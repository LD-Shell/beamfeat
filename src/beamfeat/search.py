"""Target-guided beam search over the expression DAG.

Exhaustive expansion is the scaling failure of breadth-first feature
construction: the candidate count compounds at every depth, and the cost is
paid in full before any of it is scored. This module instead expands only a
bounded frontier. At each depth, every surviving expression is extended by the
available operators, the resulting candidates are scored against the target,
and only the best ``beam_width`` survive to be extended again.

Two properties keep the beam useful rather than merely small:

**Marginal scoring.** Once features have been selected, each candidate is
scored both against the raw target and against the residual left by the
selected features, and ranked on the larger of the two. The residual score
rewards a candidate that explains something new; the raw score keeps alive a
refinement of an incumbent, such as ``log(a)`` when ``a`` is already selected,
which the residual alone would discard.

**Redundancy pruning.** Ranking on the larger score lets a beam fill with
mutually correlated variants of one expression. Before a candidate is admitted
to a depth's beam, its correlation against everything already admitted at that
depth is checked, and it is dropped if it reaches ``redundancy_threshold``.
The check is within a depth: a candidate that restates a feature from an
earlier depth can occupy a beam slot. Such restatements are removed in the
final ranking, which applies the same check across all depths.

Search is deterministic: given the same data, parameters, and ``random_state``,
the returned features and their order are identical across runs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from beamfeat.expression import (
    OPERATORS,
    EvaluationLog,
    Evaluator,
    Node,
    UnitError,
    combine,
    transform,
)
from beamfeat.scoring import ProblemType, Scorer, make_scorer

__all__ = ["BeamSearch", "SearchResult", "SearchTrace"]

logger = logging.getLogger(__name__)

# Scores closer than this are treated as tied, so the structurally simpler
# expression wins. Differences of this size are numerical noise, not evidence:
# an absolute Pearson correlation is not resolvable to anything like 1e-6 at
# realistic sample sizes. Rounding at 1e-9 was tight enough that a negligible
# extra term - e.g. (a*b) + (b*c) with c six orders smaller than a - could
# out-rank the clean (a*b) and displace it as redundant.
SCORE_TIE_TOL = 1e-6

# An affine or sign-flipped copy correlates at exactly 1 in exact arithmetic,
# but the computed value can land a rounding error either side of it. A
# correlation within this distance of 1 is therefore treated as reaching any
# threshold, so that ``redundancy_threshold=1.0`` drops exact copies
# consistently rather than according to how the rounding fell.
EXACT_COPY_TOL = 1e-9

DEFAULT_UNARY = ("log", "sqrt", "reciprocal", "square", "abs")
DEFAULT_BINARY = ("mul", "div", "add", "sub")


def report(verbose: int, level: int, message: str) -> None:
    """Print a progress line when ``verbose`` reaches ``level``.

    Progress goes to stdout rather than through :mod:`logging`, following the
    scikit-learn convention for ``verbose``: a caller who asks for progress
    should see it without first having to configure a handler. Warnings and
    diagnostics continue to use the module logger, because those are events
    the caller did not ask for and may want to capture.
    """
    if verbose >= level:
        print(f"[beamfeat] {message}", flush=True)


@dataclass(frozen=True, slots=True)
class SearchTrace:
    """Per-depth record of what the search did.

    Exposed so that the cost of a run can be attributed rather than guessed,
    and so that a beam that collapsed or saturated is visible after the fact.

    Attributes:
        depth: Search depth this record describes; depth 0 is the input columns.
        n_proposed: Candidates constructed at this depth, before evaluation.
        n_evaluated: Candidates that survived evaluation.
        n_rejected_units: Candidates rejected at construction for unit mismatch.
        n_rejected_numeric: Candidates rejected during evaluation.
        n_rejected_redundant: Candidates dropped for redundancy against the beam.
        n_kept: Candidates retained in the beam.
        best_score: Highest ranking score among the candidates kept at this
            depth. At depth 0 this is the score against the raw target.
        elapsed: Wall-clock seconds spent at this depth.
    """

    depth: int
    n_proposed: int
    n_evaluated: int
    n_rejected_units: int
    n_rejected_numeric: int
    n_rejected_redundant: int
    n_kept: int
    best_score: float
    elapsed: float

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"depth {self.depth}: proposed {self.n_proposed}, evaluated {self.n_evaluated}, "
            f"kept {self.n_kept}, best {self.best_score:.4f} ({self.elapsed:.2f}s)"
        )


@dataclass(slots=True)
class SearchResult:
    """Outcome of a completed beam search.

    Attributes:
        nodes: Selected expressions, ordered best-scoring first.
        scores: Score against the raw target for each entry of :attr:`nodes`,
            in the same order.
        trace: Per-depth :class:`SearchTrace` records.
        evaluation_log: Rejections recorded during evaluation, for auditing.
        n_proposed_total: Candidates constructed across depths 1 and above.
            The input columns are not counted; ``trace[0]`` records them.
        n_evaluated_total: Of those, the candidates that survived evaluation
            and were scored. The input columns are likewise not counted.
        n_rejected_redundant_final: Accumulated features dropped as
            near-duplicates in the final ranking across depths.
        elapsed: Total wall-clock seconds.
    """

    nodes: list[Node] = field(default_factory=list)
    scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    trace: list[SearchTrace] = field(default_factory=list)
    evaluation_log: EvaluationLog | None = None
    n_proposed_total: int = 0
    n_evaluated_total: int = 0
    n_rejected_redundant_final: int = 0
    elapsed: float = 0.0

    @property
    def names(self) -> list[str]:
        """Formula strings for the selected expressions."""
        return [node.name for node in self.nodes]

    def __len__(self) -> int:
        return len(self.nodes)

    def summary(self) -> str:
        """Multi-line human-readable summary of the run."""
        lines = [
            f"beam search: {len(self.nodes)} features from "
            f"{self.n_proposed_total} proposed / {self.n_evaluated_total} evaluated "
            f"in {self.elapsed:.2f}s"
            + (
                f", {self.n_rejected_redundant_final} near-duplicate(s) dropped across depths"
                if self.n_rejected_redundant_final
                else ""
            ),
        ]
        lines.extend(f"  {record}" for record in self.trace)
        if self.nodes:
            lines.append("  top features:")
            for node, score in list(zip(self.nodes, self.scores, strict=False))[:10]:
                lines.append(f"    {score:8.4f}  {node.name}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"SearchResult(n_features={len(self.nodes)}, elapsed={self.elapsed:.2f}s)"


class BeamSearch:
    """Guided beam search for feature construction.

    Args:
        scorer: Scoring strategy. Either a name (``"correlation"``,
            ``"mutual_information"``, ``"gradient_boosting"``, or an alias such
            as ``"mi"``) or a :class:`~beamfeat.scoring.Scorer` instance.
            Defaults to ``"correlation"``, which is the only choice cheap
            enough for wide beams; see
            :class:`~beamfeat.scoring.GradientBoostingScorer` for the tradeoff.
        problem_type: ``"regression"`` or ``"classification"``.
        max_depth: Maximum expression depth. Depth 1 permits single transforms
            and pairwise combinations of inputs; each further depth composes on
            the results of the last.
        beam_width: Expressions retained at each depth. This is the parameter
            that bounds cost: total work is roughly
            ``max_depth * beam_width * n_operators``, rather than compounding.
        max_features: Maximum features returned. Defaults to ``beam_width``.
        unary_ops: Unary operators to apply. Defaults to log, sqrt, reciprocal,
            square, and abs.
        binary_ops: Binary operators to apply. Defaults to mul, div, add, sub.
        redundancy_threshold: Absolute correlation at or above which a
            candidate is considered a duplicate of one already admitted at the
            same depth and dropped. The final ranking applies the same
            threshold across depths. ``1.0`` drops only exact affine or
            sign-flipped copies.
        include_originals: Whether input columns are eligible for the final
            result alongside constructed expressions.
        max_candidates_per_depth: Hard ceiling on candidates constructed at one
            depth, as a guard against a wide beam and many operators producing
            an unaffordable round. Candidates beyond the ceiling are not
            proposed. When the ceiling binds, parents are traversed in score
            order, so the truncation falls on the least promising ones; at
            depth 1 the parents are the input columns, ranked by their score
            against the target.
        random_state: Seed passed to the scorer. Search itself is deterministic.
        verbose: Progress reporting to stdout. ``0`` is silent, ``1`` prints a
            single summary line for the search, ``2`` adds a line per depth.

    Attributes:
        result_: :class:`SearchResult` from the most recent :meth:`run`.
    """

    def __init__(
        self,
        scorer: str | Scorer = "correlation",
        problem_type: ProblemType = "regression",
        max_depth: int = 2,
        beam_width: int = 50,
        max_features: int | None = None,
        unary_ops: Sequence[str] = DEFAULT_UNARY,
        binary_ops: Sequence[str] = DEFAULT_BINARY,
        redundancy_threshold: float = 0.95,
        include_originals: bool = True,
        max_candidates_per_depth: int = 200_000,
        random_state: int | None = 0,
        verbose: int = 0,
    ) -> None:
        if max_depth < 0:
            raise ValueError(f"max_depth must be non-negative, got {max_depth}")
        if beam_width < 1:
            raise ValueError(f"beam_width must be at least 1, got {beam_width}")
        if not 0.0 < redundancy_threshold <= 1.0:
            raise ValueError(f"redundancy_threshold must be in (0, 1], got {redundancy_threshold}")
        if max_features is not None and max_features < 1:
            raise ValueError(f"max_features must be at least 1, got {max_features}")
        if max_candidates_per_depth < 1:
            raise ValueError(f"max_candidates_per_depth must be at least 1, got {max_candidates_per_depth}")

        unknown = [op for op in (*unary_ops, *binary_ops) if op not in OPERATORS]
        if unknown:
            raise ValueError(f"unknown operator(s): {sorted(unknown)}; available: {sorted(OPERATORS)}")
        bad_unary = [op for op in unary_ops if OPERATORS[op].arity != 1]
        if bad_unary:
            raise ValueError(f"not unary operator(s): {sorted(bad_unary)}")
        bad_binary = [op for op in binary_ops if OPERATORS[op].arity != 2]
        if bad_binary:
            raise ValueError(f"not binary operator(s): {sorted(bad_binary)}")

        self.scorer = scorer
        self.problem_type = problem_type
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.max_features = max_features
        self.unary_ops = tuple(unary_ops)
        self.binary_ops = tuple(binary_ops)
        self.redundancy_threshold = redundancy_threshold
        self.include_originals = include_originals
        self.max_candidates_per_depth = max_candidates_per_depth
        self.random_state = random_state
        self.verbose = verbose

        self.result_: SearchResult | None = None

    # -- proposal ----------------------------------------------------------- #

    def _propose(
        self,
        beam: list[Node],
        pool: list[Node],
        seen: set[Node],
    ) -> tuple[list[Node], int]:
        """Construct candidates one level above the current beam.

        Unary operators are applied to each beam member; binary operators
        combine each beam member with each pool member. Candidates already
        seen are skipped, so the DAG's structural unification does the
        deduplication rather than a separate pass.

        Returns:
            Tuple of (new candidate nodes, count rejected for unit mismatch).
        """
        candidates: list[Node] = []
        n_unit_rejected = 0
        ceiling = self.max_candidates_per_depth

        for node in beam:
            for op in self.unary_ops:
                if len(candidates) >= ceiling:
                    return candidates, n_unit_rejected
                try:
                    candidate = transform(op, node)
                except UnitError:
                    n_unit_rejected += 1
                    continue
                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)

        for node in beam:
            for partner in pool:
                if node == partner:
                    continue
                for op in self.binary_ops:
                    if len(candidates) >= ceiling:
                        return candidates, n_unit_rejected
                    try:
                        candidate = combine(op, node, partner)
                    except UnitError:
                        n_unit_rejected += 1
                        continue
                    except ValueError:
                        continue
                    if candidate not in seen:
                        seen.add(candidate)
                        candidates.append(candidate)

        return candidates, n_unit_rejected

    # -- redundancy --------------------------------------------------------- #

    def _prune_redundant(
        self,
        nodes: list[Node],
        values: np.ndarray,
        scores: np.ndarray,
        limit: int,
    ) -> tuple[list[Node], np.ndarray, np.ndarray, int]:
        """Greedily admit candidates in score order, skipping near-duplicates.

        A candidate is admitted if its absolute correlation against every
        already-admitted candidate stays below ``redundancy_threshold``.
        Correlations within :data:`EXACT_COPY_TOL` of 1 count as reaching it.

        Returns:
            Tuple of (kept nodes, their values, their scores, count dropped).
        """
        if not nodes:
            return [], np.empty((values.shape[0], 0)), np.empty(0), 0

        order = _rank(nodes, scores)
        standardised = _standardise_columns(values)
        cutoff = min(self.redundancy_threshold, 1.0 - EXACT_COPY_TOL)

        kept_indices: list[int] = []
        n_dropped = 0
        n_samples = values.shape[0]

        for index in order:
            if len(kept_indices) >= limit:
                break
            if kept_indices:
                correlations = np.abs(standardised[:, kept_indices].T @ standardised[:, index]) / n_samples
                if float(np.max(correlations)) >= cutoff:
                    n_dropped += 1
                    continue
            kept_indices.append(int(index))

        kept_nodes = [nodes[i] for i in kept_indices]
        return kept_nodes, values[:, kept_indices], scores[kept_indices], n_dropped

    # -- main loop ---------------------------------------------------------- #

    def run(
        self,
        data: Any,
        target: np.ndarray,
        columns: Sequence[str] | None = None,
        units: dict[str, Any] | None = None,
        evaluator: Evaluator | None = None,
    ) -> SearchResult:
        """Execute the search.

        Args:
            data: Mapping of column name to array, or a 2-D array. Ignored if
                ``evaluator`` is supplied.
            target: Target vector, one entry per row.
            columns: Column names, when ``data`` is a 2-D array.
            units: Optional mapping of column name to pint quantity. Supplying
                units restricts the search to dimensionally valid expressions.
            evaluator: A pre-built :class:`~beamfeat.expression.Evaluator`, to
                reuse a warm cache across runs.

        Returns:
            A :class:`SearchResult`. Also stored on :attr:`result_`.
        """
        started = time.perf_counter()

        if evaluator is None:
            evaluator = Evaluator(data, columns=columns, units=units)
        target = np.asarray(target).ravel()
        if target.shape[0] != evaluator.n_rows:
            raise ValueError(f"target has {target.shape[0]} rows, data has {evaluator.n_rows}")

        scorer = make_scorer(self.scorer, problem_type=self.problem_type, random_state=self.random_state)

        # Depth 0: the input columns themselves.
        leaf_nodes = evaluator.leaf_nodes()
        beam, beam_values = evaluator.evaluate_many(leaf_nodes)
        if not beam:
            raise ValueError("no input column survived evaluation; check for constant or non-finite columns")

        beam_scores = scorer.score_batch(beam_values, target, None)
        pool = list(beam)
        pool_values = beam_values

        selected: list[Node] = list(beam) if self.include_originals else []
        selected_values = beam_values if self.include_originals else np.empty((evaluator.n_rows, 0))
        selected_scores = list(beam_scores) if self.include_originals else []

        seen: set[Node] = set(beam)
        trace: list[SearchTrace] = []
        n_proposed_total = 0
        n_evaluated_total = 0

        trace.append(
            SearchTrace(
                depth=0,
                n_proposed=len(leaf_nodes),
                n_evaluated=len(beam),
                n_rejected_units=0,
                n_rejected_numeric=len(leaf_nodes) - len(beam),
                n_rejected_redundant=0,
                n_kept=len(beam),
                best_score=float(np.max(beam_scores)) if len(beam_scores) else 0.0,
                elapsed=time.perf_counter() - started,
            )
        )
        report(
            self.verbose,
            2,
            f"  depth 0: {len(beam)} input columns, best score "
            f"{float(np.max(beam_scores)) if len(beam_scores) else 0.0:.4f}",
        )

        for depth in range(1, self.max_depth + 1):
            depth_started = time.perf_counter()

            candidates, n_unit_rejected = self._propose(beam, pool, seen)
            if depth == 1 and len(candidates) >= self.max_candidates_per_depth:
                # Later beams come out of _prune_redundant in score order, but
                # the leaves are in column order, so a ceiling that binds here
                # would cut the last columns of the table rather than the
                # weakest. Proposal order also decides between exact ties such
                # as (a - b) and (b - a), so a round the ceiling does not reach
                # keeps column order and only a truncated one is rebuilt.
                seen.difference_update(candidates)
                ranked = [beam[i] for i in _rank(beam, beam_scores)]
                candidates, n_unit_rejected = self._propose(ranked, ranked, seen)
            n_proposed_total += len(candidates)
            if not candidates:
                report(self.verbose, 2, f"  depth {depth}: no new candidates; stopping")
                break

            evaluated, values = evaluator.evaluate_many(candidates)
            n_numeric_rejected = len(candidates) - len(evaluated)
            n_evaluated_total += len(evaluated)
            if not evaluated:
                report(self.verbose, 2, f"  depth {depth}: nothing survived evaluation; stopping")
                break

            # Rank on the larger of the marginal and absolute scores.
            #
            # Marginal scoring alone is subtly wrong for survival decisions: a
            # candidate that *refines* an incumbent scores near zero against
            # the residual precisely because the incumbent is a crude version
            # of it. With log(a) as the true signal and `a` already selected,
            # log(a) scores ~0.22 marginally against ~1.00 absolutely, and
            # would be pruned in favour of noise that happens to be orthogonal.
            #
            # Absolute scoring alone has the opposite failure: the beam fills
            # with restatements of whatever dominates the target. Taking the
            # larger keeps refinements alive while still rewarding novelty,
            # and the redundancy pass below removes near-duplicates within
            # this depth; restatements of earlier depths are left to the
            # final ranking.
            incumbent = selected_values if selected_values.shape[1] else None
            absolute_scores = scorer.score_batch(values, target, None)
            if incumbent is None:
                scores = absolute_scores
            else:
                marginal_scores = scorer.score_batch(values, target, incumbent)
                scores = np.maximum(absolute_scores, marginal_scores)

            beam, beam_values, kept_scores, n_redundant = self._prune_redundant(
                evaluated, values, scores, self.beam_width
            )
            if not beam:
                report(self.verbose, 2, f"  depth {depth}: beam collapsed after redundancy pruning; stopping")
                break

            selected.extend(beam)
            selected_scores.extend(float(s) for s in kept_scores)
            selected_values = (
                beam_values if selected_values.shape[1] == 0 else np.column_stack([selected_values, beam_values])
            )

            pool = pool + beam
            pool_values = np.column_stack([pool_values, beam_values])

            trace.append(
                SearchTrace(
                    depth=depth,
                    n_proposed=len(candidates),
                    n_evaluated=len(evaluated),
                    n_rejected_units=n_unit_rejected,
                    n_rejected_numeric=n_numeric_rejected,
                    n_rejected_redundant=n_redundant,
                    n_kept=len(beam),
                    best_score=float(np.max(kept_scores)),
                    elapsed=time.perf_counter() - depth_started,
                )
            )
            dropped = []
            if n_unit_rejected:
                dropped.append(f"{n_unit_rejected} unit")
            if n_numeric_rejected:
                dropped.append(f"{n_numeric_rejected} numeric")
            if n_redundant:
                dropped.append(f"{n_redundant} near-duplicate")
            report(
                self.verbose,
                2,
                f"  depth {depth}: {len(candidates)} proposed -> {len(evaluated)} evaluated"
                f" -> {len(beam)} kept by beam"
                + (f" ({', '.join(dropped)} rejected)" if dropped else "")
                + f", best {float(np.max(kept_scores)):.4f}"
                f", {time.perf_counter() - depth_started:.2f}s",
            )

        # Final ranking across every depth, with one more redundancy pass so
        # that features admitted at different depths are also deduplicated.
        #
        # Scores accumulated during the search are partly marginal: each was
        # measured against whatever incumbent set existed when its depth was
        # reached. They are therefore not comparable across depths, and using
        # them here would let an early weak feature block a later stronger one
        # that correlates with it. Rescore everything against the raw target on
        # a common footing before deduplicating.
        final_scores = (
            scorer.score_batch(selected_values, target, None)
            if selected_values.shape[1]
            else np.asarray(selected_scores, dtype=np.float64)
        )
        result = self._finalise(selected, selected_values, final_scores, evaluator, trace, started)
        result.n_proposed_total = n_proposed_total
        result.n_evaluated_total = n_evaluated_total
        self.result_ = result
        depth_reached = trace[-1].depth
        report(
            self.verbose,
            1,
            f"search: depth {depth_reached}"
            + (f" of {self.max_depth}" if depth_reached < self.max_depth else "")
            + f", beam {self.beam_width} -> "
            f"{n_proposed_total} proposed, {len(result.nodes)} candidates "
            f"({result.elapsed:.2f}s)",
        )
        return result

    def _finalise(
        self,
        nodes: list[Node],
        values: np.ndarray,
        scores: np.ndarray,
        evaluator: Evaluator,
        trace: list[SearchTrace],
        started: float,
    ) -> SearchResult:
        """Rank, deduplicate, and truncate the accumulated selection."""
        limit = self.max_features if self.max_features is not None else self.beam_width

        n_redundant = 0
        if nodes:
            nodes, _, scores, n_redundant = self._prune_redundant(nodes, values, scores, limit)
        else:  # pragma: no cover - defensive
            scores = np.empty(0)

        return SearchResult(
            nodes=nodes,
            scores=scores,
            trace=trace,
            evaluation_log=evaluator.log,
            n_rejected_redundant_final=n_redundant,
            elapsed=time.perf_counter() - started,
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"BeamSearch(scorer={self.scorer!r}, max_depth={self.max_depth}, "
            f"beam_width={self.beam_width}, problem_type={self.problem_type!r})"
        )


def _rank(nodes: Sequence[Node], scores: np.ndarray) -> np.ndarray:
    """Order candidates best first, as beam admission sees them.

    Scores within SCORE_TIE_TOL of each other - a difference below scoring
    noise - are tied, and the structurally simpler expression goes first, so
    that the more complex near-duplicate is the one the correlation check
    drops. Remaining ties keep their given order. lexsort uses its last key as
    primary and is stable.
    """
    sizes = np.array([node.n_operators for node in nodes])
    quantised = np.round(np.asarray(scores, dtype=np.float64) / SCORE_TIE_TOL)
    return np.lexsort((sizes, -quantised))


def _standardise_columns(values: np.ndarray) -> np.ndarray:
    """Centre and scale each column to unit variance; constant columns go to zero.

    Constancy is judged relative to each column's magnitude, as in
    :func:`~beamfeat.scoring.is_constant`. Under an absolute floor on the
    spread, a column that is merely small in its units is left unscaled, looks
    uncorrelated with everything, and an exact copy of it passes the
    redundancy check.
    """
    centred = values - values.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(centred**2, axis=0, keepdims=True))
    magnitude = np.mean(np.abs(values), axis=0, keepdims=True)
    varying = scale > 1e-12 * magnitude
    standardised = np.where(varying, centred / np.where(varying, scale, 1.0), 0.0)
    return np.nan_to_num(standardised, nan=0.0, posinf=0.0, neginf=0.0)
