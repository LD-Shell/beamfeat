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

**Bounded memory.** On a wide table one depth can propose hundreds of
thousands of candidates. They are evaluated and scored a chunk at a time
(``chunk_size``, by default as many columns as fit in :data:`CHUNK_BYTES`) and
only their scores are kept; the candidates the redundancy pass reaches are
evaluated again. Peak memory then depends on the chunk and the beam rather
than on the number of candidates. A scorer whose scores depend on the whole
batch, such as mutual information, still receives a depth at once.

Search is deterministic: given the same data, parameters, and ``random_state``,
the returned features and their order are identical across runs, whatever
``n_jobs`` and ``chunk_size`` are set to.
"""

from __future__ import annotations

import logging
import numbers
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from beamfeat.expression import (
    OPERATORS,
    BatchOutcome,
    EvaluationBatch,
    EvaluationLog,
    Evaluator,
    Node,
    NodeError,
    UnitError,
    UnitRegistryError,
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

# Candidate values held per chunk when ``chunk_size`` is not given. Peak memory
# while a depth is scored is a few times this per worker, however many
# candidates the depth proposes.
CHUNK_BYTES = 8 * 2**20

# A depth with fewer candidates than this is scored in the calling thread
# whatever ``n_jobs`` says: starting workers would cost more than they save.
# Scorers that fit a model per candidate are exempt.
MIN_PARALLEL_CANDIDATES = 4096

# Candidates per chunk for a scorer that fits a model per candidate, so that
# a round of a few hundred still divides between workers.
EXPENSIVE_CHUNK = 8


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
        n_jobs: Workers that evaluate and score a depth's candidates, chunk
            by chunk, through joblib. ``1`` (the default) works in the calling
            thread, ``-1`` uses every core, and ``None`` follows the active
            joblib configuration (``joblib.parallel_config``, or
            ``parallel_backend`` before joblib 1.3) or else means 1. Workers
            are threads, since numpy releases the GIL, except for
            :class:`~beamfeat.scoring.GradientBoostingScorer`, whose model fits
            run in worker processes; a whole-batch scorer such as
            :class:`~beamfeat.scoring.MutualInformationScorer` receives
            ``n_jobs`` through the joblib backend instead. Depths of fewer than
            :data:`MIN_PARALLEL_CANDIDATES` candidates are scored in the
            calling thread unless the scorer fits models. The result does not
            depend on this value.
        chunk_size: Candidates evaluated and scored together. Bounds the
            candidate values held at once to ``chunk_size * n_rows`` per
            worker, rather than one column per candidate of the whole depth.
            Defaults to as many columns as fit in :data:`CHUNK_BYTES`. Scorers
            whose scores depend on the whole batch, such as
            :class:`~beamfeat.scoring.MutualInformationScorer`, still score a
            depth at once (see :meth:`~beamfeat.scoring.Scorer.scores_by_column`).
            The result does not depend on this value.

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
        n_jobs: int | None = 1,
        chunk_size: int | None = None,
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
        if n_jobs is not None and (isinstance(n_jobs, bool) or not isinstance(n_jobs, numbers.Integral) or n_jobs == 0):
            raise ValueError(f"n_jobs must be a non-zero integer or None, got {n_jobs!r}")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")

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
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size

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
                except UnitRegistryError:
                    raise
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
                    except UnitRegistryError:
                        raise
                    except UnitError:
                        n_unit_rejected += 1
                        continue
                    except NodeError:
                        continue
                    if candidate not in seen:
                        seen.add(candidate)
                        candidates.append(candidate)

        return candidates, n_unit_rejected

    def _can_reach_ceiling(self, leaves: list[Node]) -> bool:
        """Whether proposing from ``leaves`` alone can reach the depth ceiling.

        Counts every operator application, and each unordered pair once for a
        commutative operator, so the count bounds the distinct candidates.
        """
        n = len(leaves)
        n_proposals = n * len(self.unary_ops)
        for op in self.binary_ops:
            n_pairs = n * (n - 1)
            n_proposals += n_pairs // 2 if OPERATORS[op].commutative else n_pairs
        return n_proposals >= self.max_candidates_per_depth

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
        kept_indices, n_dropped = self._admit(order, lambda index: standardised[:, index], limit, values.shape[0])
        kept_nodes = [nodes[i] for i in kept_indices]
        return kept_nodes, values[:, kept_indices], scores[kept_indices], n_dropped

    def _prune_evaluated(
        self,
        nodes: list[Node],
        scores: np.ndarray,
        limit: int,
        evaluator: Evaluator,
    ) -> tuple[list[Node], np.ndarray, np.ndarray, int]:
        """:meth:`_prune_redundant` for a round whose values were not kept.

        The candidates the greedy pass reaches, usually a small fraction of a
        round, are evaluated again in rank order and standardised a block at
        a time, so the round never needs a matrix of all its values.
        """
        n_samples = evaluator.n_rows
        if not nodes:
            return [], np.empty((n_samples, 0)), np.empty(0), 0

        order = _rank(nodes, scores)
        columns = _RankedColumns(order, nodes, evaluator, block=max(2, min(limit, 256)))
        kept_indices, n_dropped = self._admit(order, columns.standardised, limit, n_samples)
        kept_values = np.empty((n_samples, len(kept_indices)), dtype=evaluator.dtype, order="F")
        for position, index in enumerate(kept_indices):
            kept_values[:, position] = columns.raw(index)
        return [nodes[i] for i in kept_indices], kept_values, scores[kept_indices], n_dropped

    def _admit(
        self,
        order: np.ndarray,
        standardised: Callable[[int], np.ndarray],
        limit: int,
        n_samples: int,
    ) -> tuple[list[int], int]:
        """Walk candidates in ``order``, admitting each that duplicates none admitted.

        Args:
            order: Candidate indices, best first.
            standardised: Returns a candidate's standardised column, given
                its index. Called in ``order``.
            limit: Most candidates to admit.
            n_samples: Rows per column.

        Returns:
            Tuple of (admitted indices in admission order, count dropped).
        """
        cutoff = min(self.redundancy_threshold, 1.0 - EXACT_COPY_TOL)
        # One row per admitted column: laid out as the transpose of a column
        # selection from a matrix, which is what the product is computed on.
        admitted = np.empty((0, n_samples))
        kept_indices: list[int] = []
        n_dropped = 0

        for index in order:
            if len(kept_indices) >= limit:
                break
            column = standardised(index)
            if kept_indices:
                correlations = np.abs(admitted[: len(kept_indices)] @ column) / n_samples
                if float(np.max(correlations)) >= cutoff:
                    n_dropped += 1
                    continue
            else:
                admitted = np.empty((min(limit, len(order)), n_samples), dtype=column.dtype)
            admitted[len(kept_indices)] = column
            kept_indices.append(int(index))

        return kept_indices, n_dropped

    # -- evaluation and scoring --------------------------------------------- #

    def _chunk_columns(self, evaluator: Evaluator, scorer: Scorer) -> int:
        """Candidates per chunk: ``chunk_size``, or what fits in :data:`CHUNK_BYTES`."""
        if self.chunk_size is not None:
            return int(self.chunk_size)
        column_bytes = max(1, evaluator.n_rows * evaluator.dtype.itemsize)
        step = int(np.clip(CHUNK_BYTES // column_bytes, 16, 8192))
        return min(step, EXPENSIVE_CHUNK) if scorer._expensive else step

    def _score_round(
        self,
        evaluator: Evaluator,
        candidates: list[Node],
        scorer: Scorer,
        target: np.ndarray,
        absolute: Any,
        incumbent: np.ndarray | None,
    ) -> tuple[list[Node], np.ndarray, np.ndarray | None]:
        """Evaluate and score one depth's candidates, a chunk at a time.

        Every candidate is ranked on the larger of its score against the raw
        target and its score against the residual left by ``incumbent``.

        A scorer that scores column by column (see
        :meth:`~beamfeat.scoring.Scorer.scores_by_column`) scores each chunk
        as it is evaluated, and the chunk's values are then dropped. Any other
        scorer receives the whole round at once, as one matrix.

        Args:
            evaluator: Evaluator for the search data.
            candidates: This depth's candidates.
            scorer: The scorer.
            target: Target vector.
            absolute: ``scorer.prepare(target)``, for a column-by-column scorer.
            incumbent: Values of the features selected so far, if any.

        Returns:
            Tuple of (evaluated nodes, their scores, their values as a
            matrix, or ``None`` when the values were dropped).
        """
        by_column = scorer.scores_by_column()
        references = None
        if by_column:
            references = (absolute,) if incumbent is None else (absolute, scorer.prepare(target, incumbent))

        step = self._chunk_columns(evaluator, scorer)
        chunks = [candidates[start : start + step] for start in range(0, len(candidates), step)]
        batches = [evaluator.plan_batch(chunk) for chunk in chunks]
        expensive = by_column and scorer._expensive
        n_jobs = self.n_jobs if expensive or len(candidates) >= MIN_PARALLEL_CANDIDATES else 1
        outcomes = _map(
            _run_chunk,
            [(batch, scorer, references) for batch in batches],
            n_jobs,
            prefer="processes" if expensive else "threads",
        )

        evaluated: list[Node] = []
        score_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        for chunk, (outcome, chunk_scores) in zip(chunks, outcomes, strict=True):
            evaluated.extend(evaluator.commit_batch(chunk, outcome))
            if by_column:
                score_parts.append(chunk_scores)
            else:
                value_parts.append(outcome.values)

        if not evaluated:
            return evaluated, np.empty(0), None
        if by_column:
            scores = np.concatenate(score_parts) if score_parts else np.empty(0)
            return evaluated, scores, None

        # Row-major, as evaluate_many returns: a scorer that reduces along the
        # rows sums in an order that depends on the layout.
        values = np.empty((evaluator.n_rows, len(evaluated)), dtype=evaluator.dtype)
        start = 0
        for part in value_parts:
            values[:, start : start + part.shape[1]] = part
            start += part.shape[1]
        # A scorer that parallelises through joblib, as scikit-learn's mutual
        # information estimators do, picks up n_jobs from the active backend.
        with _joblib_threads(self.n_jobs):
            scores = scorer.score_batch(values, target, None)
            if incumbent is not None:
                scores = np.maximum(scores, scorer.score_batch(values, target, incumbent))
        return evaluated, scores, values

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
        absolute = scorer.prepare(target) if scorer.scores_by_column() else None

        # Depth 0: the input columns themselves.
        leaf_nodes = evaluator.leaf_nodes()
        beam, beam_values = evaluator.evaluate_many(leaf_nodes)
        if not beam:
            raise ValueError("no input column survived evaluation; check for constant or non-finite columns")

        beam_scores = scorer.score_batch(beam_values, target, None)
        pool = list(beam)

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

            # Later beams come out of _prune_redundant in score order, but the
            # leaves are in column order, so a ceiling that binds at depth 1
            # would cut the last columns of the table rather than the weakest.
            # Proposal order also decides between exact ties such as (a - b)
            # and (b - a), so a round the ceiling does not reach keeps column
            # order and only a truncated one is proposed in score order. Both
            # orders propose the same set, so whether the ceiling binds does
            # not depend on which is tried first; the score order is tried
            # first when the ceiling can bind, which it usually then does.
            if depth == 1 and self._can_reach_ceiling(beam):
                ranked = [beam[i] for i in _rank(beam, beam_scores)]
                candidates, n_unit_rejected = self._propose(ranked, ranked, seen)
                if len(candidates) < self.max_candidates_per_depth:
                    seen.difference_update(candidates)
                    candidates, n_unit_rejected = self._propose(beam, pool, seen)
            else:
                candidates, n_unit_rejected = self._propose(beam, pool, seen)
            n_proposed_total += len(candidates)
            if not candidates:
                report(self.verbose, 2, f"  depth {depth}: no new candidates; stopping")
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
            evaluated, scores, values = self._score_round(
                evaluator, candidates, scorer, target, absolute, incumbent
            )
            n_numeric_rejected = len(candidates) - len(evaluated)
            n_evaluated_total += len(evaluated)
            if not evaluated:
                report(self.verbose, 2, f"  depth {depth}: nothing survived evaluation; stopping")
                break

            if values is None:
                beam, beam_values, kept_scores, n_redundant = self._prune_evaluated(
                    evaluated, scores, self.beam_width, evaluator
                )
            else:
                beam, beam_values, kept_scores, n_redundant = self._prune_redundant(
                    evaluated, values, scores, self.beam_width
                )
            del values
            if not beam:
                report(self.verbose, 2, f"  depth {depth}: beam collapsed after redundancy pruning; stopping")
                break

            selected.extend(beam)
            selected_scores.extend(float(s) for s in kept_scores)
            selected_values = (
                beam_values if selected_values.shape[1] == 0 else np.column_stack([selected_values, beam_values])
            )

            pool = pool + beam

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


class _RankedColumns:
    """Candidate columns evaluated again in rank order, a block at a time.

    Serves :meth:`BeamSearch._admit` the standardised columns it walks
    through, in the order it walks. Each block is standardised as a row-major
    matrix of at least two columns by :func:`_standardise_columns`, and each
    column is served as a strided view of it, so that the arithmetic, down
    to the summation order and the strides handed to BLAS, is that of
    standardising the whole round at once.
    """

    def __init__(self, order: np.ndarray, nodes: list[Node], evaluator: Evaluator, block: int) -> None:
        self._order = order
        self._nodes = nodes
        self._evaluator = evaluator
        self._block = block
        self._next = 0
        self._where: dict[int, int] = {}
        self._raw = np.empty((evaluator.n_rows, 0))
        self._standardised = self._raw

    def _fetch(self, index: int) -> None:
        """Evaluate the next block of the ranking, which starts at ``index``."""
        chosen = [int(i) for i in self._order[self._next : self._next + self._block]]
        if not chosen or chosen[0] != index:  # pragma: no cover - defensive
            raise RuntimeError("candidate columns must be requested in rank order")
        self._next += len(chosen)
        raw: list[np.ndarray] = []
        for i in chosen:
            values = self._evaluator.evaluate(self._nodes[i])
            if values is None:  # pragma: no cover - defensive
                raise RuntimeError(f"candidate {self._nodes[i].name} no longer evaluates")
            raw.append(values)
        # A lone column would be reduced as a contiguous vector, which sums in
        # a different order from a column of a wider row-major matrix.
        self._raw = np.column_stack(raw if len(raw) > 1 else raw * 2)
        self._standardised = _standardise_columns(self._raw)
        self._where = {i: column for column, i in enumerate(chosen)}

    def standardised(self, index: int) -> np.ndarray:
        """Standardised column of a candidate; candidates are requested in rank order."""
        index = int(index)
        if index not in self._where:
            self._fetch(index)
        return self._standardised[:, self._where[index]]

    def raw(self, index: int) -> np.ndarray:
        """Values of a candidate already served, if it is in the current block."""
        column = self._where.get(index)
        if column is not None:
            return self._raw[:, column]
        values = self._evaluator.evaluate(self._nodes[index])
        if values is None:  # pragma: no cover - defensive
            raise RuntimeError(f"candidate {self._nodes[index].name} no longer evaluates")
        return values


def _run_chunk(
    batch: EvaluationBatch, scorer: Scorer, references: tuple[Any, ...] | None
) -> tuple[BatchOutcome, np.ndarray | None]:
    """Evaluate one chunk and, given prepared references, score it.

    With references, each candidate's score is the largest against any of
    them, and the values are dropped from the outcome to keep it small.
    """
    outcome = batch.run()
    if references is None:
        return outcome, None
    values = outcome.values
    stats = outcome.stats if values.dtype == np.float64 else None
    scores = scorer.score_prepared(values, references[0], stats)
    for prepared in references[1:]:
        scores = np.maximum(scores, scorer.score_prepared(values, prepared, stats))
    return outcome._replace(values=None), scores


def _joblib_threads(n_jobs: int | None) -> AbstractContextManager[Any]:
    """A joblib threading backend with ``n_jobs`` workers, or no change for 1 or None."""
    if n_jobs is None or n_jobs == 1:
        return nullcontext()
    from joblib import parallel_backend

    return parallel_backend("threading", n_jobs=n_jobs)


def _map(
    function: Callable[..., Any], arguments: list[tuple[Any, ...]], n_jobs: int | None, prefer: str = "threads"
) -> list[Any]:
    """Apply ``function`` to each argument tuple through joblib, returning results in order."""
    if n_jobs == 1 or len(arguments) <= 1:
        return [function(*args) for args in arguments]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=n_jobs, prefer=prefer)(delayed(function)(*args) for args in arguments)


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
