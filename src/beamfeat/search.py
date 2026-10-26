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

**Magnitude slots** (``magnitude_slots``), off by default here and on by
default in the estimators for regression. A part of a law can be uncorrelated
with the target, as ``x0*x1`` is with ``x0*x1*x2`` when the inputs are
centred, and the score then ranks it among noise. With the option, each depth
but the last keeps a few more candidates, those whose magnitude best tracks
the target's (see :func:`magnitude_scores`), and a first depth too wide to
propose every pair proposes those inputs early.

**Residual slots** (``residual_slots``), off by default. The final ranking keeps the
``max_features`` candidates that score best against the target, and a
dominant term fills it with restatements of itself, so that a weaker term
beside it, such as ``log(x3)`` in ``x0*x1/x2 + 0.5*log(x3)``, is cut although
the search built it. With the option, the first terms of a greedy forward
path through every candidate the search kept join the ranking, each the one
that adds most to a least-squares fit of those before it; a restatement adds
nothing to the term it restates, and the weaker term comes second. The path
also chooses among the transforms of a path term's operand, which a depth
may have dropped as near copies of one another.

Memory stays bounded as well. On a wide table one depth can propose
hundreds of thousands of candidates. They are evaluated and scored a chunk at a time
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
import math
import numbers
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from beamfeat._forward import forward_order
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
    parametric_operators,
    transform,
)
from beamfeat.parametric import SNAP_LR, _correlation_rows, _statistic, fit_constants_many
from beamfeat.scoring import (
    ColumnStats,
    CorrelationScorer,
    ProblemType,
    Scorer,
    _standardise,
    _standardise_block,
    column_stats,
    make_scorer,
)

__all__ = ["BeamSearch", "SearchResult", "SearchTrace", "magnitude_scores"]

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

# The 99.9% points of chi-squared variables with 1 to 8 degrees of freedom:
# the level at which a fitted constant snaps to a simple value (SNAP_LR, one
# degree), for a difference of that many fitted constants.
_LR_CRITICAL = (SNAP_LR, 13.82, 16.27, 18.47, 20.52, 22.46, 24.32, 26.12)

DEFAULT_UNARY = ("log", "sqrt", "reciprocal", "square", "abs")
DEFAULT_BINARY = ("mul", "div", "add", "sub")
# The estimators' default fitted_ops: every built-in parametric operator but
# sign_shift, which is opt-in. BeamSearch itself defaults to none.
DEFAULT_FITTED = ("exp_scaled", "pow", "inv_shift", "log_shift")

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

# Search rows the magnitude score of ``magnitude_slots`` is computed on, at
# most: beyond this many, that many evenly spaced through them. It ranks a few
# extra slots, and a correlation over 1,000 rows has a standard error near 0.03,
# far below what a factor of the target scores (0.4 to 0.8 in
# benchmarks/search_stress/); the cap keeps its cost from growing with the rows.
MAGNITUDE_ROWS = 1000

# Search rows the forward path of ``residual_slots`` runs on, at most, and the
# bytes of candidate values it may hold, which lowers that cap on a wide
# table: beyond the cap, that many rows evenly spaced through them. A term
# explaining a hundredth of the target's variance correlates with what the
# leading terms leave at about 0.1 or more, which 1,000 rows resolve.
RESIDUAL_ROWS = 20_000
RESIDUAL_BYTES = 64 * 2**20


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
        n_rejected_constraints: Candidates rejected at construction by
            ``max_inputs``; 0 when it is not set.
        n_fitted: Of the candidates proposed, those carrying a constant
            fitted at this depth (see ``fitted_ops`` on :class:`BeamSearch`).
        n_magnitude: Of the candidates kept, those kept for their magnitude
            score in the slots ``magnitude_slots`` adds; 0 when it is not set.
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
    n_rejected_constraints: int = 0
    n_fitted: int = 0
    n_magnitude: int = 0

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"depth {self.depth}: proposed {self.n_proposed}, evaluated {self.n_evaluated}, "
            f"kept {self.n_kept}, best {self.best_score:.4f} ({self.elapsed:.2f}s)"
            + (f", {self.n_rejected_constraints} constrained" if self.n_rejected_constraints else "")
            + (f", {self.n_magnitude} by magnitude" if self.n_magnitude else "")
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


@dataclass
class _SearchState:
    """What a search carries from one depth to the next.

    ``beam`` is the current depth's kept candidates, with their values and,
    at depth 0, their scores; ``pool`` everything a proposal may pair with;
    ``selected`` every candidate kept at any depth, with its values and
    scores; ``seen`` every candidate proposed so far; ``waiting`` the fitted
    candidates held for the round of their depth; ``extra_parents`` the
    parents beyond the beam that the next round fits on.
    """

    beam: list[Node]
    beam_values: np.ndarray
    beam_scores: np.ndarray
    pool: list[Node]
    selected: list[Node]
    selected_values: np.ndarray
    selected_scores: list[float]
    seen: set[Node]
    waiting: dict[int, list[Node]]
    extra_parents: list[Node]
    extra_values: np.ndarray
    trace: list[SearchTrace]
    n_proposed_total: int
    n_evaluated_total: int


class BeamSearch:
    """Guided beam search for feature construction.

    Args:
        scorer: Scoring strategy. Either a name (``"correlation"``,
            ``"spearman"``, ``"mutual_information"``, ``"gradient_boosting"``,
            or an alias such as ``"mi"`` or ``"rank"``) or a
            :class:`~beamfeat.scoring.Scorer` instance. Defaults to
            ``"correlation"``; it and ``"spearman"`` are the choices cheap
            enough for wide beams; see
            :class:`~beamfeat.scoring.GradientBoostingScorer` for the tradeoff.
        problem_type: ``"regression"`` or ``"classification"``.
        max_depth: Maximum expression depth. Depth 1 permits single transforms
            and pairwise combinations of inputs; each further depth composes on
            the results of the last.
        beam_width: Expressions retained at each depth. This is the parameter
            that bounds cost: each depth applies the operators to the beam and
            pairs it with every expression kept so far, so the work grows
            polynomially in ``beam_width`` and ``max_depth`` rather than
            compounding with depth.
        max_features: Maximum features returned by score. Defaults to
            ``beam_width``; ``residual_slots`` adds up to that many more.
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
        max_inputs: Most distinct input columns one expression may use.
            Candidates over the limit are rejected as they are constructed,
            before evaluation. Input columns themselves always pass. ``None``
            (default) sets no limit. An expression of depth d uses at most
            2**d columns, so a value of 2**max_depth or more has no effect.
            Set it at or above the number of inputs the laws you look for
            need: one that needs more cannot be built, and the search
            returns the best expression within the limit instead.
        fitted_ops: Parametric operators whose constant is fitted to the
            data, from ``"exp_scaled"`` (``exp(a*x)``), ``"pow"`` (``x^a``,
            for positive ``x``), ``"inv_shift"`` (``1/(x + a)``),
            ``"log_shift"`` (``log(x + a)``) and ``"sign_shift"``
            (``sign(x + a)``, the threshold between the operand's 5% and 95%
            quantiles). Empty by default, which leaves
            every expression free of constants. At each depth, each operator
            is fitted to each expression of the previous depth's beam (at
            the first depth, the ``beam_width`` best input columns), and to
            the previous depth's ``beam_width`` best candidates that the
            redundancy check kept out of the beam; the fitted expressions
            join that depth's candidates, to be scored,
            pruned and kept like the rest; they are not counted against
            ``max_candidates_per_depth``. The constant maximises the absolute
            correlation with the target, alone and, while the depth limit
            leaves room for a product, as a factor of an expression in the
            beam, and is snapped to a simple value such as 1/2 when the data
            cannot tell the two apart; see :mod:`beamfeat.parametric`. A
            fitted pole keeps a margin from the data, and ``pow`` fits a
            negative power, whose pole is at 0, only where the operand stays
            more than that margin above 0 (see
            :func:`~beamfeat.expression.pole_margins`). At the end, a kept
            expression with fitted constants gives way to a near copy with
            fewer when the snapping test fails to reject the null that the
            copy fits as well, so that ``log(x0)`` is returned rather than
            ``(x0)^(-0.02)``. The fit sees only the rows the search is given.
        magnitude_slots: Beam slots added at each depth for candidates
            ranked by how well their magnitude tracks the target's, for
            laws whose parts do not correlate with the target. ``0``
            (default) adds none and leaves the search unchanged. With
            centred inputs, ``x0`` and ``x1`` do not correlate with
            ``x0*x1``, nor does ``x0*x1`` with ``x0*x1*x2``, so the ranking
            by score leaves them among noise; but the magnitude of a factor
            tracks the magnitude of the product. The magnitude score of a
            candidate ``c`` is the absolute correlation of
            ``log(1 + |c|/rms(c))`` with the same transform of the centred
            target, or of its residual on the expressions kept so far,
            whichever is larger; ``rms`` is the root mean square. When those
            expressions outnumber a quarter of the rows, the residual on
            that many of the best-scoring ones counts too: a residual on as
            many columns as rows is empty. After the
            ``beam_width`` candidates the score keeps, up to this many more
            are kept from those it left out, best magnitude score first and
            subject to the same redundancy check, at every depth but the
            last, which has no next depth to build on. At the first depth, when
            ``max_candidates_per_depth`` can bind, the inputs are proposed in
            an order that interleaves their magnitude ranking with their
            score ranking, at a share of ``magnitude_slots / (beam_width +
            magnitude_slots)`` and starting with the best magnitude score, so
            that a pair of inputs neither of which correlates with the target
            is still proposed. Regression only: a class label carries no
            magnitude. The score sees only the rows the search is given, and
            of more than :data:`MAGNITUDE_ROWS` of them, that many evenly
            spaced.
        residual_slots: Places added to the final ranking for the first
            terms of a greedy forward path through every candidate the
            search kept, each the one that most increases the R^2 of a
            least-squares fit on those before it; ``0`` (default) adds none.
            A dominant term fills the ranking by score with restatements of
            itself, and a weaker term beside it is cut; a restatement adds
            nothing to a fit that holds the term it restates, so the path
            takes the weaker term next. A path term takes the place of a
            ranked candidate that is a near copy of it, and is dropped if it
            is a near copy of an earlier path term; the ranking otherwise
            keeps every candidate the score kept. At most a quarter of the
            rows. Before the final path, the
            operand of each path term that is an input column or a transform
            of one operand, and that operand's transforms by ``unary_ops``,
            join the candidates: transforms of one operand can correlate past
            ``redundancy_threshold``, as ``log(x)``, ``sqrt(x)`` and ``1/x``
            do on a positive range of small spread, and a depth then keeps
            only the best-scoring of them. Regression only. The path
            sees only the rows the search is given, and of more than
            :data:`RESIDUAL_ROWS` of them (fewer on a wide table, so that
            the candidate values stay within :data:`RESIDUAL_BYTES`), that
            many evenly spaced.

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
        max_inputs: int | None = None,
        fitted_ops: Sequence[str] = (),
        magnitude_slots: int = 0,
        residual_slots: int = 0,
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
        if max_inputs is not None and (
            isinstance(max_inputs, bool) or not isinstance(max_inputs, numbers.Integral) or max_inputs < 1
        ):
            raise ValueError(f"max_inputs must be an integer of at least 1 or None, got {max_inputs!r}")
        if (
            isinstance(magnitude_slots, bool)
            or not isinstance(magnitude_slots, numbers.Integral)
            or magnitude_slots < 0
        ):
            raise ValueError(f"magnitude_slots must be a non-negative integer, got {magnitude_slots!r}")
        if magnitude_slots and problem_type == "classification":
            raise ValueError(
                "magnitude_slots is for regression only: a class label has no magnitude for a "
                "candidate's to track. Leave it at 0 for classification."
            )
        if (
            isinstance(residual_slots, bool)
            or not isinstance(residual_slots, numbers.Integral)
            or residual_slots < 0
        ):
            raise ValueError(f"residual_slots must be a non-negative integer, got {residual_slots!r}")
        if residual_slots and problem_type == "classification":
            raise ValueError(
                "residual_slots is for regression only: the forward path fits the target by "
                "least squares. Leave it at 0 for classification."
            )

        unknown = [op for op in (*unary_ops, *binary_ops) if op not in OPERATORS]
        if unknown:
            available = sorted(op for op, spec in OPERATORS.items() if not spec.parametric)
            raise ValueError(f"unknown operator(s): {sorted(unknown)}; available: {available}")
        bad_unary = [op for op in unary_ops if OPERATORS[op].arity != 1]
        if bad_unary:
            raise ValueError(f"not unary operator(s): {sorted(bad_unary)}")
        bad_binary = [op for op in binary_ops if OPERATORS[op].arity != 2]
        if bad_binary:
            raise ValueError(f"not binary operator(s): {sorted(bad_binary)}")
        fitted_ops = _check_fitted_ops(fitted_ops)
        needs_constant = [op for op in (*unary_ops, *binary_ops) if OPERATORS[op].parametric]
        if needs_constant:
            raise ValueError(
                f"operator(s) {sorted(needs_constant)} take a fitted constant; list them in "
                "fitted_ops rather than unary_ops"
            )

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
        self.max_inputs = max_inputs

        self._constrained = max_inputs is not None
        # The input columns of each operand, by node key, for max_inputs; None
        # marks a node over the limit. Filled during a run and cleared at its
        # start and end.
        self._profiles: dict[str, frozenset[str] | None] = {}
        self.fitted_ops = fitted_ops
        self.magnitude_slots = int(magnitude_slots)
        self.residual_slots = int(residual_slots)

        self.result_: SearchResult | None = None

    # -- proposal ----------------------------------------------------------- #

    def _propose(
        self,
        beam: list[Node],
        pool: list[Node],
        seen: set[Node],
    ) -> tuple[list[Node], int, int]:
        """Construct candidates one level above the current beam.

        Unary operators are applied to each beam member; binary operators
        combine each beam member with each pool member. Candidates already
        seen are skipped, so the DAG's structural unification does the
        deduplication rather than a separate pass. Candidates that break
        ``max_inputs`` are skipped too; they are not added to ``seen`` and do
        not count towards the ceiling.

        Returns:
            Tuple of (new candidate nodes, count rejected for unit mismatch,
            count of distinct candidates rejected by the constraints).
        """
        candidates: list[Node] = []
        n_unit_rejected = 0
        ceiling = self.max_candidates_per_depth
        constrained = self._constrained
        rejected: set[Node] = set()

        for node in beam:
            for op in self.unary_ops:
                if len(candidates) >= ceiling:
                    return candidates, n_unit_rejected, len(rejected)
                try:
                    candidate = transform(op, node)
                except UnitRegistryError:
                    raise
                except UnitError:
                    n_unit_rejected += 1
                    continue
                if candidate not in seen:
                    if constrained and not self._allowed(candidate):
                        rejected.add(candidate)
                        continue
                    seen.add(candidate)
                    candidates.append(candidate)

        for node in beam:
            for partner in pool:
                if node == partner:
                    continue
                for op in self.binary_ops:
                    if len(candidates) >= ceiling:
                        return candidates, n_unit_rejected, len(rejected)
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
                        if constrained and not self._allowed(candidate):
                            rejected.add(candidate)
                            continue
                        seen.add(candidate)
                        candidates.append(candidate)

        return candidates, n_unit_rejected, len(rejected)

    # -- constraints -------------------------------------------------------- #

    def _allowed(self, node: Node) -> bool:
        """Whether a constructed candidate satisfies ``max_inputs``.

        Takes the node as constructed, after any rewrite by
        :func:`~beamfeat.expression.transform` or
        :func:`~beamfeat.expression.combine`, since that is the expression
        that would be evaluated. The input columns of each operand are
        computed once and cached by node key, so a candidate costs a set union
        rather than a walk of its tree.
        """
        if not self._constrained:
            return True
        return self._profile(node) is not None

    def _profile(self, node: Node) -> frozenset[str] | None:
        """Input columns of ``node``, or ``None`` if it uses more than ``max_inputs``.

        Operands are looked up in, or added to, the cache; ``node`` itself is
        not cached.
        """
        if not node.children:
            # An input column; a childless node without one reads no input.
            return frozenset(()) if node.column is None else frozenset((str(node.column),))
        below = []
        for child in node.children:
            columns = self._operand_profile(child)
            if columns is None:
                return None
            below.append(columns)
        columns = frozenset().union(*below)
        if self.max_inputs is not None and len(columns) > self.max_inputs:
            return None
        return columns

    def _operand_profile(self, node: Node) -> frozenset[str] | None:
        """:meth:`_profile` of an operand, cached by node key."""
        key = node.key
        if key not in self._profiles:
            self._profiles[key] = self._profile(node)
        return self._profiles[key]

    def _can_reach_ceiling(self, leaves: list[Node]) -> bool:
        """Whether proposing from ``leaves`` alone can reach the depth ceiling.

        Counts every operator application, and each unordered pair once for a
        commutative operator, so the count bounds the distinct candidates.
        Candidates ``max_inputs`` would reject are counted as well, so with it
        the bound can overestimate. The caller then proposes in score order
        first, finds that the ceiling did not bind, and proposes again in
        column order: a second pass, but the same result.
        """
        n = len(leaves)
        n_proposals = n * len(self.unary_ops)
        for op in self.binary_ops:
            n_pairs = n * (n - 1)
            n_proposals += n_pairs // 2 if OPERATORS[op].commutative else n_pairs
        return n_proposals >= self.max_candidates_per_depth

    # -- fitted constants --------------------------------------------------- #

    def _fit(
        self,
        parents: list[Node],
        values: np.ndarray,
        prepared: np.ndarray | None,
        seen: set[Node],
        n_partners: int | None = None,
    ) -> tuple[list[Node], int]:
        """Fit each of ``fitted_ops`` to each parent.

        Each fit gives the parent with the operator applied, one level above
        it. A constant fitted as a factor of another parent (see
        :func:`~beamfeat.parametric.fit_constants`) also gives that product,
        one level higher still, when ``"mul"`` is among ``binary_ops`` and
        the product stays within ``max_depth``.

        Args:
            parents: Expressions to fit on: the beam members, best first,
                then any further parents (see :meth:`_extra_parents`).
            values: Their values, one column per parent.
            prepared: The standardised target, or ``None`` when it is
                constant and there is nothing to fit.
            seen: Nodes already proposed. New fitted nodes are added.
            n_partners: How many of the leading parents a fitted node may
                be multiplied by: the beam members. Defaults to all.

        Returns:
            Tuple of (new fitted nodes, count rejected for unit mismatch).
        """
        fitted: list[Node] = []
        n_unit_rejected = 0
        if prepared is None:
            return fitted, n_unit_rejected

        def add(node: Node) -> None:
            if node.depth <= self.max_depth and node not in seen:
                seen.add(node)
                fitted.append(node)

        partners = values if n_partners is None else values[:, :n_partners]
        # The parents each operator applies to, by the units, in order.
        applicable: dict[str, list[int]] = {op: [] for op in self.fitted_ops}
        for index, parent in enumerate(parents):
            for op in self.fitted_ops:
                spec = OPERATORS[op]
                try:
                    if spec.unit_fn is not None:
                        spec.unit_fn(parent.unit)
                except UnitRegistryError:
                    raise
                except UnitError:
                    n_unit_rejected += 1
                    continue
                applicable[op].append(index)
        # Each operator is fitted to all its parents at once; the nodes are
        # then added parent by parent, as one fit at a time would add them.
        # A product with a partner is two levels above the parent.
        fits: dict[tuple[str, int], list] = {}
        for op, indices in applicable.items():
            if not indices:
                continue
            products = ["mul" in self.binary_ops and parents[i].depth + 2 <= self.max_depth for i in indices]
            found = fit_constants_many(
                OPERATORS[op], values[:, indices], prepared, partners if any(products) else None, products
            )
            fits.update(zip(((op, index) for index in indices), found))
        for index, parent in enumerate(parents):
            for op in self.fitted_ops:
                for constant, partner in fits.get((op, index), []):
                    try:
                        node = transform(op, parent, const=constant)
                    except NodeError:
                        continue
                    add(node)
                    if partner is not None:
                        try:
                            add(combine("mul", parents[partner], node))
                        except NodeError:
                            continue
        return fitted, n_unit_rejected

    def _extra_parents(
        self,
        evaluated: list[Node],
        scores: np.ndarray,
        beam: list[Node],
        evaluator: Evaluator,
    ) -> tuple[list[Node], np.ndarray]:
        """Candidates of a round worth fitting on although the beam dropped them.

        The redundancy check judges candidates by linear correlation, and a
        near-duplicate under it need not stay one under a nonlinear
        transform: ``x0*x1`` and ``x0 + x1`` correlate at 0.95 on the unit
        square, yet only the first can be the denominator of
        ``(x0 + x1)/(1 + x0*x1)``. The round's ``beam_width`` best candidates,
        ranked as the beam ranks them but before the redundancy check, are
        therefore fitted on too; those in the beam already are skipped, so a
        depth fits on at most twice ``beam_width`` parents. Their values are
        evaluated again, since a round's values are not all kept.

        Returns:
            Tuple of (parents not in the beam, best first, their values).
        """
        in_beam = set(beam)
        order = _rank(evaluated, scores)[: self.beam_width]
        extras = [evaluated[i] for i in order if evaluated[i] not in in_beam]
        values = np.empty((evaluator.n_rows, len(extras)), dtype=evaluator.dtype)
        kept: list[Node] = []
        for node in extras:
            column = evaluator.evaluate(node)
            if column is not None:  # pragma: no branch - evaluated this round
                values[:, len(kept)] = column
                kept.append(node)
        return kept, values[:, : len(kept)]

    # -- redundancy --------------------------------------------------------- #

    def _prune_redundant(
        self,
        nodes: list[Node],
        values: np.ndarray,
        scores: np.ndarray,
        limit: int,
        shield: bool = False,
    ) -> tuple[list[Node], np.ndarray, np.ndarray, int]:
        """Greedily admit candidates in score order, skipping near-duplicates.

        A candidate is admitted if its absolute correlation against every
        already-admitted candidate stays below ``redundancy_threshold``.
        Correlations within :data:`EXACT_COPY_TOL` of 1 count as reaching it.
        With ``shield``, see :meth:`_admit`.

        Returns:
            Tuple of (kept nodes, their values, their scores, count dropped).
        """
        if not nodes:
            return [], np.empty((values.shape[0], 0)), np.empty(0), 0

        order = _rank(nodes, scores)
        standardised = _standardise_columns(values)
        kept_indices, n_dropped = self._admit(
            order, lambda index: standardised[:, index], limit, values.shape[0], _fitted_mask(nodes, shield)
        )
        kept_nodes = [nodes[i] for i in kept_indices]
        return kept_nodes, values[:, kept_indices], scores[kept_indices], n_dropped

    def _prune_evaluated(
        self,
        nodes: list[Node],
        scores: np.ndarray,
        limit: int,
        evaluator: Evaluator,
        shield: bool = False,
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
        kept_indices, n_dropped = self._admit(
            order, columns.standardised, limit, n_samples, _fitted_mask(nodes, shield)
        )
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
        fitted: np.ndarray | None = None,
    ) -> tuple[list[int], int]:
        """Walk candidates in ``order``, admitting each that duplicates none admitted.

        Args:
            order: Candidate indices, best first.
            standardised: Returns a candidate's standardised column, given
                its index. Called in ``order``.
            limit: Most candidates to admit.
            n_samples: Rows per column.
            fitted: Optional flag per candidate, true for one with a fitted
                constant. When given, a candidate without one is checked only
                against admitted candidates without one. A fitted node is its
                parent tuned to the target, so it correlates closely with the
                parent's constant-free transforms, such as ``x^2`` beside
                ``x^1.69``; were it to mask them, a depth would trade the
                building blocks of deeper expressions (``exp(-x^2/2)`` is
                built on ``x^2``) for one fit. Fitted candidates are still
                checked against everything admitted, and take beam slots
                like any other.

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
                if fitted is not None and not fitted[index]:
                    correlations = np.where(fitted[kept_indices], 0.0, correlations)
                if float(np.max(correlations)) >= cutoff:
                    n_dropped += 1
                    continue
            else:
                admitted = np.empty((min(limit, len(order)), n_samples), dtype=column.dtype)
            admitted[len(kept_indices)] = column
            kept_indices.append(int(index))

        return kept_indices, n_dropped

    # -- magnitude slots ---------------------------------------------------- #

    def _magnitude_references(
        self, target: np.ndarray, incumbent: np.ndarray | None, incumbent_scores: Sequence[float] = ()
    ) -> list[np.ndarray]:
        """Log-magnitudes of the centred target and of its residuals, standardised.

        The residuals are the target's on the incumbent set, as
        :class:`~beamfeat.scoring.CorrelationScorer` computes them, and, when
        the set holds more columns than a quarter of the rows, on that many
        of its best-scoring columns. A residual on as many columns as there
        are rows is empty, and on nearly as many it is mostly the fit's own
        noise; the shorter one still shows what the best columns leave
        unexplained, such as ``x0*x1`` behind ``x2`` in ``x0*x1 + x2`` on a
        table with more columns than rows. An empty residual is left out.
        Each is fitted on every row and kept on the rows
        :func:`_magnitude_rows` names.
        """
        scorer = CorrelationScorer(self.problem_type)
        prepared = [scorer.prepare(target)]
        if incumbent is not None:
            prepared.append(scorer.prepare(target, incumbent))
            limit = max(1, len(target) // 4)
            if incumbent.shape[1] > limit and len(incumbent_scores) == incumbent.shape[1]:
                order = np.argsort(-np.asarray(incumbent_scores, dtype=np.float64), kind="stable")
                prepared.append(scorer.prepare(target, incumbent[:, np.sort(order[:limit])]))
        rows = _magnitude_rows(len(target))
        references = []
        for vector in prepared:
            if vector is not None:
                if rows is not None:
                    vector = _standardise(vector[rows])
                # A standardised vector has a root mean square of 1.
                reference = _standardise(np.log1p(np.abs(vector)))
                if np.any(reference):
                    references.append(reference)
        return references

    def _weave(
        self, ranked: list[Node], leaves: list[Node], values: np.ndarray, scores: np.ndarray, target: np.ndarray
    ) -> list[Node]:
        """The inputs in score order, with their magnitude ranking interleaved.

        The magnitude ranking takes the first place and then its share,
        ``magnitude_slots / (beam_width + magnitude_slots)``, of the places
        that follow, skipping inputs already placed. Its references are those
        of the first depth, the inputs being the incumbent set there.
        """
        references = self._magnitude_references(target, values, scores)
        if not references:
            return ranked
        rows = _magnitude_rows(values.shape[0])
        order = _rank(leaves, magnitude_scores(values if rows is None else values[rows], references))
        by_magnitude = [leaves[i] for i in order]
        share = self.magnitude_slots / (self.beam_width + self.magnitude_slots)
        woven: list[Node] = []
        placed: set[Node] = set()
        lists = {True: iter(by_magnitude), False: iter(ranked)}
        n_magnitude = 0
        while len(woven) < len(ranked):
            from_magnitude = n_magnitude < math.ceil(share * (len(woven) + 1) - 1e-12)
            for node in lists[from_magnitude]:
                if node not in placed:
                    placed.add(node)
                    woven.append(node)
                    n_magnitude += from_magnitude
                    break
            else:  # pragma: no cover - both lists hold every input
                break
        return woven

    def _reserve(
        self,
        nodes: list[Node],
        scores: np.ndarray,
        magnitude: np.ndarray,
        beam: list[Node],
        beam_values: np.ndarray,
        evaluator: Evaluator,
    ) -> tuple[list[Node], np.ndarray, np.ndarray]:
        """Up to ``magnitude_slots`` candidates the beam left out, best magnitude first.

        Each is checked against the beam and against those reserved before it
        with the beam's redundancy threshold. Columns are evaluated again in
        the order they are needed, as the beam's are. Only the head of the
        ranking is built, a few times the slots at first and twice as much
        whenever it runs out; it is the head of the ranking of the whole
        round, ties included.

        Returns:
            Tuple of (reserved nodes, their values, their scores).
        """
        n_samples = evaluator.n_rows
        in_beam = set(beam)
        cutoff = min(self.redundancy_threshold, 1.0 - EXACT_COPY_TOL)
        admitted = np.empty((len(beam) + self.magnitude_slots, n_samples))
        admitted[: len(beam)] = _standardise_columns(beam_values).T
        n_admitted = len(beam)
        kept: list[int] = []
        values: list[np.ndarray] = []
        done = 0
        size = len(beam) + 4 * self.magnitude_slots
        while len(kept) < self.magnitude_slots and done < len(nodes):
            head = _rank_head(nodes, magnitude, size)
            order = np.array([i for i in head[done:] if nodes[i] not in in_beam], dtype=np.intp)
            done = len(head)
            size *= 2
            if not order.size:
                continue
            columns = _RankedColumns(order, nodes, evaluator, block=max(2, min(self.magnitude_slots, 256)))
            for index in order:
                if len(kept) >= self.magnitude_slots:
                    break
                column = columns.standardised(index)
                if n_admitted and float(np.max(np.abs(admitted[:n_admitted] @ column))) / n_samples >= cutoff:
                    continue
                admitted[n_admitted] = column
                n_admitted += 1
                kept.append(int(index))
                values.append(np.array(columns.raw(index), dtype=evaluator.dtype))
        if not kept:
            return [], np.empty((n_samples, 0)), np.empty(0)
        return [nodes[i] for i in kept], np.column_stack(values), scores[kept]

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
        magnitude: tuple[list[np.ndarray], np.ndarray | None] | None = None,
    ) -> tuple[list[Node], np.ndarray, np.ndarray | None, np.ndarray | None]:
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
            magnitude: References for :func:`magnitude_scores` and the rows
                they are on (``None`` for all), or ``None`` when no magnitude
                score is needed.

        Returns:
            Tuple of (evaluated nodes, their scores, their values as a
            matrix, or ``None`` when the values were dropped, and their
            magnitude scores, or ``None`` without ``magnitude``).
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
            [(batch, scorer, references, magnitude) for batch in batches],
            n_jobs,
            prefer="processes" if expensive else "threads",
        )

        evaluated: list[Node] = []
        score_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        magnitude_parts: list[np.ndarray] = []
        for chunk, (outcome, chunk_scores, chunk_magnitude) in zip(chunks, outcomes, strict=True):
            evaluated.extend(evaluator.commit_batch(chunk, outcome))
            if by_column:
                score_parts.append(chunk_scores)
            else:
                value_parts.append(outcome.values)
            if chunk_magnitude is not None:
                magnitude_parts.append(chunk_magnitude)

        if not evaluated:
            return evaluated, np.empty(0), None, None
        magnitudes = np.concatenate(magnitude_parts) if magnitude is not None else None
        if by_column:
            scores = np.concatenate(score_parts) if score_parts else np.empty(0)
            return evaluated, scores, None, magnitudes

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
        return evaluated, scores, values, magnitudes

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
        self._profiles.clear()

        scorer = make_scorer(self.scorer, problem_type=self.problem_type, random_state=self.random_state)
        absolute = scorer.prepare(target) if scorer.scores_by_column() else None
        # Constants are fitted by correlation with the raw target, whatever
        # the scorer; see beamfeat.parametric.
        fit_target = CorrelationScorer(self.problem_type).prepare(target) if self.fitted_ops else None

        state = self._depth_zero(evaluator, scorer, target, started)
        for depth in range(1, self.max_depth + 1):
            if not self._run_depth(depth, state, evaluator, scorer, target, absolute, fit_target):
                break

        if self.residual_slots:
            n_siblings = self._path_siblings(state, evaluator, target)
            if n_siblings:
                report(self.verbose, 2, f"  residual slots: {n_siblings} transforms of path operands added")
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
            scorer.score_batch(state.selected_values, target, None)
            if state.selected_values.shape[1]
            else np.asarray(state.selected_scores, dtype=np.float64)
        )
        result = self._finalise(
            state.selected, state.selected_values, final_scores, evaluator, state.trace, started, fit_target,
            target,
        )
        result.n_proposed_total = state.n_proposed_total
        result.n_evaluated_total = state.n_evaluated_total
        self.result_ = result
        self._profiles.clear()
        depth_reached = state.trace[-1].depth
        report(
            self.verbose,
            1,
            f"search: depth {depth_reached}"
            + (f" of {self.max_depth}" if depth_reached < self.max_depth else "")
            + f", beam {self.beam_width} -> "
            f"{state.n_proposed_total} proposed, {len(result.nodes)} candidates "
            f"({result.elapsed:.2f}s)",
        )
        return result

    def _depth_zero(
        self, evaluator: Evaluator, scorer: Scorer, target: np.ndarray, started: float
    ) -> _SearchState:
        """Score the input columns, the beam of depth 0, and start the search's state."""
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
        # Fitted nodes waiting for the round of their depth, and the parents
        # beyond the beam that the next round fits on.
        waiting: dict[int, list[Node]] = {}
        extra_parents: list[Node] = []
        extra_values = np.empty((evaluator.n_rows, 0))
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
        return _SearchState(
            beam=beam,
            beam_values=beam_values,
            beam_scores=beam_scores,
            pool=pool,
            selected=selected,
            selected_values=selected_values,
            selected_scores=selected_scores,
            seen=seen,
            waiting=waiting,
            extra_parents=extra_parents,
            extra_values=extra_values,
            trace=trace,
            n_proposed_total=n_proposed_total,
            n_evaluated_total=n_evaluated_total,
        )

    def _run_depth(
        self,
        depth: int,
        state: _SearchState,
        evaluator: Evaluator,
        scorer: Scorer,
        target: np.ndarray,
        absolute: np.ndarray | None,
        fit_target: np.ndarray | None,
    ) -> bool:
        """One depth of the search: propose, score, keep a beam, and record it.

        Updates ``state``; returns ``False`` when the search should stop
        here: no new candidate, none that survives evaluation, or an empty
        beam.
        """
        depth_started = time.perf_counter()

        candidates, n_unit_rejected, n_constrained, n_fitted = self._depth_candidates(
            depth, state, target, fit_target
        )
        state.n_proposed_total += len(candidates)
        if not candidates:
            report(self.verbose, 2, f"  depth {depth}: no new candidates; stopping")
            return False

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
        incumbent = state.selected_values if state.selected_values.shape[1] else None
        # The slots keep parts for the next depth to build on; the last
        # depth has none. A target whose magnitude does not vary, such as
        # one of two values either side of its mean, gives them nothing.
        magnitude = None
        if self.magnitude_slots and depth < self.max_depth:
            references = self._magnitude_references(target, incumbent, state.selected_scores)
            if references:
                magnitude = (references, _magnitude_rows(evaluator.n_rows))
        evaluated, scores, values, magnitudes = self._score_round(
            evaluator, candidates, scorer, target, absolute, incumbent, magnitude
        )
        n_numeric_rejected = len(candidates) - len(evaluated)
        state.n_evaluated_total += len(evaluated)
        if not evaluated:
            report(self.verbose, 2, f"  depth {depth}: nothing survived evaluation; stopping")
            return False

        if values is None:
            state.beam, state.beam_values, kept_scores, n_redundant = self._prune_evaluated(
                evaluated, scores, self.beam_width, evaluator, shield=bool(self.fitted_ops)
            )
        else:
            state.beam, state.beam_values, kept_scores, n_redundant = self._prune_redundant(
                evaluated, values, scores, self.beam_width, shield=bool(self.fitted_ops)
            )
        del values
        n_magnitude = 0
        if magnitudes is not None and state.beam:
            reserved, reserved_values, reserved_scores = self._reserve(
                evaluated, scores, magnitudes, state.beam, state.beam_values, evaluator
            )
            if reserved:
                n_magnitude = len(reserved)
                state.beam = state.beam + reserved
                state.beam_values = np.column_stack([state.beam_values, reserved_values])
                kept_scores = np.concatenate([kept_scores, reserved_scores])
        if not state.beam:
            report(self.verbose, 2, f"  depth {depth}: beam collapsed after redundancy pruning; stopping")
            return False
        if self.fitted_ops and depth < self.max_depth:
            state.extra_parents, state.extra_values = self._extra_parents(
                evaluated, scores, state.beam, evaluator
            )

        state.selected.extend(state.beam)
        state.selected_scores.extend(float(s) for s in kept_scores)
        state.selected_values = (
            state.beam_values
            if state.selected_values.shape[1] == 0
            else np.column_stack([state.selected_values, state.beam_values])
        )

        state.pool = state.pool + state.beam

        state.trace.append(
            SearchTrace(
                depth=depth,
                n_proposed=len(candidates),
                n_evaluated=len(evaluated),
                n_rejected_units=n_unit_rejected,
                n_rejected_numeric=n_numeric_rejected,
                n_rejected_redundant=n_redundant,
                n_kept=len(state.beam),
                best_score=float(np.max(kept_scores)),
                elapsed=time.perf_counter() - depth_started,
                n_rejected_constraints=n_constrained,
                n_fitted=n_fitted,
                n_magnitude=n_magnitude,
            )
        )
        self._report_depth(state.trace[-1], depth_started)
        return True

    def _depth_candidates(
        self, depth: int, state: _SearchState, target: np.ndarray, fit_target: np.ndarray | None
    ) -> tuple[list[Node], int, int, int]:
        """The candidates of one depth: proposed from the beam, and fitted when ``fitted_ops`` is set.

        Returns them with the number rejected for their units, the number
        rejected by ``max_inputs``, and the number carrying a constant
        fitted at this depth.
        """
        # Later beams come out of _prune_redundant in score order, but the
        # leaves are in column order, so a ceiling that binds at depth 1
        # would cut the last columns of the table rather than the weakest.
        # Proposal order also decides between exact ties such as (a - b)
        # and (b - a), so a round the ceiling does not reach keeps column
        # order and only a truncated one is proposed in score order. Both
        # orders propose the same set, so whether the ceiling binds does
        # not depend on which is tried first; the score order is tried
        # first when the ceiling can bind, which it usually then does.
        if depth == 1 and self._can_reach_ceiling(state.beam):
            ranked = [state.beam[i] for i in _rank(state.beam, state.beam_scores)]
            if self.magnitude_slots:
                ranked = self._weave(ranked, state.beam, state.beam_values, state.beam_scores, target)
            candidates, n_unit_rejected, n_constrained = self._propose(ranked, ranked, state.seen)
            if len(candidates) < self.max_candidates_per_depth:
                state.seen.difference_update(candidates)
                candidates, n_unit_rejected, n_constrained = self._propose(state.beam, state.pool, state.seen)
        else:
            candidates, n_unit_rejected, n_constrained = self._propose(state.beam, state.pool, state.seen)
        n_fitted = 0
        if self.fitted_ops:
            if depth == 1:
                order = _rank(state.beam, state.beam_scores)[: self.beam_width]
                parents, parent_values = [state.beam[i] for i in order], state.beam_values[:, order]
            else:
                parents = state.beam + state.extra_parents
                parent_values = (
                    np.column_stack([state.beam_values, state.extra_values])
                    if state.extra_parents
                    else state.beam_values
                )
            fitted, n_fit_unit_rejected = self._fit(
                parents,
                parent_values,
                fit_target,
                state.seen,
                n_partners=len(state.beam) if depth > 1 else None,
            )
            n_unit_rejected += n_fit_unit_rejected
            allowed = [node for node in fitted if self._allowed(node)]
            n_constrained += len(fitted) - len(allowed)
            fitted = allowed
            # A fitted node joins the round of its own depth: a product
            # with a partner is a level above this one, and waits for it.
            for node in fitted:
                state.waiting.setdefault(max(node.depth, depth), []).append(node)
            arrived = state.waiting.pop(depth, [])
            candidates = candidates + arrived
            n_fitted = len(arrived)
        return candidates, n_unit_rejected, n_constrained, n_fitted

    def _report_depth(self, entry: SearchTrace, depth_started: float) -> None:
        """The verbose report of one depth, from its trace record."""
        dropped = []
        if entry.n_rejected_units:
            dropped.append(f"{entry.n_rejected_units} unit")
        if entry.n_rejected_constraints:
            dropped.append(f"{entry.n_rejected_constraints} constraint")
        if entry.n_rejected_numeric:
            dropped.append(f"{entry.n_rejected_numeric} numeric")
        if entry.n_rejected_redundant:
            dropped.append(f"{entry.n_rejected_redundant} near-duplicate")
        report(
            self.verbose,
            2,
            f"  depth {entry.depth}: {entry.n_proposed} proposed"
            + (f" ({entry.n_fitted} with a fitted constant)" if entry.n_fitted else "")
            + f" -> {entry.n_evaluated} evaluated"
            f" -> {entry.n_kept} kept by beam"
            + (f" ({entry.n_magnitude} by magnitude)" if entry.n_magnitude else "")
            + (f" ({', '.join(dropped)} rejected)" if dropped else "")
            + f", best {entry.best_score:.4f}"
            f", {time.perf_counter() - depth_started:.2f}s",
        )

    def _finalise(
        self,
        nodes: list[Node],
        values: np.ndarray,
        scores: np.ndarray,
        evaluator: Evaluator,
        trace: list[SearchTrace],
        started: float,
        fit_target: np.ndarray | None = None,
        target: np.ndarray | None = None,
    ) -> SearchResult:
        """Rank, deduplicate, and truncate the accumulated selection.

        With ``residual_slots``, the terms of a forward path join the set;
        see :meth:`_add_path`. With ``fitted_ops``, a kept feature with
        fitted constants then gives way to a near copy with fewer that fits
        as well; see :meth:`_constant_free_copies`.
        """
        limit = self.max_features if self.max_features is not None else self.beam_width

        n_redundant = 0
        if nodes:
            kept, _, kept_scores, n_redundant = self._prune_redundant(nodes, values, scores, limit)
            if self.residual_slots and target is not None:
                kept, kept_scores = self._add_path(nodes, values, scores, target, limit, kept)
            if self.fitted_ops and fit_target is not None:
                kept, kept_scores = self._constant_free_copies(nodes, values, scores, kept, fit_target)
            nodes, scores = kept, kept_scores
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

    def _path(self, nodes: list[Node], values: np.ndarray, target: np.ndarray, limit: int) -> list[int]:
        """The first ``residual_slots`` terms of a greedy forward path through ``values``.

        At most ``limit`` and a quarter of the rows; see
        :func:`~beamfeat._forward.forward_order`. Gains within twice
        :data:`SCORE_TIE_TOL` of the largest, relative to it, are tied, and
        the structurally simpler expression goes first, as in :func:`_rank`.
        Of more rows than :data:`RESIDUAL_ROWS`, or than keep the values
        within :data:`RESIDUAL_BYTES`, that many evenly spaced.
        """
        memo: dict[str, tuple[int, int]] = {}
        sizes = np.array([sum(_counts(node, memo)) for node in nodes])
        n_rows = values.shape[0]
        cap = max(MAGNITUDE_ROWS, min(RESIDUAL_ROWS, RESIDUAL_BYTES // (8 * max(1, values.shape[1]))))
        if n_rows > cap:
            rows = np.unique(np.round(np.linspace(0, n_rows - 1, cap)).astype(np.intp))
            values, target = values[rows], target[rows]
        steps = min(self.residual_slots, limit, max(1, values.shape[0] // 4))
        return forward_order(values, target, steps, priority=sizes, tie=2 * SCORE_TIE_TOL)

    def _path_siblings(self, state: _SearchState, evaluator: Evaluator, target: np.ndarray) -> int:
        """Add the other transforms of each forward-path term's operand to the candidates.

        Transforms of one operand can correlate past ``redundancy_threshold``
        (``log(x)``, ``sqrt(x)`` and ``1/x`` on a positive range of small
        spread), so that a depth keeps the best-scoring and drops the rest,
        although another of them is the one that fits what the leading terms
        leave. For each term of the path that is an input column or a
        transform of one operand, the operand and its transforms by
        ``unary_ops`` join the candidates, so that the final path chooses
        among them. Returns the number added.
        """
        path = self._path(state.selected, state.selected_values, target, len(state.selected))
        present = set(state.selected)
        siblings: list[Node] = []
        for index in path:
            node = state.selected[index]
            if not node.children:
                operand = node
            elif len(node.children) == 1:
                operand = node.children[0]
            else:
                continue
            options = [operand] if operand.children or self.include_originals else []
            if operand.depth < self.max_depth:
                for op in self.unary_ops:
                    try:
                        options.append(transform(op, operand))
                    except UnitRegistryError:
                        raise
                    except UnitError:
                        continue
            for option in options:
                if option in present or (self._constrained and not self._allowed(option)):
                    continue
                present.add(option)
                siblings.append(option)
        added, values = evaluator.evaluate_many(siblings)
        if added:
            state.selected.extend(added)
            state.selected_scores.extend([0.0] * len(added))
            state.selected_values = np.column_stack([state.selected_values, values])
        return len(added)

    def _add_path(
        self,
        nodes: list[Node],
        values: np.ndarray,
        scores: np.ndarray,
        target: np.ndarray,
        limit: int,
        kept: list[Node],
    ) -> tuple[list[Node], np.ndarray]:
        """The final set with the terms of :meth:`_path` added, ranked by score.

        Each path term not already kept joins the set, in path order, and
        takes the place of any kept candidate that is a near copy of it
        (correlation at ``redundancy_threshold``), unless that copy is an
        earlier path term: the later term is then dropped, as the redundancy
        check drops a near copy. The set grows by at most ``residual_slots``,
        and the candidates the score kept stay in it otherwise.
        """
        position = {id(node): index for index, node in enumerate(nodes)}
        chosen = [position[id(node)] for node in kept]
        n_rows = values.shape[0]
        standardised = _standardise_columns(values)
        cutoff = min(self.redundancy_threshold, 1.0 - EXACT_COPY_TOL)
        path_terms: set[int] = set()
        for index in self._path(nodes, values, target, limit):
            if index in chosen:
                path_terms.add(index)
                continue
            copies = np.abs(standardised[:, chosen].T @ standardised[:, index]) / n_rows >= cutoff
            if any(copy and member in path_terms for member, copy in zip(chosen, copies)):
                continue
            chosen = [member for member, copy in zip(chosen, copies) if not copy] + [index]
            path_terms.add(index)
        order = _rank([nodes[index] for index in chosen], scores[chosen])
        return [nodes[chosen[i]] for i in order], scores[chosen][order]

    def _constant_free_copies(
        self,
        nodes: list[Node],
        values: np.ndarray,
        scores: np.ndarray,
        kept: list[Node],
        fit_target: np.ndarray,
    ) -> tuple[list[Node], np.ndarray]:
        """The final set, each feature with fitted constants swapped for a simpler near copy that fits as well.

        A constant fitted on these rows bends an expression into a near copy
        of whatever smooth law the target follows, such as ``(x0)^(-0.02)``
        of ``log(x0)``. It outscores the law here by fitting the noise, so
        the redundancy check keeps it and drops the law, and the equation
        prints the near copy. Each kept feature that carries fitted
        constants is therefore compared with the features the search
        accumulated that are near copies of it (correlation at
        ``redundancy_threshold``), carry fewer constants and are not kept.
        The likelihood-ratio test that snaps a fitted constant to a simple
        value decides, on the same rows and at the same level: the null is
        that the copy fits the target as well, the statistic is
        ``n * log(L_copy / L_kept)`` with ``L = 1 - r^2``, and it is
        rejected above the 99.9% point of a chi-squared variable with one
        degree of freedom per constant the copy does without. Of the copies
        whose null the test fails to reject, the simplest takes the
        feature's place, as the simplest snapped value does: the fewest
        constants, then the fewest operators, then the best fit, so that
        ``log(x0)`` is preferred to its exact copy ``log(sqrt(x0))``. A copy
        that is a near-duplicate of another member is passed over, as the
        redundancy check would pass it over. The set keeps its size and
        every other member, and is ranked again.
        """
        position = {id(node): index for index, node in enumerate(nodes)}
        chosen = [position[id(node)] for node in kept]
        memo: dict[str, tuple[int, int]] = {}
        sizes = [_counts(node, memo) for node in nodes]
        constants = np.array([size[1] for size in sizes])
        if not constants[chosen].any():
            return kept, scores[chosen]
        n_rows = values.shape[0]
        standardised = _standardise_columns(values)
        fits = _correlation_rows(values.T, fit_target)
        cutoff = min(self.redundancy_threshold, 1.0 - EXACT_COPY_TOL)
        taken = set(chosen)
        for slot, index in enumerate(list(chosen)):
            if constants[index] == 0 or not math.isfinite(fits[index]):
                continue
            copies = np.abs(standardised.T @ standardised[:, index]) / n_rows >= cutoff
            others = [chosen[i] for i in range(len(chosen)) if i != slot]
            best = None
            for other in np.flatnonzero(copies & (constants < constants[index])):
                other = int(other)
                if other in taken or not math.isfinite(fits[other]):
                    continue
                # The set admits no near-duplicates; neither may the copy.
                if others and float(np.max(np.abs(standardised[:, others].T @ standardised[:, other]))) / n_rows >= cutoff:
                    continue
                fewer = int(constants[index] - constants[other])
                if _statistic(fits[other], fits[index], n_rows) > _LR_CRITICAL[min(fewer, len(_LR_CRITICAL)) - 1]:
                    continue
                if best is None or (sizes[other][1], sizes[other][0], -fits[other]) < (
                    sizes[best][1], sizes[best][0], -fits[best]
                ):
                    best = other
            if best is not None:
                taken.discard(index)
                taken.add(best)
                chosen[slot] = best
        if taken == {position[id(node)] for node in kept}:
            return kept, scores[chosen]
        swapped = [nodes[index] for index in chosen]
        order = _rank(swapped, scores[chosen])
        return [swapped[i] for i in order], scores[chosen][order]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"BeamSearch(scorer={self.scorer!r}, max_depth={self.max_depth}, "
            f"beam_width={self.beam_width}, problem_type={self.problem_type!r})"
        )


def _rank(nodes: Sequence[Node], scores: np.ndarray) -> np.ndarray:
    """Order candidates best first, as beam admission sees them.

    Scores within SCORE_TIE_TOL of each other - a difference below scoring
    noise - are tied, and the structurally simpler expression (counting
    operators and fitted constants) goes first, so that the more complex
    near-duplicate is the one the correlation check drops. Remaining ties
    keep their given order. lexsort uses its last key as primary and is
    stable.
    """
    memo: dict[str, tuple[int, int]] = {}
    sizes = np.array([sum(_counts(node, memo)) for node in nodes])
    quantised = np.round(np.asarray(scores, dtype=np.float64) / SCORE_TIE_TOL)
    return np.lexsort((sizes, -quantised))


def _magnitude_rows(n_rows: int) -> np.ndarray | None:
    """Rows the magnitude score is computed on: ``None`` for all of them.

    All of them up to :data:`MAGNITUDE_ROWS`; beyond, that many evenly
    spaced through them, the first and the last included.
    """
    if n_rows <= MAGNITUDE_ROWS:
        return None
    return np.unique(np.round(np.linspace(0, n_rows - 1, MAGNITUDE_ROWS)).astype(np.intp))


def _rank_head(nodes: Sequence[Node], scores: np.ndarray, size: int) -> np.ndarray:
    """The first ``size`` or more places of :func:`_rank`, and every tie at the last.

    Ranks only the candidates whose quantised score reaches that of the
    ``size``-th best, so the structural sizes :func:`_rank` breaks ties on
    are computed for those alone.
    """
    if size >= len(nodes):
        return _rank(nodes, scores)
    quantised = np.round(np.asarray(scores, dtype=np.float64) / SCORE_TIE_TOL)
    threshold = np.partition(quantised, len(nodes) - size)[len(nodes) - size]
    chosen = np.flatnonzero(quantised >= threshold)
    return chosen[_rank([nodes[i] for i in chosen], scores[chosen])]


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


def _fitted_mask(nodes: Sequence[Node], shield: bool) -> np.ndarray | None:
    """Per node, whether it carries a fitted constant; ``None`` unless ``shield``."""
    if not shield:
        return None
    memo: dict[str, tuple[int, int]] = {}
    return np.array([_counts(node, memo)[1] > 0 for node in nodes], dtype=bool)


def _counts(node: Node, memo: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """``(node.n_operators, node.n_constants)``, sharing work through ``memo``.

    The properties recurse through the whole tree on every call, and a round
    ranks thousands of candidates built from the same few hundred
    subexpressions; the memo, keyed by structural key, visits each once.
    """
    counts = memo.get(node.key)
    if counts is None:
        if node.is_leaf:
            counts = (0, 0)
        else:
            operators, constants = 1, 0 if node.const is None else 1
            for child in node.children:
                child_operators, child_constants = _counts(child, memo)
                operators += child_operators
                constants += child_constants
            counts = (operators, constants)
        memo[node.key] = counts
    return counts


def _check_fitted_ops(fitted_ops: Sequence[str]) -> tuple[str, ...]:
    """Validate ``fitted_ops`` against the registered parametric operators."""
    if isinstance(fitted_ops, str):
        fitted_ops = (fitted_ops,)
    fitted_ops = tuple(fitted_ops)
    available = parametric_operators()
    unknown = [op for op in fitted_ops if op not in available]
    if unknown:
        raise ValueError(f"unknown fitted operator(s): {sorted(unknown)}; available: {available}")
    if len(set(fitted_ops)) != len(fitted_ops):
        raise ValueError(f"fitted_ops lists an operator more than once: {list(fitted_ops)}")
    return fitted_ops


def _run_chunk(
    batch: EvaluationBatch,
    scorer: Scorer,
    references: tuple[Any, ...] | None,
    magnitude: tuple[list[np.ndarray], np.ndarray | None] | None = None,
) -> tuple[BatchOutcome, np.ndarray | None, np.ndarray | None]:
    """Evaluate one chunk and, given prepared references, score it.

    With references, each candidate's score is the largest against any of
    them, and the values are dropped from the outcome to keep it small.
    With ``magnitude``, references and the rows they are on, the chunk's
    :func:`magnitude_scores` are computed on those rows as well, before the
    values are dropped.
    """
    outcome = batch.run()
    values = outcome.values
    stats = outcome.stats if values.dtype == np.float64 else None
    magnitudes = None
    if magnitude is not None:
        vectors, rows = magnitude
        magnitudes = (
            magnitude_scores(values, vectors, stats) if rows is None else magnitude_scores(values[rows], vectors)
        )
    if references is None:
        return outcome, None, magnitudes
    scores = scorer.score_prepared(values, references[0], stats)
    for prepared in references[1:]:
        scores = np.maximum(scores, scorer.score_prepared(values, prepared, stats))
    return outcome._replace(values=None), scores, magnitudes


def magnitude_scores(
    candidates: np.ndarray, references: Sequence[np.ndarray], stats: ColumnStats | None = None
) -> np.ndarray:
    """How well each candidate's magnitude tracks a reference's, at best.

    Each column ``c`` is transformed to ``log(1 + |c|/rms(c))``, where
    ``rms`` is its root mean square, and scored by its largest absolute
    correlation with any of ``references``, which are standardised
    transforms of the same kind (see ``magnitude_slots`` on
    :class:`BeamSearch`). The logarithm turns a product into a sum, so the
    magnitude of a factor correlates with the magnitude of the product even
    when the factor itself does not correlate with it; the 1 inside it
    treats magnitudes far below the column's scale alike, so that rows where
    a product nears zero, and the noise there, do not dominate, and it
    compresses a heavy tail. The score does not change when a column is
    scaled or changes sign.

    Each column is transformed, standardised and reduced on its own, so a
    score does not depend on the other columns it is computed with.

    Args:
        candidates: ``(n_samples, n_candidates)`` candidate values.
        references: Standardised reference vectors of length ``n_samples``.
        stats: Optional :func:`~beamfeat.scoring.column_stats` of
            ``candidates``, when the caller has them already.

    Returns:
        ``(n_candidates,)`` scores in [0, 1]; 0 for a column that is zero
        throughout or not finite.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim == 1:
        candidates = candidates.reshape(-1, 1)
    n_samples, n_candidates = candidates.shape
    scores = np.zeros(n_candidates)
    if not references:
        return scores
    step = max(1, (16 * 2**20) // max(1, 8 * n_samples))
    for start in range(0, n_candidates, step):
        block = candidates[:, start : start + step]
        block_stats = column_stats(block) if stats is None else stats.take(slice(start, start + step))
        with np.errstate(all="ignore"):
            rms = np.sqrt(block_stats.variance + block_stats.mean**2)
            transformed = np.log1p(np.abs(block) / rms)
        standardised = _standardise_block(transformed, column_stats(transformed))
        # One dot product per column and reference, as the correlation
        # scorer computes its scores: a matrix product sums in another order.
        for offset, column in enumerate(standardised.T):
            best = 0.0
            for reference in references:
                best = max(best, abs(float(column @ reference)))
            scores[start + offset] = best / n_samples
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


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
