"""Tests for guided beam search: recovery, scaling, determinism, units."""

from __future__ import annotations

import numpy as np
import pytest

import beamfeat.search
from beamfeat.expression import Evaluator, combine, leaf, transform
from beamfeat.scoring import CorrelationScorer, GradientBoostingScorer, MutualInformationScorer, Scorer
from beamfeat.search import BeamSearch


@pytest.fixture
def rng():
    return np.random.default_rng(20260723)


@pytest.fixture
def ureg():
    pint = pytest.importorskip("pint")
    return pint.UnitRegistry()


def _positive_data(rng, n=400, k=3):
    """Strictly positive inputs so log/sqrt/reciprocal are all in domain."""
    names = "abcdefgh"[:k]
    return {name: rng.uniform(1.0, 6.0, n) for name in names}


def _found(result, *fragments: str) -> bool:
    """Whether any selected formula contains all the given fragments."""
    return any(all(fragment in name for fragment in fragments) for name in result.names)


# --------------------------------------------------------------------------- #
# Recovery: the headline correctness property
# --------------------------------------------------------------------------- #


class TestRecovery:
    def test_recovers_product(self, rng):
        data = _positive_data(rng)
        target = data["a"] * data["b"] + rng.normal(0, 0.05, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=40).run(data, target)
        assert _found(result, "a", "*", "b")

    def test_recovers_ratio(self, rng):
        data = _positive_data(rng)
        target = data["a"] / data["b"] + rng.normal(0, 0.02, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=40).run(data, target)
        assert _found(result, "a", "/", "b")

    def test_recovers_log(self, rng):
        data = _positive_data(rng)
        target = np.log(data["a"]) * 5.0 + rng.normal(0, 0.05, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=40).run(data, target)
        assert _found(result, "log(a)")

    def test_recovers_square(self, rng):
        data = _positive_data(rng)
        target = data["a"] ** 2 + rng.normal(0, 0.1, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=40).run(data, target)
        assert _found(result, "a", "^2") or _found(result, "a * a")

    def test_recovers_three_way_interaction(self, rng):
        """Depth 2 must reach (a*b)/c, which no depth-1 search can express."""
        data = _positive_data(rng)
        target = (data["a"] * data["b"]) / data["c"] + rng.normal(0, 0.02, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=60).run(data, target)
        assert _found(result, "a", "b", "c")

    def test_recovers_transformed_interaction(self, rng):
        data = _positive_data(rng)
        target = np.log(data["a"]) * data["b"] + rng.normal(0, 0.05, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=60).run(data, target)
        assert _found(result, "log(a)", "b")

    @pytest.mark.parametrize("scorer", ["correlation", "mutual_information"])
    def test_recovery_holds_across_scorers(self, rng, scorer):
        data = _positive_data(rng)
        target = data["a"] * data["b"] + rng.normal(0, 0.05, len(data["a"]))
        result = BeamSearch(scorer=scorer, max_depth=2, beam_width=40).run(data, target)
        assert _found(result, "a", "*", "b")

    def test_recovery_with_gradient_boosting_scorer(self, rng):
        data = _positive_data(rng, n=300)
        target = data["a"] * data["b"] + rng.normal(0, 0.05, 300)
        scorer = GradientBoostingScorer(n_folds=2, max_iter=15, max_depth=2, subsample_size=200)
        result = BeamSearch(scorer=scorer, max_depth=1, beam_width=15).run(data, target)
        assert len(result) > 0

    def test_true_feature_outranks_distractors(self, rng):
        """Recovery is not enough; the true feature should rank near the top."""
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] + rng.normal(0, 0.02, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=50, include_originals=False).run(data, target)
        top_five = result.names[:5]
        assert any("a" in name and "b" in name and "*" in name for name in top_five)


# --------------------------------------------------------------------------- #
# Scaling: the headline efficiency property
# --------------------------------------------------------------------------- #


class TestScaling:
    def test_beam_width_bounds_retention(self, rng):
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        for width in (5, 10, 25):
            result = BeamSearch(max_depth=2, beam_width=width).run(data, target)
            for record in result.trace[1:]:
                assert record.n_kept <= width

    def test_wider_beam_proposes_more(self, rng):
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        narrow = BeamSearch(max_depth=2, beam_width=5).run(data, target)
        wide = BeamSearch(max_depth=2, beam_width=40).run(data, target)
        assert wide.n_proposed_total > narrow.n_proposed_total

    def test_cost_does_not_compound_with_depth(self, rng):
        """The central claim against exhaustive search.

        With a fixed beam, per-depth proposals stay bounded rather than growing
        multiplicatively as depth increases.
        """
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] / data["c"] + rng.normal(0, 0.1, len(data["a"]))
        result = BeamSearch(max_depth=4, beam_width=20).run(data, target)
        proposals = [record.n_proposed for record in result.trace[1:]]
        assert len(proposals) >= 2
        # Each depth's proposals are bounded by beam * (unary + beam*binary).
        ceiling = 20 * (5 + 20 * 4) * 4
        assert all(count <= ceiling for count in proposals)

    def test_max_features_caps_output(self, rng):
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=50, max_features=7).run(data, target)
        assert len(result) <= 7

    def test_candidate_ceiling_is_enforced(self, rng):
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=30, max_candidates_per_depth=50).run(data, target)
        for record in result.trace[1:]:
            assert record.n_proposed <= 50

    def test_max_features_defaults_to_beam_width(self, rng):
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=6).run(data, target)
        assert len(result) == 6

    def test_ceiling_truncates_the_weakest_columns(self, rng):
        """A binding ceiling at depth 1 must not cut the last columns of the table.

        Column order would propose only the pairs of x0 and x1 before reaching
        the ceiling; ranking the leaves by score reaches the planted pair.
        """
        n, p = 200, 12
        data = {f"x{i}": rng.uniform(1.0, 6.0, n) for i in range(p)}
        target = data["x10"] * data["x11"] + rng.normal(0, 0.05, n)
        search = BeamSearch(
            max_depth=1, beam_width=10, unary_ops=(), binary_ops=("mul",), max_candidates_per_depth=20
        )
        result = search.run(data, target)
        assert result.trace[1].n_proposed == 20
        assert "(x10 * x11)" in result.names

    def test_ceiling_that_could_bind_but_does_not_keeps_column_order(self, rng, ureg):
        """Units reject enough pairs that the ceiling, though reachable on a count, is not reached."""
        n = 200
        names = ["m1", "m2", "l1", "l2", "t1", "t2"]
        data = {name: rng.uniform(1.0, 5.0, n) for name in names}
        units = {name: 1.0 * {"m": ureg.kilogram, "l": ureg.meter, "t": ureg.second}[name[0]] for name in names}
        target = data["m1"] - data["m2"] + data["l2"] * data["t1"] + rng.normal(0, 0.05, n)
        # One depth: a ceiling at the bound would truncate the next one.
        search_kwargs = {"max_depth": 1, "beam_width": 15}
        bound = 6 * 5 + 3 * 6 * 5

        unbounded = BeamSearch(**search_kwargs).run(data, target, units=units)
        assert unbounded.trace[1].n_proposed < bound
        at_bound = BeamSearch(max_candidates_per_depth=bound, **search_kwargs).run(data, target, units=units)
        assert at_bound.names == unbounded.names
        assert at_bound.scores.tolist() == unbounded.scores.tolist()
        assert at_bound.trace[1].n_rejected_units == unbounded.trace[1].n_rejected_units

    def test_depth_zero_returns_originals_only(self, rng):
        data = _positive_data(rng)
        target = data["a"] + rng.normal(0, 0.1, len(data["a"]))
        result = BeamSearch(max_depth=0).run(data, target)
        assert set(result.names) <= set(data)


# --------------------------------------------------------------------------- #
# Redundancy control
# --------------------------------------------------------------------------- #


class TestRedundancy:
    def test_duplicate_column_not_selected_twice(self, rng):
        n = 300
        base = rng.uniform(1, 5, n)
        data = {"a": base, "a_copy": base * 2.0 + 1.0, "b": rng.uniform(1, 5, n)}
        target = base * 3.0 + rng.normal(0, 0.05, n)
        result = BeamSearch(max_depth=1, beam_width=20).run(data, target)
        assert not ("a" in result.names and "a_copy" in result.names)

    def test_copy_of_small_scale_column_not_selected_twice(self, rng):
        """Redundancy must not depend on the units a column is expressed in."""
        n = 300
        base = rng.uniform(1, 5, n) * 1e-14
        data = {"a": base, "a_copy": base.copy(), "b": rng.uniform(1, 5, n)}
        target = base * 3e14 + rng.normal(0, 0.05, n)
        result = BeamSearch(max_depth=0).run(data, target)
        assert not ("a" in result.names and "a_copy" in result.names)

    @pytest.mark.parametrize("seed", range(8))
    def test_threshold_one_drops_sign_flipped_copies(self, seed):
        """At threshold 1, (a - b) and (b - a) correlate at exactly 1.

        The computed correlation lands either side of 1 depending on rounding,
        so without a tolerance some seeds keep both.
        """
        rng = np.random.default_rng(seed)
        data = _positive_data(rng, n=300)
        target = data["a"] - data["b"] + rng.normal(0, 0.05, 300)
        search = BeamSearch(
            max_depth=1, beam_width=30, unary_ops=(), binary_ops=("sub",), redundancy_threshold=1.0
        )
        result = search.run(data, target)
        assert not ("(a - b)" in result.names and "(b - a)" in result.names)

    def test_threshold_controls_strictness(self, rng):
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        strict = BeamSearch(max_depth=2, beam_width=40, redundancy_threshold=0.5).run(data, target)
        loose = BeamSearch(max_depth=2, beam_width=40, redundancy_threshold=0.999).run(data, target)
        assert len(strict) <= len(loose)

    def test_redundant_drops_are_counted(self, rng):
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=40, redundancy_threshold=0.7).run(data, target)
        assert sum(record.n_rejected_redundant for record in result.trace) > 0

    def test_selected_features_are_mutually_uncorrelated(self, rng):
        """The property redundancy pruning exists to guarantee."""
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        search = BeamSearch(max_depth=2, beam_width=30, redundancy_threshold=0.9)
        result = search.run(data, target)

        evaluator = Evaluator(data)
        _, matrix = evaluator.evaluate_many(result.nodes)
        if matrix.shape[1] > 1:
            correlations = np.abs(np.corrcoef(matrix, rowvar=False))
            np.fill_diagonal(correlations, 0.0)
            assert np.nanmax(correlations) < 0.95


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


class TestDeterminism:
    def test_repeated_runs_identical(self, rng):
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        first = BeamSearch(max_depth=2, beam_width=20).run(data, target)
        second = BeamSearch(max_depth=2, beam_width=20).run(data, target)
        assert first.names == second.names
        np.testing.assert_allclose(first.scores, second.scores)

    def test_column_order_does_not_change_results(self, rng):
        """Commutative canonicalisation should make input order irrelevant.

        Restricted to commutative operators: ``a - c`` and ``c - a`` are
        genuinely different features, so a search including ``sub`` or ``div``
        is expected to return different (not merely reordered) results when the
        inputs are reordered.
        """
        data = _positive_data(rng, k=3)
        target = data["a"] * data["b"] + rng.normal(0, 0.05, len(data["a"]))
        reordered = {key: data[key] for key in reversed(list(data))}
        search_kwargs = {"max_depth": 1, "beam_width": 25, "binary_ops": ("mul", "add")}
        first = BeamSearch(**search_kwargs).run(data, target)
        second = BeamSearch(**search_kwargs).run(reordered, target)
        assert set(first.names) == set(second.names)

    def test_noncommutative_operators_are_order_sensitive(self, rng):
        """The converse: subtraction must distinguish operand order."""
        data = _positive_data(rng, k=3)
        target = data["a"] - data["b"] + rng.normal(0, 0.05, len(data["a"]))
        result = BeamSearch(max_depth=1, beam_width=30, unary_ops=(), binary_ops=("sub",)).run(data, target)
        assert _found(result, "(a - b)")
        assert not _found(result, "(b - a)")

    def test_warm_evaluator_does_not_change_results(self, rng):
        data = _positive_data(rng)
        target = data["a"] * data["b"] + rng.normal(0, 0.05, len(data["a"]))
        cold = BeamSearch(max_depth=2, beam_width=20).run(data, target)

        warm = Evaluator(data)
        warm.evaluate(transform("log", leaf("a")))
        warm.evaluate(combine("mul", leaf("a"), leaf("c")))
        reused = BeamSearch(max_depth=2, beam_width=20).run(data, target, evaluator=warm)
        assert cold.names == reused.names


# --------------------------------------------------------------------------- #
# Chunking and workers: neither may change the result
# --------------------------------------------------------------------------- #


def _signature(result):
    """Everything a run returns that chunking or workers could perturb, exactly."""
    return (
        result.names,
        result.scores.tolist(),
        [
            (t.n_proposed, t.n_evaluated, t.n_rejected_units, t.n_rejected_numeric, t.n_rejected_redundant,
             t.n_kept, t.best_score)
            for t in result.trace
        ],
        [(record.node.name, record.reason, record.detail) for record in result.evaluation_log],
        result.n_rejected_redundant_final,
    )


def _awkward_data(rng, n=150):
    """Inputs that trip every evaluation filter somewhere in the search.

    Signs put log and sqrt out of domain, zeros put division out of domain,
    ``c * e`` is constant to rounding, and ``d`` overflows when squared.
    """
    data = {name: rng.uniform(0.5, 4.0, n) for name in "abcdefg"}
    data["a"] = rng.normal(0.0, 1.0, n)
    data["b"][::7] = 0.0
    data["c"] = 1.0 / data["e"]
    data["d"] = rng.uniform(1e160, 1e170, n)
    target = data["a"] * data["f"] + np.log(data["g"]) + rng.normal(0, 0.05, n)
    return data, target


_SCORERS = [
    pytest.param("correlation", id="correlation"),
    pytest.param("spearman", id="spearman"),
    pytest.param(lambda: MutualInformationScorer(random_state=3), id="mutual_information"),
    pytest.param(lambda: GradientBoostingScorer(random_state=3, n_folds=2, max_iter=3), id="gradient_boosting"),
]


def _run(scorer, data, target, problem_type="regression", **kwargs):
    scorer = scorer() if callable(scorer) else scorer
    small = isinstance(scorer, GradientBoostingScorer)
    search = BeamSearch(
        scorer=scorer,
        problem_type=problem_type,
        max_depth=1 if small else 2,
        beam_width=6 if small else 12,
        unary_ops=("log", "square") if small else beamfeat.search.DEFAULT_UNARY,
        binary_ops=("mul",) if small else beamfeat.search.DEFAULT_BINARY,
        **kwargs,
    )
    return search.run(data, target)


class TestChunkingAndWorkers:
    @pytest.mark.parametrize("scorer", _SCORERS)
    def test_chunk_size_does_not_change_the_result(self, rng, scorer):
        data, target = _awkward_data(rng)
        reference = _signature(_run(scorer, data, target, chunk_size=10**9))
        for chunk_size in (1, 7):
            assert _signature(_run(scorer, data, target, chunk_size=chunk_size)) == reference

    @pytest.mark.parametrize("scorer", _SCORERS)
    def test_n_jobs_does_not_change_the_result(self, rng, scorer, monkeypatch):
        """Including the scorers that draw random numbers: their seeds do not depend on workers."""
        monkeypatch.setattr(beamfeat.search, "MIN_PARALLEL_CANDIDATES", 0)
        data, target = _awkward_data(rng)
        reference = _signature(_run(scorer, data, target, n_jobs=1, chunk_size=9))
        for n_jobs in (2, -1):
            assert _signature(_run(scorer, data, target, n_jobs=n_jobs, chunk_size=9)) == reference

    def test_n_jobs_does_not_change_a_classification_result(self, rng, monkeypatch):
        monkeypatch.setattr(beamfeat.search, "MIN_PARALLEL_CANDIDATES", 0)
        data, target = _awkward_data(rng)
        labels = (target > np.median(target)).astype(int)
        reference = _signature(_run("correlation", data, labels, "classification", chunk_size=5))
        assert _signature(_run("correlation", data, labels, "classification", chunk_size=5, n_jobs=2)) == reference

    def test_wide_round_runs_on_workers(self, rng):
        """Enough candidates to pass MIN_PARALLEL_CANDIDATES with the default chunking."""
        n, p = 120, 40
        data = {f"x{i}": rng.uniform(1.0, 6.0, n) for i in range(p)}
        target = data["x3"] * data["x7"] + rng.normal(0, 0.05, n)
        serial = BeamSearch(max_depth=1, beam_width=20, chunk_size=500).run(data, target)
        assert serial.trace[1].n_proposed >= beamfeat.search.MIN_PARALLEL_CANDIDATES
        parallel = BeamSearch(max_depth=1, beam_width=20, chunk_size=500, n_jobs=2).run(data, target)
        assert _signature(parallel) == _signature(serial)

    def test_scorer_overriding_score_batch_is_scored_whole(self, rng):
        """A subclass that overrides score_batch is never bypassed by chunked scoring."""

        class Shifted(CorrelationScorer):
            calls: list[int] = []

            def score_batch(self, candidates, target, incumbent=None):
                self.calls.append(np.asarray(candidates).shape[1])
                return super().score_batch(candidates, target, incumbent) + 1.0

        scorer = Shifted()
        assert not scorer.scores_by_column()
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] + rng.normal(0, 0.05, len(data["a"]))
        BeamSearch(scorer=scorer, max_depth=1, beam_width=10, chunk_size=3).run(data, target)
        # Depth 0, depth 1 absolute and marginal, and the final ranking: each one call.
        assert len(scorer.calls) == 4
        assert max(scorer.calls) > 3

    def test_custom_scorer_without_prepare_is_scored_whole(self, rng):
        class Variance(Scorer):
            def score_batch(self, candidates, target, incumbent=None):
                return np.var(np.asarray(candidates), axis=0)

        assert not Variance().scores_by_column()
        data = _positive_data(rng, k=3)
        result = BeamSearch(scorer=Variance(), max_depth=1, beam_width=5, chunk_size=2).run(
            data, data["a"] * data["b"]
        )
        assert len(result) > 0


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


class TestUnits:
    def test_dimensional_mismatches_are_rejected(self, rng, ureg):
        n = 300
        data = {"mass": rng.uniform(1, 5, n), "length": rng.uniform(1, 5, n)}
        units = {"mass": 1.0 * ureg.kilogram, "length": 1.0 * ureg.meter}
        target = data["mass"] * data["length"] + rng.normal(0, 0.05, n)

        result = BeamSearch(max_depth=1, beam_width=30).run(data, target, units=units)
        assert sum(record.n_rejected_units for record in result.trace) > 0
        assert not _found(result, "mass + length")
        assert not _found(result, "mass - length")

    def test_valid_combinations_still_found(self, rng, ureg):
        n = 300
        data = {"mass": rng.uniform(1, 5, n), "length": rng.uniform(1, 5, n)}
        units = {"mass": 1.0 * ureg.kilogram, "length": 1.0 * ureg.meter}
        target = data["mass"] * data["length"] + rng.normal(0, 0.05, n)
        result = BeamSearch(max_depth=1, beam_width=30).run(data, target, units=units)
        assert _found(result, "mass", "*", "length")

    def test_units_shrink_the_search(self, rng, ureg):
        """Dimensional constraints should reduce work, not just filter output."""
        n = 300
        data = {"mass": rng.uniform(1, 5, n), "length": rng.uniform(1, 5, n), "time": rng.uniform(1, 5, n)}
        units = {"mass": 1.0 * ureg.kilogram, "length": 1.0 * ureg.meter, "time": 1.0 * ureg.second}
        target = data["mass"] * data["length"] / data["time"] + rng.normal(0, 0.05, n)

        unconstrained = BeamSearch(max_depth=2, beam_width=25).run(data, target)
        constrained = BeamSearch(max_depth=2, beam_width=25).run(data, target, units=units)
        assert constrained.n_proposed_total < unconstrained.n_proposed_total


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


class TestReporting:
    def test_trace_covers_every_depth(self, rng):
        data = _positive_data(rng)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=20).run(data, target)
        assert [record.depth for record in result.trace] == [0, 1, 2]

    def test_scores_are_sorted_descending(self, rng):
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=25).run(data, target)
        assert np.all(np.diff(result.scores) <= 1e-12)

    def test_scores_align_with_nodes(self, rng):
        data = _positive_data(rng)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=20).run(data, target)
        assert len(result.nodes) == len(result.scores) == len(result)

    def test_evaluation_log_is_exposed(self, rng):
        data = _positive_data(rng)
        data["signed"] = rng.normal(0, 2, len(data["a"]))  # forces log/sqrt rejections
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        result = BeamSearch(max_depth=1, beam_width=20).run(data, target)
        assert result.evaluation_log is not None
        assert len(result.evaluation_log) > 0

    def test_totals_match_the_trace(self, rng):
        data = _positive_data(rng, k=4)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        result = BeamSearch(max_depth=2, beam_width=20).run(data, target)
        assert result.n_proposed_total == sum(record.n_proposed for record in result.trace[1:])
        assert result.n_evaluated_total == sum(record.n_evaluated for record in result.trace[1:])

    def test_final_redundant_drops_are_counted(self, rng):
        n = 300
        base = rng.uniform(1, 5, n)
        data = {"a": base, "a_copy": base * 2.0 + 1.0, "b": rng.uniform(1, 5, n)}
        target = base * 3.0 + rng.normal(0, 0.05, n)
        result = BeamSearch(max_depth=0).run(data, target)
        assert result.n_rejected_redundant_final == 1
        assert len(result) == 2

    def test_progress_reports_depth_reached(self, rng, capsys):
        data = {"a": rng.uniform(1, 5, 200)}
        target = data["a"] + rng.normal(0, 0.1, 200)
        BeamSearch(max_depth=3, unary_ops=(), binary_ops=("mul",), verbose=1).run(data, target)
        assert "search: depth 0 of 3," in capsys.readouterr().out

    def test_summary_renders(self, rng):
        data = _positive_data(rng)
        target = data["a"] * data["b"] + rng.normal(0, 0.1, len(data["a"]))
        summary = BeamSearch(max_depth=1, beam_width=10).run(data, target).summary()
        assert "beam search" in summary
        assert "depth 0" in summary

    def test_result_stored_on_instance(self, rng):
        data = _positive_data(rng)
        target = data["a"] + rng.normal(0, 0.1, len(data["a"]))
        search = BeamSearch(max_depth=1, beam_width=10)
        assert search.result_ is None
        result = search.run(data, target)
        assert search.result_ is result


# --------------------------------------------------------------------------- #
# Validation and edge cases
# --------------------------------------------------------------------------- #


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_depth": -1}, "max_depth"),
            ({"beam_width": 0}, "beam_width"),
            ({"redundancy_threshold": 0.0}, "redundancy_threshold"),
            ({"redundancy_threshold": 1.5}, "redundancy_threshold"),
            ({"max_features": 0}, "max_features"),
            ({"max_candidates_per_depth": 0}, "max_candidates_per_depth"),
            ({"n_jobs": 0}, "n_jobs"),
            ({"n_jobs": 1.5}, "n_jobs"),
            ({"n_jobs": True}, "n_jobs"),
            ({"chunk_size": 0}, "chunk_size"),
        ],
    )
    def test_bad_parameters_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            BeamSearch(**kwargs)

    def test_unknown_operator_rejected(self):
        with pytest.raises(ValueError, match="unknown operator"):
            BeamSearch(unary_ops=("teleport",))

    def test_arity_mismatch_rejected(self):
        with pytest.raises(ValueError, match="not unary"):
            BeamSearch(unary_ops=("mul",))
        with pytest.raises(ValueError, match="not binary"):
            BeamSearch(binary_ops=("log",))

    def test_target_length_mismatch_rejected(self, rng):
        data = _positive_data(rng, n=100)
        with pytest.raises(ValueError, match="rows"):
            BeamSearch().run(data, rng.normal(size=50))

    def test_array_input_accepted(self, rng):
        matrix = rng.uniform(1, 5, (300, 3))
        target = matrix[:, 0] * matrix[:, 1] + rng.normal(0, 0.05, 300)
        result = BeamSearch(max_depth=1, beam_width=20).run(matrix, target, columns=["p", "q", "r"])
        assert _found(result, "p", "*", "q")

    def test_all_constant_input_rejected(self):
        data = {"a": np.ones(100), "b": np.ones(100)}
        with pytest.raises(ValueError, match="no input column survived"):
            BeamSearch().run(data, np.arange(100.0))

    def test_single_column_input(self, rng):
        data = {"a": rng.uniform(1, 5, 200)}
        target = np.log(data["a"]) + rng.normal(0, 0.05, 200)
        result = BeamSearch(max_depth=1, beam_width=10).run(data, target)
        assert len(result) > 0

    def test_classification_runs(self, rng):
        data = _positive_data(rng, n=300)
        labels = (data["a"] * data["b"] > np.median(data["a"] * data["b"])).astype(int)
        result = BeamSearch(problem_type="classification", max_depth=1, beam_width=20).run(data, labels)
        assert len(result) > 0

    def test_include_originals_toggle(self, rng):
        data = _positive_data(rng)
        target = data["a"] * data["b"] + rng.normal(0, 0.05, len(data["a"]))
        without = BeamSearch(max_depth=1, beam_width=20, include_originals=False).run(data, target)
        assert not (set(without.names) & set(data))

    def test_custom_scorer_instance_accepted(self, rng):
        data = _positive_data(rng)
        target = data["a"] * data["b"] + rng.normal(0, 0.05, len(data["a"]))
        result = BeamSearch(scorer=CorrelationScorer(), max_depth=1, beam_width=15).run(data, target)
        assert len(result) > 0

    def test_restricted_operator_set_respected(self, rng):
        data = _positive_data(rng)
        target = data["a"] * data["b"] + rng.normal(0, 0.05, len(data["a"]))
        result = BeamSearch(max_depth=1, beam_width=20, unary_ops=(), binary_ops=("mul",)).run(data, target)
        assert not any("log" in name or "sqrt" in name for name in result.names)
