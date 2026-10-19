"""Tests for the pluggable scoring strategies."""

from __future__ import annotations

import numpy as np
import pytest

from beamfeat.scoring import (
    CorrelationScorer,
    GradientBoostingScorer,
    MutualInformationScorer,
    Scorer,
    SpearmanScorer,
    _average_ranks,
    _standardise,
    column_stats,
    is_constant,
    make_scorer,
)

ALL_SCORERS = ["correlation", "spearman", "mutual_information", "gradient_boosting"]


@pytest.fixture
def rng():
    return np.random.default_rng(20260723)


def _fast(name: str, problem_type: str = "regression") -> Scorer:
    """Construct a scorer with cheap settings so tests stay quick."""
    if name == "gradient_boosting":
        return GradientBoostingScorer(
            problem_type=problem_type, n_folds=2, max_iter=15, max_depth=2, subsample_size=300
        )
    return make_scorer(name, problem_type=problem_type)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


class TestFactory:
    @pytest.mark.parametrize("name", ALL_SCORERS)
    def test_resolves_canonical_names(self, name):
        assert make_scorer(name).name == name

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("corr", "correlation"),
            ("pearson", "correlation"),
            ("rank", "spearman"),
            ("mi", "mutual_information"),
            ("mutual_info", "mutual_information"),
            ("gb", "gradient_boosting"),
            ("boosting", "gradient_boosting"),
        ],
    )
    def test_resolves_aliases(self, alias, expected):
        assert make_scorer(alias).name == expected

    def test_case_insensitive(self):
        assert make_scorer("Correlation").name == "correlation"
        assert make_scorer("MI").name == "mutual_information"

    def test_instance_passes_through(self):
        scorer = CorrelationScorer()
        assert make_scorer(scorer) is scorer

    def test_instance_with_other_problem_type_warns(self):
        """An instance keeps its own problem type; asking for another one
        must not pass unnoticed."""
        scorer = CorrelationScorer()
        with pytest.warns(UserWarning, match="problem_type='regression'"):
            assert make_scorer(scorer, problem_type="classification") is scorer

    def test_instance_with_matching_problem_type_is_silent(self):
        import warnings

        scorer = CorrelationScorer(problem_type="classification")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert make_scorer(scorer, problem_type="classification") is scorer

    def test_unknown_name_lists_options(self):
        with pytest.raises(ValueError, match="unknown scorer"):
            make_scorer("telepathy")

    def test_wrong_type_rejected(self):
        with pytest.raises(TypeError):
            make_scorer(42)

    def test_bad_problem_type_rejected(self):
        with pytest.raises(ValueError, match="problem_type"):
            make_scorer("correlation", problem_type="clustering")

    def test_kwargs_forwarded(self):
        scorer = make_scorer("mi", n_neighbors=7)
        assert scorer.n_neighbors == 7


class TestConstructorValidation:
    def test_mi_rejects_bad_neighbors(self):
        with pytest.raises(ValueError, match="n_neighbors"):
            MutualInformationScorer(n_neighbors=0)

    def test_gb_rejects_bad_folds(self):
        with pytest.raises(ValueError, match="n_folds"):
            GradientBoostingScorer(n_folds=1)

    def test_gb_rejects_bad_max_iter(self):
        with pytest.raises(ValueError, match="max_iter"):
            GradientBoostingScorer(max_iter=0)


# --------------------------------------------------------------------------- #
# Contract obeyed by every scorer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ALL_SCORERS)
class TestScorerContract:
    def test_returns_one_score_per_candidate(self, name, rng):
        scorer = _fast(name)
        candidates = rng.normal(size=(200, 4))
        target = candidates[:, 0] * 2.0 + rng.normal(0, 0.1, 200)
        scores = scorer.score_batch(candidates, target)
        assert scores.shape == (4,)

    def test_scores_are_finite_and_non_negative(self, name, rng):
        scorer = _fast(name)
        candidates = rng.normal(size=(200, 5))
        target = candidates[:, 1] + rng.normal(0, 0.5, 200)
        scores = scorer.score_batch(candidates, target)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)

    def test_ranks_signal_above_noise(self, name, rng):
        """The core requirement: a real predictor must outrank pure noise."""
        n = 400
        signal = rng.normal(size=n)
        target = 3.0 * signal + rng.normal(0, 0.2, n)
        candidates = np.column_stack([rng.normal(size=n), signal, rng.normal(size=n)])
        scores = _fast(name).score_batch(candidates, target)
        assert int(np.argmax(scores)) == 1, f"{name} failed to rank the true signal first"

    def test_single_candidate_convenience(self, name, rng):
        n = 200
        signal = rng.normal(size=n)
        target = 2.0 * signal + rng.normal(0, 0.2, n)
        assert _fast(name).score(signal, target) > 0

    def test_deterministic_across_calls(self, name, rng):
        scorer = _fast(name)
        candidates = rng.normal(size=(200, 3))
        target = candidates[:, 0] + rng.normal(0, 0.3, 200)
        np.testing.assert_allclose(
            scorer.score_batch(candidates, target), scorer.score_batch(candidates, target)
        )

    def test_deterministic_across_instances(self, name, rng):
        candidates = rng.normal(size=(200, 3))
        target = candidates[:, 0] + rng.normal(0, 0.3, 200)
        np.testing.assert_allclose(
            _fast(name).score_batch(candidates, target),
            _fast(name).score_batch(candidates, target),
        )

    def test_1d_candidate_accepted(self, name, rng):
        n = 200
        signal = rng.normal(size=n)
        scores = _fast(name).score_batch(signal, 2.0 * signal + rng.normal(0, 0.2, n))
        assert scores.shape == (1,)

    def test_row_mismatch_rejected(self, name, rng):
        with pytest.raises(ValueError, match="rows"):
            _fast(name).score_batch(rng.normal(size=(100, 2)), rng.normal(size=50))

    def test_1d_incumbent_accepted(self, name, rng):
        n = 200
        first, second = rng.normal(size=n), rng.normal(size=n)
        target = first + second + rng.normal(0, 0.2, n)
        scorer = _fast(name)
        candidates = np.column_stack([second, rng.normal(size=n)])
        np.testing.assert_allclose(
            scorer.score_batch(candidates, target, first),
            scorer.score_batch(candidates, target, first.reshape(-1, 1)),
        )

    def test_empty_incumbent_treated_as_none(self, name, rng):
        n = 200
        signal = rng.normal(size=n)
        target = 2.0 * signal + rng.normal(0, 0.2, n)
        scorer = _fast(name)
        candidates = signal.reshape(-1, 1)
        with_empty = scorer.score_batch(candidates, target, np.empty((n, 0)))
        without = scorer.score_batch(candidates, target, None)
        np.testing.assert_allclose(with_empty, without)

    def test_classification_supported(self, name, rng):
        n = 300
        signal = rng.normal(size=n)
        labels = (signal > 0).astype(int)
        candidates = np.column_stack([rng.normal(size=n), signal])
        scores = _fast(name, problem_type="classification").score_batch(candidates, labels)
        assert scores.shape == (2,)
        assert scores[1] > scores[0]

    @pytest.mark.parametrize("n_classes", [2, 3])
    def test_string_class_labels_score_like_their_codes(self, name, rng, n_classes):
        n = 300
        signal = rng.normal(size=n)
        codes = np.digitize(signal, np.quantile(signal, np.linspace(0, 1, n_classes + 1)[1:-1]))
        names = np.array(["alpha", "beta", "gamma"])[:n_classes]
        candidates = np.column_stack([rng.normal(size=n), signal])
        incumbent = rng.normal(size=n)
        scorer = _fast(name, problem_type="classification")
        np.testing.assert_allclose(
            scorer.score_batch(candidates, names[codes], incumbent),
            scorer.score_batch(candidates, codes, incumbent),
        )


# --------------------------------------------------------------------------- #
# Marginal scoring: the anti-duplication property
# --------------------------------------------------------------------------- #


class TestMarginalScoring:
    def test_correlation_discounts_incumbent_duplicate(self, rng):
        """A candidate identical to an incumbent must lose its value."""
        n = 400
        signal = rng.normal(size=n)
        target = 3.0 * signal + rng.normal(0, 0.2, n)
        scorer = CorrelationScorer()
        alone = scorer.score(signal, target)
        given = scorer.score(signal, target, incumbent=signal.reshape(-1, 1))
        assert alone > 0.8
        assert given < 0.1

    def test_correlation_keeps_orthogonal_candidate(self, rng):
        """A genuinely new signal must survive residualisation."""
        n = 400
        first = rng.normal(size=n)
        second = rng.normal(size=n)
        target = 2.0 * first + 2.0 * second + rng.normal(0, 0.2, n)
        scorer = CorrelationScorer()
        given = scorer.score(second, target, incumbent=first.reshape(-1, 1))
        assert given > 0.5

    def test_gb_discounts_incumbent_duplicate(self, rng):
        n = 400
        signal = rng.normal(size=n)
        target = 3.0 * signal + rng.normal(0, 0.2, n)
        scorer = GradientBoostingScorer(n_folds=2, max_iter=20, max_depth=2, subsample_size=400)
        alone = scorer.score(signal, target)
        given = scorer.score(signal, target, incumbent=signal.reshape(-1, 1))
        assert alone > given

    @pytest.mark.parametrize("scorer_class", [CorrelationScorer, SpearmanScorer])
    def test_saturated_incumbent_scores_zero(self, scorer_class, rng):
        """When the incumbent explains the target exactly, nothing adds value."""
        n = 200
        signal = rng.normal(size=n)
        target = signal.copy()
        scores = scorer_class().score_batch(rng.normal(size=(n, 3)), target, signal.reshape(-1, 1))
        assert np.all(scores < 1e-6)

    @pytest.mark.parametrize("scorer_class", [CorrelationScorer, SpearmanScorer])
    def test_classification_discounts_incumbent_duplicate(self, scorer_class, rng):
        n = 400
        signal = rng.normal(size=n)
        labels = np.where(signal + rng.normal(0, 0.5, n) > 0, "yes", "no")
        scorer = scorer_class(problem_type="classification")
        alone = scorer.score(signal, labels)
        given = scorer.score(signal, labels, incumbent=signal)
        assert alone > 0.6
        assert given < 0.1


# --------------------------------------------------------------------------- #
# Where the scorers genuinely differ
# --------------------------------------------------------------------------- #


class TestScorerDifferentiation:
    def test_mi_detects_nonmonotone_dependence(self, rng):
        """The case that justifies MI's cost: correlation is blind here."""
        n = 600
        feature = rng.uniform(-3, 3, n)
        target = feature**2 + rng.normal(0, 0.3, n)  # symmetric, so corr ~ 0
        noise = rng.normal(size=n)
        candidates = np.column_stack([feature, noise])

        corr_scores = CorrelationScorer().score_batch(candidates, target)
        mi_scores = MutualInformationScorer().score_batch(candidates, target)

        assert corr_scores[0] < 0.2, "correlation should be near-blind to a symmetric relationship"
        assert mi_scores[0] > mi_scores[1], "MI should detect the quadratic dependence"

    def test_correlation_suffices_when_transform_is_explicit(self, rng):
        """Why correlation remains a sensible default in this library.

        A nonlinearity already encoded in the expression is visible to a linear
        scorer applied to the transformed column.
        """
        n = 400
        raw = rng.uniform(-3, 3, n)
        target = raw**2 + rng.normal(0, 0.3, n)
        transformed = raw**2  # what the DAG would hand the scorer
        assert CorrelationScorer().score(transformed, target) > 0.8

    def test_correlation_is_faster_than_alternatives(self, rng):
        """Documents the cost ordering the docstrings claim.

        Each scorer is timed as the best of several runs, since a single
        wall-clock reading on a shared runner can be inflated by anything
        else the machine is doing; the minimum is the reading least affected.
        The gap being checked is one to two orders of magnitude."""
        import time

        n, k = 500, 30
        candidates = rng.normal(size=(n, k))
        target = candidates[:, 0] + rng.normal(0, 0.3, n)

        def best_of(scorer, repeats):
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                scorer.score_batch(candidates, target)
                times.append(time.perf_counter() - start)
            return min(times)

        corr_time = best_of(CorrelationScorer(), 7)
        mi_time = best_of(MutualInformationScorer(), 3)
        assert corr_time < mi_time


# --------------------------------------------------------------------------- #
# Degenerate inputs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ALL_SCORERS)
class TestDegenerateInputs:
    def test_constant_candidate_scores_zero(self, name, rng):
        n = 200
        target = rng.normal(size=n)
        scores = _fast(name).score_batch(np.ones((n, 1)), target)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_constant_target_handled(self, name, rng):
        n = 200
        scores = _fast(name).score_batch(rng.normal(size=(n, 3)), np.ones(n))
        assert np.all(np.isfinite(scores))

    def test_tiny_sample_handled(self, name, rng):
        scores = _fast(name).score_batch(rng.normal(size=(4, 2)), rng.normal(size=4))
        assert scores.shape == (2,)
        assert np.all(np.isfinite(scores))

    def test_single_candidate_column(self, name, rng):
        n = 150
        signal = rng.normal(size=n)
        scores = _fast(name).score_batch(signal.reshape(-1, 1), signal + rng.normal(0, 0.2, n))
        assert scores.shape == (1,)


class TestGradientBoostingSpecifics:
    def test_subsampling_is_deterministic(self, rng):
        n = 800
        signal = rng.normal(size=n)
        target = 2.0 * signal + rng.normal(0, 0.3, n)
        candidates = np.column_stack([signal, rng.normal(size=n)])
        scorer = GradientBoostingScorer(n_folds=2, max_iter=15, max_depth=2, subsample_size=200)
        np.testing.assert_allclose(
            scorer.score_batch(candidates, target), scorer.score_batch(candidates, target)
        )

    def test_subsampling_bounds_rows_seen(self, rng, monkeypatch):
        """Subsampling must bound the rows the model is fitted on.

        Asserted by observing fit sizes directly rather than wall-clock time:
        at these input sizes runtime is overhead-dominated and shared CI
        runners make timing comparisons non-deterministic."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        n = 4000
        signal = rng.normal(size=n)
        target = 2.0 * signal + rng.normal(0, 0.3, n)
        candidates = np.column_stack([signal, rng.normal(size=n)])

        seen: list[int] = []
        original_fit = HistGradientBoostingRegressor.fit

        def recording_fit(self, X, y, **kwargs):
            seen.append(len(X))
            return original_fit(self, X, y, **kwargs)

        monkeypatch.setattr(HistGradientBoostingRegressor, "fit", recording_fit)

        scorer = GradientBoostingScorer(n_folds=2, max_iter=20, max_depth=2, subsample_size=300)
        scorer.score_batch(candidates, target)
        assert seen and max(seen) <= 300

        seen.clear()
        full = GradientBoostingScorer(n_folds=2, max_iter=20, max_depth=2, subsample_size=None)
        full.score_batch(candidates, target)
        assert max(seen) > 300

    def test_single_class_fold_handled(self, rng):
        """A rare class must not crash the stratified path."""
        n = 100
        labels = np.zeros(n, dtype=int)
        labels[:3] = 1
        scorer = GradientBoostingScorer(problem_type="classification", n_folds=2, max_iter=10, max_depth=2)
        scores = scorer.score_batch(rng.normal(size=(n, 2)), labels)
        assert np.all(np.isfinite(scores))


class TestSpearmanScorer:
    def test_monotone_nonlinear_beats_pearson(self):
        """exp() of a uniform is monotone but convex: rank correlation is
        exact where product-moment correlation is diluted by the curvature."""
        from beamfeat.scoring import CorrelationScorer, SpearmanScorer

        rng = np.random.default_rng(0)
        x = rng.uniform(0, 3, (400, 1))
        y = np.exp(2 * x[:, 0])
        pearson = CorrelationScorer().score_batch(x, y)[0]
        spearman = SpearmanScorer().score_batch(x, y)[0]
        assert spearman > 0.999
        assert spearman > pearson

    def test_outlier_insensitive(self):
        from beamfeat.scoring import CorrelationScorer, SpearmanScorer

        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, (300, 1))
        y = x[:, 0] + 0.01 * rng.standard_normal(300)
        y[:5] += 500.0  # five wild rows
        assert SpearmanScorer().score_batch(x, y)[0] > 0.9
        assert CorrelationScorer().score_batch(x, y)[0] < 0.5

    def test_factory_resolves(self):
        from beamfeat.scoring import SpearmanScorer, make_scorer

        assert isinstance(make_scorer("spearman"), SpearmanScorer)
        assert isinstance(make_scorer("rank"), SpearmanScorer)

    def test_ties_do_not_depend_on_row_order(self, rng):
        """Ordinal ranks broke ties by row position, so reordering the rows
        changed the score of a column with tied values."""
        n = 300
        candidate = rng.integers(0, 3, n).astype(np.float64)
        target = candidate + rng.normal(0, 1.0, n)
        scorer = SpearmanScorer()
        expected = scorer.score(candidate, target)
        for _ in range(3):
            order = rng.permutation(n)
            assert scorer.score(candidate[order], target[order]) == pytest.approx(expected, rel=1e-12)

    def test_matches_spearman_rho_with_ties(self, rng):
        stats = pytest.importorskip("scipy.stats")
        n = 200
        candidate = rng.integers(0, 4, n).astype(np.float64)
        target = np.round(candidate + rng.normal(0, 1.5, n))
        expected = abs(stats.spearmanr(candidate, target)[0])
        assert SpearmanScorer().score(candidate, target) == pytest.approx(expected, rel=1e-10)

    def test_average_ranks_match_rankdata(self, rng):
        stats = pytest.importorskip("scipy.stats")
        matrix = rng.integers(0, 5, size=(60, 6)).astype(np.float64)
        matrix[:, 0] = 2.0
        matrix[:, 1] = rng.normal(size=60)
        np.testing.assert_array_equal(_average_ranks(matrix), stats.rankdata(matrix, axis=0))

    def test_non_finite_candidate_scores_zero(self, rng):
        n = 100
        signal = rng.normal(size=n)
        candidates = np.column_stack([signal, signal, signal])
        candidates[0, 1] = np.inf
        candidates[0, 2] = np.nan
        scores = SpearmanScorer().score_batch(candidates, signal)
        assert scores[0] == pytest.approx(1.0)
        np.testing.assert_array_equal(scores[1:], [0.0, 0.0])
        with np.errstate(invalid="ignore"):
            pearson = CorrelationScorer().score_batch(candidates, signal)
        np.testing.assert_array_equal(pearson[1:], [0.0, 0.0])


class TestStringLabelsThroughTheTransformer:
    def test_classification_transformer_accepts_string_labels(self, rng):
        """The scorers cast the target to float, so a string-labelled
        classification target failed in the search."""
        from beamfeat import BeamFeatTransformer

        n = 300
        data = rng.normal(size=(n, 3))
        codes = (data[:, 0] * data[:, 1] + rng.normal(0, 0.2, n) > 0).astype(int)
        labels = np.array(["no", "yes"])[codes]

        def fit(target):
            return BeamFeatTransformer(
                problem_type="classification", parsimony=None, max_depth=1, beam_width=10, random_state=0
            ).fit(data, target)

        assert fit(labels).feature_formulas_ == fit(codes).feature_formulas_


class TestIsConstant:
    """Constancy is relative to magnitude, not an absolute floor on the
    spread. An absolute floor makes the verdict depend on the column's units,
    which is how a genuinely varying small-magnitude column gets discarded and
    a stuck sensor gets kept."""

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            (np.full(500, 9.81), True),
            (np.full(500, 9.81) + np.linspace(0, 1e-15, 500), True),
            (np.zeros(500), True),
            (np.linspace(0, 1e-9, 500), False),
            (300.0 + np.linspace(0, 1e-4, 500), False),
            (np.linspace(0, 1, 500), False),
        ],
    )
    def test_verdicts(self, values, expected):
        assert is_constant(values) is expected

    def test_an_absolute_floor_would_disagree(self):
        """Pins the reason for the relative test. This column varies over its
        whole range, but its variance sits below any floor low enough to be
        worth writing down, so an absolute test would discard it."""
        varying = np.linspace(0, 1e-9, 500)
        assert np.var(varying) < 1e-8
        assert is_constant(varying) is False

    def test_standardise_agrees_with_the_predicate(self):
        for values in (np.full(200, 4.0), np.linspace(0, 1e-9, 200), np.linspace(1, 9, 200)):
            standardised = _standardise(values)
            assert np.allclose(standardised, 0.0) is is_constant(values)


# --------------------------------------------------------------------------- #
# Scoring in pieces
# --------------------------------------------------------------------------- #


def _awkward_candidates(rng, n=240):
    """Columns at very different scales, with a constant, a near-constant and a NaN column."""
    base = rng.uniform(1.0, 3.0, (n, 9)) * np.array([1e-9, 1e-3, 1.0, 1.0, 1e3, 1e9, 1.0, 1.0, 1.0])
    base[:, 6] = 2.5
    base[:, 7] = 1.0 + 1e-14 * rng.normal(size=n)
    base[3, 8] = np.nan
    return base


SEPARABLE = ["correlation", "spearman", "gradient_boosting"]


class TestScoringInPieces:
    @pytest.mark.parametrize("name", SEPARABLE)
    @pytest.mark.parametrize("problem_type", ["regression", "classification"])
    @pytest.mark.parametrize("with_incumbent", [False, True])
    def test_prepared_scores_equal_score_batch(self, rng, name, problem_type, with_incumbent):
        candidates = _awkward_candidates(rng)
        signal = candidates[:, 2] * candidates[:, 3]
        target = signal + rng.normal(0, 0.1, len(signal))
        if problem_type == "classification":
            target = (target > np.median(target)).astype(int)
        incumbent = candidates[:, [3]] if with_incumbent else None
        scorer = _fast(name, problem_type)
        assert scorer.scores_by_column()

        whole = scorer.score_batch(candidates, target, incumbent)
        prepared = scorer.prepare(target, incumbent)
        pieces = np.concatenate(
            [scorer.score_prepared(candidates[:, start : start + 2], prepared) for start in range(0, 9, 2)]
        )
        np.testing.assert_array_equal(pieces, whole)

    @pytest.mark.parametrize("name", ["correlation", "gradient_boosting"])
    def test_supplied_statistics_give_the_same_scores(self, rng, name):
        candidates = _awkward_candidates(rng)
        target = candidates[:, 2] + rng.normal(0, 0.1, len(candidates))
        scorer = _fast(name)
        prepared = scorer.prepare(target)
        np.testing.assert_array_equal(
            scorer.score_prepared(candidates, prepared, column_stats(candidates)),
            scorer.score_prepared(candidates, prepared),
        )

    def test_correlation_equals_the_column_by_column_definition(self, rng):
        """Bit for bit: vectorising the standardisation must not move a score."""
        candidates = _awkward_candidates(rng)
        target = candidates[:, 2] * candidates[:, 4] + rng.normal(0, 0.1, len(candidates))
        residual = _standardise(target)
        expected = np.empty(candidates.shape[1])
        with np.errstate(all="ignore"):
            for index in range(candidates.shape[1]):
                column = _standardise(candidates[:, index])
                expected[index] = abs(float(column @ residual) / len(target))
        expected = np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)
        np.testing.assert_array_equal(CorrelationScorer().score_batch(candidates, target), expected)
        np.testing.assert_array_equal(
            CorrelationScorer().score_batch(np.ascontiguousarray(candidates), target),
            CorrelationScorer().score_batch(np.asfortranarray(candidates), target),
        )

    def test_column_stats_match_the_one_column_reductions(self, rng):
        candidates = _awkward_candidates(rng)[:, :8]
        stats = column_stats(candidates)
        for index in range(candidates.shape[1]):
            column = candidates[:, index]
            assert stats.mean[index] == column.mean()
            assert stats.variance[index] == np.var(column)
            assert stats.magnitude[index] == np.mean(np.abs(column))

    def test_mutual_information_is_scored_whole(self, rng):
        """Its noise is drawn for the whole batch, so a column's score depends on the batch."""
        scorer = MutualInformationScorer()
        assert not scorer.scores_by_column()
        with pytest.raises(NotImplementedError):
            scorer.prepare(rng.normal(size=50))

    def test_nothing_left_to_explain_scores_zero(self, rng):
        candidates = rng.uniform(1.0, 3.0, (100, 3))
        target = 2.0 * candidates[:, 0] + 1.0
        for scorer in (CorrelationScorer(), SpearmanScorer()):
            prepared = scorer.prepare(target, candidates[:, [0]])
            assert prepared is None
            np.testing.assert_array_equal(scorer.score_prepared(candidates, prepared), np.zeros(3))
