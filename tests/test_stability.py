"""Tests for the stability report."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from beamfeat import (
    BeamFeatClassifier,
    BeamFeatRegressor,
    BeamFeatTransformer,
    NoDiscoveriesWarning,
    StabilityReport,
    stability_report,
)
from beamfeat.expression import combine, leaf, transform
from beamfeat.stability import _feature_labels


@pytest.fixture
def planted():
    """y = x0 * x1, recovered at depth 1 by every split."""
    rng = np.random.default_rng(7)
    X = rng.uniform(1.0, 6.0, (200, 4))
    return X, X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 200)


def _fast(cls, **kwargs):
    """Estimator with cheap settings so the suite stays quick."""
    kwargs.setdefault("max_depth", 2)
    kwargs.setdefault("beam_width", 10)
    return cls(**kwargs)


def _splits(report):
    """What each split fitted and found, comparable across reports."""
    return [
        (split.seed, split.train.tolist(), split.test.tolist(), split.formulas, split.equation,
         split.score, split.n_undefined)
        for split in report.splits
    ]


def _classes(report):
    """The classes a report found, comparable across reports."""
    return (
        [(feature.formula, feature.spellings, feature.splits) for feature in report.features],
        [(equation.features, equation.splits) for equation in report.equations],
        [(split.feature_classes, split.equation_class) for split in report.splits],
    )


class TestDeterminism:
    def test_the_same_random_state_gives_the_same_report(self, planted):
        X, y = planted
        estimator = _fast(BeamFeatRegressor)
        first = stability_report(estimator, X, y, n_splits=4, random_state=3)
        second = stability_report(estimator, X, y, n_splits=4, random_state=3)
        assert _splits(first) == _splits(second)
        assert _classes(first) == _classes(second)
        assert first.summary() == second.summary()
        # The estimator passed in is cloned, never fitted.
        assert not hasattr(estimator, "features_")

    def test_another_random_state_draws_other_splits_and_seeds(self, planted):
        X, y = planted
        first = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=3, random_state=3)
        other = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=3, random_state=4)
        assert [s.seed for s in first.splits] != [s.seed for s in other.splits]
        assert all(not np.array_equal(a.test, b.test) for a, b in zip(first.splits, other.splits))

    def test_more_splits_extend_the_study_rather_than_redraw_it(self, planted):
        X, y = planted
        short = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=3, random_state=3)
        long = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=5, random_state=3)
        assert _splits(long)[:3] == _splits(short)

    def test_the_clone_seeds_are_spawned_from_random_state(self, planted):
        X, y = planted
        report = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=3, random_state=11)
        children = np.random.SeedSequence(11).spawn(2)[0].spawn(3)
        assert [split.seed for split in report.splits] == [int(c.generate_state(2)[1]) for c in children]
        assert [split.estimator.random_state for split in report.splits] == [
            split.seed for split in report.splits
        ]

    def test_n_jobs_does_not_change_the_report(self, planted):
        X, y = planted
        serial = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=4, random_state=5)
        parallel = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=4, random_state=5, n_jobs=2)
        assert _splits(parallel) == _splits(serial)
        assert _classes(parallel) == _classes(serial)


class TestHeldOut:
    def test_every_row_is_in_exactly_one_part_of_each_split(self, planted):
        X, y = planted
        report = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=5, test_size=0.3, random_state=0)
        for split in report.splits:
            assert len(split.test) == math.ceil(0.3 * len(y))
            rows = np.concatenate([split.train, split.test])
            assert np.array_equal(np.bincount(rows, minlength=len(y)), np.ones(len(y), dtype=int))
            assert np.array_equal(split.train, np.sort(split.train))

    def test_no_test_row_reaches_the_fit_and_the_test_rows_are_scored_once(self):
        rng = np.random.default_rng(1)
        X = np.column_stack([np.arange(160.0), rng.uniform(1.0, 6.0, (160, 2))])
        y = X[:, 1] * X[:, 2] + rng.normal(0, 0.05, 160)

        class Spy(BeamFeatRegressor):
            """Records the row numbers, in column 0, that fit and score receive.

            The regressor scores itself on its selection rows inside fit, as
            a diagnostic; those calls are recorded apart.
            """

            def fit(self, X, y):
                self.calls_, self.fitting_ = [("fit", np.asarray(X)[:, 0].astype(int).tolist())], True
                super().fit(X, y)
                self.fitting_ = False
                return self

            def score(self, X, y, sample_weight=None):
                call = "score inside fit" if self.fitting_ else "score"
                self.calls_.append((call, np.asarray(X)[:, 0].astype(int).tolist()))
                return super().score(X, y, sample_weight)

        report = stability_report(_fast(Spy), X, y, n_splits=4, random_state=0)
        for split in report.splits:
            calls = split.estimator.calls_
            assert calls[0] == ("fit", split.train.tolist())
            assert calls[-1] == ("score", split.test.tolist())
            assert [call for call, _ in calls].count("score") == 1
            for call, rows in calls[1:-1]:
                assert call == "score inside fit" and set(rows) <= set(split.train.tolist())

    def test_each_split_is_the_fit_on_its_training_rows_alone(self, planted):
        X, y = planted
        estimator = _fast(BeamFeatRegressor)
        report = stability_report(estimator, X, y, n_splits=3, random_state=2)
        for split in report.splits:
            alone = clone(estimator).set_params(random_state=split.seed).fit(X[split.train], y[split.train])
            assert alone.formulas() == list(split.formulas)
            assert alone.equation() == split.equation
            assert alone.score(X[split.test], y[split.test]) == split.score

    def test_a_test_row_outside_a_formulas_domain_leaves_its_split_unscored(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(1.0, 6.0, (120, 2))
        X[17, 0] = -1.0
        y = 3 * np.log(np.abs(X[:, 0])) + rng.normal(0, 0.05, 120)
        report = stability_report(_fast(BeamFeatRegressor, max_depth=1), X, y, n_splits=4, random_state=0)
        unscored = [split for split in report.splits if np.isnan(split.score)]
        assert unscored
        for split in unscored:
            assert 17 in split.test
            assert "log(x0)" in split.formulas
            assert split.n_undefined == 1
        scored = report.scores[np.isfinite(report.scores)]
        assert len(scored) == len(report.splits) - len(unscored)
        assert (
            f"over the {len(scored)} splits scored; {len(unscored)} splits were not "
            "(test rows outside a selected formula's domain)"
        ) in report.summary()


class TestEquivalence:
    def test_spelling_variants_of_one_formula_are_one_feature(self):
        rng = np.random.default_rng(0)
        x0, x1, x2 = rng.uniform(1.0, 6.0, (3, 100))
        formulas = [("x0*x1",), ("x1*x0",), ("x2", "x1*x0")]
        values = [np.column_stack([x0 * x1]), np.column_stack([x1 * x0]), np.column_stack([x2, x1 * x0])]
        labels, n_classes = _feature_labels(formulas, values, min_rows=50)
        assert labels == [[0], [0], [1, 0]]
        assert n_classes == 2

    def test_rescalings_shifts_and_sign_flips_are_one_feature(self):
        rng = np.random.default_rng(0)
        x0, x1 = rng.uniform(1.0, 6.0, (2, 100))
        formulas = [("(x0 - x1)",), ("(x1 - x0)",), ("2*(x0 - x1) + 3",)]
        values = [np.column_stack([x0 - x1]), np.column_stack([x1 - x0]), np.column_stack([2 * (x0 - x1) + 3])]
        assert _feature_labels(formulas, values, min_rows=50) == ([[0], [0], [0]], 1)

    def test_features_correlated_below_the_threshold_stay_apart(self):
        rng = np.random.default_rng(0)
        x0 = rng.uniform(1.0, 6.0, 500)
        close = x0 + rng.normal(0, 0.06 * x0.std(), 500)  # correlation about 0.998
        assert 0.99 < abs(np.corrcoef(x0, close)[0, 1]) < 0.999
        labels, n_classes = _feature_labels(
            [("x0",), ("x0b",)], [x0[:, None], close[:, None]], min_rows=50
        )
        assert labels == [[0], [1]] and n_classes == 2

    def test_a_feature_undefined_on_most_rows_is_matched_by_formula_alone(self):
        rng = np.random.default_rng(0)
        x0 = rng.uniform(1.0, 6.0, 100)
        sparse = np.where(np.arange(100) < 10, x0, np.nan)  # defined on 10 rows of 100
        formulas = [("log(x0)",), ("log(x0)",), ("x0",)]
        values = [sparse[:, None], sparse[:, None], x0[:, None]]
        # Identical formulas are one class; the sparse one cannot be compared
        # by value, so it does not absorb x0, with which it agrees.
        assert _feature_labels(formulas, values, min_rows=50) == ([[0], [0], [1]], 2)

    def test_respelled_features_from_real_fits_are_one_class(self, planted):
        X, y = planted

        class Respelled(BeamFeatRegressor):
            """Prints (x0 * x1) as (abs(x0) * x1), its equal on positive data, for odd seeds."""

            def fit(self, X, y):
                super().fit(X, y)
                if self.random_state % 2:
                    product = combine("mul", leaf("x0"), leaf("x1"))
                    respelled = combine("mul", transform("abs", leaf("x0")), leaf("x1"))
                    self.features_ = [respelled if node == product else node for node in self.features_]
                    self.feature_formulas_ = [node.name for node in self.features_]
                return self

        report = stability_report(_fast(Respelled), X, y, n_splits=6, random_state=0)
        assert {split.formulas for split in report.splits} == {("(x0 * x1)",), ("(abs(x0) * x1)",)}
        assert len(report.features) == 1
        assert set(report.features[0].spellings) == {"(x0 * x1)", "(abs(x0) * x1)"}
        assert sum(report.features[0].spellings.values()) == 6
        assert len(report.equations) == 1 and report.equations[0].count == 6


class TestPlantedAndNull:
    def test_a_planted_feature_is_one_equation_in_every_split(self, planted):
        X, y = planted
        report = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=6, random_state=0)
        assert isinstance(report, StabilityReport)
        assert len(report.equations) == 1
        assert report.equations[0].formulas == ("(x0 * x1)",)
        assert report.equations[0].splits == tuple(range(6))
        assert [feature.formula for feature in report.features] == ["(x0 * x1)"]
        assert all(split.fdr_controlled for split in report.splits)
        assert np.all(report.scores > 0.99)
        text = report.summary()
        assert "1 distinct equation in 6 splits: every split printed it." in text
        assert "6/6  y ~ (x0 * x1)" in text

    def test_a_null_target_without_screening_prints_many_equations(self, planted):
        X, _ = planted
        noise = np.random.default_rng(3).standard_normal(len(X))
        report = stability_report(_fast(BeamFeatRegressor, selector=None), X, noise, n_splits=6, random_state=0)
        assert len(report.equations) == 6
        assert report.equations[0].count == 1
        assert len(report.features) > 6
        assert "6 distinct equations in 6 splits; none recurs." in report.summary()
        assert "Equations printed in more than one split" not in report.summary()
        # Every split is in exactly one equation class, and every selection
        # is counted once in the feature frequencies.
        assert sorted(i for equation in report.equations for i in equation.splits) == list(range(6))
        assert sum(feature.count for feature in report.features) == sum(
            len(set(split.feature_classes)) for split in report.splits
        )

    def test_a_null_target_with_screening_prints_the_empty_equation(self, planted):
        X, _ = planted
        noise = np.random.default_rng(3).standard_normal(len(X))
        with pytest.warns(NoDiscoveriesWarning):
            report = stability_report(_fast(BeamFeatRegressor), X, noise, n_splits=4, random_state=0)
        assert [equation.formulas for equation in report.equations] == [()]
        assert report.features == ()
        text = report.summary()
        assert "No split selected a feature." in text
        assert "4/4  (no feature selected)" in text

    def test_a_partly_ambiguous_target_states_how_often_the_commonest_equation_recurs(self, planted):
        X, _ = planted
        y = X[:, 0] + X[:, 1] + np.random.default_rng(0).normal(0, 2.0, len(X))
        report = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=6, random_state=0)
        counts = [equation.count for equation in report.equations]
        assert counts == sorted(counts, reverse=True)
        match = re.search(r"(\d+) distinct equations in 6 splits; (?:the most frequent in (\d+)|none recurs)\.",
                          report.summary())
        assert match and int(match.group(1)) == len(report.equations)
        for split in report.splits:
            assert split.index in report.equations[split.equation_class].splits
            for formula, position in zip(split.formulas, split.feature_classes, strict=True):
                assert formula in report.features[position].spellings
                assert split.index in report.features[position].splits


class TestEstimators:
    def test_the_classifier_is_scored_by_accuracy_on_stratified_splits(self, planted):
        X, _ = planted
        labels = (X[:, 0] * X[:, 1] > X[:, 2] * X[:, 3]).astype(int)
        report = stability_report(_fast(BeamFeatClassifier), X, labels, n_splits=4, random_state=0)
        assert report.score_name == "accuracy"
        for split in report.splits:
            assert split.score == split.estimator.score(X[split.test], labels[split.test])
            assert abs(labels[split.test].mean() - labels.mean()) <= 1 / len(split.test)
            assert split.equation.startswith("logit P(y = 1) = ")
        assert "Held-out accuracy: mean" in report.summary()

    def test_the_transformer_is_scored_by_a_ridge_regression_on_its_features(self, planted):
        X, y = planted
        report = stability_report(_fast(BeamFeatTransformer), X, y, n_splits=3, random_state=0)
        assert report.score_name == "R^2 of a ridge regression on the features"
        for split in report.splits:
            assert split.equation is None
            model = split.estimator
            scaler = StandardScaler().fit(model.transform(X[split.train]))
            ridge = RidgeCV(alphas=np.logspace(-4, 4, 25)).fit(
                scaler.transform(model.transform(X[split.train])), y[split.train]
            )
            expected = r2_score(y[split.test], ridge.predict(scaler.transform(model.transform(X[split.test]))))
            assert split.score == pytest.approx(expected, rel=1e-12)
        assert "Held-out R^2 of a ridge regression on the features" in report.summary()

    def test_a_transformer_that_selects_nothing_is_scored_as_the_training_mean(self, planted):
        X, _ = planted
        noise = np.random.default_rng(3).standard_normal(len(X))
        with pytest.warns(NoDiscoveriesWarning):
            report = stability_report(_fast(BeamFeatTransformer), X, noise, n_splits=2, random_state=0)
        for split in report.splits:
            mean = np.full(len(split.test), noise[split.train].mean())
            assert split.score == pytest.approx(r2_score(noise[split.test], mean), rel=1e-12)

    def test_a_classifying_transformer_is_scored_by_accuracy(self, planted):
        X, _ = planted
        labels = np.where(X[:, 0] * X[:, 1] > 12.0, "high", "low")
        estimator = _fast(BeamFeatTransformer, problem_type="classification")
        report = stability_report(estimator, X, labels, n_splits=3, random_state=0)
        assert report.score_name == "accuracy of a logistic regression on the features"
        assert all(0.8 < split.score <= 1.0 for split in report.splits)

    def test_pandas_input_names_the_formulas(self, planted):
        pd = pytest.importorskip("pandas")
        X, y = planted
        frame = pd.DataFrame(X, columns=["mass", "speed", "height", "time"])
        report = stability_report(_fast(BeamFeatRegressor), frame, pd.Series(y), n_splits=3, random_state=0)
        assert report.equations[0].formulas == ("(mass * speed)",)
        assert all(list(split.estimator.feature_names_in_) == list(frame.columns) for split in report.splits)
        on_arrays = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=3, random_state=0)
        assert [s.score for s in report.splits] == [s.score for s in on_arrays.splits]


class TestResample:
    def test_seeds_keep_the_first_splits_rows_and_vary_only_the_seed(self, planted):
        X, y = planted
        splits = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=4, random_state=3)
        seeds = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=4, random_state=3, resample="seeds")
        assert (splits.resample, seeds.resample) == ("splits", "seeds")
        assert [s.seed for s in seeds.splits] == [s.seed for s in splits.splits]
        assert len({s.seed for s in seeds.splits}) == 4
        first = splits.splits[0]
        for split in seeds.splits:
            assert np.array_equal(split.train, first.train) and np.array_equal(split.test, first.test)
        assert any(not np.array_equal(s.test, first.test) for s in splits.splits[1:])
        # The first split is the same in both designs, and the design is deterministic.
        assert _splits(seeds)[0] == _splits(splits)[0]
        again = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=4, random_state=3, resample="seeds")
        assert _splits(again) == _splits(seeds) and _classes(again) == _classes(seeds)

    def test_seeds_measure_only_what_the_seed_changes(self, planted):
        X, _ = planted
        y = X[:, 0] + X[:, 1] + np.random.default_rng(0).normal(0, 1.0, len(X))
        # Without a selector no step of the fit draws on the seed, so refits
        # on fixed rows are identical, while new rows move the score.
        estimator = _fast(BeamFeatRegressor, selector=None)
        seeds = stability_report(estimator, X, y, n_splits=4, random_state=0, resample="seeds")
        assert len(seeds.equations) == 1 and np.ptp(seeds.scores) == 0.0
        splits = stability_report(estimator, X, y, n_splits=4, random_state=0)
        assert np.ptp(splits.scores) > 0.0

    def test_the_summary_of_seeds_counts_fits(self, planted):
        X, y = planted
        report = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=3, random_state=0, resample="seeds")
        text = report.summary()
        assert text.startswith(
            "BeamFeatRegressor refitted with 3 seeds on one random split (random_state=0), "
            "fitted on 150 rows and scored on the other 50; only the search/selection split "
            "inside each fit varies."
        )
        assert "1 distinct equation in 3 fits: every fit printed it." in text
        assert "1 distinct feature, selected in 3 of the 3 fits." in text
        assert "Equations printed in more than one fit, most frequent first:" in text


class TestValidation:
    @pytest.mark.parametrize("estimator", [Ridge(), Ridge, BeamFeatRegressor, "BeamFeatRegressor"])
    def test_rejects_anything_but_an_estimator_instance(self, planted, estimator):
        X, y = planted
        with pytest.raises(TypeError, match="estimator must be"):
            stability_report(estimator, X, y)

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("n_splits", 1), ("n_splits", 0), ("n_splits", 2.5), ("n_splits", True),
            ("test_size", 0.0), ("test_size", 1.0), ("test_size", 3), ("test_size", -0.2),
            ("test_size", True), ("test_size", "0.25"),
            ("random_state", -1), ("random_state", 1.5), ("random_state", True), ("random_state", "0"),
            ("n_jobs", 0), ("n_jobs", True), ("n_jobs", 1.5),
            ("resample", "bootstrap"), ("resample", "Seeds"), ("resample", None),
        ],
    )
    def test_rejects_an_argument_out_of_range(self, planted, name, value):
        X, y = planted
        with pytest.raises(ValueError, match=name):
            stability_report(BeamFeatRegressor(), X, y, **{name: value})

    def test_rejects_a_missing_or_mismatched_target(self, planted):
        X, y = planted
        with pytest.raises(ValueError, match="requires y"):
            stability_report(BeamFeatRegressor(), X, None)
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            stability_report(BeamFeatRegressor(), X, y[:-1])

    def test_rejects_too_few_rows_for_the_test_size(self, planted):
        X, y = planted
        with pytest.raises(ValueError, match="both parts need at least 2"):
            stability_report(BeamFeatRegressor(), X[:5], y[:5], test_size=0.2)

    def test_accepts_fresh_entropy(self, planted):
        X, y = planted
        report = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=2, random_state=None)
        assert report.random_state is None and len(report.splits) == 2

    def test_summary_rejects_a_negative_row_count(self, planted):
        X, y = planted
        report = stability_report(_fast(BeamFeatRegressor), X, y, n_splits=2, random_state=0)
        with pytest.raises(ValueError, match="max_rows"):
            report.summary(max_rows=-1)
        assert "  ... and 1 more feature\n" in report.summary(max_rows=0)
        assert "2/2  (x0 * x1)" not in report.summary(max_rows=0)
