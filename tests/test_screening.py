"""Tests for the screen inside a fit: selection on held-out rows, selector
settings, and the input test."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from beamfeat import (
    BeamFeatClassifier,
    BeamFeatRegressor,
    BeamFeatTransformer,
    NoDiscoveriesError,
    NoDiscoveriesWarning,
    PermutationSelector,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20260723)


@pytest.fixture
def regression_data(rng):
    """y = (a * b) / c, so the true expression needs depth 2."""
    n = 400
    X = rng.uniform(1.0, 6.0, (n, 4))
    y = (X[:, 0] * X[:, 1]) / X[:, 2] + rng.normal(0, 0.05, n)
    return X, y


@pytest.fixture
def classification_data(rng):
    n = 400
    X = rng.uniform(1.0, 6.0, (n, 4))
    score = X[:, 0] * X[:, 1]
    return X, (score > np.median(score)).astype(int)


def _fast(cls, **kwargs):
    """Estimator with cheap settings so the suite stays quick."""
    kwargs.setdefault("max_depth", 2)
    kwargs.setdefault("beam_width", 20)
    kwargs.setdefault("selector", None)
    return cls(**kwargs)


class TestSelectionIntegration:
    def test_selection_result_is_exposed(self, regression_data):
        X, y = regression_data
        model = BeamFeatRegressor(max_depth=2, beam_width=20, selector="permutation").fit(X, y)
        assert model.selection_result_ is not None
        assert model.selection_result_.method == "permutation"

    def test_search_result_is_exposed(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        assert model.search_result_ is not None
        assert model.search_result_.n_proposed_total > 0

    def test_selector_none_skips_selection(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector=None).fit(X, y)
        assert model.selection_result_ is None
        assert model.fdr_controlled_ is None

    def test_flag_true_when_the_screen_rejects_a_null(self, regression_data):
        X, y = regression_data
        model = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", target_fdr=0.2, random_state=0
        ).fit(X, y)
        if model.fdr_controlled_:
            # Parsimony keeps a subset of the screened set, never more.
            assert 0 < model.n_features_out_ <= model.selection_result_.n_selected

    def test_empty_default_on_pure_noise(self):
        """Default on_no_discoveries="empty": on a noise target where nothing
        passes, the model keeps no features, warns visibly, and predicts the
        training mean — the honest null model."""
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = rng.standard_normal(400)
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, selector="permutation", target_fdr=0.05, random_state=3
            ).fit(X, y)
        assert model.n_features_out_ == 0
        assert model.fdr_controlled_ is False
        np.testing.assert_allclose(model.predict(X[:7]), np.full(7, y.mean()), rtol=1e-9)
        assert "no feature selected" in model.equation()

    def test_fallback_mode_keeps_search_output_flagged(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = rng.standard_normal(400)
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3,
                on_no_discoveries="fallback",
            ).fit(X, y)
        assert model.n_features_out_ > 0
        assert model.fdr_controlled_ is False

    def test_raise_mode_raises(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = rng.standard_normal(400)
        with pytest.raises(NoDiscoveriesError):
            BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3,
                on_no_discoveries="raise",
            ).fit(X, y)

    @pytest.mark.parametrize("cls", [BeamFeatRegressor, BeamFeatClassifier, BeamFeatTransformer])
    def test_parsimony_none_returns_the_whole_screened_set(self, cls, regression_data,
                                                           classification_data):
        """``parsimony=None`` fits the screened set itself rather than a subset
        chosen from it, which is what makes the printed equation the object the
        guarantee covers."""
        X, y = (classification_data if cls is BeamFeatClassifier else regression_data)
        model = cls(max_depth=2, beam_width=20, selector="permutation",
                    random_state=0, parsimony=None).fit(X, y)
        screened = {row["formula"] for row in model.selection_report_ if row["screened"]}
        assert screened, "the fixture is meant to produce discoveries"
        assert set(model.feature_formulas_) == screened
        assert model.n_features_out_ == model.selection_result_.n_selected
        assert model.fdp_inflation_ == pytest.approx(1.0)

    @pytest.mark.parametrize("cls", [BeamFeatRegressor, BeamFeatClassifier, BeamFeatTransformer])
    def test_default_prunes_within_the_screened_set(self, cls, regression_data,
                                                    classification_data):
        """The default is ``parsimony="forward"``: what is fitted is a subset of
        what was screened, and the screen rejected the null of every member."""
        X, y = (classification_data if cls is BeamFeatClassifier else regression_data)
        model = cls(max_depth=2, beam_width=20, selector="permutation",
                    random_state=0).fit(X, y)
        assert model.parsimony == "forward"
        screened = {row["formula"] for row in model.selection_report_ if row["screened"]}
        assert set(model.feature_formulas_) <= screened
        assert model.n_features_out_ <= model.selection_result_.n_selected
        assert model.fdp_inflation_ >= 1.0

    def test_parsimony_compacts_without_losing_recovery(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (400, 4))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)
        compact = BeamFeatRegressor(max_depth=2, beam_width=25, random_state=0).fit(X, y)
        full = BeamFeatRegressor(
            max_depth=2, beam_width=25, random_state=0, parsimony=None
        ).fit(X, y)
        assert compact.n_features_out_ < full.n_features_out_
        assert any("x0" in f and "x1" in f for f in compact.formulas())
        assert compact.score(X, y) > 0.99
        # the screen rejected the null of every kept feature
        report = {r["formula"]: r for r in compact.selection_report_}
        assert all(report[f]["screened"] for f in compact.formulas())

    def test_selection_report_is_consistent(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(1, 6, (400, 4))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)
        model = BeamFeatRegressor(max_depth=2, beam_width=20, random_state=1).fit(X, y)
        report = model.selection_report_
        assert report and all(
            set(row) == {"formula", "p_value", "q_value", "statistic", "screened", "kept"}
            for row in report
        )
        for row in report:
            if row["kept"]:
                assert row["screened"]
                assert row["q_value"] <= model.target_fdr + 1e-9

    def test_nan_error_is_actionable(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (50, 3))
        X[3, 1] = np.nan
        with pytest.raises(ValueError, match="SimpleImputer"):
            BeamFeatRegressor(max_depth=1, beam_width=5).fit(X, rng.standard_normal(50))

    def test_same_data_selection_never_claims_control(self, regression_data):
        """With the holdout disabled, candidates are tested on the rows that
        chose them; the estimator must not claim the FDR guarantee."""
        X, y = regression_data
        model = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", selection_holdout=None, random_state=0
        ).fit(X, y)
        assert model.fdr_controlled_ is False

    def test_holdout_split_is_disjoint_and_seed_stable(self, regression_data, monkeypatch):
        """The rows the search sees, the rows screening tests on and the rows
        the re-test uses must not overlap. Each target value is unique, so the
        rows a stage saw are identified by the targets it was given."""
        from beamfeat.search import BeamSearch
        from beamfeat.selection import PermutationSelector

        X, y = regression_data
        assert len(np.unique(y)) == len(y)
        seen = {"search": [], "select": []}
        run, select = BeamSearch.run, PermutationSelector.select

        def recording_run(self, data, target, *args, **kwargs):
            seen["search"].append(np.asarray(target))
            return run(self, data, target, *args, **kwargs)

        def recording_select(self, features, target):
            seen["select"].append(np.asarray(target))
            return select(self, features, target)

        monkeypatch.setattr(BeamSearch, "run", recording_run)
        monkeypatch.setattr(PermutationSelector, "select", recording_select)

        first = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", random_state=7,
            parsimony_holdout=0.5,
        ).fit(X, y)
        stages = [seen["search"][0], *seen["select"]]
        assert len(stages) == 3
        for i, earlier in enumerate(stages):
            for later in stages[i + 1:]:
                assert not set(earlier) & set(later)
        assert sum(len(stage) for stage in stages) == len(y)
        np.testing.assert_array_equal(np.sort(y[first._holdout_rows_]),
                                      np.sort(np.concatenate(seen["select"])))

        second = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", random_state=7,
            parsimony_holdout=0.5,
        ).fit(X, y)
        np.testing.assert_array_equal(first._holdout_rows_, second._holdout_rows_)
        assert first.formulas() == second.formulas()

    def test_selected_features_still_predict(self, regression_data):
        X, y = regression_data
        model = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", target_fdr=0.2
        ).fit(X, y)
        assert model.score(X, y) > 0.9

    def test_pipeline_level_fdr_calibration(self):
        """End-to-end control: search on one split, selection on the other.

        A feature is a false discovery iff its formula references only
        irrelevant columns. FDR is an expectation, so the mean FDP over
        trials is what must respect the nominal level (with Monte Carlo
        slack) — not each individual trial.
        """
        nominal = 0.2
        fdps = []
        recovered = 0
        n_trials = 12
        for trial in range(n_trials):
            rng = np.random.default_rng(100 + trial)
            X = rng.uniform(1, 6, (400, 6))
            signal = X[:, 0] * X[:, 1]
            y = signal + rng.normal(0, 0.05 * np.std(signal), 400)
            model = BeamFeatRegressor(
                max_depth=2, beam_width=20, selector="permutation", target_fdr=nominal,
                selection_holdout=0.5, random_state=trial,
            ).fit(X, y)
            if not model.fdr_controlled_:
                continue
            relevant = {"x0", "x1"}
            false = sum(1 for node in model.features_ if not (node.columns() & relevant))
            fdps.append(false / len(model.features_))
            recovered += any("x0" in f and "x1" in f for f in model.formulas())
        assert fdps, "the screen never rejected a null; the pipeline has no power"
        assert float(np.mean(fdps)) <= nominal + 0.15
        assert recovered >= len(fdps) - 2


class TestSelectorDiagnosticsSurfaced:
    def test_knockoff_unsatisfiability_is_a_visible_warning(self):
        """A knockoff configuration that cannot select must warn through the
        warnings module, not only the logger, so notebook users see it."""
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (300, 4))
        y = X[:, 0] / X[:, 1] + rng.normal(0, 0.02, 300)
        with pytest.warns(UserWarning, match="offset"):
            BeamFeatRegressor(
                max_depth=2, beam_width=15, selector="knockoff", target_fdr=0.1, random_state=0
            ).fit(X, y)


class TestSelectorInstance:
    """An instance keeps its own settings; a disagreement with the estimator's
    is named rather than resolved in silence."""

    def test_disagreeing_correction_is_named(self, regression_data):

        X, y = regression_data
        with pytest.warns(UserWarning, match=r"selection_correction='by'.*correction='bh'"):
            model = _fast(BeamFeatRegressor, selector=PermutationSelector()).fit(X, y)
        assert model.selection_result_ is not None

    def test_disagreeing_target_fdr_is_named(self, regression_data):

        X, y = regression_data
        with pytest.warns(UserWarning, match=r"target_fdr=0.05.*target_fdr=0.1"):
            _fast(
                BeamFeatRegressor, target_fdr=0.05, selector=PermutationSelector(correction="by")
            ).fit(X, y)

    def test_agreeing_instance_is_silent(self, regression_data):

        X, y = regression_data
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fast(BeamFeatRegressor, selector=PermutationSelector(correction="by")).fit(X, y)
        assert not [entry for entry in caught if "instance" in str(entry.message)]

    def test_verbose_reports_the_correction_that_ran(self, regression_data, capsys):

        X, y = regression_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _fast(BeamFeatRegressor, selector=PermutationSelector(), verbose=1).fit(X, y)
        assert "permutation (BH)" in capsys.readouterr().out


class TestSingleCandidate:
    def test_a_lone_candidate_is_screened(self, rng):
        """With one candidate, screening used to be skipped and the term
        returned unvetted."""
        X = rng.uniform(1.0, 6.0, (300, 1))
        y = 2 * X[:, 0] + rng.normal(0, 0.1, 300)
        model = BeamFeatRegressor(max_depth=0, random_state=0).fit(X, y)
        assert model.selection_result_ is not None
        assert model.selection_result_.n_candidates == 1
        assert model.fdr_controlled_ is True
        assert model.formulas() == ["x0"]

    def test_a_lone_null_candidate_can_be_screened_out(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1.0, 6.0, (300, 1))
        y = rng.standard_normal(300)
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(max_depth=0, random_state=0).fit(X, y)
        assert model.selection_result_.p_values[0] > 0.1
        assert model.n_features_out_ == 0


class TestInputTest:
    """``input_test=True`` drops screened terms that read a column they do not need."""

    @staticmethod
    def _distractors(seed=0, n=500):
        rng = np.random.default_rng(seed)
        X = rng.uniform(1.0, 6.0, (n, 12))
        signal = X[:, 0] * X[:, 1]
        return X, signal + rng.normal(0, 0.25 * np.std(signal), n)

    @staticmethod
    def _columns(model):
        return set().union(*(node.columns() for node in model.features_))

    def test_removes_junk_columns_from_the_equation(self):
        # Without fitted constants, whose search keeps no rider on this seed.
        X, y = self._distractors(seed=0)
        plain = BeamFeatRegressor(random_state=0, fitted_ops=()).fit(X, y)
        tested = BeamFeatRegressor(random_state=0, input_test=True, fitted_ops=()).fit(X, y)
        assert self._columns(plain) - {"x0", "x1"}
        assert self._columns(tested) == {"x0", "x1"}
        assert tested.fdr_controlled_ is True and tested.fdr_scope_ == "screened set"
        assert "input_test dropped" in tested.equation()

    def test_report_lists_every_pair(self):
        X, y = self._distractors(seed=0)
        model = BeamFeatRegressor(random_state=0, input_test=True).fit(X, y)
        report = model.input_report_
        assert report and set(report[0]) == {"formula", "column", "p_value", "q_value", "null_rejected"}
        tested = {(row["formula"], row["column"]) for row in report}
        for formula in model.formulas():
            node = next(n for n in model.features_ if n.name == formula)
            if len(node.columns()) > 1:
                assert all((formula, c) in tested for c in node.columns())
                assert all(row["null_rejected"] for row in report if row["formula"] == formula)

    def test_off_by_default_and_changes_nothing(self):
        X, y = self._distractors(seed=1)
        default = BeamFeatRegressor(random_state=0).fit(X, y)
        explicit = BeamFeatRegressor(random_state=0, input_test=False).fit(X, y)
        assert default.input_report_ is None
        assert default.formulas() == explicit.formulas()
        np.testing.assert_array_equal(default.predict(X), explicit.predict(X))

    def test_with_parsimony_none_and_holdout(self):
        X, y = self._distractors(seed=2, n=800)
        whole = BeamFeatRegressor(random_state=0, input_test=True, parsimony=None).fit(X, y)
        assert self._columns(whole) <= {"x0", "x1"}
        certified = BeamFeatRegressor(random_state=0, input_test=True, parsimony_holdout=0.5).fit(X, y)
        assert self._columns(certified) <= {"x0", "x1"}

    def test_scope_after_a_re_test_that_rejects_no_null(self):
        # The input test drops terms and the conditional re-test of the rest
        # rejects no null, so the screened set less those terms is returned:
        # a subset of the set the guarantee covers.
        X, y = self._distractors(seed=11)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = BeamFeatRegressor(
                random_state=0, input_test=True, parsimony=None, parsimony_holdout=0.5,
                certify_test="conditional",
            ).fit(X, y)
        assert model.certification_result_.n_selected == 0
        assert model.n_features_out_ < model.selection_result_.n_selected
        assert model.fdr_scope_ == "screened set"
        assert "input_test dropped" in model.equation()

    def test_n_jobs_gives_the_same_fit(self):
        X, y = self._distractors(seed=4)
        serial = BeamFeatRegressor(random_state=0, input_test=True).fit(X, y)
        threaded = BeamFeatRegressor(random_state=0, input_test=True, n_jobs=2).fit(X, y)
        assert serial.input_report_ == threaded.input_report_
        assert serial.formulas() == threaded.formulas()

    def test_classifier(self):
        X, y = self._distractors(seed=3)
        labels = (y > np.median(y)).astype(int)
        model = BeamFeatClassifier(random_state=0, input_test=True).fit(X, labels)
        assert model.input_report_ is not None
        assert self._columns(model) <= {"x0", "x1"}

    def test_falls_back_when_it_would_drop_every_term(self, monkeypatch):
        from beamfeat.selection import SelectionResult, _InputPermutationTest

        def rejects_no_null(self, nodes, data, target):
            pairs = [(i, c) for i, n in enumerate(nodes) if len(n.columns()) > 1 for c in sorted(n.columns())]
            ones = np.ones(len(pairs))
            return pairs, SelectionResult(
                statistics=0 * ones, p_values=ones, q_values=ones, n_candidates=len(pairs),
                method="input_permutation",
            )

        monkeypatch.setattr(_InputPermutationTest, "test_inputs", rejects_no_null)
        X, y = self._distractors(seed=0)
        with pytest.warns(NoDiscoveriesWarning, match="as if input_test were False"):
            # Without fitted constants, whose one-column terms need no test.
            model = BeamFeatRegressor(
                random_state=0, input_test=True, include_originals=False, fitted_ops=()
            ).fit(X, y)
        assert model.n_features_out_ > 0

    @pytest.mark.parametrize("value", ["yes", 1, None])
    def test_rejects_non_boolean(self, value):
        X, y = self._distractors()
        with pytest.raises(ValueError, match="input_test must be True or False"):
            BeamFeatRegressor(input_test=value).fit(X, y)

    def test_rejects_without_a_selector(self):
        X, y = self._distractors()
        with pytest.raises(ValueError, match="runs no screen"):
            BeamFeatRegressor(input_test=True, selector=None).fit(X, y)
