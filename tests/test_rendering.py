"""Tests for the renderings of a fit: equation(), latex(), the equation list and export."""

from __future__ import annotations

import re

import numpy as np
import pytest

from beamfeat import (
    BeamFeatClassifier,
    BeamFeatRegressor,
    BeamFeatTransformer,
    NoDiscoveriesWarning,
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


def _reference_forward_path(matrix, target):
    """The forward step re-solved by least squares at every step, to the end."""
    n_rows, n_columns = matrix.shape
    total_variance = float(np.var(target))
    chosen, remaining, scores = [], list(range(n_columns)), []
    while remaining:
        step = []
        for index in remaining:
            design = np.column_stack([np.ones(n_rows)] + [matrix[:, j] for j in chosen + [index]])
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
            step.append(1.0 - float(np.var(target - design @ coefficients)) / total_variance)
        best = int(np.argmax(step))
        chosen.append(remaining.pop(best))
        scores.append(step[best])
    return chosen, scores


class TestClassifierEquation:
    def test_binary_equation_reproduces_decision_function(self):
        """The printed log-odds equation, evaluated on raw feature values,
        must equal decision_function — the same model-equation identity the
        regressor guarantees."""
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (400, 4))
        y = (X[:, 0] * X[:, 1] > X[:, 2] * X[:, 3]).astype(int)
        model = BeamFeatClassifier(max_depth=2, beam_width=20, random_state=0).fit(X, y)
        equation = model.equation(precision=10)
        assert equation.startswith("logit P(y = ")

        features = model._apply(X[:12])
        scale = np.where(model.scaler_.scale_ == 0, 1.0, model.scaler_.scale_)
        raw = np.atleast_2d(model.coef_)[0] / scale
        offset = float(model.intercept_[0] - np.sum(raw * model.scaler_.mean_))
        manual = features @ raw + offset
        np.testing.assert_allclose(manual, model.decision_function(X[:12]), rtol=1e-9)

    def test_labels_print_as_plain_values(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (400, 4))
        y = (X[:, 0] * X[:, 1] > X[:, 2] * X[:, 3]).astype(np.int64)
        model = BeamFeatClassifier(max_depth=1, beam_width=10, random_state=0).fit(X, y)
        equation = model.equation()
        assert equation.startswith("logit P(y = 1) = ")
        assert "np." not in equation
        named = BeamFeatClassifier(max_depth=1, beam_width=10, random_state=0).fit(
            X, np.where(y == 1, "yes", "no")
        )
        assert named.equation().startswith("logit P(y = 'yes') = ")

    def test_multiclass_equation_has_one_line_per_class(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(1, 6, (450, 4))
        y = np.digitize(X[:, 0] * X[:, 1], [8.0, 18.0])
        model = BeamFeatClassifier(max_depth=1, beam_width=10, random_state=0).fit(X, y)
        lines = model.equation().splitlines()
        assert sum(line.startswith("score(") for line in lines) == len(model.classes_)
        assert lines[0].startswith("score(0) = ")

    def test_null_model_equation_reports_priors(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = (rng.uniform(size=400) > 0.5).astype(int)
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatClassifier(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3
            ).fit(X, y)
        assert "class priors only" in model.equation()


class TestParsimonyNote:
    """``equation()`` marks a pruned equation as a subset of the certified set.

    The guarantee is a statement about the screened set, and the printed
    equation is normally a subset of it. A caller who reads the equation and
    the ``fdr_controlled_`` flag side by side would otherwise have no signal
    that the two do not refer to the same set of terms."""

    MARKER = "certified terms; parsimony=None prints all"

    def test_present_when_terms_were_dropped(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", beam_width=25).fit(X, y)
        if model.n_features_out_ >= model.selection_result_.n_selected:
            pytest.skip("this fit pruned nothing, so there is no gap to mark")
        equation = model.equation()
        assert self.MARKER in equation
        assert f"[{model.n_features_out_} of {model.selection_result_.n_selected}" in equation

    def test_absent_when_parsimony_is_off(self, regression_data):
        """The terms then are the certified set, so there is nothing to mark."""
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation",
                      beam_width=25, parsimony=None).fit(X, y)
        assert self.MARKER not in model.equation()

    def test_absent_without_a_selector(self, regression_data):
        """Nothing was certified, so no claim about a certified set is implied."""
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector=None).fit(X, y)
        assert self.MARKER not in model.equation()

    def test_absent_when_the_subset_was_re_certified(self, regression_data):
        """``parsimony_holdout`` re-tests the subset on rows of its own, so the
        guarantee does cover the printed terms and the note would be wrong."""
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation",
                      parsimony_holdout=0.5).fit(X, y)
        if model.certification_result_ is None:
            pytest.skip("the rows did not support a second split")
        assert self.MARKER not in model.equation()

    def test_absent_when_no_null_is_rejected(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3
            ).fit(X, rng.standard_normal(400))
        assert self.MARKER not in model.equation()

    def test_does_not_disturb_the_equation_itself(self, regression_data):
        """The note is appended; the coefficients and formulas are untouched."""
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", beam_width=25).fit(X, y)
        head = model.equation().split("   [")[0]
        assert head.startswith("y = ")
        assert any(formula in head for formula in model.formulas())

    def test_classifier_marks_the_block_once(self, rng):
        """The terms are shared across a multiclass equation's lines, so one
        note belongs at the end of the block rather than on every line."""
        X = rng.uniform(1.0, 6.0, (400, 4))
        score = X[:, 0] * X[:, 1]
        y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
        model = _fast(BeamFeatClassifier, selector="permutation", beam_width=25).fit(X, y)
        equation = model.equation()
        if self.MARKER in equation:
            assert equation.count(self.MARKER) == 1


class TestEquationList:
    """``equations_`` records the forward path through the whole screened set.

    The fitted model must be exactly the one the forward step picks at
    ``parsimony_tol``; the path only adds a record of the sizes it did not
    pick."""

    SCOPE = "not an equation chosen from this list"

    @pytest.fixture
    def model(self, regression_data):
        X, y = regression_data
        return _fast(BeamFeatRegressor, selector="permutation", beam_width=25).fit(X, y)

    def test_one_record_per_screened_size(self, model):
        records = model.equations_
        assert len(records) == model.selection_result_.n_selected
        assert [record["n_terms"] for record in records] == list(range(1, len(records) + 1))
        screened = {row["formula"] for row in model.selection_report_ if row["screened"]}
        assert set(records[-1]["formulas"]) == screened

    def test_records_are_nested_and_monotone(self, model):
        records = model.equations_
        for shorter, longer in zip(records, records[1:]):
            assert longer["formulas"][:-1] == shorter["formulas"]
            assert longer["complexity"] > shorter["complexity"]
            assert longer["r2"] >= shorter["r2"] - 1e-9
        assert all(0.0 <= record["r2"] <= 1.0 + 1e-12 for record in records)

    def test_complexity_counts_nodes_and_coefficients(self, model):
        def tree_size(node):
            return 1 + sum(tree_size(child) for child in node.children)

        for record in model.equations_:
            assert record["complexity"] == sum(
                tree_size(node) + 1 + node.n_constants for node in record["features"]
            )
            assert [node.name for node in record["features"]] == record["formulas"]

    def test_the_marked_record_is_the_fitted_model(self, model):
        marked = [record for record in model.equations_ if record["model"]]
        assert len(marked) == 1
        assert sorted(marked[0]["formulas"]) == sorted(model.feature_formulas_)
        assert marked[0]["n_terms"] == model._forward_stop_

    def test_the_marked_record_is_where_the_forward_step_stops(self, model):
        records = model.equations_
        stop = next(record["n_terms"] for record in records if record["model"])
        if stop < len(records):
            assert records[stop]["r2"] - records[stop - 1]["r2"] < model.parsimony_tol + 1e-9
        for record, previous in zip(records[1:stop], records[: stop - 1]):
            assert record["r2"] - previous["r2"] >= model.parsimony_tol

    def test_matches_a_path_re_solved_at_every_step(self, rng):
        """The steps past the forward step's stopping point are taken by
        projection rather than re-solving; they must agree with re-solving,
        including around an exactly collinear column."""
        latent = rng.standard_normal((150, 4))
        matrix = np.column_stack(
            [latent @ rng.standard_normal(4) + 0.3 * rng.standard_normal(150) for _ in range(12)]
        )
        matrix[:, 5] = 2.0 * matrix[:, 1] + 1.0
        target = matrix[:, :3] @ np.array([1.0, -0.5, 0.25]) + rng.standard_normal(150)
        reference, reference_scores = _reference_forward_path(matrix, target)

        estimator = BeamFeatRegressor(parsimony_tol=0.05)
        order, scores, stop = estimator._forward_path(matrix, target, list(range(12)))
        assert 1 <= stop < 12
        assert order[:stop] == reference[:stop]
        assert scores[:stop] == reference_scores[:stop]
        np.testing.assert_allclose(scores, reference_scores, atol=1e-10)
        assert sorted(order) == list(range(12))
        assert estimator._parsimonious_subset(matrix, target, list(range(12))) == sorted(order[:stop])

    def test_does_not_change_the_fitted_model(self, regression_data):
        """The path is the same whatever ``parsimony`` is, and only the
        marked record moves."""
        X, y = regression_data
        forward = _fast(BeamFeatRegressor, selector="permutation", beam_width=25).fit(X, y)
        full = _fast(BeamFeatRegressor, selector="permutation", beam_width=25,
                     parsimony=None).fit(X, y)
        assert [r["formulas"] for r in forward.equations_] == [r["formulas"] for r in full.equations_]
        assert [r["r2"] for r in forward.equations_] == [r["r2"] for r in full.equations_]
        assert full.equations_[-1]["model"]
        assert sorted(full.equations_[-1]["formulas"]) == sorted(full.feature_formulas_)

    def test_parsimony_none_marks_the_screened_set(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", beam_width=25,
                      parsimony=None).fit(X, y)
        text = model.equations()
        assert "parsimony=None keeps the whole screened set" in text
        assert self.SCOPE in text
        assert [r["model"] for r in model.equations_].count(True) == 1

    def test_text_marks_the_model_and_states_the_scope(self, model):
        text = model.equations()
        lines = text.splitlines()
        stop = next(record["n_terms"] for record in model.equations_ if record["model"])
        marked = [line for line in lines if line.startswith("*") and " ~ " in line]
        assert len(marked) == 1
        assert marked[0].split()[1] == str(stop)
        assert lines[-1].startswith("The FDR guarantee covers the screened set of "
                                    f"{len(model.equations_)} terms")
        assert self.SCOPE in lines[-1]
        assert f"parsimony_tol={model.parsimony_tol:g}" in text
        for formula in model.equations_[0]["formulas"]:
            assert formula in text

    def test_text_can_be_shortened(self, model):
        records = model.equations_
        if len(records) < 4:
            pytest.skip("too few screened terms to shorten the list")
        text = model.equations(max_n_terms=2)
        rows = [line for line in text.splitlines() if " ~ " in line]
        sizes = [int(line.lstrip("* ").split()[0]) for line in rows]
        stop = next(record["n_terms"] for record in records if record["model"])
        assert sizes == sorted({1, 2, stop})
        assert "..." in text
        assert text.splitlines()[-1].endswith(self.SCOPE + ".")

    def test_rejects_a_bad_precision(self, model):
        with pytest.raises(ValueError, match="precision"):
            model.equations(precision=0)

    def test_none_without_a_selector(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector=None).fit(X, y)
        assert model.equations_ is None
        assert "no selector was configured" in model.equations()

    def test_empty_when_no_null_is_rejected(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3
            ).fit(X, rng.standard_normal(400))
        assert model.equations_ == []
        assert model.equations() == "no equation list: the screen rejected no null"

    def test_a_single_screened_feature(self, rng):
        X = rng.uniform(1.0, 6.0, (300, 1))
        y = 2 * X[:, 0] + rng.normal(0, 0.1, 300)
        model = BeamFeatRegressor(max_depth=0, random_state=0).fit(X, y)
        assert len(model.equations_) == 1
        record = model.equations_[0]
        assert record["formulas"] == ["x0"]
        assert record["complexity"] == 2
        assert record["model"]
        assert record["r2"] > 0.99
        text = model.equations()
        assert "the screened set of 1 term (line 1)" in text
        assert "the forward step keeps every screened term" in text

    def test_classification(self, classification_data):
        X, y = classification_data
        model = _fast(BeamFeatClassifier, selector="permutation").fit(X, y)
        records = model.equations_
        assert len(records) == model.selection_result_.n_selected
        assert [record["model"] for record in records].count(True) == 1
        marked = next(record for record in records if record["model"])
        assert sorted(marked["formulas"]) == sorted(model.feature_formulas_)
        text = model.equations()
        assert "logit P(y = 1) ~ " in text
        assert "integer-coded labels" in text

    def test_multiclass_and_string_labels(self, rng):
        X = rng.uniform(1.0, 6.0, (400, 4))
        score = X[:, 0] * X[:, 1]
        labels = np.array(["low", "mid", "high"])[np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))]
        model = _fast(BeamFeatClassifier, selector="permutation", beam_width=25).fit(X, labels)
        assert model.equations_
        assert "scores ~ " in model.equations()

    def test_transformer_records_it_too(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatTransformer, selector="permutation", beam_width=25).fit(X, y)
        assert len(model.equations_) == model.selection_result_.n_selected
        assert "* the returned features" in model.equations()

    def test_same_data_selection_claims_no_guarantee(self, regression_data):
        X, y = regression_data
        with pytest.warns(UserWarning, match="same rows"):
            model = _fast(BeamFeatRegressor, selector="permutation",
                          selection_holdout=None).fit(X, y)
        text = model.equations()
        assert "No line carries the FDR guarantee" in text
        assert "The FDR guarantee covers" not in text

    def test_re_certified_model(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation",
                      parsimony_holdout=0.5).fit(X, y)
        if model.certification_result_ is None:
            pytest.skip("the rows did not support a second split")
        text = model.equations()
        marked = [record for record in model.equations_ if record["model"]]
        if marked:
            assert sorted(marked[0]["formulas"]) == sorted(model.feature_formulas_)
        else:
            assert "No line is the fitted model" in text
        assert "re-test" in text
        assert "does not cover another equation chosen from this list" in text

    def test_unfitted_raises(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            BeamFeatRegressor().equations()


class TestLatex:
    """``latex()`` is ``equation()`` typeset: the same terms, order,
    coefficients and note, with each feature rendered by sympy."""

    @pytest.fixture
    def model(self, regression_data):
        pytest.importorskip("sympy")
        X, y = regression_data
        return _fast(BeamFeatRegressor, selector="permutation", beam_width=25,
                     parsimony_tol=1e-4).fit(X, y)

    def test_same_coefficients_as_the_equation(self, model):
        equation, latex = model.equation(), model.latex()
        assert latex.startswith("y = ")
        coefficients = [
            match.group(1)
            for match in re.finditer(r"(\d+(?:\.\d+)?(?:e[+-]\d+)?)\*", equation)
        ]
        assert coefficients
        for coefficient in coefficients:
            assert coefficient in latex
        assert equation.split("   [")[0].rsplit(" ", 1)[-1] in latex

    def test_contains_every_feature(self, model):
        latex = model.latex()
        for formula in model.latex_formulas():
            assert formula in latex

    def test_latex_formulas_come_from_sympy(self, model):
        sympy = pytest.importorskip("sympy")
        assert model.latex_formulas() == [sympy.latex(e) for e in model.to_sympy()]

    def test_carries_the_parsimony_note(self, model):
        latex = model.latex()
        if model.fdr_scope_ == "screened set":
            assert r"\quad \text{[" in latex
            assert "certified terms; parsimony=None prints all" in latex
        else:
            assert r"\text{[" not in latex

    def test_note_false_returns_the_bare_equation(self, model):
        bare = model.latex(note=False)
        assert r"\quad" not in bare and r"\text{" not in bare
        assert model.latex().startswith(bare)

    def test_no_note_when_parsimony_is_off(self, regression_data):
        pytest.importorskip("sympy")
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", beam_width=25,
                      parsimony=None).fit(X, y)
        assert "certified terms" not in model.latex(max_terms=3)

    def test_options_follow_equation(self, model):
        scientific = model.latex(style="scientific", precision=2)
        assert r"\times 10^{" in scientific
        assert "e+" not in scientific and "e-" not in scientific
        assert model.latex(max_terms=1).count(r"\quad") <= 1
        with pytest.raises(ValueError, match="precision"):
            model.latex(precision=0)
        with pytest.raises(ValueError, match="style"):
            model.latex(style="roman")

    def test_sums_are_parenthesised(self):
        sympy = pytest.importorskip("sympy")
        from beamfeat._rendering import _render_linear_latex

        class Scaler:
            mean_ = np.array([0.0, 0.0])
            scale_ = np.array([1.0, 1.0])

        a, b = sympy.symbols("a b", real=True)
        rhs = _render_linear_latex(
            np.array([2.0, -0.5]), 1.0, Scaler(), [a + b, a * b], 4, None, "significant"
        )
        assert rhs == r"2 \left(a + b\right) - 0.5 a b + 1"

    def test_numbers_and_text(self):
        from beamfeat._rendering import _latex_label, _latex_number, _latex_text

        assert _latex_number("3.2e-07") == r"3.2 \times 10^{-7}"
        assert _latex_number("1.50e+03") == r"1.50 \times 10^{3}"
        assert _latex_number("0.003943") == "0.003943"
        assert _latex_text("a_b & 50%") == r"\text{a\_b \& 50\%}"
        assert _latex_label(np.int64(1)) == "1"
        assert _latex_label("yes") == r"\text{yes}"

    def test_null_model(self):
        pytest.importorskip("sympy")
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3
            ).fit(X, rng.standard_normal(400))
        assert model.latex().endswith(r"\quad \text{(no feature selected)}")
        assert r"\text{" not in model.latex(note=False)

    def test_binary_classifier(self, classification_data):
        pytest.importorskip("sympy")
        X, y = classification_data
        model = _fast(BeamFeatClassifier).fit(X, y)
        latex = model.latex()
        assert latex.startswith(r"\operatorname{logit} P(y = 1) = ")
        for formula in model.latex_formulas():
            assert formula in latex

    def test_classifier_with_string_labels(self, rng):
        pytest.importorskip("sympy")
        X = rng.uniform(1.0, 6.0, (300, 3))
        labels = np.where(X[:, 0] * X[:, 1] > 9, "yes", "no")
        model = _fast(BeamFeatClassifier, max_depth=1).fit(X, labels)
        assert model.latex().startswith(r"\operatorname{logit} P(y = \text{yes}) = ")

    def test_multiclass_classifier(self, rng):
        pytest.importorskip("sympy")
        X = rng.uniform(1.0, 6.0, (400, 4))
        score = X[:, 0] * X[:, 1]
        y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
        model = _fast(BeamFeatClassifier, selector="permutation", beam_width=25).fit(X, y)
        latex = model.latex()
        assert latex.startswith("\\begin{aligned}\n")
        assert latex.count(r"\operatorname{score}(") == 3
        assert latex.count(" &= ") == 3
        assert "argmax of the softmax scores" in latex
        assert model.latex(note=False).endswith("\\end{aligned}")
        if model.fdr_scope_ == "screened set":
            assert latex.count("certified terms") == 1

    def test_classifier_null_model(self):
        pytest.importorskip("sympy")
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatClassifier(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3
            ).fit(X, rng.integers(0, 2, 400))
        latex = model.latex()
        assert latex.startswith("P(y = 0) = ")
        assert "no feature selected" in latex

    def test_missing_sympy_fails_as_to_sympy_does(self, model, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "sympy", None)
        with pytest.raises(ImportError) as from_to_sympy:
            model.to_sympy()
        for method in (model.latex, model.latex_formulas):
            with pytest.raises(ImportError) as raised:
                method()
            assert type(raised.value) is type(from_to_sympy.value)
            assert "sympy" in str(raised.value)


class TestLambdifyExport:
    """The export recipe in the guide: ``sympy.lambdify`` on ``to_sympy()``
    gives functions of the input columns that reproduce the fitted features,
    in NumPy here and in JAX or PyTorch with ``modules="jax"`` or
    ``"torch"``."""

    @staticmethod
    def _lambdified(model, n_columns):
        sympy = pytest.importorskip("sympy")
        columns = [sympy.Symbol(f"x{i}", real=True) for i in range(n_columns)]
        return sympy.lambdify(columns, model.to_sympy(), modules="numpy")

    def test_features_reproduce_transform(self, regression_data, rng):
        X, y = regression_data
        model = _fast(BeamFeatTransformer, selector="permutation", beam_width=25,
                      parsimony=None).fit(X, y)
        features = self._lambdified(model, X.shape[1])
        X_new = rng.uniform(1.0, 6.0, (200, X.shape[1]))
        exported = np.column_stack(features(*X_new.T))
        np.testing.assert_allclose(exported, model.transform(X_new), rtol=1e-10, atol=1e-12)

    def test_equation_reproduces_predict(self, regression_data, rng):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", beam_width=25).fit(X, y)
        features = self._lambdified(model, X.shape[1])
        X_new = rng.uniform(1.0, 6.0, (200, X.shape[1]))
        raw = model.coef_ / model.scaler_.scale_
        offset = model.intercept_ - raw @ model.scaler_.mean_
        exported = np.column_stack(features(*X_new.T)) @ raw + offset
        np.testing.assert_allclose(exported, model.predict(X_new), rtol=1e-10, atol=1e-10)
