"""How much a fitted equation depends on the split it was fitted on.

A fit prints one equation, found on one random split of its training rows
into a search part and a selection part. Where many expressions predict about
equally well, another split, or another sample of rows, prints another
equation, and nothing in a single fit shows whether the printed one is *the*
discovery or one draw among many equally good ones. :func:`stability_report`
measures it, as the stability study of the beamfeat paper (TMLR 2026,
Appendix E) did, by refitting on many random splits and counting the distinct
equations they print.

**Splits.** Each of ``n_splits`` splits divides the rows at random into a
training part and a test part of ``test_size`` (stratified by class when the
estimator classifies). A clone of the estimator, with a ``random_state`` of
its own, is fitted on the training part: the search, the screen, the
parsimony step and the downstream model never see the test rows. The clone is
then scored once on the test part. The partitions and the clones' seeds come
from a :class:`numpy.random.SeedSequence` of ``random_state``, one child per
split, so a report is the same on every run with the same ``random_state``,
whatever ``n_jobs`` is, and the first ``k`` splits are the same whatever
``n_splits`` is.

**What varies.** By default (``resample="splits"``) each split draws its own
partition as well as its own seed. The report then measures split-to-split
variability: how far the equation and the score move with another sample of
training rows, another test part and another seed, and the spread of the
scores includes the variation between test parts. With ``resample="seeds"``
every split keeps the partition of the first, and only the clones' seeds
differ, as in the paper's study. That measures seed-to-seed variability on
fixed data: how far the equation moves with the random search/selection
split inside the fit alone, which is the split the guarantee is conditional
on, and the scores vary only through it.

**Equivalence.** Formulas spelled differently can be the same feature:
``log((x0 * x1))`` and ``(log(x0) + log(x1))``, or ``(x0 - x1)`` and
``(x1 - x0)``. Every selected feature is evaluated on the rows of ``X``, or on
1500 of them drawn at random when there are more, and the target is not used.
A feature belongs to the first class, in order of appearance, whose first
member it correlates with above 0.999 in absolute value, on at least 50 rows
(or on every row, when there are fewer) where both are defined. This is the
value-equivalence of the paper's study. It merges rewritings, and rescalings,
shifts and sign flips, which a linear model's coefficient and intercept
absorb. It also merges near-duplicates whose values agree that closely on the
data, such as one expression on two almost identical columns. Identical
formulas are always one class, and a feature that cannot be compared, being
defined on too few rows or constant on them, shares its class only with
identical formulas. An equation is the set of the classes of its features:
two splits print the same equation when their features fall in the same
classes, whatever the coefficients.
"""

from __future__ import annotations

import math
import numbers
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import clone

from beamfeat.estimators import (
    BeamFeatClassifier,
    BeamFeatRegressor,
    BeamFeatTransformer,
    _BeamFeatBase,
)
from beamfeat.scoring import is_constant
from beamfeat.search import _map

__all__ = ["EquationClass", "FeatureClass", "SplitResult", "StabilityReport", "stability_report"]

# Two features are one when their values correlate above this, in absolute
# value: the threshold of the paper's stability study.
EQUIVALENCE_THRESHOLD = 0.999

# Rows of X the features are compared on, at most; more are subsampled.
PROBE_ROWS = 1500

# Rows on which two features must both be defined to be compared, unless X
# has fewer, in which case every row is needed.
MIN_PROBE_ROWS = 50

# The downstream ridge penalties for a transformer's features, as in the
# paper's stability study.
_RIDGE_ALPHAS = np.logspace(-4, 4, 25)


@dataclass(frozen=True, eq=False)
class SplitResult:
    """One split of a :class:`StabilityReport`.

    Attributes:
        index: Position of the split, from 0.
        seed: The ``random_state`` the clone was fitted with.
        train: The rows the clone was fitted on, as positions in ``X``,
            ascending.
        test: The rows it was scored on: every row not in :attr:`train`.
        formulas: The selected features' formulas, as ``formulas()``
            returns them.
        equation: The fitted equation, as ``equation()`` prints it; ``None``
            for a transformer, which has none.
        score: The score on the test rows, NaN when it is undefined.
        n_undefined: Test rows on which a selected formula is undefined
            (outside its domain, or not finite). They leave a regressor's or
            transformer's score undefined; the classifier predicts its most
            frequent training class on them, as its ``predict`` does.
        fdr_controlled: The clone's ``fdr_controlled_``.
        feature_classes: The class of each formula, as a position in
            :attr:`StabilityReport.features`.
        equation_class: The class of the equation, as a position in
            :attr:`StabilityReport.equations`.
        estimator: The fitted clone.
    """

    index: int
    seed: int
    train: np.ndarray = field(repr=False)
    test: np.ndarray = field(repr=False)
    formulas: tuple[str, ...]
    equation: str | None
    score: float
    n_undefined: int
    fdr_controlled: bool | None
    feature_classes: tuple[int, ...]
    equation_class: int
    estimator: Any = field(repr=False)


@dataclass(frozen=True, eq=False)
class FeatureClass:
    """Selected features that are one feature, up to equivalence.

    Attributes:
        formula: The spelling selected in the most splits, which names the
            class; the first to appear, among equally frequent ones.
        spellings: Every formula of the class, with the number of splits that
            selected it, most frequent first.
        splits: The splits that selected a feature of the class, ascending.
    """

    formula: str
    spellings: dict[str, int]
    splits: tuple[int, ...]

    @property
    def count(self) -> int:
        """Number of splits that selected a feature of the class."""
        return len(self.splits)


@dataclass(frozen=True, eq=False)
class EquationClass:
    """Printed equations that are one equation, up to equivalence.

    Attributes:
        features: The classes of its features, as positions in
            :attr:`StabilityReport.features`, ascending, so the most
            frequent first. Empty for the equation of a split that selected
            no feature.
        formulas: The formulas naming those classes, in the same order.
        splits: The splits that printed it, ascending.
    """

    features: tuple[int, ...]
    formulas: tuple[str, ...]
    splits: tuple[int, ...]

    @property
    def count(self) -> int:
        """Number of splits that printed the equation."""
        return len(self.splits)


@dataclass(frozen=True, eq=False)
class StabilityReport:
    """What :func:`stability_report` returns; ``print`` it for the summary.

    Attributes:
        estimator: An unfitted clone of the estimator the splits refitted.
        score_name: What :attr:`scores` measures: ``"R^2"`` or
            ``"accuracy"``, followed for a transformer by the model fitted on
            its features.
        test_size: The fraction of the rows each split held out.
        resample: ``"splits"``, when each split drew its own partition and
            seed, or ``"seeds"``, when they shared one partition and only
            the seeds differed.
        random_state: The seed the splits and the clones' seeds came from.
        splits: One :class:`SplitResult` per split, in order.
        features: The distinct features selected, one
            :class:`FeatureClass` each, the most frequent first, and in order
            of first appearance among equally frequent ones.
        equations: The distinct equations printed, one
            :class:`EquationClass` each, ordered in the same way.
    """

    estimator: Any = field(repr=False)
    score_name: str
    test_size: float
    resample: str
    random_state: int | None
    splits: tuple[SplitResult, ...] = field(repr=False)
    features: tuple[FeatureClass, ...] = field(repr=False)
    equations: tuple[EquationClass, ...] = field(repr=False)

    @property
    def scores(self) -> np.ndarray:
        """The held-out score of each split, NaN where it is undefined."""
        return np.array([split.score for split in self.splits], dtype=np.float64)

    @property
    def _unit(self) -> str:
        """What the summary calls one refit: a split, or a fit when only the seed varied."""
        return "fit" if self.resample == "seeds" else "split"

    def summary(self, max_rows: int = 5) -> str:
        """The report as text: scores, equations and features.

        Args:
            max_rows: Most equations, and most features, listed by name. The
                equations listed are those printed in more than one split;
                the rest of either list is counted.

        Returns:
            The text, ending with the definition of equivalence it counts by.
        """
        if isinstance(max_rows, bool) or not isinstance(max_rows, numbers.Integral) or max_rows < 0:
            raise ValueError(f"max_rows must be a non-negative integer, got {max_rows!r}")
        n_splits = len(self.splits)
        first = self.splits[0]
        name = type(self.estimator).__name__
        if self.resample == "seeds":
            header = (
                f"{name} refitted with {n_splits} seeds on one random split "
                f"(random_state={self.random_state}), fitted on {len(first.train)} rows and "
                f"scored on the other {len(first.test)}; only the search/selection split "
                "inside each fit varies."
            )
        else:
            header = (
                f"{name} refitted on {n_splits} random splits "
                f"(random_state={self.random_state}), each fitted on {len(first.train)} rows "
                f"and scored on the other {len(first.test)}."
            )
        lines = [header, "", self._score_line(), self._equation_line(), self._feature_line()]
        lhs = first.estimator._path_lhs()
        recurring = [equation for equation in self.equations if equation.count > 1]
        if recurring:
            rows = [
                (f"{equation.count}/{n_splits}",
                 f"{lhs} ~ " + " + ".join(equation.formulas) if equation.formulas
                 else "(no feature selected)")
                for equation in recurring
            ]
            lines += [
                "",
                f"Equations printed in more than one {self._unit}, most frequent first:",
                *_table(rows, max_rows, "equation"),
            ]
        rows = [(f"{feature.count}/{n_splits}", feature.formula) for feature in self.features]
        if rows:
            lines += ["", "Features, most frequent first:", *_table(rows, max_rows, "feature")]
        lines += [
            "",
            f"Features whose values on the rows of X correlate above {EQUIVALENCE_THRESHOLD:g} in "
            "absolute value are counted as one; an equation is the set of its features, whatever "
            "its coefficients.",
        ]
        return "\n".join(lines)

    def _score_line(self) -> str:
        """The spread of the held-out scores, and the splits left unscored."""
        scores = self.scores
        scored = scores[np.isfinite(scores)]
        unscored = len(scores) - len(scored)
        reason = "test rows outside a selected formula's domain"
        unit = self._unit
        if not len(scored):
            return f"Held-out {self.score_name}: undefined in every {unit} ({reason})."
        line = (
            f"Held-out {self.score_name}: mean {np.mean(scored):.3f}, sd {np.std(scored):.3f}, "
            f"min {np.min(scored):.3f}, max {np.max(scored):.3f}"
        )
        if unscored:
            verb = f"{unit} was" if unscored == 1 else f"{unit}s were"
            line += f", over the {len(scored)} {unit}s scored; {unscored} {verb} not ({reason})"
        return line + "."

    def _equation_line(self) -> str:
        """The ambiguity of the printed equation, in one sentence."""
        n_splits, n_equations, unit = len(self.splits), len(self.equations), self._unit
        if n_equations == 1:
            return f"1 distinct equation in {n_splits} {unit}s: every {unit} printed it."
        top = self.equations[0].count
        if top == 1:
            return f"{n_equations} distinct equations in {n_splits} {unit}s; none recurs."
        return f"{n_equations} distinct equations in {n_splits} {unit}s; the most frequent in {top}."

    def _feature_line(self) -> str:
        """How many distinct features were selected, and how often the commonest was."""
        unit = self._unit
        if not self.features:
            return f"No {unit} selected a feature."
        n_features, top = len(self.features), self.features[0].count
        if n_features == 1:
            return f"1 distinct feature, selected in {top} of the {len(self.splits)} {unit}s."
        return (
            f"{n_features} distinct features; the most frequent in {top} of the "
            f"{len(self.splits)} {unit}s."
        )

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"StabilityReport({type(self.estimator).__name__}, n_splits={len(self.splits)}, "
            f"equations={len(self.equations)}, features={len(self.features)})"
        )


def _table(rows: list[tuple[str, str]], max_rows: int, noun: str) -> list[str]:
    """Indented ``count  text`` lines, the first ``max_rows`` of ``rows``."""
    if not rows:
        return []
    width = max(len(count) for count, _ in rows)
    lines = [f"  {count.rjust(width)}  {text}" for count, text in rows[:max_rows]]
    hidden = len(rows) - max_rows
    if hidden > 0:
        lines.append(f"  ... and {hidden} more {noun}{'' if hidden == 1 else 's'}")
    return lines


# --------------------------------------------------------------------------- #
# The study
# --------------------------------------------------------------------------- #


def stability_report(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    n_splits: int = 30,
    test_size: float = 0.25,
    resample: str = "splits",
    random_state: int | None = 0,
    n_jobs: int | None = 1,
) -> StabilityReport:
    """Refit an estimator on random splits and count the distinct equations.

    Each split fits a clone of ``estimator`` on a random training part of the
    rows, with a seed of its own, and scores it once on the remaining test
    part, which no step of the fit sees. The report counts the distinct
    equations and features the splits printed, up to the equivalence the
    module documentation defines, and records the spread of the scores. See
    the module documentation for how the splits and seeds are drawn, and for
    what each choice of ``resample`` measures.

    The score is the estimator's own ``score`` on the test part: R^2 for
    :class:`~beamfeat.estimators.BeamFeatRegressor`, accuracy for
    :class:`~beamfeat.estimators.BeamFeatClassifier`. A transformer has no
    score of its own, so a model is fitted on its features over the training
    part: for regression a ridge regression on the standardised features, its
    penalty chosen by leave-one-out cross-validation among 25 values from
    1e-4 to 1e4, scored by R^2, as in the paper's stability study; for
    classification the logistic regression
    :class:`~beamfeat.estimators.BeamFeatClassifier` fits, scored by
    accuracy. When the transformer selects nothing, the prediction is the
    training mean or the most frequent training class.

    A test row on which a selected formula is undefined leaves the R^2 of a
    regressor undefined, as its ``score`` says by raising, and likewise a
    transformer's score: that split's score is NaN, and the summary counts
    it. The classifier predicts its most frequent training class on such a
    row, as its ``predict`` does, and is scored.

    Args:
        estimator: A :class:`~beamfeat.estimators.BeamFeatRegressor`,
            :class:`~beamfeat.estimators.BeamFeatClassifier` or
            :class:`~beamfeat.estimators.BeamFeatTransformer`, fitted or not.
            It is cloned and left as it is. Each clone's ``random_state`` is
            the split's seed; every other setting is kept, including a scorer
            or selector instance's own seed.
        X (array-like): ``(n_samples, n_features)`` data. A DataFrame's column
            names flow into the formulas, as they do in a fit.
        y (array-like): Target, one entry per row of ``X``.
        n_splits: Number of splits, at least 2.
        test_size: Fraction of the rows each split holds out of its fit and
            scores on, strictly between 0 and 1. The test part has
            ``ceil(test_size * n_samples)`` rows, as in scikit-learn's
            ``ShuffleSplit``, and both parts need at least 2.
        resample: ``"splits"`` (default) draws a partition and a seed for
            each split, and measures split-to-split variability.
            ``"seeds"`` keeps the first split's partition for every split
            and varies only the seeds, as the paper's stability study did,
            and measures seed-to-seed variability on fixed data. The seeds
            are the same either way, so the first split is too.
        random_state: Seed of the partitions and of the clones'
            ``random_state``: a non-negative integer, or ``None`` for fresh
            entropy, in which case the report differs from run to run.
        n_jobs: Splits fitted at the same time, through joblib. ``1``
            (default) fits them one after another in the calling process,
            ``-1`` uses every core, and ``None`` follows an enclosing joblib
            configuration (``joblib.parallel_config``, or
            ``parallel_backend`` before joblib 1.3) or else means 1. The
            workers are processes unless that configuration names another
            backend. The report is the same for every value. Each clone
            still runs its own search with the estimator's ``n_jobs``, so
            parallelise one or the other.

    Returns:
        The :class:`StabilityReport`.

    Raises:
        TypeError: If ``estimator`` is not one of the three estimators.
        ValueError: If an argument is out of range, or ``X`` and ``y``
            disagree in length or have too few rows for ``test_size``.
    """
    from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
    from sklearn.utils import _safe_indexing
    from sklearn.utils.validation import _num_samples, indexable

    _check_estimator(estimator)
    _check_arguments(n_splits, test_size, random_state, n_jobs)
    if resample not in ("splits", "seeds"):
        raise ValueError(f"resample must be 'splits' or 'seeds', got {resample!r}")
    if y is None:
        raise ValueError("stability_report requires y; feature construction is supervised")
    X, y = indexable(X, y)
    n_samples = _num_samples(X)
    n_test = math.ceil(float(test_size) * n_samples)
    if n_test < 2 or n_samples - n_test < 2:
        raise ValueError(
            f"test_size={test_size!r} splits {n_samples} rows into {n_samples - n_test} to fit "
            f"and {n_test} to score; both parts need at least 2"
        )

    # The workers receive an unfitted copy, which is all a clone keeps.
    template = clone(estimator)
    splitter = StratifiedShuffleSplit if _classifies(estimator) else ShuffleSplit
    entropy = None if random_state is None else int(random_state)
    split_sequence, probe_sequence = np.random.SeedSequence(entropy).spawn(2)
    plans = []
    for child in split_sequence.spawn(int(n_splits)):
        partition_seed, clone_seed = (int(word) for word in child.generate_state(2))
        if plans and resample == "seeds":
            # The first split's partition, for every split: only the seed varies.
            train, test = plans[0][0], plans[0][1]
        else:
            train, test = (np.sort(rows) for rows in next(
                splitter(n_splits=1, test_size=float(test_size), random_state=partition_seed).split(X, y)
            ))
        plans.append((train, test, clone_seed))

    if n_samples > PROBE_ROWS:
        probe = np.sort(np.random.default_rng(probe_sequence).choice(n_samples, PROBE_ROWS, replace=False))
    else:
        probe = np.arange(n_samples)
    X_probe = _safe_indexing(X, probe)

    outcomes = _map(
        _fit_split,
        [(template, X, y, train, test, seed, X_probe) for train, test, seed in plans],
        n_jobs,
        prefer="processes",
    )

    formulas = [tuple(model.formulas()) for model, *_ in outcomes]
    labels, n_classes = _feature_labels(
        formulas, [values for *_, values in outcomes], min(MIN_PROBE_ROWS, len(probe))
    )

    # Features: the splits and the spellings of each class, then the classes
    # ordered by frequency, the earlier first among equals.
    members: list[list[int]] = [[] for _ in range(n_classes)]
    spellings: list[dict[str, int]] = [{} for _ in range(n_classes)]
    for index, (split_formulas, split_labels) in enumerate(zip(formulas, labels, strict=True)):
        for formula, label in zip(split_formulas, split_labels, strict=True):
            spellings[label][formula] = spellings[label].get(formula, 0) + 1
            if not members[label] or members[label][-1] != index:
                members[label].append(index)
    order = sorted(range(n_classes), key=lambda label: (-len(members[label]), label))
    position = {label: rank for rank, label in enumerate(order)}
    features = []
    for label in order:
        ranked = dict(sorted(spellings[label].items(), key=lambda item: -item[1]))
        features.append(FeatureClass(next(iter(ranked)), ranked, tuple(members[label])))

    # Equations: the set of their features' classes.
    equation_splits: dict[tuple[int, ...], list[int]] = {}
    for index, split_labels in enumerate(labels):
        key = tuple(sorted({position[label] for label in split_labels}))
        equation_splits.setdefault(key, []).append(index)
    keys = sorted(equation_splits, key=lambda key: -len(equation_splits[key]))
    equations = tuple(
        EquationClass(key, tuple(features[rank].formula for rank in key), tuple(equation_splits[key]))
        for key in keys
    )
    equation_of = {index: rank for rank, key in enumerate(keys) for index in equation_splits[key]}

    splits = tuple(
        SplitResult(
            index=index,
            seed=seed,
            train=train,
            test=test,
            formulas=formulas[index],
            equation=None if isinstance(model, BeamFeatTransformer) else model.equation(),
            score=score,
            n_undefined=n_undefined,
            fdr_controlled=model.fdr_controlled_,
            feature_classes=tuple(position[label] for label in labels[index]),
            equation_class=equation_of[index],
            estimator=model,
        )
        for index, ((train, test, seed), (model, score, n_undefined, _)) in enumerate(
            zip(plans, outcomes, strict=True)
        )
    )
    return StabilityReport(
        estimator=template,
        score_name=_score_name(estimator),
        test_size=float(test_size),
        resample=resample,
        random_state=random_state,
        splits=splits,
        features=tuple(features),
        equations=equations,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _check_estimator(estimator: Any) -> None:
    """Accept an instance of one of the three estimators, and nothing else."""
    if isinstance(estimator, type) and issubclass(estimator, _BeamFeatBase):
        raise TypeError(
            f"estimator must be an instance, such as {estimator.__name__}(), not the class itself"
        )
    if not isinstance(estimator, (BeamFeatRegressor, BeamFeatClassifier, BeamFeatTransformer)):
        name = estimator.__name__ if isinstance(estimator, type) else type(estimator).__name__
        raise TypeError(
            "estimator must be a BeamFeatRegressor, BeamFeatClassifier or BeamFeatTransformer "
            f"instance, got {name}"
        )


def _check_arguments(n_splits: Any, test_size: Any, random_state: Any, n_jobs: Any) -> None:
    """Reject arguments that are out of range or of the wrong type."""
    if isinstance(n_splits, bool) or not isinstance(n_splits, numbers.Integral) or n_splits < 2:
        raise ValueError(f"n_splits must be an integer of at least 2, got {n_splits!r}")
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, numbers.Real)
        or not 0.0 < float(test_size) < 1.0
    ):
        raise ValueError(f"test_size must be a fraction strictly between 0 and 1, got {test_size!r}")
    if random_state is not None and (
        isinstance(random_state, bool)
        or not isinstance(random_state, numbers.Integral)
        or random_state < 0
    ):
        raise ValueError(f"random_state must be a non-negative integer or None, got {random_state!r}")
    if n_jobs is not None and (
        isinstance(n_jobs, bool) or not isinstance(n_jobs, numbers.Integral) or n_jobs == 0
    ):
        raise ValueError(f"n_jobs must be a non-zero integer or None, got {n_jobs!r}")


def _classifies(estimator: Any) -> bool:
    """Whether the estimator's target is a class label."""
    if isinstance(estimator, BeamFeatTransformer):
        return estimator.problem_type == "classification"
    return isinstance(estimator, BeamFeatClassifier)


def _score_name(estimator: Any) -> str:
    """What the held-out score measures, as the summary names it."""
    if isinstance(estimator, BeamFeatTransformer):
        if _classifies(estimator):
            return "accuracy of a logistic regression on the features"
        return "R^2 of a ridge regression on the features"
    return "accuracy" if _classifies(estimator) else "R^2"


# --------------------------------------------------------------------------- #
# One split
# --------------------------------------------------------------------------- #


def _fit_split(
    estimator: Any, X: Any, y: Any, train: np.ndarray, test: np.ndarray, seed: int, X_probe: Any
) -> tuple[Any, float, int, np.ndarray]:
    """Fit a seeded clone on the training rows, then score it on the test rows.

    Returns the fitted clone, its held-out score, the number of test rows on
    which a selected formula is undefined, and the selected features' values
    on the probe rows.
    """
    from sklearn.utils import _safe_indexing

    model = clone(estimator).set_params(random_state=seed)
    X_train, y_train = _safe_indexing(X, train), _safe_indexing(y, train)
    model.fit(X_train, y_train)
    # The test rows are indexed only now, after the fit has returned.
    X_test, y_test = _safe_indexing(X, test), _safe_indexing(y, test)
    test_values = _feature_values(model, X_test)
    n_undefined = int(np.count_nonzero(np.isnan(test_values).any(axis=1)))
    if isinstance(model, BeamFeatTransformer):
        score = (
            math.nan if n_undefined
            else _transformer_score(model, _feature_values(model, X_train), y_train, test_values, y_test)
        )
    elif n_undefined and isinstance(model, BeamFeatRegressor):
        score = math.nan
    else:
        score = float(model.score(X_test, y_test))
    return model, score, n_undefined, _feature_values(model, X_probe)


def _feature_values(model: Any, X: Any) -> np.ndarray:
    """The selected features on the rows of ``X``, NaN where undefined.

    Evaluated as ``transform`` evaluates them, without its warning about the
    undefined rows: the caller counts those itself.
    """
    from sklearn.utils.validation import check_array

    matrix = check_array(X, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return model._apply(matrix)


def _transformer_score(
    model: BeamFeatTransformer, train_values: np.ndarray, y_train: Any, test_values: np.ndarray, y_test: Any
) -> float:
    """The documented score of a transformer: a downstream model on its features."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.preprocessing import StandardScaler

    from beamfeat._compat import logistic_regression

    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    if _classifies(model):
        labels, counts = np.unique(y_train, return_counts=True)
        if train_values.shape[1] == 0 or len(labels) < 2:
            prediction = np.full(len(y_test), labels[np.argmax(counts)])
        else:
            scaler = StandardScaler().fit(train_values)
            classifier = logistic_regression(C=1.0, max_iter=2000, random_state=model.random_state)
            prediction = classifier.fit(scaler.transform(train_values), y_train).predict(
                scaler.transform(test_values)
            )
        return float(accuracy_score(y_test, prediction))
    if train_values.shape[1] == 0:
        return float(r2_score(y_test, np.full(len(y_test), np.mean(y_train))))
    scaler = StandardScaler().fit(train_values)
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS).fit(scaler.transform(train_values), y_train)
    return float(r2_score(y_test, ridge.predict(scaler.transform(test_values))))


# --------------------------------------------------------------------------- #
# Equivalence
# --------------------------------------------------------------------------- #


class _Representatives:
    """The first member of each value class found so far, scaled to [-1, 1]."""

    def __init__(self, n_rows: int) -> None:
        self._values = np.empty((16, n_rows))
        self._labels: list[int] = []

    def match(self, column: np.ndarray, min_rows: int) -> int | None:
        """The label of the first class ``column`` correlates with, or ``None``."""
        if not self._labels:
            return None
        block = self._values[: len(self._labels)]
        both = np.isfinite(block) & np.isfinite(column)
        n_rows = both.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            left = np.where(both, column, 0.0)
            left = np.where(both, left - (left.sum(axis=1) / n_rows)[:, None], 0.0)
            right = np.where(both, block, 0.0)
            right = np.where(both, right - (right.sum(axis=1) / n_rows)[:, None], 0.0)
            correlation = (left * right).sum(axis=1) / np.sqrt(
                (left * left).sum(axis=1) * (right * right).sum(axis=1)
            )
        hits = np.flatnonzero((n_rows >= min_rows) & (np.abs(correlation) > EQUIVALENCE_THRESHOLD))
        return self._labels[int(hits[0])] if hits.size else None

    def add(self, column: np.ndarray, label: int) -> None:
        """Make ``column`` the representative of a new class."""
        if len(self._labels) == len(self._values):
            self._values = np.concatenate([self._values, np.empty_like(self._values)])
        self._values[len(self._labels)] = column
        self._labels.append(label)


def _feature_labels(
    formulas: list[tuple[str, ...]], values: list[np.ndarray], min_rows: int
) -> tuple[list[list[int]], int]:
    """Assign every selected feature to an equivalence class.

    ``formulas`` and ``values`` hold, split by split, the selected formulas
    and their values on the probe rows, one column per formula. Returns the
    class of each formula, split by split, with classes numbered in order of
    first appearance, and the number of classes.
    """
    n_rows = values[0].shape[0] if values else 0
    representatives = _Representatives(n_rows)
    known: dict[str, int] = {}
    labels: list[list[int]] = []
    n_classes = 0
    for split_formulas, split_values in zip(formulas, values, strict=True):
        split_labels = []
        for formula, column in zip(split_formulas, split_values.T, strict=True):
            label = known.get(formula)
            if label is None:
                finite = np.isfinite(column)
                comparable = (
                    np.count_nonzero(finite) >= min_rows and not is_constant(column[finite])
                )
                if comparable:
                    column = column / np.max(np.abs(column[finite]))
                    label = representatives.match(column, min_rows)
                if label is None:
                    label, n_classes = n_classes, n_classes + 1
                    if comparable:
                        representatives.add(column, label)
                known[formula] = label
            split_labels.append(label)
        labels.append(split_labels)
    return labels, n_classes
