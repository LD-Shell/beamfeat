"""Compatibility with the dependency versions beamfeat supports.

scikit-learn is supported from 1.6 on, and its logistic regression has
changed across that range: 1.7 deprecates fitting more than two classes
with the ``liblinear`` solver, which fits two classes only from 1.8; 1.8
deprecates ``penalty`` in favour of ``l1_ratio`` and ``l1_ratios``; 1.10
deprecates ``C`` in favour of ``alpha = 1/C``, and ``liblinear`` itself.
The helpers here build the models beamfeat fits internally with the
arguments the installed version expects, so that none of those
deprecations is triggered, and a fit on a version older than a change is
the one it always was. The exception is 1.7's warning about ``liblinear``
with more than two classes, which is silenced: on 1.7 the fit is the one
beamfeat has always made, and from 1.8 on each class is fitted on its own.
"""

from __future__ import annotations

import inspect
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np


def _sklearn_version() -> tuple[int, int]:
    """The installed scikit-learn's major and minor version."""
    import sklearn

    match = re.match(r"(\d+)\.(\d+)", sklearn.__version__)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


@contextmanager
def quiet() -> Iterator[None]:
    """Silence the user and runtime warnings of models fitted internally.

    Convergence notices and floating-point warnings from the auxiliary models
    beamfeat fits are noise to its users. Deprecations are not: a
    ``DeprecationWarning``, ``PendingDeprecationWarning`` or ``FutureWarning``
    still follows the caller's filters, so a dependency deprecating an
    argument beamfeat passes reaches the test suite, which fails on it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def logistic_regression(C: float = 1.0, **kwargs: Any) -> Any:
    """A ``LogisticRegression`` with inverse regularisation strength ``C``.

    From scikit-learn 1.10 the strength is given as ``alpha = 1/C``, which
    fits the same model.
    """
    from sklearn.linear_model import LogisticRegression

    if "alpha" in inspect.signature(LogisticRegression).parameters:
        return LogisticRegression(alpha=1.0 / C, **kwargs)
    return LogisticRegression(C=C, **kwargs)


def l1_logistic_coefficients(
    design: np.ndarray, target: np.ndarray, cv: int, random_state: int | None
) -> np.ndarray:
    """Coefficients of a cross-validated L1-penalised logistic fit.

    Args:
        design: ``(n_samples, n_features)`` matrix.
        target: Class labels, two or more classes.
        cv: Number of cross-validation folds.
        random_state: Seed for the solver.

    Returns:
        ``(1, n_features)`` for two classes, otherwise one row per class from
        fitting that class against the rest, as ``liblinear`` always has.
    """
    from sklearn.linear_model import LogisticRegressionCV

    parameters = inspect.signature(LogisticRegressionCV).parameters
    if parameters["penalty"].default != "deprecated":
        model = LogisticRegressionCV(
            cv=cv, penalty="l1", solver="liblinear", max_iter=2000, random_state=random_state
        )
        with warnings.catch_warnings():
            # 1.7 warns that liblinear will fit two classes only from 1.8,
            # where the branch below fits each class on its own.
            warnings.filterwarnings(
                "ignore", message="Using the 'liblinear' solver for multiclass", category=FutureWarning
            )
            return model.fit(design, target).coef_

    options: dict[str, Any] = {
        "cv": cv,
        "l1_ratios": (1.0,),
        # Accuracy is the earlier default; 1.11 changes it to log loss, which
        # would select a different penalty.
        "scoring": "accuracy",
        "solver": "liblinear" if _sklearn_version() < (1, 10) else "newton-cd",
        "max_iter": 2000,
        "random_state": random_state,
    }
    if "use_legacy_attributes" in parameters:
        options["use_legacy_attributes"] = False
    classes = np.unique(target)
    if len(classes) <= 2:
        return LogisticRegressionCV(**options).fit(design, target).coef_
    return np.vstack(
        [LogisticRegressionCV(**options).fit(design, target == klass).coef_[0] for klass in classes]
    )
