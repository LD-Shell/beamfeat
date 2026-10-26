"""beamfeat: scalable symbolic feature construction with calibrated selection."""

from beamfeat.estimators import (
    BeamFeatClassifier,
    BeamFeatRegressor,
    BeamFeatTransformer,
    DegenerateFitWarning,
    ExtrapolationWarning,
    NoDiscoveriesError,
    NoDiscoveriesWarning,
)
from beamfeat.expression import (
    EvaluationLog,
    Evaluator,
    ExclusionReason,
    Node,
    NodeError,
    OperatorSpec,
    RejectedNode,
    UnitError,
    UnitRegistryError,
    combine,
    leaf,
    transform,
)
from beamfeat.scoring import (
    CorrelationScorer,
    GradientBoostingScorer,
    MutualInformationScorer,
    Scorer,
    SpearmanScorer,
    is_constant,
    make_scorer,
)
from beamfeat.screen import EquationScreen, screen_equation
from beamfeat.search import BeamSearch, SearchResult, SearchTrace
from beamfeat.selection import (
    KnockoffSelector,
    PermutationSelector,
    SelectionResult,
    Selector,
    knockoff_threshold,
    make_selector,
)
from beamfeat.stability import StabilityReport, stability_report

__version__ = "0.7.0"

__all__ = [
    "BeamFeatClassifier", "BeamFeatRegressor", "BeamFeatTransformer",
    "DegenerateFitWarning",
    "ExtrapolationWarning",
    "NoDiscoveriesError",
    "NoDiscoveriesWarning",
    "BeamSearch", "CorrelationScorer", "EquationScreen", "EvaluationLog", "Evaluator",
    "ExclusionReason", "GradientBoostingScorer", "KnockoffSelector",
    "MutualInformationScorer", "Node", "NodeError", "OperatorSpec",
    "PermutationSelector", "RejectedNode", "Scorer", "SearchResult",
    "SearchTrace", "SelectionResult", "Selector", "SpearmanScorer", "StabilityReport",
    "UnitError", "UnitRegistryError", "combine",
    "is_constant", "knockoff_threshold", "leaf", "make_scorer", "make_selector",
    "screen_equation", "stability_report", "transform",
]
