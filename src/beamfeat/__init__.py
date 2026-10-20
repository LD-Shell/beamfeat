"""beamfeat: scalable symbolic feature construction with calibrated selection."""

from beamfeat.estimators import (
    BeamFeatClassifier,
    BeamFeatRegressor,
    BeamFeatTransformer,
    DegenerateFitWarning,
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
from beamfeat.search import BeamSearch, SearchResult, SearchTrace
from beamfeat.selection import (
    KnockoffSelector,
    PermutationSelector,
    SelectionResult,
    Selector,
    knockoff_threshold,
    make_selector,
)

__version__ = "0.5.1.dev0"

__all__ = [
    "BeamFeatClassifier", "BeamFeatRegressor", "BeamFeatTransformer",
    "DegenerateFitWarning",
    "NoDiscoveriesError",
    "NoDiscoveriesWarning",
    "BeamSearch", "CorrelationScorer", "EvaluationLog", "Evaluator",
    "ExclusionReason", "GradientBoostingScorer", "KnockoffSelector",
    "MutualInformationScorer", "Node", "NodeError", "OperatorSpec",
    "PermutationSelector", "RejectedNode", "Scorer", "SearchResult",
    "SearchTrace", "SelectionResult", "Selector", "SpearmanScorer", "UnitError",
    "UnitRegistryError", "combine",
    "is_constant", "knockoff_threshold", "leaf", "make_scorer", "make_selector",
    "transform",
]
