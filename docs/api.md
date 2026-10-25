# API reference

Most code needs only the three estimators. The layers beneath them are public
as well, for running a search or a selector on its own, writing a scorer, or
building and evaluating expressions directly.

## Estimators

::: beamfeat.BeamFeatRegressor
    options:
      inherited_members: [equations, formulas, get_feature_names_out, latex_formulas, to_sympy]

::: beamfeat.BeamFeatClassifier
    options:
      inherited_members: [equations, formulas, get_feature_names_out, latex_formulas, to_sympy]

::: beamfeat.BeamFeatTransformer
    options:
      inherited_members: [equations, formulas, get_feature_names_out, latex_formulas, to_sympy]

## Warnings and errors

::: beamfeat.NoDiscoveriesWarning
::: beamfeat.NoDiscoveriesError
::: beamfeat.DegenerateFitWarning
::: beamfeat.ExtrapolationWarning

## Stability over splits

::: beamfeat.stability
    options:
      filters: public

## Search

::: beamfeat.search
    options:
      filters: public

::: beamfeat.parametric
    options:
      filters: public

## Scoring

::: beamfeat.scoring
    options:
      filters: public

## Selection

::: beamfeat.selection
    options:
      filters: public

## Expressions

::: beamfeat.expression
    options:
      filters: public
