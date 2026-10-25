# Architecture

beamfeat builds candidate formulas from the columns of a table with a beam
search, tests each against the null that it is unrelated to the target on rows
the search never saw, and fits a linear model on the terms it keeps. This page
maps the code onto that: the stages of a fit, the stages of the search, and
the modules. The user documentation is at
[ld-shell.github.io/beamfeat](https://ld-shell.github.io/beamfeat/); the
guarantee and its scope are on its guarantees page.

## A fit, stage by stage

```mermaid
---
config:
  flowchart:
    wrappingWidth: 600
---
flowchart TD
    fit["fit()<br/>BeamFeatRegressor, BeamFeatClassifier,<br/>BeamFeatTransformer · <i>estimators.py</i>"]
    fit --> pipe["_run_pipeline · <i>_base.py</i>"]
    pipe --> rows["_split_rows<br/>search rows · selection rows<br/>· re-test rows"]
    rows --> search["_search → BeamSearch.run<br/><i>search.py</i>"]
    search -->|"a selector (the default)"| screen["_select<br/>evaluate the candidates<br/>on the selection rows,<br/>then selector.select · <i>selection.py</i>"]
    search -->|"selector=None"| finish
    screen -->|"a null rejected"| choose["_choose_terms, within _select<br/>input test if on → forward step<br/>(<i>_forward.py</i>) → re-test<br/>if parsimony_holdout"]
    screen -->|"none rejected,<br/>or nothing testable"| nodisc["_no_discoveries, within _select<br/>on_no_discoveries:<br/>empty · fallback · raise"]
    choose --> finish["_finalise<br/>evaluate the kept terms on all rows;<br/>features_, equations_, fdr_scope_"]
    nodisc --> finish
    finish --> model["back in fit()<br/>regressor: StandardScaler + ridge<br/>classifier: StandardScaler + logistic<br/>(the mean or the class prior<br/>when no term is kept)<br/>transformer: the columns, no model"]
    model --> check["_holdout_fit_check,<br/>regressor and classifier:<br/>DegenerateFitWarning when the model<br/>is degenerate on the selection rows"]
```

- **Rows.** With a selector (the default) and `selection_holdout` (0.5 by
  default), the search sees one part of the rows and the screen the other,
  provided both parts have at least 10 rows; otherwise the screen runs on the
  search rows and `fdr_controlled_` is `False`. That split is what the FDR
  guarantee rests on: the screen tests a candidate set fixed before it looked
  at its rows. `parsimony_holdout` splits the selection rows again for the
  re-test; without the first split it is ignored.
- **Screen.** The default, `selector="permutation"`, is a
  `PermutationSelector` with the estimator's `selection_correction`,
  Benjamini-Yekutieli by default. A selector instance passed in keeps its own
  settings (a `PermutationSelector()` uses Benjamini-Hochberg), with a warning
  where they differ from the estimator's. `selection_report_` records each
  candidate's statistic, its p- and q-value when the selector computes them,
  and whether its null was rejected. The guarantee covers the screened set.
- **Choice of terms.** The screened set, less, with `input_test=True`, any
  term with a column whose null the input test fails to reject (if that would
  leave none, the screened set is kept, with a warning); then, at the default
  `parsimony="forward"`, the forward step, which adds terms while each adds at
  least `parsimony_tol` to R² (and, with `parsimony_alpha`, rejects the null
  that chance reaches its gain); then, with `parsimony_holdout`, the terms the
  re-test certifies on rows of their own. When the re-test certifies none,
  or the selection rows are too few to split for it (with a warning), the
  terms before the forward step are returned instead: the screened set, less
  any the input test removed.
- **Prediction.** `predict` and `transform` replay the kept expressions on new
  rows without searching again (`_BeamFeatBase._apply`). An expression that
  fails on some rows, the logarithm of a negative value say, is evaluated row
  so that each row gets the value it would get alone (`_prediction.py`), and
  is NaN on the rows it fails on only. `_model_output` returns the model's
  output with NaN on those rows where the output is a float (predictions,
  probabilities), and the mask of rows that are defined; the classifier's
  `predict` gives an undefined row the most frequent training class, as its
  labels cannot be NaN.

## Inside the search

```mermaid
---
config:
  flowchart:
    wrappingWidth: 600
---
flowchart TD
    zero["_depth_zero<br/>score the input columns:<br/>the beam of depth 0"]
    zero --> depth{"_run_depth<br/>for depth 1 … max_depth"}
    depth --> cand["_depth_candidates<br/>_propose: transforms and pairs<br/>from the beam and pool<br/>_fit: constants of fitted_ops"]
    cand --> score["_score_round<br/>evaluate; rank on the larger of<br/>the score against the target<br/>and against the residual"]
    score --> keep["_prune_redundant / _prune_evaluated:<br/>keep beam_width<br/>_reserve: magnitude_slots<br/>_extra_parents: more parents<br/>for the next round's fits"]
    keep --> record["trace and verbose report"]
    record --> depth
    depth -->|"last depth, nothing new,<br/>nothing evaluated,<br/>or an empty beam"| fin["rescore everything kept<br/>against the target;<br/>_finalise: remove near-duplicates,<br/>keep max_features (default beam_width);<br/>_constant_free_copies"]
```

`_SearchState` carries the beam, the pool, everything kept so far and the
trace from one depth to the next. The constants of `fitted_ops` are fitted in
`parametric.py`. `_constant_free_copies` then swaps a kept feature that has
fitted constants for a near copy with fewer, among the expressions the search
accumulated but left out of the final set, when a likelihood-ratio test fails to reject the null that
the copy fits as well; of those, the simplest. The expressions are `Node`
objects from `expression.py`, evaluated and cached by its `Evaluator`.

## Modules

```mermaid
---
config:
  flowchart:
    wrappingWidth: 600
---
flowchart LR
    %% scoring's arrows come first so that it sits top right, clear of GitHub's zoom buttons
    expression --> scoring
    search --> scoring
    _base --> scoring
    estimators --> scoring
    stability --> scoring
    estimators --> _base
    estimators --> _rendering
    estimators --> search
    estimators --> selection
    stability --> estimators
    _base --> search
    _base --> selection
    _base --> _forward
    _base --> _prediction
    _base --> _rendering
    _base --> expression
    search --> parametric
    search --> expression
    selection --> expression
    parametric --> expression
    _rendering --> expression
    _prediction --> expression
    _operators -->|"registers the<br/>built-in operators"| expression
    stability --> search
```

Arrows point from a module to one it imports. Not drawn: `__init__`, which
re-exports the public names of `estimators`, `expression`, `scoring`,
`search`, `selection` and `stability`, and `_compat`, the shim for
scikit-learn versions that `estimators`, `scoring`, `selection` and
`stability` use. `expression` imports `_operators` once, at its end, so that
the operators register in `OPERATORS` after everything they are built from
exists.

| module | what it holds |
| :-- | :-- |
| `estimators.py` | the three public estimators: their parameters, `fit`, `predict`, `transform`, and `equation` and `latex` for the regressor and the classifier |
| `_base.py` | `_BeamFeatBase`, what the three share: validation, units, the row split and the stages above, prediction on new rows (`_apply`, `_model_output`), the introspection methods (`formulas`, `to_sympy`, `latex_formulas`, `get_feature_names_out`), and the public warnings and errors |
| `_forward.py` | the forward step's path through the screened set and its chance check (`parsimony_alpha`) |
| `_rendering.py` | number and LaTeX formatting, the equation list (`equations`, for all three estimators) and the notes on what the guarantee covers |
| `_prediction.py` | the row-by-row evaluation of an expression that fails on some rows |
| `search.py` | `BeamSearch`, `SearchResult`, `SearchTrace`; the default operator sets (`DEFAULT_UNARY`, `DEFAULT_BINARY`, and `DEFAULT_FITTED`, the estimators' default; `BeamSearch` itself fits no constants unless given `fitted_ops`), `magnitude_scores`, and `report`, the verbose output |
| `parametric.py` | the constants inside `exp(a*x)`, `x^a`, `1/(x + a)` and `log(x + a)` |
| `expression.py` | `Node`, the expression DAG; `Evaluator`; units; the operator registry |
| `_operators.py` | the built-in operators: function, formula, unit rule, domain, constant grid |
| `scoring.py` | the scorers that rank candidates in the search |
| `selection.py` | the FDR-controlled selectors, the input test, and the Freedman-Lane test of the conditional re-test (`_ConditionalPermutationSelector`) |
| `stability.py` | `stability_report`: refits on many splits and counts the distinct equations |
| `_compat.py` | differences between scikit-learn versions |

## Where to make a change

| change | where |
| :-- | :-- |
| a new operator | a `register_operator(OperatorSpec(...))` call in `_operators.py`, and its sympy form in `_sympy_builders` in `expression.py` (without one, `to_sympy` and the LaTeX show it as an opaque function); a fitted operator whose constant can make it a plain operator or a constant also goes in `_PLAIN_FORMS` or `_DEGENERATE` there. To use it by default, add it to `DEFAULT_UNARY`, `DEFAULT_BINARY` or `DEFAULT_FITTED` in `search.py`. Tests in `tests/test_expression.py`; a fitted operator also in `tests/test_parametric.py`, whose `FITTED` names every one and which pins `DEFAULT_FITTED` |
| a new scorer | a `Scorer` subclass in `scoring.py`, named in `_SCORERS`; `tests/test_scoring.py` |
| a new selector | a `Selector` subclass in `selection.py`, named in `_SELECTORS`; `tests/test_selection.py` |
| how terms are chosen after the screen | `_BeamFeatBase._choose_terms` in `_base.py`, the forward step in `_forward.py`; `tests/test_parsimony.py` |
| prediction on new rows | `_apply` and `_model_output` in `_base.py`, `_prediction.py`, and each estimator's `predict` in `estimators.py`; `tests/test_estimators.py` |
| the printed equation or LaTeX | `_rendering.py`, and `equation` / `latex` in `estimators.py`; `tests/test_rendering.py` |
| the search | `search.py`: `_depth_candidates`, `_score_round`, `_prune_redundant`; `tests/test_search.py` |
| a new estimator parameter | `_BeamFeatBase.__init__` and each public class's `__init__` (scikit-learn reads a class's parameters from its own signature), `_validate_params`, and the `Args` of `BeamFeatTransformer`'s docstring, which the other two refer to |
| the documentation | `docs/`, with the pages listed in `mkdocs.yml`; the tutorials are generated by `notebooks/build_notebooks.py` (see `CONTRIBUTING.md`) |

The test files follow the code by topic: `tests/test_estimators.py` for the
estimators' API and prediction, `tests/test_screening.py` for the screen
inside a fit, `tests/test_parsimony.py` for the forward step, the re-test and
what the guarantee covers, `tests/test_rendering.py` for the renderings, and
one file each for `expression`, `parametric`, `scoring`, `search`,
`selection`, `stability` and `_compat`. ruff (`pyproject.toml`) limits the
complexity and the number of statements of a function everywhere but the
tests and the benchmark scripts, so a stage that grows is split rather than
lengthened.
