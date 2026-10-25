# User guide

The guide explains each part of a fit with examples you can run; the outputs
shown are from real runs. Each page runs on its own: its first code block sets
up the data the rest of the page uses. New to beamfeat? Start with the
[quickstart](../quickstart.md) and [choosing a setting](../settings.md).

| page | what it covers |
| :-- | :-- |
| [Reading a fit](reading-a-fit.md) | what the printed equation and its note mean, the equation at every size, the selection report, refits on other splits, an empty fit, LaTeX and export |
| [The printed equation](printed-equation.md) | how the equation is chosen from the certified set, and the options that shorten it, test the columns inside it or re-test it |
| [Search options](search.md) | units, fitted constants, laws whose parts the score cannot see, limits on the search, speed and memory |
| [Data and prediction](data.md) | constant columns, missing values and categoricals, rows outside a formula's domain, predicting with the original columns as well, new rows beyond the training data |

## Find a parameter

| parameter, method or attribute | where |
| :-- | :-- |
| `parsimony_alpha` | [A short equation](printed-equation.md#a-short-equation-parsimony_alpha) |
| `input_test`, `input_report_` | [Every column tested](printed-equation.md#every-column-tested-input_test) |
| `parsimony`, `parsimony_tol` | [The printed equation](printed-equation.md) |
| `parsimony_holdout`, `certify_test` | [Re-testing the printed terms](printed-equation.md#re-testing-the-printed-terms-parsimony_holdout) |
| `fdr_controlled_`, `fdr_scope_` | [Quickstart](../quickstart.md#sixty-seconds-to-a-vetted-equation) |
| `fdp_inflation_` | [Why the equation says "1 of 25"](reading-a-fit.md#why-the-equation-says-1-of-25) |
| `equations()`, `equations_` | [The equation at every size](reading-a-fit.md#the-equation-at-every-size) |
| `selection_report_` | [Audit what selection did](reading-a-fit.md#audit-what-selection-did) |
| `stability_report` | [One equation or one of many](reading-a-fit.md#one-equation-or-one-of-many) |
| `on_no_discoveries` | [When no null is rejected](reading-a-fit.md#when-no-null-is-rejected) |
| `latex()`, `latex_formulas()` | [LaTeX](reading-a-fit.md#latex) |
| `to_sympy()` | [Export to NumPy, JAX or PyTorch](reading-a-fit.md#export-to-numpy-jax-or-pytorch) |
| `units` | [Dimensional analysis](search.md#dimensional-analysis) |
| `fitted_ops` | [Constants inside expressions](search.md#constants-inside-expressions) |
| `magnitude_slots` | [Laws whose parts the score cannot see](search.md#laws-whose-parts-the-score-cannot-see) |
| `max_inputs` | [Search constraints](search.md#search-constraints) |
| `n_jobs` | [Speed and memory on wide data](search.md#speed-and-memory-on-wide-data) |
| `feature_ranges_`, `ExtrapolationWarning` | [New rows beyond the training data](data.md#new-rows-beyond-the-training-data) |

`max_depth`, `beam_width` and the scorers are covered in the
[Search and scoring](../notebooks/02_search_and_scoring.ipynb) tutorial, and
`target_fdr`, the selectors and their corrections in
[Selection and units](../notebooks/03_selection_and_units.ipynb) and on the
[guarantees page](../guarantees.md). The [API reference](../api.md) lists
every parameter.
