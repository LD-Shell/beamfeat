# What the parsimony default costs

`parsimony="forward"` is the default: the fitted equation is a compact subset
of the FDR-screened set, chosen by greedy forward selection on the selection
rows. It buys readability and gives up the guarantee over the printed terms.

The documentation and the paper both assert that exchange. This file measures
it, and records which studies depend on the setting at all.
`benchmarks/parsimony_cost.py` produces the numbers; the records are in
`benchmarks/parsimony_cost/`.

## The measurement

308 paired fits, each study run once at each setting on identical data with
identical seeds. Deltas are default minus `parsimony=None`, so a negative
number means the compact equation fits worse.

| study | fits | mean ΔR² | worst ΔR² | best ΔR² | terms, None ÷ default | `fdr_controlled_` disagreements |
|---|---|---|---|---|---|---|
| end-to-end calibration (200 signal + 60 null) | 260 | −0.0001 | −0.0003 | +0.0000 | 25.0× | 0 |
| known-formula + stress suites (18 datasets) | 18 | −0.0008 | −0.0148 | +0.0037 | 26.0× | 0 |
| Feynman physics panel (12 laws × 2 configs) | 24 | −0.0009 | −0.0111 | +0.0000 | 30.6× | 0 |
| Friedman #1 decomposition (6 draws) | 6 | −0.0095 | −0.0192 | +0.0019 | 3.6× | 0 |

Compactness costs about a thousandth of R² on average and at most
0.019 on a single fit, and buys an equation roughly twenty-five times shorter.
R² is held out in every study except end-to-end calibration, which fits and
scores on the same 500 rows, so its row is an in-sample comparison.
`fdr_controlled_` never disagreed between the settings on any of the 308 fits.

Two results worth naming separately.

**The end-to-end calibration figures are identical at both settings**, to every
digit reported: empirical FDR 0.0000, 95% upper bound 0.0149, power 1.000,
fallbacks 0, null selections 0. On this design all 25 screened formulas touch a
signal column, so widening the reported set from 1 to 25 adds no false
discovery. That is a fact about the design, not a property of the procedure,
and it should not be generalised.

**Pruning was never observed to inflate the realised false discovery
proportion.** This is an empirical observation, not a theorem: |S|/|S'| remains
the only available bound and it is weak. The mechanism behind the observation
is that greedy forward selection ranks by predictive contribution, and a
marginally null feature contributes nothing, so it tends to be dropped first.
On a selector-level stress designed to force false discoveries through
(`selector_calibration.gaussian_design`, nominal q = 0.5, 300 trials), the
screened set realised FDP 0.397 while the greedily pruned subset realised
0.000, and the proportion rose in none of the 300 trials.

## The heavy studies, measured

`parsimony_cost.py` covers the four studies that finish in minutes. The four
that take hours were left unmeasured when it was written. They have since been
run at `parsimony=None` on the maintainer's hardware, and those records are in
`parsimony_cost/parsimony_none/` beside the others: `depth_ladder.json`,
`scalability.json`, `split_stability.json`, `highdim.json`, and the two
`independent_*.json` files from the 360-fit study.

They pair with the shipped archives, which are the same studies at the default:
`additional_experiments/results/` and `independent/results_as_reported/`.

| study | quantity | default | `parsimony=None` |
|---|---|---|---|
| high-dimensional panel (9 datasets) | mean features | 16.1 | 48.2 |
| high-dimensional panel | mean held-out R² | 0.671 | 0.675 |
| high-dimensional panel | tecator features | 3.6 | 50.0 |
| independent study, synthetic | mean `n_new` | 4.08 | 49.96 |
| independent study, synthetic | mean held-out R² | 0.9483 | 0.9488 |
| split stability, `three_way` | features / Jaccard | 1.0 / 1.000 | 50.0 / 0.454 |
| scalability, every width | recovery / false features | 1.00 / 0.00 | 1.00 / 0.00 |
| scalability, p = 1000 | peak MB | 2614 | 2616 |

The pattern of the light studies holds on the heavy ones: **held-out fit barely
moves and set size moves by an order of magnitude.** Every compactness figure
the report quotes is a default-setting figure and none of them survives the
switch; every accuracy figure does.

Three observations that only the heavy studies could give.

**Scalability reproduces in full.** Recovery stays 1.00, false features stay
0.00, peak memory matches to a few MB and the FDR flag holds in all 100 cells.
The table in the report quotes none of the quantities that move, so §4 is
setting-independent as printed, even though the underlying kept set is not
(1 feature at the default against 50 here, at p = 10). The earlier note in this
file that "false 0 at every width would need re-measuring" is now discharged:
it holds at both settings.

**Split stability changes character, not just scale.** `three_way` is the
report's clean case — one feature, recurring in 30 of 30 splits, Jaccard 1.000.
At `parsimony=None` the kept set is the beam cap in every split and Jaccard
falls to 0.454. The sentence "predictive performance is stable; the selected
set is not" is a claim about the pruned set and does not transfer.

**Two fits exceeded the budget.** `ujiindoorloc` splits 1 and 3 raised
`BudgetExceeded` at 900 s, so its `None` mean is over three splits, not five.
The report states that no fit exceeded the budget at the default. Whether the
cause is the larger kept set or slower hardware is not separable from this
record alone.

One confound is worth stating plainly rather than leaving for a reader to find.
The `None` arm here and the default arm in the archives were produced in
**different environments, months apart**. Large structural differences — set
sizes at the beam cap against single-figure counts — are safely attributable to
the setting. Small per-cell differences are not: on the depth ladder, 13 of 60
recovery cells differ from the archived table and they differ in *both*
directions, which the nesting argument below does not predict. A same-machine
paired run is what would separate those two causes, and it has not been done.

These records exist because the harnesses were run without the parsimony flag
at a point when omitting it selected `None` rather than the library default.
That defect is fixed — omitting the flag now reproduces the shipped default
everywhere — and the run it produced is kept here rather than discarded,
because it is the arm that was missing.

## The criterion that only holds at the default

`feynman_panel._exact_form` requires the kept formulas to match the expected
terms **as a multiset**: `if len(fitted) != len(expected_terms): return False`.
That is a statement about the *printed equation*, which is what the paper's
exact-form claim is about, so it means what it says at the default.

At `parsimony=None` the printed equation is the whole screened set, forty terms
against one expected, and the test returns `False` for every law whether or not
the law was recovered — 8/12 exact form becomes 0/12. It is not measuring
capability at that setting, it is measuring set size.

`parsimony_cost.py` therefore records `contains_exact_form` separately: whether
*any* kept formula is proportional to the law. That figure is **16/18 at both
settings**, which is how the record shows the search finds the law either way.
It is a weaker claim than exact form and must not be reported as if it were
the same criterion.

## Which studies depend on the setting

Parsimony acts in exactly one place, `_BeamFeatBase._parsimonious_subset`, and
only when screening returned something:

```python
if selection_result.n_selected > 0:
    keep_order = self._parsimonious_subset(...)
```

Three consequences follow, and every determination below is one of them.

1. **A study that never constructs a `BeamFeat*` estimator is unaffected.**
   `PermutationSelector`, `KnockoffSelector` and `BeamSearch` used directly do
   not run the step at all.
2. **A study reporting a quantity computed from the *kept* set depends on the
   setting**: the returned formulas, their count, the held-out score of the
   model fitted on them, `fdp_inflation_`, and anything derived from those.
   The kept sets nest — the default returns a subset of what `parsimony=None`
   returns, from identical screening on identical rows — so counts can only
   fall and recovery can only hold or worsen relative to `None`.
3. **A study reporting a quantity fixed upstream of the step does not**: the
   split, the search, `selection_result_`, `selection_report_`'s `screened`
   column, p- and q-values, and whether screening selected anything at all.
   `fdr_controlled_` is unaffected too: it is `bool(nodes)` over a set
   parsimony never empties, because the greedy pass always keeps at least one
   feature.

### Depends on the setting

| study | what moves | why |
|---|---|---|
| `run_benchmarks.py` (core, robustness, real) | `n_features`, `complexity`, `r2`, `false_feature_rate`, `recovered` | `run_beamfeat` fits a `BeamFeatRegressor`/`BeamFeatClassifier`; every column is read off the kept set. `false_feature_rate` is a *fraction of returned formulas*, so set size changes it even when no false formula is added |
| `feynman_panel.py` | `solved`, `exact_form`, `top_formula`, `seconds` | fits a `BeamFeatRegressor`; see the criterion note above |
| `calibration_study.py` | in principle `empirical_fdr`, `power`, `null_selections` | FDP is `#{kept formulas touching no relevant column} / #{kept}`, both over the kept set. Measured: identical at both settings on this design |
| `friedman_decomposition.py` (partly) | `achieved`, and therefore "lost to the search" | `achieved` fits a `BeamFeatRegressor`. `oracle` and `ceiling` are least-squares fits on a fixed basis and `admitted` comes from `PermutationSelector` used directly, so "lost to marginal screening" does not move. The script asserts `achieved < ceiling < oracle`; `parsimony=None` raises `achieved`, so that assertion sits closer to its bound there |
| `additional_experiments/scalability.py` | `n_selected`, `false_features`, `recovered`, `seconds` | fits a `BeamFeatTransformer`; all four are read off the kept set. Measured at p = 10: 1 feature at the default against 50 at `parsimony=None`. Both arms are now on record and the reported quantities — recovery, false features, memory, the FDR flag — are identical at the two settings; only `n_selected`, which the report does not print, moves |
| `additional_experiments/depth_ladder.py` | `n_selected`, `recovered_all`, `recovered_any`, `seconds` | `equivalent()` scores `model.transform(probe)`, the kept features. The kept sets nest, so recovery at `None` should dominate the default; measured against the archived table it does not, differing in both directions on 13 of 60 cells. The two arms come from different environments, so that comparison cannot separate the setting from the environment |
| `additional_experiments/split_stability.py` | `r2_*`, `n_selected`, `jaccard_*`, `n_classes`, `frequency_table` | fits a `BeamFeatTransformer` per split; Jaccard overlap is computed between kept sets, and set size changes it directly |
| `additional_experiments/bench.py` (`highdim`) | `r2`, `n_new` for the `beamfeat` and `beamfeat_ridge` rows | both wrappers fit a `BeamFeat*` estimator; the "mean count of certified features" column is the kept count |
| `independent/bench.py` (the 360-fit study) | `r2`, `n_new`, `recovered` for the two `beamfeat` rows | same wrappers. The other six methods are untouched, so two of eight rows depend on the setting |
| the consumer ablation (`run_all_clean.ipynb`, cell 12) | `k`, and the `bf_*` columns | fits a `BeamFeatTransformer` and feeds its kept features to three consumers |

### Does not

| study | why |
|---|---|
| `selector_calibration.py`, studies 1 and 2 | `PermutationSelector` and `KnockoffSelector` are called directly. No estimator, no parsimony step |
| `selector_calibration.py`, study 3 (global-null stress) | fits a `BeamFeatRegressor`, but the recorded quantity is `selected_any` — trials with `fdr_controlled_ and n_features_out_ > 0`. Parsimony runs only when screening selected something and always keeps at least one feature, so it can turn neither a zero into a nonzero nor the reverse. The measured BH 6/100 against BY 1/100 is invariant |
| `additional_experiments/selector_comparison.py` | selectors called directly on candidate matrices; the `beam` regime uses `BeamSearch` directly, upstream of the step |
| The label-permuted controls (`run_all_clean.ipynb`, cell 10) | the reported quantity is "five permuted replicates of riboflavin selected zero features each". Under a permuted label nothing passes screening, so the parsimony branch is never entered, and parsimony cannot create a feature from an empty set |
| Every p-value, q-value and `screened` flag in `selection_report_` | computed by the selector before the step runs |
| Dataset characterisation and the cost profile | neither fits the estimator for the reported numbers |
| The search/selection split-size sweep | affected only through the fitted model, and both arms move together, so the claim is a difference that survives |
| Anything in `independent/` other than the two `beamfeat` rows | `autofeat`, `featuretools`, `OpenFE` and the three raw baselines never touch the estimator |

## Archives that must not be overwritten

- `independent/results_as_reported/` — the published 360-fit study.
- `additional_experiments/results/` — the records the revision's report and
  figures are built from, along with `additional_experiments/figures/`.
- `independent/results_fresh_run/` — already holds a regeneration.

The second of these is tracked and has been overwritten in place once, by a
run at the wrong setting. `git checkout` recovered it. Anything that is not
the shipped default belongs under `parsimony_cost/`, which is what that
directory is for.

`parsimony_cost.py` writes only under `benchmarks/parsimony_cost/`, which it
creates. It reads nothing from the archives and writes nothing to them.

## Running it

```bash
python benchmarks/parsimony_cost.py --dry-run     # estimates only, runs nothing
python benchmarks/parsimony_cost.py               # the four studies above
python benchmarks/parsimony_cost.py --heavy       # adds the three long ones
```

Wall time per setting, measured on one core; the runner prints its own
estimates before doing anything.

| study | per setting | both settings |
|---|---|---|
| `feynman_panel` (two operator configurations) | ~25 s | ~50 s |
| `friedman_decomposition` (6 draws) | ~10 s | ~20 s |
| `calibration_study` (200 signal + 60 null) | ~100 s | ~3.5 min |
| `run_benchmarks` core + robustness (18 datasets) | ~20 s | ~40 s |
| `run_benchmarks` real (8 datasets, needs network on first run) | ~10 min | ~20 min |
| `scalability` (100 cells to p = 1000) | ~30 min | ~1 h |
| `depth_ladder` (15 targets × 2 scorers × 4 beams × 20 seeds) | ~45–90 min | ~1.5–3 h |
| `split_stability` (30 splits × 12 datasets) | ~30–60 min | ~1–2 h |

The two `bench.py` harnesses are not driven by the runner: each needs its own
environment and its own data, and both already checkpoint per dataset. Run them
directly with `BEAMFEAT_PARSIMONY`, writing to a new directory:

```bash
cd benchmarks/additional_experiments
BEAMFEAT_PARSIMONY=none      python bench.py highdim beamfeat,beamfeat_ridge 5 \
    ../parsimony_cost/parsimony_none/highdim_beamfeat.json
BEAMFEAT_PARSIMONY=forward   python bench.py highdim beamfeat,beamfeat_ridge 5 \
    ../parsimony_cost/parsimony_forward/highdim_beamfeat.json
```

Unset, `BEAMFEAT_PARSIMONY` gives the library default, `forward`; `none` keeps
the whole screened set. The value in force is recorded on every row as
`parsimony`, and the same convention holds for the `--parsimony` flag on the
scripts above: omitting it reproduces the shipped default, which is what every
archived result in this repository was produced at.
