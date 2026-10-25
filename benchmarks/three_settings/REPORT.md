# Three settings

The default, `parsimony_alpha=0.05`, `input_test=True` and both options together, fitted on the same data with the same seeds; the documentation's "Choosing a setting" page summarises these tables. beamfeat 0.7.0, numpy 2.5.4, scikit-learn 1.9.1; `PROVENANCE.md` says how to rerun.

## Noise sweep

`y = x0/x1` among 12 columns, 10 of them unrelated, with Gaussian noise at a multiple of the signal's standard deviation; 1,000 training rows and 10 seeds per level. R² is against the noiseless target on 2,000 new rows. A term reads an unrelated column when it uses any column but `x0` and `x1`.

| noise | setting | median terms | terms reading an unrelated column | median R² | median fit (s) |
| --: | :-- | --: | --: | --: | --: |
| 0.1 | default | 1 | 0 of 10 | 1.000 | 0.7 |
| 0.1 | `parsimony_alpha=0.05` | 1 | 0 of 10 | 1.000 | 0.7 |
| 0.1 | `input_test=True` | 1 | 0 of 10 | 1.000 | 5.1 |
| 0.1 | both | 1 | 0 of 10 | 1.000 | 5.4 |
| 0.3 | default | 1 | 5 of 15 | 1.000 | 0.8 |
| 0.3 | `parsimony_alpha=0.05` | 1 | 0 of 10 | 1.000 | 0.8 |
| 0.3 | `input_test=True` | 1 | 0 of 10 | 1.000 | 5.7 |
| 0.3 | both | 1 | 0 of 10 | 1.000 | 5.5 |
| 0.5 | default | 4 | 25 of 36 | 0.997 | 0.8 |
| 0.5 | `parsimony_alpha=0.05` | 1 | 0 of 10 | 0.999 | 0.8 |
| 0.5 | `input_test=True` | 1 | 2 of 16 | 0.999 | 5.4 |
| 0.5 | both | 1 | 0 of 10 | 0.999 | 5.1 |
| 1.0 | default | 9.5 | 78 of 92 | 0.981 | 0.8 |
| 1.0 | `parsimony_alpha=0.05` | 1 | 0 of 10 | 0.998 | 0.8 |
| 1.0 | `input_test=True` | 2 | 3 of 21 | 0.995 | 5.1 |
| 1.0 | both | 1 | 0 of 10 | 0.998 | 5.3 |

## Real datasets

The 12 real datasets of `benchmarks/release_comparison/`, with its five 70/30 splits and seeds: mean held-out score (R², accuracy for breast_cancer), with the mean number of terms in brackets. Better or worse means a change of more than 0.005 from the default.

| dataset | default | `parsimony_alpha=0.05` | `input_test=True` | both |
| :-- | --: | --: | --: | --: |
| diabetes | 0.433 (6.6) | 0.381 (1.4) | 0.425 (6.6) | 0.386 (1.4) |
| breast_cancer | 0.971 (25.8) | 0.952 (2.8) | 0.963 (19.4) | 0.950 (2.4) |
| mpg | 0.868 (9.4) | 0.858 (1.6) | 0.866 (6.4) | 0.862 (2.0) |
| diamonds | 0.852 (3.4) | 0.851 (1.0) | 0.825 (1.8) | 0.824 (1.4) |
| penguins | 0.803 (7.8) | 0.768 (1.4) | 0.787 (5.0) | 0.771 (1.6) |
| tips | 0.472 (8.6) | 0.433 (1.0) | 0.411 (4.2) | 0.361 (1.0) |
| california_housing | 0.690 (21.2) | 0.681 (13.4) | 0.688 (16.2) | 0.665 (8.4) |
| communities | 0.647 (11.6) | 0.629 (2.4) | 0.647 (11.6) | 0.629 (2.4) |
| superconduct | 0.747 (18.8) | 0.737 (12.6) | 0.747 (18.8) | 0.737 (12.6) |
| tecator | 0.992 (3.0) | 0.991 (2.2) | 0.991 (3.0) | 0.991 (2.2) |
| eyedata | 0.321 (30.6) | 0.330 (1.0) | 0.284 (23.8) | 0.366 (1.0) |
| riboflavin | 0.516 (18.6) | 0.489 (1.0) | 0.472 (9.0) | 0.464 (1.0) |
| **mean** | **0.693** (13.8) | **0.675** (3.5) | **0.675** (10.5) | **0.667** (3.1) |
| better / same / worse | — | 1 / 2 / 9 | 0 / 5 / 7 | 1 / 1 / 10 |
| fit time (× default) | 1 | 1.0 | 3.6 | 3.6 |

Fit time is the geometric mean, over the 60 fits, of each fit's time over the default's on the same split. The settings ran in parallel processes, one per setting, on one core each.

## Synthetic datasets

The 18 synthetic datasets of the same suite. On 15 of them no setting moved the mean held-out score by more than 0.005 from the default. The others:

| dataset | default | `parsimony_alpha=0.05` | `input_test=True` | both |
| :-- | --: | --: | --: | --: |
| distractors_noise50 | 0.808 (11.0) | 0.815 (1.2) | 0.813 (4.8) | 0.805 (1.4) |
| friedman1 | 0.796 (11.8) | 0.759 (3.4) | 0.800 (11.4) | 0.759 (3.4) |
| threshold_step | 0.841 (12.0) | 0.808 (1.0) | 0.814 (2.4) | 0.808 (1.0) |

## At the paper's own settings

Each study of the paper rerun with its own script and settings (`run.py` lists them). The paper column gives the paper's figures, or, per dataset, the archived results its tables quote (`benchmarks/results*.csv`, `feynman_results.json`).

| claim | paper | default | `parsimony_alpha=0.05` | `input_test=True` | both |
| :-- | :-- | :-- | :-- | :-- | :-- |
| generating columns recovered (Table 4) | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| mean R², core suite (Table 4) | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 0.9990 |
| stress sets with false features (Table 4) | 0 of 5 | 0 of 5 | 0 of 5 | 0 of 5 | 0 of 5 |
| Feynman panel solved, exact (§4.3) | 10/12, 8/12 | 10/12, 8/12 | 10/12, 8/12 | 10/12, 8/12 | 10/12, 8/12 |

Real datasets of `run_benchmarks.py` (§4.4; one split):

| dataset | paper | default | `parsimony_alpha=0.05` | `input_test=True` | both |
| :-- | --: | --: | --: | --: | --: |
| diabetes | 0.376 (5) | 0.370 (5) | 0.297 (1) | 0.375 (5) | 0.297 (1) |
| breast_cancer | 0.965 (11) | 0.959 (14) | 0.942 (3) | 0.982 (16) | 0.942 (3) |
| mpg | 0.864 (17) | 0.855 (23) | 0.860 (2) | 0.863 (4) | 0.860 (2) |
| diamonds | 0.871 (1) | 0.880 (2) | 0.881 (1) | 0.881 (1) | 0.881 (1) |
| penguins | 0.744 (7) | 0.760 (7) | 0.714 (2) | 0.726 (3) | 0.726 (3) |
| tips | 0.496 (15) | 0.472 (15) | 0.470 (1) | 0.407 (5) | 0.357 (1) |
| california_housing | 0.700 (15) | 0.663 (11) | 0.660 (10) | 0.673 (10) | 0.657 (6) |
| **mean** | **0.717** | **0.709** | **0.689** | **0.701** | **0.674** |

The comparison study (Table 5; 9 datasets, 5 splits):

| | paper | default | `parsimony_alpha=0.05` | `input_test=True` | both |
| :-- | --: | --: | --: | --: | --: |
| mean R² | 0.803 | 0.803 | 0.784 | 0.803 | 0.786 |
| worst of 45 | 0.355 | 0.302 | 0.284 | 0.295 | 0.282 |
| fits below zero | 0 | 0 | 0 | 0 | 0 |
| features | 9.2 | 9.2 | 2.2 | 7.9 | 2.4 |

The high-dimensional study (Table 6; the five datasets the repository stores, 5 splits):

| dataset | paper | default | `parsimony_alpha=0.05` | `input_test=True` | both |
| :-- | --: | --: | --: | --: | --: |
| communities | 0.641 (11.6) | 0.644 (13.2) | 0.627 (2.6) | 0.644 (12.6) | 0.629 (3.0) |
| superconduct | 0.731 (16.2) | 0.741 (20.0) | 0.732 (13.0) | 0.741 (20.0) | 0.732 (13.0) |
| tecator | 0.991 (3.6) | 0.990 (3.2) | 0.990 (2.8) | 0.989 (4.8) | 0.988 (3.0) |
| eyedata | 0.416 (34.0) | 0.418 (32.2) | 0.455 (1.2) | 0.447 (13.6) | 0.424 (1.2) |
| riboflavin | 0.544* (17.4) | 0.389 (20.4) | 0.196 (1.2) | 0.354 (10.2) | 0.196 (1.2) |

\* 0.340 with the column-order fix of 0.5.0 (docs, Benchmarks, Reading the paper today).

## Against the release comparison

- default: 150 of 150 fits give the held-out score and the number of terms of the release comparison's 0.7.0 run on the same split.
- `parsimony_alpha=0.05`: 150 of 150 fits give the held-out score and the number of terms of the release comparison's 0.7.0 run on the same split.

