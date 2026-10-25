# tecator and the baselines of chemometrics

beamfeat 0.7.0. The fat content of 240 meat samples from 100 near-infrared absorbance channels (850 to 1048 nm), on the five 75/25 splits of the paper's high-dimensional study (180 training and 60 test rows each). Mean ± standard deviation over the splits; RMSE is in percent fat. `run.py` describes each method and its tuning; `PROVENANCE.md` how to rerun. These baselines were added after the paper, which compared ridge, a random forest and LightGBM only.

| method | held-out R² | RMSE | size |
| :-- | --: | --: | :-- |
| SVR, derivative | 0.998 ± 0.001 | 0.66 ± 0.18 | all 100 channels |
| kernel ridge, derivative | 0.997 ± 0.002 | 0.79 ± 0.30 | all 100 channels |
| Gaussian process, derivative | 0.996 ± 0.002 | 0.85 ± 0.27 | all 100 channels |
| beamfeat, derivative | 0.995 ± 0.002 | 1.01 ± 0.21 | 1 to 2 terms |
| beamfeat | 0.991 ± 0.001 | 1.37 ± 0.13 | 2 to 4 terms |
| PLS, SNV | 0.971 ± 0.017 | 2.33 ± 0.58 | 5 to 18 components |
| PLS, derivative | 0.958 ± 0.014 | 2.89 ± 0.36 | 12 to 20 components |
| ridge | 0.957 ± 0.009 | 2.94 ± 0.40 | all 100 channels |
| PLS | 0.951 ± 0.016 | 3.14 ± 0.38 | 17 to 18 components |
| random forest (Table 6) | 0.717 | — | all 100 channels |
| LightGBM (Table 6) | 0.756 | — | all 100 channels |

"derivative" is a Savitzky-Golay derivative of the spectra (polynomial order 2), whose window, and for PLS whose order, was tuned; for beamfeat it is the second derivative with a window of 15.

## The formulas beamfeat printed

`beamfeat`:

- split 0: `((x35 - x42) / (x37 - x39))`, `((x41 - x55) / (x50 - x57))`, `((x39 - x42) / (x29 - x54))`
- split 1: `((x41 - x43) / (x35 - x40))`, `((x35 - x40) / (x21 - x49))`, `((x46 - x48) - (x38 - x42))`, `((x41 - x43) / (x28 - x29))`
- split 2: `((x39 - x45) / (x34 - x41))`, `((x31 - x47) / (x39 - x40))`, `((x46 - x61) / (x39 - x40))`, `((x5 - x46) / (x23 - x33))`
- split 3: `((x34 - x43) / (x37 - x38))`, `((x47 - x48) / (x34 - x65))`
- split 4: `((x32 - x46) / (x36 - x41))`, `((x36 - x41) / (x27 - x46))`, `(exp(-0.278*x48) * x40)`

Channels used: 30, at 860 to 980 nm; the most frequent: x41 (932 nm, 4 of 5 splits), x40 (930 nm, 3 of 5 splits), x46 (942 nm, 3 of 5 splits), x48 (946 nm, 3 of 5 splits).

`beamfeat, derivative`:

- split 0: `((x38 / x47) - (x25 / x48))`, `((x15 / x46) / (x56 / x48))`
- split 1: `((x25 / x57) + (x39 / x47))`, `((x15 - x27) / x32)`
- split 2: `((x39 / x47) - (x26 / x48))`, `((x29 + x76) - x9)`
- split 3: `((x39 / x47) - (x65 / x47))`, `(x18 + x77)`
- split 4: `((x38 / x47) - (x27 / x47))`

Channels used: 18, at 868 to 1004 nm; the most frequent: x47 (944 nm, 5 of 5 splits), x39 (928 nm, 3 of 5 splits), x38 (926 nm, 2 of 5 splits), x15 (880 nm, 2 of 5 splits).

## The split of the data's documentation

The documentation of the data (StatLib) divides the samples into the sets C (129), M (43), T (43), E1 (8) and E2 (17), in file order, and asks that a model be tuned on C and M, tested on T, and that a model with an element of randomness be quoted as the average of a handful of fits, not the best on T. E1 holds more fat than any training sample (54.7 to 58.5% against at most 49.1%) and E2 more protein, to test extrapolation. RMSE is in percent fat; beamfeat's figures are means over five seeds, with the seeds' E1 errors listed.

| method | T: RMSE | T: R² | E1: RMSE | E2: RMSE | size |
| :-- | --: | --: | --: | --: | :-- |
| ridge | 2.20 | 0.971 | 8.47 | 2.19 | all 100 channels |
| PLS | 2.10 | 0.974 | 8.43 | 1.76 | 13 components |
| PLS, SNV | 2.10 | 0.974 | 8.08 | 1.66 | 9 components |
| PLS, derivative | 1.98 | 0.977 | 8.22 | 2.00 | 8 components |
| kernel ridge, derivative | 0.54 | 0.998 | 2.69 | 0.42 | all 100 channels |
| SVR, derivative | 0.55 | 0.998 | 3.76 | 0.44 | all 100 channels |
| Gaussian process, derivative | 0.54 | 0.998 | 3.61 | 0.41 | all 100 channels |
| beamfeat | 1.21 | 0.991 | 4.09 (13.0, 1.8, 2.3, 1.7, 1.6) | 1.19 | 2 to 5 terms |
| beamfeat, parsimony_alpha=0.05 | 1.28 | 0.990 | 4.02 (13.0, 1.3, 2.5, 1.6, 1.6) | 1.22 | 1 to 3 terms |
| beamfeat, magnitude_slots=0 | 1.23 | 0.991 | 47.70 (11.0, 13.5, 211.0, 1.8, 1.2) | 1.00 | 3 to 6 terms |
| beamfeat, derivative | 0.62 | 0.998 | 1.79 (1.2, 2.5, 1.8, 1.6, 1.9) | 0.50 | 2 terms |

The documentation quotes, for T, 0.65 to 0.36 for three neural networks of the early 1990s on the principal components (Borggaard & Thodberg, 1992; Thodberg) and 2.78 for a linear model with 10 inputs.

### How far the formulas reach beyond the training data

For each beamfeat fit, "outside" is the largest distance by which one of its terms goes beyond its range on the training samples, in widths of that range, over the rows of each set.

| method | seed | outside on T | outside on E1 | E1: RMSE | E1: predictions |
| :-- | --: | --: | --: | --: | :-- |
| beamfeat | 0 | 0.00 | 2.09 | 13.04 | 26 to 54 |
| beamfeat | 1 | 0.00 | 0.27 | 1.84 | 53 to 60 |
| beamfeat | 2 | 0.03 | 0.53 | 2.27 | 53 to 62 |
| beamfeat | 3 | 0.00 | 0.16 | 1.72 | 54 to 60 |
| beamfeat | 4 | 0.00 | 0.19 | 1.57 | 53 to 59 |
| beamfeat, parsimony_alpha=0.05 | 0 | 0.00 | 2.09 | 13.04 | 26 to 54 |
| beamfeat, parsimony_alpha=0.05 | 1 | 0.00 | 0.15 | 1.33 | 53 to 59 |
| beamfeat, parsimony_alpha=0.05 | 2 | 0.00 | 0.53 | 2.53 | 53 to 62 |
| beamfeat, parsimony_alpha=0.05 | 3 | 0.00 | 0.16 | 1.65 | 54 to 61 |
| beamfeat, parsimony_alpha=0.05 | 4 | 0.00 | 0.19 | 1.57 | 53 to 59 |
| beamfeat, magnitude_slots=0 | 0 | 0.00 | 2.09 | 10.96 | 31 to 53 |
| beamfeat, magnitude_slots=0 | 1 | 0.00 | 3.64 | 13.50 | 56 to 84 |
| beamfeat, magnitude_slots=0 | 2 | 0.03 | 24.45 | 211.02 | -299 to 507 |
| beamfeat, magnitude_slots=0 | 3 | 0.00 | 0.16 | 1.79 | 55 to 61 |
| beamfeat, magnitude_slots=0 | 4 | 0.00 | 0.16 | 1.22 | 53 to 58 |
| beamfeat, derivative | 0 | 0.07 | 0.25 | 1.15 | 53 to 58 |
| beamfeat, derivative | 1 | 0.00 | 0.28 | 2.46 | 54 to 63 |
| beamfeat, derivative | 2 | 0.00 | 0.20 | 1.80 | 53 to 58 |
| beamfeat, derivative | 3 | 0.00 | 0.20 | 1.59 | 53 to 60 |
| beamfeat, derivative | 4 | 0.00 | 0.20 | 1.93 | 53 to 59 |

