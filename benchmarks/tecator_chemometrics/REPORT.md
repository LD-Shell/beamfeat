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

