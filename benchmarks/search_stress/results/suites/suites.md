## Main suite (24 datasets; left out, refused by the option: breast_cancer)

| | default | option |
| :-- | --: | --: |
| mean held-out score | 0.8851 | 0.8841 |
| identical fits | | 19/24 |
| formula recovered | 15/15 | 15/15 |

Fits that differ: friedman1 0.7449 → 0.7449, distractors_noise50 0.7764 → 0.7748, threshold_step 0.8432 → 0.8429, diabetes 0.3817 → 0.3703, california_housing 0.6989 → 0.6868.

## High-dimensional study (mean held-out R² over 5 splits)

| dataset | default | option | splits up / down | fit time, default → option |
| :-- | --: | --: | :-- | :-- |
| communities_p100 | 0.641 | 0.640 | 2 / 3 | 16 s → 22 s |
| superconduct_p81 | 0.731 | 0.740 | 4 / 1 | 22 s → 31 s |
| tecator_p100 | 0.991 | 0.991 | 3 / 2 | 24 s → 29 s |
| eyedata_p200 | 0.416 | 0.420 | 1 / 0 | 25 s → 30 s |
| riboflavin_p4088 | 0.340 | 0.338 | 1 / 2 | 66 s → 59 s |
| geomusic_p117 | 0.805 | 0.804 | 0 / 2 | 8 s → 16 s |
| blogfeedback_p280 | 0.415 | 0.427 | 1 / 1 | 126 s → 134 s |
| ct_slices_p384 | 0.664 | 0.664 | 0 / 1 | 154 s → 214 s |
| ujiindoorloc_p520 | 0.816 | 0.816 | 0 / 0 | 121 s → 224 s |
| mean of datasets | 0.647 | 0.649 | | 562 s → 758 s |

Peak memory of the runs timed with `/usr/bin/time`: 541 MB → 548 MB.

## Feynman panel (12 laws)

- default, default operators: 10 solved, 8 exact, 0.28 s per law
- default, with exp: 10 solved, 8 exact, 0.28 s per law
- magnitude_slots=10, default operators: 10 solved, 8 exact, 0.43 s per law
- magnitude_slots=10, with exp: 10 solved, 8 exact, 0.49 s per law
