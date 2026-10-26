# A weak term beside a dominant one, and switches at a threshold

Two targeted studies of the options 0.8.0 adds, all off by default: the
residual slots (`residual_slots`) and the switch operators (`sign`,
`sign_shift`). 0.8.0's default fits as 0.7.0's does, and gives the same
records here. The full tables are in `TABLES.md`; how they were made is in
`PROVENANCE.md`.

## A weak term: `x0*x1/x2 + 0.5*log(x3)`

The logarithm carries 0.6% of the target's variance. The default never
keeps it: on [1, 5], `log(x3)`, `sqrt(x3)` and `1/x3` correlate above the
redundancy threshold, so a depth keeps one of them, and the final ranking
then cuts the survivor below fifty restatements of `x0*x1/x2`. The residual
slots add the first terms of a forward path through the search's
candidates to that ranking, after comparing each with the other transforms
of its operand.

| rows | default (0.7.0, 0.8.0): exact law printed | `residual_slots=10` | with `parsimony_alpha=0.05` too | R² against the law, default → slots |
| --: | --: | --: | --: | :-- |
| 500 | 0 of 20 | 3 of 20 | 3 of 20 | 0.99864 → 0.99868 |
| 2,000 | 0 of 20 | 13 of 20 | 13 of 20 | 0.99868 → 0.99960 |
| 5,000 | 0 of 20 | 17 of 20 | 17 of 20 | 0.99865 → 0.99998 |

Each row pools 4 and 10 columns and noise of 0.1% and 10% of the signal's
standard deviation, five seeds each. At 500 rows the search kept `log(x3)`
in 16 of 20 fits, but the screen, on 250 selection rows, rejected its null
in 3: at a correlation of 0.08 with the target, that takes a few thousand
rows. In no cell was the mean R² against the law lower with the slots, and
`parsimony_alpha` beside them changed no fit's equation here.

The simpler settings do not do it (`alternatives.py`, the three cells with
0.1% noise and 2,000 or 5,000 rows, five seeds each, all with
`parsimony_alpha=0.05`):

| setting | exact law printed | `log(x3)` returned by the search |
| :-- | --: | --: |
| `parsimony_alpha` alone | 0 of 15 | 0 of 15 |
| `max_features=200` | 4 of 15 | 4 of 15 |
| `beam_width=200` | 0 of 15 | 0 of 15 |
| `residual_slots=10` | 14 of 15 | 15 of 15 |

A longer final list sometimes reaches the surviving transform, and then
often prints `x3` instead; a wider beam keeps more restatements of
`x0*x1/x2` and never `log(x3)`, and its fits took about five times as long
with 10 columns.

## Switches: `y = s(x0)*x2 + x1`

`s` is 2 above a threshold on `x0` and -1 below it; 1,000 rows, the
threshold at 0 or 0.7, noise of standard deviation 0.3 or 1.0, five seeds.
R² against the noiseless target on 5,000 new rows, and the fits that print
the switch term:

| | threshold 0 | threshold 0.7 | switch printed |
| :-- | --: | --: | --: |
| default (0.7.0, 0.8.0) | 0.765 to 0.767 | 0.870 to 0.875 | 0 of 20 |
| `sign` | 0.999 | 0.885 | 10 of 20 (threshold 0) |
| `sign_shift` | 0.999 | 0.997 to 0.998 | 20 of 20 |
| `sign_shift`, `residual_slots=10` | 1.000 | 1.000 | 20 of 20 |
| gradient-boosted trees | 0.987 to 0.996 | 0.983 to 0.992 | — |

The default cannot build a switch, so it fits a smooth surrogate.
`sign_shift` fits the threshold and snaps it to 0.7. The rest of the law,
`x1` and the switch's offset `0.5*x2`, is weaker than the switch term, and
without the residual slots a surrogate stands in for it. `sign` covers a
switch at zero only, which `sign_shift` also finds. Trees approximate the
step with many splits and lose a little at its edges.
