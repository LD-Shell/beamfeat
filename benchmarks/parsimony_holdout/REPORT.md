# `parsimony_holdout` and the conditional re-test: calibration

48,000 fits: twenty design cells, 200 replicates each, six parsimony
settings, two levels. `calibration.py` produced them at commit `b264d46`.
Later commits change docstrings, documentation and these records, and make
the script record the path it imported beamfeat from relative to the
repository; none changes what a fit computes. `PROVENANCE.md` says what was
fixed in advance, what was added after a smoke run (the second noise level)
and what a rerun reproduces. Every figure is a mean over replicates with its
Monte Carlo standard error, and each design is pooled over its four cells
(n = 300 and 1,000, with 5 and 20 distractors or columns) with equal weight.
The 480 fits checked against the unmemoised search all matched to the bit,
and no test-set R² was undefined.

A printed term is false in the **marginal** sense when it is built only from
columns outside `a*b/c`, and in the **conditional** sense when it uses any
column outside it, as a rider such as `(a*b/c) * x7` does. **Power** is the
share of fits printing a term equivalent to `a*b/c`; **fallback** the share
in which the re-test certified nothing and the screened set was printed.
`snr1` and `snr3` are noise with the signal's standard deviation and with a
third of it (population R² 0.5 and 0.9).

## q = 0.1

| design | setting | FDR, marginal | FDR, conditional | power | terms | fallback | held-out R² |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| null | forward | 0.009 ± 0.003 | 0.009 ± 0.003 | — | 0.0 | — | -0.003 ± 0.000 |
| null | none | 0.009 ± 0.003 | 0.009 ± 0.003 | — | 0.1 | — | -0.003 ± 0.000 |
| null | holdout_0.3 | 0.010 ± 0.004 | 0.010 ± 0.004 | — | 0.1 | 0.009 | -0.003 ± 0.000 |
| null | holdout_0.5 | 0.018 ± 0.005 | 0.018 ± 0.005 | — | 0.3 | 0.016 | -0.003 ± 0.000 |
| null | conditional_0.3 | 0.010 ± 0.004 | 0.010 ± 0.004 | — | 0.1 | 0.009 | -0.003 ± 0.000 |
| null | conditional_0.5 | 0.018 ± 0.005 | 0.018 ± 0.005 | — | 0.3 | 0.016 | -0.003 ± 0.000 |
| planted_snr1 | forward | 0.000 ± 0.000 | 0.717 ± 0.004 | 0.908 ± 0.010 | 18.4 | — | 0.483 ± 0.001 |
| planted_snr1 | none | 0.000 ± 0.000 | 0.726 ± 0.002 | 0.929 ± 0.009 | 50.0 | — | 0.484 ± 0.001 |
| planted_snr1 | holdout_0.3 | 0.000 ± 0.000 | 0.728 ± 0.003 | 0.903 ± 0.010 | 22.5 | 0.003 | 0.484 ± 0.001 |
| planted_snr1 | holdout_0.5 | 0.000 ± 0.000 | 0.733 ± 0.003 | 0.874 ± 0.011 | 27.5 | 0.000 | 0.484 ± 0.001 |
| planted_snr1 | conditional_0.3 | 0.000 ± 0.000 | 0.682 ± 0.006 | 0.912 ± 0.010 | 45.7 | 0.912 | 0.483 ± 0.001 |
| planted_snr1 | conditional_0.5 | 0.000 ± 0.000 | 0.689 ± 0.006 | 0.909 ± 0.010 | 46.2 | 0.923 | 0.482 ± 0.001 |
| planted_snr3 | forward | 0.000 ± 0.000 | 0.423 ± 0.008 | 0.996 ± 0.002 | 3.9 | — | 0.897 ± 0.000 |
| planted_snr3 | none | 0.000 ± 0.000 | 0.690 ± 0.002 | 0.996 ± 0.002 | 50.0 | — | 0.894 ± 0.000 |
| planted_snr3 | holdout_0.3 | 0.000 ± 0.000 | 0.532 ± 0.008 | 0.996 ± 0.002 | 5.6 | 0.000 | 0.897 ± 0.000 |
| planted_snr3 | holdout_0.5 | 0.000 ± 0.000 | 0.596 ± 0.007 | 0.996 ± 0.002 | 8.1 | 0.000 | 0.897 ± 0.000 |
| planted_snr3 | conditional_0.3 | 0.000 ± 0.000 | 0.187 ± 0.009 | 0.991 ± 0.003 | 12.4 | 0.231 | 0.897 ± 0.000 |
| planted_snr3 | conditional_0.5 | 0.000 ± 0.000 | 0.204 ± 0.009 | 0.993 ± 0.003 | 13.9 | 0.263 | 0.896 ± 0.001 |
| proxy_snr1 | forward | 0.000 ± 0.000 | 0.748 ± 0.004 | 0.885 ± 0.011 | 18.3 | — | 0.483 ± 0.001 |
| proxy_snr1 | none | 0.000 ± 0.000 | 0.765 ± 0.002 | 0.903 ± 0.010 | 50.0 | — | 0.484 ± 0.001 |
| proxy_snr1 | holdout_0.3 | 0.000 ± 0.000 | 0.758 ± 0.003 | 0.876 ± 0.011 | 22.9 | 0.004 | 0.483 ± 0.001 |
| proxy_snr1 | holdout_0.5 | 0.000 ± 0.000 | 0.767 ± 0.003 | 0.855 ± 0.012 | 27.7 | 0.000 | 0.483 ± 0.001 |
| proxy_snr1 | conditional_0.3 | 0.000 ± 0.000 | 0.711 ± 0.007 | 0.891 ± 0.011 | 45.7 | 0.912 | 0.483 ± 0.001 |
| proxy_snr1 | conditional_0.5 | 0.000 ± 0.000 | 0.723 ± 0.006 | 0.882 ± 0.011 | 45.9 | 0.917 | 0.482 ± 0.001 |
| proxy_snr3 | forward | 0.000 ± 0.000 | 0.435 ± 0.008 | 0.996 ± 0.002 | 3.8 | — | 0.897 ± 0.000 |
| proxy_snr3 | none | 0.000 ± 0.000 | 0.736 ± 0.001 | 0.996 ± 0.002 | 50.0 | — | 0.894 ± 0.000 |
| proxy_snr3 | holdout_0.3 | 0.000 ± 0.000 | 0.538 ± 0.008 | 0.996 ± 0.002 | 5.5 | 0.000 | 0.897 ± 0.000 |
| proxy_snr3 | holdout_0.5 | 0.000 ± 0.000 | 0.621 ± 0.007 | 0.996 ± 0.002 | 7.9 | 0.000 | 0.897 ± 0.000 |
| proxy_snr3 | conditional_0.3 | 0.000 ± 0.000 | 0.180 ± 0.010 | 0.990 ± 0.004 | 11.1 | 0.205 | 0.896 ± 0.001 |
| proxy_snr3 | conditional_0.5 | 0.000 ± 0.000 | 0.224 ± 0.010 | 0.994 ± 0.003 | 13.8 | 0.259 | 0.897 ± 0.000 |

## q = 0.2

| design | setting | FDR, marginal | FDR, conditional | power | terms | fallback | held-out R² |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| null | forward | 0.024 ± 0.005 | 0.024 ± 0.005 | — | 0.1 | — | -0.003 ± 0.000 |
| null | none | 0.024 ± 0.005 | 0.024 ± 0.005 | — | 0.3 | — | -0.003 ± 0.000 |
| null | holdout_0.3 | 0.021 ± 0.005 | 0.021 ± 0.005 | — | 0.3 | 0.016 | -0.003 ± 0.000 |
| null | holdout_0.5 | 0.029 ± 0.006 | 0.029 ± 0.006 | — | 0.4 | 0.026 | -0.003 ± 0.000 |
| null | conditional_0.3 | 0.021 ± 0.005 | 0.021 ± 0.005 | — | 0.3 | 0.019 | -0.003 ± 0.000 |
| null | conditional_0.5 | 0.029 ± 0.006 | 0.029 ± 0.006 | — | 0.3 | 0.024 | -0.003 ± 0.000 |
| planted_snr1 | forward | 0.000 ± 0.000 | 0.717 ± 0.004 | 0.908 ± 0.010 | 18.4 | — | 0.483 ± 0.001 |
| planted_snr1 | none | 0.000 ± 0.000 | 0.726 ± 0.002 | 0.929 ± 0.009 | 50.0 | — | 0.484 ± 0.001 |
| planted_snr1 | holdout_0.3 | 0.000 ± 0.000 | 0.730 ± 0.003 | 0.904 ± 0.010 | 22.8 | 0.001 | 0.484 ± 0.001 |
| planted_snr1 | holdout_0.5 | 0.000 ± 0.000 | 0.733 ± 0.003 | 0.874 ± 0.011 | 27.6 | 0.000 | 0.484 ± 0.001 |
| planted_snr1 | conditional_0.3 | 0.000 ± 0.000 | 0.668 ± 0.007 | 0.906 ± 0.010 | 44.1 | 0.879 | 0.482 ± 0.001 |
| planted_snr1 | conditional_0.5 | 0.000 ± 0.000 | 0.683 ± 0.006 | 0.896 ± 0.011 | 44.8 | 0.893 | 0.482 ± 0.001 |
| planted_snr3 | forward | 0.000 ± 0.000 | 0.423 ± 0.008 | 0.996 ± 0.002 | 3.9 | — | 0.897 ± 0.000 |
| planted_snr3 | none | 0.000 ± 0.000 | 0.690 ± 0.002 | 0.996 ± 0.002 | 50.0 | — | 0.894 ± 0.000 |
| planted_snr3 | holdout_0.3 | 0.000 ± 0.000 | 0.532 ± 0.008 | 0.996 ± 0.002 | 5.6 | 0.000 | 0.897 ± 0.000 |
| planted_snr3 | holdout_0.5 | 0.000 ± 0.000 | 0.596 ± 0.007 | 0.996 ± 0.002 | 8.1 | 0.000 | 0.897 ± 0.000 |
| planted_snr3 | conditional_0.3 | 0.000 ± 0.000 | 0.185 ± 0.009 | 0.987 ± 0.004 | 11.0 | 0.203 | 0.897 ± 0.001 |
| planted_snr3 | conditional_0.5 | 0.000 ± 0.000 | 0.207 ± 0.010 | 0.986 ± 0.004 | 12.6 | 0.235 | 0.895 ± 0.001 |
| proxy_snr1 | forward | 0.000 ± 0.000 | 0.748 ± 0.004 | 0.885 ± 0.011 | 18.3 | — | 0.483 ± 0.001 |
| proxy_snr1 | none | 0.000 ± 0.000 | 0.765 ± 0.002 | 0.903 ± 0.010 | 50.0 | — | 0.484 ± 0.001 |
| proxy_snr1 | holdout_0.3 | 0.000 ± 0.000 | 0.760 ± 0.003 | 0.876 ± 0.011 | 23.1 | 0.003 | 0.483 ± 0.001 |
| proxy_snr1 | holdout_0.5 | 0.000 ± 0.000 | 0.767 ± 0.003 | 0.857 ± 0.012 | 27.8 | 0.000 | 0.483 ± 0.001 |
| proxy_snr1 | conditional_0.3 | 0.000 ± 0.000 | 0.698 ± 0.008 | 0.878 ± 0.011 | 43.9 | 0.874 | 0.482 ± 0.001 |
| proxy_snr1 | conditional_0.5 | 0.000 ± 0.000 | 0.715 ± 0.007 | 0.870 ± 0.012 | 44.7 | 0.891 | 0.481 ± 0.001 |
| proxy_snr3 | forward | 0.000 ± 0.000 | 0.435 ± 0.008 | 0.996 ± 0.002 | 3.8 | — | 0.897 ± 0.000 |
| proxy_snr3 | none | 0.000 ± 0.000 | 0.736 ± 0.001 | 0.996 ± 0.002 | 50.0 | — | 0.894 ± 0.000 |
| proxy_snr3 | holdout_0.3 | 0.000 ± 0.000 | 0.538 ± 0.008 | 0.996 ± 0.002 | 5.5 | 0.000 | 0.897 ± 0.000 |
| proxy_snr3 | holdout_0.5 | 0.000 ± 0.000 | 0.621 ± 0.007 | 0.996 ± 0.002 | 7.9 | 0.000 | 0.897 ± 0.000 |
| proxy_snr3 | conditional_0.3 | 0.000 ± 0.000 | 0.183 ± 0.010 | 0.990 ± 0.004 | 10.2 | 0.185 | 0.896 ± 0.001 |
| proxy_snr3 | conditional_0.5 | 0.000 ± 0.000 | 0.223 ± 0.010 | 0.991 ± 0.003 | 12.0 | 0.221 | 0.896 ± 0.001 |

## What the tables show

**In the marginal sense every setting controls the rate, with room to spare,
and the re-test adds nothing measurable.** On the global null the printed
equation held a term in 7 to 14 of 800 fits per setting at q = 0.1 (realised
FDR 0.009 to 0.018) and in 17 to 23 at q = 0.2 (0.021 to 0.029). On the
planted and proxy designs no setting printed a term built only from
distractors in any of its 3,200 fits per level, a one-sided 95% bound below
0.001. The re-test did its part on the null: at q = 0.1, either test
certified nothing in 20 of the 22 fits in which the screen had picked a
false term (7 of 8 at a held-back fraction of 0.3, 13 of 14 at 0.5). The
fallback to the screened set then printed those terms anyway, so the printed
equation's error under `parsimony_holdout` is that of the screen, run on
fewer rows.

**In the conditional sense no setting controls the rate except the
conditional re-test on a short subset.** At signal to noise 1 the forward
step keeps 18 terms on average (27 at n = 300), almost all of them riders
that clear `parsimony_tol` by fitting noise, and the conditional rate is
0.67 to 0.77 under every setting at both levels. At signal to noise 3 the
default prints about four terms, with a conditional rate of 0.42 to 0.44.
The marginal re-test makes this worse, 0.53 to 0.62, paired +0.10 to +0.19
against the default: on its smaller share of rows the forward step keeps
more riders, and a rider is associated with the target, so the re-test keeps
it. The conditional re-test brings it to 0.18 to 0.22, paired −0.21 to
−0.26.

**The conditional re-test works when the subset it receives is short, and
otherwise falls back.** In the fits where it certified something it printed
1.1 to 1.2 terms on average at signal to noise 3, with a conditional rate of
0.026 to 0.040 at q = 0.1 and 0.052 to 0.073 at q = 0.2, and power 0.98 to
0.99; at n = 1,000 it fell back in at most 0.5% of fits and printed 1.1 to
1.3 terms, with rates of 0.019 to 0.033 at q = 0.1 and power 1.00. When the
forward subset is long and its riders nearly reproduce the true term, no
term adds anything given the others and the screened set is returned, which
carries the screen's marginal guarantee only: in 87 to 92% of fits at signal
to noise 1 (over 95% at n = 300) and 18 to 26% at signal to noise 3 (37 to
52% at n = 300). The conditional settings therefore print 44 to 46 terms on
average at signal to noise 1, against the default's 18.

**Power and fit hardly move.** Power is 0.85 to 0.93 at signal to noise 1
and 0.99 to 1.00 at 3 under every setting; holding back half the selection
rows costs the most (at signal to noise 1, 0.87 on the planted design
against the default's 0.91, and 0.86 on the proxy design against 0.89).
Held-out R² stays within 0.0033 of the default, paired, under every setting.

**The proxy column changes little.** A column correlated 0.9 with `a` moves
the conditional rate by −0.01 to +0.05 across settings and noise levels, and
lowers power by 0.02 to 0.03 at signal to noise 1. The conditional re-test
does about as well as on the planted design: 0.18 and 0.22 against 0.19 and
0.20 at signal to noise 3 and q = 0.1.

## Scope

One formula, one operator set and the estimator's defaults otherwise
(Benjamini-Yekutieli, `parsimony_tol = 1e-3`, depth 2, beam 50), on inputs
from a bounded positive range. The marginal-sense rate is barely stressed,
since a distractor-only term must first pass a conservative screen; the
contrast between the settings lies in the conditional sense and in the
fallback. Nothing here bears on real data or on classification, where the
conditional re-test is not available.

## Files

`results/records.csv.gz` holds one row per fit, `summary.json` the
provenance and every aggregate (per cell and pooled), `tables.md` these
tables and the per-cell ones, `examples.json` the printed formulas of the
two replicates per cell that were checked against the unmemoised search, and
`calibration.log` the run's output. The run wrote to a directory outside the
repository, shown as `<out>` in the log, from which these files were copied;
the path beamfeat was imported from is given relative to the repository.
`python benchmarks/parsimony_holdout/calibration.py --tabulate` rebuilds
`summary.json` and `tables.md` from `records.csv.gz` alone, byte for byte.
