# A search on a sample of its rows (`max_search_rows`)

`max_search_rows` runs the search on that many of its rows, drawn at random,
and checks the expressions it returns on every search row; the screen still
runs on every selection row, so its guarantee is unchanged. What a sample can
cost is a weak term, ranked on fewer rows. This study measures both sides on
a tall table. The full table is in `TABLES.md`, the method in `run.py` and
`PROVENANCE.md`.

200,000 rows of ten columns on [1, 5]; the target is `x0*x1/x2 + w*x4*x5`
plus noise as large as the signal, with `w` giving the weak term 2%, 0.5% or
0.1% of the signal's variance. The default split leaves 100,000 search rows.
Five seeds per cell, every other setting at its default.

| search rows | fit time, s (median over 15 fits) | peak memory of the fit, MB | 2% term: `x4` and `x5` printed | R² against the signal, 2% term |
| --: | --: | --: | --: | --: |
| 100,000 (all) | 110 to 128 | 777 | 5 of 5 | 0.9972 |
| 30,000 | 45 to 48 | 648 | 5 of 5 | 0.9962 |
| 10,000 | 28 to 37 | 646 | 5 of 5 | 0.9953 |
| 3,000 | 24 to 35 | 648 | 5 of 5 | 0.9955 |

- **Time** falls by 2.5 to 4.5 times. The screen on 100,000 selection rows
  and the final fit on all 200,000 take the rest, about 25 seconds here.
- **Memory** falls by a sixth: its peak is set by the screen and the final
  fit, which still see every row.
- **The weak term.** At 2% of the signal it was printed at every sample size,
  each time as `x4` and `x5` terms rather than their product (on [1, 5] a
  product of two independent columns is close to a sum of functions of
  each), and R² against the signal fell by 0.001 to 0.002. At 0.5% and 0.1%
  the default search did not find it on all 100,000 rows either, so the
  sample cost nothing there (R² 0.9950 and 0.9990, one minus the term's
  share).
- **The strong term** `x0*x1/x2` was printed in every fit.

Memory is measured for the 0.5% cell, one fit per sample size, each in a
fresh process; times are of the fits of the main table, on a shared machine.
