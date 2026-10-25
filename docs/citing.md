---
description: "How to cite beamfeat: the TMLR paper, the Zenodo software archive, and the statistical methods behind each option, with BibTeX."
---

# Citing beamfeat

If you use beamfeat in published work, please cite:

1. [**The paper**](#the-paper), which describes the method. Always cite it.
2. [**The software release**](#the-software-release), when your results
   depend on a particular version.
3. [**The statistical methods**](#the-statistical-methods) behind the options
   you used, which beamfeat implements but did not invent.

## The paper

Daramola, O. M. (2026). Symbolic feature construction with false discovery
rate control. *Transactions on Machine Learning Research*.
<https://openreview.net/forum?id=UibXoy4h3u>

```bibtex
@article{daramola2026symbolic,
  title   = {Symbolic feature construction with false discovery rate control},
  author  = {Daramola, Olanrewaju M.},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=UibXoy4h3u}
}
```

The paper describes version 0.4.2. Later releases correct two of its results:
a harness bug understated featuretools, and fixing a search bug lowers the
number of datasets where beamfeat leads both tree ensembles from five to
four. [Reading the paper today](benchmarks.md#reading-the-paper-today) has
the details.

## The software release

Each release is archived on Zenodo with its own DOI:

| what to cite | DOI |
| :-- | :-- |
| beamfeat in general (resolves to the latest release) | [10.5281/zenodo.21619567](https://doi.org/10.5281/zenodo.21619567) |
| 0.4.2, the version the paper describes, with the run records behind its results | [10.5281/zenodo.23417132](https://doi.org/10.5281/zenodo.23417132) |

To cite the release you used, open the first DOI, choose that version
from the list of versions on the Zenodo page, and use its DOI. The page
also gives the citation in several formats.

```bibtex
@software{daramola_beamfeat,
  author    = {Daramola, Olanrewaju M.},
  title     = {beamfeat: beam-search feature construction with FDR-controlled selection},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21619567},
  url       = {https://doi.org/10.5281/zenodo.21619567}
}
```

For one release, add `version = {...}` and replace the DOI with that
release's. [`CITATION.cff`](https://github.com/LD-Shell/beamfeat/blob/main/CITATION.cff)
in the repository has the same details in machine-readable form, and
GitHub's "Cite this repository" button reads it.

## The statistical methods

beamfeat's guarantee rests on established procedures. If your work relies on
one of them, please cite its source as well:

| if you use | beamfeat relies on | please also cite |
| :-- | :-- | :-- |
| any fit with the default selector | permutation p-values with the add-one estimator, and the Benjamini–Yekutieli correction | [Phipson & Smyth (2010)](#ref-phipson2010); [Benjamini & Yekutieli (2001)](#ref-benjamini2001) |
| `selection_correction="bh"` | the Benjamini–Hochberg correction, in place of Benjamini–Yekutieli | [Benjamini & Hochberg (1995)](#ref-benjamini1995) |
| `selector="knockoff"` or `KnockoffSelector` | fixed-X knockoffs with knockoff+ when there are more than twice as many rows as columns, model-X Gaussian knockoffs otherwise | [Barber & Candès (2015)](#ref-barber2015); [Candès et al. (2018)](#ref-candes2018) |
| `KnockoffSelector(s_method="ci")` | the conditional-independence knockoff | [Liu & Rigollet (2019)](#ref-liu2019) |
| `parsimony_alpha` | the multiplier bootstrap for the best fit noise can reach | [Fan & Zhou (2016)](#ref-fan2016) |
| `input_test=True` | sequential Monte Carlo p-values | [Besag & Clifford (1991)](#ref-besag1991) |
| `parsimony_holdout` with `certify_test="conditional"` | the Freedman–Lane permutation test | [Freedman & Lane (1983)](#ref-freedman1983); [Anderson & Robinson (2001)](#ref-anderson2001); [Winkler et al. (2014)](#ref-winkler2014) |

[Statistical guarantees](guarantees.md) explains what each procedure
guarantees in beamfeat, and under which assumptions.

### References

- <a id="ref-anderson2001"></a>Anderson, M. J., & Robinson, J. (2001).
  Permutation tests for linear models. *Australian & New Zealand Journal of
  Statistics*, 43(1), 75–88.
  <https://doi.org/10.1111/1467-842X.00156>
- <a id="ref-barber2015"></a>Barber, R. F., & Candès, E. J. (2015).
  Controlling the false discovery rate via knockoffs. *The Annals of
  Statistics*, 43(5), 2055–2085.
  <https://doi.org/10.1214/15-AOS1337>
- <a id="ref-benjamini1995"></a>Benjamini, Y., & Hochberg, Y. (1995).
  Controlling the false discovery rate: a practical and powerful approach to
  multiple testing. *Journal of the Royal Statistical Society: Series B
  (Methodological)*, 57(1), 289–300.
  <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>
- <a id="ref-benjamini2001"></a>Benjamini, Y., & Yekutieli, D. (2001). The
  control of the false discovery rate in multiple testing under dependency.
  *The Annals of Statistics*, 29(4), 1165–1188.
  <https://doi.org/10.1214/aos/1013699998>
- <a id="ref-besag1991"></a>Besag, J., & Clifford, P. (1991). Sequential
  Monte Carlo p-values. *Biometrika*, 78(2), 301–304.
  <https://doi.org/10.1093/biomet/78.2.301>
- <a id="ref-candes2018"></a>Candès, E., Fan, Y., Janson, L., & Lv, J.
  (2018). Panning for gold: 'model-X' knockoffs for high dimensional
  controlled variable selection. *Journal of the Royal Statistical Society:
  Series B (Statistical Methodology)*, 80(3), 551–577.
  <https://doi.org/10.1111/rssb.12265>
- <a id="ref-fan2016"></a>Fan, J., & Zhou, W.-X. (2016). Guarding against
  spurious discoveries in high dimensions. *Journal of Machine Learning
  Research*, 17(203), 1–34.
  <https://jmlr.org/papers/v17/16-068.html>
- <a id="ref-freedman1983"></a>Freedman, D., & Lane, D. (1983). A
  nonstochastic interpretation of reported significance levels. *Journal of
  Business & Economic Statistics*, 1(4), 292–298.
  <https://doi.org/10.1080/07350015.1983.10509354>
- <a id="ref-liu2019"></a>Liu, J., & Rigollet, P. (2019). Power analysis of
  knockoff filters for correlated designs. *Advances in Neural Information
  Processing Systems*, 32.
  <https://proceedings.neurips.cc/paper_files/paper/2019/hash/09ab23b6b607496f095feed7aaa1259b-Abstract.html>
- <a id="ref-phipson2010"></a>Phipson, B., & Smyth, G. K. (2010).
  Permutation p-values should never be zero: calculating exact p-values when
  permutations are randomly drawn. *Statistical Applications in Genetics and
  Molecular Biology*, 9(1), Article 39.
  <https://doi.org/10.2202/1544-6115.1585>
- <a id="ref-winkler2014"></a>Winkler, A. M., Ridgway, G. R., Webster, M. A.,
  Smith, S. M., & Nichols, T. E. (2014). Permutation inference for the
  general linear model. *NeuroImage*, 92, 381–397.
  <https://doi.org/10.1016/j.neuroimage.2014.01.060>

??? note "BibTeX for the paper and these references"

    ```bibtex
    @article{daramola2026symbolic,
      title   = {Symbolic feature construction with false discovery rate control},
      author  = {Daramola, Olanrewaju M.},
      journal = {Transactions on Machine Learning Research},
      issn    = {2835-8856},
      year    = {2026},
      url     = {https://openreview.net/forum?id=UibXoy4h3u}
    }
    @article{anderson2001permutation,
      title   = {Permutation tests for linear models},
      author  = {Anderson, Marti J. and Robinson, John},
      journal = {Australian \& New Zealand Journal of Statistics},
      volume  = {43}, number = {1}, pages = {75--88}, year = {2001},
      doi     = {10.1111/1467-842X.00156}
    }
    @article{barber2015controlling,
      title   = {Controlling the false discovery rate via knockoffs},
      author  = {Barber, Rina Foygel and Cand{\`e}s, Emmanuel J.},
      journal = {The Annals of Statistics},
      volume  = {43}, number = {5}, pages = {2055--2085}, year = {2015},
      doi     = {10.1214/15-AOS1337}
    }
    @article{benjamini1995controlling,
      title   = {Controlling the false discovery rate: a practical and powerful approach to multiple testing},
      author  = {Benjamini, Yoav and Hochberg, Yosef},
      journal = {Journal of the Royal Statistical Society: Series B (Methodological)},
      volume  = {57}, number = {1}, pages = {289--300}, year = {1995},
      doi     = {10.1111/j.2517-6161.1995.tb02031.x}
    }
    @article{benjamini2001control,
      title   = {The control of the false discovery rate in multiple testing under dependency},
      author  = {Benjamini, Yoav and Yekutieli, Daniel},
      journal = {The Annals of Statistics},
      volume  = {29}, number = {4}, pages = {1165--1188}, year = {2001},
      doi     = {10.1214/aos/1013699998}
    }
    @article{besag1991sequential,
      title   = {Sequential {Monte Carlo} p-values},
      author  = {Besag, Julian and Clifford, Peter},
      journal = {Biometrika},
      volume  = {78}, number = {2}, pages = {301--304}, year = {1991},
      doi     = {10.1093/biomet/78.2.301}
    }
    @article{candes2018panning,
      title   = {Panning for gold: `model-{X}' knockoffs for high dimensional controlled variable selection},
      author  = {Cand{\`e}s, Emmanuel and Fan, Yingying and Janson, Lucas and Lv, Jinchi},
      journal = {Journal of the Royal Statistical Society: Series B (Statistical Methodology)},
      volume  = {80}, number = {3}, pages = {551--577}, year = {2018},
      doi     = {10.1111/rssb.12265}
    }
    @article{fan2016guarding,
      title   = {Guarding against spurious discoveries in high dimensions},
      author  = {Fan, Jianqing and Zhou, Wen-Xin},
      journal = {Journal of Machine Learning Research},
      volume  = {17}, number = {203}, pages = {1--34}, year = {2016},
      url     = {https://jmlr.org/papers/v17/16-068.html}
    }
    @article{freedman1983nonstochastic,
      title   = {A nonstochastic interpretation of reported significance levels},
      author  = {Freedman, David and Lane, David},
      journal = {Journal of Business \& Economic Statistics},
      volume  = {1}, number = {4}, pages = {292--298}, year = {1983},
      doi     = {10.1080/07350015.1983.10509354}
    }
    @inproceedings{liu2019power,
      title     = {Power analysis of knockoff filters for correlated designs},
      author    = {Liu, Jingbo and Rigollet, Philippe},
      booktitle = {Advances in Neural Information Processing Systems},
      publisher = {Curran Associates, Inc.},
      volume    = {32}, year = {2019},
      url       = {https://proceedings.neurips.cc/paper_files/paper/2019/hash/09ab23b6b607496f095feed7aaa1259b-Abstract.html}
    }
    @article{phipson2010permutation,
      title   = {Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn},
      author  = {Phipson, Belinda and Smyth, Gordon K.},
      journal = {Statistical Applications in Genetics and Molecular Biology},
      volume  = {9}, number = {1}, year = {2010},
      note    = {Article 39},
      doi     = {10.2202/1544-6115.1585}
    }
    @article{winkler2014permutation,
      title   = {Permutation inference for the general linear model},
      author  = {Winkler, Anderson M. and Ridgway, Gerard R. and Webster, Matthew A. and Smith, Stephen M. and Nichols, Thomas E.},
      journal = {NeuroImage},
      volume  = {92}, pages = {381--397}, year = {2014},
      doi     = {10.1016/j.neuroimage.2014.01.060}
    }
    ```
