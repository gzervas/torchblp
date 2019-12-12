# torchblp
PyTorch implementation of BLP'95.

## Acknowledgements

The data comes from:

Andrews, Isaiah; Gentzkow, Matthew; Shapiro, Jesse M., 2017, "Replication Data for: "Measuring the Sensitivity of Parameter Estimates to Estimation Moments"", https://doi.org/10.7910/DVN/LLARSN, Harvard Dataverse, V1

I used the MATLAB code of Andrews, Gentzkow, and Shapiro to construct GMM instruments.

The ```squarem``` implementation is adapted from https://github.com/jeffgortmaker/pyblp/tree/master/pyblp.

The code relies on various undocumented tricks to make this work on a GPU.

## Usage

You will need Python 3.

Clone the repo, and from within the ```src``` directory run:

```
python torchblp.py  --data-filename ../data/blp95.csv --unobs-filename ../data/blp95unobs.csv 
```

If you do not have a GPU use the ```--disable-cuda``` option. The code should produce identical results, though it will run slower.

## Results

With float64 precision, ```torchblp``` almost exactly replicates the estimates of Andrews, Gentzkow, and Shapiro (AGS'17). With float32 precision, the ```torchblp``` estimates are different from AGS'17 but not by much. The AGS'17 estimates themselves are different from the published BLP'95 estimates for various reasons including a bug in instrument construction in BLP'95 (see the AGS'17 appendix for details). The code takes about 20 secs to run on an NVIDIA Quadro RTX 4000 (about 5 secs are wasted loading and transforming data, the rest is estimation.) Inverting all market shares once takes about 0.12 secs using float64, and 0.07 secs using float32. The code does not produce standard errors, yet.

| param_name      | BLP'95 | AGS'17 | torchblp (float64) | torchblp (float32) |
|-----------------|-------:|-------:|-------------------:|-------------------:|
| alpha_price     | 43.501 | 42.870 | 42.870             | 43.712             |
| sigma_const     | 3.612  | 2.522  | 2.522              | 2.108              |
| sigma_hpwt      | 4.628  | 3.525  | 3.525              | 5.780              |
| sigma_air       | 1.818  | 4.167  | 4.166              | 3.312              |
| sigma_mpd       | 1.050  | 0.393  | 0.393              | 0.290              |
| sigma_space     | 2.056  | 1.937  | 1.937              | 1.823              |
| demand_const    | -7.061 | -7.728 | -7.728             | -7.394             |
| demand_hpwt     | 2.883  | 4.620  | 4.620              | 3.471              |
| demand_air      | 1.521  | -1.227 | -1.226             | -0.512             |
| demand_mpd      | -0.122 | 0.293  | 0.293              | 0.338              |
| demand_space    | 3.460  | 3.992  | 3.992              | 4.160              |
| supply_const    | 0.952  | 2.751  | 2.751              | 2.804              |
| supply_loghpwt  | 0.477  | 0.812  | 0.812              | 0.853              |
| supply_air      | 0.619  | 0.430  | 0.430              | 0.393              |
| supply_logmpg   | -0.415 | -0.610 | -0.610             | -0.617             |
| supply_logspace | -0.046 | -0.352 | -0.352             | -0.362             |
| supply_trend    | 0.019  | 0.027  | 0.027              | 0.028              |

## References

1. https://academic.oup.com/qje/article/132/4/1553/3861634
2. https://doi.org/10.7910/DVN/LLARSN
3. https://chrisconlon.github.io/site/pyblp.pdf
4. https://github.com/jeffgortmaker/pyblp/tree/master/pyblp
