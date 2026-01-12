# empfin - Empirical Finance Tools in Python
`empfin` is a Python toolkit for empirical asset pricing models and risk premia estimation. This library is in active development and aims to implement models from all corners of the literature.

# What's Inside
Currently available models for estimation of risk premia:
- `TimeseriesReg`: single-pass OLS time-series regression, described in [Cochrane (2005)](https://press.princeton.edu/books/hardcover/9780691121376/asset-pricing?srsltid=AfmBOoobXP_DmuPEfu1g7gm1ppk4h69GFHtwJqq0ugoZwSYKW60gLXZ6), Section 12.1
- `CrossSectionReg`: two-pass cross-sectional regression, described in [Cochrane (2005)](https://press.princeton.edu/books/hardcover/9780691121376/asset-pricing?srsltid=AfmBOoobXP_DmuPEfu1g7gm1ppk4h69GFHtwJqq0ugoZwSYKW60gLXZ6), Section 12.2
- `NonTradableFactors`: iterative maximum-likelihood estimator for non-tradable factors, described in [Campbell, Lo & MacKinlay (2012)](https://www.amazon.com/Econometrics-Financial-Markets-John-Campbell/dp/0691043019), Section 6.2.3 
- `RiskPremiaTermStructure`: term structure of risk premia with a single factor, tradable or not, following [Bryzgalova, Huang & Julliard (2024)](https://doi.org/10.2139/ssrn.4752696). I would like to thank the authors for sharing their replication files.

# Examples
For each model, there is a jupyter notebook with [examples](https://github.com/gusamarante/empfin/tree/main/examples) of their use.

# Installation
```bash
pip install empfin
```

# References
Bryzgalova, Huang, and Julliard (2024) [“_Macro Strikes Back: Term Structure of Risk Premia and Market Segmentation_”](https://doi.org/10.2139/ssrn.4752696) Working Paper

Cochrane (2009) ["_Asset Pricing: Revised Edition_"](https://press.princeton.edu/books/hardcover/9780691121376/asset-pricing?srsltid=AfmBOoobXP_DmuPEfu1g7gm1ppk4h69GFHtwJqq0ugoZwSYKW60gLXZ6). Princeton University Press.

Campbell, Lo, and MacKinlay (2012) ["_The Econometrics of Financial Markets_"](https://www.amazon.com/Econometrics-Financial-Markets-John-Campbell/dp/0691043019)

# Library Citation
> Gustavo Amarante (2026). empfin - Empirical Finance Tools in Python. Retrieved from https://github.com/gusamarante/empfin