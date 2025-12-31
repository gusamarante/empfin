# empfin
_**UNDER CONSTRUCTUTION**_

Available models for estimation of risk premia:
- `TimeseriesReg`: Timeseries regression. The single-pass OLS described in Section 12.1 of Cochrane (2009)
- `CrossSectionReg`: Cross-Sectional regression. The two-pass OLS described in Section 12.2 of Cochrane (2009)
- `RiskPremiaTermStructure`: Estimation of the term structure of risk premia based on a single factor, tradeable or non-tradable. Model from Bryzgalova, Huang and Julliard (2024)

# References
Bryzgalova, Svetlana, Jiantao Huang, and Christian Julliard. 2024. [“Macro Strikes Back: Term Structure of Risk Premia and Market Segmentation.”](https://doi.org/10.2139/ssrn.4752696) Working Paper

Cochrane, John H. 2009. ["Asset Pricing: Revised Edition"](https://press.princeton.edu/books/hardcover/9780691121376/asset-pricing?srsltid=AfmBOoobXP_DmuPEfu1g7gm1ppk4h69GFHtwJqq0ugoZwSYKW60gLXZ6). Princeton University Press.

Campbell, Lo, and MacKinlay. 2012. _"The Econometrics of Financial Markets"_.