# SPY GARCH vs EGARCH -- Walk-Forward Evaluation Results
## Scripts: `egarch_vs_garch.py` (baseline) + `egarch_evaluation.py`  |  Run: 2026-07-06

---

## Overview

| Field | Value |
|---|---|
| Ticker | SPY |
| Sample | 2015-01-05 -- 2026-07-06 |
| Forecast observations | 2,492 |
| Monthly refit windows | 120 |
| Walk-forward type | Expanding window (all past data retained) |
| Evaluation targets | Yang-Zhang RV (21-day) + 20-day Rolling RV |
| EGARCH distribution | Student-t (best by AIC in all refit windows) |

---

## Walk-Forward Forecasting Performance  (h = 1 day)

### Target 1: Yang-Zhang (2000) Realized Volatility

| Metric | GARCH(1,1) | EGARCH(1,1,1,t) | Winner |
|---|---|---|---|
| Avg RMSE (%/day) | 0.344913 | 0.418734 | **GARCH** |
| Avg MAE  (%/day) | 0.212402 | 0.245276 | **GARCH** |
| % months EGARCH wins | -- | 38.3% | -- |
| % months GARCH wins  | 61.7% | -- | -- |

### Target 2: 20-Day Rolling Realized Volatility

| Metric | GARCH(1,1) | EGARCH(1,1,1,t) | Winner |
|---|---|---|---|
| Avg RMSE (%/day) | 0.316293 | 0.397080 | **GARCH** |
| Avg MAE  (%/day) | 0.197191 | 0.230796 | **GARCH** |

---

## Multi-Horizon Forecasting Performance

> h=1 forecasts use h=1 sequential state update; h=5 and h=21 use the
> analytic mean-reverting formula from the h=1 state.  Target is Yang-Zhang RV
> at the forecast date + h trading days (row-shift on the same rv_yz series).

### RMSE (%/day) Ã¢â‚¬â€ Yang-Zhang target

| Horizon | GARCH(1,1) | EGARCH(1,1,1,t) | Winner |
|---|---|---|---|
| h =  1 | 0.344913 | 0.418734 | **GARCH** |
| h =  5 | 0.283746 | 0.386362 | **GARCH** |
| h = 21 | 0.524885 | 0.574831 | **GARCH** |

### MAE (%/day) Ã¢â‚¬â€ Yang-Zhang target

| Horizon | GARCH(1,1) | EGARCH(1,1,1,t) | Winner |
|---|---|---|---|
| h =  1 | 0.212402 | 0.245276 | **GARCH** |
| h =  5 | 0.196446 | 0.204765 | **GARCH** |
| h = 21 | 0.337410 | 0.297752 | **EGARCH** |

### Diebold-Mariano Tests Ã¢â‚¬â€ Multi-Horizon (HLN 1997, squared loss)

| Horizon | DM Stat | p-value | Conclusion |
|---|---|---|---|
| h =  1 | -4.7243 | 0.0000 | GARCH is significantly MORE accurate (DM=-4.724, p=0.0000, alpha=0.05). |
| h =  5 | -1.7769 | 0.0757 | No significant difference in predictive accuracy (DM=-1.777, p=0.0757, cannot reject H0). |
| h = 21 | -0.7369 | 0.4612 | No significant difference in predictive accuracy (DM=-0.737, p=0.4612, cannot reject H0). |

---

## Forecast Accuracy by VIX Regime

> VIX source: CBOE via Yahoo Finance (^VIX).  Regimes defined on the daily
> closing VIX at each forecast date.

| Regime | Definition | N Days | VIX Range | GARCH RMSE | EGARCH RMSE | GARCH MAE | EGARCH MAE | Winner |
|---|---|---|---|---|---|---|---|---|
| Low | VIX < 15 | 864 | 9.1 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“ 15.0 | 0.152393 | 0.148341 | 0.116511 | 0.112235 | EGARCH |
| Medium | 15 ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¤ VIX < 25 | 1263 | 15.0 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“ 25.0 | 0.269149 | 0.310625 | 0.196657 | 0.232953 | GARCH |
| High | VIX ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¥ 25 | 365 | 25.0 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“ 82.7 | 0.711745 | 0.900633 | 0.493874 | 0.602844 | GARCH |

---

## Diebold-Mariano Test (Harvey, Leybourne & Newbold 1997 Corrected)

H0: GARCH and EGARCH have equal predictive accuracy (squared error, h=1)

### vs Yang-Zhang Target

| Statistic | Value |
|---|---|
| DM statistic (HLN corrected) | -4.7243 |
| p-value (two-sided) | 0.0000 |
| Mean GARCH squared loss | 0.118965 |
| Mean EGARCH squared loss | 0.175338 |
| **Conclusion** | **GARCH is significantly MORE accurate (DM=-4.724, p=0.0000, alpha=0.05).** |

### vs 20-Day Rolling RV Target

| Statistic | Value |
|---|---|
| DM statistic (HLN corrected) | -5.0646 |
| p-value (two-sided) | 0.0000 |
| Mean GARCH squared loss | 0.100041 |
| Mean EGARCH squared loss | 0.157673 |
| **Conclusion** | **GARCH is significantly MORE accurate (DM=-5.065, p=0.0000, alpha=0.05).** |

---

## Rolling Parameter Estimates

> Statistics computed across all monthly refit windows.

### GARCH(1,1)

| Parameter | Mean | Min | Max |
|---|---|---|---|
| omega | 0.045779 | 0.036867 | 0.097639 |
| alpha | 0.211907 | 0.173679 | 0.256340 |
| beta  | 0.744514  | 0.676116  | 0.792429 |
| alpha+beta (persistence) | 0.956421 | 0.887526 | 0.981133 |
| Log-likelihood | -2032.58 | -3727.56 | -496.79 |
| AIC | 4073.15 | 1001.58 | 7463.12 |
| BIC | 4094.26 | 1017.30 | 7486.99 |

### EGARCH(1,1,1,t)

| Parameter | Mean | Min | Max |
|---|---|---|---|
| omega | -0.016808 | -0.042304 | -0.004804 |
| alpha | 0.199719 | -0.087944 | 0.259065 |
| **gamma (leverage)** | **-0.201666** | -0.338987 | -0.173275 |
| beta  | 0.959593  | 0.916518  | 0.990577 |
| nu (d.o.f.) | 5.9863 | 3.9993 | 11.4207 |
| Log-likelihood | -1940.83 | -3566.04 | -477.16 |
| AIC | 3893.66 | 966.31 | 7144.09 |
| BIC | 3925.33 | 989.89 | 7179.90 |

---

## Leverage Effect  --  Structural Robustness

- gamma remained **consistently negative** across all 120 monthly refit windows
- Mean gamma = -0.2017  (range: [-0.3390,  -0.1733])
- gamma was negative in every single refit -- the leverage effect is a **structural feature of SPY volatility**, not a single-sample artefact
- COVID crash and 2022 rate hike regimes show the most negative gamma,
  consistent with elevated fear asymmetry during high-stress periods

---

## Files Generated

| File | Description |
|---|---|
| `egarch_vs_garch.py` | Original baseline -- **unchanged** |
| `egarch_evaluation.py` | This evaluation module |
| `egarch_vs_garch.png` | Original 4-panel baseline chart |
| `egarch_param_evolution.png` | Parameter evolution + gamma + regime shading |
| `egarch_rolling_eval.png` | 2-panel: forecast vs YZ + rolling RMSE |
| `forecasts.csv` | Every forecast + realized vol + multi-horizon errors |
| `adaptive_forecast_data.csv` | Clean export for adaptive model training |
| `params_garch.csv` | GARCH params, LL, AIC, BIC at every refit |
| `params_egarch.csv` | EGARCH params, LL, AIC, BIC at every refit |

---

## Next Steps

1. **Adaptive EGARCH** -- time-varying gamma via regime switching or kernel weighting
2. **HMM Regime Detection** -- per-regime EGARCH to capture structural breaks
3. **Mincer-Zarnowitz Regression** -- formal unbiasedness test (slope=1, intercept=0)
4. **Realized GARCH** -- incorporate realized variance directly into the recursion
5. **Intraday RV** -- use 5-min returns for cleaner daily variance proxies