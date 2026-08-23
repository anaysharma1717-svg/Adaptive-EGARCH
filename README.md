# Adaptive EGARCH — SPY Volatility Forecasting & Regime-Conditional Bias Correction

**A walk-forward research project testing whether a documented EGARCH crisis-period edge can be preserved through bias correction and exploited via model combination — and honestly reporting that, after five iterations, it can't be done with statistical significance.**

This is not a "here's a model that beats the market" repo. It's a record of a rigorous, negative-result-friendly research process: a real discovery (raw EGARCH beats a HAR baseline during volatility crises), a real bug found by that discovery (the standard bias-correction method destroys the edge it's supposed to fix), four different attempted fixes, and an honest conclusion that none of them clear the statistical bar. That process — hypothesize, test, get a null result, diagnose *why*, iterate, and still report the null result plainly — is the actual deliverable.

---

## Table of Contents
1. [What this project forecasts](#what-this-project-forecasts)
2. [Headline results](#headline-results)
3. [The investigation — how the story unfolds](#the-investigation)
4. [Methodology](#methodology)
5. [What failed (and why that's the point)](#what-failed)
6. [Repository structure](#repository-structure)
7. [How to run it](#how-to-run-it)
8. [Tech stack](#tech-stack)

---

## What this project forecasts

The target is **1-day-ahead realized volatility (RV)** of SPY, measured with the **Garman-Klass estimator** (a volatility proxy built from a day's Open/High/Low/Close prices, more efficient than squared close-to-close returns because it uses the whole day's price range, not just the endpoints). Nine forecasting models are compared out-of-sample, walk-forward, with a strict no-look-ahead rule enforced and independently audited: **no forecast for day *t* is ever allowed to see day *t*'s own data.**

## Headline results

**Baseline model comparison** (376 out-of-sample trading days, 2025‑02‑24 → 2026‑08‑21):

| Model | RMSE | QLIKE | Notes |
|---|---|---|---|
| M1 — HAR+TS (OLS on lagged RV) | 0.368 | 0.388 | Simple baseline, hard to beat |
| M2 — Raw EGARCH(1,1,1,t) | 0.435 | 0.422 | Worst overall RMSE, **but see crisis result below** |
| M2b — MZ-bias-corrected EGARCH | 0.341 | 0.359 | Standard fix for EGARCH's known upward bias |
| **M3 — HAR + corrected-EGARCH (OLS combination)** | **0.329** | **0.341** | Best overall — static combination wins on average |

**The discovery that drove the rest of the project:** in the **crisis subsample** (top-decile realized-vol days), raw EGARCH significantly beats the HAR baseline on QLIKE loss (a loss function built for volatility forecasts specifically — see [Methodology](#methodology)):

> **Diebold-Mariano test, HLN-corrected, crisis QLIKE: DM = +2.797, p = 0.0081, N = 38**

That's a real, statistically significant edge. The problem: **M2b — the standard bias correction — destroys it** (DM = −3.83, p = 0.0005 vs. raw EGARCH, same subsample). Everything from Step 1 onward is the investigation into why, and four attempts to fix it.

## The investigation

```
Task 1  Crisis DM test on the static combination (M3 vs M1)     → NOT significant (p≈0.15–0.88, honestly reported)
Task 2  Leakage audit of the bias correction                     → confirmed leak-free
Task 3  Found & fixed a real bug in the Log-HAR model             → Jensen's-inequality retransformation bias
Task 4  Built M9 — logistic regime-weighted combination            → doesn't beat M3 in crisis (negative result)
Step 1  Diagnosed WHY: MZ correction is 90% fit on calm days       → confirmed, β differs by regime (t≈5.85)
Step 2  M2c — split the correction by regime                       → real improvement, still loses to raw EGARCH
Step 3  M2d — refit the correction to minimize QLIKE, not OLS      → the motivating PREDICTION WAS WRONG
Step 4  M2e — stop correcting in crisis, use raw EGARCH there       → best result yet, still not significant vs M3
```

Full numbers, per-step, are in [`research/results/`](research/results/) — every table above is a saved CSV, not a claim. **117 Diebold-Mariano tests** were run across this investigation; that multiplicity is tracked and disclosed, not swept under the rug (see the detailed report).

## Methodology

- **Garman-Klass RV**: `σ² = 0.5·(ln(H/L))² − (2ln2−1)·(ln(C/O))²` — an intraday-range volatility estimator, more statistically efficient than squared returns.
- **EGARCH(1,1,1,t)**: an asymmetric GARCH variant (captures the *leverage effect* — volatility rises more after a price drop than after an equal-sized rise) with Student-t innovations, fit via the `arch` package.
- **Walk-forward evaluation**: expanding training window, refit every 21 trading days, forecast made using only data through *t*−1. Independently audited for leakage.
- **QLIKE loss**: `mean(r − ln(r) − 1)` where `r = actual²/forecast²` — the standard loss function in the volatility-forecasting literature, because unlike MSE it stays well-behaved (asymptotically unbiased) even when the realized-vol *proxy* itself is noisy.
- **Diebold-Mariano test with the Harvey-Leybourne-Newbold (HLN) small-sample correction** — tests whether one forecast's loss series is significantly lower than another's.
- **Mincer-Zarnowitz regression**: `actual = α + β·forecast + ε`; efficiency requires α=0, β=1, tested jointly with an F-test.

## What failed

This is deliberately not hidden in an appendix:

- **Task 1's headline claim didn't replicate** when tested properly: corrected-M3 vs M1 is *not* significant in any subsample (p ranges 0.15–0.88).
- **M9** (the first combination model, logistic-weighted by a volatility-regime signal) does not beat the static M3 combination in crisis — directionally worse on QLIKE and MSE.
- **The core theoretical prediction in Step 3 was falsified**: QLIKE-optimal correction coefficients came out *lower* than the OLS ones, the opposite of what the diagnosis predicted.
- **Every corrected EGARCH variant (M2b, M2c, M2d) remains significantly worse than raw, uncorrected EGARCH in crisis** (p ≈ 0.0002–0.0005) — bias correction, in any form tested, costs you the crisis edge.
- **The best variant (M9e) still doesn't reach significance** against the static M3 baseline (p = 0.18–0.28) — directionally the closest yet, but not proven.

## Repository structure

Two layers, deliberately kept separate:

**Read layer** — `research/01_*.py` through `research/07_*.py`. Clean, numbered, heavily-commented scripts that each answer one question and reproduce the exact already-saved numbers from `research/results/`. Start here if you want to understand the project.

**Compute layer** — `research/extended_model_zoo.py` (the core engine: data, features, all model definitions, DM test, MZ regression) plus `stepB1-4_*.py`, `taskA_m9_regime_eval.py`, `task1_dm_significance.py`, `task3_loghar_fix.py`, `appendix_h5_and_macro.py`. This is where the EGARCH walk-forward is actually fit, refit, and simulated — the code that originally produced every number the read layer reproduces. Kept, not deleted, because it's the only way to regenerate results from raw data; the read layer only replays already-computed CSVs.

```
research/
  01_data_and_rv_engine.py             data, Garman-Klass RV, ACF/PACF
  02_model_comparison.py               the 8-model + M2/M2b/M3 comparison
  03_crisis_regime_test.py             the crisis DM tests (both the null-result
                                        original test and the p=0.008 discovery)
  04_bias_correction_chain.py          M2b -> M2c -> M2d -> M2e, why each step happened
  05_regime_weighted_combination.py    M9 -> M9c -> M9d -> M9e
  06_horizon_and_calendar_appendix.py  h=5 replication, FOMC dummy
  07_pairs_trading_pilot.py            six pre-specified pairs, null result

  extended_model_zoo.py                core engine (compute layer)
  stepB1-4_*.py, taskA_*.py, task1/3_*.py, appendix_h5_and_macro.py   (compute layer)
  results/                             every metric, DM test, per-day forecast, as CSV
  pairs_trading/                       shelved pilot: data + results, cointegration/half-life analysis
```

## How to run it

```bash
pip install -r requirements.txt

# read layer -- fast, just reproduces already-saved numbers
python research/01_data_and_rv_engine.py
python research/02_model_comparison.py
python research/03_crisis_regime_test.py
python research/04_bias_correction_chain.py
python research/05_regime_weighted_combination.py
python research/06_horizon_and_calendar_appendix.py
python research/07_pairs_trading_pilot.py

# compute layer -- slow, actually refits EGARCH via simulation from raw data
python research/extended_model_zoo.py
python research/stepB1_diagnostic_regime_mz.py
python research/stepB2_m2c_m9c.py
python research/stepB3_m2d_m9d.py
python research/stepB4_m2e_m9e.py
```

## Tech stack

Python · `arch` (EGARCH/GARCH) · `scikit-learn` (OLS/Ridge) · `statsmodels` (ADF, diagnostics) · `scipy.optimize` (QLIKE numerical fitting) · `pandas`/`numpy` · `yfinance`
