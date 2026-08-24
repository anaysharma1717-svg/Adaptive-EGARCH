<title>Adaptive EGARCH</title>

# Adaptive EGARCH — SPY Volatility Forecasting

**Question:** can next-day S&P 500 volatility be forecast better than a simple statistical baseline, and does a documented volatility-forecasting model's edge during market crises survive the standard fix for that model's known bias? **Headline finding:** yes to the first (marginally, and not with statistical significance across the board) — a static combination model is the best all-round forecaster, but no significance test in this project shows it, or anything else, *significantly* beating the simplest baseline overall. For the second question: no — a real, statistically significant crisis-period edge exists in one specific model, and every method tried to preserve it through bias correction measurably weakens or destroys it.

## Table of Contents
1. [Data](#1-data)
2. [Methodology](#2-methodology)
3. [Headline finding](#3-headline-finding)
4. [The correction chain: M2b → M2e](#4-the-correction-chain-m2b--m2e)
5. [Appendix: horizon and calendar robustness checks](#5-appendix-horizon-and-calendar-robustness-checks)
6. [Limitations](#6-limitations)
7. [How to reproduce](#7-how-to-reproduce)

---

## 1. Data

SPY (S&P 500 ETF) daily OHLC (Open/High/Low/Close), fetched via `yfinance`.

- **Raw data range:** 2011-01-03 → 2026-08-21 (3,932 trading days).
- **Feature-engineered range:** 2012-01-03 onward (the earliest ~250 trading days are consumed as warmup by a 252-day rolling percentile feature).
- **Out-of-sample test window:** 2025-02-24 → 2026-08-21 (376 trading days).

**Target — Garman-Klass realized volatility (GK-RV):**

```
GK_variance = 0.5 * ln(High/Low)^2 − (2·ln2 − 1) * ln(Close/Open)^2
RV = sqrt(GK_variance) * 100          # daily % volatility
```

A single day's squared close-to-close return is a very noisy estimate of that day's true variance — it only uses two price points. Garman-Klass uses the day's full trading range (all four OHLC prices) and is a standard, more statistically efficient (lower-variance, still unbiased) estimator.

## 2. Methodology

**Nine models compared**, walk-forward, expanding training window, refit every 21 trading days, strict no-look-ahead (every forecast for day *t* uses only data through *t*-1 — independently leakage-audited by tracing the actual code path, not inferred from variable names):

M1 HAR+TS (lagged-RV OLS baseline) · M2 raw EGARCH(1,1,1,t) · M2b bias-corrected EGARCH · M3 (HAR + M2b combined) · M4 Log-HAR · M5 Asymmetric HAR · M6 HAR+Returns · M7 Rolling-window HAR · M8 Ridge "kitchen sink".

**Why QLIKE, not just RMSE:** QLIKE (`mean(r − ln(r) − 1)`, where `r = actual²/forecast²`) is the standard loss function in the volatility-forecasting literature because, unlike RMSE, it stays well-behaved even when scored against a noisy *proxy* for the true (unobservable) variance — which GK-RV is, being a good but imperfect stand-in. RMSE alone can rank two forecasts in the wrong order under that kind of proxy noise; QLIKE is specifically robust to it (Patton, 2011).

**Diebold-Mariano (DM) test, HLN-corrected:** answers "is model A's average loss significantly lower than model B's?", accounting for the fact that both models are usually wrong on the *same* days (their errors are correlated, so you can't just eyeball two average losses). The Harvey-Leybourne-Newbold correction adjusts for small-sample bias in the basic test.

**Mincer-Zarnowitz (MZ) regression:** regresses the actual outcome on the forecast (`actual = α + β·forecast + ε`). An "efficient" forecast — one that isn't systematically biased or mis-scaled — has α=0 and β=1, tested jointly with an F-test.

## 3. Headline finding

**8-model comparison, full out-of-sample test window (N=376):**

| model | RMSE | QLIKE | rank |
|---|---|---|---|
| M3 (HAR + corrected EGARCH) | 0.329 | 0.341 | 1 |
| M2b (corrected EGARCH) | 0.341 | 0.359 | 2 |
| M8 (Ridge kitchen sink) | 0.353 | 0.368 | 3 |
| M6 (HAR + returns) | 0.353 | 0.368 | 4 |
| M5 (asymmetric HAR) | 0.362 | 0.374 | 5 |
| M1 (HAR baseline) | 0.368 | 0.388 | 6 |
| M7 (rolling-window HAR) | 0.374 | 0.387 | 7 |
| M4 (Log-HAR) | 0.386 | 0.448 | 8 |
| M2 (raw EGARCH) | 0.435 | 0.422 | 9 |

**Honest result:** of the 8 models tested against the M1 baseline (16 DM tests: 8 models × 2 loss functions), **only one reaches significance at the 5% level — and it's M4 Log-HAR being significantly *worse* than the baseline** (p=0.0004). Every other model, including the best-ranked M3, is statistically indistinguishable from the simple HAR baseline overall. Rank order is not the same as proven superiority, and this project doesn't conflate the two.

**The actual discovery** is not about the overall ranking — it's about one subsample. In the top-decile realized-volatility ("crisis") days, **raw, uncorrected EGARCH (M2) significantly beats the M1 baseline on QLIKE**:

> **Diebold-Mariano test, HLN-corrected: DM = +2.797, p = 0.0081, N = 38**

This is real and significant. It's also the model ranked *worst* overall (M2, RMSE 0.435) — a real edge, hiding inside a model that looks bad on average. That combination is what makes the rest of this project worth doing.

## 4. The correction chain: M2b → M2e

EGARCH's known bias is normally fixed with an MZ correction (§2) — that's exactly what M2b already is. The question this chain answers: **does that standard fix preserve the crisis edge, or erase it?** Four iterations, each one a direct response to what the previous one found — not four independent experiments.

**M2b (the existing fix).** A single MZ correction, `corrected = α + β·raw`, fit once on all training days. **Diagnosis:** that correction is fit on a sample that's 90% calm days, so its one β reflects calm-day dynamics. Splitting the same regression by regime confirms it directly: β_crisis = 0.830 vs. β_calm = 0.617 (t ≈ 5.85 — a large, unambiguous difference). Since β < 1 means "shrink the raw signal," the pooled correction over-shrinks specifically on crisis days. Measured cost: M2b vs. raw M2, crisis QLIKE, DM = −3.83, p = 0.0005 — the fix measurably destroys the edge it was meant to preserve.

**M2c: split the correction by regime.** Two separate MZ regressions (crisis-only, calm-only training days), applied by each day's own regime, known only from data through *t*-1. **Mechanism 1 — pooled-vs-regime:** this directly targets the diagnosis above, and it works, partially — a real, significant improvement over M2b (crisis QLIKE roughly halves). But M2c is *still* significantly worse than raw EGARCH in crisis (p=0.0002). Splitting the regression fixes the *pooling* problem but not the underlying one: an OLS-based correction shrinks toward the mean by construction, and QLIKE punishes shrinkage during a real crisis hard.

**M2d: fit the correction to minimize QLIKE directly, not squared error.** If OLS's own optimization target is the problem, minimize the metric that actually matters instead (numerically, via `scipy.optimize`, since there's no closed form). The specific prediction going in — QLIKE-optimal β should come out *higher* than OLS's, resisting shrinkage — **was wrong**. QLIKE-fit β came out *lower* than OLS β, in every regime, at every one of 18 refits, with zero exceptions. Reported as a falsified prediction, not dropped quietly. (M2d-regime was still a large, significant improvement over M2c, just not for the hypothesized reason — and still lost to raw EGARCH in crisis, p=0.0002.)

**M2e: stop correcting in crisis at all.** If every correction tried so far loses to no correction, use raw EGARCH directly on ex-ante crisis days, and only correct on calm days. **Mechanism 2 — ex-ante detection:** the best crisis numbers of the whole chain — but the ex-ante regime flag (yesterday's volatility percentile) is an imperfect predictor of an *actual* crisis day: only 19 of 45 flagged days turned out to be real crisis days in the test window. The other 26 are false alarms, where the noisier raw signal gets used on a day that turns out calm — which shows up as a real, statistically significant calm-day cost (several comparisons at p=0.002–0.02). The chain's final answer: **no correction method tested recovers the crisis edge with statistical significance against the best static baseline (M3)** — M2e/M9e get numerically closest, never significantly.

*(M9 through M9e are the same four EGARCH-correction variants combined smoothly with the HAR baseline via a logistic weight, rather than used standalone — same finding, same mechanism, detailed results in `research/results/05_regime_weighted_combination/`.)*

## 5. Appendix: horizon and calendar robustness checks

Two checks on whether the headline crisis finding is fragile, run after the main investigation concluded.

**h=5 (next-week) horizon:** does the crisis-QLIKE edge survive forecasting the *average* volatility over the next 5 trading days instead of just tomorrow? **Large effect, partially replicates:** the crisis edge for raw EGARCH remains significant (DM=+2.46, p=0.02, N=29) — but EGARCH's h=5 forecast is dramatically worse than the baseline everywhere *else* (overall and calm both p<0.0002), a much bigger overall degradation than at h=1. Not a clean "yes."

**FOMC-announcement dummy:** does knowing a day is a scheduled Fed announcement day improve the baseline model? **Large effect, underpowered test:** the coefficient is real and highly significant in-sample (+0.224, t=7.17, p<0.0001 — FOMC days are more volatile, as expected), and out-of-sample error on FOMC days themselves drops substantially (QLIKE 0.593 → 0.141). But with only 12 FOMC days in the 376-day test window, no DM test reaches significance (p=0.17–0.39). This is a real effect the test is underpowered to confirm, not a null result — those are different things and this project keeps them distinct.

## 6. Limitations

- **Daily range-based volatility, not intraday.** GK-RV uses daily OHLC, not tick data — a real ceiling on how precisely "true" volatility is measured here.
- **N=38 crisis subsample.** The central finding rests on 38 out-of-sample crisis days. Real and significant, but a small sample — later results (§4, §5) sometimes flip on small movements within it.
- **~127 Diebold-Mariano tests were run over the course of this investigation** (crisis test → M9's failure → the M2b diagnosis → M2c/M2d/M2e → the M9 combination chain → the two robustness checks), plus a further ~18 summary comparisons added when this README's tables were assembled. This is disclosed, not hidden, because of the multiple-comparisons problem: at a 5% threshold, a handful of false positives are expected by chance alone across that many tests. No formal correction (e.g. Bonferroni) is applied, because each test followed from a specific question the *prior* result raised — this was an iterative investigation, not a search across many independent hypotheses for the first one to clear p<0.05. That said, any single nominally-significant result here should be read with that base rate in mind.
- **No trading strategy, backtest, or profitability claim of any kind exists in this project.** Everything above is a forecast-accuracy comparison, evaluated on statistical loss functions (RMSE/MAE/QLIKE) and significance tests (DM/MZ) — not a P&L, not a position-sizing rule, not a deployment.

## 7. How to reproduce

`research/pipeline/` contains the original analysis scripts, run in sequence (see `research/pipeline/README.md` for the exact order — Task 1 → Task 3 → Task 4/Task A → Step 1 → Step 2 → Step 3 → Step 4 → h=5/FOMC appendix), that produced every result in this project.

`research/01-06` are a **verification/walkthrough layer**, not the analysis itself: `01` reproduces the volatility engine from scratch (re-fetches data, rebuilds RV and ACF/PACF); `02-04` recompute statistical tests (metrics, DM tests, MZ regressions) from already-saved forecasts, without re-fitting any model; `05-06` display already-computed results as-is, each with a pointer in its own docstring to the exact `research/pipeline/` script that actually produced what it's showing.

| Result | Verification script | Original computation |
|---|---|---|
| §1 Data, RV, ACF/PACF | `research/01_data_and_rv_engine.py` | *(recomputes from raw data itself)* |
| §3 8-model comparison | `research/02_model_comparison.py` | `research/pipeline/extended_model_zoo.py` |
| §3 crisis DM test | `research/03_crisis_regime_test.py` | `research/pipeline/task1_dm_significance.py`, `taskA_m9_regime_eval.py` |
| §4 correction chain (M2b–M2e) | `research/04_bias_correction_chain.py` | `research/pipeline/stepB1-4_*.py` |
| §4 combination chain (M9–M9e) | `research/05_regime_weighted_combination.py` | `research/pipeline/taskA_m9_regime_eval.py`, `stepB2-4_*.py` |
| §5 h=5 / FOMC appendix | `research/06_horizon_and_calendar_appendix.py` | `research/pipeline/appendix_h5_and_macro.py` |

```bash
pip install -r requirements.txt

# verification layer -- fast, replays already-saved numbers
python research/01_data_and_rv_engine.py   # ... through 06_horizon_and_calendar_appendix.py

# pipeline -- slow, actually re-fits/refits models against real data
python research/pipeline/extended_model_zoo.py
python research/pipeline/task1_dm_significance.py
# ... see research/pipeline/README.md for the rest, in order
```

**Dependencies** (`requirements.txt`): `yfinance`, `arch` (EGARCH/GARCH), `scikit-learn` (OLS/Ridge), `statsmodels` (ADF, ACF/PACF, Ljung-Box), `scipy` (DM/MZ test statistics, QLIKE numerical fitting), `pandas`, `numpy`, `matplotlib`.
