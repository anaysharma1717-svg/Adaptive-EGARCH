# Adaptive EGARCH Project — Failure Report & Research Log

> **Purpose:** Comprehensive documentation of every approach attempted, why it failed, and what was learned. Serves as a research portfolio artifact showing evidence-driven, iterative methodology.
> **Date:** 2026-07-11
> **Conclusion:** GARCH(1,1) is the unbeaten baseline for h=1 daily SPY volatility forecasting.

---

## Final Scoreboard (Yang-Zhang RV Target, h=1)

| Model | RMSE | MAE | vs GARCH | Status |
|:---|:---:|:---:|:---:|:---:|
| **GARCH(1,1)** | **0.266** | **0.177** | — | ✅ Winner |
| EGARCH(1,1,1,t) | 0.295 | 0.202 | −11% worse | ✅ Kept (v3 best) |
| Adaptive v4 (smoothness + clip) | 0.299 | 0.204 | −12% worse | ✅ Kept (best adaptive) |
| Adaptive v2 (smoothness only) | 0.301 | 0.205 | −13% worse | ✅ Kept |
| Adaptive v3 (clip only) | 0.458 | 0.333 | −72% worse | ❌ Archived |
| Adaptive v1 (raw) | 0.438 | 0.319 | −65% worse | ❌ Archived |
| Adaptive v5 (HMM-driven) | N/A | N/A | Failed validation | ❌ Archived |
| Mod A: VIX-EGARCH | 0.329 | 0.227 | −24% worse | ❌ Archived |
| Mod B: Realized-EGARCH | 0.361 | 0.242 | −36% worse | ❌ Archived |
| Mod C: GJR-GARCH | 0.358 | 0.233 | −35% worse | ❌ Archived |

**Note:** At h=21, standard EGARCH beats GARCH on MAE (0.298 vs 0.337, p=0.0003). The leverage effect compounds at long horizons.

---

## Phase 1 — Adaptive EGARCH Framework (dp_egarch v1–v5)

### What we built
A custom walk-forward EGARCH where the parameters ω and α are no longer fixed scalars but time-varying functions of macroeconomic features:

```
omega_t = f(VIX, credit spread, yield curve, ...)
alpha_t = g(VIX, credit spread, yield curve, ...)
```

The idea: macro context should tell the model whether shocks are large (crisis) or small (calm), so the GARCH update weights should adapt accordingly.

### v1 — Raw Adaptive (RMSE 0.438)
**Failure:** Daily parameter swings of ±2.17 %/day. The optimizer treated each monthly refit window as a blank slate and found completely different local optima each time. Without any continuity constraint, the forecast series looked like random noise.

**Root cause:** The objective function `neg_loglik` is smooth globally but has many local optima. L-BFGS-B finds a different one every 21 days, producing wildly inconsistent forecasts.

### v2 — Temporal Smoothness (RMSE 0.301)
**Result:** Stabilized the model. Adding a penalty `λ·Σ(ω_t − ω_{t-1})²` forced consecutive parameter values to be similar, eliminating the daily jumps.
**Still lost to GARCH** but came close (0.301 vs 0.266). This was the first stable adaptive model.

### v3 — Feature Clipping Only (RMSE 0.458)
**Failure:** Worse than v1. Clipping z-scored macro features to ±3σ without the smoothness penalty caused the optimizer to pin parameters at the clip boundaries. The sigmoid saturated and the gradient died — the model stopped learning.

**Lesson:** Feature clipping alone is harmful. It only works as a secondary safeguard after smoothness is already enforced.

### v4 — Smoothness + Clipping (RMSE 0.299)
**Result:** Best adaptive version. Combining smoothness (smooth gradient landscape) with clipping (prevents extreme feature values from saturating sigmoid) achieved the lowest adaptive RMSE. Daily jumps reduced from 2.17 to 1.02 %/day.
**Still lost to GARCH.** The improvement from v2→v4 is marginal (0.301→0.299).

### v5 — HMM-Driven (Failed validation)
**Failure:** Attempted to use Hidden Markov Model regime probabilities as the sole driver of ω_t. The HMM itself failed all validation checks (see Phase 2 below). v5 was abandoned before OOS evaluation.

### Why Adaptive EGARCH Cannot Beat GARCH (Core Finding)
Adapting the structural parameters `(ω, α)` from macroeconomic features introduces more estimation noise than it removes structural bias. The GARCH(1,1) model's two-parameter structure (`α`, `β`) is near-optimal for SPY daily variance. Every additional parameter adds estimation variance that outweighs its in-sample explanatory power out-of-sample. This is a manifestation of the **bias-variance tradeoff** in a high-noise, low-signal environment (daily financial returns).

---

## Phase 2 — HMM Regime Detection (7 Architecture Variants)

### Motivation
To supply a stable, economically-meaningful regime signal to the Adaptive EGARCH. A Hidden Markov Model with 6 features (log VIX, VIX term structure, credit spread, yield curve, volume z-score, momentum) was the natural first choice.

### Variant 1: Expanding-Window, Cold-Restart
**Failure:** 85% label-switching rate across 56 refits (Hungarian algorithm diagnostic).
**Root cause:** EM algorithm finds different local optima at each refit, randomly relabelling "Calm" as "Crisis" across windows.

### Variant 2: Expanding-Window, Warm-Start
**Failure:** 0% label-switching but permanently stuck post-COVID. P(Crisis) = 1.0 from March 2020 onward forever.
**Root cause:** Warm-starting traps the EM in a deep local minimum after the COVID structural break.

### Variant 3: Frozen 3-State, Global Z-Score
**Failure:** EM split "Calm" into two era-specific sub-states instead of learning Calm/Transition/Crisis.
**Root cause:** 2022-23 yield-curve inversion was only 15% of training data. EM found it more useful to model 2011-14 vs 2015-19 macro differences.

### Variant 4: Frozen 2-State, Global Z-Score
**Failures:** (1) COVID stickiness (P(Crisis) never recovered below 0.5 after March 2020). (2) 2021 bull market classified as Crisis (P(Calm) = 0.019, VIX = 14).
**Root cause:** The Gaussian Calm state's mean vector encoded pre-COVID macro levels. Post-COVID observations (different yield curve, credit spread baselines) fell outside the trained Calm distribution despite low VIX.

### Variants 5 & 6: Rolling-Relative Z-Score (3yr window)
**Failure:** Rolling-relative z-scores introduced their own structural break at COVID. The 3-year rolling mean includes the COVID spike throughout 2020-2023, distorting the relative z-scores for the entire post-COVID recovery. Economic inversion of state meanings (Calm state had wide credit spreads; Crisis had tight spreads).

### Variant 7: Frozen 2-State, Rolling-Relative Z-Score
**Same failures as variants 4 & 6.**

### Universal Root Cause (All HMM Variants)
A Gaussian HMM learns an **absolute mean vector** per state during training. When macro baselines shift permanently (COVID-era monetary expansion, 2022-23 Fed hiking cycle), the trained mean vectors no longer match any post-break observation cluster. The model cannot separate "low VIX but unusual macro baseline" from "Crisis." This is a fundamental parametric family limitation — not fixable via feature engineering or preprocessing.

---

## Phase 3 — Percentile Composite Stress Score

### Motivation
Replace the HMM with a non-parametric, stationary composite indicator using 3-year rolling percentiles of RV, Credit, Yield Curve, VRP, and Momentum. No optimization, no local optima, no label-switching.

### Failure
The Yield Curve and Credit Spread percentiles — even after 3-year rolling normalization — remained elevated through all of 2021-2022 because the COVID shock contaminated the 3-year trailing mean/std throughout the post-COVID recovery period.

- 2021 bull market avg composite score: 0.638 (need < 0.4 for "Calm")
- Post-COVID min composite: 0.594 (need < 0.5 for "non-sticky")

Passed crisis responsiveness (all events > 0.6) but failed chronological consistency and non-stickiness — the same two tests the HMM failed.

**This confirmed the fundamental problem is not the HMM parametric family — it is using multi-year macro features at all.** Macro features (yield curve, credit) operate on business-cycle timescales (2-5 years). A daily volatility model trying to use these for day-to-day regime switches will always struggle with structural breaks.

---

## Phase 4 — EGARCH Structural Modifications

### Mod A: VIX-Augmented EGARCH (RMSE 0.329)
Added `δ·log(VIX_{t-1})` to the EGARCH variance equation as a new freely-estimated parameter.
**Failure:** Mean forecast = 0.856 %/day vs GARCH ~0.62. The VIX permanently trades at a premium over realized vol (the VIX risk premium). Adding raw log(VIX) inflates all forecasts systematically. The fix would be the VIX-RV spread, not raw VIX.
**Additional issue:** Custom Python MLE (scipy.optimize) vs arch library's C/Cython optimization — different convergence quality.

### Mod B: Realized EGARCH (RMSE 0.361, QLIKE 0.849)
Replaced the `|z_{t-1}| − E|z|` innovation with the Yang-Zhang RV.
**Failure:** QLIKE=0.849 vs GARCH's 0.122 — catastrophically miscalibrated. The lagged 21-day Yang-Zhang RV is a slow-moving, autocorrelated signal. When volatility changes quickly (spike then recovery), the model's "yesterday's RV" lags reality by 21 days and produces systematically wrong forecasts.

### Mod C: GJR-GARCH (RMSE 0.358)
Hard-threshold leverage: extra shock weight on negative returns.
**Failure:** Custom Python MLE could not match the convergence quality of the arch library. Mean forecast = 0.868, MaxJump = 1.93 %/day — significant estimation instability.
**Unresolved question:** Whether GJR-GARCH via the arch library would match or beat GARCH is untested. This is the one modification where the implementation quality may have distorted the result.

---

## Key Research Findings

### 1. GARCH(1,1) is the h=1 benchmark that cannot be beaten
Across 8 distinct model architectures and ~25 variants, no model beat GARCH(1,1) at h=1 on SPY. This is consistent with the academic literature (Poon & Granger 2003, Hansen & Lunde 2005 review of 330 ARCH models).

### 2. EGARCH wins at h=21 (leverage effect compounds)
At the 21-day horizon, EGARCH beats GARCH on MAE (0.298 vs 0.337, p=0.0003). Asymmetric leverage (`γ·z_{t-1}`) becomes meaningful over multi-week horizons when the direction of the initial shock matters for the mean-reversion path.

### 3. Structural complexity always hurts OOS
Every additional parameter or external signal (HMM, macro features, VIX loading, RV innovation) increased OOS RMSE relative to GARCH. This is a high-noise environment where estimation variance dominates bias reduction.

### 4. Macro features are incompatible with daily vol forecasting
Yield Curve and Credit Spread operate on 1-5 year business-cycle timescales. Forcing them into daily volatility models creates structural break sensitivity and chronological inconsistency. This invalidated both the HMM and the composite approaches.

### 5. The leverage effect is structurally robust
EGARCH's gamma parameter was **consistently negative in every single refit** across 120 monthly windows (mean = −0.202, range [−0.339, −0.173]). This confirms that SPY's leverage effect (negative returns → higher future vol) is a genuine structural property, not a sample artifact.

---

## What Comes Next: GARCH Improvement

Starting point: **GARCH(1,1) RMSE = 0.266, MAE = 0.177.**

The next phase will focus on improving the GARCH model itself, not replacing it. Approaches to test:
- ML residual correction (GARCH forecast + XGBoost error prediction)
- Intraday realized variance as a GARCH input
- VIX-informed GARCH (not EGARCH) via proper risk-premium adjustment
- HAR-RV as an alternative benchmark

---

*Files preserved: `egarch_vs_garch.py`, `dp_egarch_v2.py`, `dp_egarch_v4.py`, `forecasts.csv`, `params_egarch.csv`, `params_garch.csv`*  
*Files archived: everything else (see `archive/` folder)*
