# Regime Detection Module — Design History

> **Status:** HMM module deprecated. Replaced by stationary percentile-composite stress score.
> **Date:** 2026-07-10
> **Files:** `hmm_final_study.py` (archived), `composite_regime_score.py` (active)

---

## 1. Why the HMM Was Originally Chosen

The Adaptive EGARCH project requires a regime signal `s_t ∈ [0, 1]` that:
1. Conditions the EGARCH `omega_t` parameter on market stress
2. Distinguishes structurally different market environments (calm vs. crisis)
3. Produces a continuous probability rather than a hard threshold

A Hidden Markov Model was the natural first choice because:
- It models latent state transitions (Markov property matches volatility regime persistence)
- Gaussian emissions provide a principled probabilistic framework
- The `hmmlearn` library provides a clean Python implementation
- It is well-established in quantitative finance literature (Hamilton 1989, Ang & Timmermann 2012)

The initial design used a single feature (`log VIX`) with 2 states (calm/crisis). Later studies added 5 additional economically distinct features: VIX Term Structure, Credit Spread (`log HYG/LQD`), Yield Curve (10Y − IRX), Volume Z-score, and 21-day SPY momentum.

---

## 2. What Worked

| Result | Finding |
|:---|:---|
| Crisis detection (acute events) | The HMM reliably identified COVID (VIX > 30), 2018 Volmageddon, and the 2025 Tariff Shock with P(Crisis) → 1.0 |
| 6-feature multi-dimensional view | Full-covariance captured cross-feature correlations (Credit ↔ YC during hiking cycle) that diagonal models miss |
| AIC/BIC model selection | 2-state consistently won over 3-state on information criteria |
| Transition matrix | Learned state persistence (avg crisis duration ~65 days) economically reasonable |

---

## 3. What Failed — Seven Architecture Variants Tested

### 3.1 Expanding-Window, Cold-Restart
**Failure:** 85% of 56 refits required a non-identity state permutation (Hungarian algorithm diagnostic).
**Root cause:** EM's likelihood objective is indifferent to labelling. Cold-restarts find different local optima, randomly permuting states.

### 3.2 Expanding-Window, Warm-Start
**Failure:** 0% label-switching, but P(Crisis) permanently pinned to 1.0 from March 2020 onward.
**Root cause:** EM is a local optimizer. COVID-scale structural breaks push warm-started parameters into a deep local minimum they never escape.

### 3.3 Frozen 3-State, Global Z-Score
**Failure:** EM split "Calm" into two era-specific sub-states (steep vs. flat curve) rather than learning Calm/Transition/Crisis.
**Root cause:** 2022-2023 inversion was only ~15% of training data. EM found it more useful to model 2011-2014 vs 2015-2019 macro differences.

### 3.4 Frozen 2-State, Global Z-Score
**Failures:** (1) COVID stickiness — P(Crisis) never recovered below 0.5 after March 2020. (2) 2021 bull market classified as Crisis (P(Calm) = 0.019).
**Root cause:** After COVID, macro features (YC, Credit) shifted to levels not seen during pre-COVID "Calm" periods. The Gaussian Calm state's mean vector pointed to pre-COVID baselines; 2021 observations fell outside the Calm distribution despite low VIX.

### 3.5 Frozen 3-State, Rolling-Relative Z-Score
**Failure:** Learned state means economically inverted (Calm state had wide credit spreads; Crisis had tight spreads). COVID stickiness persisted.
**Root cause:** The 3-year rolling window includes the COVID spike in its lookback throughout 2020-2023, contaminating relative z-scores for the entire post-COVID recovery.

### 3.6 Frozen 2-State, Rolling-Relative Z-Score
**Failure:** Same COVID stickiness and 2021 misclassification as 3.4.
**Root cause:** Rolling-relative z-scores introduce their own structural break at COVID: post-COVID "normal" looks abnormal relative to COVID-inflated rolling baseline.

### 3.7 Universal Root Cause
A Gaussian HMM learns an **absolute mean vector** per state during training. When macro baselines shift permanently (COVID-era monetary expansion), the trained mean vectors no longer match any post-break "Calm" observation. The model has no mechanism to separate "low VIX but unusual macro baseline" from "Crisis." This is a fundamental parametric family limitation — not fixable via feature engineering or preprocessing.

---

## 4. Why the Stationary Percentile-Composite Is More Robust

### Non-parametric and distribution-free
Each feature is transformed to its empirical CDF rank within a 3-year trailing window:
- A yield curve at −1.0 in 2023 is correctly evaluated as "flat relative to recent 3-year history" 
- The same value in 2010 would be "extremely flat relative to its own 3-year history"
- No absolute baseline is ever used — immune to structural breaks in levels

### No optimization, no local optima, no label-switching
```
stress_t = mean( RV_pct_t, CreditSpread_pct_t, YC_pct_t, VRP_pct_t, Mom_pct_t )
```
Deterministic, interpretable, reproducible across any random seed.

### Stationary by construction
Rolling percentile ranks are bounded to [0, 1] and mean-revert to 0.5 by construction. ADF tests confirm stationarity for all retained features (p < 0.01 for all).

### Computationally O(n) vs O(n²)
HMM causal OOS filtering required `predict_proba(X[:t+1])` for each day t — quadratic time. The composite requires one rolling window pass per feature — linear time.

---

## 5. Why the EGARCH Regression Structure Was Unchanged

`dp_egarch_v5.py` uses the regime signal only to condition `omega_t`:
```python
omega_t = -0.05 + 0.10 * sigmoid(w_omega @ [1, stress_score_t])
```

Keeping the EGARCH regression identical to the HMM version:
1. Creates a **controlled A/B comparison** — any forecast improvement is attributable solely to the regime signal
2. Preserves all existing validation infrastructure (`egarch_evaluation.py`, `forecasts.csv`, `params_egarch.csv`)
3. Avoids overfitting risks from introducing new EGARCH parameters alongside the signal change

---

## 6. Files

| File | Status | Purpose |
|:---|:---|:---|
| `hmm_final_study.py` | Archived | Original 3-state expanding-window HMM |
| `hmm_label_switching.py` | Archived | Hungarian-algorithm label-switching diagnostic |
| `hmm_warmstart_test.py` | Archived | Warm vs. cold restart mismatch comparison |
| `hmm_frozen_study.py` | Archived | 3-state frozen, causal OOS filtering |
| `hmm_2state_frozen_test.py` | Archived | 2-state frozen validation |
| `hmm_rolling_relative.py` | Archived | Rolling-relative standardization attempt |
| `hmm_validation.py` | Archived | 4-part stress test suite |
| `composite_regime_score.py` | **Active** | Stationary percentile-composite (Steps 1–5) |
| `dp_egarch_v5.py` | Pending | Will replace `prob_crisis` with `stress_score` |
| `data/composite_regime_score.csv` | **Active** | Daily composite scores for EGARCH |

---

## 7. Evidence Chain

The design decisions were strictly empirical:

1. HMM chosen (theoretically principled for latent regime detection)
2. 3-state tested (AIC/BIC, economic reasoning)
3. Label-switching quantified (Hungarian diagnostic, not assumed)
4. Warm-start tested before abandoning HMM
5. Frozen architecture tested to eliminate refit instability
6. Rolling-relative standardization tested as targeted fix
7. Composite adopted only after exhausting all HMM remediation options, with **identical** validation checks on both

> This document preserves the full research trail. The portfolio shows an evidence-driven, iterative design process — not a silent model swap.
