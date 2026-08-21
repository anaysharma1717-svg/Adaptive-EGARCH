# Volatility Forecasting: Why Simple Models Win and Data Quality Dominates

**Author:** [Your Name]  
**Date:** July 2026  
**Focus:** SPY Daily Volatility Forecasting (2016–2026 Out-of-Sample)

## Executive Summary
This project evaluated whether modern adaptive architectures and Hidden Markov Models (HMMs) could outperform standard GARCH baselines for forecasting daily equity volatility. After testing over 20 variants—including macro-adaptive EGARCH and regime-switching models—the conclusion was stark: **complex parametric models consistently underperform simple baselines out-of-sample.**

The breakthrough came not from a better algorithm, but from better data. Replacing squared daily returns with the Garman-Klass intraday volatility proxy, and using the simple linear HAR-RV (Heterogeneous Autoregressive) model, resulted in a **26.5% reduction in RMSE** compared to GARCH(1,1).

This confirms a fundamental tenet of financial modeling: in high-noise environments, reducing measurement error in the target variable is vastly more effective than increasing model complexity.

---

## 1. The Baseline: GARCH(1,1) is Extremely Hard to Beat
The initial baseline was a standard GARCH(1,1) and EGARCH(1,1,1,t) evaluated using an expanding-window walk-forward protocol (monthly refits, no lookahead bias) from 2016 to 2026.
- **Result:** GARCH(1,1) proved incredibly robust for 1-day ahead forecasts (RMSE = 0.266 against a moving-average target).
- **Finding:** EGARCH's asymmetric leverage effect didn't beat GARCH at 1-day horizons, but became statistically significant at the 21-day horizon (p=0.0003), correctly capturing slower mean-reversion after market crashes.

## 2. The Failure of Complexity: Adaptive & HMM Approaches
I hypothesized that GARCH structural parameters ($\omega, \alpha, \beta$) should not be static, but should adapt to the macroeconomic environment (VIX, yield curve, credit spreads).
- **Adaptive EGARCH:** Modeled parameters as dynamic functions of macro features. 
  - *Result:* Failed. The optimizer chased local noise in every 21-day refit window. Adding temporal smoothness penalties stabilized the model, but it still underperformed GARCH(1,1) by 12%.
- **HMM Regime Switching:** Attempted to use a Gaussian HMM to classify the market into "Calm" and "Crisis" states based on macro variables.
  - *Result:* Failed due to structural breaks. The COVID-19 monetary expansion permanently shifted the baseline of macro features (e.g., credit spreads). The HMM interpreted this structural shift as a permanent "Crisis" regime, leading to 100% label stickiness post-2020. 

**Core Lesson:** Macroeconomic features operate on 2-5 year business cycle timescales. Forcing them into daily volatility models introduces severe structural break sensitivity.

## 3. The Breakthrough: HAR-RV and Data Quality
If complex parameterization fails, is the bottleneck the model architecture, or the data? Standard GARCH uses squared daily returns ($r_t^2$) as a proxy for variance, which is notoriously noisy. 

I replaced the GARCH models with the Heterogeneous Autoregressive (HAR-RV) model by Corsi (2009), which uses simple OLS regression on lagged daily, weekly, and monthly realized volatility:
$RV_{t+1} = \beta_0 + \beta_d RV_t + \beta_w RV_{t,w} + \beta_m RV_{t,m} + \epsilon$

Instead of squared close-to-close returns, I used the **Garman-Klass** estimator, which incorporates Open, High, Low, and Close prices to dramatically reduce measurement noise.

### Out-of-Sample Results (Target: Daily Garman-Klass Volatility)
| Model | RMSE | MAE |
|:---|:---:|:---:|
| GARCH(1,1) | 0.485 | 0.364 |
| EGARCH(1,1,1,t) | 0.452 | 0.352 |
| **HAR-RV** | **0.356** | **0.230** |

**Statistical Significance:** Diebold-Mariano test yields a p-value of 0.0000. HAR-RV definitively beats GARCH(1,1) by **26.5%**.

## Conclusion
This research project validates three critical principles for quantitative research:
1. **Bias-Variance Tradeoff:** In financial data, estimation variance usually dominates. Adding parameters (Adaptive EGARCH, HMMs) overfits the noise and degrades out-of-sample performance.
2. **Horizon Mismatch:** Macro data cannot reliably predict daily volatility.
3. **Data Quality > Model Complexity:** A simple linear regression (HAR) on high-quality intraday data (Garman-Klass) absolutely destroyed complex non-linear optimization (GARCH) on low-quality data (daily returns).
