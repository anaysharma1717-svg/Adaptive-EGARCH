"""
Time Series Diagnostics — ACF, PACF, Long-Memory & Lag Optimization
====================================================================
Purpose:
  1. ACF/PACF of Garman-Klass RV to understand autocorrelation structure
  2. ACF/PACF of HAR-RV residuals to check for leftover predictable structure
  3. Long-memory tests (Hurst exponent) — volatility is known to exhibit long memory
  4. Data-driven lag selection for an improved HAR-RV model
  5. ARMA model comparison informed by ACF/PACF

Author: Anay Sharma
Date: August 2026
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from scipy.stats import t as t_dist
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data(ticker="SPY", start="2011-01-01"):
    print(f"Fetching {ticker} data from {start}...")
    df = yf.download(ticker, start=start, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize()
    return df

def garman_klass_vol(df):
    """Garman-Klass (1980) daily volatility estimator using OHLC prices."""
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    gk_vol = np.sqrt(np.maximum(gk_var, 1e-8)) * 100.0
    return gk_vol.dropna()

def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))
def mae(a, b): return np.mean(np.abs(a - b))

def dm_test(e1, e2):
    """Diebold-Mariano test with HLN (1997) correction."""
    d = e1**2 - e2**2
    n = len(d)
    if n == 0: return np.nan, np.nan
    d_bar = d.mean()
    var_d = np.var(d, ddof=1) / n
    dm = d_bar / np.sqrt(var_d + 1e-12)
    hln = dm * np.sqrt((n + 1 - 2 + 1.0/n) / n)
    p = 2 * t_dist.sf(np.abs(hln), df=n-1)
    return round(hln, 4), round(p, 4)

def hurst_exponent(ts, max_lag=100):
    """
    Estimate the Hurst exponent via the R/S (rescaled range) method.
    H > 0.5 → long memory (persistent), H = 0.5 → random walk, H < 0.5 → mean-reverting.
    """
    ts = np.asarray(ts)
    lags = range(10, min(max_lag, len(ts) // 4))
    rs_values = []
    
    for lag in lags:
        # Divide into sub-series of length 'lag'
        n_subseries = len(ts) // lag
        rs_subseries = []
        
        for i in range(n_subseries):
            sub = ts[i * lag : (i + 1) * lag]
            mean_sub = np.mean(sub)
            deviations = np.cumsum(sub - mean_sub)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(sub, ddof=1)
            if s > 1e-10:
                rs_subseries.append(r / s)
        
        if rs_subseries:
            rs_values.append((lag, np.mean(rs_subseries)))
    
    if len(rs_values) < 5:
        return np.nan
    
    log_lags = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])
    
    # Linear regression: log(R/S) = H * log(n) + c
    coeffs = np.polyfit(log_lags, log_rs, 1)
    return coeffs[0]


def ljung_box_test(residuals, lags=20):
    """Ljung-Box Q-test for residual autocorrelation."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    result = acorr_ljungbox(residuals, lags=lags, return_df=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: ACF/PACF of the Raw Garman-Klass RV Series
# ─────────────────────────────────────────────────────────────────────────────

def analyze_rv_autocorrelation(rv_series):
    """Full autocorrelation analysis of the realized volatility series."""
    print("\n" + "=" * 70)
    print("  PART 1: Autocorrelation Structure of Garman-Klass RV")
    print("=" * 70)
    
    # Stationarity test
    adf_stat, adf_p, _, _, _, _ = adfuller(rv_series, maxlag=30)
    print(f"\n  ADF Test: stat = {adf_stat:.4f}, p = {adf_p:.6f}")
    print(f"  Stationary: {'YES' if adf_p < 0.05 else 'NO (consider differencing)'}")
    
    # Compute ACF and PACF
    n_lags = 60
    acf_vals = acf(rv_series, nlags=n_lags, fft=True)
    pacf_vals = pacf(rv_series, nlags=n_lags, method='ywm')
    
    # Print significant PACF lags (beyond 95% CI)
    ci_bound = 1.96 / np.sqrt(len(rv_series))
    
    print(f"\n  95% Confidence Bound: ±{ci_bound:.4f}")
    print(f"\n  Significant PACF Lags (|PACF| > {ci_bound:.4f}):")
    print(f"  {'Lag':>5}  {'PACF':>8}  {'Significance':>12}")
    print(f"  {'---':>5}  {'---':>8}  {'---':>12}")
    
    significant_lags = []
    for lag in range(1, n_lags + 1):
        if abs(pacf_vals[lag]) > ci_bound:
            sig = abs(pacf_vals[lag]) / ci_bound
            significant_lags.append((lag, pacf_vals[lag], sig))
            marker = "***" if sig > 3 else "**" if sig > 2 else "*"
            print(f"  {lag:>5}  {pacf_vals[lag]:>8.4f}  {sig:>8.1f}x CI {marker}")
    
    # Key HAR lags check
    print(f"\n  HAR-RV Standard Lags Check:")
    for lag in [1, 5, 22]:
        val = pacf_vals[lag] if lag <= n_lags else np.nan
        in_har = "[Y] USED" if lag in [1, 5, 22] else "[N] NOT USED"
        print(f"    Lag {lag:>2}: PACF = {val:>8.4f}  ({in_har})")
    
    # Identify top PACF lags NOT in HAR
    print(f"\n  Top PACF Lags NOT Captured by HAR-RV (1, 5, 22):")
    non_har = [(l, p, s) for l, p, s in significant_lags if l not in [1, 5, 22]]
    for lag, pval, sig in sorted(non_har, key=lambda x: -abs(x[1]))[:10]:
        print(f"    Lag {lag:>2}: PACF = {pval:>8.4f}  ({sig:.1f}x CI)")
    
    # ACF decay analysis — check for long memory
    print(f"\n  ACF Decay Analysis (Long Memory Check):")
    for lag in [1, 5, 10, 20, 40, 60]:
        if lag <= n_lags:
            print(f"    ACF({lag:>2}) = {acf_vals[lag]:.4f}")
    
    # Hurst exponent
    H = hurst_exponent(rv_series.values)
    print(f"\n  Hurst Exponent (R/S method): H = {H:.4f}")
    if H > 0.5:
        print(f"  → Long-memory process (H > 0.5). Volatility is persistent.")
        print(f"  → Fractional differencing parameter d ≈ {H - 0.5:.3f}")
        print(f"  → This suggests ARFIMA or HAR-type models are appropriate.")
    else:
        print(f"  → Short-memory / mean-reverting process.")
    
    return acf_vals, pacf_vals, significant_lags


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: ACF/PACF of HAR-RV Residuals
# ─────────────────────────────────────────────────────────────────────────────

def analyze_har_residuals(rv_series, har_df, base_df):
    """Check if HAR-RV residuals still contain exploitable autocorrelation."""
    print("\n" + "=" * 70)
    print("  PART 2: HAR-RV Residual Diagnostics")
    print("=" * 70)
    
    # Rebuild HAR-RV forecasts using the same walk-forward protocol
    refit_dates = pd.to_datetime(base_df['refit_date'].unique()).sort_values()
    common_dates = base_df.index.intersection(har_df.index)
    har_forecasts = pd.Series(index=common_dates, dtype=float)
    
    model = LinearRegression()
    for rdate in refit_dates:
        train_df = har_df[har_df.index < rdate]
        if len(train_df) < 100:
            continue
        X_train = train_df[['rv_1', 'rv_5', 'rv_22']].values
        y_train = train_df['rv'].values
        model.fit(X_train, y_train)
        
        test_mask = (base_df['refit_date'] == str(rdate.date())) | (base_df['refit_date'] == str(rdate))
        test_dates = base_df[test_mask].index.intersection(har_df.index)
        
        for tdate in test_dates:
            X_test = har_df.loc[tdate, ['rv_1', 'rv_5', 'rv_22']].values.reshape(1, -1)
            pred = model.predict(X_test)[0]
            har_forecasts.loc[tdate] = max(pred, 0.01)
    
    har_forecasts = har_forecasts.dropna()
    aligned = har_df.loc[har_forecasts.index]
    residuals = aligned['rv'].values - har_forecasts.values
    
    print(f"\n  Residual Statistics:")
    print(f"    N observations: {len(residuals)}")
    print(f"    Mean: {np.mean(residuals):.4f} (should be ~0)")
    print(f"    Std:  {np.std(residuals):.4f}")
    print(f"    Skew: {pd.Series(residuals).skew():.4f}")
    print(f"    Kurt: {pd.Series(residuals).kurtosis():.4f}")
    
    # ACF/PACF of residuals
    n_lags = 40
    res_acf = acf(residuals, nlags=n_lags, fft=True)
    res_pacf = pacf(residuals, nlags=n_lags, method='ywm')
    ci_bound = 1.96 / np.sqrt(len(residuals))
    
    print(f"\n  Significant Residual ACF Lags (leftover autocorrelation):")
    sig_acf = []
    for lag in range(1, n_lags + 1):
        if abs(res_acf[lag]) > ci_bound:
            sig = abs(res_acf[lag]) / ci_bound
            sig_acf.append((lag, res_acf[lag], sig))
    
    if sig_acf:
        for lag, val, sig in sig_acf[:15]:
            marker = "***" if sig > 3 else "**" if sig > 2 else "*"
            print(f"    ACF({lag:>2}) = {val:>8.4f}  ({sig:.1f}x CI) {marker}")
    else:
        print("    NONE — residuals are white noise (model is well-specified)")
    
    print(f"\n  Significant Residual PACF Lags (direct lag effects missed by HAR):")
    sig_pacf = []
    for lag in range(1, n_lags + 1):
        if abs(res_pacf[lag]) > ci_bound:
            sig = abs(res_pacf[lag]) / ci_bound
            sig_pacf.append((lag, res_pacf[lag], sig))
    
    if sig_pacf:
        for lag, val, sig in sig_pacf[:15]:
            marker = "***" if sig > 3 else "**" if sig > 2 else "*"
            print(f"    PACF({lag:>2}) = {val:>8.4f}  ({sig:.1f}x CI) {marker}")
    else:
        print("    NONE — no additional lag structure to exploit")
    
    # Ljung-Box test
    lb_results = ljung_box_test(residuals, lags=20)
    lb_20 = lb_results.iloc[-1]
    print(f"\n  Ljung-Box Q-Test (H0: no autocorrelation up to lag 20):")
    print(f"    Q-stat = {lb_20['lb_stat']:.2f}, p-value = {lb_20['lb_pvalue']:.6f}")
    if lb_20['lb_pvalue'] < 0.05:
        print(f"    → REJECT H0: Residuals have significant autocorrelation!")
        print(f"    → The HAR-RV model is MISSING predictable structure.")
    else:
        print(f"    → FAIL to reject H0: Residuals appear white-noise-like.")
    
    return residuals, har_forecasts, res_acf, res_pacf, sig_pacf


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Augmented HAR-RV with Data-Driven Lags
# ─────────────────────────────────────────────────────────────────────────────

def build_augmented_har(rv_series, har_df, base_df, extra_lags):
    """
    Build HAR-RV with additional lags informed by PACF analysis.
    Walk-forward protocol identical to har_rv_model.py.
    """
    if not extra_lags:
        print("\n  No significant extra lags found — skipping augmented model.")
        return None, None
    
    print(f"\n" + "=" * 70)
    print(f"  PART 3: Augmented HAR-RV with Extra Lags {extra_lags}")
    print("=" * 70)
    
    # Add extra lag features
    aug_df = har_df.copy()
    extra_cols = []
    for lag in extra_lags:
        col_name = f'rv_{lag}'
        if col_name not in aug_df.columns:
            aug_df[col_name] = rv_series.shift(lag)
            extra_cols.append(col_name)
        else:
            extra_cols.append(col_name)
    
    aug_df = aug_df.dropna()
    
    feature_cols = ['rv_1', 'rv_5', 'rv_22'] + extra_cols
    feature_cols = list(dict.fromkeys(feature_cols))  # deduplicate
    
    print(f"  Features: {feature_cols}")
    print(f"  Training samples available: {len(aug_df)}")
    
    # Walk-forward
    refit_dates = pd.to_datetime(base_df['refit_date'].unique()).sort_values()
    common_dates = base_df.index.intersection(aug_df.index)
    
    aug_forecasts = pd.Series(index=common_dates, dtype=float)
    har_forecasts = pd.Series(index=common_dates, dtype=float)
    
    model_aug = LinearRegression()
    model_har = LinearRegression()
    
    for rdate in refit_dates:
        train_aug = aug_df[aug_df.index < rdate]
        if len(train_aug) < 100:
            continue
        
        # Augmented model
        X_train_aug = train_aug[feature_cols].values
        y_train = train_aug['rv'].values
        model_aug.fit(X_train_aug, y_train)
        
        # Standard HAR for fair comparison
        X_train_har = train_aug[['rv_1', 'rv_5', 'rv_22']].values
        model_har.fit(X_train_har, y_train)
        
        test_mask = (base_df['refit_date'] == str(rdate.date())) | (base_df['refit_date'] == str(rdate))
        test_dates = base_df[test_mask].index.intersection(aug_df.index)
        
        for tdate in test_dates:
            X_aug = aug_df.loc[tdate, feature_cols].values.reshape(1, -1)
            X_har = aug_df.loc[tdate, ['rv_1', 'rv_5', 'rv_22']].values.reshape(1, -1)
            
            aug_forecasts.loc[tdate] = max(model_aug.predict(X_aug)[0], 0.01)
            har_forecasts.loc[tdate] = max(model_har.predict(X_har)[0], 0.01)
    
    aug_forecasts = aug_forecasts.dropna()
    har_forecasts = har_forecasts.loc[aug_forecasts.index].dropna()
    
    common = aug_forecasts.index.intersection(har_forecasts.index).intersection(aug_df.index)
    target = aug_df.loc[common, 'rv'].values
    aug_fc = aug_forecasts.loc[common].values
    har_fc = har_forecasts.loc[common].values
    
    aug_rmse = rmse(aug_fc, target)
    har_rmse = rmse(har_fc, target)
    aug_mae = mae(aug_fc, target)
    har_mae = mae(har_fc, target)
    
    err_aug = aug_fc - target
    err_har = har_fc - target
    dm_stat, dm_p = dm_test(err_aug, err_har)
    
    print(f"\n  Walk-Forward Results (N = {len(target)}):")
    print(f"  {'Model':<30} {'RMSE':>8} {'MAE':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    print(f"  {'HAR-RV (1, 5, 22)':<30} {har_rmse:>8.4f} {har_mae:>8.4f}")
    print(f"  {'HAR-RV + extra lags':<30} {aug_rmse:>8.4f} {aug_mae:>8.4f}")
    print(f"\n  DM test (Augmented vs Standard HAR):")
    print(f"    DM stat = {dm_stat}, p-value = {dm_p}")
    
    if aug_rmse < har_rmse:
        pct = (har_rmse - aug_rmse) / har_rmse * 100
        print(f"    → Augmented model WINS by {pct:.2f}%")
        if dm_p < 0.05:
            print(f"    → Statistically significant (p < 0.05)")
        else:
            print(f"    → NOT statistically significant")
    else:
        pct = (aug_rmse - har_rmse) / har_rmse * 100
        print(f"    → Standard HAR WINS by {pct:.2f}% — extra lags add noise, not signal")
    
    return aug_forecasts, har_forecasts


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: ARMA Model Comparison
# ─────────────────────────────────────────────────────────────────────────────

def fit_arma_models(rv_series, har_df, base_df):
    """
    Fit ARMA models with orders informed by ACF/PACF analysis.
    Walk-forward evaluation using identical protocol.
    """
    print("\n" + "=" * 70)
    print("  PART 4: ARMA Models (ACF/PACF-Informed Order Selection)")
    print("=" * 70)
    
    from statsmodels.tsa.arima.model import ARIMA
    
    # Candidate ARMA orders based on typical PACF/ACF patterns for vol
    candidates = [
        (1, 0, 0),  # AR(1)
        (2, 0, 0),  # AR(2)
        (1, 0, 1),  # ARMA(1,1)
        (2, 0, 1),  # ARMA(2,1)
        (3, 0, 1),  # ARMA(3,1) 
        (5, 0, 1),  # ARMA(5,1) — motivated by weekly PACF
    ]
    
    refit_dates = pd.to_datetime(base_df['refit_date'].unique()).sort_values()
    common_dates = base_df.index.intersection(har_df.index)
    
    results = {}
    
    for order in candidates:
        label = f"ARMA({order[0]},{order[2]})"
        print(f"\n  Fitting {label} (walk-forward)...", end="", flush=True)
        
        forecasts = pd.Series(index=common_dates, dtype=float)
        n_failed = 0
        
        for rdate in refit_dates:
            train_rv = rv_series[rv_series.index < rdate]
            if len(train_rv) < 200:
                continue
            
            try:
                model = ARIMA(train_rv.values[-1000:], order=order)  # Use last 1000 obs for speed
                fit = model.fit()
                pred = fit.forecast(steps=1)[0]
            except Exception:
                n_failed += 1
                continue
            
            test_mask = (base_df['refit_date'] == str(rdate.date())) | (base_df['refit_date'] == str(rdate))
            test_dates = base_df[test_mask].index.intersection(har_df.index)
            
            # Use same forecast for all days until next refit (the ARMA updates daily via recursion,
            # but for fair comparison with monthly-refit HAR, we use refit-level forecasts)
            for tdate in test_dates:
                # For each test date, re-forecast from that date's position
                idx = rv_series.index.get_loc(tdate)
                if idx > 0:
                    try:
                        # Use the fitted model's parameters to forecast from the latest data
                        recent = rv_series.iloc[max(0, idx-1000):idx].values
                        model_t = ARIMA(recent, order=order)
                        fit_t = model_t.fit()
                        forecasts.loc[tdate] = max(fit_t.forecast(steps=1)[0], 0.01)
                    except Exception:
                        n_failed += 1
                        continue
        
        forecasts = forecasts.dropna()
        if len(forecasts) < 100:
            print(f" SKIPPED (only {len(forecasts)} valid forecasts, {n_failed} failures)")
            continue
        
        aligned = har_df.loc[forecasts.index]
        target = aligned['rv'].values
        fc = forecasts.values
        
        arma_rmse = rmse(fc, target)
        arma_mae = mae(fc, target)
        
        results[label] = {
            'rmse': arma_rmse,
            'mae': arma_mae,
            'n': len(target),
            'failed': n_failed,
            'order': order,
            'forecasts': forecasts
        }
        print(f" RMSE={arma_rmse:.4f}, MAE={arma_mae:.4f} (N={len(target)}, {n_failed} failures)")
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Part 5: Generate All Plots
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(rv_series, acf_vals, pacf_vals, residuals, res_acf, res_pacf):
    """Generate comprehensive diagnostic plots."""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Time Series Diagnostics — SPY Garman-Klass Realized Volatility", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    ci_bound_rv = 1.96 / np.sqrt(len(rv_series))
    ci_bound_res = 1.96 / np.sqrt(len(residuals))
    n_lags_rv = len(acf_vals) - 1
    n_lags_res = len(res_acf) - 1
    
    # ── Row 1: RV Series ACF / PACF ──
    ax1 = axes[0, 0]
    lags = np.arange(1, n_lags_rv + 1)
    ax1.bar(lags, acf_vals[1:], color='steelblue', alpha=0.7, width=0.8)
    ax1.axhline(ci_bound_rv, color='red', linestyle='--', alpha=0.5, label='95% CI')
    ax1.axhline(-ci_bound_rv, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(0, color='black', linewidth=0.5)
    # Highlight HAR lags
    for hl in [1, 5, 22]:
        if hl <= n_lags_rv:
            ax1.bar(hl, acf_vals[hl], color='darkorange', alpha=0.9, width=0.8)
    ax1.set_title("ACF — Garman-Klass RV", fontweight='bold')
    ax1.set_xlabel("Lag (days)")
    ax1.set_ylabel("Autocorrelation")
    ax1.legend(fontsize=8)
    
    ax2 = axes[0, 1]
    ax2.bar(lags, pacf_vals[1:], color='steelblue', alpha=0.7, width=0.8)
    ax2.axhline(ci_bound_rv, color='red', linestyle='--', alpha=0.5, label='95% CI')
    ax2.axhline(-ci_bound_rv, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(0, color='black', linewidth=0.5)
    for hl in [1, 5, 22]:
        if hl <= n_lags_rv:
            ax2.bar(hl, pacf_vals[hl], color='darkorange', alpha=0.9, width=0.8)
    ax2.set_title("PACF — Garman-Klass RV", fontweight='bold')
    ax2.set_xlabel("Lag (days)")
    ax2.set_ylabel("Partial Autocorrelation")
    ax2.legend(fontsize=8)
    ax2.annotate("Orange = HAR lags (1, 5, 22)", xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=8, color='darkorange', fontweight='bold')
    
    # ── Row 2: HAR-RV Residual ACF / PACF ──
    ax3 = axes[1, 0]
    res_lags = np.arange(1, n_lags_res + 1)
    colors_acf = ['crimson' if abs(res_acf[i+1]) > ci_bound_res else 'steelblue' 
                  for i in range(n_lags_res)]
    ax3.bar(res_lags, res_acf[1:], color=colors_acf, alpha=0.7, width=0.8)
    ax3.axhline(ci_bound_res, color='red', linestyle='--', alpha=0.5, label='95% CI')
    ax3.axhline(-ci_bound_res, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_title("ACF — HAR-RV Residuals (red = significant)", fontweight='bold')
    ax3.set_xlabel("Lag (days)")
    ax3.set_ylabel("Autocorrelation")
    ax3.legend(fontsize=8)
    
    ax4 = axes[1, 1]
    colors_pacf = ['crimson' if abs(res_pacf[i+1]) > ci_bound_res else 'steelblue' 
                   for i in range(n_lags_res)]
    ax4.bar(res_lags, res_pacf[1:], color=colors_pacf, alpha=0.7, width=0.8)
    ax4.axhline(ci_bound_res, color='red', linestyle='--', alpha=0.5, label='95% CI')
    ax4.axhline(-ci_bound_res, color='red', linestyle='--', alpha=0.5)
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_title("PACF — HAR-RV Residuals (red = significant)", fontweight='bold')
    ax4.set_xlabel("Lag (days)")
    ax4.set_ylabel("Partial Autocorrelation")
    ax4.legend(fontsize=8)
    
    # ── Row 3: ACF Log-Log Decay (Long Memory) + Residual Distribution ──
    ax5 = axes[2, 0]
    lags_for_decay = np.arange(1, n_lags_rv + 1)
    ax5.plot(np.log(lags_for_decay), np.log(np.maximum(acf_vals[1:], 1e-6)), 
             'o-', color='steelblue', markersize=3, alpha=0.7, label='ACF')
    # Fit linear trend (long memory → slow linear decay in log-log)
    valid = acf_vals[1:] > 0
    if np.sum(valid) > 5:
        x_valid = np.log(lags_for_decay[valid])
        y_valid = np.log(acf_vals[1:][valid])
        slope, intercept = np.polyfit(x_valid, y_valid, 1)
        ax5.plot(x_valid, slope * x_valid + intercept, '--', color='red', 
                 label=f'Decay slope = {slope:.3f}', linewidth=2)
        ax5.annotate(f"d ≈ {-slope/2:.3f} (long memory)", xy=(0.5, 0.05), 
                     xycoords='axes fraction', fontsize=9, color='red', fontweight='bold')
    ax5.set_title("Log-Log ACF Decay (Long Memory Test)", fontweight='bold')
    ax5.set_xlabel("log(Lag)")
    ax5.set_ylabel("log(ACF)")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)
    
    ax6 = axes[2, 1]
    ax6.hist(residuals, bins=80, color='steelblue', alpha=0.7, density=True, edgecolor='white')
    from scipy.stats import norm
    x_norm = np.linspace(np.min(residuals), np.max(residuals), 200)
    ax6.plot(x_norm, norm.pdf(x_norm, np.mean(residuals), np.std(residuals)), 
             'r-', linewidth=2, label='Normal fit')
    ax6.set_title("HAR-RV Residual Distribution", fontweight='bold')
    ax6.set_xlabel("Residual")
    ax6.set_ylabel("Density")
    ax6.legend(fontsize=8)
    
    skew = pd.Series(residuals).skew()
    kurt = pd.Series(residuals).kurtosis()
    ax6.annotate(f"Skew={skew:.2f}, Kurt={kurt:.2f}", xy=(0.95, 0.95), 
                 xycoords='axes fraction', ha='right', va='top', fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('plots/ts_diagnostics.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: plots/ts_diagnostics.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Part 6: Summary Report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(H, significant_lags, sig_pacf_res, arma_results, har_rmse_baseline):
    """Print final summary and recommendations."""
    print("\n" + "=" * 70)
    print("  SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"\n  1. LONG MEMORY:")
    print(f"     Hurst exponent H = {H:.4f} → d ≈ {H - 0.5:.3f}")
    if H > 0.6:
        print(f"     → Strong long memory. ACF decays hyperbolically, not exponentially.")
        print(f"     → HAR-RV's multi-scale averaging (1,5,22) is a good approximation")
        print(f"       of the true long-memory structure.")
    
    print(f"\n  2. PACF INSIGHTS (Raw RV):")
    top_non_har = [(l, p) for l, p, s in significant_lags if l not in [1, 5, 22]][:5]
    if top_non_har:
        print(f"     Significant lags NOT in HAR-RV: {[l for l, p in top_non_har]}")
        print(f"     → These represent direct lag effects the HAR model ignores.")
    else:
        print(f"     → All significant lags are captured by HAR-RV (1, 5, 22).")
    
    print(f"\n  3. RESIDUAL ANALYSIS:")
    if sig_pacf_res:
        missed = [l for l, v, s in sig_pacf_res][:5]
        print(f"     HAR-RV residuals have significant PACF at lags: {missed}")
        print(f"     → There IS leftover predictable structure in the residuals.")
    else:
        print(f"     → HAR-RV residuals are white noise. No additional structure to exploit.")
    
    print(f"\n  4. ARMA COMPARISON (vs HAR-RV RMSE ≈ {har_rmse_baseline:.4f}):")
    if arma_results:
        best_arma = min(arma_results.items(), key=lambda x: x[1]['rmse'])
        print(f"     Best ARMA: {best_arma[0]} → RMSE = {best_arma[1]['rmse']:.4f}")
        if best_arma[1]['rmse'] < har_rmse_baseline:
            pct = (har_rmse_baseline - best_arma[1]['rmse']) / har_rmse_baseline * 100
            print(f"     → {best_arma[0]} BEATS HAR-RV by {pct:.2f}%!")
        else:
            pct = (best_arma[1]['rmse'] - har_rmse_baseline) / har_rmse_baseline * 100
            print(f"     → HAR-RV still wins by {pct:.2f}%.")
    
    print(f"\n  5. BOTTOM LINE:")
    print(f"     ACF/PACF analysis reveals whether the HAR-RV lag structure is optimal")
    print(f"     and whether ARMA/ARFIMA models can extract additional predictive signal")
    print(f"     from the volatility series' autocorrelation structure.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TIME SERIES DIAGNOSTICS — ACF, PACF, Long-Memory Analysis")
    print("  Target: SPY Garman-Klass Realized Volatility")
    print("=" * 70)
    
    # 1. Load data
    df_spy = fetch_data(start="2011-01-01")
    rv_series = garman_klass_vol(df_spy)
    
    # Build HAR features
    har_df = pd.DataFrame({'rv': rv_series})
    har_df['rv_1'] = har_df['rv'].shift(1)
    har_df['rv_5'] = har_df['rv_1'].rolling(5).mean()
    har_df['rv_22'] = har_df['rv_1'].rolling(22).mean()
    har_df = har_df.dropna()
    
    # Load baseline data
    base_df = pd.read_csv('data/forecasts.csv', index_col='date', parse_dates=True)
    
    # 2. Part 1: ACF/PACF of raw RV series
    acf_vals, pacf_vals, significant_lags = analyze_rv_autocorrelation(rv_series)
    
    # Hurst for later
    H = hurst_exponent(rv_series.values)
    
    # 3. Part 2: HAR-RV Residual diagnostics
    residuals, har_fc, res_acf, res_pacf, sig_pacf_res = analyze_har_residuals(
        rv_series, har_df, base_df
    )
    
    # 4. Part 3: Augmented HAR with data-driven lags
    # Use top PACF lags from residual analysis that aren't already in HAR
    extra_lags = [l for l, v, s in sig_pacf_res if l not in [1, 5, 22] and s > 2.0][:5]
    if extra_lags:
        print(f"\n  Data-driven extra lags from residual PACF: {extra_lags}")
        aug_fc, har_fc_compare = build_augmented_har(rv_series, har_df, base_df, extra_lags)
    else:
        # Fall back to raw RV PACF-driven lags
        extra_lags_raw = [l for l, v, s in significant_lags if l not in [1, 5, 22] and s > 3.0][:5]
        if extra_lags_raw:
            print(f"\n  Trying top PACF lags from raw RV: {extra_lags_raw}")
            aug_fc, har_fc_compare = build_augmented_har(rv_series, har_df, base_df, extra_lags_raw)
        else:
            aug_fc, har_fc_compare = None, None
    
    # 5. Part 4: ARMA model comparison
    arma_results = fit_arma_models(rv_series, har_df, base_df)
    
    # HAR-RV baseline RMSE
    har_aligned = har_df.loc[pd.Series(har_fc, dtype=float).dropna().index if isinstance(har_fc, pd.Series) else []]
    # Compute HAR baseline from Part 2
    if isinstance(har_fc, pd.Series):
        har_common = har_fc.dropna().index.intersection(har_df.index)
        har_baseline_rmse = rmse(har_fc.loc[har_common].values, har_df.loc[har_common, 'rv'].values)
    else:
        har_baseline_rmse = 0.356  # From previous results
    
    # 6. Generate plots
    generate_plots(rv_series, acf_vals, pacf_vals, residuals, res_acf, res_pacf)
    
    # 7. Summary
    print_summary(H, significant_lags, sig_pacf_res, arma_results, har_baseline_rmse)


if __name__ == '__main__':
    main()
