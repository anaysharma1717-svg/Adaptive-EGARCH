"""Quick script to generate just the diagnostic plots (ACF/PACF)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf, pacf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
print("Fetching data...")
df = yf.download("SPY", start="2011-01-01", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.index = pd.to_datetime(df.index).normalize()

# Garman-Klass
log_hl = np.log(df['High'] / df['Low'])
log_co = np.log(df['Close'] / df['Open'])
gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
rv_series = (np.sqrt(np.maximum(gk_var, 1e-8)) * 100.0).dropna()

# HAR features
har_df = pd.DataFrame({'rv': rv_series})
har_df['rv_1'] = har_df['rv'].shift(1)
har_df['rv_5'] = har_df['rv_1'].rolling(5).mean()
har_df['rv_22'] = har_df['rv_1'].rolling(22).mean()
har_df = har_df.dropna()

# HAR-RV walk-forward residuals
base_df = pd.read_csv('data/forecasts.csv', index_col='date', parse_dates=True)
common_dates = base_df.index.intersection(har_df.index)
base_df = base_df.loc[common_dates]
refit_dates = pd.to_datetime(base_df['refit_date'].unique()).sort_values()

har_forecasts = pd.Series(index=common_dates, dtype=float)
model = LinearRegression()
for rdate in refit_dates:
    train_df = har_df[har_df.index < rdate]
    if len(train_df) < 100:
        continue
    model.fit(train_df[['rv_1', 'rv_5', 'rv_22']].values, train_df['rv'].values)
    test_mask = (base_df['refit_date'] == str(rdate.date())) | (base_df['refit_date'] == str(rdate))
    test_dates = base_df[test_mask].index.intersection(har_df.index)
    for tdate in test_dates:
        X = har_df.loc[tdate, ['rv_1', 'rv_5', 'rv_22']].values.reshape(1, -1)
        har_forecasts.loc[tdate] = max(model.predict(X)[0], 0.01)

har_forecasts = har_forecasts.dropna()
aligned = har_df.loc[har_forecasts.index]
residuals = aligned['rv'].values - har_forecasts.values

# Compute ACF/PACF
n_lags_rv = 60
n_lags_res = 40
acf_vals = acf(rv_series, nlags=n_lags_rv, fft=True)
pacf_vals = pacf(rv_series, nlags=n_lags_rv, method='ywm')
res_acf = acf(residuals, nlags=n_lags_res, fft=True)
res_pacf = pacf(residuals, nlags=n_lags_res, method='ywm')

ci_bound_rv = 1.96 / np.sqrt(len(rv_series))
ci_bound_res = 1.96 / np.sqrt(len(residuals))

print("Generating plots...")

# ── MAIN 6-PANEL DIAGNOSTIC PLOT ──
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Time Series Diagnostics - SPY Garman-Klass Realized Volatility", 
             fontsize=14, fontweight='bold', y=0.98)

# Row 1: RV ACF / PACF
ax1 = axes[0, 0]
lags = np.arange(1, n_lags_rv + 1)
ax1.bar(lags, acf_vals[1:], color='steelblue', alpha=0.7, width=0.8)
ax1.axhline(ci_bound_rv, color='red', linestyle='--', alpha=0.5, label='95% CI')
ax1.axhline(-ci_bound_rv, color='red', linestyle='--', alpha=0.5)
ax1.axhline(0, color='black', linewidth=0.5)
for hl in [1, 5, 22]:
    ax1.bar(hl, acf_vals[hl], color='darkorange', alpha=0.9, width=0.8)
ax1.set_title("ACF - Garman-Klass RV", fontweight='bold')
ax1.set_xlabel("Lag (days)")
ax1.set_ylabel("Autocorrelation")
ax1.legend(fontsize=8)

ax2 = axes[0, 1]
ax2.bar(lags, pacf_vals[1:], color='steelblue', alpha=0.7, width=0.8)
ax2.axhline(ci_bound_rv, color='red', linestyle='--', alpha=0.5, label='95% CI')
ax2.axhline(-ci_bound_rv, color='red', linestyle='--', alpha=0.5)
ax2.axhline(0, color='black', linewidth=0.5)
for hl in [1, 5, 22]:
    ax2.bar(hl, pacf_vals[hl], color='darkorange', alpha=0.9, width=0.8)
# Highlight lag 2 (the big missed one)
ax2.bar(2, pacf_vals[2], color='crimson', alpha=0.9, width=0.8)
ax2.set_title("PACF - Garman-Klass RV", fontweight='bold')
ax2.set_xlabel("Lag (days)")
ax2.set_ylabel("Partial Autocorrelation")
ax2.legend(fontsize=8)
ax2.annotate("Orange = HAR lags (1,5,22)\nRed = Missed lag 2 (PACF=0.30)", 
             xy=(0.95, 0.95), xycoords='axes fraction',
             ha='right', va='top', fontsize=8, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Row 2: Residual ACF / PACF
ax3 = axes[1, 0]
res_lags = np.arange(1, n_lags_res + 1)
colors_acf = ['crimson' if abs(res_acf[i+1]) > ci_bound_res else 'steelblue' 
              for i in range(n_lags_res)]
ax3.bar(res_lags, res_acf[1:], color=colors_acf, alpha=0.7, width=0.8)
ax3.axhline(ci_bound_res, color='red', linestyle='--', alpha=0.5, label='95% CI')
ax3.axhline(-ci_bound_res, color='red', linestyle='--', alpha=0.5)
ax3.axhline(0, color='black', linewidth=0.5)
ax3.set_title("ACF - HAR-RV Residuals (red = significant)", fontweight='bold')
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
ax4.set_title("PACF - HAR-RV Residuals (red = significant)", fontweight='bold')
ax4.set_xlabel("Lag (days)")
ax4.set_ylabel("Partial Autocorrelation")
ax4.legend(fontsize=8)

# Row 3: Log-Log ACF Decay + Residual Distribution
ax5 = axes[2, 0]
lags_for_decay = np.arange(1, n_lags_rv + 1)
ax5.plot(np.log(lags_for_decay), np.log(np.maximum(acf_vals[1:], 1e-6)), 
         'o-', color='steelblue', markersize=3, alpha=0.7, label='ACF')
valid = acf_vals[1:] > 0
if np.sum(valid) > 5:
    x_valid = np.log(lags_for_decay[valid])
    y_valid = np.log(acf_vals[1:][valid])
    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    ax5.plot(x_valid, slope * x_valid + intercept, '--', color='red', 
             label=f'Decay slope = {slope:.3f}', linewidth=2)
    ax5.annotate(f"d = {-slope/2:.3f} (long memory)", xy=(0.5, 0.05), 
                 xycoords='axes fraction', fontsize=9, color='red', fontweight='bold')
ax5.set_title("Log-Log ACF Decay (Long Memory Test)", fontweight='bold')
ax5.set_xlabel("log(Lag)")
ax5.set_ylabel("log(ACF)")
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

ax6 = axes[2, 1]
ax6.hist(residuals, bins=80, color='steelblue', alpha=0.7, density=True, edgecolor='white')
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
print("Saved: plots/ts_diagnostics.png")
plt.close()

# ── SCOREBOARD BAR CHART ──
fig2, ax = plt.subplots(figsize=(10, 6))
models = ['ARMA(1,0)', 'AR(2)', 'HAR-RV\n(1,5,22)', 'ARMA(2,1)', 'ARMA(1,1)', 'HAR-RV\n+ lag 2']
rmses =  [0.3783,      0.3578,  0.3564,          0.3547,     0.3535,     0.3533]
colors = ['#95a5a6',   '#95a5a6', '#3498db',     '#95a5a6',  '#e74c3c',  '#2ecc71']

bars = ax.barh(models, rmses, color=colors, edgecolor='white', height=0.6)
ax.set_xlabel("RMSE (lower is better)", fontsize=12)
ax.set_title("Model Comparison: ACF/PACF-Informed Models vs HAR-RV Baseline", 
             fontsize=13, fontweight='bold')
ax.axvline(0.3564, color='#3498db', linestyle='--', alpha=0.5, label='HAR-RV baseline')

for bar, val in zip(bars, rmses):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
            va='center', fontsize=10, fontweight='bold')

ax.set_xlim(0.345, 0.39)
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.2)
plt.tight_layout()
plt.savefig('plots/ts_model_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: plots/ts_model_comparison.png")
plt.close()

print("Done!")
