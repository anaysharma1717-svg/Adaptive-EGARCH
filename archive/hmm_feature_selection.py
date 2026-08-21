"""
hmm_feature_selection.py  --  Step 1: Feature selection before multi-feature HMM

Goal: Find 4-6 features from DIFFERENT economic dimensions that are
      genuinely independent, to discover true market regimes.

Features evaluated (7 candidates, 6 dimensions):
  1. log(VIX)                      → Market fear / realized vol expectation
  2. VIX_TERM = VIX9D - VIX3M     → Volatility term structure (contango/backwardation)
  3. CREDIT  = log(HYG/LQD)       → Credit risk premium
  4. YIELD_CURVE = 10Y - 2Y yield  → Macro/recession indicator
  5. VOL_ZSCORE  = rolling z-score of SPY log-volume → Liquidity / market activity
  6. MOMENTUM    = SPY 1M log-return (21-day)         → Trend / market direction
  7. GAP = SPY 1-day momentum (detrended)              → Short-term sentiment

Selection process:
  a) Compute pairwise correlation matrix
  b) Compute VIF for each feature
  c) Drop features with |corr| > 0.7 or VIF > 5 (keep lower-VIF one)
  d) Print recommendation

Outputs:
  - plots/feature_correlation_matrix.png
  - data/hmm_candidate_features.csv
  - Printed correlation + VIF table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

warnings.filterwarnings("ignore")
os.makedirs('plots', exist_ok=True)
os.makedirs('data',  exist_ok=True)

START = '2010-01-01'
END   = '2026-07-01'

# ─────────────────────────────────────────────────────────────────────────────
# 1. Download raw data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  DOWNLOADING RAW DATA")
print("=" * 70)

def dl(ticker):
    d = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    return d['Close'].squeeze().rename(ticker)

vix   = dl('^VIX')
vix9d = dl('^VIX9D')
vix3m = dl('^VIX3M')
hyg   = dl('HYG')
lqd   = dl('LQD')
spy   = dl('SPY')
t10y  = dl('^TNX')   # 10-year Treasury yield
t2y   = dl('^IRX')   # 2-year proxy (^IRX = 13-week, we'll note this)

print(f"  VIX  : {len(vix)}  rows  ({vix.index[0].date()} → {vix.index[-1].date()})")
print(f"  VIX9D: {len(vix9d)} rows")
print(f"  VIX3M: {len(vix3m)} rows")
print(f"  HYG  : {len(hyg)}  rows")
print(f"  LQD  : {len(lqd)}  rows")
print(f"  SPY  : {len(spy)}  rows")
print(f"  10Y  : {len(t10y)} rows")
print(f"  IRX  : {len(t2y)}  rows (13-week, used as short-rate proxy)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build feature DataFrame (all on common dates)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  BUILDING FEATURES")
print("=" * 70)

raw = pd.DataFrame({
    'VIX':   vix,
    'VIX9D': vix9d,
    'VIX3M': vix3m,
    'HYG':   hyg,
    'LQD':   lqd,
    'SPY':   spy,
    'T10Y':  t10y,
    'IRX':   t2y,
}).dropna()

print(f"  Common rows after dropna: {len(raw)} ({raw.index[0].date()} → {raw.index[-1].date()})")

# ── Feature 1: log(VIX) ──────────────────────────────────────────────────────
raw['F1_logVIX'] = np.log(raw['VIX'])
print("  F1: log(VIX) [Market fear]")

# ── Feature 2: VIX term structure ────────────────────────────────────────────
# VIX9D - VIX3M: positive = backwardation (fear spike), negative = contango (calm)
raw['F2_VIX_TERM'] = raw['VIX9D'] - raw['VIX3M']
print("  F2: VIX9D - VIX3M [Volatility term structure]")

# ── Feature 3: Credit spread proxy ───────────────────────────────────────────
# log(HYG/LQD): lower ratio = wider credit spread = more credit stress
raw['F3_CREDIT'] = np.log(raw['HYG'] / raw['LQD'])
print("  F3: log(HYG/LQD) [Credit risk, inverse of spread]")

# ── Feature 4: Yield curve ───────────────────────────────────────────────────
# 10Y - IRX (short rate proxy). Negative = inverted = recession signal.
raw['F4_YIELD_CURVE'] = raw['T10Y'] - raw['IRX']
print("  F4: 10Y - IRX [Yield curve / macro environment]")

# ── Feature 5: Equity momentum ───────────────────────────────────────────────
# 21-day (1-month) log return on SPY. Negative = drawdown environment.
raw['F5_MOMENTUM'] = np.log(raw['SPY'] / raw['SPY'].shift(21))
print("  F5: SPY 21-day log return [Equity momentum / trend]")

# ── Feature 6: Volume Z-score ────────────────────────────────────────────────
# SPY volume requires downloading separately with auto_adjust=False
spy_vol = yf.download('SPY', start=START, end=END, progress=False, auto_adjust=False)['Volume'].squeeze()
spy_vol = spy_vol.reindex(raw.index)
ROLL = 63  # ~3-month rolling window
raw['F6_VOL_Z'] = (spy_vol - spy_vol.rolling(ROLL).mean()) / (spy_vol.rolling(ROLL).std() + 1e-8)
print(f"  F6: Volume Z-score (rolling {ROLL}d) [Liquidity / market activity]")

# ── Drop rows needed for warmup ───────────────────────────────────────────────
feat_cols = ['F1_logVIX', 'F2_VIX_TERM', 'F3_CREDIT', 'F4_YIELD_CURVE', 'F5_MOMENTUM', 'F6_VOL_Z']
feat_df = raw[feat_cols].dropna()
print(f"\n  Final feature matrix: {feat_df.shape[0]} rows × {feat_df.shape[1]} features")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Descriptive statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FEATURE DESCRIPTIVE STATISTICS")
print("=" * 70)
desc = feat_df.describe().T[['mean', 'std', 'min', 'max']]
desc.columns = ['Mean', 'Std', 'Min', 'Max']
print(desc.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 4. Correlation matrix
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PAIRWISE CORRELATION MATRIX  (Pearson)")
print("=" * 70)
corr = feat_df.corr()
print(corr.round(3).to_string())

# Flag high correlations
labels = {
    'F1_logVIX':    'log(VIX)',
    'F2_VIX_TERM':  'VIX Term',
    'F3_CREDIT':    'Credit',
    'F4_YIELD_CURVE':'YieldCurve',
    'F5_MOMENTUM':  'Momentum',
    'F6_VOL_Z':     'VolZscore',
}
print("\n  Pairs with |corr| > 0.50 (potential redundancy):")
high_corr = []
for i in range(len(feat_cols)):
    for j in range(i+1, len(feat_cols)):
        c = corr.iloc[i, j]
        if abs(c) > 0.50:
            print(f"    {feat_cols[i]:20s} vs {feat_cols[j]:20s}  corr = {c:+.3f}")
            high_corr.append((feat_cols[i], feat_cols[j], c))

if not high_corr:
    print("    None — all pairwise correlations are below 0.50 ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Variance Inflation Factor (VIF)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  VARIANCE INFLATION FACTOR (VIF)")
print("  VIF > 5 → concerning multicollinearity")
print("  VIF > 10 → severe")
print("=" * 70)

X_vif = feat_df.values
vif_scores = []
for i in range(X_vif.shape[1]):
    v = variance_inflation_factor(X_vif, i)
    vif_scores.append(v)
    flag = "  ← HIGH" if v > 5 else ""
    print(f"  {feat_cols[i]:22s}: VIF = {v:6.2f}{flag}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature selection recommendation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FEATURE SELECTION RECOMMENDATION")
print("=" * 70)

selected = []
dropped  = {}

for i, col in enumerate(feat_cols):
    vif = vif_scores[i]
    # Check if this feature is highly correlated with an already-selected feature
    redundant_with = None
    for sel in selected:
        sel_idx = feat_cols.index(sel)
        c = abs(corr.loc[col, sel])
        if c > 0.70:
            # Keep the one with lower VIF
            if vif < vif_scores[sel_idx]:
                # Replace sel with col
                dropped[sel] = f"Dropped: |corr({col})| = {c:.2f} and VIF worse"
                selected.remove(sel)
                redundant_with = None
            else:
                redundant_with = sel
            break
    if redundant_with is None:
        if vif <= 10:   # Generous threshold — let results speak
            selected.append(col)
        else:
            dropped[col] = f"Dropped: VIF = {vif:.1f} (severe multicollinearity)"

print(f"\n  SELECTED ({len(selected)} features):")
for s in selected:
    print(f"    ✓  {s:22s}  VIF={vif_scores[feat_cols.index(s)]:.2f}  → {labels[s]}")

if dropped:
    print(f"\n  DROPPED:")
    for d, reason in dropped.items():
        print(f"    ✗  {d:22s}  {reason}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Plot correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────
short_labels = [labels[c] for c in feat_cols]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Feature Independence Analysis for Multi-Feature HMM", fontsize=14, fontweight='bold')

# Heatmap
im = axes[0].imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(len(feat_cols)))
axes[0].set_yticks(range(len(feat_cols)))
axes[0].set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
axes[0].set_yticklabels(short_labels, fontsize=9)
axes[0].set_title("Pearson Correlation Matrix", fontsize=11)
for i in range(len(feat_cols)):
    for j in range(len(feat_cols)):
        axes[0].text(j, i, f'{corr.values[i,j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='black' if abs(corr.values[i,j]) < 0.5 else 'white')
plt.colorbar(im, ax=axes[0])

# VIF bar chart
colors = ['green' if v <= 5 else 'orange' if v <= 10 else 'red' for v in vif_scores]
axes[1].barh(short_labels, vif_scores, color=colors, edgecolor='black', linewidth=0.5)
axes[1].axvline(5,  color='orange', linestyle='--', linewidth=1.5, label='VIF=5 (moderate)')
axes[1].axvline(10, color='red',    linestyle='--', linewidth=1.5, label='VIF=10 (severe)')
axes[1].set_xlabel('VIF Score')
axes[1].set_title('Variance Inflation Factor (VIF)', fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='x')
# Add value labels
for i, v in enumerate(vif_scores):
    axes[1].text(v + 0.1, i, f'{v:.1f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('plots/feature_correlation_matrix.png', dpi=150, bbox_inches='tight')
print("\nSaved: plots/feature_correlation_matrix.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 8. Save feature CSV for next step
# ─────────────────────────────────────────────────────────────────────────────
feat_df['SPY'] = raw['SPY'].reindex(feat_df.index)
feat_df['VIX'] = raw['VIX'].reindex(feat_df.index)
feat_df.to_csv('data/hmm_candidate_features.csv')
print(f"Saved: data/hmm_candidate_features.csv  ({feat_df.shape[0]} rows × {feat_df.shape[1]} cols)")
print(f"\nReady for HMM: use columns {selected}")
