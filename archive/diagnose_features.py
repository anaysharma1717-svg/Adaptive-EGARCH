"""
Diagnostic script: inspect raw feature distributions and what omega_t / alpha_t
would look like BEFORE any regularization or clamping.
Run this first to understand how bad the instability actually is.
"""

import pandas as pd
import numpy as np

# ── Load features ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/dp_egarch_features.csv', index_col=0, parse_dates=True)

# The 11 Z-scored features we'll feed into the model (already normalized)
feature_cols = [
    'VIX_Z', 'VVIX_Z', 'VIX_TERM_Z', 'YZ_VOL_21_Z',
    'VOL_Z_Z', 'CREDIT_SPREAD_Z', 'YIELD_CURVE_Z',
    'OVERNIGHT_GAP_Z', 'VOL_MOMENTUM_Z',
    'DAYS_TO_FOMC', 'DAYS_TO_CPI_NFP'
]

# Keep only rows where all features are present
X = df[feature_cols].dropna()

print(f"Feature matrix shape: {X.shape}")
print(f"Date range: {X.index[0].date()} → {X.index[-1].date()}\n")

# ── Step 1: Raw feature distribution ──────────────────────────────────────────
print("=" * 65)
print("RAW FEATURE STATISTICS (after Z-scoring)")
print("=" * 65)
stats = X.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
stats = stats[['min', '1%', '5%', '50%', '95%', '99%', 'max', 'std']]
print(stats.round(3).to_string())

# ── Step 2: Simulate raw scores with random unit-normal weights ────────────────
np.random.seed(42)
n_features = len(feature_cols)
n_sims = 5

print("\n" + "=" * 65)
print("SIMULATED RAW SCORES  w^T X_t  (5 random weight vectors)")
print("(These are the inputs to exp() and sigmoid() in DP-EGARCH)")
print("=" * 65)

X_arr = X.values

for i in range(n_sims):
    w = np.random.randn(n_features)  # random weights, unit normal
    raw_scores = X_arr @ w

    print(f"\n  Sim {i+1}  ||w|| = {np.linalg.norm(w):.2f}")
    print(f"    Raw score min  : {raw_scores.min():.2f}")
    print(f"    Raw score max  : {raw_scores.max():.2f}")
    print(f"    Raw score std  : {raw_scores.std():.2f}")
    print(f"    Raw score p1/p99: {np.percentile(raw_scores, 1):.2f} / {np.percentile(raw_scores, 99):.2f}")

    omega  = np.exp(raw_scores)
    alpha  = 1 / (1 + np.exp(-raw_scores))  # sigmoid

    print(f"    → omega_t  range: [{omega.min():.4f}, {omega.max():.2f}]  (ratio max/min = {omega.max()/omega.min():.0f}x)")
    print(f"    → alpha_t  range: [{alpha.min():.4f}, {alpha.max():.4f}]")

# ── Step 3: Worst-case with slightly larger weights ────────────────────────────
print("\n" + "=" * 65)
print("WORST-CASE: what happens with larger weights  (||w|| = 3)")
print("=" * 65)

w_big = np.random.randn(n_features)
w_big = w_big / np.linalg.norm(w_big) * 3.0
raw_big = X_arr @ w_big
omega_big = np.exp(raw_big)

print(f"  Raw score min/max : {raw_big.min():.2f} / {raw_big.max():.2f}")
print(f"  omega_t min/max   : {omega_big.min():.6f} / {omega_big.max():.2f}")
print(f"  omega_t ratio     : {omega_big.max()/omega_big.min():.0f}x")
print(f"  # days omega > 10 : {(omega_big > 10).sum()}")
print(f"  # days omega < 0.01: {(omega_big < 0.01).sum()}")

# ── Step 4: Day-to-day jumps in omega (the real danger) ───────────────────────
print("\n" + "=" * 65)
print("DAY-TO-DAY OMEGA JUMPS  (consecutive ratio)")
print("=" * 65)

w_unit = np.random.randn(n_features)
raw_unit = X_arr @ w_unit
omega_unit = np.exp(raw_unit)
daily_ratio = np.abs(omega_unit[1:] / omega_unit[:-1])

print(f"  Median daily omega ratio : {np.median(daily_ratio):.3f}x")
print(f"  95th pct daily ratio     : {np.percentile(daily_ratio, 95):.2f}x")
print(f"  99th pct daily ratio     : {np.percentile(daily_ratio, 99):.2f}x")
print(f"  Max daily ratio          : {daily_ratio.max():.2f}x")
print(f"\n  Top 5 worst daily jumps:")
top5_idx = np.argsort(daily_ratio)[-5:][::-1]
for idx in top5_idx:
    print(f"    {X.index[idx].date()} → {X.index[idx+1].date()}  ratio = {daily_ratio[idx]:.2f}x")

print("\n✓ Done. Use these numbers to decide if clamping/L2 is needed.")
