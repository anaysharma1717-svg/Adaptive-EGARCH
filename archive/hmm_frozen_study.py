"""
hmm_frozen_study.py  --  Final Frozen 3-State HMM Validation
════════════════════════════════════════════════════════════════════════════════

1. Train frozen 3-state HMM on data from 2010 through mid-2024 (In-Sample).
2. Verify Transition state (inverted YC) emerges in the training parameters.
3. Generate OOS probabilities (2024-mid to 2026) using causal forward filtering.
   Note: hmmlearn's predict_proba(X[:t])[-1] is exactly the forward filtered 
   probability P(state_t | X_1...X_t), because at the final step T, the backward 
   variable beta_T is 1, making smoothed == filtered.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
from hmmlearn.hmm import GaussianHMM
import os

warnings.filterwarnings("ignore")
os.makedirs('plots', exist_ok=True)

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# ═════════════════════════════════════════════════════════════════════════════
#  DATA PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  REBUILDING DATA")
print("=" * 80)

START = '2010-01-01'

def dl_close(ticker):
    d = yf.download(ticker, start=START, progress=False, auto_adjust=True)
    return d['Close'].squeeze().rename(ticker)

vix   = dl_close('^VIX')
vix9d = dl_close('^VIX9D')
vix3m = dl_close('^VIX3M')
hyg   = dl_close('HYG')
lqd   = dl_close('LQD')
spy   = dl_close('SPY')
t10y  = dl_close('^TNX')
t2y   = dl_close('^IRX')
spy_ohlcv = yf.download('SPY', start=START, progress=False, auto_adjust=False)
spy_vol   = spy_ohlcv['Volume'].squeeze().rename('SPY_VOL')

raw = pd.DataFrame({
    'VIX': vix, 'VIX9D': vix9d, 'VIX3M': vix3m,
    'HYG': hyg, 'LQD': lqd, 'SPY': spy,
    'T10Y': t10y, 'IRX': t2y, 'SPY_VOL': spy_vol,
}).dropna().sort_index()
raw.index = pd.to_datetime(raw.index)

raw['F1_logVIX']      = np.log(raw['VIX'])
raw['F2_VIX_TERM']    = raw['VIX9D'] - raw['VIX3M']
raw['F3_CREDIT']      = np.log(raw['HYG'] / raw['LQD'])
raw['F4_YIELD_CURVE'] = raw['T10Y'] - raw['IRX']
raw['F5_VOL_Z']       = (raw['SPY_VOL'] - raw['SPY_VOL'].rolling(63).mean()) / \
                         (raw['SPY_VOL'].rolling(63).std() + 1e-8)
raw['F6_MOMENTUM']    = np.log(raw['SPY'] / raw['SPY'].shift(21))
raw['RV_21d']         = np.log(raw['SPY'] / raw['SPY'].shift(1)).rolling(21).std() * np.sqrt(252) * 100

feat_cols = ['F1_logVIX', 'F2_VIX_TERM', 'F3_CREDIT', 'F4_YIELD_CURVE', 'F5_VOL_Z', 'F6_MOMENTUM']
feat_names = ['logVIX', 'VIX_Term', 'Credit', 'YldCrv', 'Vol_Z', 'Momentum']

df = raw.dropna(subset=feat_cols + ['RV_21d']).copy()

# Standardize over FULL sample to avoid leak in scaling (or scale on IS only)
# For strictness, let's scale on IS only.
TRAIN_END = pd.Timestamp('2024-06-30')

is_mask = df.index <= TRAIN_END
oos_mask = df.index > TRAIN_END

g_means = df.loc[is_mask, feat_cols].mean()
g_stds  = df.loc[is_mask, feat_cols].std()
X_std = ((df[feat_cols] - g_means) / g_stds).values

print(f"  Total Data: {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")
print(f"  In-Sample : {is_mask.sum()} rows (up to {TRAIN_END.date()})")
print(f"  OOS       : {oos_mask.sum()} rows ({df.index[oos_mask][0].date()} → {df.index[-1].date()})")
print(f"  OOS Length: {oos_mask.sum() / 252:.1f} years")

# ═════════════════════════════════════════════════════════════════════════════
#  PART 1: TRAIN FROZEN MODEL (IS)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PART 1 — TRAINING IN-SAMPLE (2011 to Mid-2024)")
print("=" * 80)

X_is = X_std[is_mask]

best_model = None
best_ll = -np.inf
for seed in [42, 123, 456, 789, 1024]:
    m = GaussianHMM(n_components=3, covariance_type='full', n_iter=500, random_state=seed)
    m.fit(X_is)
    ll = m.score(X_is)
    if ll > best_ll:
        best_ll = ll
        best_model = m

m = best_model

# Sort states by VIX mean to ensure 0=Calm, 1=Trans/Mid, 2=Crisis
vix_order = np.argsort(m.means_[:, 0])
m.means_ = m.means_[vix_order]
m.covars_ = m.covars_[vix_order]
m.startprob_ = m.startprob_[vix_order]
# Reorder transmat
P = m.transmat_
P = P[vix_order, :]
P = P[:, vix_order]
m.transmat_ = P

# Un-standardize means for interpretation
means_raw = m.means_ * g_stds.values + g_means.values

print("\n  State Characteristics (In-Sample):")
labels = []
for i in range(3):
    yc = means_raw[i, 3]
    vix_val = np.exp(means_raw[i, 0])
    # Identify Transition state (low/med VIX, negative YC)
    if yc < 0 and vix_val < 25:
        name = "Transition"
    elif vix_val > 22:
        name = "Crisis"
    else:
        name = "Calm"
    labels.append(name)
    
    print(f"\n  State {i}: {name}")
    print(f"    VIX:      {vix_val:6.1f}   (Mean logVIX: {means_raw[i,0]:.2f})")
    print(f"    YldCrv:   {yc:6.2f}")
    print(f"    Credit:   {means_raw[i,2]:6.2f}")
    print(f"    Momentum: {means_raw[i,5]:6.3f}")

print("\n  Transition Matrix:")
for i in range(3):
    print(f"    From {labels[i]:<10}: " + "  ".join(f"{labels[j]}={P[i,j]:.3f}" for j in range(3)))

if "Transition" not in labels:
    print("\n  ⚠ WARNING: No state was identified as 'Transition' (negative Yield Curve).")
    print("    The 2022-2023 hiking cycle might not be distinct enough to claim a 3rd state.")

# ═════════════════════════════════════════════════════════════════════════════
#  PART 2: OUT-OF-SAMPLE CAUSAL FILTERING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PART 2 — OUT-OF-SAMPLE CAUSAL FILTERING (Mid-2024 to 2026)")
print("=" * 80)

# We want P(state_t | X_1...X_t) for all t.
# For IS, we can just do predict_proba which gives smoothed (since it's IS anyway).
# But for strict OOS timeline tracking, we should simulate walking forward.
# Actually, calling m.predict_proba(X[:t+1])[-1] is exactly the forward filtered prob.

prob_series = np.full((len(X_std), 3), np.nan)

# IS probs (smoothed is fine for IS analysis)
prob_series[is_mask] = m.predict_proba(X_is)

# OOS probs (strictly causal filtered)
oos_indices = np.where(oos_mask)[0]
print(f"  Filtering {len(oos_indices)} OOS days...")

for t in oos_indices:
    # m.predict_proba(X_std[:t+1]) runs forward-backward on data up to t.
    # At time t, backward variable is 1.0, so the result for t is exactly 
    # the forward filtered probability.
    p_t = m.predict_proba(X_std[:t+1])[-1]
    prob_series[t] = p_t

df['P_Calm'] = prob_series[:, 0]
df['P_Trans'] = prob_series[:, 1]
df['P_Crisis'] = prob_series[:, 2]

# Map names if they aren't exactly [Calm, Trans, Crisis] due to sorting
# By construction above, we sorted by VIX, but labels[] holds the true names.
c_idx = labels.index('Calm') if 'Calm' in labels else 0
x_idx = labels.index('Transition') if 'Transition' in labels else 1
r_idx = labels.index('Crisis') if 'Crisis' in labels else 2

df['P_Calm'] = prob_series[:, c_idx]
df['P_Trans'] = prob_series[:, x_idx]
df['P_Crisis'] = prob_series[:, r_idx]

print("\n  OOS Monthly Snapshots (First day of each month):")
print(f"  {'Date':<14} {'VIX':>6} {'YldCrv':>8} {'P(Calm)':>8} {'P(Trans)':>8} {'P(Crisis)':>9}")
print("  " + "-" * 60)

prev_month = None
for idx in oos_indices:
    d = df.index[idx]
    if d.month != prev_month:
        print(f"  {str(d.date()):<14} {df['VIX'].values[idx]:>6.1f} {df['F4_YIELD_CURVE'].values[idx]:>+8.2f} "
              f"{df['P_Calm'].values[idx]:>8.4f} {df['P_Trans'].values[idx]:>8.4f} {df['P_Crisis'].values[idx]:>9.4f}")
        prev_month = d.month

# Check if OOS gets stuck
oos_df = df.iloc[oos_indices]
stuck_crisis = (oos_df['P_Crisis'] > 0.9).sum() / len(oos_df)
print(f"\n  OOS 'Stuck in Crisis' check:")
print(f"    Days with P(Crisis) > 0.9: {stuck_crisis*100:.1f}% of OOS")

# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═════════════════════════════════════════════════════════════════════════════
print("\n  Generating plots...")

fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
fig.suptitle(f"Frozen 3-State HMM (Trained through {TRAIN_END.date()}) — Causal OOS", fontsize=14, fontweight='bold')

axes[0].plot(df.index, df['VIX'], 'steelblue', linewidth=0.5)
axes[0].axvline(TRAIN_END, color='black', linestyle='--', linewidth=2, label='OOS Start')
axes[0].axhline(20, color='orange', linestyle='--', linewidth=0.5)
axes[0].axhline(30, color='red', linestyle='--', linewidth=0.5)
axes[0].set_ylabel('VIX')
axes[0].legend()
axes[0].grid(True, alpha=0.2)

colors = ['steelblue', 'orange', 'crimson']
state_cols = ['P_Calm', 'P_Trans', 'P_Crisis']

for ci, (col, st) in enumerate(zip(colors, state_cols)):
    axes[1].fill_between(df.index, df[st], alpha=0.5, color=col, label=st)
axes[1].axvline(TRAIN_END, color='black', linestyle='--', linewidth=2)
axes[1].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
axes[1].set_ylabel('Probability')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].set_ylim(-0.05, 1.1)
axes[1].grid(True, alpha=0.2)

axes[2].plot(df.index, df['F4_YIELD_CURVE'], 'purple', linewidth=0.5)
axes[2].axvline(TRAIN_END, color='black', linestyle='--', linewidth=2)
axes[2].axhline(0, color='red', linestyle='--', linewidth=0.8)
axes[2].set_ylabel('Yield Curve\n(10Y−IRX)')
axes[2].grid(True, alpha=0.2)
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('plots/frozen_3_state_oos.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/frozen_3_state_oos.png")
print("\n" + "=" * 80)
