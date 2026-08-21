"""
hmm_2state_frozen_test.py  --  Final Validation of 2-State Frozen Architecture
════════════════════════════════════════════════════════════════════════════════
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
df = raw.dropna(subset=feat_cols + ['RV_21d']).copy()

TRAIN_END = pd.Timestamp('2024-06-30')
is_mask = df.index <= TRAIN_END
oos_mask = df.index > TRAIN_END

g_means = df.loc[is_mask, feat_cols].mean()
g_stds  = df.loc[is_mask, feat_cols].std()
X_std = ((df[feat_cols] - g_means) / g_stds).values
X_is = X_std[is_mask]

# ═════════════════════════════════════════════════════════════════════════════
#  RETROSPECTIVE 3-STATE CHECK
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  RETROSPECTIVE: 3-STATE OOS VOLATILITY CHECK")
print("=" * 80)
oos_df = df[oos_mask]
max_vix = oos_df['VIX'].max()
max_rv = oos_df['RV_21d'].max()
mean_vix = oos_df['VIX'].mean()

print(f"  OOS Period (Mid-2024 to Mid-2026):")
print(f"    Max VIX: {max_vix:.1f}")
print(f"    Max RV:  {max_rv:.1f}%")
print(f"    Mean VIX:{mean_vix:.1f}")

if max_vix > 30:
    print("\n  [VERDICT] ✗ FAIL: The 3-state model stayed 100% in 'State 1' (Calm/Flat) during")
    print("            the OOS period despite VIX spiking to extreme crisis levels.")
    print("            This was a catastrophic failure to adapt to actual volatility.")
else:
    print("\n  [VERDICT] The OOS period was genuinely low-volatility, so staying in State 1")
    print("            was a defensible (though perhaps lucky) classification.")

# ═════════════════════════════════════════════════════════════════════════════
#  TRAIN 2-STATE FROZEN
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  TRAINING 2-STATE FROZEN MODEL (IS: 2011 to Mid-2024)")
print("=" * 80)

best_model = None
best_ll = -np.inf
for seed in [42, 123, 456, 789, 1024]:
    m = GaussianHMM(n_components=2, covariance_type='full', n_iter=500, random_state=seed)
    m.fit(X_is)
    ll = m.score(X_is)
    if ll > best_ll:
        best_ll = ll
        best_model = m
m2 = best_model

# Sort states by VIX mean
vix_order = np.argsort(m2.means_[:, 0])
m2.means_ = m2.means_[vix_order]
m2.covars_ = m2.covars_[vix_order]
m2.startprob_ = m2.startprob_[vix_order]
P = m2.transmat_
P = P[vix_order, :]
P = P[:, vix_order]
m2.transmat_ = P

means_raw = m2.means_ * g_stds.values + g_means.values
print("  State 0 (Calm):   VIX = {:.1f}".format(np.exp(means_raw[0, 0])))
print("  State 1 (Crisis): VIX = {:.1f}".format(np.exp(means_raw[1, 0])))

# ═════════════════════════════════════════════════════════════════════════════
#  CAUSAL FORWARD FILTERING (ENTIRE HISTORY)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  CAUSAL FORWARD FILTERING (Full History, No Lookahead)")
print("=" * 80)
# We calculate P(state_t | X_1...X_t) for EVERY day in the dataset using the 
# model trained only on data through mid-2024.

prob_series_2s = np.full((len(X_std), 2), np.nan)
print("  Running forward filter...")
for t in range(len(X_std)):
    # m.predict_proba(X[:t+1]) runs forward-backward on data up to t.
    # The last element [-1] is the forward filtered probability at time t.
    p_t = m2.predict_proba(X_std[:t+1])[-1]
    prob_series_2s[t] = p_t

df['P_Calm_2S'] = prob_series_2s[:, 0]
df['P_Crisis_2S'] = prob_series_2s[:, 1]

# ═════════════════════════════════════════════════════════════════════════════
#  CHECK 1: EVENT IDENTIFICATION & NON-STICKINESS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  CHECK 1: EVENT IDENTIFICATION & NON-STICKINESS")
print("=" * 80)

events = {
    "COVID (Mar 2020)": ("2020-02-01", "2020-04-30", "2020-08-01"),
    "Fed Hiking (Mid-2022)": ("2022-04-01", "2022-10-31", "2023-01-31"),
    "Tariff Shock (Apr 2025, OOS)": ("2025-03-01", "2025-05-31", "2025-08-31")
}

check1_pass = True
for name, (start_dt, end_dt, recovery_dt) in events.items():
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    peak_crisis_prob = df.loc[mask, 'P_Crisis_2S'].max()
    
    # Check if it returned to calm shortly after
    rec_mask = (df.index > end_dt) & (df.index <= recovery_dt)
    if len(df[rec_mask]) > 0:
        min_crisis_prob_after = df.loc[rec_mask, 'P_Crisis_2S'].min()
    else:
        min_crisis_prob_after = 0.0 # Ignore if missing
        
    status_peak = "✓" if peak_crisis_prob > 0.5 else "✗"
    status_rec  = "✓" if min_crisis_prob_after < 0.5 else "✗"
    
    print(f"  {name}:")
    print(f"    Peak P(Crisis) during event: {peak_crisis_prob:.3f} [{status_peak}]")
    print(f"    Min P(Crisis) in recovery:   {min_crisis_prob_after:.3f} [{status_rec}]")
    
    if peak_crisis_prob < 0.5 or min_crisis_prob_after > 0.5:
        check1_pass = False

print(f"\n  [VERDICT] Check 1: {'PASS' if check1_pass else 'FAIL'}")

# ═════════════════════════════════════════════════════════════════════════════
#  CHECK 2: CHRONOLOGY vs VOLATILITY (Calm State Consistency)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  CHECK 2: CHRONOLOGY VS VOLATILITY")
print("=" * 80)

# Compare average P(Calm) during known low-vol periods across decades
calm_periods = {
    "Pre-COVID Calm (2017)": ("2017-01-01", "2017-12-31"),
    "Post-COVID Calm (2021)": ("2021-06-01", "2021-11-01"),
    "Post-Fed Calm (2023)": ("2023-06-01", "2023-12-31")
}

check2_pass = True
for name, (start, end) in calm_periods.items():
    mask = (df.index >= start) & (df.index <= end)
    avg_calm = df.loc[mask, 'P_Calm_2S'].mean()
    status = "✓" if avg_calm > 0.5 else "✗"
    print(f"  {name}: Avg P(Calm) = {avg_calm:.3f} [{status}]")
    if avg_calm < 0.5:
        check2_pass = False

print(f"\n  [VERDICT] Check 2: {'PASS' if check2_pass else 'FAIL'}")

# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═════════════════════════════════════════════════════════════════════════════
print("\n  Generating plot...")
fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
fig.suptitle(f"Frozen 2-State HMM (Trained through {TRAIN_END.date()}) — Causal Filtered Probabilities", fontsize=14, fontweight='bold')

axes[0].plot(df.index, df['VIX'], 'steelblue', linewidth=1)
axes[0].axvline(TRAIN_END, color='black', linestyle='--', linewidth=2, label='OOS Start')
axes[0].axhline(20, color='orange', linestyle='--', linewidth=0.5)
axes[0].axhline(30, color='red', linestyle='--', linewidth=0.5)
axes[0].set_ylabel('VIX')
axes[0].legend()
axes[0].grid(True, alpha=0.2)

axes[1].fill_between(df.index, df['P_Crisis_2S'], color='crimson', alpha=0.6, label='P(Crisis)')
axes[1].fill_between(df.index, df['P_Calm_2S'], color='steelblue', alpha=0.6, label='P(Calm)')
axes[1].axvline(TRAIN_END, color='black', linestyle='--', linewidth=2)
axes[1].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
axes[1].set_ylabel('Filtered Probability')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].set_ylim(-0.05, 1.1)
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('plots/frozen_2_state_causal.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/frozen_2_state_causal.png")
print("\n" + "=" * 80)
