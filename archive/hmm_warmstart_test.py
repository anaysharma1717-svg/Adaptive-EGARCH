"""
hmm_warmstart_test.py  --  Testing Warm-Start vs Cold-Restart for Label Switching
════════════════════════════════════════════════════════════════════════════════

1. 3-State Cold-Restart (Baseline)
2. 3-State Warm-Start (Initialize with previous window's parameters)
3. 2-State Cold-Restart (Test "inherent stability" assumption)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
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
df = raw.dropna(subset=feat_cols + ['RV_21d']).copy()

g_means = df[feat_cols].mean()
g_stds  = df[feat_cols].std()
X_std = ((df[feat_cols] - g_means) / g_stds).values

print(f"  Data: {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")

BURN_IN = 500
REFIT   = 60

# ═════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def run_diagnostic(n_components, warm_start=False):
    """
    Runs the expanding window loop and tracks label switching via Hungarian matching.
    Returns: mismatch_rate, prob_series (aligned using Hungarian if warm_start is false,
    otherwise just identity since warm-start implicitly aligns)
    """
    prob_series = np.full((len(df), n_components), np.nan)
    prev_means = None
    anchor_means = None
    
    switches = 0
    total_refits = 0
    
    current_model = None
    current_perm = None # For aligning probabilities to anchor
    
    # Need to keep the latest fitted parameters for warm start
    last_startprob = None
    last_transmat = None
    last_means = None
    last_covars = None

    for t in range(BURN_IN, len(df)):
        is_refit = (t == BURN_IN) or (t % REFIT == 0)

        if is_refit:
            if warm_start and t > BURN_IN:
                # Warm start from previous parameters
                m = GaussianHMM(n_components=n_components, covariance_type='full',
                                n_iter=200, init_params='', params='stmc', random_state=42)
                m.startprob_ = last_startprob.copy()
                m.transmat_ = last_transmat.copy()
                m.means_ = last_means.copy()
                m.covars_ = last_covars.copy()
                m.fit(X_std[:t])
            else:
                # Cold restart - find best of 5 seeds
                best_model, best_ll = None, -np.inf
                for seed in [42, 123, 456, 789, 1024]:
                    m_temp = GaussianHMM(n_components=n_components, covariance_type='full',
                                         n_iter=300, random_state=seed)
                    m_temp.fit(X_std[:t])
                    ll = m_temp.score(X_std[:t])
                    if ll > best_ll:
                        best_ll = ll
                        best_model = m_temp
                m = best_model
                
            new_means = m.means_.copy()
            total_refits += 1
            
            # 1. Label-switching mismatch check (against PREVIOUS window)
            if prev_means is not None:
                D_prev = cdist(prev_means, new_means, metric='euclidean')
                row_ind, col_ind = linear_sum_assignment(D_prev)
                optimal_perm_prev = col_ind
                identity = np.arange(n_components)
                if not np.array_equal(optimal_perm_prev, identity):
                    switches += 1
            
            prev_means = new_means.copy()
            
            # 2. Alignment to ANCHOR for plotting
            if anchor_means is None:
                # Define anchor order by VIX mean (feature 0 = logVIX)
                vix_order = np.argsort(new_means[:, 0]) 
                current_perm = np.argsort(vix_order) # canonical -> raw
                anchor_means = new_means[vix_order].copy()
            else:
                D_anchor = cdist(anchor_means, new_means, metric='euclidean')
                _, col_ind = linear_sum_assignment(D_anchor)
                current_perm = col_ind # canonical -> raw

            current_model = m
            last_startprob = m.startprob_
            last_transmat = m.transmat_
            last_means = m.means_
            last_covars = m.covars_

        # Predict probability for day t and apply canonical alignment
        p_raw = current_model.predict_proba(X_std[:t+1])[-1]
        for ci in range(n_components):
            prob_series[t, ci] = p_raw[current_perm[ci]]

    # Exclude the very first fit from switch calculation since there's no previous
    mismatch_rate = switches / (total_refits - 1) if total_refits > 1 else 0
    return mismatch_rate, prob_series, total_refits - 1

# ═════════════════════════════════════════════════════════════════════════════
#  RUN TESTS
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  TEST 1: 3-State Cold-Restart (Baseline)")
print("=" * 80)
rate_3_cold, prob_3_cold, n_refits = run_diagnostic(3, warm_start=False)
print(f"  Mismatch rate: {rate_3_cold*100:.1f}%")

print("\n" + "=" * 80)
print("  TEST 2: 3-State Warm-Start")
print("=" * 80)
rate_3_warm, prob_3_warm, _ = run_diagnostic(3, warm_start=True)
print(f"  Mismatch rate: {rate_3_warm*100:.1f}%")

print("\n" + "=" * 80)
print("  TEST 3: 2-State Cold-Restart")
print("=" * 80)
rate_2_cold, prob_2_cold, _ = run_diagnostic(2, warm_start=False)
print(f"  Mismatch rate: {rate_2_cold*100:.1f}%")

# ═════════════════════════════════════════════════════════════════════════════
#  PLOT WARM-STARTED 3-STATE SERIES
# ═════════════════════════════════════════════════════════════════════════════
print("\n  Generating plot for 3-State Warm-Start...")

fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
fig.suptitle("Full History: Aligned OOS Probabilities (3-State WARM-START)", fontsize=14, fontweight='bold')

axes[0].plot(df.index, df['VIX'], 'steelblue', linewidth=0.5)
axes[0].axhline(20, color='orange', linestyle='--', linewidth=0.5)
axes[0].axhline(30, color='red', linestyle='--', linewidth=0.5)
axes[0].set_ylabel('VIX')
axes[0].grid(True, alpha=0.2)

colors = ['steelblue', 'orange', 'crimson']
labels = ['P(Calm)', 'P(Transition)', 'P(Crisis)']

for ci, (col, lab) in enumerate(zip(colors, labels)):
    axes[1].fill_between(df.index, prob_3_warm[:, ci], alpha=0.5, color=col, label=lab)
axes[1].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
axes[1].set_ylabel('Warm-Started\nProbabilities')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].set_ylim(-0.05, 1.1)
axes[1].grid(True, alpha=0.2)

axes[2].plot(df.index, df['F4_YIELD_CURVE'], 'purple', linewidth=0.5)
axes[2].axhline(0, color='red', linestyle='--', linewidth=0.8)
axes[2].set_ylabel('Yield Curve\n(10Y−IRX)')
axes[2].grid(True, alpha=0.2)
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('plots/warm_start_3_state.png', dpi=150, bbox_inches='tight')
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
#  REPORT TABLE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  FINAL DIAGNOSTIC REPORT")
print("=" * 80)
print(f"  Total Refit Boundaries Checked: {n_refits}")
print(f"")
print(f"  Model Configuration             | Label Mismatch Rate")
print(f"  --------------------------------+--------------------")
print(f"  3-State (Cold Restarts, 5-seed) | {rate_3_cold*100:>5.1f}%")
print(f"  3-State (Warm Starts)           | {rate_3_warm*100:>5.1f}%")
print(f"  2-State (Cold Restarts, 5-seed) | {rate_2_cold*100:>5.1f}%")
print(f"")
