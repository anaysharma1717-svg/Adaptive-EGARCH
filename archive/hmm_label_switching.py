"""
hmm_label_switching.py  --  Detect and fix label-switching across refits
═══════════════════════════════════════════════════════════════════════════

1. At each 60-day refit, compare old vs new state means
2. Use Hungarian algorithm to find optimal label alignment
3. Flag every refit where labels permuted
4. Implement alignment, rerun COVID + 2022 timelines
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
#  DATA PIPELINE (identical to hmm_final_study.py)
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

N_STATES = 3
BURN_IN  = 500
REFIT    = 60

feat_short = ['logVIX', 'VIXTerm', 'Credit', 'YldCrv', 'VolZ', 'Mom']

# ═════════════════════════════════════════════════════════════════════════════
#  PART 1: DETECT LABEL-SWITCHING (UNALIGNED)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PART 1 — DETECTING LABEL-SWITCHING ACROSS REFITS")
print("=" * 80)

# Run expanding window, track model means at each refit
refit_log = []        # list of dicts per refit
prev_means = None     # (3, 6) array of previous state means

prob_unaligned = np.full((len(df), N_STATES), np.nan)

current_model = None
current_permutation = None  # identity initially

for t in range(BURN_IN, len(df)):
    is_refit = (t == BURN_IN) or (t % REFIT == 0)

    if is_refit:
        m = GaussianHMM(n_components=N_STATES, covariance_type='full',
                        n_iter=300, random_state=42)
        m.fit(X_std[:t])
        new_means = m.means_.copy()  # (3, 6) in standardized space

        entry = {
            'refit_idx': t,
            'date': df.index[t],
            'new_means': new_means.copy(),
        }

        if prev_means is not None:
            # Compute pairwise distance matrix: prev_means vs new_means
            D = cdist(prev_means, new_means, metric='euclidean')  # (3, 3)
            entry['dist_matrix'] = D.copy()

            # Hungarian algorithm: find optimal assignment
            row_ind, col_ind = linear_sum_assignment(D)
            optimal_perm = col_ind  # maps old_state_i → new_state_j
            identity = np.arange(N_STATES)
            is_switched = not np.array_equal(optimal_perm, identity)

            entry['optimal_perm'] = optimal_perm.copy()
            entry['is_switched'] = is_switched
            entry['total_cost'] = D[row_ind, col_ind].sum()

            if is_switched:
                entry['switch_detail'] = {int(i): int(optimal_perm[i]) for i in range(N_STATES)}
        else:
            entry['is_switched'] = False
            entry['optimal_perm'] = np.arange(N_STATES)

        prev_means = new_means
        current_model = m
        refit_log.append(entry)

    # Predict probability for day t
    p = current_model.predict_proba(X_std[:t+1])
    prob_unaligned[t] = p[-1]

# Report label switches
n_switches = sum(1 for e in refit_log if e['is_switched'])
print(f"\n  Total refits: {len(refit_log)}")
print(f"  Label switches detected: {n_switches} ({100*n_switches/len(refit_log):.1f}%)")

print(f"\n  ── Refits with label-switching ──")
print(f"  {'Date':<14} {'Permutation':<20} {'Cost':>8} {'Distance Matrix (row=old, col=new)'}")
print("  " + "-" * 90)

for e in refit_log:
    if e['is_switched']:
        perm = e['optimal_perm']
        D = e['dist_matrix']
        perm_str = " → ".join(f"{i}→{perm[i]}" for i in range(N_STATES))
        d_str = "  ".join(f"d({i},{j})={D[i,j]:.2f}" for i in range(N_STATES) for j in range(N_STATES))
        print(f"  {str(e['date'].date()):<14} {perm_str:<20} {e['total_cost']:>8.3f}")
        # Print full distance matrix
        for i in range(N_STATES):
            row = "      " + "  ".join(f"{D[i,j]:6.3f}" for j in range(N_STATES))
            print(row)

# ═════════════════════════════════════════════════════════════════════════════
#  PART 2: ALIGNED OOS PROBABILITIES
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PART 2 — COMPUTING ALIGNED OOS PROBABILITIES")
print("=" * 80)

# Re-run with alignment: at each refit, permute new model's state indices
# to best match the ANCHOR (first model's states, sorted by VIX mean)

prob_aligned = np.full((len(df), N_STATES), np.nan)
anchor_means = None
refit_idx_counter = 0

for t in range(BURN_IN, len(df)):
    is_refit = (t == BURN_IN) or (t % REFIT == 0)

    if is_refit:
        m = GaussianHMM(n_components=N_STATES, covariance_type='full',
                        n_iter=300, random_state=42)
        m.fit(X_std[:t])
        new_means = m.means_.copy()

        if anchor_means is None:
            # First model: sort states by VIX mean (feature 0 = logVIX)
            vix_order = np.argsort(new_means[:, 0])  # ascending: calm, trans, crisis
            perm = np.argsort(vix_order)  # maps original idx → canonical idx
            anchor_means = new_means[vix_order]  # reorder to canonical
            anchor_labels = {0: 'Calm', 1: 'Transition', 2: 'Crisis'}
            print(f"  Anchor set at t={t} ({df.index[t].date()})")
            print(f"    Canonical order (by logVIX): {list(vix_order)}")
            for ci in range(N_STATES):
                print(f"    Canonical {ci} ({anchor_labels[ci]}): logVIX={anchor_means[ci, 0]:.3f}")
        else:
            # Align new model to anchor using Hungarian
            D = cdist(anchor_means, new_means, metric='euclidean')
            _, col_ind = linear_sum_assignment(D)
            # col_ind[canonical_i] = new_model_raw_state
            perm = col_ind  # maps canonical → raw

        current_model = m
        current_perm = perm
        refit_idx_counter += 1

    # Predict and apply permutation
    p_raw = current_model.predict_proba(X_std[:t+1])[-1]  # (3,)
    # Remap: prob_aligned[canonical_i] = p_raw[current_perm[canonical_i]]
    for ci in range(N_STATES):
        prob_aligned[t, ci] = p_raw[current_perm[ci]]

# Canonical labels: 0=Calm, 1=Transition, 2=Crisis
print(f"\n  Aligned probabilities computed for {np.sum(~np.isnan(prob_aligned[:, 0]))} days.")

# ═════════════════════════════════════════════════════════════════════════════
#  PART 3: COVID TIMELINE — UNALIGNED vs ALIGNED
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PART 3 — COVID TIMELINE: UNALIGNED vs ALIGNED")
print("=" * 80)

covid_mask = (df.index >= '2019-12-01') & (df.index <= '2020-03-15')
covid_idx = np.where(covid_mask)[0]

# For unaligned: identify crisis state at each point using VIX heuristic
# This is what hmm_final_study.py did — pick crisis as highest-VIX state
# But the label might flip. Let's show raw state 0, 1, 2 probabilities.

print(f"\n  {'Date':<14} {'VIX':>6} {'--- UNALIGNED (raw) ---':^30} {'--- ALIGNED (canonical) ---':^30}")
print(f"  {'':14} {'':>6} {'P(s0)':>8} {'P(s1)':>8} {'P(s2)':>8}   {'P(Calm)':>8} {'P(Trans)':>8} {'P(Crisis)':>9}")
print("  " + "-" * 100)

for i, idx in enumerate(covid_idx):
    d = df.index[idx]
    v = df['VIX'].values[idx]
    pu = prob_unaligned[idx]
    pa = prob_aligned[idx]

    if np.isnan(pu[0]):
        continue

    # Highlight dates of interest
    mark = ""
    if d.strftime('%Y-%m-%d') == '2020-01-30':
        mark = " ← REFIT"
    elif d.strftime('%Y-%m-%d') == '2020-02-24':
        mark = " ← VIX SPIKE"

    print(f"  {str(d.date()):<14} {v:>6.1f} {pu[0]:>8.4f} {pu[1]:>8.4f} {pu[2]:>8.4f}"
          f"   {pa[0]:>8.4f} {pa[1]:>8.4f} {pa[2]:>9.4f}{mark}")

# Find first date P(Crisis) > 0.5 in aligned series
aligned_crisis = prob_aligned[:, 2]  # canonical state 2 = Crisis
for idx in covid_idx:
    if not np.isnan(aligned_crisis[idx]) and aligned_crisis[idx] > 0.5:
        print(f"\n  ALIGNED: First P(Crisis) > 50% at {df.index[idx].date()} (VIX={df['VIX'].values[idx]:.1f})")
        break

# ═════════════════════════════════════════════════════════════════════════════
#  PART 4: 2022 FED HIKING TIMELINE — ALIGNED
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PART 4 — 2022 FED HIKING CYCLE: ALIGNED PROBABILITIES")
print("=" * 80)

fed_mask = (df.index >= '2021-10-01') & (df.index <= '2023-06-01')
fed_idx = np.where(fed_mask)[0]

print(f"\n  Monthly snapshots:")
print(f"  {'Date':<14} {'VIX':>6} {'YldCrv':>8} {'P(Calm)':>8} {'P(Trans)':>8} {'P(Crisis)':>9}")
print("  " + "-" * 60)

prev_month = None
for idx in fed_idx:
    d = df.index[idx]
    m = d.month
    if m == prev_month:
        continue
    prev_month = m
    pa = prob_aligned[idx]
    if np.isnan(pa[0]):
        continue
    v = df['VIX'].values[idx]
    yc = df['F4_YIELD_CURVE'].values[idx]
    print(f"  {str(d.date()):<14} {v:>6.1f} {yc:>+8.2f} {pa[0]:>8.4f} {pa[1]:>8.4f} {pa[2]:>9.4f}")

# ═════════════════════════════════════════════════════════════════════════════
#  PART 5: MAJOR EVENTS — ALIGNED EARLY WARNING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PART 5 — ALIGNED EARLY-WARNING FOR MAJOR EVENTS")
print("=" * 80)

events = [
    ("2015 China Crash",     "2015-08-24", "2015-06-01", "2015-10-01"),
    ("2018 Volmageddon",     "2018-02-05", "2017-12-01", "2018-04-01"),
    ("2020 COVID",           "2020-03-16", "2019-12-01", "2020-06-01"),
    ("2022 Fed Hiking",      "2022-06-13", "2022-01-01", "2022-10-01"),
    ("2025 Tariff Shock",    "2025-04-03", "2025-02-01", "2025-06-01"),
]

print(f"\n  {'Event':<22} {'Peak':>12} {'Aligned P(Crisis)>50%':>22} {'Lead':>8} {'VIX at detection':>16}")
print("  " + "-" * 82)

for name, peak_str, look_start, look_end in events:
    peak_date = pd.Timestamp(peak_str)
    mask = (df.index >= pd.Timestamp(look_start)) & (df.index <= pd.Timestamp(look_end))
    sub_idx = np.where(mask)[0]
    if len(sub_idx) == 0:
        print(f"  {name:<22} {peak_str:>12} {'N/A':>22}")
        continue

    # Find first date aligned P(Crisis) > 0.5
    detect_date = None
    detect_vix = None
    for si in sub_idx:
        if not np.isnan(aligned_crisis[si]) and aligned_crisis[si] > 0.5:
            detect_date = df.index[si]
            detect_vix = df['VIX'].values[si]
            break

    if detect_date is not None:
        lead = (peak_date - detect_date).days
        print(f"  {name:<22} {peak_str:>12} {str(detect_date.date()):>22} {lead:>+8d}d {detect_vix:>16.1f}")
    else:
        print(f"  {name:<22} {peak_str:>12} {'No signal':>22} {'':>8} {'':>16}")

# ═════════════════════════════════════════════════════════════════════════════
#  PLOT: ALIGNED vs UNALIGNED COVID COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
print("\n  Generating plots...")

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig.suptitle("Label-Switching Fix: Aligned vs Unaligned OOS Probabilities (COVID Window)",
             fontsize=14, fontweight='bold')

dates_covid = df.index[covid_idx]
vix_covid = df['VIX'].values[covid_idx]

# Panel 1: VIX
ax = axes[0]
ax.plot(dates_covid, vix_covid, 'steelblue', linewidth=2)
ax.axhline(20, color='orange', linestyle='--', linewidth=0.8)
ax.axhline(30, color='red', linestyle='--', linewidth=0.8)
ax.set_ylabel('VIX', fontsize=11)
ax.set_title('VIX Level', fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 2: Unaligned (raw state with highest VIX heuristic — what hmm_final_study did)
ax = axes[1]
# For unaligned, we need to pick "crisis" state using the same heuristic as before
# Show all 3 raw state probabilities
for si in range(N_STATES):
    ax.plot(dates_covid, prob_unaligned[covid_idx, si], linewidth=1.5, label=f'Raw State {si}', alpha=0.8)
ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8)
ax.set_ylabel('Probability', fontsize=11)
ax.set_title('UNALIGNED (raw state indices — labels may switch at refits)', fontsize=11)
ax.legend(fontsize=8, loc='center left')
ax.set_ylim(-0.05, 1.1)
ax.grid(True, alpha=0.3)
# Mark refits
for e in refit_log:
    if e['date'] >= dates_covid[0] and e['date'] <= dates_covid[-1]:
        ax.axvline(e['date'], color='blue', linestyle=':', linewidth=2, alpha=0.7)

# Panel 3: Aligned
ax = axes[2]
colors = ['steelblue', 'orange', 'crimson']
labels = ['P(Calm)', 'P(Transition)', 'P(Crisis)']
for ci in range(N_STATES):
    ax.plot(dates_covid, prob_aligned[covid_idx, ci], color=colors[ci], linewidth=2, label=labels[ci])
ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8)
ax.set_ylabel('Probability', fontsize=11)
ax.set_title('ALIGNED (Hungarian-matched to canonical Calm/Transition/Crisis)', fontsize=11)
ax.legend(fontsize=9, loc='center left')
ax.set_ylim(-0.05, 1.1)
ax.grid(True, alpha=0.3)
for e in refit_log:
    if e['date'] >= dates_covid[0] and e['date'] <= dates_covid[-1]:
        ax.axvline(e['date'], color='blue', linestyle=':', linewidth=2, alpha=0.7)

# Panel 4: Credit + Momentum features
ax = axes[3]
ax2 = ax.twinx()
ax.plot(dates_covid, df['F3_CREDIT'].values[covid_idx], 'darkgreen', linewidth=1.5, label='Credit')
ax2.plot(dates_covid, df['F6_MOMENTUM'].values[covid_idx], 'purple', linewidth=1.5, label='Momentum')
ax.set_ylabel('Credit log(HYG/LQD)', color='darkgreen', fontsize=10)
ax2.set_ylabel('Momentum (21d ret)', color='purple', fontsize=10)
ax.legend(loc='upper left', fontsize=9)
ax2.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))

plt.tight_layout()
plt.savefig('plots/label_switching_covid.png', dpi=150, bbox_inches='tight')
plt.close()

# Full history plot: aligned probabilities
fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
fig.suptitle("Full History: Aligned OOS Regime Probabilities (3-state HMM)", fontsize=14, fontweight='bold')

axes[0].plot(df.index, df['VIX'], 'steelblue', linewidth=0.5)
axes[0].axhline(20, color='orange', linestyle='--', linewidth=0.5)
axes[0].axhline(30, color='red', linestyle='--', linewidth=0.5)
axes[0].set_ylabel('VIX')
axes[0].grid(True, alpha=0.2)

for ci, (col, lab) in enumerate(zip(colors, labels)):
    axes[1].fill_between(df.index, prob_aligned[:, ci], alpha=0.5, color=col, label=lab)
axes[1].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
axes[1].set_ylabel('Aligned\nProbabilities')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].set_ylim(-0.05, 1.1)
axes[1].grid(True, alpha=0.2)

axes[2].plot(df.index, df['F4_YIELD_CURVE'], 'purple', linewidth=0.5)
axes[2].axhline(0, color='red', linestyle='--', linewidth=0.8)
axes[2].set_ylabel('Yield Curve\n(10Y−IRX)')
axes[2].grid(True, alpha=0.2)
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('plots/label_switching_aligned_full.png', dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: plots/label_switching_covid.png")
print("  Saved: plots/label_switching_aligned_full.png")

# ═════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  SUMMARY")
print("=" * 80)
print(f"\n  Label switches detected: {n_switches} / {len(refit_log)} refits ({100*n_switches/len(refit_log):.1f}%)")

if n_switches > 0:
    print(f"\n  ⚠ LABEL-SWITCHING CONFIRMED.")
    print(f"    The 'COVID 74-day lead' was likely caused by a label permutation")
    print(f"    where the model's 'Calm' state was mislabeled as 'Crisis' before a refit.")
    print(f"\n    AFTER ALIGNMENT:")
    # Find COVID aligned detection
    covid_window = (df.index >= '2019-12-01') & (df.index <= '2020-03-31')
    cw_idx = np.where(covid_window)[0]
    for si in cw_idx:
        if not np.isnan(aligned_crisis[si]) and aligned_crisis[si] > 0.5:
            print(f"    First aligned P(Crisis) > 50%: {df.index[si].date()} (VIX={df['VIX'].values[si]:.1f})")
            break
else:
    print(f"\n  ✓ No label-switching found. The previous results were correctly labeled.")

print("\n" + "=" * 80)
