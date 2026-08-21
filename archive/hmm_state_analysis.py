"""
hmm_state_analysis.py  --  Rigorous HMM validation and feature expansion

Steps:
  1. Fit 2-state and 3-state HMM on log(VIX), compare AIC/BIC
  2. Print transition matrices and implied average regime durations
  3. Measure early-warning capability: how often HMM enters crisis BEFORE VIX > 30
  4. Experiment with adding VVIX as a second feature (one at a time)
  5. Print full comparative report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm
import os

warnings.filterwarnings("ignore")
os.makedirs('plots', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Download data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  DOWNLOADING DATA")
print("=" * 70)

vix_raw  = yf.download('^VIX',  start='2010-01-01', progress=False)
vvix_raw = yf.download('^VVIX', start='2010-01-01', progress=False)

vix  = vix_raw['Close'].squeeze().dropna().sort_index()
vvix = vvix_raw['Close'].squeeze().dropna().sort_index()

# Align to common dates
common_idx = vix.index.intersection(vvix.index)
vix  = vix.loc[common_idx]
vvix = vvix.loc[common_idx]

print(f"  VIX:  {len(vix)} days  ({vix.index[0].date()} → {vix.index[-1].date()})")
print(f"  VVIX: {len(vvix)} days  ({vvix.index[0].date()} → {vvix.index[-1].date()})")

# Feature matrices
X1 = np.log(vix.values).reshape(-1, 1)               # 1-feature: log(VIX)
X2 = np.column_stack([np.log(vix.values),            # 2-feature: log(VIX) + log(VVIX)
                       np.log(vvix.values)])
n  = len(vix)

# ─────────────────────────────────────────────────────────────────────────────
# 2. AIC / BIC helper
# ─────────────────────────────────────────────────────────────────────────────
def count_params(n_states, n_features):
    """
    Free parameters in a full-covariance Gaussian HMM:
      - Transition matrix:   n_states * (n_states - 1)   (each row sums to 1)
      - Initial probs:       n_states - 1
      - Means:               n_states * n_features
      - Covariance (full):   n_states * n_features * (n_features + 1) / 2
    """
    trans   = n_states * (n_states - 1)
    init    = n_states - 1
    means   = n_states * n_features
    covars  = n_states * n_features * (n_features + 1) // 2
    return trans + init + means + covars

def fit_hmm(X, n_states, n_iter=300, random_state=42):
    model = GaussianHMM(n_components=n_states, covariance_type='full',
                        n_iter=n_iter, random_state=random_state)
    model.fit(X)
    ll   = model.score(X) * len(X)       # total log-likelihood
    k    = count_params(n_states, X.shape[1])
    N    = len(X)
    aic  = 2 * k - 2 * ll
    bic  = k * np.log(N) - 2 * ll
    return model, ll, aic, bic, k

def avg_duration(transition_matrix):
    """Average time in each state = 1 / (1 - P(self-transition))."""
    return [1.0 / (1.0 - transition_matrix[i, i]) for i in range(len(transition_matrix))]

def crisis_state_idx(model, feature_idx=0):
    """Return index of state with highest mean on feature_idx (= crisis state)."""
    return int(np.argmax([m[feature_idx] for m in model.means_]))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fit models: 2-state and 3-state, both feature sets
# ─────────────────────────────────────────────────────────────────────────────
configs = [
    ("2-state | log(VIX)",        X1, 2),
    ("3-state | log(VIX)",        X1, 3),
    ("2-state | log(VIX)+log(VVIX)", X2, 2),
    ("3-state | log(VIX)+log(VVIX)", X2, 3),
]

results = []
models  = {}

print("\n" + "=" * 70)
print("  AIC / BIC COMPARISON")
print("=" * 70)
print(f"{'Model':<38} {'K':>5} {'LogLik':>10} {'AIC':>10} {'BIC':>10}")
print("-" * 70)

for label, X, n_states in configs:
    m, ll, aic, bic, k = fit_hmm(X, n_states)
    results.append({'label': label, 'model': m, 'X': X,
                    'n_states': n_states, 'll': ll, 'aic': aic, 'bic': bic, 'k': k})
    models[label] = (m, X, n_states)
    print(f"  {label:<36} {k:>5} {ll:>10.1f} {aic:>10.1f} {bic:>10.1f}")

best_aic = min(results, key=lambda r: r['aic'])
best_bic = min(results, key=lambda r: r['bic'])
print(f"\n  Best AIC: {best_aic['label']}")
print(f"  Best BIC: {best_bic['label']}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Transition matrices and regime durations
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TRANSITION MATRICES & AVERAGE REGIME DURATIONS")
print("=" * 70)

for r in results:
    m, n_s = r['model'], r['n_states']
    ci      = crisis_state_idx(m)
    durations = avg_duration(m.transmat_)
    means_vix = [np.exp(m.means_[s][0]) for s in range(n_s)]  # back to VIX level

    print(f"\n  --- {r['label']} ---")
    print(f"  State means (VIX equiv): {['%.1f' % v for v in means_vix]}")
    print(f"  Crisis state index: {ci}  (VIX ≈ {means_vix[ci]:.1f})")
    print(f"  Transition matrix:")
    for i in range(n_s):
        row = "  " + "  ".join(f"P({i}→{j})={m.transmat_[i,j]:.4f}" for j in range(n_s))
        print(row)
    print(f"  Avg durations (days): {['%.1f' % d for d in durations]}")
    print(f"  Crisis avg duration: {durations[ci]:.1f} days = {durations[ci]/21:.1f} months")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Early-warning analysis (1-feature 2-state model, full expanding window)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  EARLY-WARNING ANALYSIS  (2-state log(VIX), expanding window)")
print("=" * 70)

BURN_IN = 500
REFIT   = 60

prob_crisis_oos = np.zeros(n)
crisis_st = 0

for t in range(BURN_IN, n):
    if t == BURN_IN or t % REFIT == 0:
        m_t = GaussianHMM(n_components=2, covariance_type='full',
                          n_iter=100, random_state=42)
        m_t.fit(X1[:t])
        crisis_st = crisis_state_idx(m_t)
    probs = m_t.predict_proba(X1[:t+1])
    prob_crisis_oos[t] = probs[-1, crisis_st]

prob_crisis_oos[:BURN_IN] = prob_crisis_oos[BURN_IN]

# Find VIX-30 crossing events
vix_arr  = vix.values
vix_above_30 = vix_arr > 30

# Identify start of each VIX>30 episode
episodes = []
in_ep = False
for t in range(1, n):
    if vix_above_30[t] and not vix_above_30[t-1]:
        in_ep = True
        ep_start = t
    elif not vix_above_30[t] and vix_above_30[t-1] and in_ep:
        in_ep = False
        episodes.append((ep_start, t-1))
if in_ep:
    episodes.append((ep_start, n-1))

print(f"\n  VIX > 30 episodes found: {len(episodes)}")
lead_days = []

for (es, ee) in episodes:
    if es < BURN_IN:
        continue
    ep_date = vix.index[es]
    # Look back up to 30 days before episode start for HMM signal
    lookback = max(BURN_IN, es - 30)
    # Find first day HMM crossed 50% in the 30 days before the episode
    pre = prob_crisis_oos[lookback:es]
    pre_idx = np.where(pre > 0.5)[0]
    if len(pre_idx) > 0:
        first_signal = lookback + pre_idx[0]
        lead = es - first_signal
        lead_days.append(lead)
        print(f"  Episode start: {ep_date.date()}  VIX={vix_arr[es]:.1f}  "
              f"HMM crossed 50% on {vix.index[first_signal].date()}  "
              f"=> Lead: {lead:+d} days ({'EARLY' if lead > 0 else 'LATE'})")
    else:
        print(f"  Episode start: {ep_date.date()}  VIX={vix_arr[es]:.1f}  "
              f"HMM had NO early signal in prior 30 days")
        lead_days.append(np.nan)

valid_leads = [x for x in lead_days if not np.isnan(x)]
if valid_leads:
    early = [x for x in valid_leads if x > 0]
    print(f"\n  Episodes with early signal: {len(early)} / {len(valid_leads)}")
    print(f"  Median lead time (days): {np.median(valid_leads):.1f}")
    print(f"  Mean lead time (days):   {np.mean(valid_leads):.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Plot: 2-state vs 3-state regime probabilities
# ─────────────────────────────────────────────────────────────────────────────
m2, _, _ = models["2-state | log(VIX)"]
m3, _, _ = models["3-state | log(VIX)"]

p2 = m2.predict_proba(X1)[:, crisis_state_idx(m2)]
# For 3-state: crisis = highest-VIX state; mid = middle; calm = lowest
state_means3 = [float(m3.means_[s][0]) for s in range(3)]
order3 = np.argsort(state_means3)   # calm, mid, crisis by increasing mean
calm3, mid3, crisis3 = order3[0], order3[1], order3[2]

p3_calm   = m3.predict_proba(X1)[:, calm3]
p3_mid    = m3.predict_proba(X1)[:, mid3]
p3_crisis = m3.predict_proba(X1)[:, crisis3]

fig, axes = plt.subplots(4, 1, figsize=(17, 13), sharex=True)
fig.suptitle("2-State vs 3-State HMM: Regime Probabilities", fontsize=14, fontweight='bold')

axes[0].plot(vix.index, vix.values, color='steelblue', linewidth=0.8)
axes[0].axhline(30, color='red', linestyle='--', linewidth=0.8, label='VIX=30')
axes[0].axhline(20, color='orange', linestyle='--', linewidth=0.8, label='VIX=20')
axes[0].set_ylabel('VIX')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].fill_between(vix.index, p2, alpha=0.7, color='crimson', label='P(Crisis) - 2-state')
axes[1].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
axes[1].set_ylabel('P(Crisis)\n2-state')
axes[1].set_ylim(-0.02, 1.05)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

axes[2].fill_between(vix.index, p3_crisis, alpha=0.7, color='crimson', label='P(Crisis)')
axes[2].fill_between(vix.index, p3_mid,    alpha=0.5, color='orange',  label='P(Transition)')
axes[2].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
axes[2].set_ylabel('Probabilities\n3-state')
axes[2].set_ylim(-0.02, 1.05)
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

axes[3].fill_between(vix.index, prob_crisis_oos, alpha=0.7, color='darkred',
                     label='P(Crisis) - OOS (expanding window)')
axes[3].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
axes[3].set_ylabel('P(Crisis) OOS\n2-state')
axes[3].set_ylim(-0.02, 1.05)
axes[3].legend(fontsize=8)
axes[3].grid(True, alpha=0.3)

axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[3].xaxis.set_major_locator(mdates.YearLocator())
plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/hmm_2state_vs_3state.png', dpi=150, bbox_inches='tight')
print("\nSaved: plots/hmm_2state_vs_3state.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Plot: log(VIX) vs log(VIX)+log(VVIX) crisis probabilities
# ─────────────────────────────────────────────────────────────────────────────
m2v, _, _ = models["2-state | log(VIX)+log(VVIX)"]
p2v = m2v.predict_proba(X2)[:, crisis_state_idx(m2v)]

fig, axes = plt.subplots(3, 1, figsize=(17, 10), sharex=True)
fig.suptitle("Feature Expansion: 1-Feature vs 2-Feature HMM", fontsize=14, fontweight='bold')

axes[0].plot(vix.index, vix.values, 'steelblue', linewidth=0.8, label='VIX')
ax0b = axes[0].twinx()
ax0b.plot(vix.index, vvix.values, 'purple', linewidth=0.6, alpha=0.7, label='VVIX')
ax0b.set_ylabel('VVIX', color='purple')
axes[0].set_ylabel('VIX')
axes[0].grid(True, alpha=0.3)
lines1, labels1 = axes[0].get_legend_handles_labels()
lines2, labels2 = ax0b.get_legend_handles_labels()
axes[0].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

axes[1].fill_between(vix.index, p2,  alpha=0.7, color='crimson', label='P(Crisis) 1-feat: log(VIX)')
axes[1].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
axes[1].set_ylabel('P(Crisis)')
axes[1].set_ylim(-0.02, 1.05)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

axes[2].fill_between(vix.index, p2v, alpha=0.7, color='darkorchid', label='P(Crisis) 2-feat: log(VIX)+log(VVIX)')
axes[2].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
axes[2].set_ylabel('P(Crisis)')
axes[2].set_ylim(-0.02, 1.05)
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[2].xaxis.set_major_locator(mdates.YearLocator())
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/hmm_feature_expansion.png', dpi=150, bbox_inches='tight')
print("Saved: plots/hmm_feature_expansion.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 8. Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SUMMARY TABLE")
print("=" * 70)
print(f"{'Model':<40} {'AIC':>10} {'BIC':>10} {'ΔAIC':>8} {'ΔBIC':>8}")
print("-" * 70)
min_aic = min(r['aic'] for r in results)
min_bic = min(r['bic'] for r in results)
for r in sorted(results, key=lambda x: x['bic']):
    da = r['aic'] - min_aic
    db = r['bic'] - min_bic
    star_aic = " ★" if da == 0 else ""
    star_bic = " ★" if db == 0 else ""
    print(f"  {r['label']:<38} {r['aic']:>10.1f}{star_aic}  {r['bic']:>10.1f}{star_bic}  "
          f"{da:>+8.1f}  {db:>+8.1f}")
print()
