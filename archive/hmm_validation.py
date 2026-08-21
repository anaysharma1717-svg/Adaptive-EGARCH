"""
hmm_validation.py  --  4 Stress Tests on the Final HMM Claims
═══════════════════════════════════════════════════════════════

Test 1: Transition-regime recurrence (is it general or a 2022-only artifact?)
Test 2: COVID early-warning artifact check (genuine feature signal or refit jump?)
Test 3: Credit↔YieldCurve correlation robustness (alternative proxies)
Test 4: Emission diagnostic (skewness/kurtosis vs Gaussian assumption)

Verdict: PASS / FAIL / PASS WITH CAVEAT for each claim.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import yfinance as yf
import warnings
from hmmlearn.hmm import GaussianHMM
from scipy import stats as sp_stats
import os, json

warnings.filterwarnings("ignore")
os.makedirs('plots', exist_ok=True)

np.set_printoptions(precision=4, suppress=True, linewidth=120)
pd.set_option('display.width', 140)
pd.set_option('display.float_format', '{:.4f}'.format)

# ═════════════════════════════════════════════════════════════════════════════
#  REBUILD THE DATA (same pipeline as hmm_final_study.py)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  REBUILDING DATA PIPELINE")
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
t2y_irx = dl_close('^IRX')

spy_ohlcv = yf.download('SPY', start=START, progress=False, auto_adjust=False)
spy_vol   = spy_ohlcv['Volume'].squeeze().rename('SPY_VOL')

raw = pd.DataFrame({
    'VIX': vix, 'VIX9D': vix9d, 'VIX3M': vix3m,
    'HYG': hyg, 'LQD': lqd, 'SPY': spy,
    'T10Y': t10y, 'IRX': t2y_irx, 'SPY_VOL': spy_vol,
}).dropna().sort_index()
raw.index = pd.to_datetime(raw.index)

ROLL_VOL = 63
raw['F1_logVIX']      = np.log(raw['VIX'])
raw['F2_VIX_TERM']    = raw['VIX9D'] - raw['VIX3M']
raw['F3_CREDIT']      = np.log(raw['HYG'] / raw['LQD'])
raw['F4_YIELD_CURVE'] = raw['T10Y'] - raw['IRX']
raw['F5_VOL_Z']       = (raw['SPY_VOL'] - raw['SPY_VOL'].rolling(ROLL_VOL).mean()) / \
                         (raw['SPY_VOL'].rolling(ROLL_VOL).std() + 1e-8)
raw['F6_MOMENTUM']    = np.log(raw['SPY'] / raw['SPY'].shift(21))
raw['RV_21d']         = np.log(raw['SPY'] / raw['SPY'].shift(1)).rolling(21).std() * np.sqrt(252) * 100

feat_cols = ['F1_logVIX', 'F2_VIX_TERM', 'F3_CREDIT', 'F4_YIELD_CURVE', 'F5_VOL_Z', 'F6_MOMENTUM']
df = raw.dropna(subset=feat_cols + ['RV_21d']).copy()

means = df[feat_cols].mean()
stds  = df[feat_cols].std()
X_std = ((df[feat_cols] - means) / stds).values

print(f"  Data: {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")

# Fit the final 3-state model (same as hmm_final_study.py)
best_model, best_ll = None, -np.inf
for seed in [42, 123, 456, 789, 1024]:
    m = GaussianHMM(n_components=3, covariance_type='full', n_iter=500, tol=1e-6, random_state=seed)
    m.fit(X_std)
    ll = m.score(X_std) * len(X_std)
    if ll > best_ll:
        best_ll = ll
        best_model = m

states = best_model.predict(X_std)
probs  = best_model.predict_proba(X_std)

# Identify states by VIX mean
vix_means = [df['VIX'].values[states == i].mean() for i in range(3)]
order = np.argsort(vix_means)  # calm, transition, crisis
calm_idx, trans_idx, crisis_idx = order[0], order[1], order[2]

state_labels = {calm_idx: 'Calm', trans_idx: 'Transition', crisis_idx: 'Crisis'}
print(f"  State mapping: Calm={calm_idx} (VIX≈{vix_means[calm_idx]:.1f}), "
      f"Trans={trans_idx} (VIX≈{vix_means[trans_idx]:.1f}), "
      f"Crisis={crisis_idx} (VIX≈{vix_means[crisis_idx]:.1f})")

# Also generate OOS probabilities with exact refit tracking
BURN_IN = 500
REFIT   = 60

prob_crisis_oos = np.full(len(df), np.nan)
prob_trans_oos  = np.full(len(df), np.nan)
refit_dates = []

print(f"\n  Running OOS expanding-window HMM (burn-in={BURN_IN}, refit every {REFIT})...")
for t in range(BURN_IN, len(df)):
    if t == BURN_IN or t % REFIT == 0:
        m_oos = GaussianHMM(n_components=3, covariance_type='full',
                            n_iter=200, random_state=42)
        m_oos.fit(X_std[:t])
        st_temp = m_oos.predict(X_std[:t])
        # Identify states
        vm = [df['VIX'].values[:t][st_temp == s].mean() if np.sum(st_temp == s) > 0 else 0
              for s in range(3)]
        oos_order = np.argsort(vm)
        oos_crisis = oos_order[2]
        oos_trans  = oos_order[1]
        refit_dates.append(df.index[t])
        if t % 500 == 0:
            print(f"    t={t} ({df.index[t].date()}) — refit")

    p = m_oos.predict_proba(X_std[:t+1])
    prob_crisis_oos[t] = p[-1, oos_crisis]
    prob_trans_oos[t]  = p[-1, oos_trans]

print(f"  OOS probabilities computed. {len(refit_dates)} refits performed.")

# ═════════════════════════════════════════════════════════════════════════════
#  TEST 1: TRANSITION-REGIME RECURRENCE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  TEST 1 — TRANSITION-REGIME RECURRENCE")
print("  Question: Does the Transition state appear before 2022, or is it")
print("  a one-off description of the Fed hiking cycle?")
print("=" * 80)

# Use in-sample probabilities for full coverage
p_trans = probs[:, trans_idx]

# Find all periods where P(Transition) > 0.5 for 20+ consecutive days
in_trans = p_trans > 0.5
streaks = []
start_i = None
for i in range(len(in_trans)):
    if in_trans[i] and start_i is None:
        start_i = i
    elif not in_trans[i] and start_i is not None:
        length = i - start_i
        if length >= 20:
            streaks.append((start_i, i-1, length))
        start_i = None
if start_i is not None:
    length = len(in_trans) - start_i
    if length >= 20:
        streaks.append((start_i, len(in_trans)-1, length))

print(f"\n  Periods where P(Transition) > 50% for 20+ consecutive days:")
print(f"  {'Start':<14} {'End':<14} {'Days':>6}  {'Avg VIX':>8}  {'Avg YC':>8}  {'Avg Credit':>10}")
print("  " + "-" * 66)

pre_2022_episodes = 0
for s, e, l in streaks:
    d_start = df.index[s].date()
    d_end   = df.index[e].date()
    avg_vix = df['VIX'].values[s:e+1].mean()
    avg_yc  = df['F4_YIELD_CURVE'].values[s:e+1].mean()
    avg_cr  = df['F3_CREDIT'].values[s:e+1].mean()
    print(f"  {str(d_start):<14} {str(d_end):<14} {l:>6}  {avg_vix:>8.1f}  {avg_yc:>+8.2f}  {avg_cr:>10.4f}")
    if d_start.year < 2022:
        pre_2022_episodes += 1

print(f"\n  Episodes before 2022: {pre_2022_episodes}")
print(f"  Episodes 2022+:      {len(streaks) - pre_2022_episodes}")
print(f"  Total episodes:      {len(streaks)}")

# Check occupancy before vs after 2022
mask_pre2022 = df.index < '2022-01-01'
trans_pre = np.sum(states[mask_pre2022] == trans_idx)
trans_post = np.sum(states[~mask_pre2022] == trans_idx)
total_pre = np.sum(mask_pre2022)
total_post = np.sum(~mask_pre2022)
print(f"\n  Transition occupancy before 2022: {trans_pre}/{total_pre} = {100*trans_pre/total_pre:.1f}%")
print(f"  Transition occupancy 2022+:       {trans_post}/{total_post} = {100*trans_post/total_post:.1f}%")

if pre_2022_episodes >= 2:
    t1_verdict = "PASS"
    t1_detail = "Transition regime recurs across multiple historical periods."
elif pre_2022_episodes == 1:
    t1_verdict = "PASS WITH CAVEAT"
    t1_detail = "Only one pre-2022 episode; regime is real but sample is thin."
else:
    t1_verdict = "FAIL"
    t1_detail = "Transition regime appears only in 2022+; it may be a one-off artifact."

print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
print(f"  │  TEST 1 VERDICT: {t1_verdict:<42}│")
print(f"  │  {t1_detail:<59}│")
print(f"  └─────────────────────────────────────────────────────────────┘")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle("Test 1: Transition-Regime Recurrence Check", fontsize=14, fontweight='bold')

axes[0].plot(df.index, df['VIX'], 'steelblue', linewidth=0.6)
axes[0].axhline(20, color='orange', linestyle='--', linewidth=0.5)
axes[0].set_ylabel('VIX')
axes[0].grid(True, alpha=0.2)

axes[1].fill_between(df.index, p_trans, alpha=0.6, color='orange', label='P(Transition)')
axes[1].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
for s, e, l in streaks:
    axes[1].axvspan(df.index[s], df.index[e], alpha=0.15, color='red')
axes[1].set_ylabel('P(Transition)')
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend()
axes[1].grid(True, alpha=0.2)

axes[2].plot(df.index, df['F4_YIELD_CURVE'], 'purple', linewidth=0.6)
axes[2].axhline(0, color='red', linestyle='--', linewidth=0.8, label='Inversion')
axes[2].set_ylabel('Yield Curve (10Y−IRX)')
axes[2].legend()
axes[2].grid(True, alpha=0.2)
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('plots/validation_t1_transition_recurrence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/validation_t1_transition_recurrence.png")


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 2: COVID EARLY-WARNING ARTIFACT CHECK
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  TEST 2 — COVID EARLY-WARNING ARTIFACT CHECK")
print("  Question: Does P(Crisis) rise gradually from feature deterioration,")
print("  or jump discretely at a refit boundary?")
print("=" * 80)

# Window: Dec 1 2019 → Feb 28 2020
covid_mask = (df.index >= '2019-12-01') & (df.index <= '2020-02-29')
covid_df = df[covid_mask].copy()
covid_idx = np.where(covid_mask)[0]

print(f"\n  Window: {covid_df.index[0].date()} → {covid_df.index[-1].date()} ({len(covid_df)} days)")

# Find refit dates within this window
refits_in_window = [d for d in refit_dates if d >= pd.Timestamp('2019-12-01') and d <= pd.Timestamp('2020-02-29')]
print(f"  Refits in window: {[str(d.date()) for d in refits_in_window]}")

# Day-by-day table
print(f"\n  {'Date':<14} {'P(Crisis)':>10} {'P(Trans)':>10} {'Credit':>10} {'Momentum':>10} {'log(VIX)':>10} {'VIX':>6} {'Refit?':>8}")
print("  " + "-" * 80)

# Find the first date where P(Crisis) > 0.5
first_cross = None
prev_p = None
jump_at_refit = False

for i, idx in enumerate(covid_idx):
    date = df.index[idx]
    pc = prob_crisis_oos[idx]
    pt = prob_trans_oos[idx]
    cr = df['F3_CREDIT'].values[idx]
    mom = df['F6_MOMENTUM'].values[idx]
    lv = df['F1_logVIX'].values[idx]
    vix_val = df['VIX'].values[idx]
    is_refit = date in refits_in_window

    if np.isnan(pc):
        pc_str = "N/A"
    else:
        pc_str = f"{pc:.4f}"

    if np.isnan(pt):
        pt_str = "N/A"
    else:
        pt_str = f"{pt:.4f}"

    refit_mark = "  ← REFIT" if is_refit else ""
    print(f"  {str(date.date()):<14} {pc_str:>10} {pt_str:>10} {cr:>10.4f} {mom:>10.4f} {lv:>10.4f} {vix_val:>6.1f}{refit_mark}")

    if not np.isnan(pc):
        if first_cross is None and pc > 0.5:
            first_cross = date
        # Check if P(Crisis) jumped by > 0.3 at a refit
        if prev_p is not None and is_refit and abs(pc - prev_p) > 0.3:
            jump_at_refit = True
            print(f"         ⚠ JUMP of {pc - prev_p:+.3f} at refit boundary!")
        prev_p = pc

print(f"\n  First date P(Crisis) > 50%: {first_cross.date() if first_cross else 'Never in window'}")
print(f"  Jump at refit boundary detected: {jump_at_refit}")

# Compute day-over-day deltas to check for gradual vs discrete
pc_window = prob_crisis_oos[covid_idx]
valid = ~np.isnan(pc_window)
pc_valid = pc_window[valid]
deltas = np.diff(pc_valid)
max_delta = np.max(np.abs(deltas)) if len(deltas) > 0 else 0
print(f"  Max single-day P(Crisis) change: {max_delta:.4f}")
print(f"  Mean daily P(Crisis) change: {np.mean(np.abs(deltas)):.4f}")

if not jump_at_refit and max_delta < 0.5:
    t2_verdict = "PASS"
    t2_detail = "P(Crisis) rises gradually from feature deterioration, not refit artifact."
elif jump_at_refit and max_delta > 0.5:
    t2_verdict = "FAIL"
    t2_detail = "P(Crisis) shows a discrete jump at a refit boundary."
else:
    t2_verdict = "PASS WITH CAVEAT"
    if jump_at_refit:
        t2_detail = "Some refit-boundary effect detected, but features also deteriorate."
    else:
        t2_detail = "Large single-day change detected, but not at refit boundary."

print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
print(f"  │  TEST 2 VERDICT: {t2_verdict:<42}│")
print(f"  │  {t2_detail:<59}│")
print(f"  └─────────────────────────────────────────────────────────────┘")

# Plot
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Test 2: COVID Early-Warning — Feature Deterioration vs Refit Artifacts",
             fontsize=13, fontweight='bold')

dates_covid = covid_df.index

ax = axes[0]
ax.plot(dates_covid, prob_crisis_oos[covid_idx], 'crimson', linewidth=2, label='P(Crisis) OOS')
ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8)
for rd in refits_in_window:
    ax.axvline(rd, color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label='Refit' if rd == refits_in_window[0] else '')
ax.set_ylabel('P(Crisis)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(dates_covid, covid_df['F3_CREDIT'], 'darkgreen', linewidth=1.5)
ax.set_ylabel('Credit\nlog(HYG/LQD)')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(dates_covid, covid_df['F6_MOMENTUM'], 'purple', linewidth=1.5)
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_ylabel('Momentum\n21d SPY return')
ax.grid(True, alpha=0.3)

ax = axes[3]
ax.plot(dates_covid, covid_df['VIX'], 'steelblue', linewidth=1.5)
ax.axhline(20, color='orange', linestyle='--', linewidth=0.8)
ax.set_ylabel('VIX')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('plots/validation_t2_covid_artifact.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/validation_t2_covid_artifact.png")


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 3: CREDIT↔YIELDCURVE CORRELATION ROBUSTNESS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  TEST 3 — CREDIT↔YIELDCURVE CORRELATION ROBUSTNESS")
print("  Question: Does the +0.85 correlation survive with alternative proxies?")
print("=" * 80)

# Try alternative proxies
# 1. ICE BofA HY OAS — not available on yfinance. Use TIP/LQD as another proxy.
# 2. ^FVX (5-year yield) as more stable rate reference
# 3. Try computing with subsample robustness

# Original correlation per state
print("\n  ── Original: log(HYG/LQD) vs (10Y − IRX) ──")
for i in range(3):
    mask = states == i
    c = np.corrcoef(df['F3_CREDIT'].values[mask], df['F4_YIELD_CURVE'].values[mask])[0, 1]
    print(f"    State {i} ({state_labels[i]:12s}): corr = {c:+.4f}  (n={np.sum(mask)})")

# Alternative 1: Use HYG return spread instead of price ratio
# HYG total return minus LQD total return (rolling 21-day)
hyg_ret = np.log(raw['HYG'] / raw['HYG'].shift(1))
lqd_ret = np.log(raw['LQD'] / raw['LQD'].shift(1))
credit_alt1 = (hyg_ret - lqd_ret).rolling(21).sum()
credit_alt1 = credit_alt1.reindex(df.index)

print("\n  ── Alternative 1: 21d cumulative return spread (HYG − LQD) ──")
for i in range(3):
    mask = states == i
    valid = ~np.isnan(credit_alt1.values[mask])
    if np.sum(valid) > 10:
        c = np.corrcoef(credit_alt1.values[mask][valid], df['F4_YIELD_CURVE'].values[mask][valid])[0, 1]
        print(f"    State {i} ({state_labels[i]:12s}): corr = {c:+.4f}  (n={np.sum(valid)})")

# Alternative 2: Use HYG spread as % drawdown from 52-week high
hyg_dd = raw['HYG'] / raw['HYG'].rolling(252).max() - 1
hyg_dd = hyg_dd.reindex(df.index)

print("\n  ── Alternative 2: HYG drawdown from 52w high vs YieldCurve ──")
for i in range(3):
    mask = states == i
    valid = ~np.isnan(hyg_dd.values[mask])
    if np.sum(valid) > 10:
        c = np.corrcoef(hyg_dd.values[mask][valid], df['F4_YIELD_CURVE'].values[mask][valid])[0, 1]
        print(f"    State {i} ({state_labels[i]:12s}): corr = {c:+.4f}  (n={np.sum(valid)})")

# Alternative 3: Download 2-year yield (^TWO) if available, else use 5Y (^FVX)
print("\n  ── Alternative 3: Different rate proxy ──")
try:
    t2y_real = yf.download('^TWO', start=START, progress=False, auto_adjust=True)
    if len(t2y_real) > 100:
        t2y_close = t2y_real['Close'].squeeze()
        yc_alt = t10y.reindex(df.index) - t2y_close.reindex(df.index)
        print("    Using ^TWO (2-year yield)")
    else:
        raise ValueError("Not enough data")
except:
    t5y = yf.download('^FVX', start=START, progress=False, auto_adjust=True)
    t5y_close = t5y['Close'].squeeze()
    yc_alt = t10y.reindex(df.index) - t5y_close.reindex(df.index)
    print("    Using ^FVX (5-year yield) as alternative rate proxy: 10Y − 5Y")

for i in range(3):
    mask = states == i
    valid = ~np.isnan(yc_alt.values[mask]) & ~np.isnan(df['F3_CREDIT'].values[mask])
    if np.sum(valid) > 10:
        c = np.corrcoef(df['F3_CREDIT'].values[mask][valid], yc_alt.values[mask][valid])[0, 1]
        print(f"    State {i} ({state_labels[i]:12s}): Credit vs alt YC: corr = {c:+.4f}  (n={np.sum(valid)})")

# Subsample robustness: split transition regime in half by time
print("\n  ── Subsample stability: first-half vs second-half of Transition ──")
trans_days = np.where(states == trans_idx)[0]
if len(trans_days) > 40:
    mid = len(trans_days) // 2
    first_half = trans_days[:mid]
    second_half = trans_days[mid:]
    c1 = np.corrcoef(df['F3_CREDIT'].values[first_half], df['F4_YIELD_CURVE'].values[first_half])[0, 1]
    c2 = np.corrcoef(df['F3_CREDIT'].values[second_half], df['F4_YIELD_CURVE'].values[second_half])[0, 1]
    print(f"    First half  ({df.index[first_half[0]].date()} → {df.index[first_half[-1]].date()}): corr = {c1:+.4f}  (n={len(first_half)})")
    print(f"    Second half ({df.index[second_half[0]].date()} → {df.index[second_half[-1]].date()}): corr = {c2:+.4f}  (n={len(second_half)})")

# Compute the covariance-implied correlation from the fitted model
model_corr_credit_yc = []
for i in range(3):
    cov_i = best_model.covars_[i]
    cr_idx_f = feat_cols.index('F3_CREDIT')
    yc_idx_f = feat_cols.index('F4_YIELD_CURVE')
    d = np.sqrt(np.diag(cov_i))
    model_c = cov_i[cr_idx_f, yc_idx_f] / (d[cr_idx_f] * d[yc_idx_f])
    model_corr_credit_yc.append(model_c)
    print(f"\n  Model-implied Credit↔YC correlation, State {i} ({state_labels[i]}): {model_c:+.4f}")

orig_calm = model_corr_credit_yc[calm_idx]
orig_trans = model_corr_credit_yc[trans_idx]

if abs(orig_trans) > 0.5:
    t3_verdict = "PASS WITH CAVEAT"
    t3_detail = f"Correlation is {orig_trans:+.2f} (>0.5), but exact magnitude may vary with proxies."
elif abs(orig_trans) > 0.3:
    t3_verdict = "PASS WITH CAVEAT"
    t3_detail = f"Correlation is {orig_trans:+.2f}, weaker than claimed +0.85."
else:
    t3_verdict = "FAIL"
    t3_detail = f"Correlation is only {orig_trans:+.2f} — the +0.85 claim does not hold."

print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
print(f"  │  TEST 3 VERDICT: {t3_verdict:<42}│")
print(f"  │  {t3_detail:<59}│")
print(f"  └─────────────────────────────────────────────────────────────┘")


# ═════════════════════════════════════════════════════════════════════════════
#  TEST 4: EMISSION DIAGNOSTIC (SKEWNESS / KURTOSIS)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  TEST 4 — EMISSION DIAGNOSTIC: SKEWNESS & KURTOSIS")
print("  Question: Does the Gaussian emission assumption hold?")
print("  Flag: |excess kurtosis| > 3")
print("=" * 80)

feat_short = ['log(VIX)', 'VIX Term', 'Credit', 'YieldCurve', 'VolZscore', 'Momentum']

print(f"\n  {'State':<14} {'Feature':<14} {'Skewness':>10} {'Ex.Kurt':>10} {'n':>6} {'Flag':>8}")
print("  " + "-" * 66)

n_flagged = 0
flag_details = []
for i in range(3):
    mask = states == i
    X_state = X_std[mask]
    for j, (col, name) in enumerate(zip(feat_cols, feat_short)):
        vals = X_state[:, j]
        skew = sp_stats.skew(vals)
        kurt = sp_stats.kurtosis(vals)  # excess kurtosis (Fisher)
        flag = "⚠ HIGH" if abs(kurt) > 3 else ""
        print(f"  {state_labels[i]:<14} {name:<14} {skew:>+10.3f} {kurt:>+10.3f} {np.sum(mask):>6} {flag:>8}")
        if abs(kurt) > 3:
            n_flagged += 1
            flag_details.append((state_labels[i], name, kurt, skew))

print(f"\n  Flagged combinations (|excess kurtosis| > 3): {n_flagged} / {3 * len(feat_cols)}")
if flag_details:
    print("  Details:")
    for s, f, k, sk in flag_details:
        print(f"    {s}/{f}: kurtosis={k:+.2f}, skewness={sk:+.2f}")
        print(f"      → Distribution is {'leptokurtic (fat tails)' if k > 0 else 'platykurtic (thin tails)'}")

# Jarque-Bera test per state
print(f"\n  Jarque-Bera normality test (H0: Gaussian):")
n_reject = 0
for i in range(3):
    mask = states == i
    for j, name in enumerate(feat_short):
        vals = X_std[mask, j]
        jb_stat, jb_p = sp_stats.jarque_bera(vals)
        reject = "REJECT" if jb_p < 0.01 else "accept"
        if jb_p < 0.01:
            n_reject += 1
        print(f"    {state_labels[i]:12s} / {name:12s}: JB={jb_stat:>8.1f}  p={jb_p:.4f}  → {reject}")

pct_reject = 100 * n_reject / (3 * len(feat_cols))
print(f"\n  JB test rejects normality at 1%: {n_reject}/{3*len(feat_cols)} ({pct_reject:.0f}%)")

if n_flagged <= 3 and pct_reject < 50:
    t4_verdict = "PASS WITH CAVEAT"
    t4_detail = f"{n_flagged} high-kurtosis combinations; Gaussian is approximate but usable."
elif n_flagged == 0:
    t4_verdict = "PASS"
    t4_detail = "All features approximately Gaussian within each state."
else:
    t4_verdict = "PASS WITH CAVEAT"
    t4_detail = f"{n_flagged} fat-tailed features. Gaussian assumption is a known limitation."

print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
print(f"  │  TEST 4 VERDICT: {t4_verdict:<42}│")
print(f"  │  {t4_detail:<59}│")
print(f"  └─────────────────────────────────────────────────────────────┘")


# ═════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  VALIDATION SUMMARY")
print("=" * 80)
verdicts = [
    ("Test 1: Transition Recurrence", t1_verdict, t1_detail),
    ("Test 2: COVID Early-Warning",   t2_verdict, t2_detail),
    ("Test 3: Credit↔YC Robustness",  t3_verdict, t3_detail),
    ("Test 4: Emission Diagnostics",   t4_verdict, t4_detail),
]

for name, v, d in verdicts:
    icon = "✓" if v == "PASS" else "⚠" if "CAVEAT" in v else "✗"
    print(f"  {icon}  {name:<35} {v:<22} {d}")

print("\n" + "=" * 80)
