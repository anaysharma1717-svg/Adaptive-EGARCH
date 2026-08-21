"""
hmm_rolling_relative.py  --  Rolling-Relative Standardization Fix for Non-Stationary Features
═══════════════════════════════════════════════════════════════════════════════════════════════
Root cause: Yield Curve and Credit are non-stationary. Their "normal" baseline drifts
across Fed regimes. Fix: replace global z-score with 756-day (3yr) rolling z-score so
the HMM sees relative positioning, not absolute chronological levels.

Features:
  Globally standardized (mean-reverting, keep as-is):
    F1_logVIX, F2_VIX_TERM, F5_VOL_Z, F6_MOMENTUM
  Rolling-relative (non-stationary, apply fix):
    F3_CREDIT     (log HYG/LQD) — 756-day rolling z-score
    F4_YIELD_CURVE (10Y−IRX)    — 756-day rolling z-score
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
print("  REBUILDING DATA WITH ROLLING-RELATIVE STANDARDIZATION")
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

# Raw feature construction
raw['F1_logVIX']      = np.log(raw['VIX'])
raw['F2_VIX_TERM']    = raw['VIX9D'] - raw['VIX3M']
raw['F3_CREDIT_RAW']  = np.log(raw['HYG'] / raw['LQD'])
raw['F4_YC_RAW']      = raw['T10Y'] - raw['IRX']
raw['F5_VOL_Z']       = (raw['SPY_VOL'] - raw['SPY_VOL'].rolling(63).mean()) / \
                         (raw['SPY_VOL'].rolling(63).std() + 1e-8)
raw['F6_MOMENTUM']    = np.log(raw['SPY'] / raw['SPY'].shift(21))
raw['RV_21d']         = np.log(raw['SPY'] / raw['SPY'].shift(1)).rolling(21).std() * np.sqrt(252) * 100

ROLL_WIN = 756  # 3 years of trading days

# Apply rolling-relative z-score ONLY to the two non-stationary features
for col, out in [('F3_CREDIT_RAW', 'F3_CREDIT_REL'), ('F4_YC_RAW', 'F4_YC_REL')]:
    roll_mean = raw[col].rolling(ROLL_WIN, min_periods=252).mean()
    roll_std  = raw[col].rolling(ROLL_WIN, min_periods=252).std()
    raw[out]  = (raw[col] - roll_mean) / (roll_std + 1e-8)

# Feature set: 4 stationary (global z-score) + 2 rolling-relative
feat_cols_raw   = ['F1_logVIX', 'F2_VIX_TERM', 'F3_CREDIT_REL', 'F4_YC_REL', 'F5_VOL_Z', 'F6_MOMENTUM']
feat_names      = ['logVIX', 'VIXTerm', 'Credit(3yr-rel)', 'YldCrv(3yr-rel)', 'VolZ', 'Mom']

df = raw.dropna(subset=feat_cols_raw + ['RV_21d']).copy()

TRAIN_END = pd.Timestamp('2024-06-30')
is_mask  = df.index <= TRAIN_END
oos_mask = df.index > TRAIN_END

# Global z-score the 4 stationary features on IS data only
stationary_cols  = ['F1_logVIX', 'F2_VIX_TERM', 'F5_VOL_Z', 'F6_MOMENTUM']
rolling_rel_cols = ['F3_CREDIT_REL', 'F4_YC_REL']

g_means = df.loc[is_mask, stationary_cols].mean()
g_stds  = df.loc[is_mask, stationary_cols].std()

X_all = df[feat_cols_raw].copy()
X_all[stationary_cols] = (X_all[stationary_cols] - g_means) / g_stds
# Rolling-relative cols are already z-scores, no further global scaling needed
X_std = X_all.values

X_is = X_std[is_mask]

print(f"  Total rows: {len(df)} ({df.index[0].date()} → {df.index[-1].date()})")
print(f"  IS rows  : {is_mask.sum()} (up to {TRAIN_END.date()})")
print(f"  OOS rows : {oos_mask.sum()} ({df.index[oos_mask][0].date()} → {df.index[-1].date()})")

# Quick sanity: show rolling-relative YC value at key dates
print("\n  Rolling-relative YC values at key dates (should be ~0 regardless of regime):")
key_dates = ['2014-01-02', '2018-12-28', '2021-06-01', '2022-10-03', '2023-06-01']
for d in key_dates:
    try:
        row = df.loc[d]
        print(f"    {d}: YC_raw={row['F4_YC_RAW']:+.2f}  YC_rel={row['F4_YC_REL']:+.2f}  "
              f"Credit_raw={row['F3_CREDIT_RAW']:.3f}  Credit_rel={row['F3_CREDIT_REL']:+.2f}")
    except KeyError:
        print(f"    {d}: not in index")

# ═════════════════════════════════════════════════════════════════════════════
#  HELPER: FIT FROZEN MODEL
# ═════════════════════════════════════════════════════════════════════════════
def fit_frozen(n_states, X_is, seeds=[42, 123, 456, 789, 1024]):
    best_model, best_ll = None, -np.inf
    for seed in seeds:
        m = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=500, random_state=seed)
        m.fit(X_is)
        ll = m.score(X_is)
        if ll > best_ll:
            best_ll = ll
            best_model = m
    # Sort states by VIX mean (feature 0)
    vix_order = np.argsort(best_model.means_[:, 0])
    best_model.means_ = best_model.means_[vix_order]
    best_model.covars_ = best_model.covars_[vix_order]
    best_model.startprob_ = best_model.startprob_[vix_order]
    P = best_model.transmat_[vix_order, :]
    best_model.transmat_ = P[:, vix_order]
    return best_model

# ═════════════════════════════════════════════════════════════════════════════
#  HELPER: CAUSAL FORWARD FILTER
# ═════════════════════════════════════════════════════════════════════════════
def causal_filter(model, X_std_full, is_mask, oos_mask):
    n_states = model.n_components
    prob = np.full((len(X_std_full), n_states), np.nan)
    # IS: use smoothed (fine for IS analysis)
    prob[is_mask] = model.predict_proba(X_std_full[is_mask])
    # OOS: strictly causal (forward-only at each step)
    oos_indices = np.where(oos_mask)[0]
    print(f"    Forward-filtering {len(oos_indices)} OOS days...", flush=True)
    for t in oos_indices:
        prob[t] = model.predict_proba(X_std_full[:t+1])[-1]
    return prob

# ═════════════════════════════════════════════════════════════════════════════
#  HELPER: RUN CHECKS
# ═════════════════════════════════════════════════════════════════════════════
def run_checks(df, prob, n_states, label):
    # Assign columns: Calm = lowest VIX state, Crisis = highest VIX state
    # For 2-state: col 0 = Calm, col 1 = Crisis
    # For 3-state: col 0 = Calm, col 1 = Transition, col 2 = Crisis
    crisis_col = n_states - 1
    calm_col   = 0

    df = df.copy()
    df['P_Crisis'] = prob[:, crisis_col]
    df['P_Calm']   = prob[:, calm_col]
    if n_states == 3:
        df['P_Trans'] = prob[:, 1]

    results = {}

    # ── Check 1: Non-stickiness (COVID recovery) ──────────────────────────
    covid_peak_mask = (df.index >= '2020-02-01') & (df.index <= '2020-04-30')
    covid_rec_mask  = (df.index >= '2020-05-01') & (df.index <= '2020-07-31')
    peak_crisis = df.loc[covid_peak_mask, 'P_Crisis'].max()
    min_crisis_rec = df.loc[covid_rec_mask, 'P_Crisis'].min()
    check1_pass = (peak_crisis > 0.5) and (min_crisis_rec < 0.5)
    results['C1_covid_peak']   = peak_crisis
    results['C1_covid_rec_min'] = min_crisis_rec
    results['C1_pass'] = check1_pass

    # ── Check 2: Chronology vs Volatility ─────────────────────────────────
    calm_periods = {
        '2017': ('2017-01-01', '2017-12-31'),
        '2021': ('2021-06-01', '2021-11-01'),
        '2023': ('2023-06-01', '2023-12-31'),
    }
    check2_results = {}
    check2_pass = True
    for period, (s, e) in calm_periods.items():
        m = (df.index >= s) & (df.index <= e)
        avg = df.loc[m, 'P_Calm'].mean()
        ok = avg > 0.5
        check2_results[period] = (avg, ok)
        if not ok:
            check2_pass = False
    results['C2_periods'] = check2_results
    results['C2_pass'] = check2_pass

    # ── Check 3: OOS Tariff Shock (Apr 2025) ──────────────────────────────
    tariff_peak_mask = (df.index >= '2025-03-01') & (df.index <= '2025-05-31')
    tariff_rec_mask  = (df.index >= '2025-06-01') & (df.index <= '2025-08-31')
    tariff_peak = df.loc[tariff_peak_mask, 'P_Crisis'].max()
    tariff_rec_min = df.loc[tariff_rec_mask, 'P_Crisis'].min() if df.loc[tariff_rec_mask].shape[0] > 0 else 0.0
    check3_pass = (tariff_peak > 0.5) and (tariff_rec_min < 0.5)
    results['C3_tariff_peak']   = tariff_peak
    results['C3_tariff_rec_min'] = tariff_rec_min
    results['C3_pass'] = check3_pass

    # ── Print ──────────────────────────────────────────────────────────────
    print(f"\n  ── {label} ──")
    v1 = '✓' if check1_pass else '✗'
    v2 = '✓' if check2_pass else '✗'
    v3 = '✓' if check3_pass else '✗'
    print(f"  [{v1}] CHECK 1 Non-Stickiness (COVID recovery):")
    print(f"        Peak P(Crisis) Feb-Apr 2020: {peak_crisis:.3f}")
    print(f"        Min  P(Crisis) May-Jul 2020: {min_crisis_rec:.3f}  (need <0.5 to pass)")
    print(f"  [{v2}] CHECK 2 Chronology vs Volatility:")
    for period, (avg, ok) in check2_results.items():
        print(f"        {period}: Avg P(Calm) = {avg:.3f}  {'✓' if ok else '✗'}")
    print(f"  [{v3}] CHECK 3 OOS Tariff Shock (Apr 2025):")
    print(f"        Peak P(Crisis) Mar-May 2025: {tariff_peak:.3f}")
    print(f"        Min  P(Crisis) Jun-Aug 2025: {tariff_rec_min:.3f}  (need <0.5 to pass)")

    return results, df

# ═════════════════════════════════════════════════════════════════════════════
#  RUN 2-STATE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  MODEL A: 2-STATE FROZEN (ROLLING-RELATIVE FEATURES)")
print("=" * 80)
m2 = fit_frozen(2, X_is)
means_raw_2 = m2.means_.copy()
print(f"  Calm   state: VIX-z={means_raw_2[0,0]:.2f}  YC-rel={means_raw_2[0,3]:.2f}  Cred-rel={means_raw_2[0,2]:.2f}")
print(f"  Crisis state: VIX-z={means_raw_2[1,0]:.2f}  YC-rel={means_raw_2[1,3]:.2f}  Cred-rel={means_raw_2[1,2]:.2f}")
prob2 = causal_filter(m2, X_std, is_mask, oos_mask)
res2, df2 = run_checks(df, prob2, 2, "2-State Frozen (Rolling-Relative)")

# ═════════════════════════════════════════════════════════════════════════════
#  RUN 3-STATE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  MODEL B: 3-STATE FROZEN (ROLLING-RELATIVE FEATURES)")
print("=" * 80)
m3 = fit_frozen(3, X_is)
means_raw_3 = m3.means_.copy()
state_labels3 = []
for i in range(3):
    vix_z = means_raw_3[i, 0]
    yc_rel = means_raw_3[i, 3]
    cr_rel = means_raw_3[i, 2]
    if yc_rel < -0.3 or cr_rel > 0.3:
        name = "Transition"
    elif vix_z > 0.5:
        name = "Crisis"
    else:
        name = "Calm"
    state_labels3.append(name)
    print(f"  State {i} ({name}): VIX-z={vix_z:.2f}  YC-rel={yc_rel:.2f}  "
          f"Cred-rel={cr_rel:.2f}  Mom={means_raw_3[i,5]:.3f}")

if "Transition" in state_labels3:
    print(f"\n  ✓ Transition-like state FOUND at state index {state_labels3.index('Transition')}")
else:
    print(f"\n  ✗ No Transition state emerged — EM still split Calm into sub-states")

prob3 = causal_filter(m3, X_std, is_mask, oos_mask)
res3, df3 = run_checks(df, prob3, 3, "3-State Frozen (Rolling-Relative)")

# 3-state extra check: does Transition activate during 2022–2023?
if "Transition" in state_labels3:
    trans_idx = state_labels3.index('Transition')
    trans_2022_mask = (df.index >= '2022-01-01') & (df.index <= '2023-12-31')
    avg_trans_2022 = prob3[trans_2022_mask, trans_idx].mean()
    peak_trans_2022 = prob3[trans_2022_mask, trans_idx].max()
    print(f"\n  3-State Transition Activation (2022–2023):")
    print(f"    Avg P(Transition): {avg_trans_2022:.3f}")
    print(f"    Peak P(Transition): {peak_trans_2022:.3f}")
    v = '✓' if avg_trans_2022 > 0.3 else '✗'
    print(f"    [{v}] Transition state activates meaningfully during 2022-2023 Fed cycle")

# ═════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  FINAL VERDICT TABLE")
print("=" * 80)
print(f"\n  {'Check':<45} {'2-State':>8} {'3-State':>8}")
print("  " + "-" * 62)

checks = [
    ("C1 Non-Stickiness (COVID recovery <0.5)", 'C1_pass'),
    ("C2 2017 Calm correctly classified",       None),
    ("C2 2021 Calm correctly classified",       None),
    ("C2 2023 Calm correctly classified",       None),
    ("C3 OOS Tariff Shock detected (>0.5)",     'C3_pass'),
]

for chk_name, key in checks:
    if key:
        v2 = '✓ PASS' if res2.get(key) else '✗ FAIL'
        v3 = '✓ PASS' if res3.get(key) else '✗ FAIL'
    else:
        # Extract from C2_periods
        period_key = {'2017': '2017', '2021': '2021', '2023': '2023'}
        for p in ['2017', '2021', '2023']:
            if p in chk_name:
                _, ok2 = res2['C2_periods'].get(p, (0, False))
                _, ok3 = res3['C2_periods'].get(p, (0, False))
                v2 = '✓ PASS' if ok2 else '✗ FAIL'
                v3 = '✓ PASS' if ok3 else '✗ FAIL'
    print(f"  {chk_name:<45} {v2:>8} {v3:>8}")

all_pass_2 = res2['C1_pass'] and res2['C2_pass'] and res2['C3_pass']
all_pass_3 = res3['C1_pass'] and res3['C2_pass'] and res3['C3_pass']

print("\n  Overall:")
print(f"    2-State: {'ALL PASS → FINALISE' if all_pass_2 else 'FAIL → CHECK DETAILS'}")
print(f"    3-State: {'ALL PASS → FINALISE' if all_pass_3 else 'FAIL → CHECK DETAILS'}")

if not all_pass_2 and not all_pass_3:
    print("\n  ⚠ RECOMMENDATION: ABANDON HMM ARCHITECTURE.")
    print("    Rolling-relative fix did not resolve structural misclassification.")
    print("    Fall back to stationary percentile-composite metric for Adaptive EGARCH.")
elif all_pass_2 and not all_pass_3:
    print("\n  ✓ RECOMMENDATION: FINALISE 2-STATE FROZEN HMM.")
elif all_pass_3:
    print("\n  ✓ RECOMMENDATION: FINALISE 3-STATE FROZEN HMM.")
    if "Transition" not in state_labels3:
        print("    CAVEAT: Transition state did not emerge; treat as 2-state in practice.")

# ═════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n  Generating plots...")

fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
fig.suptitle("Rolling-Relative Standardization Fix — 2-State vs 3-State Frozen HMM",
             fontsize=14, fontweight='bold')

axes[0].plot(df.index, df['VIX'], 'steelblue', linewidth=0.8, label='VIX')
axes[0].axhline(20, color='orange', linestyle='--', linewidth=0.5)
axes[0].axhline(30, color='red', linestyle='--', linewidth=0.5)
axes[0].axvline(TRAIN_END, color='black', linestyle='--', linewidth=1.5, label='OOS Start')
axes[0].set_ylabel('VIX')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.2)

# Panel 2: 2-State
axes[1].fill_between(df.index, df2['P_Crisis'], color='crimson', alpha=0.6, label='P(Crisis)')
axes[1].fill_between(df.index, df2['P_Calm'],   color='steelblue', alpha=0.6, label='P(Calm)')
axes[1].axvline(TRAIN_END, color='black', linestyle='--', linewidth=1.5)
axes[1].axhline(0.5, color='black', linestyle='--', linewidth=0.6)
axes[1].set_ylabel('2-State\nProbabilities')
axes[1].legend(fontsize=8, loc='upper left')
axes[1].set_ylim(-0.05, 1.1)
axes[1].grid(True, alpha=0.2)

# Panel 3: 3-State
colors3 = ['steelblue', 'orange', 'crimson']
cols3   = ['P_Calm', 'P_Trans', 'P_Crisis'] if 'P_Trans' in df3.columns else ['P_Calm', 'P_Crisis']
for ci, (col3, lbl3) in enumerate(zip(cols3, ['P(Calm)', 'P(Trans)', 'P(Crisis)'])):
    if col3 in df3.columns:
        axes[2].fill_between(df3.index, df3[col3], color=colors3[ci], alpha=0.5, label=lbl3)
axes[2].axvline(TRAIN_END, color='black', linestyle='--', linewidth=1.5)
axes[2].axhline(0.5, color='black', linestyle='--', linewidth=0.6)
axes[2].set_ylabel('3-State\nProbabilities')
axes[2].legend(fontsize=8, loc='upper left')
axes[2].set_ylim(-0.05, 1.1)
axes[2].grid(True, alpha=0.2)

# Panel 4: Rolling-relative YC and Credit
ax4b = axes[3].twinx()
axes[3].plot(df.index, df['F4_YC_REL'],      'purple', linewidth=0.7, label='YC (3yr-rel)')
ax4b.plot(   df.index, df['F3_CREDIT_REL'],  'green',  linewidth=0.7, label='Credit (3yr-rel)', alpha=0.8)
axes[3].axhline(0, color='purple', linestyle=':', linewidth=0.5)
ax4b.axhline(0, color='green', linestyle=':', linewidth=0.5)
axes[3].axvline(TRAIN_END, color='black', linestyle='--', linewidth=1.5)
axes[3].set_ylabel('YC 3yr-rel z-score', color='purple')
ax4b.set_ylabel('Credit 3yr-rel z-score', color='green')
axes[3].legend(loc='upper left', fontsize=8)
ax4b.legend(loc='upper right', fontsize=8)
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[3].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('plots/rolling_relative_hmm.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/rolling_relative_hmm.png")
print("=" * 80)
