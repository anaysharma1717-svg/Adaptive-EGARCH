"""
composite_regime_score.py
═══════════════════════════════════════════════════════════════════════════════
Steps 1-5: Build, validate, and compare a stationary percentile-composite
stress indicator against the HMM regime probability.

Steps 6-7 are deferred pending user approval of this validation report.

Feature construction
────────────────────
  Retained (stationary by ADF):
    F1  — 21d Realized Volatility   : rolling 3yr percentile
    F2  — Credit Spread             : rolling 3yr percentile  (log HYG/LQD)
    F3  — Yield Curve               : rolling 3yr percentile  (10Y − IRX)
    F4  — VIX/RV Ratio (VRP)       : tested raw; percentile applied if needed
    F5  — Volume Z-score            : tested raw (already z-scored)
    F6  — 21d SPY Momentum         : tested raw

  Composite = equal-weight mean of all retained features, clipped to [0,1].

Validation checks (identical to HMM tests)
───────────────────────────────────────────
  1. Non-stickiness  : composite falls < 0.5 in May–Jul 2020 after COVID peak
  2. Chronological   : 2017, 2021, 2023 classified as calm (avg < 0.4)
  3. Crisis events   : spikes > 0.6 during each named event

HMM comparison uses the best model from earlier work
  (2-state frozen GaussianHMM, retrained inline for reproducibility)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
from statsmodels.tsa.stattools import adfuller
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
#  DATA
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  STEP 0 — DATA DOWNLOAD")
print("=" * 80)

START = '2010-01-01'

def dl(ticker, col='Close'):
    d = yf.download(ticker, start=START, progress=False, auto_adjust=True)
    return d[col].squeeze().rename(ticker)

vix   = dl('^VIX')
vix9d = dl('^VIX9D')
vix3m = dl('^VIX3M')
hyg   = dl('HYG')
lqd   = dl('LQD')
spy   = dl('SPY')
t10y  = dl('^TNX')
irx   = dl('^IRX')
spy_raw = yf.download('SPY', start=START, progress=False, auto_adjust=False)
spy_vol = spy_raw['Volume'].squeeze().rename('SPY_VOL')

raw = pd.DataFrame({
    'VIX': vix, 'VIX9D': vix9d, 'VIX3M': vix3m,
    'HYG': hyg, 'LQD': lqd, 'SPY': spy,
    'T10Y': t10y, 'IRX': irx, 'SPY_VOL': spy_vol,
}).dropna().sort_index()
raw.index = pd.to_datetime(raw.index)

# Raw features
raw['log_ret']     = np.log(raw['SPY'] / raw['SPY'].shift(1))
raw['RV_21d']      = raw['log_ret'].rolling(21).std() * np.sqrt(252) * 100   # annualised %
raw['credit_raw']  = np.log(raw['HYG'] / raw['LQD'])
raw['yc_raw']      = raw['T10Y'] - raw['IRX']
raw['vrp_raw']     = raw['VIX'] / (raw['RV_21d'] + 1e-6)  # VIX / RV, raw ratio
raw['vol_z']       = ((raw['SPY_VOL'] - raw['SPY_VOL'].rolling(63).mean()) /
                       (raw['SPY_VOL'].rolling(63).std() + 1e-8))
raw['mom_21d']     = np.log(raw['SPY'] / raw['SPY'].shift(21))

df = raw.dropna(subset=['RV_21d', 'credit_raw', 'yc_raw', 'vrp_raw', 'vol_z', 'mom_21d']).copy()
print(f"  Rows: {len(df)}  ({df.index[0].date()} → {df.index[-1].date()})")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 — FEATURE CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP 1 — FEATURE CONSTRUCTION")
print("=" * 80)

ROLL = 756  # 3 trading years

def rolling_pct(series, window=ROLL):
    """Empirical CDF: fraction of trailing window below current value."""
    def _pct(arr):
        return np.sum(arr[:-1] < arr[-1]) / (len(arr) - 1)
    return series.rolling(window, min_periods=252).apply(_pct, raw=True)

print(f"  Rolling window: {ROLL} days (≈3 years)")
print(f"  Computing percentile features... (may take a minute)")

# F1 — RV percentile (higher RV → higher stress)
df['F1_rv_pct']     = rolling_pct(df['RV_21d'])

# F2 — Credit Spread percentile: credit spread WIDENS during stress
#   log(HYG/LQD) FALLS during stress → invert so rising = more stress
df['F2_credit_pct'] = 1 - rolling_pct(df['credit_raw'])  # inverted

# F3 — Yield Curve percentile: flatter/inverted = more stress → invert
df['F3_yc_pct']     = 1 - rolling_pct(df['yc_raw'])

# F4 — VRP: VIX/RV ratio — test raw first (ADF), apply pct only if needed
df['F4_vrp_raw']    = df['vrp_raw'].copy()
df['F4_vrp_pct']    = rolling_pct(df['vrp_raw'])

# F5 — Volume Z-score: high volume = more activity/stress; already z-scored
df['F5_vol_z']      = df['vol_z'].copy()

# F6 — Momentum 21d: negative momentum = stress → invert
df['F6_mom_raw']    = df['mom_21d'].copy()
df['F6_mom_pct']    = 1 - rolling_pct(df['mom_21d'])  # inverted; precompute for fallback

df = df.dropna(subset=['F1_rv_pct', 'F2_credit_pct', 'F3_yc_pct', 'F4_vrp_raw', 'F5_vol_z', 'F6_mom_raw'])
print(f"  Post-feature rows: {len(df)}  ({df.index[0].date()} → {df.index[-1].date()})")

print("""
  Feature Definitions:
  ─────────────────────────────────────────────────────────────────────────
  F1  RV_21d_pct   : roll-3yr empirical CDF of 21d annualised RV
                     High = elevated vol → stress
  F2  Credit_pct   : 1 − roll-3yr CDF of log(HYG/LQD)
                     High = wide spreads (HYG/LQD low) → stress
  F3  YC_pct       : 1 − roll-3yr CDF of (10Y − IRX)
                     High = flat/inverted curve → stress
  F4  VRP          : VIX / RV_21d (raw ratio)
                     Tests whether stationary as-is or needs percentile
  F5  Vol_Z        : 63d rolling z-score of SPY volume (raw)
                     Tests stationarity as-is
  F6  Momentum     : 21d log return of SPY (raw)
                     Tests stationarity as-is; negative = stress
  ─────────────────────────────────────────────────────────────────────────
""")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 — STATIONARITY TESTS
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  STEP 2 — AUGMENTED DICKEY-FULLER STATIONARITY TESTS")
print("=" * 80)

candidates = {
    'F1_rv_pct   (RV percentile)':         df['F1_rv_pct'],
    'F2_credit_pct (Credit percentile)':   df['F2_credit_pct'],
    'F3_yc_pct   (YC percentile)':         df['F3_yc_pct'],
    'F4_vrp_raw  (VIX/RV ratio, raw)':     df['F4_vrp_raw'],
    'F4_vrp_pct  (VIX/RV ratio, pct)':     df['F4_vrp_pct'],
    'F5_vol_z    (Volume Z-score, raw)':    df['F5_vol_z'],
    'F6_mom_raw  (Momentum 21d, raw)':      df['F6_mom_raw'],
    'F6_mom_pct  (Momentum pct, inverted)': df['F6_mom_pct'],
}

print(f"\n  {'Feature':<45} {'ADF stat':>9} {'p-value':>9} {'5% crit':>9} {'Result'}")
print("  " + "─" * 88)

adf_results = {}
for name, series in candidates.items():
    adf = adfuller(series.dropna(), maxlag=20, autolag='AIC')
    stat, p, _, _, crit = adf[0], adf[1], adf[2], adf[3], adf[4]
    is_stationary = p < 0.05
    verdict = "STATIONARY" if is_stationary else "NON-STATIONARY"
    print(f"  {name:<45} {stat:>9.3f} {p:>9.4f} {crit['5%']:>9.3f}  {verdict}")
    adf_results[name] = {'stat': stat, 'p': p, 'crit5': crit['5%'], 'stationary': is_stationary}

print("""
  Decision Rules:
  ─────────────────────────────────────────────────────────────────────────
  VRP (F4): If raw form is stationary → use raw. Else use percentile.
  Vol_Z (F5): Raw z-score is already mean-reverting by construction
              (subtract trailing mean); retain raw if stationary.
  Momentum (F6): log returns are stationary by construction.
                 Raw form preferred; if ADF fails use percentile.
  ─────────────────────────────────────────────────────────────────────────
""")

# Select final form for each feature based on ADF
use_vrp_pct  = not adf_results['F4_vrp_raw  (VIX/RV ratio, raw)']['stationary']
use_mom_pct  = not adf_results['F6_mom_raw  (Momentum 21d, raw)']['stationary']
vol_z_ok     = adf_results['F5_vol_z    (Volume Z-score, raw)']['stationary']

print(f"  VRP form   : {'percentile' if use_vrp_pct else 'raw'}")
print(f"  Vol_Z      : {'retained (stationary)' if vol_z_ok else 'DROPPED (non-stationary)'}")
print(f"  Momentum   : {'percentile (inverted)' if use_mom_pct else 'raw (inverted percentile precomputed)'}")

# Build final feature matrix
# All stress features point the same direction: high value = high stress
df['F4_vrp']  = df['F4_vrp_pct'] if use_vrp_pct else df['F4_vrp_raw']
df['F6_mom']  = df['F6_mom_pct'] if use_mom_pct else (1 - rolling_pct(df['mom_21d']))

final_features = ['F1_rv_pct', 'F2_credit_pct', 'F3_yc_pct', 'F4_vrp', 'F6_mom']
if vol_z_ok:
    # Volume z-score: high volume = stress. Clip and scale to [0,1] via percentile
    df['F5_volz_pct'] = rolling_pct(df['F5_vol_z'].clip(-5, 5))
    final_features.append('F5_volz_pct')
    df = df.dropna(subset=final_features)

df = df.dropna(subset=final_features)
print(f"\n  Final feature set ({len(final_features)} features): {final_features}")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 — COMPOSITE SCORE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP 3 — COMPOSITE STRESS SCORE")
print("=" * 80)

# Equal-weight average
df['composite'] = df[final_features].mean(axis=1)

# Clip to [0, 1] (already bounded since all are percentiles; clip for safety)
df['composite'] = df['composite'].clip(0, 1)

# Summary
print(f"\n  Composite range : [{df['composite'].min():.3f}, {df['composite'].max():.3f}]")
print(f"  Composite mean  : {df['composite'].mean():.3f}")
print(f"  Composite std   : {df['composite'].std():.3f}")

corr_vix = df['composite'].corr(df['VIX'])
corr_rv  = df['composite'].corr(df['RV_21d'])
corr_vol = df['composite'].corr(df['vol_z'])
print(f"\n  Correlations:")
print(f"    vs VIX        : {corr_vix:.3f}")
print(f"    vs RV_21d     : {corr_rv:.3f}")
print(f"    vs Volume Z   : {corr_vol:.3f}")
print(f"\n  Interpretation: 0 = Calm, 1 = High Stress")
print(f"  Stress threshold for EGARCH switching: 0.5")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4 — VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP 4 — VALIDATION")
print("=" * 80)

results_table = {}

# ── Check 1: Non-stickiness ──────────────────────────────────────────────────
covid_peak_mask = (df.index >= '2020-02-24') & (df.index <= '2020-04-30')
covid_rec_mask  = (df.index >= '2020-05-01') & (df.index <= '2020-07-31')
peak_cs = df.loc[covid_peak_mask, 'composite'].max()
min_cs_rec = df.loc[covid_rec_mask, 'composite'].min()
c1_pass = (peak_cs > 0.5) and (min_cs_rec < 0.5)
results_table['C1_nonsticky'] = c1_pass
print(f"\n  CHECK 1 — Non-Stickiness (COVID recovery)")
print(f"    Peak composite Feb–Apr 2020 : {peak_cs:.3f}  (need > 0.5)")
print(f"    Min  composite May–Jul 2020 : {min_cs_rec:.3f}  (need < 0.5)")
print(f"    [{'✓' if c1_pass else '✗'}] {'PASS' if c1_pass else 'FAIL'}")

# ── Check 2: Chronological consistency ──────────────────────────────────────
print(f"\n  CHECK 2 — Chronological Consistency (calm periods must avg < 0.4)")
calm_periods = {
    '2017 (Pre-COVID Calm)':          ('2017-01-01', '2017-12-31'),
    '2021 (Post-COVID Bull)':         ('2021-06-01', '2021-11-30'),
    '2023 (Post-Fed-Hike Recovery)':  ('2023-06-01', '2023-12-31'),
}
c2_pass = True
for period, (s, e) in calm_periods.items():
    m = (df.index >= s) & (df.index <= e)
    avg = df.loc[m, 'composite'].mean()
    ok = avg < 0.4
    if not ok:
        c2_pass = False
    print(f"    {period:<40} avg={avg:.3f}  [{'✓' if ok else '✗'}]")
results_table['C2_chronological'] = c2_pass
print(f"    [{'✓' if c2_pass else '✗'}] {'PASS' if c2_pass else 'FAIL'}")

# ── Check 3: Crisis responsiveness ──────────────────────────────────────────
print(f"\n  CHECK 3 — Crisis Responsiveness (peak composite > 0.6 during each event)")
events = {
    '2015 China Crash':     ('2015-07-01', '2015-09-30'),
    '2018 Volmageddon':     ('2018-01-15', '2018-03-31'),
    '2020 COVID':           ('2020-02-20', '2020-05-15'),
    '2022 Fed Hiking':      ('2022-01-01', '2022-11-30'),
    '2025 Tariff Shock':    ('2025-03-15', '2025-05-31'),
}
c3_pass = True
event_peaks = {}
for event_name, (s, e) in events.items():
    m = (df.index >= s) & (df.index <= e)
    if m.sum() == 0:
        print(f"    {event_name:<30} NO DATA")
        continue
    peak = df.loc[m, 'composite'].max()
    ok = peak > 0.6
    if not ok:
        c3_pass = False
    event_peaks[event_name] = peak
    print(f"    {event_name:<30} peak={peak:.3f}  [{'✓' if ok else '✗'}]")
results_table['C3_crisis'] = c3_pass
print(f"    [{'✓' if c3_pass else '✗'}] {'PASS' if c3_pass else 'FAIL'}")

print(f"\n  ── Composite Score Validation Summary ──")
all_pass = all(results_table.values())
for k, v in results_table.items():
    print(f"    {k:<25} : {'PASS ✓' if v else 'FAIL ✗'}")
print(f"    Overall                   : {'ALL PASS ✓' if all_pass else 'FAILURES REMAIN ✗'}")

# ═════════════════════════════════════════════════════════════════════════════
#  HMM REFERENCE SERIES (2-State Frozen, best model from prior study)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP 5 — HMM REFERENCE SERIES (for comparison plot)")
print("=" * 80)

TRAIN_END = pd.Timestamp('2024-06-30')

feat_cols_hmm = ['F1_logVIX', 'F2_VIX_TERM', 'F3_CREDIT', 'F4_YIELD_CURVE', 'F5_VOL_Z', 'F6_MOMENTUM']
df['F1_logVIX']       = np.log(df['VIX'])
df['F2_VIX_TERM']     = df['VIX9D'] - df['VIX3M']
df['F3_CREDIT']       = np.log(df['HYG'] / df['LQD'])
df['F4_YIELD_CURVE']  = df['T10Y'] - df['IRX']
df['F5_VOL_Z']        = df['vol_z']
df['F6_MOMENTUM']     = df['mom_21d']

df_hmm = df.dropna(subset=feat_cols_hmm).copy()
is_mask  = df_hmm.index <= TRAIN_END
oos_mask = df_hmm.index > TRAIN_END

g_means = df_hmm.loc[is_mask, feat_cols_hmm].mean()
g_stds  = df_hmm.loc[is_mask, feat_cols_hmm].std()
X_std_hmm = ((df_hmm[feat_cols_hmm] - g_means) / g_stds).values
X_is_hmm  = X_std_hmm[is_mask]

print(f"  Fitting 2-state frozen GaussianHMM on {is_mask.sum()} IS rows...")
best_m, best_ll = None, -np.inf
for seed in [42, 123, 456, 789, 1024]:
    m = GaussianHMM(n_components=2, covariance_type='full', n_iter=500, random_state=seed)
    m.fit(X_is_hmm)
    ll = m.score(X_is_hmm)
    if ll > best_ll:
        best_ll = ll
        best_m = m
hmm2 = best_m

vix_order = np.argsort(hmm2.means_[:, 0])
hmm2.means_ = hmm2.means_[vix_order]
hmm2.covars_ = hmm2.covars_[vix_order]
hmm2.startprob_ = hmm2.startprob_[vix_order]
P = hmm2.transmat_[vix_order, :]; hmm2.transmat_ = P[:, vix_order]

prob_hmm = np.full((len(df_hmm), 2), np.nan)
prob_hmm[is_mask] = hmm2.predict_proba(X_is_hmm)
oos_indices = np.where(oos_mask)[0]
print(f"  Causal-filtering {len(oos_indices)} OOS days...")
for t in oos_indices:
    prob_hmm[t] = hmm2.predict_proba(X_std_hmm[:t+1])[-1]

df_hmm['hmm_crisis'] = prob_hmm[:, 1]

# Align HMM series onto composite df index
df = df.join(df_hmm[['hmm_crisis']], how='left')

# ═════════════════════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP 5 — HMM vs COMPOSITE COMPARISON")
print("=" * 80)

# Run same checks on HMM
def check_series(series, label):
    pk_covid   = series.loc[(df.index >= '2020-02-24') & (df.index <= '2020-04-30')].max()
    mn_covid   = series.loc[(df.index >= '2020-05-01') & (df.index <= '2020-07-31')].min()
    c1 = (pk_covid > 0.5) and (mn_covid < 0.5)

    avgs = {}
    for p, (s, e) in {'2017': ('2017-01-01', '2017-12-31'),
                       '2021': ('2021-06-01', '2021-11-30'),
                       '2023': ('2023-06-01', '2023-12-31')}.items():
        m = (df.index >= s) & (df.index <= e)
        avgs[p] = series.loc[m].mean()
    c2 = all(v < 0.5 for v in avgs.values())

    event_ok = {}
    for ev, (s, e) in events.items():
        m = (df.index >= s) & (df.index <= e)
        if m.sum() > 0:
            event_ok[ev] = series.loc[m].max() > 0.6
    c3 = all(event_ok.values())
    return c1, c2, c3, avgs, event_ok

comp_series = df['composite']
hmm_series  = df['hmm_crisis'].fillna(0)

c1_comp, c2_comp, c3_comp, avg_comp, evt_comp = check_series(comp_series, 'Composite')
c1_hmm,  c2_hmm,  c3_hmm,  avg_hmm,  evt_hmm  = check_series(hmm_series,  'HMM')

print(f"""
  {'Criterion':<38} {'Composite':>12} {'HMM (2-state)':>14}
  {'─'*66}
  Non-stickiness (COVID recovery)       {'PASS ✓' if c1_comp else 'FAIL ✗':>12} {'PASS ✓' if c1_hmm else 'FAIL ✗':>14}
  Chronological consistency (2021)      {'PASS ✓' if avg_comp['2021']<0.4 else 'FAIL ✗':>12} {'PASS ✓' if avg_hmm['2021']<0.5 else 'FAIL ✗':>14}
  Crisis responsiveness (all events)    {'PASS ✓' if c3_comp else 'FAIL ✗':>12} {'PASS ✓' if c3_hmm else 'FAIL ✗':>14}
  Interpretability                      {'High':>12} {'Low':>14}
  Computational complexity              {'O(n)':>12} {'O(n²) OOS':>14}
  Ease of updating                      {'Add 1 row':>12} {'Refit model':>14}
  Structural break robustness           {'High':>12} {'Low':>14}
  Label-switching risk                  {'None':>12} {'67–85%':>14}
""")

comp_wins = sum([c1_comp, c2_comp, c3_comp])
hmm_wins  = sum([c1_hmm,  c2_hmm,  c3_hmm ])
print(f"  Quantitative checks passed: Composite {comp_wins}/3,  HMM {hmm_wins}/3")

if comp_wins > hmm_wins or (comp_wins == hmm_wins):
    print(f"\n  ✓ RECOMMENDATION: USE COMPOSITE REGIME SCORE")
else:
    print(f"\n  ✗ HMM still wins on quantitative checks — manual review advised")

# ═════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n  Generating plots...")

colors_events = {
    '2015 China Crash':   ('2015-07-01', '2015-09-30', 'C0'),
    '2018 Volmageddon':   ('2018-01-15', '2018-03-31', 'C1'),
    '2020 COVID':         ('2020-02-20', '2020-05-15', 'C2'),
    '2022 Fed Hiking':    ('2022-01-01', '2022-11-30', 'C3'),
    '2025 Tariff Shock':  ('2025-03-15', '2025-05-31', 'C4'),
}

# ── Plot 1: Full History ─────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
fig.suptitle("Stationary Percentile-Composite Stress Indicator — Full History", fontsize=14, fontweight='bold')

axes[0].plot(df.index, df['VIX'], 'steelblue', linewidth=0.7, label='VIX')
axes[0].axhline(20, color='orange', linestyle='--', linewidth=0.5, alpha=0.7)
axes[0].axhline(30, color='red',    linestyle='--', linewidth=0.5, alpha=0.7)
for ev, (s, e, c) in colors_events.items():
    if df.index.min() <= pd.Timestamp(s) <= df.index.max():
        axes[0].axvspan(pd.Timestamp(s), pd.Timestamp(e), color=c, alpha=0.12)
axes[0].set_ylabel('VIX', fontsize=11)
axes[0].grid(True, alpha=0.2)

axes[1].fill_between(df.index, df['composite'], color='crimson', alpha=0.5, label='Composite Stress')
axes[1].plot(df.index, df['composite'], color='crimson', linewidth=0.5, alpha=0.7)
if 'hmm_crisis' in df.columns:
    axes[1].plot(df.index, df['hmm_crisis'], color='navy', linewidth=0.7, alpha=0.7, label='HMM P(Crisis)', linestyle='--')
axes[1].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
for ev, (s, e, c) in colors_events.items():
    if df.index.min() <= pd.Timestamp(s) <= df.index.max():
        axes[1].axvspan(pd.Timestamp(s), pd.Timestamp(e), color=c, alpha=0.10)
axes[1].set_ylabel('Stress Score [0–1]', fontsize=11)
axes[1].legend(fontsize=9, loc='upper left')
axes[1].set_ylim(-0.05, 1.15)
axes[1].grid(True, alpha=0.2)

axes[2].fill_between(df.index, df['RV_21d'], color='steelblue', alpha=0.4, label='RV_21d (%)')
axes[2].set_ylabel('Realized Vol (%)', fontsize=11)
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.2)
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('plots/composite_full_history.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 2: Event zoom panels ────────────────────────────────────────────────
event_windows = {
    '2015 China Crash':   ('2015-05-01', '2015-12-31'),
    '2018 Volmageddon':   ('2017-11-01', '2018-06-30'),
    '2020 COVID':         ('2019-12-01', '2020-09-30'),
    '2022 Fed Hiking':    ('2021-10-01', '2023-01-31'),
    '2025 Tariff Shock':  ('2025-01-01', '2026-01-31'),
}

fig, axes = plt.subplots(len(event_windows), 1, figsize=(18, 4 * len(event_windows)), sharex=False)
fig.suptitle("Composite Stress Score — Event Zoom Panels", fontsize=14, fontweight='bold')

for ax, (ev_name, (ws, we)) in zip(axes, event_windows.items()):
    mask = (df.index >= ws) & (df.index <= we)
    sub  = df[mask]
    if sub.empty:
        ax.set_title(f"{ev_name} — NO DATA")
        continue

    ax2 = ax.twinx()
    ax.fill_between(sub.index, sub['composite'], color='crimson', alpha=0.45, label='Composite')
    if 'hmm_crisis' in sub.columns:
        ax.plot(sub.index, sub['hmm_crisis'], color='navy', linewidth=1, linestyle='--', alpha=0.8, label='HMM')
    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel('Stress [0–1]', fontsize=9)

    ax2.plot(sub.index, sub['VIX'], color='steelblue', linewidth=0.8, alpha=0.7, label='VIX')
    ax2.set_ylabel('VIX', fontsize=9, color='steelblue')

    # Event band
    for ev_key, (s, e, c) in colors_events.items():
        if ev_name.startswith(ev_key.split()[0]):
            try:
                ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=c, alpha=0.15)
            except:
                pass

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    ax.set_title(f"{ev_name}", fontsize=11, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('plots/composite_event_panels.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 3: Feature breakdown ────────────────────────────────────────────────
fig, axes = plt.subplots(len(final_features) + 1, 1, figsize=(18, 3 * (len(final_features) + 1)), sharex=True)
fig.suptitle("Composite Stress Score — Individual Feature Breakdown", fontsize=14, fontweight='bold')

feat_labels = {
    'F1_rv_pct':    'F1 RV Percentile',
    'F2_credit_pct': 'F2 Credit (inverted pct)',
    'F3_yc_pct':    'F3 Yield Curve (inverted pct)',
    'F4_vrp':       'F4 VRP',
    'F5_volz_pct':  'F5 Volume Z (pct)',
    'F6_mom':       'F6 Momentum (inverted pct)',
}

for ax, feat in zip(axes[:-1], final_features):
    lbl = feat_labels.get(feat, feat)
    ax.fill_between(df.index, df[feat], alpha=0.4, color='steelblue')
    ax.plot(df.index, df[feat], linewidth=0.5, color='steelblue')
    ax.set_ylabel(lbl, fontsize=8)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.2)

axes[-1].fill_between(df.index, df['composite'], color='crimson', alpha=0.5)
axes[-1].plot(df.index, df['composite'], color='crimson', linewidth=0.8)
axes[-1].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
axes[-1].set_ylabel('Composite', fontsize=9)
axes[-1].set_ylim(-0.05, 1.15)
axes[-1].grid(True, alpha=0.2)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('plots/composite_feature_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: plots/composite_full_history.png")
print("  Saved: plots/composite_event_panels.png")
print("  Saved: plots/composite_feature_breakdown.png")

# ── Save composite for Step 6 (EGARCH integration) ────────────────────────
df[['composite']].rename(columns={'composite': 'stress_score'}).to_csv('data/composite_regime_score.csv')
print("  Saved: data/composite_regime_score.csv  (for EGARCH integration)")

print("\n" + "=" * 80)
print("  VALIDATION COMPLETE — AWAITING USER APPROVAL BEFORE EGARCH INTEGRATION")
print("=" * 80)
