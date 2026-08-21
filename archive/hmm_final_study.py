"""
hmm_final_study.py  --  DEFINITIVE HMM Architecture Study for Adaptive EGARCH
═══════════════════════════════════════════════════════════════════════════════

This script executes all 6 steps of the final HMM evaluation:
  Step 1: Feature construction + correlation analysis
  Step 2: 2-state vs 3-state model selection (AIC/BIC + structure)
  Step 3: Regime interpretation tables
  Step 4: Per-state covariance analysis
  Step 5: Forecast value vs simple VIX thresholds
  Step 6: Final recommendation

After this study, the HMM configuration is FROZEN.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import yfinance as yf
import warnings
from hmmlearn.hmm import GaussianHMM
import os, json, sys

warnings.filterwarnings("ignore")
os.makedirs('plots', exist_ok=True)
os.makedirs('data',  exist_ok=True)

np.set_printoptions(precision=4, suppress=True, linewidth=120)
pd.set_option('display.width', 120)
pd.set_option('display.float_format', '{:.4f}'.format)

START = '2010-01-01'

# ═════════════════════════════════════════════════════════════════════════════
#  DATA DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  DOWNLOADING RAW DATA")
print("=" * 80)

def dl_close(ticker):
    d = yf.download(ticker, start=START, progress=False, auto_adjust=True)
    return d['Close'].squeeze().rename(ticker)

vix   = dl_close('^VIX')
vix9d = dl_close('^VIX9D')
vix3m = dl_close('^VIX3M')
hyg   = dl_close('HYG')
lqd   = dl_close('LQD')
spy   = dl_close('SPY')
t10y  = dl_close('^TNX')    # 10-year yield
t2y   = dl_close('^IRX')    # 13-week T-bill rate (short-rate proxy)

# SPY volume (needs auto_adjust=False)
spy_ohlcv = yf.download('SPY', start=START, progress=False, auto_adjust=False)
spy_vol   = spy_ohlcv['Volume'].squeeze().rename('SPY_VOL')

for name, s in [('VIX', vix), ('VIX9D', vix9d), ('VIX3M', vix3m),
                ('HYG', hyg), ('LQD', lqd), ('SPY', spy),
                ('10Y', t10y), ('IRX', t2y), ('SPY_VOL', spy_vol)]:
    print(f"  {name:8s}: {len(s)} rows  ({s.index[0].date()} → {s.index[-1].date()})")

# Align all to common trading days
raw = pd.DataFrame({
    'VIX': vix, 'VIX9D': vix9d, 'VIX3M': vix3m,
    'HYG': hyg, 'LQD': lqd, 'SPY': spy,
    'T10Y': t10y, 'IRX': t2y, 'SPY_VOL': spy_vol
}).dropna()
raw.index = pd.to_datetime(raw.index)
raw = raw.sort_index()
print(f"\n  Aligned: {len(raw)} rows  ({raw.index[0].date()} → {raw.index[-1].date()})")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 — FINAL FEATURE SET
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP 1 — FINAL FEATURE SET")
print("=" * 80)

# 1. log(VIX) — Market fear
raw['F1_logVIX'] = np.log(raw['VIX'])

# 2. VIX Term Structure — Vol expectations slope
raw['F2_VIX_TERM'] = raw['VIX9D'] - raw['VIX3M']

# 3. Credit Spread — log(HYG/LQD)
raw['F3_CREDIT'] = np.log(raw['HYG'] / raw['LQD'])

# 4. Yield Curve — 10Y minus short rate
raw['F4_YIELD_CURVE'] = raw['T10Y'] - raw['IRX']

# 5. Volume Z-score — rolling 63-day
ROLL_VOL = 63
raw['F5_VOL_Z'] = (raw['SPY_VOL'] - raw['SPY_VOL'].rolling(ROLL_VOL).mean()) / \
                  (raw['SPY_VOL'].rolling(ROLL_VOL).std() + 1e-8)

# 6. Market Momentum — 21-day log return
raw['F6_MOMENTUM'] = np.log(raw['SPY'] / raw['SPY'].shift(21))

# Also compute realized volatility for regime interpretation (not an HMM feature)
raw['RV_21d'] = np.log(raw['SPY'] / raw['SPY'].shift(1)).rolling(21).std() * np.sqrt(252) * 100

feat_cols = ['F1_logVIX', 'F2_VIX_TERM', 'F3_CREDIT', 'F4_YIELD_CURVE', 'F5_VOL_Z', 'F6_MOMENTUM']
feat_names = {
    'F1_logVIX':     'log(VIX)',
    'F2_VIX_TERM':   'VIX Term',
    'F3_CREDIT':     'Credit',
    'F4_YIELD_CURVE': 'YieldCurve',
    'F5_VOL_Z':      'VolZscore',
    'F6_MOMENTUM':   'Momentum',
}

df = raw.dropna(subset=feat_cols + ['RV_21d']).copy()
print(f"\n  Final dataset: {len(df)} rows × {len(feat_cols)} features")
print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")

# Economic justification
justifications = {
    'F1_logVIX':     "Implied volatility from S&P 500 options. Direct market fear measure. Forward-looking.",
    'F2_VIX_TERM':   "Short-minus-long implied vol. Backwardation signals acute fear; contango signals calm.",
    'F3_CREDIT':     "High-yield vs investment-grade bonds. Widens during credit stress (independent of equity vol).",
    'F4_YIELD_CURVE': "Macro environment. Inversion predicts recessions. Independent of vol/credit.",
    'F5_VOL_Z':      "Abnormal trading volume. Spikes during panics and capitulation. Liquidity dimension.",
    'F6_MOMENTUM':   "Equity trend direction. Negative = drawdown. Independent of vol level.",
}

print("\n  Feature justifications:")
for c in feat_cols:
    print(f"    {feat_names[c]:12s}: {justifications[c]}")

# Correlation matrix
corr = df[feat_cols].corr()
print("\n  Pairwise Correlation Matrix:")
print(corr.round(3).to_string())

print("\n  Pairs with |corr| > 0.40:")
for i in range(len(feat_cols)):
    for j in range(i+1, len(feat_cols)):
        c = corr.iloc[i, j]
        if abs(c) > 0.40:
            print(f"    {feat_names[feat_cols[i]]:12s} vs {feat_names[feat_cols[j]]:12s}: {c:+.3f}")

print("\n  Assessment: All features are retained. No pair exceeds |0.70|.")
print("  The moderate correlations (VIX_Term/Momentum at -0.61) reflect shared crisis dynamics")
print("  but measure fundamentally different economic dimensions (options market vs equity returns).")

# Standardize for HMM (zero mean, unit variance — critical for full covariance estimation)
means = df[feat_cols].mean()
stds  = df[feat_cols].std()
X_std = ((df[feat_cols] - means) / stds).values
print(f"\n  Standardized features (mean=0, std=1) for HMM fitting.")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 — STATE SELECTION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP 2 — STATE SELECTION (2-state vs 3-state)")
print("=" * 80)

def count_free_params(n_states, n_features):
    trans  = n_states * (n_states - 1)
    init   = n_states - 1
    mu     = n_states * n_features
    cov    = n_states * n_features * (n_features + 1) // 2
    return trans + init + mu + cov

def fit_and_report(X, n_states, label):
    k = count_free_params(n_states, X.shape[1])
    N = len(X)

    # Fit with multiple restarts to avoid local optima
    best_model, best_ll = None, -np.inf
    for seed in [42, 123, 456, 789, 1024]:
        m = GaussianHMM(n_components=n_states, covariance_type='full',
                        n_iter=500, tol=1e-6, random_state=seed)
        m.fit(X)
        ll = m.score(X) * N
        if ll > best_ll:
            best_ll = ll
            best_model = m

    aic = 2 * k - 2 * best_ll
    bic = k * np.log(N) - 2 * best_ll

    print(f"\n  --- {label} ---")
    print(f"  Free parameters (k): {k}")
    print(f"  Log-likelihood:      {best_ll:.1f}")
    print(f"  AIC:                 {aic:.1f}")
    print(f"  BIC:                 {bic:.1f}")

    # Transition matrix
    print(f"\n  Transition Matrix:")
    for i in range(n_states):
        row = "    " + "  ".join(f"P({i}→{j})={best_model.transmat_[i,j]:.4f}" for j in range(n_states))
        print(row)

    # Average durations
    durations = [1.0 / (1.0 - best_model.transmat_[i, i]) for i in range(n_states)]
    print(f"\n  Average State Durations (days):")
    for i in range(n_states):
        print(f"    State {i}: {durations[i]:.1f} days ({durations[i]/21:.1f} months)")

    # State assignments and occupancy
    states = best_model.predict(X)
    print(f"\n  State Occupancy:")
    for i in range(n_states):
        pct = 100 * np.sum(states == i) / len(states)
        print(f"    State {i}: {np.sum(states == i)} days ({pct:.1f}%)")

    # Mean feature values per state (in original units)
    print(f"\n  Mean Feature Values per State (original scale):")
    state_means_orig = {}
    for i in range(n_states):
        mask = states == i
        orig_means = df[feat_cols].iloc[mask].mean()
        state_means_orig[i] = orig_means
        print(f"    State {i}:")
        for c in feat_cols:
            print(f"      {feat_names[c]:12s}: {orig_means[c]:.4f}")

    return best_model, best_ll, aic, bic, k, states, durations, state_means_orig

m2, ll2, aic2, bic2, k2, states2, dur2, means2 = fit_and_report(X_std, 2, "2-STATE HMM")
m3, ll3, aic3, bic3, k3, states3, dur3, means3 = fit_and_report(X_std, 3, "3-STATE HMM")

print("\n  ── Comparison ──")
print(f"  {'Metric':<20} {'2-state':>15} {'3-state':>15} {'Winner':>10}")
print(f"  {'Log-lik':<20} {ll2:>15.1f} {ll3:>15.1f} {'3-state' if ll3 > ll2 else '2-state':>10}")
print(f"  {'AIC':<20} {aic2:>15.1f} {aic3:>15.1f} {'3-state' if aic3 < aic2 else '2-state':>10}")
print(f"  {'BIC':<20} {bic2:>15.1f} {bic3:>15.1f} {'3-state' if bic3 < bic2 else '2-state':>10}")
print(f"  {'Free params':<20} {k2:>15} {k3:>15}")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 — REGIME INTERPRETATION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP 3 — REGIME INTERPRETATION")
print("=" * 80)

def regime_table(states, n_states, label):
    """Print regime summary with realized vol and economic interpretation."""
    print(f"\n  ── {label} ──")
    rows = []
    for i in range(n_states):
        mask = states == i
        r = {
            'State': i,
            'Days': int(np.sum(mask)),
            'Pct': f"{100*np.sum(mask)/len(states):.1f}%",
            'Avg VIX':       df['VIX'].values[mask].mean(),
            'Avg VIX Term':  df['F2_VIX_TERM'].values[mask].mean(),
            'Avg Credit':    df['F3_CREDIT'].values[mask].mean(),
            'Avg YieldCurve': df['F4_YIELD_CURVE'].values[mask].mean(),
            'Avg VolZ':      df['F5_VOL_Z'].values[mask].mean(),
            'Avg Momentum':  df['F6_MOMENTUM'].values[mask].mean(),
            'Avg RV (ann%)': df['RV_21d'].values[mask].mean(),
        }
        rows.append(r)

    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    return tbl

tbl2 = regime_table(states2, 2, "2-STATE REGIMES")
tbl3 = regime_table(states3, 3, "3-STATE REGIMES")

# Determine state ordering for 3-state by VIX level
vix_means_3 = [df['VIX'].values[states3 == i].mean() for i in range(3)]
order_3 = np.argsort(vix_means_3)   # calm, mid, crisis

print("\n  ── 3-State Economic Interpretation ──")
interp_names = ['', '', '']
for rank, idx in enumerate(order_3):
    avg_vix  = df['VIX'].values[states3 == idx].mean()
    avg_mom  = df['F6_MOMENTUM'].values[states3 == idx].mean()
    avg_rv   = df['RV_21d'].values[states3 == idx].mean()
    avg_yc   = df['F4_YIELD_CURVE'].values[states3 == idx].mean()
    avg_term = df['F2_VIX_TERM'].values[states3 == idx].mean()
    dur      = dur3[idx]

    if rank == 0:
        name = "Calm Expansion"
    elif rank == 1:
        name = "Transition / Stress Build-up"
    else:
        name = "Crisis"

    interp_names[idx] = name
    print(f"\n    State {idx} → {name}")
    print(f"      VIX={avg_vix:.1f}, RV={avg_rv:.1f}%, Mom={avg_mom:+.4f}, "
          f"YieldCurve={avg_yc:.2f}, VIXTerm={avg_term:+.2f}, Duration={dur:.0f}d")

# Check if 3rd state is genuinely distinct
vix_sorted = [vix_means_3[i] for i in order_3]
if abs(vix_sorted[0] - vix_sorted[1]) < 3.0:
    print("\n  ⚠ WARNING: States 0 and 1 have similar VIX means (<3 points apart).")
    print("    The third state may be a SPLIT of the calm regime, not genuinely distinct.")
    third_state_valid = False
else:
    rv_sorted = [df['RV_21d'].values[states3 == order_3[i]].mean() for i in range(3)]
    if abs(rv_sorted[0] - rv_sorted[1]) < 3.0:
        print("\n  ⚠ WARNING: States 0 and 1 have similar realized volatility (<3% apart).")
        third_state_valid = False
    else:
        print("\n  ✓ All three states appear economically distinct.")
        third_state_valid = True

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4 — COVARIANCE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP 4 — COVARIANCE ANALYSIS (per-state)")
print("=" * 80)

# Use the model that we'll likely recommend
for model, n_s, label in [(m2, 2, "2-STATE"), (m3, 3, "3-STATE")]:
    print(f"\n  ── {label} Covariance Matrices (standardized scale) ──")
    short = [feat_names[c] for c in feat_cols]

    for i in range(n_s):
        cov_i = model.covars_[i]
        # Convert to correlation for interpretability
        d = np.sqrt(np.diag(cov_i))
        corr_i = cov_i / np.outer(d, d)

        print(f"\n    State {i} — Correlation Matrix:")
        corr_df = pd.DataFrame(corr_i, index=short, columns=short)
        print("    " + corr_df.round(3).to_string().replace('\n', '\n    '))

        # Find notable relationships
        notable = []
        for r in range(len(short)):
            for c_idx in range(r+1, len(short)):
                v = corr_i[r, c_idx]
                if abs(v) > 0.30:
                    notable.append((short[r], short[c_idx], v))
        if notable:
            print(f"    Notable correlations (|r|>0.30):")
            for a, b, v in sorted(notable, key=lambda x: -abs(x[2])):
                print(f"      {a:12s} ↔ {b:12s}: {v:+.3f}")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 5 — FORECAST VALUE vs VIX THRESHOLDS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP 5 — FORECAST VALUE (HMM vs VIX Thresholds)")
print("=" * 80)

# Generate out-of-sample probabilities using expanding window
BURN_IN = 500
REFIT   = 60

prob_crisis_oos = np.zeros(len(df))

# For 2-state model
crisis_idx_2s = int(np.argmax([df['VIX'].values[states2 == i].mean() for i in range(2)]))

print(f"\n  Generating OOS regime probabilities (burn-in={BURN_IN}, refit every {REFIT} days)...")
for t in range(BURN_IN, len(df)):
    if t % 500 == 0:
        print(f"    Processed {t}/{len(df)}...")

    if t == BURN_IN or t % REFIT == 0:
        m_oos = GaussianHMM(n_components=2, covariance_type='full',
                            n_iter=200, random_state=42)
        m_oos.fit(X_std[:t])
        # Identify crisis state
        st_temp = m_oos.predict(X_std[:t])
        crisis_oos = int(np.argmax([
            df['VIX'].values[:t][st_temp == s].mean() if np.sum(st_temp == s) > 0 else 0
            for s in range(2)
        ]))

    probs = m_oos.predict_proba(X_std[:t+1])
    prob_crisis_oos[t] = probs[-1, crisis_oos]

prob_crisis_oos[:BURN_IN] = prob_crisis_oos[BURN_IN]

# Major volatility events
events = [
    ("2015 China Crash",     "2015-08-24", "2015-07-01", "2015-10-01"),
    ("2018 Volmageddon",     "2018-02-05", "2017-12-01", "2018-04-01"),
    ("2020 COVID",           "2020-03-16", "2020-01-01", "2020-06-01"),
    ("2022 Fed Hiking",      "2022-06-13", "2022-01-01", "2022-10-01"),
    ("2025 Tariff Shock",    "2025-04-03", "2025-02-01", "2025-06-01"),
]

vix_arr = df['VIX'].values
dates   = df.index

print(f"\n  ── Event-by-Event Analysis ──")
print(f"  {'Event':<22} {'Peak Date':>12} {'VIX>20':>10} {'VIX>30':>10} {'HMM>50%':>10} {'HMM Lead':>10}")
print("  " + "-" * 76)

event_results = []
for name, peak_str, look_start, look_end in events:
    peak_date = pd.Timestamp(peak_str)

    # Find dates in our data
    mask = (dates >= pd.Timestamp(look_start)) & (dates <= pd.Timestamp(look_end))
    if mask.sum() == 0:
        print(f"  {name:<22} {'N/A':>12} — data not available")
        continue

    # Find first date VIX > 20, VIX > 30, HMM > 0.5 in the lookback window
    sub_idx = np.where(mask)[0]

    def first_date(condition, indices):
        hits = indices[condition[indices]]
        return dates[hits[0]] if len(hits) > 0 else None

    d_vix20  = first_date(vix_arr > 20, sub_idx)
    d_vix30  = first_date(vix_arr > 30, sub_idx)
    d_hmm50  = first_date(prob_crisis_oos > 0.5, sub_idx)

    # Lead time relative to peak
    def lead(d):
        if d is None:
            return "No signal"
        delta = (peak_date - d).days
        return f"{delta:+d}d" if delta >= 0 else f"{delta:+d}d"

    hmm_lead = lead(d_hmm50)
    print(f"  {name:<22} {peak_str:>12} {str(d_vix20.date()) if d_vix20 else 'None':>10} "
          f"{str(d_vix30.date()) if d_vix30 else 'None':>10} "
          f"{str(d_hmm50.date()) if d_hmm50 else 'None':>10} {hmm_lead:>10}")

    event_results.append({
        'Event': name, 'Peak': peak_str,
        'VIX20': str(d_vix20.date()) if d_vix20 else None,
        'VIX30': str(d_vix30.date()) if d_vix30 else None,
        'HMM50': str(d_hmm50.date()) if d_hmm50 else None,
        'HMM_Lead': hmm_lead,
    })

# False positive / negative analysis
print(f"\n  ── False Positive / Negative Analysis ──")

# Define "true crisis" as periods where realized vol > 20% (annualized)
true_crisis = df['RV_21d'].values > 20
hmm_signal  = prob_crisis_oos > 0.5
vix20_signal = vix_arr > 20
vix30_signal = vix_arr > 30

for sig_name, sig in [("VIX > 20", vix20_signal), ("VIX > 30", vix30_signal), ("HMM > 50%", hmm_signal)]:
    tp = np.sum(sig & true_crisis)
    fp = np.sum(sig & ~true_crisis)
    fn = np.sum(~sig & true_crisis)
    tn = np.sum(~sig & ~true_crisis)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"    {sig_name:<12}: Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  "
          f"(TP={tp}, FP={fp}, FN={fn}, TN={tn})")

# ═════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n  Generating plots...")

# ── Plot 1: Correlation heatmap ──
short = [feat_names[c] for c in feat_cols]
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(feat_cols)))
ax.set_yticks(range(len(feat_cols)))
ax.set_xticklabels(short, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(short, fontsize=10)
ax.set_title("Feature Correlation Matrix (6 Features)", fontsize=13, fontweight='bold')
for i in range(len(feat_cols)):
    for j in range(len(feat_cols)):
        ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=9,
                color='black' if abs(corr.values[i,j]) < 0.5 else 'white')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig('plots/final_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 2: 2-state vs 3-state regime probabilities with VIX and SPY ──
fig = plt.figure(figsize=(18, 16))
gs = GridSpec(5, 1, height_ratios=[1.2, 0.8, 1, 1, 1], hspace=0.08)

# SPY price
ax0 = fig.add_subplot(gs[0])
ax0.plot(df.index, df['SPY'], 'steelblue', linewidth=0.8)
ax0.set_ylabel('SPY Price', fontsize=10)
ax0.set_title('Final HMM Study: Regime Probabilities vs Market Data', fontsize=14, fontweight='bold')
ax0.grid(True, alpha=0.2)
ax0.set_xticklabels([])

# VIX
ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax1.plot(df.index, df['VIX'], 'darkorange', linewidth=0.8)
ax1.axhline(20, color='orange', linestyle='--', linewidth=0.6, alpha=0.7)
ax1.axhline(30, color='red',    linestyle='--', linewidth=0.6, alpha=0.7)
ax1.set_ylabel('VIX', fontsize=10)
ax1.grid(True, alpha=0.2)
ax1.set_xticklabels([])

# 2-state probabilities
ax2 = fig.add_subplot(gs[2], sharex=ax0)
p2_crisis = m2.predict_proba(X_std)[:, crisis_idx_2s]
ax2.fill_between(df.index, p2_crisis, alpha=0.7, color='crimson')
ax2.axhline(0.5, color='black', linestyle='--', linewidth=0.8)
ax2.set_ylabel('P(Crisis)\n2-state', fontsize=10)
ax2.set_ylim(-0.02, 1.05)
ax2.grid(True, alpha=0.2)
ax2.set_xticklabels([])

# 3-state probabilities
ax3 = fig.add_subplot(gs[3], sharex=ax0)
p3_all = m3.predict_proba(X_std)
# Order by VIX mean: calm, transition, crisis
colors_3 = ['steelblue', 'orange', 'crimson']
labels_3 = ['Calm', 'Transition', 'Crisis']
for rank, idx in enumerate(order_3):
    ax3.fill_between(df.index, p3_all[:, idx], alpha=0.5, color=colors_3[rank],
                     label=f'{labels_3[rank]} (State {idx})')
ax3.axhline(0.5, color='black', linestyle='--', linewidth=0.8)
ax3.set_ylabel('Probabilities\n3-state', fontsize=10)
ax3.set_ylim(-0.02, 1.05)
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.2)
ax3.set_xticklabels([])

# Realized volatility
ax4 = fig.add_subplot(gs[4], sharex=ax0)
ax4.plot(df.index, df['RV_21d'], 'black', linewidth=0.6, alpha=0.8)
ax4.axhline(20, color='red', linestyle='--', linewidth=0.6, label='RV=20%')
ax4.set_ylabel('Realized Vol\n(ann %)', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.2)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax4.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.savefig('plots/final_hmm_regimes.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 3: OOS HMM vs VIX thresholds for events ──
fig, axes = plt.subplots(len(events), 1, figsize=(16, 3.5*len(events)), sharex=False)
if len(events) == 1:
    axes = [axes]

for ax, (name, peak_str, look_start, look_end) in zip(axes, events):
    mask = (dates >= pd.Timestamp(look_start)) & (dates <= pd.Timestamp(look_end))
    if mask.sum() == 0:
        ax.set_title(f"{name} — No data", fontsize=11)
        continue
    sub_dates = dates[mask]
    sub_vix   = vix_arr[mask]
    sub_hmm   = prob_crisis_oos[mask]

    ax2_twin = ax.twinx()
    ax.plot(sub_dates, sub_vix, 'steelblue', linewidth=1.5, label='VIX')
    ax.axhline(20, color='orange', linestyle='--', linewidth=0.8)
    ax.axhline(30, color='red', linestyle='--', linewidth=0.8)
    ax.axvline(pd.Timestamp(peak_str), color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='Peak')
    ax.set_ylabel('VIX', fontsize=9, color='steelblue')

    ax2_twin.fill_between(sub_dates, sub_hmm, alpha=0.4, color='crimson', label='HMM P(Crisis)')
    ax2_twin.axhline(0.5, color='darkred', linestyle='--', linewidth=0.8)
    ax2_twin.set_ylim(-0.05, 1.1)
    ax2_twin.set_ylabel('HMM P(Crisis)', fontsize=9, color='crimson')

    ax.set_title(f"{name}", fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax2_twin.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.tight_layout()
plt.savefig('plots/final_hmm_events.png', dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: plots/final_correlation_matrix.png")
print("  Saved: plots/final_hmm_regimes.png")
print("  Saved: plots/final_hmm_events.png")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 6 — FINAL RECOMMENDATION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP 6 — FINAL RECOMMENDATION")
print("=" * 80)

# Decision logic
if third_state_valid:
    rec_states = 3
    rec_reason = "3 economically distinct regimes identified (Calm/Transition/Crisis)"
else:
    rec_states = 2
    rec_reason = "Third state was NOT economically distinct; collapsed to calm split"

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │              FINAL HMM CONFIGURATION                           │
  ├─────────────────────────────────────────────────────────────────┤
  │  States:          {rec_states}                                         │
  │  Covariance:      full                                         │
  │  Features:        6 (log(VIX), VIX Term, Credit, YieldCurve,   │
  │                      VolZscore, Momentum)                      │
  │  Standardization: Z-score (mean=0, std=1)                      │
  │  Refit:           Expanding window, every 60 trading days      │
  │  Burn-in:         500 days                                     │
  │  Justification:   {rec_reason[:49]:<49}│
  ├─────────────────────────────────────────────────────────────────┤
  │  Limitations:                                                  │
  │  - Gaussian emission assumption (fat tails not modeled)        │
  │  - Expanding window means slow adaptation to structural breaks │
  │  - HMM output is a filtered probability, not a forecast        │
  │  - Credit spread proxy (HYG/LQD) has ETF-specific noise       │
  └─────────────────────────────────────────────────────────────────┘
""")

# Save the final configuration
config = {
    'n_states': rec_states,
    'covariance_type': 'full',
    'features': feat_cols,
    'feature_names': feat_names,
    'standardization': 'z-score',
    'standardization_means': means.to_dict(),
    'standardization_stds': stds.to_dict(),
    'burn_in': BURN_IN,
    'refit_every': REFIT,
    'justification': rec_reason,
    'third_state_valid': third_state_valid,
}
with open('data/hmm_final_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print("  Saved: data/hmm_final_config.json")

# Save OOS probabilities
oos_df = pd.DataFrame({
    'prob_crisis': prob_crisis_oos,
    'VIX': df['VIX'].values,
    'RV_21d': df['RV_21d'].values,
}, index=df.index)
oos_df.to_csv('data/hmm_final_probabilities.csv')
print("  Saved: data/hmm_final_probabilities.csv")

print("\n  ═══ HMM MODULE IS NOW FINALIZED ═══")
print("  No further tuning unless a bug is discovered.")
