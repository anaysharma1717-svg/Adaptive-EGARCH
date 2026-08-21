"""
hmm_diagnostics.py  --  Verify HMM model and generate all diagnostic plots

Plots generated:
  1. HMM regime probability vs VIX (full history)
  2. Zoom into 2020 COVID crash
  3. omega_t path vs regime
  4. Distribution of omega in calm vs crisis regime
  5. HMM state transition visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings("ignore")

# ── Load data ────────────────────────────────────────────────────────────────
hmm_df   = pd.read_csv('data/hmm_features.csv', index_col=0)
hmm_df.index = pd.to_datetime(hmm_df.index)

v5_fc    = pd.read_csv('data/adaptive_v5_forecasts.csv', index_col=0)
v5_fc.index = pd.to_datetime(v5_fc.index)

feat_df  = pd.read_csv('data/dp_egarch_features.csv', index_col=0)
feat_df.index = pd.to_datetime(feat_df.index)

# ── Align ────────────────────────────────────────────────────────────────────
combined = hmm_df.join(v5_fc[['sigma_v5_h1', 'egarch_fc_vol', 'garch_fc_vol', 'rv_target']], how='inner')
combined = combined.dropna()
print(f"Combined dataset: {len(combined)} rows, {combined.index[0].date()} → {combined.index[-1].date()}")

# ── 1. Full-history: VIX and HMM regime probability ─────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle("HMM Regime Verification: VIX vs Crisis Probability", fontsize=15, fontweight='bold')

ax1, ax2, ax3 = axes

# Panel 1: VIX
ax1.plot(hmm_df.index, hmm_df['VIX'], color='steelblue', linewidth=0.8, label='VIX')
ax1.axhline(20, color='orange', linestyle='--', linewidth=0.8, label='VIX=20 (moderate stress)')
ax1.axhline(30, color='red',    linestyle='--', linewidth=0.8, label='VIX=30 (crisis)')
ax1.set_ylabel('VIX Level', fontsize=10)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)

# Panel 2: HMM crisis probability
p = hmm_df['prob_crisis']
ax2.fill_between(hmm_df.index, p, alpha=0.6, color='crimson', label='P(Crisis State)')
ax2.axhline(0.5, color='black', linestyle='--', linewidth=0.8, label='50% threshold')
ax2.set_ylabel('P(Crisis State)', fontsize=10)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.3)

# Panel 3: Binary regime call (>50% = crisis)
regime = (p > 0.5).astype(int)
ax3.fill_between(hmm_df.index, regime, alpha=0.5, color='darkred', label='Crisis (P>0.5)')
ax3.fill_between(hmm_df.index, 1 - regime, alpha=0.3, color='steelblue', label='Calm (P≤0.5)')
ax3.set_ylabel('Regime', fontsize=10)
ax3.set_ylim(0, 1.05)
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.3)

ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.savefig('plots/hmm_regime_verification.png', dpi=150, bbox_inches='tight')
print("Saved: plots/hmm_regime_verification.png")
plt.close()

# ── 2. Zoom into 2020 COVID crash ────────────────────────────────────────────
z_hmm = hmm_df.loc['2019-10-01':'2021-01-01']
print(f"COVID zoom rows: {len(z_hmm)}  ({z_hmm.index[0].date()} \u2192 {z_hmm.index[-1].date()})")
print(f"  VIX range: {z_hmm['VIX'].min():.1f} \u2013 {z_hmm['VIX'].max():.1f}")
print(f"  P(Crisis) range: {z_hmm['prob_crisis'].min():.4f} \u2013 {z_hmm['prob_crisis'].max():.4f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle("HMM Zoom: 2020 COVID Crash  (Oct 2019 – Jan 2021)", fontsize=14, fontweight='bold')

ax1.plot(z_hmm.index, z_hmm['VIX'], color='steelblue', linewidth=1.4, label='VIX')
ax1.fill_between(z_hmm.index, z_hmm['VIX'], alpha=0.15, color='steelblue')
ax1.axhline(20, color='orange', linestyle='--', linewidth=0.9, label='VIX = 20')
ax1.axhline(30, color='red',    linestyle='--', linewidth=0.9, label='VIX = 30 (crisis)')
ax1.set_ylabel('VIX Level', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.fill_between(z_hmm.index, z_hmm['prob_crisis'], alpha=0.75, color='crimson', label='P(Crisis State)')
ax2.axhline(0.5, color='black', linestyle='--', linewidth=1.1, label='50% threshold')
ax2.set_ylabel('P(Crisis State)', fontsize=11)
ax2.set_ylim(-0.02, 1.05)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/hmm_covid_zoom.png', dpi=150, bbox_inches='tight')
print("Saved: plots/hmm_covid_zoom.png")
plt.close()

# ── 3. Omega_t path (derived from prob_crisis) vs regime ────────────────────
# omega_t = -0.05 + 0.10 * sigmoid(w @ X)
# In v5, X = [1, prob_crisis] and w was learned.
# We approximate the omega_t range here to show what the model was doing.
# Use actual forecasts as a proxy: compare sigma_v5 vs regime.

def sigmoid(x): return 1 / (1 + np.exp(-x))

# Load what omega_t looked like given the v5 design
# omega range: (-0.05, +0.05)
# When prob_crisis=0 => omega near -0.05 (calm baseline)
# When prob_crisis=1 => omega near +0.05 (crisis baseline)
p_crisis = combined['prob_crisis']

# Approximate omega given the sigmoid transform
# (w_intercept + w_crisis * prob_crisis): at prob_crisis=0, omega=-0.05; at 1, omega=+0.05
omega_approx_low  = -0.05 + 0.10 * sigmoid(-2 + 0 * p_crisis)   # if w=[−2, 0]
omega_approx_opt  = -0.05 + 0.10 * sigmoid(0 * (1 - p_crisis) + 4 * p_crisis)  # shifted

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle("omega_t Path vs HMM Regime vs Forecast vs Realized", fontsize=13, fontweight='bold')

# Panel 1: Regime probability
axes[0].fill_between(combined.index, combined['prob_crisis'], alpha=0.6, color='crimson', label='P(Crisis)')
axes[0].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
axes[0].set_ylabel('P(Crisis)', fontsize=10)
axes[0].set_ylim(0, 1.05)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Panel 2: Realized vol vs model forecasts
axes[1].plot(combined.index, combined['rv_target'],    color='black',   linewidth=0.8, label='Realized Vol (YZ)', alpha=0.9)
axes[1].plot(combined.index, combined['garch_fc_vol'], color='steelblue', linewidth=0.8, label='GARCH(1,1)', alpha=0.8)
axes[1].plot(combined.index, combined['egarch_fc_vol'],color='darkorange', linewidth=0.8, label='EGARCH', alpha=0.8)
axes[1].plot(combined.index, combined['sigma_v5_h1'],  color='crimson',  linewidth=0.8, label='v5 HMM-Adaptive', alpha=0.8)
axes[1].set_ylabel('Volatility (% /day)', fontsize=10)
axes[1].legend(fontsize=8, loc='upper left')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 6)

# Panel 3: Forecast error (v5 vs GARCH): who is closer to realized?
err_garch = (combined['garch_fc_vol'] - combined['rv_target']).abs()
err_v5    = (combined['sigma_v5_h1']  - combined['rv_target']).abs()
diff      = err_v5 - err_garch   # positive = v5 is WORSE than GARCH
axes[2].fill_between(combined.index, diff, where=(diff < 0), alpha=0.6, color='green',  label='v5 BETTER than GARCH')
axes[2].fill_between(combined.index, diff, where=(diff >= 0), alpha=0.5, color='red',    label='v5 WORSE than GARCH')
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_ylabel('|err_v5| − |err_GARCH|', fontsize=10)
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[2].xaxis.set_major_locator(mdates.YearLocator())
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.savefig('plots/omega_vs_regime_vs_forecast.png', dpi=150, bbox_inches='tight')
print("Saved: plots/omega_vs_regime_vs_forecast.png")
plt.close()

# ── 4. Distribution of sigma_v5 in Calm vs Crisis ────────────────────────────
calm   = combined[combined['prob_crisis'] < 0.5]
crisis = combined[combined['prob_crisis'] >= 0.5]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Forecast Distribution: Calm vs Crisis Regime", fontsize=13, fontweight='bold')

for ax, col, label in zip(axes,
    ['sigma_v5_h1', 'garch_fc_vol', 'rv_target'],
    ['v5 Adaptive Forecast', 'GARCH Forecast', 'Realized Vol']):
    ax.hist(calm[col],   bins=60, alpha=0.6, color='steelblue', label='Calm',   density=True)
    ax.hist(crisis[col], bins=60, alpha=0.6, color='crimson',   label='Crisis', density=True)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel('Vol (% /day)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/hmm_distribution_calm_vs_crisis.png', dpi=150, bbox_inches='tight')
print("Saved: plots/hmm_distribution_calm_vs_crisis.png")
plt.close()

# ── 5. What factor drives the HMM? Retrain once and inspect ──────────────────
print("\n--- HMM Model Inspection ---")
vix_data = np.log(hmm_df['VIX'].values).reshape(-1, 1)
model = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=42)
model.fit(vix_data)

crisis_state = np.argmax(model.means_.flatten())
calm_state   = 1 - crisis_state

print(f"HMM Input Feature: log(VIX)")
print(f"State 0 mean log(VIX) = {model.means_[0][0]:.4f}  => VIX ≈ {np.exp(model.means_[0][0]):.1f}")
print(f"State 1 mean log(VIX) = {model.means_[1][0]:.4f}  => VIX ≈ {np.exp(model.means_[1][0]):.1f}")
print(f"Crisis state = State {crisis_state}")
print(f"\nTransition matrix:")
print(f"  P(calm  → calm)   = {model.transmat_[calm_state][calm_state]:.4f}")
print(f"  P(calm  → crisis) = {model.transmat_[calm_state][crisis_state]:.4f}")
print(f"  P(crisis→ crisis) = {model.transmat_[crisis_state][crisis_state]:.4f}")
print(f"  P(crisis→ calm)   = {model.transmat_[crisis_state][calm_state]:.4f}")

# ── 6. Print regime summary statistics ───────────────────────────────────────
print(f"\n--- Regime Summary (by P>0.5 threshold) ---")
print(f"Calm days   : {len(calm)} ({100*len(calm)/len(combined):.1f}%)")
print(f"Crisis days : {len(crisis)} ({100*len(crisis)/len(combined):.1f}%)")
print(f"\nRMSE in Calm regime:")
for m, c in [('GARCH', 'garch_fc_vol'), ('EGARCH', 'egarch_fc_vol'), ('v5 HMM', 'sigma_v5_h1')]:
    rmse = np.sqrt(np.mean((calm[c] - calm['rv_target'])**2))
    print(f"  {m:12s}: {rmse:.4f}")
print(f"\nRMSE in Crisis regime:")
for m, c in [('GARCH', 'garch_fc_vol'), ('EGARCH', 'egarch_fc_vol'), ('v5 HMM', 'sigma_v5_h1')]:
    rmse = np.sqrt(np.mean((crisis[c] - crisis['rv_target'])**2))
    print(f"  {m:12s}: {rmse:.4f}")

print("\nAll plots saved to plots/")
