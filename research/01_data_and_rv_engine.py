"""
STAGE 1 -- Data and the Realized-Volatility engine.

WHAT THIS SCRIPT ACTUALLY DOES: recomputes from raw data. It re-fetches SPY
and rebuilds the RV series and ACF/PACF from scratch -- this is the one
script in research/ (top level) that does not depend on any already-saved
CSV. For the scripts that originally computed everything downstream of this,
see research/pipeline/.

Question this stage answers: what are we actually forecasting, and why is it
measured this way?

This is the foundation everything downstream depends on: SPY daily OHLC data,
turned into a Garman-Klass realized-volatility (GK-RV) series, plus a look at
that series' own autocorrelation structure -- which is the empirical
justification for the HAR (Heterogeneous Autoregressive) lag structure
(1-day, 5-day, 22-day averages) used by every HAR-based model in this
project (stages 2, 4, 5).

Note on scope: the ACF/PACF diagnostic below is computed fresh here (it was
never saved as its own artifact earlier in the project -- the model-adequacy
PACF check inside extended_model_zoo.py's full_report() operates on model
RESIDUALS, a different, later-stage diagnostic, not this one). Everything
else in this script (the data and the RV formula) reproduces exactly what
every later stage already used.
"""
import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))
from extended_model_zoo import fetch_data, compute_features

OUT_DIR = os.path.join("research", "results", "01_data_rv_engine")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 90)
print("  STAGE 1: DATA AND THE RV ENGINE")
print("=" * 90)

# --- 1. Raw data ---
# yfinance daily OHLC for SPY. auto_adjust=True means splits/dividends are
# already folded into the price series, which matters for a *return* series
# (an un-adjusted split would otherwise look like a fake -50% return).
df = fetch_data()
print(f"\nRaw OHLC: {len(df)} trading days, {df.index[0].date()} -> {df.index[-1].date()}")

# --- 2. Garman-Klass realized volatility ---
# Why not just use squared close-to-close returns as the volatility target?
# A single day's squared return only uses 2 price points (yesterday's close,
# today's close) and is a very noisy (high-variance) estimate of that day's
# true variance. Garman-Klass uses all four OHLC prices -- the day's full
# trading range -- and is a textbook-standard MORE STATISTICALLY EFFICIENT
# (lower-variance, still unbiased under a random-walk assumption) estimator.
# A less noisy target makes every downstream model comparison less noisy too.
log_hl = np.log(df["High"] / df["Low"])
log_co = np.log(df["Close"] / df["Open"])
gk_var = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
rv = np.sqrt(np.maximum(gk_var, 1e-10)) * 100.0     # daily % volatility, floored to avoid log(0)/sqrt(neg) issues

print(f"\nGK-RV summary statistics (daily %):")
print(f"  mean={rv.mean():.4f}  std={rv.std():.4f}  min={rv.min():.4f}  max={rv.max():.4f}")
print(f"  90th percentile (the crisis threshold used throughout this project): {np.percentile(rv.dropna(), 90):.4f}")

# --- 3. ACF / PACF of the RV series -- why HAR uses lags 1, 5, 22 ---
# ACF (autocorrelation function): how correlated is RV_t with RV_{t-k}, for
# each lag k? PACF (partial ACF): the SAME thing but with the effect of all
# shorter lags already removed -- it isolates the "direct" contribution of
# lag k. Volatility is famous for having a slowly-decaying ACF (a value
# called "long memory") rather than the sharp cutoff a simple AR(1) process
# would show. HAR's specific fix -- averaging over 1-day, 5-day (~1 trading
# week), and 22-day (~1 trading month) windows -- is a cheap, three-parameter
# way to approximate that slow decay without fitting a full long-memory model.
clean_rv = rv.dropna().values
acf_vals = acf(clean_rv, nlags=30, fft=True)
pacf_vals = pacf(clean_rv, nlags=30, method="ywm")
ci = 1.96 / np.sqrt(len(clean_rv))     # approx 95% significance band for a white-noise series

print(f"\nACF of RV, selected lags (95% significance band = +/-{ci:.4f}):")
for lag in [1, 2, 3, 5, 10, 22, 30]:
    print(f"  lag {lag:>3}: ACF={acf_vals[lag]:+.4f}  PACF={pacf_vals[lag]:+.4f}  "
          f"{'(significant)' if abs(acf_vals[lag]) > ci else ''}")

acf_df = pd.DataFrame({"lag": range(len(acf_vals)), "acf": acf_vals, "pacf": pacf_vals})
acf_df.to_csv(os.path.join(OUT_DIR, "acf_pacf_rv.csv"), index=False)

# --- 4. Feature engineering (unchanged, imported from the shared engine) ---
# Every feature below uses .shift(1) or later -- this is the single most
# important line in the whole project: it's what guarantees no model ever
# sees day t's own information when predicting day t.
data = compute_features(df)
print(f"\nFeature-engineered dataset: {len(data)} rows (shorter than raw due to warmup:"
      f" rv22 needs 22 prior days, rv1_pctile needs 252 prior days)")
print(f"Columns: {list(data.columns)}")

summary = pd.DataFrame([{
    "raw_rows": len(df), "feature_rows": len(data),
    "date_start": data.index[0].date().isoformat(), "date_end": data.index[-1].date().isoformat(),
    "rv_mean": float(rv.mean()), "rv_std": float(rv.std()),
    "rv_90th_pctile_full_sample": float(np.percentile(clean_rv, 90)),
}])
summary.to_csv(os.path.join(OUT_DIR, "data_rv_summary.csv"), index=False)

print(f"\nSaved: {OUT_DIR}/acf_pacf_rv.csv")
print(f"Saved: {OUT_DIR}/data_rv_summary.csv")
