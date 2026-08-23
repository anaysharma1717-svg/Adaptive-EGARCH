"""
Read-only confirmation run. Imports the EXISTING, unmodified functions from
extended_model_zoo.py and executes the default walk-forward pipeline, then
dumps the day-by-day actual + all model forecasts to CSV.

Does not modify extended_model_zoo.py. Purpose: verify the test window length
and reproduce the cited baseline numbers (corrected M3 RMSE, M4 MZ beta) before
any task begins.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from extended_model_zoo import (
    fetch_data, compute_features, walk_forward, MODELS,
    dm_test, mincer_zarnowitz, qlike,
)

df = fetch_data()
data = compute_features(df)
print(f"Full feature set: {len(data)} obs, {data.index[0].date()} -> {data.index[-1].date()}")

actuals, results, test_dates = walk_forward(data)  # all defaults, unmodified

print(f"\nTEST WINDOW: N={len(actuals)}  {test_dates[0].date()} -> {test_dates[-1].date()}")

out = pd.DataFrame({"date": test_dates, "actual": actuals})
for name, fc in results.items():
    out[name] = fc
os.makedirs("research/results", exist_ok=True)
out.to_csv("research/results/baseline_confirm_forecasts.csv", index=False)
print(f"Saved: research/results/baseline_confirm_forecasts.csv  ({out.shape[0]} rows, {out.shape[1]} cols)")

def metrics(a, f):
    return dict(RMSE=float(np.sqrt(np.mean((a-f)**2))), MAE=float(np.mean(np.abs(a-f))), QLIKE=qlike(a,f))

print("\n--- Key metrics ---")
for name in ["M1 HAR+TS", "M2 EGARCH", "M2b Corrected EGARCH", "M3 Combined", "M4 Log-HAR"]:
    m = metrics(actuals, results[name])
    print(f"  {name:<24} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  QLIKE={m['QLIKE']:.4f}")

print("\n--- M4 Log-HAR MZ regression ---")
mz4 = mincer_zarnowitz(actuals, results["M4 Log-HAR"])
print(f"  alpha={mz4['alpha']:.4f}  beta={mz4['beta']:.4f}  R2={mz4['R2']:.4f}  F_pval={mz4['F_pval']:.6f}")

print("\n--- M4 vs M1 DM test (qlike) ---")
stat, p = dm_test(actuals, results["M1 HAR+TS"], results["M4 Log-HAR"], loss="qlike")
print(f"  DM stat={stat:.4f}  p={p:.6f}")

print("\n--- M2b Corrected EGARCH MZ regression ---")
mz2b = mincer_zarnowitz(actuals, results["M2b Corrected EGARCH"])
print(f"  alpha={mz2b['alpha']:.4f}  beta={mz2b['beta']:.4f}")
