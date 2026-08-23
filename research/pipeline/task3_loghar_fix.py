"""
Task 3: Log-HAR (M4) Jensen/smearing retransformation fix -- before/after.
"before" numbers reused from the already-confirmed baseline run (unmodified code);
"after" numbers from a fresh run of the now-patched ModelSpec.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from extended_model_zoo import fetch_data, compute_features, walk_forward, qlike, mincer_zarnowitz

before = pd.read_csv("research/results/baseline_confirm_forecasts.csv", parse_dates=["date"])
before_actual = before["actual"].values
before_m4 = before["M4 Log-HAR"].values

def metrics(a, f):
    return dict(RMSE=float(np.sqrt(np.mean((a-f)**2))), MAE=float(np.mean(np.abs(a-f))), QLIKE=qlike(a,f))

before_m = metrics(before_actual, before_m4)
before_mz = mincer_zarnowitz(before_actual, before_m4)

df = fetch_data()
data = compute_features(df)
actuals, results, test_dates = walk_forward(data)  # now uses the patched ModelSpec
after_m4 = results["M4 Log-HAR"]

after_m = metrics(actuals, after_m4)
after_mz = mincer_zarnowitz(actuals, after_m4)

print(f"N before={len(before_actual)}  N after={len(actuals)}")
print(f"\n{'':10} {'RMSE':>10} {'MAE':>10} {'QLIKE':>10} {'alpha':>10} {'beta':>10}")
print(f"{'BEFORE':10} {before_m['RMSE']:>10.4f} {before_m['MAE']:>10.4f} {before_m['QLIKE']:>10.4f} {before_mz['alpha']:>10.4f} {before_mz['beta']:>10.4f}")
print(f"{'AFTER':10} {after_m['RMSE']:>10.4f} {after_m['MAE']:>10.4f} {after_m['QLIKE']:>10.4f} {after_mz['alpha']:>10.4f} {after_mz['beta']:>10.4f}")

out = pd.DataFrame({"date": test_dates, "actual": actuals, "M4_before": before_m4, "M4_after": after_m4})
out.to_csv("research/results/task3_loghar_before_after.csv", index=False)
print("\nSaved: research/results/task3_loghar_before_after.csv")

summary = pd.DataFrame([
    {"stage": "before", **before_m, "alpha": before_mz["alpha"], "beta": before_mz["beta"]},
    {"stage": "after", **after_m, "alpha": after_mz["alpha"], "beta": after_mz["beta"]},
])
summary.to_csv("research/results/task3_loghar_metrics.csv", index=False)
print("Saved: research/results/task3_loghar_metrics.csv")
