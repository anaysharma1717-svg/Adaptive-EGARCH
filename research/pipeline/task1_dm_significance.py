"""
Task 1: HLN-corrected Diebold-Mariano, corrected-M3 vs M1 HAR+TS, on QLIKE and MSE.
Overall + crisis (actual RV > 90th pct of TRAINING distribution) + calm remainder.

Reuses the forecast arrays already confirmed in baseline_confirm_forecasts.csv
(no re-run of the walk-forward, so no new EGARCH-simulation noise is introduced).
Crisis threshold is computed from data strictly before the test window starts
(the training set at the point the test window begins) -- not from the test
actuals themselves, per the task's explicit instruction.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from extended_model_zoo import fetch_data, compute_features, dm_test

# --- reload the same data used for the confirmed run, to get the training threshold ---
df = fetch_data()
data = compute_features(df)

fc = pd.read_csv("research/results/baseline_confirm_forecasts.csv", parse_dates=["date"])
test_start = fc["date"].iloc[0]

train_rv = data.loc[data.index < test_start, "rv"]
crisis_threshold = float(np.percentile(train_rv, 90))
print(f"Training set: {train_rv.index[0].date()} -> {train_rv.index[-1].date()}  (N={len(train_rv)})")
print(f"90th percentile of TRAINING actual RV: {crisis_threshold:.4f}")

actual = fc["actual"].values
m1 = fc["M1 HAR+TS"].values
m3 = fc["M3 Combined"].values

crisis_mask = actual > crisis_threshold
calm_mask = ~crisis_mask
print(f"Test days: N={len(actual)}  crisis(N={crisis_mask.sum()})  calm(N={calm_mask.sum()})")

rows = []
for label, mask in [("overall", np.ones(len(actual), dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    a, f1, f3 = actual[mask], m1[mask], m3[mask]
    for loss in ("mse", "qlike"):
        stat, pval = dm_test(a, f1, f3, loss=loss)
        rows.append({"subsample": label, "loss": loss, "N": int(mask.sum()), "dm_stat": stat, "p_value": pval})
        print(f"  [{label:<8}] {loss.upper():>6}  N={mask.sum():4d}  DM={stat:+.4f}  p={pval:.4f}")

out = pd.DataFrame(rows)
out.to_csv("research/results/task1_dm_results.csv", index=False)
print("\nSaved: research/results/task1_dm_results.csv")

# also save the crisis/calm labeling used, for audit
fc["crisis"] = crisis_mask
fc[["date", "actual", "M1 HAR+TS", "M3 Combined", "crisis"]].to_csv(
    "research/results/task1_labeled_forecasts.csv", index=False)
print("Saved: research/results/task1_labeled_forecasts.csv")
