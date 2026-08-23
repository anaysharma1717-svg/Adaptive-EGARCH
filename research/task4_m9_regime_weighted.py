"""
Task 4: M9 regime-weighted combination of M1 HAR+TS and M2b Corrected EGARCH.
Weight: logistic in the lagged (ex-ante) percentile rank of rv1 within its own
trailing 252-day window. (c,k) fit every 21-day refit on the expanding training
window only, k capped at 20. Benchmarked against corrected-M3, M1, and
corrected-EGARCH (M2b) alone: RMSE/MAE/QLIKE overall + crisis/calm, plus
HLN-corrected DM tests of M9 vs M3 (comparison model) on both losses.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from extended_model_zoo import fetch_data, compute_features, walk_forward, qlike, dm_test

df = fetch_data()
data = compute_features(df)

m9_params = []
actuals, results, test_dates = walk_forward(data, m9_param_log=m9_params)

print(f"\nN={len(actuals)}  {test_dates[0].date()} -> {test_dates[-1].date()}")

# --- M9 fitted-parameter audit trail ---
params_df = pd.DataFrame(m9_params)
params_df.to_csv("research/results/task4_m9_params_history.csv", index=False)
print(f"\nM9 fitted (c,k) at each of the {len(params_df)} refits:")
print(params_df.to_string(index=False))
print(f"\nk hit the cap (20) in {int((params_df['k'] >= 20).sum())}/{len(params_df)} refits")

# --- crisis threshold: 90th pct of TRAINING actual RV (same definition as Task 1) ---
test_start = test_dates[0]
train_rv = data.loc[data.index < test_start, "rv"]
crisis_threshold = float(np.percentile(train_rv, 90))
print(f"\nCrisis threshold (90th pct of training RV, N={len(train_rv)}): {crisis_threshold:.4f}")

crisis_mask = actuals > crisis_threshold
calm_mask = ~crisis_mask
print(f"Crisis days: {crisis_mask.sum()}  Calm days: {calm_mask.sum()}")

# --- save all per-day forecasts + losses ---
out = pd.DataFrame({"date": test_dates, "actual": actuals, "crisis": crisis_mask})
for name in ["M1 HAR+TS", "M2b Corrected EGARCH", "M3 Combined", "M9 Regime"]:
    out[name] = results[name]
    out[f"{name}_sq_err"] = (actuals - results[name]) ** 2
    out[f"{name}_abs_err"] = np.abs(actuals - results[name])
out.to_csv("research/results/task4_m9_forecasts_losses.csv", index=False)
print(f"\nSaved: research/results/task4_m9_forecasts_losses.csv ({out.shape[0]} rows)")

# --- metrics ---
def metrics(a, f):
    return dict(RMSE=float(np.sqrt(np.mean((a - f) ** 2))), MAE=float(np.mean(np.abs(a - f))), QLIKE=qlike(a, f))

models_to_bench = ["M9 Regime", "M3 Combined", "M1 HAR+TS", "M2b Corrected EGARCH"]
rows = []
for label, mask in [("overall", np.ones(len(actuals), dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for name in models_to_bench:
        m = metrics(actuals[mask], results[name][mask])
        rows.append({"subsample": label, "model": name, "N": int(mask.sum()), **m})

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv("research/results/task4_m9_benchmark_metrics.csv", index=False)
print(f"\n{'subsample':<10}{'model':<24}{'N':>5}{'RMSE':>10}{'MAE':>10}{'QLIKE':>10}")
for _, r in metrics_df.iterrows():
    print(f"{r['subsample']:<10}{r['model']:<24}{r['N']:>5}{r['RMSE']:>10.4f}{r['MAE']:>10.4f}{r['QLIKE']:>10.4f}")

# --- DM tests: M9 vs M3 (comparison model), HLN-corrected, both losses, 3 subsamples ---
m3 = results["M3 Combined"]
m9 = results["M9 Regime"]
dm_rows = []
for label, mask in [("overall", np.ones(len(actuals), dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for loss in ("mse", "qlike"):
        stat, pval = dm_test(actuals[mask], m3[mask], m9[mask], loss=loss)
        dm_rows.append({"subsample": label, "loss": loss, "N": int(mask.sum()), "dm_stat": stat, "p_value": pval})

dm_df = pd.DataFrame(dm_rows)
dm_df.to_csv("research/results/task4_m9_vs_m3_dm.csv", index=False)
print(f"\nDM tests: M9 vs M3 (positive stat => M9 better)")
print(dm_df.to_string(index=False))
print("\nSaved: research/results/task4_m9_params_history.csv")
print("Saved: research/results/task4_m9_benchmark_metrics.csv")
print("Saved: research/results/task4_m9_vs_m3_dm.csv")
