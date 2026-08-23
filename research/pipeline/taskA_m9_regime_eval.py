"""
Task A: M9 regime-weighted EGARCH/HAR combination -- full evaluation.

Prerequisite check (reported, not re-run as new work -- verified by direct
inspection of the current extended_model_zoo.py, unmodified since it was
written):
  Task 2 (MZ-correction leakage audit): walk_forward()'s `hist = data[data.index
    < tdate]` (line 275) is the only data the MZ fit (lines 296-301) ever sees --
    strictly expanding, no day-t-or-later leakage. Confirmed, not re-run.
  Task 3 (Log-HAR Jensen/smearing retransformation): ModelSpec.fit()/predict()
    (lines 207-237) still carries exp(sigma^2/2), computed from in-sample
    residuals on the same `hist` window as the fit. Confirmed, not re-run.

Step 0.5 (this script, below): the motivating claim -- raw EGARCH (M2) beats
M1 HAR+TS on QLIKE in the crisis regime, DM=+2.791 p=0.0083 N=38 -- is
verified directly here, since Task 1 tested corrected-M3 vs M1 only, not raw
EGARCH vs M1. Reported as it comes out, not assumed.

M9's construction (regime variable = ex-ante trailing-252d percentile rank of
rv1; logistic weight; (c,k) fit by grid search on the expanding TRAINING
window only, every 21-day refit, k capped at 20; forecast = w*corrected_EGARCH
+ (1-w)*HAR+TS) is UNCHANGED from Task 4 and matches Task A's spec exactly --
reused as-is, not rebuilt. See extended_model_zoo.py lines 98-103, 311-334,
363-373 for the implementation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from extended_model_zoo import fetch_data, compute_features, walk_forward, qlike, dm_test, mincer_zarnowitz

RESULTS_DIR = "research/results"

df = fetch_data()
data = compute_features(df)

m9_params = []
actuals, results, test_dates = walk_forward(data, m9_param_log=m9_params)
actuals = np.asarray(actuals)
n = len(actuals)
print(f"\nN={n}  {test_dates[0].date()} -> {test_dates[-1].date()}")

m1 = np.asarray(results["M1 HAR+TS"])
m2 = np.asarray(results["M2 EGARCH"])
m2b = np.asarray(results["M2b Corrected EGARCH"])
m3 = np.asarray(results["M3 Combined"])
m9 = np.asarray(results["M9 Regime"])

# --- training crisis threshold: 90th pct of TRAINING actual RV (same convention as Task 1/4) ---
test_start = test_dates[0]
train_rv = data.loc[data.index < test_start, "rv"]
crisis_threshold = float(np.percentile(train_rv, 90))
crisis_mask = actuals > crisis_threshold
calm_mask = ~crisis_mask
print(f"Crisis threshold (90th pct training RV, N={len(train_rv)}): {crisis_threshold:.4f}")
print(f"Crisis days: {crisis_mask.sum()}  Calm days: {calm_mask.sum()}")

# ---------------------------------------------------------------------------
# Step 0.5: verify the motivating claim directly (not assumed)
# ---------------------------------------------------------------------------
stat_v, pval_v = dm_test(actuals[crisis_mask], m1[crisis_mask], m2[crisis_mask], loss="qlike")
print(f"\n[VERIFY MOTIVATION] Raw EGARCH (M2) vs M1 HAR+TS, QLIKE, crisis: "
      f"DM={stat_v:+.4f}  p={pval_v:.4f}  N={int(crisis_mask.sum())}")
print(f"  Cited: DM=+2.791 p=0.0083 N=38")
match = abs(stat_v - 2.791) < 0.05 and abs(pval_v - 0.0083) < 0.005 and int(crisis_mask.sum()) == 38
print(f"  {'MATCHES cited figures.' if match else 'DOES NOT exactly match cited figures -- reporting actual numbers above, proceeding on the ACTUAL result.'}")

# ---------------------------------------------------------------------------
# M9 functional form (restated, unchanged from Task 4's already-approved spec)
# ---------------------------------------------------------------------------
print("\nM9 functional form (unchanged from Task 4, already approved -- reused, not rebuilt):")
print("  regime p_t  = ex-ante trailing-252d percentile rank of rv1 (uses data through t-1 only)")
print("  w_t         = 1 / (1 + exp(-k * (p_t - c)))          [logistic, smooth, no hard switch]")
print("  M9_t        = w_t * corrected_EGARCH_t + (1 - w_t) * M1_HAR+TS_t")
print("  (c,k) fit by grid search (c in [0.10,0.90] step 0.05, k in {1..20} integer),")
print("  minimizing in-sample QLIKE, refit every 21 trading days, TRAINING window only.")
params_df = pd.DataFrame(m9_params)
print(f"  {len(params_df)} refits over the test window; k hit the cap (20) in "
      f"{int((params_df['k'] >= 20).sum())}/{len(params_df)} refits")
params_df.to_csv(os.path.join(RESULTS_DIR, "taskA_m9_params_history.csv"), index=False)

# ---------------------------------------------------------------------------
# Reconstruct per-day w9 from the logged (c,k) history + rv1_pctile, for the
# audit CSV (walk_forward computes w9 internally but doesn't return it).
# ---------------------------------------------------------------------------
p_series = data.loc[test_dates, "rv1_pctile"].values
refit_dates = pd.to_datetime(params_df["refit_date"]).values
active_c = np.empty(n)
active_k = np.empty(n)
ci = -1
for idx, td in enumerate(pd.to_datetime(test_dates)):
    while ci + 1 < len(refit_dates) and refit_dates[ci + 1] <= td:
        ci += 1
    active_c[idx] = params_df["c"].iloc[ci]
    active_k[idx] = params_df["k"].iloc[ci]
w9 = np.where(np.isfinite(p_series), 1.0 / (1.0 + np.exp(-active_k * (p_series - active_c))), 0.5)

m9_check = np.maximum(w9 * np.maximum(m2b, 1e-4) + (1 - w9) * m1, 1e-4)
max_diff = float(np.max(np.abs(m9_check - m9)))
print(f"\n[SANITY] reconstructed M9 (from logged c,k + rv1_pctile) matches walk_forward's M9 to within {max_diff:.2e}")

# ---------------------------------------------------------------------------
# Metrics: RMSE/MAE/QLIKE overall/crisis/calm
# ---------------------------------------------------------------------------
def metrics(a, f):
    return dict(RMSE=float(np.sqrt(np.mean((a - f) ** 2))), MAE=float(np.mean(np.abs(a - f))), QLIKE=qlike(a, f))

model_arrays = {"M9 Regime": m9, "M3 Combined": m3, "M1 HAR+TS": m1, "M2b Corrected EGARCH": m2b}
rows = []
for label, mask in [("overall", np.ones(n, dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for name, arr in model_arrays.items():
        m = metrics(actuals[mask], arr[mask])
        rows.append({"subsample": label, "model": name, "N": int(mask.sum()), **m})
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(os.path.join(RESULTS_DIR, "taskA_m9_metrics.csv"), index=False)
print(f"\n{'subsample':<10}{'model':<24}{'N':>5}{'RMSE':>10}{'MAE':>10}{'QLIKE':>10}")
for _, r in metrics_df.iterrows():
    print(f"{r['subsample']:<10}{r['model']:<24}{r['N']:>5}{r['RMSE']:>10.4f}{r['MAE']:>10.4f}{r['QLIKE']:>10.4f}")

# ---------------------------------------------------------------------------
# Mincer-Zarnowitz for M9
# ---------------------------------------------------------------------------
mz9 = mincer_zarnowitz(actuals, m9)
print(f"\nM9 Mincer-Zarnowitz: alpha={mz9['alpha']:.4f} (se={mz9['alpha_se']:.4f})  "
      f"beta={mz9['beta']:.4f} (se={mz9['beta_se']:.4f})  R2={mz9['R2']:.4f}  "
      f"F={mz9['F_stat']:.4f}  p={mz9['F_pval']:.4f}")
pd.DataFrame([mz9]).to_csv(os.path.join(RESULTS_DIR, "taskA_m9_mz.csv"), index=False)

# ---------------------------------------------------------------------------
# DM tests: M9 vs M3 (static combination), M9 vs M1 (baseline), MSE + QLIKE,
# overall/crisis/calm. Positive stat => M9 beats the comparison model.
# ---------------------------------------------------------------------------
dm_rows = []
for label, mask in [("overall", np.ones(n, dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for comp_name, comp in [("M3 Combined", m3), ("M1 HAR+TS", m1)]:
        for loss in ("mse", "qlike"):
            stat, pval = dm_test(actuals[mask], comp[mask], m9[mask], loss=loss)
            dm_rows.append({"subsample": label, "vs": comp_name, "loss": loss,
                             "N": int(mask.sum()), "dm_stat": stat, "p_value": pval})
dm_df = pd.DataFrame(dm_rows)
dm_df.to_csv(os.path.join(RESULTS_DIR, "taskA_m9_dm_tests.csv"), index=False)
print("\nDM tests: M9 vs {M3 Combined, M1 HAR+TS}  (positive stat => M9 better)")
print(dm_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Save all per-day forecasts, weights, and losses
# ---------------------------------------------------------------------------
out = pd.DataFrame({
    "date": test_dates, "actual": actuals, "crisis": crisis_mask,
    "M1_HAR_TS": m1, "M2_EGARCH_raw": m2, "M2b_Corrected_EGARCH": m2b,
    "M3_Combined": m3, "M9_Regime": m9,
    "M9_weight_w": w9, "M9_active_c": active_c, "M9_active_k": active_k,
})
for name, arr in [("M1_HAR_TS", m1), ("M2b_Corrected_EGARCH", m2b), ("M3_Combined", m3), ("M9_Regime", m9)]:
    out[f"{name}_sq_err"] = (actuals - arr) ** 2
    out[f"{name}_abs_err"] = np.abs(actuals - arr)
out.to_csv(os.path.join(RESULTS_DIR, "taskA_m9_forecasts_weights_losses.csv"), index=False)

print(f"\nSaved: {RESULTS_DIR}/taskA_m9_params_history.csv")
print(f"Saved: {RESULTS_DIR}/taskA_m9_metrics.csv")
print(f"Saved: {RESULTS_DIR}/taskA_m9_mz.csv")
print(f"Saved: {RESULTS_DIR}/taskA_m9_dm_tests.csv")
print(f"Saved: {RESULTS_DIR}/taskA_m9_forecasts_weights_losses.csv")
