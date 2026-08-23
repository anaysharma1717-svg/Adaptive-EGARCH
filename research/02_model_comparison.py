"""
STAGE 2 -- The 8-model comparison (+ M2/M2b/M3).

WHAT THIS SCRIPT ACTUALLY DOES: recomputes metrics/DM/MZ from saved
forecasts (does not re-fit models). It loads research/results/
baseline_confirm_forecasts.csv -- already-generated per-day forecasts -- and
derives RMSE/MAE/QLIKE/DM/MZ from that fixed data. The models themselves
were fit by research/pipeline/extended_model_zoo.py, not here.

Question this stage answers: of nine reasonable ways to forecast next-day
volatility, which one is best, and by how much?

This stage does NOT re-run the walk-forward (that requires refitting EGARCH
by simulation, which is expensive and reintroduces small run-to-run Monte
Carlo noise). Instead it loads the per-day forecasts already saved by the
original walk-forward run (baseline_confirm_forecasts.csv) and recomputes
every summary statistic FROM that fixed, already-saved data -- so the numbers
below are guaranteed to exactly match what was originally reported, not an
approximate re-run.

Models compared (see extended_model_zoo.py for the exact fit/predict code):
  M1 HAR+TS    - OLS on (rv1, rv2, rv5, rv22)                    [baseline]
  M2 EGARCH    - standalone EGARCH(1,1,1,t) conditional volatility
  M2b Corrected EGARCH - M2 rescaled by a Mincer-Zarnowitz regression (see
                 stage 4 for why this "fix" turns out to be more complicated
                 than it looks)
  M3 Combined  - HAR features + M2b as a 5th regressor            [best overall]
  M4 Log-HAR   - OLS on log(RV); needed a bug fix, see stage 4's sibling note
                 in the project report (Jensen's-inequality retransformation)
  M5 Asym-HAR  - HAR + a negative-semi-variance term (captures the leverage
                 effect -- vol rising more after down days -- without EGARCH)
  M6 HAR+Ret   - HAR + lagged return + lagged |return|
  M7 Roll-HAR  - HAR+TS but fit on a fixed 1000-day ROLLING window instead
                 of an expanding one (tests whether "forgetting" old data helps)
  M8 KitchSink - all of the above features at once, fit with Ridge (L2-
                 regularized OLS, to control overfitting with many features)
"""
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))
from extended_model_zoo import qlike, dm_test, mincer_zarnowitz

SRC = os.path.join("research", "results", "baseline_confirm_forecasts.csv")
OUT_DIR = os.path.join("research", "results", "02_model_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 90)
print("  STAGE 2: 8-MODEL COMPARISON (reproducing baseline_confirm_forecasts.csv)")
print("=" * 90)

df = pd.read_csv(SRC, parse_dates=["date"])
actual = df["actual"].values
model_cols = [c for c in df.columns if c not in ("date", "actual")]
print(f"\nLoaded {len(df)} test days, {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"Models: {model_cols}")

rows = []
for name in model_cols:
    fc = df[name].values
    rmse = float(np.sqrt(np.mean((actual - fc) ** 2)))
    mae = float(np.mean(np.abs(actual - fc)))
    ql = qlike(actual, fc)
    rows.append({"model": name, "RMSE": rmse, "MAE": mae, "QLIKE": ql})
metrics_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
metrics_df["rank_by_RMSE"] = metrics_df.index + 1
metrics_df.to_csv(os.path.join(OUT_DIR, "model_comparison_metrics.csv"), index=False)

print(f"\n{'model':<24}{'RMSE':>10}{'MAE':>10}{'QLIKE':>10}{'rank':>6}")
for _, r in metrics_df.iterrows():
    marker = "  <-- best RMSE" if r["rank_by_RMSE"] == 1 else ""
    print(f"{r['model']:<24}{r['RMSE']:>10.4f}{r['MAE']:>10.4f}{r['QLIKE']:>10.4f}{int(r['rank_by_RMSE']):>6}{marker}")

# --- DM tests: every model vs the M1 HAR+TS baseline ---
# HLN-corrected Diebold-Mariano: is the difference in average loss between
# two forecasts big relative to how noisy that difference itself is (given
# both models are often wrong on the SAME days)? Positive stat = the
# challenger has LOWER loss than M1 (better).
m1 = df["M1 HAR+TS"].values
dm_rows = []
for name in model_cols:
    if name == "M1 HAR+TS":
        continue
    for loss in ("mse", "qlike"):
        stat, pval = dm_test(actual, m1, df[name].values, loss=loss)
        dm_rows.append({"vs_M1": name, "loss": loss, "N": len(actual), "dm_stat": stat, "p_value": pval})
dm_df = pd.DataFrame(dm_rows)
dm_df.to_csv(os.path.join(OUT_DIR, "dm_tests_vs_m1.csv"), index=False)
print(f"\nDM tests vs M1 HAR+TS (positive stat => challenger better):")
print(dm_df.to_string(index=False))

# --- Mincer-Zarnowitz efficiency for every model ---
# actual = alpha + beta*forecast + eps. An efficient forecast has alpha=0,
# beta=1 -- tested jointly with an F-test (not two separate t-tests, since
# alpha and beta's estimation errors are correlated).
mz_rows = []
for name in model_cols:
    mz = mincer_zarnowitz(actual, df[name].values)
    mz_rows.append({"model": name, **mz})
mz_df = pd.DataFrame(mz_rows)
mz_df.to_csv(os.path.join(OUT_DIR, "mincer_zarnowitz_all_models.csv"), index=False)
print(f"\nMincer-Zarnowitz efficiency (alpha=0, beta=1 jointly tested):")
print(mz_df[["model", "alpha", "beta", "R2", "F_stat", "F_pval"]].to_string(index=False))

print(f"\nSaved: {OUT_DIR}/model_comparison_metrics.csv")
print(f"Saved: {OUT_DIR}/dm_tests_vs_m1.csv")
print(f"Saved: {OUT_DIR}/mincer_zarnowitz_all_models.csv")
