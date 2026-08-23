"""
STEP 1 diagnostic: does the crisis advantage of raw EGARCH over M1 HAR+TS
(established in Task A: DM=+2.797, p=0.0081, N=38, QLIKE) survive the MZ bias
correction (M2 -> M2b)? No new models are built here -- pure diagnostic.

(a)/(b)/(c): HLN-corrected DM tests, crisis subsample (test period), QLIKE+MSE:
  (a) M2  (raw EGARCH)       vs M1 HAR+TS   -- re-confirm Task A's baseline
  (b) M2b (corrected EGARCH) vs M1 HAR+TS
  (c) M2b (corrected EGARCH) vs M2 (raw EGARCH)

Plus: crisis-subsample RMSE/QLIKE for M2 vs M2b side by side, and the MZ
correction's beta estimated SEPARATELY on crisis-only vs calm-only TRAINING
days (with SEs and day counts), to test whether the correction is dominated
by calm-day data.

Regime split for the TRAINING-day analysis: this must use the same ex-ante
rule M2c/M9c would apply prospectively to test days, i.e. rv1_pctile (already
computed ex-ante through t-1, see compute_features) thresholded at 0.90 --
the same "90th percentile" convention used everywhere else in this project
(rv1_pctile is already itself a 0-1 percentile rank, so "90th percentile of
rv1_pctile" = rv1_pctile >= 0.90). This is a judgment call since the prompt
doesn't pin a numeric threshold -- flagged here rather than decided silently.
Note this is a DIFFERENT quantity from the evaluation crisis mask used
elsewhere (actual RV > 90th pct of training RV): that mask looks at day t's
OWN realized vol (fine for scoring a forecast after the fact, since it's not
used to produce the forecast), whereas rv1_pctile only ever uses data through
t-1, which is required here because it's the same rule a regime-conditional
correction would need to apply prospectively.

The TRAINING window used for this analysis is the initial training set
(data before the test window starts, N=3304 -- identical to the training set
already used project-wide for the crisis-threshold computation). EGARCH is
fit ONCE on this window to get its full in-sample conditional volatility
(this exactly reproduces walk_forward()'s first refit, i=0, since hist at
i=0 IS data[data.index < test_start]) -- not re-derived from any later,
larger refit window.

No look-ahead: MZ regressions use only training-window (pre-test) data;
rv1_pctile is ex-ante by construction; crisis subsample for the DM tests is
the test-period evaluation mask already established in Task 1/4/A.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from arch import arch_model
from extended_model_zoo import fetch_data, compute_features, walk_forward, qlike, dm_test, mincer_zarnowitz

RESULTS_DIR = "research/results"
REGIME_PCTILE_THRESHOLD = 0.90  # rv1_pctile >= this => "crisis" training day (see docstring)

# ---------------------------------------------------------------------------
# Running DM-test tally across the project (for the multiplicity caveat)
# ---------------------------------------------------------------------------
DM_TEST_LOG = [
    ("Task 1 (task1_dm_significance.py): corrected-M3 vs M1, 3 subsamples x 2 losses", 6),
    ("Task 4 (task4_m9_regime_weighted.py): M9 vs M3, 3 subsamples x 2 losses", 6),
    ("Task A (taskA_m9_regime_eval.py): motivation check (1) + M9 vs {M3,M1}, 3 subsamples x 2 comparisons x 2 losses (12)", 13),
]

df = fetch_data()
data = compute_features(df)

actuals, results, test_dates = walk_forward(data)
actuals = np.asarray(actuals)
n = len(actuals)
m1 = np.asarray(results["M1 HAR+TS"])
m2 = np.asarray(results["M2 EGARCH"])
m2b = np.asarray(results["M2b Corrected EGARCH"])

test_start = test_dates[0]
train_rv = data.loc[data.index < test_start, "rv"]
crisis_threshold = float(np.percentile(train_rv, 90))
crisis_mask = actuals > crisis_threshold
print(f"\nN={n}  {test_dates[0].date()} -> {test_dates[-1].date()}")
print(f"Crisis threshold (90th pct training RV, N={len(train_rv)}): {crisis_threshold:.4f}")
print(f"Crisis days: {int(crisis_mask.sum())}  Calm days: {int((~crisis_mask).sum())}")

# ---------------------------------------------------------------------------
# (a)/(b)/(c) DM tests, crisis subsample, QLIKE + MSE
# ---------------------------------------------------------------------------
dm_rows = []
comparisons = [
    ("(a) M2 (raw) vs M1", m1, m2),
    ("(b) M2b (corrected) vs M1", m1, m2b),
    ("(c) M2b (corrected) vs M2 (raw)", m2, m2b),
]
print("\n" + "=" * 100)
print("  (a)/(b)/(c) DM TESTS -- crisis subsample only (positive stat => 2nd-named model better)")
print("=" * 100)
for label, ref, target in comparisons:
    for loss in ("qlike", "mse"):
        stat, pval = dm_test(actuals[crisis_mask], ref[crisis_mask], target[crisis_mask], loss=loss)
        dm_rows.append({"comparison": label, "loss": loss, "N": int(crisis_mask.sum()), "dm_stat": stat, "p_value": pval})
        print(f"  {label:<35} {loss:>6}  N={int(crisis_mask.sum())}  DM={stat:+.4f}  p={pval:.4f}")
dm_df = pd.DataFrame(dm_rows)
dm_df.to_csv(os.path.join(RESULTS_DIR, "stepB1_dm_tests_crisis.csv"), index=False)
DM_TEST_LOG.append(("Step 1 diagnostic (stepB1_diagnostic_regime_mz.py): (a)/(b)/(c) x 2 losses, crisis only", len(dm_rows)))

# ---------------------------------------------------------------------------
# crisis-subsample RMSE/QLIKE for M2 vs M2b, side by side
# ---------------------------------------------------------------------------
def metrics(a, f):
    return dict(RMSE=float(np.sqrt(np.mean((a - f) ** 2))), MAE=float(np.mean(np.abs(a - f))), QLIKE=qlike(a, f))

m2_crisis = metrics(actuals[crisis_mask], m2[crisis_mask])
m2b_crisis = metrics(actuals[crisis_mask], m2b[crisis_mask])
print("\n" + "=" * 100)
print("  CRISIS-SUBSAMPLE METRICS -- M2 (raw) vs M2b (corrected)")
print("=" * 100)
print(f"  {'model':<10}{'N':>5}{'RMSE':>10}{'MAE':>10}{'QLIKE':>10}")
print(f"  {'M2 raw':<10}{int(crisis_mask.sum()):>5}{m2_crisis['RMSE']:>10.4f}{m2_crisis['MAE']:>10.4f}{m2_crisis['QLIKE']:>10.4f}")
print(f"  {'M2b corr':<10}{int(crisis_mask.sum()):>5}{m2b_crisis['RMSE']:>10.4f}{m2b_crisis['MAE']:>10.4f}{m2b_crisis['QLIKE']:>10.4f}")
pd.DataFrame([
    {"model": "M2 raw", "N": int(crisis_mask.sum()), **m2_crisis},
    {"model": "M2b corrected", "N": int(crisis_mask.sum()), **m2b_crisis},
]).to_csv(os.path.join(RESULTS_DIR, "stepB1_crisis_metrics_m2_vs_m2b.csv"), index=False)

# ---------------------------------------------------------------------------
# MZ beta: crisis-only vs calm-only TRAINING days (regime = rv1_pctile >= 0.90)
# EGARCH fit ONCE on the initial training window (data before test_start) --
# this reproduces walk_forward()'s first refit (i=0) exactly.
# ---------------------------------------------------------------------------
hist0 = data[data.index < test_start]
am = arch_model(hist0["return"], vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
eg_res = am.fit(disp="off", show_warning=False)
eg_cv0 = eg_res.conditional_volatility.values
n0 = min(len(eg_cv0), len(hist0))
eg_cv0 = eg_cv0[-n0:]
rv0 = hist0["rv"].values[-n0:]
pctile0 = hist0["rv1_pctile"].values[-n0:]

train_crisis_mask = pctile0 >= REGIME_PCTILE_THRESHOLD
train_calm_mask = ~train_crisis_mask
n_train_crisis = int(train_crisis_mask.sum())
n_train_calm = int(train_calm_mask.sum())

print("\n" + "=" * 100)
print(f"  MZ REGRESSION SPLIT BY TRAINING-DAY REGIME (rv1_pctile >= {REGIME_PCTILE_THRESHOLD}, N_train={n0})")
print("=" * 100)
print(f"  Crisis training days: {n_train_crisis}  ({n_train_crisis/n0:.1%})")
print(f"  Calm training days:   {n_train_calm}  ({n_train_calm/n0:.1%})")

mz_pooled = mincer_zarnowitz(rv0, eg_cv0)
print(f"\n  POOLED   (N={n0}):    alpha={mz_pooled['alpha']:+.4f} (se={mz_pooled['alpha_se']:.4f})   "
      f"beta={mz_pooled['beta']:.4f} (se={mz_pooled['beta_se']:.4f})")

MIN_STABLE_N = 30
mz_rows = [{"regime": "pooled", "N": n0, **mz_pooled}]

if n_train_crisis >= MIN_STABLE_N:
    mz_crisis_train = mincer_zarnowitz(rv0[train_crisis_mask], eg_cv0[train_crisis_mask])
    print(f"  CRISIS   (N={n_train_crisis}):    alpha={mz_crisis_train['alpha']:+.4f} (se={mz_crisis_train['alpha_se']:.4f})   "
          f"beta={mz_crisis_train['beta']:.4f} (se={mz_crisis_train['beta_se']:.4f})")
    mz_rows.append({"regime": "crisis_train", "N": n_train_crisis, **mz_crisis_train})
else:
    print(f"  CRISIS   (N={n_train_crisis}): TOO FEW for a stable fit (< {MIN_STABLE_N}) -- not fit.")
    mz_crisis_train = None

if n_train_calm >= MIN_STABLE_N:
    mz_calm_train = mincer_zarnowitz(rv0[train_calm_mask], eg_cv0[train_calm_mask])
    print(f"  CALM     (N={n_train_calm}):    alpha={mz_calm_train['alpha']:+.4f} (se={mz_calm_train['alpha_se']:.4f})   "
          f"beta={mz_calm_train['beta']:.4f} (se={mz_calm_train['beta_se']:.4f})")
    mz_rows.append({"regime": "calm_train", "N": n_train_calm, **mz_calm_train})
else:
    print(f"  CALM     (N={n_train_calm}): TOO FEW for a stable fit (< {MIN_STABLE_N}) -- not fit.")
    mz_calm_train = None

pd.DataFrame(mz_rows).to_csv(os.path.join(RESULTS_DIR, "stepB1_mz_by_regime.csv"), index=False)

if mz_crisis_train is not None and mz_calm_train is not None:
    beta_diff = mz_crisis_train["beta"] - mz_calm_train["beta"]
    se_diff = float(np.sqrt(mz_crisis_train["beta_se"]**2 + mz_calm_train["beta_se"]**2))
    t_diff = beta_diff / se_diff if se_diff > 0 else np.nan
    print(f"\n  beta_crisis - beta_calm = {beta_diff:+.4f}  (approx SE={se_diff:.4f}, t~{t_diff:+.2f})")
else:
    beta_diff, se_diff, t_diff = np.nan, np.nan, np.nan

# ---------------------------------------------------------------------------
# Verdict on the hypothesis (reported plainly, no action taken)
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("  HYPOTHESIS CHECK")
print("=" * 100)
print("  H: MZ correction is fit predominantly on calm days and scales down EGARCH's")
print("     over-reaction, which is the source of its crisis edge -- so M2b should lose")
print("     the crisis advantage M2 has over M1, and beta_crisis should differ materially")
print("     from beta_calm.")
b_qlike = dm_df[(dm_df["comparison"] == "(a) M2 (raw) vs M1") & (dm_df["loss"] == "qlike")].iloc[0]
c_qlike = dm_df[(dm_df["comparison"] == "(b) M2b (corrected) vs M1") & (dm_df["loss"] == "qlike")].iloc[0]
d_qlike = dm_df[(dm_df["comparison"] == "(c) M2b (corrected) vs M2 (raw)") & (dm_df["loss"] == "qlike")].iloc[0]
print(f"\n  (a) raw EGARCH vs M1, crisis, QLIKE:        DM={b_qlike['dm_stat']:+.4f} p={b_qlike['p_value']:.4f}")
print(f"  (b) corrected EGARCH vs M1, crisis, QLIKE:  DM={c_qlike['dm_stat']:+.4f} p={c_qlike['p_value']:.4f}")
print(f"  (c) corrected vs raw EGARCH, crisis, QLIKE: DM={d_qlike['dm_stat']:+.4f} p={d_qlike['p_value']:.4f}")
print(f"  Training-day fraction crisis: {n_train_crisis/n0:.1%}  ({n_train_crisis}/{n0})")
if not np.isnan(t_diff):
    print(f"  beta_crisis vs beta_calm difference: {beta_diff:+.4f} (t~{t_diff:+.2f})")

print(f"\nSaved: {RESULTS_DIR}/stepB1_dm_tests_crisis.csv")
print(f"Saved: {RESULTS_DIR}/stepB1_crisis_metrics_m2_vs_m2b.csv")
print(f"Saved: {RESULTS_DIR}/stepB1_mz_by_regime.csv")

total_dm = sum(c for _, c in DM_TEST_LOG)
print("\n" + "=" * 100)
print(f"  DM-TEST TALLY ACROSS THE PROJECT SO FAR: {total_dm}")
print("=" * 100)
for label, count in DM_TEST_LOG:
    print(f"  {count:>3}  {label}")
