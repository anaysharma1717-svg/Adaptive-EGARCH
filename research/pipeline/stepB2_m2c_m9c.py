"""
STEP 2: regime-conditional MZ correction (M2c) and M9c.

Confirmed in Step 1: the pooled MZ correction is calm-dominated (90.1% of
training days) and beta differs materially by regime (beta_crisis=0.830 vs
beta_calm=0.617, t~5.85), which erodes raw EGARCH's significant crisis QLIKE
edge over M1 (DM=+2.797,p=0.008 raw -> DM=+0.997,p=0.325 corrected).

This script is SELF-CONTAINED rather than calling extended_model_zoo's
walk_forward() a second time: M2, M2b, M2c, M3, M9, M9c all need to come from
the SAME per-refit EGARCH fit (same eg_cv, same eg_pred) at each iteration --
calling walk_forward() again to get M1/M2/M2b/M3/M9 and separately re-fitting
EGARCH for M2c's regime split would risk two independently-converged EGARCH
fits drifting apart numerically. The loop below reproduces walk_forward()'s
logic for M1/M2/M2b/M3/M9 EXACTLY (same refit schedule, same formulas, same
grid search) so those five are unchanged from what Task A/Step 1 already
validated, and adds M2c/M9c on top of the same fit object.

M2c: two MZ regressions per refit (alpha_calm,beta_calm) and
(alpha_crisis,beta_crisis), fit on the SAME expanding training window as
M2b's pooled correction, split by the same ex-ante regime rule M9 already
uses (rv1_pctile >= 0.90, i.e. rv1's own trailing-252d percentile rank --
never touches day t's realized vol). Applied to day t using day t's own
ex-ante regime classification. If a regime has < MIN_STABLE_N=30 training
days at a given refit, that refit's split is not fit -- the pooled
correction is used as a fallback for that regime at that refit only, and
this is logged and reported (expected not to trigger here: Step 1 found
already 326 crisis training days at the very first refit).

M9c: identical grid-search mechanics to M9 (c in [0.10,0.90] step 0.05, k in
{1..20}, minimize in-sample QLIKE, training-only, every 21-day refit) but
run separately against M2c's in-sample regime-corrected series instead of
M2b's pooled one. Not a retune of M9 -- a fresh application of the same
unmodified procedure to a different input series, which is expected to
produce a different (c,k) than M9's own fit.

No look-ahead: hist = data[data.index < tdate] throughout; rv1_pctile is
ex-ante by construction; every correction and every grid search uses only
data strictly before the day it's applied to.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.linear_model import LinearRegression
from extended_model_zoo import fetch_data, compute_features, MODELS, qlike, dm_test, mincer_zarnowitz

RESULTS_DIR = "research/results"
REGIME_THRESHOLD = 0.90
MIN_STABLE_N = 30
REFIT_FREQ = 21
MIN_TRAIN = 500
TEST_MONTHS = 18
SIM_PATHS = 500
M9_K_CAP = 20.0

df = fetch_data()
data = compute_features(df)

split_date = data.index[-1] - pd.DateOffset(months=TEST_MONTHS)
test_data = data[data.index > split_date]
test_dates = test_data.index
print(f"\nTest: {test_dates[0].date()} -> {test_dates[-1].date()} ({len(test_dates)} obs)")

har_spec = MODELS["M1 HAR+TS"]  # fresh fit each refit below, exactly like walk_forward()

results = {k: [] for k in ["M1", "M2_raw", "M2b_pooled", "M2c_regime", "M3", "M9", "M9c",
                            "regime_exante", "w9", "w9c", "m9_c", "m9_k", "m9c_c", "m9c_k"]}
actuals = []

eg_params = None
comb_coef = None
eg_alpha, eg_beta = 0.0, 1.0
alpha_crisis, beta_crisis = 0.0, 1.0
alpha_calm, beta_calm = 0.0, 1.0
m9_c, m9_k = 0.5, 1.0
m9c_c, m9c_k = 0.5, 1.0

refit_audit = []

for i, tdate in enumerate(test_dates):
    hist = data[data.index < tdate]
    if len(hist) < MIN_TRAIN:
        continue

    do_refit = (i % REFIT_FREQ == 0) or (eg_params is None)

    if do_refit:
        print(f"Refit at {tdate.date()} (step {i}/{len(test_dates)})...")
        har_spec.fit(hist)

        am = arch_model(hist["return"], vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
        eg_res = am.fit(starting_values=eg_params, disp="off", show_warning=False)
        eg_params = eg_res.params.values
        eg_cv = eg_res.conditional_volatility.values
        n_a = min(len(eg_cv), len(hist))
        eg_cv = eg_cv[-n_a:]
        rv_train = hist["rv"].values[-n_a:]
        pctile_train = hist["rv1_pctile"].values[-n_a:]

        # ---- pooled MZ (M2b, unchanged) ----
        mz_pooled = mincer_zarnowitz(rv_train, eg_cv)
        eg_alpha, eg_beta = mz_pooled["alpha"], mz_pooled["beta"]
        eg_cv_corr_pooled = eg_alpha + eg_beta * eg_cv

        # ---- regime split (M2c, new) ----
        crisis_mask_t = pctile_train >= REGIME_THRESHOLD
        calm_mask_t = ~crisis_mask_t
        n_crisis_t, n_calm_t = int(crisis_mask_t.sum()), int(calm_mask_t.sum())

        if n_crisis_t >= MIN_STABLE_N:
            mz_c = mincer_zarnowitz(rv_train[crisis_mask_t], eg_cv[crisis_mask_t])
            alpha_crisis, beta_crisis = mz_c["alpha"], mz_c["beta"]
            crisis_stable = True
        else:
            alpha_crisis, beta_crisis = eg_alpha, eg_beta  # pooled fallback, logged
            crisis_stable = False

        if n_calm_t >= MIN_STABLE_N:
            mz_k_ = mincer_zarnowitz(rv_train[calm_mask_t], eg_cv[calm_mask_t])
            alpha_calm, beta_calm = mz_k_["alpha"], mz_k_["beta"]
            calm_stable = True
        else:
            alpha_calm, beta_calm = eg_alpha, eg_beta
            calm_stable = False

        eg_cv_corr_regime = np.where(crisis_mask_t, alpha_crisis + beta_crisis * eg_cv,
                                      alpha_calm + beta_calm * eg_cv)

        # ---- M3 (unchanged, pooled-based) ----
        X_comb = np.column_stack([hist[["rv1", "rv2", "rv5", "rv22"]].values[-n_a:], eg_cv_corr_pooled])
        y_comb = rv_train
        comb_mod = LinearRegression().fit(X_comb, y_comb)
        comb_coef = (comb_mod.intercept_, comb_mod.coef_)

        # ---- M9 grid search (unchanged, pooled-based) ----
        X_har_hist = hist[har_spec.features].values[-n_a:]
        har_fitted = np.maximum(har_spec.coef[0] + X_har_hist @ har_spec.coef[1], 1e-4)
        eg_cv_corr_pooled_floor = np.maximum(eg_cv_corr_pooled, 1e-4)
        eg_cv_corr_regime_floor = np.maximum(eg_cv_corr_regime, 1e-4)
        p_hist = pctile_train
        y9 = rv_train
        valid9 = np.isfinite(p_hist)

        best_c, best_k, best_loss = 0.5, 1.0, np.inf
        for c_try in np.arange(0.10, 0.901, 0.05):
            for k_try in range(1, int(M9_K_CAP) + 1):
                w_try = 1.0 / (1.0 + np.exp(-k_try * (p_hist[valid9] - c_try)))
                pred9_try = np.maximum(w_try * eg_cv_corr_pooled_floor[valid9] + (1 - w_try) * har_fitted[valid9], 1e-4)
                loss_try = qlike(y9[valid9], pred9_try)
                if loss_try < best_loss:
                    best_loss, best_c, best_k = loss_try, float(c_try), float(k_try)
        m9_c, m9_k = best_c, min(best_k, M9_K_CAP)

        # ---- M9c grid search (new, identical mechanics, M2c-based) ----
        best_c9c, best_k9c, best_loss9c = 0.5, 1.0, np.inf
        for c_try in np.arange(0.10, 0.901, 0.05):
            for k_try in range(1, int(M9_K_CAP) + 1):
                w_try = 1.0 / (1.0 + np.exp(-k_try * (p_hist[valid9] - c_try)))
                pred9c_try = np.maximum(w_try * eg_cv_corr_regime_floor[valid9] + (1 - w_try) * har_fitted[valid9], 1e-4)
                loss_try = qlike(y9[valid9], pred9c_try)
                if loss_try < best_loss9c:
                    best_loss9c, best_c9c, best_k9c = loss_try, float(c_try), float(k_try)
        m9c_c, m9c_k = best_c9c, min(best_k9c, M9_K_CAP)

        refit_audit.append({
            "refit_date": tdate, "n_train": n_a,
            "n_crisis_train": n_crisis_t, "n_calm_train": n_calm_t,
            "crisis_stable": crisis_stable, "calm_stable": calm_stable,
            "alpha_crisis": alpha_crisis, "beta_crisis": beta_crisis,
            "alpha_calm": alpha_calm, "beta_calm": beta_calm,
            "eg_alpha_pooled": eg_alpha, "eg_beta_pooled": eg_beta,
            "m9_c": m9_c, "m9_k": m9_k, "m9c_c": m9c_c, "m9c_k": m9c_k,
        })

    # ---- per-day forecasts ----
    am_fix = arch_model(hist["return"], vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
    eg_fixed = am_fix.fix(eg_params)
    try:
        fc = eg_fixed.forecast(horizon=1, method="simulation", simulations=SIM_PATHS, reindex=False)
        eg_pred = float(np.sqrt(fc.variance.iloc[-1, 0]))
    except Exception:
        eg_pred = float(eg_fixed.conditional_volatility.iloc[-1])
    eg_pred = max(eg_pred, 1e-4)

    row = data.loc[tdate]
    m1_pred = har_spec.predict(row)
    m2_raw = eg_pred
    m2b_pred = max(eg_alpha + eg_beta * eg_pred, 1e-4)

    p_t = row["rv1_pctile"]
    is_crisis_day = bool(np.isfinite(p_t) and p_t >= REGIME_THRESHOLD)
    if is_crisis_day:
        m2c_pred = max(alpha_crisis + beta_crisis * eg_pred, 1e-4)
    else:
        m2c_pred = max(alpha_calm + beta_calm * eg_pred, 1e-4)

    x_comb = np.array([row["rv1"], row["rv2"], row["rv5"], row["rv22"], m2b_pred])
    m3_pred = max(float(comb_coef[0] + np.dot(comb_coef[1], x_comb)), 1e-4)

    if np.isfinite(p_t):
        w9 = 1.0 / (1.0 + np.exp(-m9_k * (float(p_t) - m9_c)))
        w9c = 1.0 / (1.0 + np.exp(-m9c_k * (float(p_t) - m9c_c)))
    else:
        w9 = w9c = 0.5
    m9_pred = max(w9 * m2b_pred + (1 - w9) * m1_pred, 1e-4)
    m9c_pred = max(w9c * m2c_pred + (1 - w9c) * m1_pred, 1e-4)

    results["M1"].append(m1_pred)
    results["M2_raw"].append(m2_raw)
    results["M2b_pooled"].append(m2b_pred)
    results["M2c_regime"].append(m2c_pred)
    results["M3"].append(m3_pred)
    results["M9"].append(m9_pred)
    results["M9c"].append(m9c_pred)
    results["regime_exante"].append("crisis" if is_crisis_day else "calm")
    results["w9"].append(w9)
    results["w9c"].append(w9c)
    results["m9_c"].append(m9_c); results["m9_k"].append(m9_k)
    results["m9c_c"].append(m9c_c); results["m9c_k"].append(m9c_k)
    actuals.append(float(row["rv"]))

actuals = np.array(actuals)
n = len(actuals)
test_dates = test_dates[-n:]
for k in results:
    if k != "regime_exante":
        results[k] = np.array(results[k])

refit_df = pd.DataFrame(refit_audit)
refit_df.to_csv(os.path.join(RESULTS_DIR, "stepB2_refit_audit.csv"), index=False)
print("\n" + "=" * 100)
print("  PER-REFIT CRISIS/CALM TRAINING DAY COUNTS (regime = rv1_pctile >= 0.90)")
print("=" * 100)
print(refit_df[["refit_date", "n_train", "n_crisis_train", "n_calm_train", "crisis_stable", "calm_stable",
                "beta_crisis", "beta_calm"]].to_string(index=False))
if not refit_df["crisis_stable"].all():
    n_unstable = int((~refit_df["crisis_stable"]).sum())
    print(f"\n  {n_unstable} refit(s) had < {MIN_STABLE_N} crisis training days -- pooled correction used as fallback there.")
else:
    print(f"\n  All {len(refit_df)} refits had >= {MIN_STABLE_N} crisis training days -- no fallback needed.")
print(f"  beta_crisis range across refits: {refit_df['beta_crisis'].min():.4f} -> {refit_df['beta_crisis'].max():.4f}")
print(f"  beta_calm    range across refits: {refit_df['beta_calm'].min():.4f} -> {refit_df['beta_calm'].max():.4f}")

# --- evaluation crisis mask (same convention as Task 1/4/A: 90th pct of TRAINING actual RV) ---
test_start = test_dates[0]
train_rv = data.loc[data.index < test_start, "rv"]
crisis_threshold = float(np.percentile(train_rv, 90))
crisis_mask = actuals > crisis_threshold
calm_mask = ~crisis_mask
print(f"\nEvaluation crisis threshold (90th pct training RV, N={len(train_rv)}): {crisis_threshold:.4f}")
print(f"Crisis days: {int(crisis_mask.sum())}  Calm days: {int(calm_mask.sum())}")

m1, m2, m2b, m2c, m3, m9, m9c = (results[k] for k in ["M1", "M2_raw", "M2b_pooled", "M2c_regime", "M3", "M9", "M9c"])

# ---------------------------------------------------------------------------
# M2c alone: RMSE/MAE/QLIKE overall/crisis/calm + MZ
# ---------------------------------------------------------------------------
def metrics(a, f):
    return dict(RMSE=float(np.sqrt(np.mean((a - f) ** 2))), MAE=float(np.mean(np.abs(a - f))), QLIKE=qlike(a, f))

print("\n" + "=" * 100)
print("  M2c ALONE vs M2 (raw) vs M2b (pooled) -- RMSE/MAE/QLIKE")
print("=" * 100)
m2c_rows = []
for label, mask in [("overall", np.ones(n, dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for name, arr in [("M2 raw", m2), ("M2b pooled", m2b), ("M2c regime", m2c)]:
        met = metrics(actuals[mask], arr[mask])
        m2c_rows.append({"subsample": label, "model": name, "N": int(mask.sum()), **met})
m2c_df = pd.DataFrame(m2c_rows)
m2c_df.to_csv(os.path.join(RESULTS_DIR, "stepB2_m2c_metrics.csv"), index=False)
print(m2c_df.to_string(index=False))

mz_m2c = mincer_zarnowitz(actuals, m2c)
print(f"\nM2c Mincer-Zarnowitz (full test window): alpha={mz_m2c['alpha']:+.4f} (se={mz_m2c['alpha_se']:.4f})  "
      f"beta={mz_m2c['beta']:.4f} (se={mz_m2c['beta_se']:.4f})  R2={mz_m2c['R2']:.4f}  F={mz_m2c['F_stat']:.4f}  p={mz_m2c['F_pval']:.4f}")
pd.DataFrame([mz_m2c]).to_csv(os.path.join(RESULTS_DIR, "stepB2_m2c_mz.csv"), index=False)

# ---------------------------------------------------------------------------
# M2c vs raw M2, crisis subsample, DM (the "result I care about most" check)
# ---------------------------------------------------------------------------
dm_rows = []
print("\n" + "=" * 100)
print("  M2c vs raw M2 (M2), crisis subsample -- does M2c preserve the crisis edge?")
print("=" * 100)
for loss in ("qlike", "mse"):
    stat, pval = dm_test(actuals[crisis_mask], m2[crisis_mask], m2c[crisis_mask], loss=loss)
    dm_rows.append({"comparison": "M2c vs M2 (raw)", "subsample": "crisis", "loss": loss,
                     "N": int(crisis_mask.sum()), "dm_stat": stat, "p_value": pval})
    print(f"  {loss:>6}  N={int(crisis_mask.sum())}  DM={stat:+.4f}  p={pval:.4f}  "
          f"(negative & significant => M2c still worse than raw M2 in crisis)")
# (calm-side M2c vs M2 not requested as a DM test -- the M2c-alone metrics table
# above already covers M2c's calm performance without spending an extra test.)

# ---------------------------------------------------------------------------
# M9c evaluation: metrics + DM vs M3, M9, M1
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("  M9c -- RMSE/MAE/QLIKE overall/crisis/calm, vs M9/M3/M1")
print("=" * 100)
m9c_metric_rows = []
for label, mask in [("overall", np.ones(n, dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for name, arr in [("M9c", m9c), ("M9", m9), ("M3", m3), ("M1", m1)]:
        met = metrics(actuals[mask], arr[mask])
        m9c_metric_rows.append({"subsample": label, "model": name, "N": int(mask.sum()), **met})
m9c_metrics_df = pd.DataFrame(m9c_metric_rows)
m9c_metrics_df.to_csv(os.path.join(RESULTS_DIR, "stepB2_m9c_metrics.csv"), index=False)
print(m9c_metrics_df.to_string(index=False))

print("\nDM tests: M9c vs {M3, M9, M1}  (positive stat => M9c better)")
for label, mask in [("overall", np.ones(n, dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for comp_name, comp in [("M3", m3), ("M9", m9), ("M1", m1)]:
        for loss in ("mse", "qlike"):
            stat, pval = dm_test(actuals[mask], comp[mask], m9c[mask], loss=loss)
            dm_rows.append({"comparison": f"M9c vs {comp_name}", "subsample": label, "loss": loss,
                             "N": int(mask.sum()), "dm_stat": stat, "p_value": pval})
            print(f"  [{label:<8}] M9c vs {comp_name:<3} {loss:>6}  N={mask.sum():4d}  DM={stat:+.4f}  p={pval:.4f}")

dm_df = pd.DataFrame(dm_rows)
dm_df.to_csv(os.path.join(RESULTS_DIR, "stepB2_dm_tests.csv"), index=False)

# ---------------------------------------------------------------------------
# Save all per-day forecasts, weights, losses
# ---------------------------------------------------------------------------
out = pd.DataFrame({
    "date": test_dates, "actual": actuals, "crisis_eval": crisis_mask,
    "regime_exante": results["regime_exante"],
    "M1": m1, "M2_raw": m2, "M2b_pooled": m2b, "M2c_regime": m2c, "M3": m3, "M9": m9, "M9c": m9c,
    "w9": results["w9"], "w9c": results["w9c"],
    "m9_c_active": results["m9_c"], "m9_k_active": results["m9_k"],
    "m9c_c_active": results["m9c_c"], "m9c_k_active": results["m9c_k"],
})
for name, arr in [("M2c_regime", m2c), ("M9c", m9c), ("M9", m9), ("M3", m3), ("M1", m1)]:
    out[f"{name}_sq_err"] = (actuals - arr) ** 2
    out[f"{name}_abs_err"] = np.abs(actuals - arr)
out.to_csv(os.path.join(RESULTS_DIR, "stepB2_forecasts_weights_losses.csv"), index=False)

print(f"\nSaved: {RESULTS_DIR}/stepB2_refit_audit.csv")
print(f"Saved: {RESULTS_DIR}/stepB2_m2c_metrics.csv")
print(f"Saved: {RESULTS_DIR}/stepB2_m2c_mz.csv")
print(f"Saved: {RESULTS_DIR}/stepB2_m9c_metrics.csv")
print(f"Saved: {RESULTS_DIR}/stepB2_dm_tests.csv")
print(f"Saved: {RESULTS_DIR}/stepB2_forecasts_weights_losses.csv")

# ---------------------------------------------------------------------------
# Running DM-test tally
# ---------------------------------------------------------------------------
prior_total = 31  # Task 1 (6) + Task 4 (6) + Task A (13) + Step 1 (6)
this_step_count = len(dm_rows)
print("\n" + "=" * 100)
print(f"  DM-TEST TALLY: {prior_total} (through Step 1) + {this_step_count} (Step 2) = {prior_total + this_step_count}")
print("=" * 100)
