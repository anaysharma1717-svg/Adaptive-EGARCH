"""
STEP 3: QLIKE-fitted bias correction (M2d).

Step 2 found the mechanism: even a crisis-only OLS/MZ fit gives beta=0.83,
still well below 1 -- OLS shrinks toward the conditional mean, which is
exactly what QLIKE punishes when a big move actually happens. This script
tests the fix that follows directly from that diagnosis: fit alpha,beta by
numerically minimizing QLIKE on the training window instead of OLS.

Same affine form (corrected = alpha + beta*raw), same expanding-window
training, same ex-ante regime rule (rv1_pctile >= 0.90) and MIN_STABLE_N=30
threshold as Step 2 -- nothing retuned. Two variants:
  M2d-pooled  -- QLIKE-fit on all training days (like M2b, but QLIKE-fit)
  M2d-regime  -- QLIKE-fit separately on crisis/calm training days (like
                 M2c, but QLIKE-fit)

Self-contained (extends Step 2's loop) rather than re-importing walk_forward
a third time: M1/M2/M2b/M2c/M3/M9/M9c must come from the exact same per-refit
EGARCH fit as M2d/M9d for the requested M2d-regime vs M2c and M9d vs
M9c/M9/M3/M1 comparisons to be apples-to-apples. This duplicates Step 2's
loop body and layers the QLIKE-fit correction and M9d on top.

QLIKE-fit optimizer: scipy.optimize.minimize, Nelder-Mead (derivative-free --
the forecast floor at 1e-4 makes the objective non-smooth at that boundary,
so a gradient method risks getting stuck exactly there), initialized at the
OLS solution for that same window/regime (a reasonable, non-arbitrary warm
start). Objective: mean(r - log(r) - 1) where r = actual^2 / max(alpha +
beta*raw, 1e-4)^2. Convergence (res.success) logged and reported per refit,
per regime.

No look-ahead: everything fit on hist = data[data.index < tdate]; rv1_pctile
ex-ante; M9d's grid search training-only, same mechanics as M9/M9c, not
retuned.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
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


def qlike_fit(y, x, a0, b0):
    """Numerically minimize QLIKE over (alpha,beta) for corrected = alpha+beta*x,
    initialized at the OLS solution. Returns (alpha, beta, converged)."""
    def obj(p):
        a, b = p
        f = np.maximum(a + b * x, 1e-4)
        r = y ** 2 / f ** 2
        return float(np.mean(r - np.log(r) - 1))
    res = minimize(obj, x0=[a0, b0], method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000, "maxfev": 5000})
    return float(res.x[0]), float(res.x[1]), bool(res.success)


df = fetch_data()
data = compute_features(df)

split_date = data.index[-1] - pd.DateOffset(months=TEST_MONTHS)
test_data = data[data.index > split_date]
test_dates = test_data.index
print(f"\nTest: {test_dates[0].date()} -> {test_dates[-1].date()} ({len(test_dates)} obs)")

har_spec = MODELS["M1 HAR+TS"]

results = {k: [] for k in ["M1", "M2_raw", "M2b_pooled", "M2c_regime", "M2d_pooled", "M2d_regime",
                            "M3", "M9", "M9c", "M9d", "regime_exante", "w9", "w9c", "w9d"]}
actuals = []

eg_params = None
comb_coef = None
eg_alpha, eg_beta = 0.0, 1.0
alpha_crisis, beta_crisis = 0.0, 1.0
alpha_calm, beta_calm = 0.0, 1.0
alpha_d_pooled, beta_d_pooled = 0.0, 1.0
alpha_d_crisis, beta_d_crisis = 0.0, 1.0
alpha_d_calm, beta_d_calm = 0.0, 1.0
m9_c, m9_k = 0.5, 1.0
m9c_c, m9c_k = 0.5, 1.0
m9d_c, m9d_k = 0.5, 1.0

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

        # ---- pooled OLS (M2b) ----
        mz_pooled = mincer_zarnowitz(rv_train, eg_cv)
        eg_alpha, eg_beta = mz_pooled["alpha"], mz_pooled["beta"]
        eg_cv_corr_pooled = eg_alpha + eg_beta * eg_cv

        # ---- regime OLS split (M2c) ----
        crisis_mask_t = pctile_train >= REGIME_THRESHOLD
        calm_mask_t = ~crisis_mask_t
        n_crisis_t, n_calm_t = int(crisis_mask_t.sum()), int(calm_mask_t.sum())

        if n_crisis_t >= MIN_STABLE_N:
            mz_c = mincer_zarnowitz(rv_train[crisis_mask_t], eg_cv[crisis_mask_t])
            alpha_crisis, beta_crisis = mz_c["alpha"], mz_c["beta"]
            crisis_stable = True
        else:
            alpha_crisis, beta_crisis = eg_alpha, eg_beta
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

        # ---- NEW: QLIKE-fit pooled (M2d-pooled), init at OLS pooled solution ----
        alpha_d_pooled, beta_d_pooled, conv_d_pooled = qlike_fit(rv_train, eg_cv, eg_alpha, eg_beta)
        eg_cv_corr_d_pooled = np.maximum(alpha_d_pooled + beta_d_pooled * eg_cv, 1e-4)

        # ---- NEW: QLIKE-fit regime split (M2d-regime), init at OLS regime solutions ----
        if n_crisis_t >= MIN_STABLE_N:
            alpha_d_crisis, beta_d_crisis, conv_d_crisis = qlike_fit(
                rv_train[crisis_mask_t], eg_cv[crisis_mask_t], alpha_crisis, beta_crisis)
        else:
            alpha_d_crisis, beta_d_crisis, conv_d_crisis = alpha_d_pooled, beta_d_pooled, True

        if n_calm_t >= MIN_STABLE_N:
            alpha_d_calm, beta_d_calm, conv_d_calm = qlike_fit(
                rv_train[calm_mask_t], eg_cv[calm_mask_t], alpha_calm, beta_calm)
        else:
            alpha_d_calm, beta_d_calm, conv_d_calm = alpha_d_pooled, beta_d_pooled, True

        eg_cv_corr_d_regime = np.where(crisis_mask_t, alpha_d_crisis + beta_d_crisis * eg_cv,
                                        alpha_d_calm + beta_d_calm * eg_cv)

        # ---- M3 (unchanged, pooled-OLS-based) ----
        X_comb = np.column_stack([hist[["rv1", "rv2", "rv5", "rv22"]].values[-n_a:], eg_cv_corr_pooled])
        comb_mod = LinearRegression().fit(X_comb, rv_train)
        comb_coef = (comb_mod.intercept_, comb_mod.coef_)

        # ---- M9/M9c/M9d grid searches (identical mechanics, different input series) ----
        X_har_hist = hist[har_spec.features].values[-n_a:]
        har_fitted = np.maximum(har_spec.coef[0] + X_har_hist @ har_spec.coef[1], 1e-4)
        p_hist = pctile_train
        y9 = rv_train
        valid9 = np.isfinite(p_hist)

        def grid_search(eg_series):
            eg_floor = np.maximum(eg_series, 1e-4)
            best_c, best_k, best_loss = 0.5, 1.0, np.inf
            for c_try in np.arange(0.10, 0.901, 0.05):
                for k_try in range(1, int(M9_K_CAP) + 1):
                    w_try = 1.0 / (1.0 + np.exp(-k_try * (p_hist[valid9] - c_try)))
                    pred_try = np.maximum(w_try * eg_floor[valid9] + (1 - w_try) * har_fitted[valid9], 1e-4)
                    loss_try = qlike(y9[valid9], pred_try)
                    if loss_try < best_loss:
                        best_loss, best_c, best_k = loss_try, float(c_try), float(k_try)
            return best_c, min(best_k, M9_K_CAP)

        m9_c, m9_k = grid_search(eg_cv_corr_pooled)
        m9c_c, m9c_k = grid_search(eg_cv_corr_regime)
        m9d_c, m9d_k = grid_search(eg_cv_corr_d_regime)

        refit_audit.append({
            "refit_date": tdate, "n_train": n_a, "n_crisis_train": n_crisis_t, "n_calm_train": n_calm_t,
            "beta_ols_pooled": eg_beta, "beta_ols_crisis": beta_crisis, "beta_ols_calm": beta_calm,
            "alpha_d_pooled": alpha_d_pooled, "beta_d_pooled": beta_d_pooled, "converged_d_pooled": conv_d_pooled,
            "alpha_d_crisis": alpha_d_crisis, "beta_d_crisis": beta_d_crisis, "converged_d_crisis": conv_d_crisis,
            "alpha_d_calm": alpha_d_calm, "beta_d_calm": beta_d_calm, "converged_d_calm": conv_d_calm,
            "m9_c": m9_c, "m9_k": m9_k, "m9c_c": m9c_c, "m9c_k": m9c_k, "m9d_c": m9d_c, "m9d_k": m9d_k,
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
        m2d_regime_pred = max(alpha_d_crisis + beta_d_crisis * eg_pred, 1e-4)
    else:
        m2c_pred = max(alpha_calm + beta_calm * eg_pred, 1e-4)
        m2d_regime_pred = max(alpha_d_calm + beta_d_calm * eg_pred, 1e-4)
    m2d_pooled_pred = max(alpha_d_pooled + beta_d_pooled * eg_pred, 1e-4)

    x_comb = np.array([row["rv1"], row["rv2"], row["rv5"], row["rv22"], m2b_pred])
    m3_pred = max(float(comb_coef[0] + np.dot(comb_coef[1], x_comb)), 1e-4)

    if np.isfinite(p_t):
        w9 = 1.0 / (1.0 + np.exp(-m9_k * (float(p_t) - m9_c)))
        w9c = 1.0 / (1.0 + np.exp(-m9c_k * (float(p_t) - m9c_c)))
        w9d = 1.0 / (1.0 + np.exp(-m9d_k * (float(p_t) - m9d_c)))
    else:
        w9 = w9c = w9d = 0.5
    m9_pred = max(w9 * m2b_pred + (1 - w9) * m1_pred, 1e-4)
    m9c_pred = max(w9c * m2c_pred + (1 - w9c) * m1_pred, 1e-4)
    m9d_pred = max(w9d * m2d_regime_pred + (1 - w9d) * m1_pred, 1e-4)

    results["M1"].append(m1_pred)
    results["M2_raw"].append(m2_raw)
    results["M2b_pooled"].append(m2b_pred)
    results["M2c_regime"].append(m2c_pred)
    results["M2d_pooled"].append(m2d_pooled_pred)
    results["M2d_regime"].append(m2d_regime_pred)
    results["M3"].append(m3_pred)
    results["M9"].append(m9_pred)
    results["M9c"].append(m9c_pred)
    results["M9d"].append(m9d_pred)
    results["regime_exante"].append("crisis" if is_crisis_day else "calm")
    results["w9"].append(w9); results["w9c"].append(w9c); results["w9d"].append(w9d)
    actuals.append(float(row["rv"]))

actuals = np.array(actuals)
n = len(actuals)
test_dates = test_dates[-n:]
for k in results:
    if k != "regime_exante":
        results[k] = np.array(results[k])

refit_df = pd.DataFrame(refit_audit)
refit_df.to_csv(os.path.join(RESULTS_DIR, "stepB3_refit_audit.csv"), index=False)
print("\n" + "=" * 100)
print("  PER-REFIT: QLIKE-FIT vs OLS-FIT beta, POOLED / CRISIS / CALM")
print("=" * 100)
print(refit_df[["refit_date", "beta_ols_pooled", "beta_d_pooled", "converged_d_pooled",
                "beta_ols_crisis", "beta_d_crisis", "converged_d_crisis",
                "beta_ols_calm", "beta_d_calm", "converged_d_calm"]].to_string(index=False))

n_nonconv = int((~refit_df["converged_d_pooled"]).sum() + (~refit_df["converged_d_crisis"]).sum() + (~refit_df["converged_d_calm"]).sum())
if n_nonconv:
    print(f"\n  {n_nonconv} optimizer run(s) across all refits x {{pooled,crisis,calm}} did NOT converge -- see flags above.")
else:
    print(f"\n  All {len(refit_df)*3} QLIKE-fit optimizations (pooled+crisis+calm x {len(refit_df)} refits) converged.")

print(f"\n  beta_d_pooled range: {refit_df['beta_d_pooled'].min():.4f} -> {refit_df['beta_d_pooled'].max():.4f}   "
      f"(OLS pooled range: {refit_df['beta_ols_pooled'].min():.4f} -> {refit_df['beta_ols_pooled'].max():.4f})")
print(f"  beta_d_crisis range: {refit_df['beta_d_crisis'].min():.4f} -> {refit_df['beta_d_crisis'].max():.4f}   "
      f"(OLS crisis range: {refit_df['beta_ols_crisis'].min():.4f} -> {refit_df['beta_ols_crisis'].max():.4f})")
print(f"  beta_d_calm range: {refit_df['beta_d_calm'].min():.4f} -> {refit_df['beta_d_calm'].max():.4f}   "
      f"(OLS calm range: {refit_df['beta_ols_calm'].min():.4f} -> {refit_df['beta_ols_calm'].max():.4f})")

# --- evaluation crisis mask (same convention throughout the project) ---
test_start = test_dates[0]
train_rv = data.loc[data.index < test_start, "rv"]
crisis_threshold = float(np.percentile(train_rv, 90))
crisis_mask = actuals > crisis_threshold
calm_mask = ~crisis_mask
print(f"\nEvaluation crisis threshold (90th pct training RV, N={len(train_rv)}): {crisis_threshold:.4f}")
print(f"Crisis days: {int(crisis_mask.sum())}  Calm days: {int(calm_mask.sum())}")

m1, m2, m2b, m2c, m2d_p, m2d_r, m3, m9, m9c, m9d = (
    results[k] for k in ["M1", "M2_raw", "M2b_pooled", "M2c_regime", "M2d_pooled", "M2d_regime", "M3", "M9", "M9c", "M9d"])

def metrics(a, f):
    return dict(RMSE=float(np.sqrt(np.mean((a - f) ** 2))), MAE=float(np.mean(np.abs(a - f))), QLIKE=qlike(a, f))

# ---------------------------------------------------------------------------
# M2d-pooled and M2d-regime alone: metrics + MZ
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("  M2d-pooled / M2d-regime ALONE vs M2 raw / M2b / M2c -- RMSE/MAE/QLIKE")
print("=" * 100)
m2d_rows = []
for label, mask in [("overall", np.ones(n, dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for name, arr in [("M2 raw", m2), ("M2b pooled-OLS", m2b), ("M2c regime-OLS", m2c),
                       ("M2d-pooled", m2d_p), ("M2d-regime", m2d_r)]:
        met = metrics(actuals[mask], arr[mask])
        m2d_rows.append({"subsample": label, "model": name, "N": int(mask.sum()), **met})
m2d_df = pd.DataFrame(m2d_rows)
m2d_df.to_csv(os.path.join(RESULTS_DIR, "stepB3_m2d_metrics.csv"), index=False)
print(m2d_df.to_string(index=False))

mz_rows = []
for name, arr in [("M2d-pooled", m2d_p), ("M2d-regime", m2d_r)]:
    mzr = mincer_zarnowitz(actuals, arr)
    mz_rows.append({"model": name, **mzr})
    print(f"\n{name} MZ: alpha={mzr['alpha']:+.4f} (se={mzr['alpha_se']:.4f})  beta={mzr['beta']:.4f} (se={mzr['beta_se']:.4f})  "
          f"R2={mzr['R2']:.4f}  F={mzr['F_stat']:.4f}  p={mzr['F_pval']:.4f}")
pd.DataFrame(mz_rows).to_csv(os.path.join(RESULTS_DIR, "stepB3_m2d_mz.csv"), index=False)

# ---------------------------------------------------------------------------
# DM tests: M2d-regime vs raw M2 (KEY), M2d-regime vs M2c, M2d-pooled vs M2b -- crisis only
# ---------------------------------------------------------------------------
dm_rows = []
print("\n" + "=" * 100)
print("  DM TESTS -- crisis subsample (positive stat => 2nd-named model better)")
print("=" * 100)
for label, ref, target in [
    ("M2d-regime vs raw M2 [KEY]", m2, m2d_r),
    ("M2d-regime vs M2c", m2c, m2d_r),
    ("M2d-pooled vs M2b", m2b, m2d_p),
]:
    for loss in ("qlike", "mse"):
        stat, pval = dm_test(actuals[crisis_mask], ref[crisis_mask], target[crisis_mask], loss=loss)
        dm_rows.append({"comparison": label, "subsample": "crisis", "loss": loss,
                         "N": int(crisis_mask.sum()), "dm_stat": stat, "p_value": pval})
        print(f"  {label:<30} {loss:>6}  N={int(crisis_mask.sum())}  DM={stat:+.4f}  p={pval:.4f}")

# ---------------------------------------------------------------------------
# M9d: metrics + DM vs M3, M9, M9c, M1 -- overall/crisis/calm, both losses
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("  M9d -- RMSE/MAE/QLIKE overall/crisis/calm, vs M9c/M9/M3/M1")
print("=" * 100)
m9d_metric_rows = []
for label, mask in [("overall", np.ones(n, dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for name, arr in [("M9d", m9d), ("M9c", m9c), ("M9", m9), ("M3", m3), ("M1", m1)]:
        met = metrics(actuals[mask], arr[mask])
        m9d_metric_rows.append({"subsample": label, "model": name, "N": int(mask.sum()), **met})
m9d_metrics_df = pd.DataFrame(m9d_metric_rows)
m9d_metrics_df.to_csv(os.path.join(RESULTS_DIR, "stepB3_m9d_metrics.csv"), index=False)
print(m9d_metrics_df.to_string(index=False))

print("\nDM tests: M9d vs {M3, M9, M9c, M1}  (positive stat => M9d better)")
for label, mask in [("overall", np.ones(n, dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for comp_name, comp in [("M3", m3), ("M9", m9), ("M9c", m9c), ("M1", m1)]:
        for loss in ("mse", "qlike"):
            stat, pval = dm_test(actuals[mask], comp[mask], m9d[mask], loss=loss)
            dm_rows.append({"comparison": f"M9d vs {comp_name}", "subsample": label, "loss": loss,
                             "N": int(mask.sum()), "dm_stat": stat, "p_value": pval})
            print(f"  [{label:<8}] M9d vs {comp_name:<3} {loss:>6}  N={mask.sum():4d}  DM={stat:+.4f}  p={pval:.4f}")

dm_df = pd.DataFrame(dm_rows)
dm_df.to_csv(os.path.join(RESULTS_DIR, "stepB3_dm_tests.csv"), index=False)

# ---------------------------------------------------------------------------
# Save all per-day forecasts/weights/losses
# ---------------------------------------------------------------------------
out = pd.DataFrame({
    "date": test_dates, "actual": actuals, "crisis_eval": crisis_mask, "regime_exante": results["regime_exante"],
    "M1": m1, "M2_raw": m2, "M2b_pooled": m2b, "M2c_regime": m2c,
    "M2d_pooled": m2d_p, "M2d_regime": m2d_r, "M3": m3, "M9": m9, "M9c": m9c, "M9d": m9d,
    "w9": results["w9"], "w9c": results["w9c"], "w9d": results["w9d"],
})
for name, arr in [("M2d_regime", m2d_r), ("M9d", m9d), ("M9c", m9c), ("M9", m9), ("M3", m3), ("M1", m1)]:
    out[f"{name}_sq_err"] = (actuals - arr) ** 2
    out[f"{name}_abs_err"] = np.abs(actuals - arr)
out.to_csv(os.path.join(RESULTS_DIR, "stepB3_forecasts_weights_losses.csv"), index=False)

print(f"\nSaved: {RESULTS_DIR}/stepB3_refit_audit.csv")
print(f"Saved: {RESULTS_DIR}/stepB3_m2d_metrics.csv")
print(f"Saved: {RESULTS_DIR}/stepB3_m2d_mz.csv")
print(f"Saved: {RESULTS_DIR}/stepB3_m9d_metrics.csv")
print(f"Saved: {RESULTS_DIR}/stepB3_dm_tests.csv")
print(f"Saved: {RESULTS_DIR}/stepB3_forecasts_weights_losses.csv")

prior_total = 51
this_step_count = len(dm_rows)
print("\n" + "=" * 100)
print(f"  DM-TEST TALLY: {prior_total} (through Step 2) + {this_step_count} (Step 3) = {prior_total + this_step_count}")
print("=" * 100)
