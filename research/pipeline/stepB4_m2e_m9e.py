"""
STEP 4 (final variant): M2e -- regime-switched correction APPLICATION, not a
new fit. Step 3 showed every corrected variant (M2b, M2c, M2d-pooled,
M2d-regime) is significantly worse than raw M2 in crisis at ~p=0.0002 --
implying the fix isn't a better correction, it's no correction in crisis.

M2e:
  ex-ante regime = crisis  -> raw EGARCH forecast, UNCORRECTED
  ex-ante regime = calm    -> M2d-regime's calm branch (QLIKE-fitted calm
                               correction), unchanged from Step 3
No new fitting for M2e itself -- purely a routing change at application time,
reusing the already-fit calm QLIKE correction. M9e still needs its OWN
logistic (c,k) grid search (same unmodified mechanics as M9/M9c/M9d, just
applied to M2e's in-sample equivalent series) since that's the established
procedure for every M9 variant, not a "refit of the correction."

Self-contained (reproduces Step 3's per-refit EGARCH fit and QLIKE-fit calm
branch exactly) so M2e/M9e vs M2d-regime/M9d/M3/M1 are apples-to-apples from
one consistent run. Trimmed to only what Step 4 needs: M1, raw M2, M2d-regime
(crisis+calm QLIKE branches), M3, M9d, plus the new M2e/M9e. M2b/M2c/M9/M9c
are not needed for any comparison requested in this step and are omitted.

No look-ahead: hist = data[data.index < tdate] throughout; regime rule is
rv1_pctile >= 0.90 (ex-ante, unchanged); no threshold/window/grid retuning.
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

results = {k: [] for k in ["M1", "M2_raw", "M2d_regime", "M2e", "M3", "M9d", "M9e",
                            "regime_exante", "w9d", "w9e"]}
actuals = []

eg_params = None
comb_coef = None
eg_alpha, eg_beta = 0.0, 1.0
alpha_crisis, beta_crisis = 0.0, 1.0
alpha_calm, beta_calm = 0.0, 1.0
alpha_d_crisis, beta_d_crisis = 0.0, 1.0
alpha_d_calm, beta_d_calm = 0.0, 1.0
m9d_c, m9d_k = 0.5, 1.0
m9e_c, m9e_k = 0.5, 1.0

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

        # ---- pooled OLS (feeds M3 only, unchanged design) ----
        mz_pooled = mincer_zarnowitz(rv_train, eg_cv)
        eg_alpha, eg_beta = mz_pooled["alpha"], mz_pooled["beta"]
        eg_cv_corr_pooled = eg_alpha + eg_beta * eg_cv

        # ---- regime OLS split (warm start for QLIKE fit only) ----
        crisis_mask_t = pctile_train >= REGIME_THRESHOLD
        calm_mask_t = ~crisis_mask_t
        n_crisis_t, n_calm_t = int(crisis_mask_t.sum()), int(calm_mask_t.sum())

        if n_crisis_t >= MIN_STABLE_N:
            mz_c = mincer_zarnowitz(rv_train[crisis_mask_t], eg_cv[crisis_mask_t])
            alpha_crisis, beta_crisis = mz_c["alpha"], mz_c["beta"]
        if n_calm_t >= MIN_STABLE_N:
            mz_k_ = mincer_zarnowitz(rv_train[calm_mask_t], eg_cv[calm_mask_t])
            alpha_calm, beta_calm = mz_k_["alpha"], mz_k_["beta"]

        # ---- QLIKE-fit regime split (M2d-regime, unchanged from Step 3) ----
        if n_crisis_t >= MIN_STABLE_N:
            alpha_d_crisis, beta_d_crisis, _ = qlike_fit(rv_train[crisis_mask_t], eg_cv[crisis_mask_t], alpha_crisis, beta_crisis)
        if n_calm_t >= MIN_STABLE_N:
            alpha_d_calm, beta_d_calm, _ = qlike_fit(rv_train[calm_mask_t], eg_cv[calm_mask_t], alpha_calm, beta_calm)

        eg_cv_corr_d_regime = np.where(crisis_mask_t, alpha_d_crisis + beta_d_crisis * eg_cv,
                                        alpha_d_calm + beta_d_calm * eg_cv)
        # ---- M2e in-sample series: crisis -> RAW eg_cv, calm -> QLIKE-fit calm branch. No new fit. ----
        eg_cv_corr_e = np.where(crisis_mask_t, eg_cv, alpha_d_calm + beta_d_calm * eg_cv)

        # ---- M3 (unchanged, pooled-OLS-based) ----
        X_comb = np.column_stack([hist[["rv1", "rv2", "rv5", "rv22"]].values[-n_a:], eg_cv_corr_pooled])
        comb_mod = LinearRegression().fit(X_comb, rv_train)
        comb_coef = (comb_mod.intercept_, comb_mod.coef_)

        # ---- M9d / M9e grid searches (identical mechanics, different input series) ----
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

        m9d_c, m9d_k = grid_search(eg_cv_corr_d_regime)
        m9e_c, m9e_k = grid_search(eg_cv_corr_e)

        refit_audit.append({
            "refit_date": tdate, "n_train": n_a, "n_crisis_train": n_crisis_t, "n_calm_train": n_calm_t,
            "alpha_d_calm": alpha_d_calm, "beta_d_calm": beta_d_calm,
            "m9d_c": m9d_c, "m9d_k": m9d_k, "m9e_c": m9e_c, "m9e_k": m9e_k,
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

    p_t = row["rv1_pctile"]
    is_crisis_day = bool(np.isfinite(p_t) and p_t >= REGIME_THRESHOLD)
    if is_crisis_day:
        m2d_regime_pred = max(alpha_d_crisis + beta_d_crisis * eg_pred, 1e-4)
        m2e_pred = eg_pred  # crisis -> raw, uncorrected
    else:
        m2d_regime_pred = max(alpha_d_calm + beta_d_calm * eg_pred, 1e-4)
        m2e_pred = max(alpha_d_calm + beta_d_calm * eg_pred, 1e-4)  # calm -> QLIKE-fit calm branch

    x_comb = np.array([row["rv1"], row["rv2"], row["rv5"], row["rv22"], eg_alpha + eg_beta * eg_pred])
    m3_pred = max(float(comb_coef[0] + np.dot(comb_coef[1], x_comb)), 1e-4)

    if np.isfinite(p_t):
        w9d = 1.0 / (1.0 + np.exp(-m9d_k * (float(p_t) - m9d_c)))
        w9e = 1.0 / (1.0 + np.exp(-m9e_k * (float(p_t) - m9e_c)))
    else:
        w9d = w9e = 0.5
    m9d_pred = max(w9d * m2d_regime_pred + (1 - w9d) * m1_pred, 1e-4)
    m9e_pred = max(w9e * m2e_pred + (1 - w9e) * m1_pred, 1e-4)

    results["M1"].append(m1_pred)
    results["M2_raw"].append(m2_raw)
    results["M2d_regime"].append(m2d_regime_pred)
    results["M2e"].append(m2e_pred)
    results["M3"].append(m3_pred)
    results["M9d"].append(m9d_pred)
    results["M9e"].append(m9e_pred)
    results["regime_exante"].append("crisis" if is_crisis_day else "calm")
    results["w9d"].append(w9d)
    results["w9e"].append(w9e)
    actuals.append(float(row["rv"]))

actuals = np.array(actuals)
n = len(actuals)
test_dates = test_dates[-n:]
for k in results:
    if k != "regime_exante":
        results[k] = np.array(results[k])

refit_df = pd.DataFrame(refit_audit)
refit_df.to_csv(os.path.join(RESULTS_DIR, "stepB4_refit_audit.csv"), index=False)

test_start = test_dates[0]
train_rv = data.loc[data.index < test_start, "rv"]
crisis_threshold = float(np.percentile(train_rv, 90))
crisis_mask = actuals > crisis_threshold
calm_mask = ~crisis_mask
print(f"\nEvaluation crisis threshold (90th pct training RV, N={len(train_rv)}): {crisis_threshold:.4f}")
print(f"Crisis days: {int(crisis_mask.sum())}  Calm days: {int(calm_mask.sum())}")

m1, m2, m2d_r, m2e, m3, m9d, m9e = (
    results[k] for k in ["M1", "M2_raw", "M2d_regime", "M2e", "M3", "M9d", "M9e"])
regime_exante = np.array(results["regime_exante"])

def metrics(a, f):
    return dict(RMSE=float(np.sqrt(np.mean((a - f) ** 2))), MAE=float(np.mean(np.abs(a - f))), QLIKE=qlike(a, f))

# ---------------------------------------------------------------------------
# M2e alone: metrics + MZ
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("  M2e ALONE vs M2 raw / M2d-regime -- RMSE/MAE/QLIKE")
print("=" * 100)
m2e_rows = []
for label, mask in [("overall", np.ones(n, dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for name, arr in [("M2 raw", m2), ("M2d-regime", m2d_r), ("M2e", m2e)]:
        met = metrics(actuals[mask], arr[mask])
        m2e_rows.append({"subsample": label, "model": name, "N": int(mask.sum()), **met})
m2e_df = pd.DataFrame(m2e_rows)
m2e_df.to_csv(os.path.join(RESULTS_DIR, "stepB4_m2e_metrics.csv"), index=False)
print(m2e_df.to_string(index=False))

mz_e = mincer_zarnowitz(actuals, m2e)
print(f"\nM2e MZ: alpha={mz_e['alpha']:+.4f} (se={mz_e['alpha_se']:.4f})  beta={mz_e['beta']:.4f} (se={mz_e['beta_se']:.4f})  "
      f"R2={mz_e['R2']:.4f}  F={mz_e['F_stat']:.4f}  p={mz_e['F_pval']:.4f}")
pd.DataFrame([mz_e]).to_csv(os.path.join(RESULTS_DIR, "stepB4_m2e_mz.csv"), index=False)

# ---------------------------------------------------------------------------
# DM: M2e vs raw M2, M2e vs M2d-regime, M2e vs M1 -- overall/crisis/calm, both losses
# ---------------------------------------------------------------------------
dm_rows = []
print("\n" + "=" * 100)
print("  DM TESTS -- M2e vs {raw M2, M2d-regime, M1}  (positive stat => M2e better)")
print("=" * 100)
for label, mask in [("overall", np.ones(n, dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for comp_name, comp in [("M2 raw", m2), ("M2d-regime", m2d_r), ("M1", m1)]:
        for loss in ("qlike", "mse"):
            stat, pval = dm_test(actuals[mask], comp[mask], m2e[mask], loss=loss)
            dm_rows.append({"comparison": f"M2e vs {comp_name}", "subsample": label, "loss": loss,
                             "N": int(mask.sum()), "dm_stat": stat, "p_value": pval})
            print(f"  [{label:<8}] M2e vs {comp_name:<11} {loss:>6}  N={mask.sum():4d}  DM={stat:+.4f}  p={pval:.4f}")

# ---------------------------------------------------------------------------
# M9e: metrics + DM vs M3, M9d, M1
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("  M9e -- RMSE/MAE/QLIKE overall/crisis/calm, vs M9d/M3/M1")
print("=" * 100)
m9e_metric_rows = []
for label, mask in [("overall", np.ones(n, dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for name, arr in [("M9e", m9e), ("M9d", m9d), ("M3", m3), ("M1", m1)]:
        met = metrics(actuals[mask], arr[mask])
        m9e_metric_rows.append({"subsample": label, "model": name, "N": int(mask.sum()), **met})
m9e_metrics_df = pd.DataFrame(m9e_metric_rows)
m9e_metrics_df.to_csv(os.path.join(RESULTS_DIR, "stepB4_m9e_metrics.csv"), index=False)
print(m9e_metrics_df.to_string(index=False))

print("\nDM tests: M9e vs {M3, M9d, M1}  (positive stat => M9e better)")
for label, mask in [("overall", np.ones(n, dtype=bool)), ("crisis", crisis_mask), ("calm", calm_mask)]:
    for comp_name, comp in [("M3", m3), ("M9d", m9d), ("M1", m1)]:
        for loss in ("mse", "qlike"):
            stat, pval = dm_test(actuals[mask], comp[mask], m9e[mask], loss=loss)
            dm_rows.append({"comparison": f"M9e vs {comp_name}", "subsample": label, "loss": loss,
                             "N": int(mask.sum()), "dm_stat": stat, "p_value": pval})
            print(f"  [{label:<8}] M9e vs {comp_name:<3} {loss:>6}  N={mask.sum():4d}  DM={stat:+.4f}  p={pval:.4f}")

dm_df = pd.DataFrame(dm_rows)
dm_df.to_csv(os.path.join(RESULTS_DIR, "stepB4_dm_tests.csv"), index=False)

# ---------------------------------------------------------------------------
# Realized weight distribution of w9e by ex-ante regime
# ---------------------------------------------------------------------------
w9e = results["w9e"]
w9d = results["w9d"]
print("\n" + "=" * 100)
print("  w9e (M9e logistic weight on M2e) DISTRIBUTION BY EX-ANTE REGIME")
print("=" * 100)
weight_rows = []
for regime_label in ("crisis", "calm"):
    mask = regime_exante == regime_label
    w_sub = w9e[mask]
    stats = {
        "regime": regime_label, "N": int(mask.sum()),
        "mean": float(np.mean(w_sub)), "median": float(np.median(w_sub)),
        "min": float(np.min(w_sub)), "max": float(np.max(w_sub)),
        "p10": float(np.percentile(w_sub, 10)), "p90": float(np.percentile(w_sub, 90)),
        "frac_above_0.95": float(np.mean(w_sub > 0.95)),
        "frac_above_0.99": float(np.mean(w_sub > 0.99)),
    }
    weight_rows.append(stats)
    print(f"  {regime_label:<8} N={stats['N']:>4}  mean={stats['mean']:.4f}  median={stats['median']:.4f}  "
          f"min={stats['min']:.4f}  max={stats['max']:.4f}  p10={stats['p10']:.4f}  p90={stats['p90']:.4f}  "
          f"frac>0.95={stats['frac_above_0.95']:.1%}  frac>0.99={stats['frac_above_0.99']:.1%}")
pd.DataFrame(weight_rows).to_csv(os.path.join(RESULTS_DIR, "stepB4_w9e_distribution.csv"), index=False)

print(f"\n  For comparison, m9e_c/m9e_k fitted per refit (final refit): "
      f"c={refit_df['m9e_c'].iloc[-1]:.3f}  k={refit_df['m9e_k'].iloc[-1]:.1f}  "
      f"(k range across refits: {refit_df['m9e_k'].min():.1f} -> {refit_df['m9e_k'].max():.1f})")

# ---------------------------------------------------------------------------
# Save all per-day forecasts/weights/losses
# ---------------------------------------------------------------------------
out = pd.DataFrame({
    "date": test_dates, "actual": actuals, "crisis_eval": crisis_mask, "regime_exante": regime_exante,
    "M1": m1, "M2_raw": m2, "M2d_regime": m2d_r, "M2e": m2e, "M3": m3, "M9d": m9d, "M9e": m9e,
    "w9d": w9d, "w9e": w9e,
})
for name, arr in [("M2e", m2e), ("M9e", m9e), ("M9d", m9d), ("M3", m3), ("M1", m1)]:
    out[f"{name}_sq_err"] = (actuals - arr) ** 2
    out[f"{name}_abs_err"] = np.abs(actuals - arr)
out.to_csv(os.path.join(RESULTS_DIR, "stepB4_forecasts_weights_losses.csv"), index=False)

print(f"\nSaved: {RESULTS_DIR}/stepB4_refit_audit.csv")
print(f"Saved: {RESULTS_DIR}/stepB4_m2e_metrics.csv")
print(f"Saved: {RESULTS_DIR}/stepB4_m2e_mz.csv")
print(f"Saved: {RESULTS_DIR}/stepB4_m9e_metrics.csv")
print(f"Saved: {RESULTS_DIR}/stepB4_dm_tests.csv")
print(f"Saved: {RESULTS_DIR}/stepB4_w9e_distribution.csv")
print(f"Saved: {RESULTS_DIR}/stepB4_forecasts_weights_losses.csv")

prior_total = 81
this_step_count = len(dm_rows)
print("\n" + "=" * 100)
print(f"  DM-TEST TALLY: {prior_total} (through Step 3) + {this_step_count} (Step 4) = {prior_total + this_step_count}")
print("=" * 100)
