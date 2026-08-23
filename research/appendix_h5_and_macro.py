"""
appendix_h5_and_macro.py
==========================
ROBUSTNESS-CHECK APPENDIX to the concluded forecasting work. NOT a re-run or
replacement of the 8-model comparison, the M9 combination chain, or the
crisis-regime findings -- those results/CSVs/models are untouched by this
script. All outputs go to research/results/appendix_horizon_calendar/.

STEP 1 (reported, not code): confirmed the existing models forecast h=1
(next-trading-day) RV -- see extended_model_zoo.py lines 70 (rv, unshifted)
vs 77-80 (all predictors shifted) vs 375 (actuals.append(row["rv"])).

STEP 2: h=5 (next-week) horizon for M1 HAR+TS, M2 EGARCH, M3 Combined ONLY.
Target = average RV over t+1..t+5 (Corsi HAR multi-horizon convention):
    rv_fwd5_t = mean(RV_{t+1}, ..., RV_{t+5})
Predictors UNCHANGED (rv1/rv2/rv5/rv22, still only using data through t-1).

IMPORTANT no-look-ahead subtlety specific to a FORWARD-looking target: a
naive `hist = data[data.index < tdate]` is NOT enough here, because the last
few rows of that slice have their OWN rv_fwd5 target reaching at-or-past
tdate (e.g. the row at tdate-1's target covers tdate..tdate+4 -- it would
"know" about tdate's own RV). Fixed by additionally dropping the last 5 rows
of the naive slice, so every training example's target is fully resolved
strictly before tdate. See the `iloc[:-5]` line below.

STEP 3: single FOMC-day dummy added to h=1 M1 (-> "M1-macro"). CPI/NFP
dropped per your decision (BLS.gov and FRED blocked automated fetches with
HTTP 403 across three attempts; fabricating 15 years of release dates was
not an acceptable substitute). FOMC dates:
  - 2021-2027: fetched directly from https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    (single official page, high confidence).
  - 2011-2020: assembled via web search referencing individual official
    federalreserve.gov press-release pages (each has a canonical dated URL,
    e.g. monetary20130405a.htm = a real April 5, 2013 release), but not
    independently re-verified page-by-page the way 2021-2027 was -- treat
    this range as lower-confidence than the direct-fetch range. Full list
    saved to appendix_fomc_dates_used.csv for audit. The "FOMC day" used is
    the decision/statement day (2nd day of any 2-day meeting).

No look-ahead: FOMC meeting dates are scheduled and published well in
advance of the meeting itself (confirmed by construction -- these are the
MEETING dates, not some after-the-fact data release), so using them as a
same-day dummy for day t's forecast (built from t-1 and earlier data plus
the KNOWN-IN-ADVANCE fact that t is a scheduled FOMC day) introduces no
leakage.
"""
import os
import sys
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extended_model_zoo import fetch_data, compute_features, qlike, dm_test, mincer_zarnowitz

APPENDIX_DIR = os.path.join("research", "results", "appendix_horizon_calendar")
os.makedirs(APPENDIX_DIR, exist_ok=True)

REFIT_FREQ = 21
MIN_TRAIN = 500
TEST_MONTHS = 18
SIM_PATHS = 500
HAR_FEATURES = ["rv1", "rv2", "rv5", "rv22"]


def metrics(a, f):
    return dict(RMSE=float(np.sqrt(np.mean((a - f) ** 2))), MAE=float(np.mean(np.abs(a - f))), QLIKE=qlike(a, f))


def ols_with_inference(X, y):
    """Manual OLS with SE/t/p, same pattern as extended_model_zoo.mincer_zarnowitz."""
    n, k = X.shape
    Xc = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(Xc, y, rcond=None)[0]
    resid = y - Xc @ beta
    sigma2 = np.sum(resid ** 2) / (n - k - 1)
    XtX_inv = np.linalg.inv(Xc.T @ Xc)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    t_stat = beta / se
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k - 1))
    return beta, se, t_stat, p_val


print("=" * 100)
print("  LOADING DATA (shared with core project -- extended_model_zoo.py untouched)")
print("=" * 100)
df = fetch_data()
data = compute_features(df)
print(f"data: {len(data)} rows, {data.index[0].date()} -> {data.index[-1].date()}")

# same crisis threshold definition/methodology as the core project, computed
# fresh here on the UNMODIFIED h=1 data/split so both Step 2 and Step 3 use
# an identical, self-consistent reference value.
split_date_h1 = data.index[-1] - pd.DateOffset(months=TEST_MONTHS)
train_rv_h1 = data.loc[data.index < split_date_h1, "rv"]
crisis_threshold = float(np.percentile(train_rv_h1, 90))
print(f"Crisis threshold (90th pct training RV, N={len(train_rv_h1)}): {crisis_threshold:.4f}")

# ============================================================================
# STEP 2 -- h=5 horizon: M1, M2, M3
# ============================================================================
print("\n" + "=" * 100)
print("  STEP 2 -- h=5 (next-week) horizon")
print("=" * 100)

data5 = data.copy()
data5["rv_fwd5"] = data5["rv"].rolling(5).mean().shift(-5)
data5 = data5.dropna(subset=["rv_fwd5"])
print(f"h=5 usable rows (after dropping trailing rows with no forward target): {len(data5)}")

split_date_h5 = data5.index[-1] - pd.DateOffset(months=TEST_MONTHS)
test_data5 = data5[data5.index > split_date_h5]
test_dates5 = test_data5.index
print(f"h=5 test window: {test_dates5[0].date()} -> {test_dates5[-1].date()}  N={len(test_dates5)}")
print("(Ends 5 trading days earlier than the h=1 test window, by construction -- ")
print(" data5's own last usable date is 5 rows short of data's last date.)")

m1_coef = None
eg_params5 = None
comb_coef5 = None
eg_alpha5, eg_beta5 = 0.0, 1.0

actuals5, m1_5, m2_5, m3_5, out_dates5 = [], [], [], [], []

for i, tdate in enumerate(test_dates5):
    hist_naive = data5[data5.index < tdate]
    # drop the last 5 rows: their OWN rv_fwd5 target reaches at-or-past tdate
    hist = hist_naive.iloc[:-5] if len(hist_naive) > 5 else hist_naive.iloc[0:0]
    if len(hist) < MIN_TRAIN:
        continue

    do_refit = (i % REFIT_FREQ == 0) or (eg_params5 is None)
    if do_refit:
        print(f"  h=5 refit at {tdate.date()} (step {i}/{len(test_dates5)})...")
        X1 = hist[HAR_FEATURES].values
        y = hist["rv_fwd5"].values
        m1_mod = LinearRegression().fit(X1, y)
        m1_coef = (m1_mod.intercept_, m1_mod.coef_)

        am = arch_model(hist["return"], vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
        eg_res = am.fit(starting_values=eg_params5, disp="off", show_warning=False)
        eg_params5 = eg_res.params.values

        # In-sample proxy for the h=5 EGARCH signal, used only to fit the pooled
        # bias correction feeding M3 -- same role eg_cv plays for M2b in the h=1
        # code (EGARCH's own filtered one-step vol as the in-sample regressor).
        eg_cv = eg_res.conditional_volatility.values
        n_a = min(len(eg_cv), len(hist))
        mz = mincer_zarnowitz(hist["rv_fwd5"].values[-n_a:], eg_cv[-n_a:])
        eg_alpha5, eg_beta5 = mz["alpha"], mz["beta"]

        X_comb = np.column_stack([hist[HAR_FEATURES].values[-n_a:], eg_alpha5 + eg_beta5 * eg_cv[-n_a:]])
        y_comb = hist["rv_fwd5"].values[-n_a:]
        comb_mod = LinearRegression().fit(X_comb, y_comb)
        comb_coef5 = (comb_mod.intercept_, comb_mod.coef_)

    am_fix = arch_model(hist["return"], vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
    eg_fixed = am_fix.fix(eg_params5)
    try:
        fc = eg_fixed.forecast(horizon=5, method="simulation", simulations=SIM_PATHS, reindex=False)
        var_path = fc.variance.iloc[-1].values          # 5 values, one per day ahead
        eg_pred5 = float(np.mean(np.sqrt(np.maximum(var_path, 1e-10))))
    except Exception:
        eg_pred5 = float(eg_fixed.conditional_volatility.iloc[-1])
    eg_pred5 = max(eg_pred5, 1e-4)

    row = data5.loc[tdate]
    x1 = np.array([row[f] for f in HAR_FEATURES])
    m1_pred = max(float(m1_coef[0] + np.dot(m1_coef[1], x1)), 1e-4)

    eg_pred5_corr = max(eg_alpha5 + eg_beta5 * eg_pred5, 1e-4)
    x_comb = np.array(list(x1) + [eg_pred5_corr])
    m3_pred = max(float(comb_coef5[0] + np.dot(comb_coef5[1], x_comb)), 1e-4)

    actuals5.append(float(row["rv_fwd5"]))
    m1_5.append(m1_pred)
    m2_5.append(eg_pred5)      # RAW EGARCH -- matches the ORIGINAL crisis comparison's model identity
    m3_5.append(m3_pred)
    out_dates5.append(tdate)

actuals5 = np.array(actuals5)
m1_5, m2_5, m3_5 = np.array(m1_5), np.array(m2_5), np.array(m3_5)
n5 = len(actuals5)
print(f"\nh=5 evaluation N={n5}")

crisis_mask5 = actuals5 > crisis_threshold
calm_mask5 = ~crisis_mask5
print(f"h=5 crisis days: {int(crisis_mask5.sum())}  calm days: {int(calm_mask5.sum())}  "
      f"(threshold={crisis_threshold:.4f}, same value as h=1)")

h5_metric_rows = []
for label, mask in [("overall", np.ones(n5, dtype=bool)), ("crisis", crisis_mask5), ("calm", calm_mask5)]:
    for name, arr in [("M1_h5", m1_5), ("M2_h5_raw", m2_5), ("M3_h5", m3_5)]:
        met = metrics(actuals5[mask], arr[mask])
        h5_metric_rows.append({"subsample": label, "model": name, "N": int(mask.sum()), **met})
h5_metrics_df = pd.DataFrame(h5_metric_rows)
h5_metrics_df.to_csv(os.path.join(APPENDIX_DIR, "appendix_h5_metrics.csv"), index=False)
print("\nh=5 metrics:")
print(h5_metrics_df.to_string(index=False))

print("\nh=5 DM test: RAW EGARCH (M2) vs M1 HAR+TS -- does the original crisis finding replicate?")
h5_dm_rows = []
for label, mask in [("overall", np.ones(n5, dtype=bool)), ("crisis", crisis_mask5), ("calm", calm_mask5)]:
    for loss in ("qlike", "mse"):
        stat, pval = dm_test(actuals5[mask], m1_5[mask], m2_5[mask], loss=loss)
        h5_dm_rows.append({"subsample": label, "loss": loss, "N": int(mask.sum()), "dm_stat": stat, "p_value": pval})
        print(f"  [{label:<8}] {loss:>6}  N={int(mask.sum())}  DM={stat:+.4f}  p={pval:.4f}")
h5_dm_df = pd.DataFrame(h5_dm_rows)
h5_dm_df.to_csv(os.path.join(APPENDIX_DIR, "appendix_h5_dm_tests.csv"), index=False)

out5 = pd.DataFrame({"date": out_dates5, "actual_rv_fwd5": actuals5, "crisis": crisis_mask5,
                      "M1_h5": m1_5, "M2_h5_raw": m2_5, "M3_h5": m3_5})
out5.to_csv(os.path.join(APPENDIX_DIR, "appendix_h5_forecasts.csv"), index=False)

# ============================================================================
# STEP 3 -- FOMC macro dummy, h=1, M1 only
# ============================================================================
print("\n" + "=" * 100)
print("  STEP 3 -- FOMC-day dummy added to h=1 M1 (-> M1-macro)")
print("=" * 100)

FOMC_DATES_2021_2027_SOURCE = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (direct fetch, single official page)"
FOMC_DATES_2011_2020_SOURCE = "web search referencing individual official federalreserve.gov press-release pages; NOT independently re-verified page-by-page like 2021-2027"

FOMC_DATES = pd.to_datetime([
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-07-31", "2012-09-13", "2012-10-24", "2012-12-12",
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-18", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
])

fomc_audit = pd.DataFrame({"fomc_date": FOMC_DATES})
fomc_audit["source"] = np.where(fomc_audit["fomc_date"].dt.year >= 2021,
                                 FOMC_DATES_2021_2027_SOURCE, FOMC_DATES_2011_2020_SOURCE)
fomc_audit.to_csv(os.path.join(APPENDIX_DIR, "appendix_fomc_dates_used.csv"), index=False)

data["fomc_day"] = data.index.isin(FOMC_DATES).astype(float)
MACRO_FEATURES = HAR_FEATURES + ["fomc_day"]

test_data_h1 = data[data.index > split_date_h1]
test_dates_h1 = test_data_h1.index
print(f"h=1 test window (same as core project): {test_dates_h1[0].date()} -> {test_dates_h1[-1].date()}  "
      f"N={len(test_dates_h1)}")
n_fomc_in_test = int(data.loc[test_dates_h1, "fomc_day"].sum())
print(f"FOMC days within this test window: {n_fomc_in_test}")

m1_coef_h1, m1macro_coef = None, None
actuals_m, m1_m, m1macro_m, fomc_flag = [], [], [], []
coef_history = []

for i, tdate in enumerate(test_dates_h1):
    hist = data[data.index < tdate]
    if len(hist) < MIN_TRAIN:
        continue
    do_refit = (i % REFIT_FREQ == 0) or (m1_coef_h1 is None)
    if do_refit:
        X1 = hist[HAR_FEATURES].values
        y = hist["rv"].values
        m1_mod = LinearRegression().fit(X1, y)
        m1_coef_h1 = (m1_mod.intercept_, m1_mod.coef_)

        Xm = hist[MACRO_FEATURES].values
        m1macro_mod = LinearRegression().fit(Xm, y)
        m1macro_coef = (m1macro_mod.intercept_, m1macro_mod.coef_)
        coef_history.append({"refit_date": tdate, "n_train": len(hist),
                              "n_fomc_train": int(hist["fomc_day"].sum()),
                              "fomc_coef": float(m1macro_mod.coef_[-1])})

    row = data.loc[tdate]
    x1 = np.array([row[f] for f in HAR_FEATURES])
    m1_pred = max(float(m1_coef_h1[0] + np.dot(m1_coef_h1[1], x1)), 1e-4)
    xm = np.array([row[f] for f in MACRO_FEATURES])
    m1macro_pred = max(float(m1macro_coef[0] + np.dot(m1macro_coef[1], xm)), 1e-4)

    actuals_m.append(float(row["rv"]))
    m1_m.append(m1_pred)
    m1macro_m.append(m1macro_pred)
    fomc_flag.append(bool(row["fomc_day"]))

actuals_m = np.array(actuals_m)
m1_m, m1macro_m = np.array(m1_m), np.array(m1macro_m)
fomc_flag = np.array(fomc_flag)
nM = len(actuals_m)
print(f"\nM1-macro evaluation N={nM}, announcement days in this evaluation set: {int(fomc_flag.sum())}")

# --- final-refit coefficient inference (largest, most representative training window) ---
final_hist = data[data.index < test_dates_h1[-1]]
final_hist = final_hist.iloc[-(coef_history[-1]["n_train"]):]  # same window as the last refit used
beta, se, t_stat, p_val = ols_with_inference(final_hist[MACRO_FEATURES].values, final_hist["rv"].values)
fomc_beta, fomc_se, fomc_t, fomc_p = beta[-1], se[-1], t_stat[-1], p_val[-1]
print(f"\nFinal-refit FOMC-day coefficient: {fomc_beta:+.4f}  SE={fomc_se:.4f}  t={fomc_t:+.3f}  p={fomc_p:.4f}  "
      f"(N_train={len(final_hist)}, N_fomc_train={int(final_hist['fomc_day'].sum())})")

coef_df = pd.DataFrame(coef_history)
coef_df.to_csv(os.path.join(APPENDIX_DIR, "appendix_macro_coef_history.csv"), index=False)
pd.DataFrame([{"feature": f, "coef": b, "se": s, "t": t, "p": p}
              for f, b, s, t, p in zip(["intercept"] + MACRO_FEATURES, beta, se, t_stat, p_val)]
             ).to_csv(os.path.join(APPENDIX_DIR, "appendix_macro_final_ols_summary.csv"), index=False)

macro_metric_rows = []
for label, mask in [("overall", np.ones(nM, dtype=bool)), ("announcement_days_only", fomc_flag)]:
    for name, arr in [("M1", m1_m), ("M1-macro", m1macro_m)]:
        met = metrics(actuals_m[mask], arr[mask])
        macro_metric_rows.append({"subsample": label, "model": name, "N": int(mask.sum()), **met})
macro_metrics_df = pd.DataFrame(macro_metric_rows)
macro_metrics_df.to_csv(os.path.join(APPENDIX_DIR, "appendix_macro_metrics.csv"), index=False)
print("\nM1 vs M1-macro metrics:")
print(macro_metrics_df.to_string(index=False))

print("\nDM test: M1-macro vs M1 (positive => M1-macro better)")
macro_dm_rows = []
for label, mask in [("overall", np.ones(nM, dtype=bool)), ("announcement_days_only", fomc_flag)]:
    for loss in ("mse", "qlike"):
        stat, pval = dm_test(actuals_m[mask], m1_m[mask], m1macro_m[mask], loss=loss)
        macro_dm_rows.append({"subsample": label, "loss": loss, "N": int(mask.sum()), "dm_stat": stat, "p_value": pval})
        print(f"  [{label:<22}] {loss:>6}  N={int(mask.sum())}  DM={stat:+.4f}  p={pval:.4f}")
macro_dm_df = pd.DataFrame(macro_dm_rows)
macro_dm_df.to_csv(os.path.join(APPENDIX_DIR, "appendix_macro_dm_tests.csv"), index=False)

out_m = pd.DataFrame({"date": test_dates_h1[-nM:], "actual": actuals_m, "fomc_day": fomc_flag,
                       "M1": m1_m, "M1_macro": m1macro_m})
out_m.to_csv(os.path.join(APPENDIX_DIR, "appendix_macro_forecasts.csv"), index=False)

print(f"\nAll appendix outputs saved to: {APPENDIX_DIR}/")
print("Core project files (research/results/*.csv, research/extended_model_zoo.py, etc.) NOT modified.")
