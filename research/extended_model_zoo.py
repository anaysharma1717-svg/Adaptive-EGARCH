"""
extended_model_zoo.py
=====================
Extended walk-forward OOS model comparison for SPY GK-RV.

Beyond the three models already tested, this script evaluates:

  M1  HAR+TS         : OLS on (RV1, RV2, RV5, RV22)  [baseline]
  M2  EGARCH         : standalone conditional vol
  M3  Combined       : HAR+TS + EGARCH sigma
  M4  Log-HAR        : OLS on log(RV) ~ log(RV1) + log(RV5) + log(RV22) + log(RV2)
  M5  Asymmetric HAR : HAR + negative semi-variance (RV_neg captures leverage via RV)
  M6  HAR + Returns  : HAR + lagged return + lagged |return| (momentum/mean-reversion)
  M7  Rolling HAR    : HAR+TS with fixed 1000-day rolling window (instead of expanding)
  M8  Kitchen Sink   : RV1,RV2,RV5,RV22,RV_neg,ret_lag,abs_ret_lag,EGARCH_sigma

Additional diagnostics beyond standard RMSE/MAE/QLIKE/DM:
  - Mincer-Zarnowitz efficiency test (intercept=0, slope=1)
  - Forecast encompassing test (Fair-Shiller / Harvey-Leybourne-Newbold)
  - Conditional performance: crisis vs calm regime analysis
  - Cumulative squared error difference plot (CSED)

Dependencies: yfinance, arch, statsmodels, sklearn, numpy, pandas, scipy, matplotlib
"""

import sys
import io
import os
import warnings
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.linear_model import LinearRegression, Ridge
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import pacf

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────

def fetch_data(ticker="SPY", start="2011-01-01"):
    log.info(f"Fetching {ticker} from {start}...")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        sys.exit("No data fetched.")
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    log.info(f"  {len(df)} bars  |  {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def compute_features(df):
    """Build all candidate features. Every feature uses shift(1+) to prevent leakage."""
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    rv = np.sqrt(np.maximum(gk_var, 1e-10)) * 100.0

    log_ret = np.log(df["Close"] / df["Close"].shift(1)) * 100.0

    d = pd.DataFrame({"rv": rv, "return": log_ret})

    # Standard HAR lags
    d["rv1"]  = d["rv"].shift(1)
    d["rv2"]  = d["rv"].shift(2)
    d["rv5"]  = d["rv"].shift(1).rolling(5).mean()
    d["rv22"] = d["rv"].shift(1).rolling(22).mean()

    # Log-HAR features
    d["log_rv"]   = np.log(np.maximum(d["rv"], 1e-6))
    d["log_rv1"]  = d["log_rv"].shift(1)
    d["log_rv2"]  = d["log_rv"].shift(2)
    d["log_rv5"]  = d["log_rv"].shift(1).rolling(5).mean()
    d["log_rv22"] = d["log_rv"].shift(1).rolling(22).mean()

    # Asymmetric HAR: negative semi-variance (captures leverage via RV)
    # RV_neg = sqrt(GK_var) only on days when return < 0, else 0
    neg_mask = (d["return"].shift(1) < 0).astype(float)
    d["rv_neg"] = d["rv1"] * neg_mask  # Yesterday's RV, but only if yesterday was a down day

    # Return features (lagged by 1 to avoid leakage)
    d["ret_lag"]     = d["return"].shift(1)
    d["abs_ret_lag"] = np.abs(d["return"].shift(1))

    # Task 4 (M9): rolling percentile rank of rv1 within its OWN trailing 252-day
    # window. rv1 is already t-1-lagged; the comparison window is rv1's own past
    # values (t-2 and earlier), so this never touches day-t's actual RV.
    def _last_pct(x):
        return float((x <= x[-1]).mean())
    d["rv1_pctile"] = d["rv1"].rolling(252, min_periods=252).apply(_last_pct, raw=True)

    return d.dropna()


# ─────────────────────────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────────────────────────

def qlike(a, f):
    r = a**2 / f**2
    return float(np.mean(r - np.log(r) - 1))

def dm_test(a, f1, f2, loss="mse"):
    a, p1, p2 = np.asarray(a), np.asarray(f1), np.asarray(f2)
    if loss == "mse":
        d = (a - p1)**2 - (a - p2)**2
    elif loss == "mae":
        d = np.abs(a - p1) - np.abs(a - p2)
    elif loss == "qlike":
        def _ql(x, h): return x**2/h**2 - np.log(x**2/h**2) - 1
        d = _ql(a, p1) - _ql(a, p2)
    n = len(d)
    mu = np.mean(d)
    v = np.var(d, ddof=1)
    if v < 1e-15:
        return 0.0, 1.0
    k = np.sqrt((n + 1 - 2 + 1/n) / n)
    stat = mu / np.sqrt(v / n) * k
    pval = 2 * (1 - stats.t.cdf(abs(stat), df=n-1))
    return float(stat), float(pval)


def mincer_zarnowitz(actual, forecast):
    """
    MZ regression: actual = alpha + beta * forecast + eps
    Under efficiency: alpha=0, beta=1.
    Returns alpha, beta, R^2, F-stat p-value for joint test (alpha=0, beta=1).
    """
    from scipy.stats import f as f_dist
    a, f = np.asarray(actual), np.asarray(forecast)
    n = len(a)
    X = np.column_stack([np.ones(n), f])
    beta_hat = np.linalg.lstsq(X, a, rcond=None)[0]
    alpha, beta = beta_hat
    residuals = a - X @ beta_hat
    sse = np.sum(residuals**2)
    sigma2 = sse / (n - 2)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv * sigma2))
    R2 = 1 - sse / np.sum((a - np.mean(a))**2)

    # Joint F-test: H0: alpha=0, beta=1
    R = np.array([[1, 0], [0, 1]])
    r = np.array([0, 1])
    diff = R @ beta_hat - r
    F_stat = (diff @ np.linalg.inv(R @ XtX_inv @ R.T * sigma2) @ diff) / 2
    p_val = 1 - f_dist.cdf(F_stat, 2, n - 2)
    return {
        "alpha": float(alpha), "alpha_se": float(se[0]),
        "beta": float(beta), "beta_se": float(se[1]),
        "R2": float(R2), "F_stat": float(F_stat), "F_pval": float(p_val),
    }


def forecast_encompassing(actual, f1, f2, label1="M1", label2="M2"):
    """
    Fair-Shiller encompassing test.
    actual = lambda1 * f1 + lambda2 * f2 + eps
    If lambda2=0 => f1 encompasses f2 (f2 adds nothing).
    If lambda1=0 => f2 encompasses f1.
    """
    a = np.asarray(actual)
    X = np.column_stack([np.asarray(f1), np.asarray(f2)])
    n = len(a)
    beta = np.linalg.lstsq(X, a, rcond=None)[0]
    resid = a - X @ beta
    sigma2 = np.sum(resid**2) / (n - 2)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv * sigma2))
    t1 = beta[0] / se[0]
    t2 = beta[1] / se[1]
    p1 = 2 * (1 - stats.t.cdf(abs(t1), df=n-2))
    p2 = 2 * (1 - stats.t.cdf(abs(t2), df=n-2))
    return {
        "lambda1": float(beta[0]), "t1": float(t1), "p1": float(p1),
        "lambda2": float(beta[1]), "t2": float(t2), "p2": float(p2),
    }


# ─────────────────────────────────────────────────────────────────
# 3. MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────

class ModelSpec:
    """Generic OLS model specification."""
    def __init__(self, name, features, use_log=False, rolling_window=None, use_ridge=False):
        self.name = name
        self.features = features
        self.use_log = use_log
        self.rolling_window = rolling_window
        self.use_ridge = use_ridge
        self.coef = None

    def fit(self, hist):
        X = hist[self.features].values
        y = np.log(np.maximum(hist["rv"].values, 1e-6)) if self.use_log else hist["rv"].values
        if self.rolling_window:
            X = X[-self.rolling_window:]
            y = y[-self.rolling_window:]
        if self.use_ridge:
            mod = Ridge(alpha=1.0).fit(X, y)
        else:
            mod = LinearRegression().fit(X, y)
        self.coef = (mod.intercept_, mod.coef_)
        if self.use_log:
            # Jensen / smearing retransformation term: for y = log(RV) = Xb + eps,
            # E[RV|X] = exp(Xb) * E[exp(eps)]; under eps ~ N(0, sigma^2) this is
            # exp(Xb) * exp(sigma^2/2), NOT exp(Xb) alone. Without this factor,
            # exp(Xb) estimates the MEDIAN of RV, not the MEAN, and is systematically
            # too low (exp is convex). sigma^2 is the in-sample residual variance of
            # THIS fit, i.e. computed over the same expanding window as the fit itself
            # (recomputed every refit alongside the coefficients -- never uses data
            # past the current `hist` cutoff).
            resid = y - mod.predict(X)
            self.log_resid_var = float(np.var(resid, ddof=1))
        else:
            self.log_resid_var = None

    def predict(self, row):
        x = np.array([row[f] for f in self.features])
        pred = self.coef[0] + np.dot(self.coef[1], x)
        if self.use_log:
            pred = np.exp(pred) * np.exp(self.log_resid_var / 2.0)
        return max(pred, 1e-4)


# Model definitions
MODELS = {
    "M1 HAR+TS": ModelSpec("M1 HAR+TS", ["rv1", "rv2", "rv5", "rv22"]),
    "M4 Log-HAR": ModelSpec("M4 Log-HAR", ["log_rv1", "log_rv2", "log_rv5", "log_rv22"], use_log=True),
    "M5 Asym-HAR": ModelSpec("M5 Asym-HAR", ["rv1", "rv2", "rv5", "rv22", "rv_neg"]),
    "M6 HAR+Ret": ModelSpec("M6 HAR+Ret", ["rv1", "rv2", "rv5", "rv22", "ret_lag", "abs_ret_lag"]),
    "M7 Roll-HAR": ModelSpec("M7 Roll-HAR", ["rv1", "rv2", "rv5", "rv22"], rolling_window=1000),
    "M8 KitchSink": ModelSpec("M8 KitchSink",
        ["rv1", "rv2", "rv5", "rv22", "rv_neg", "ret_lag", "abs_ret_lag"],
        use_ridge=True),
}


# ─────────────────────────────────────────────────────────────────
# 4. WALK-FORWARD ENGINE
# ─────────────────────────────────────────────────────────────────

def walk_forward(data, test_months=18, refit_freq=21, min_train=500, sim_paths=500, m9_param_log=None):
    """m9_param_log: optional list; if provided, (refit_date, c, k) tuples for M9's
    logistic weighting parameters are appended to it at every refit, for audit."""
    split_date = data.index[-1] - pd.DateOffset(months=test_months)
    test_data = data[data.index > split_date]
    test_dates = test_data.index
    log.info(f"Test: {test_dates[0].date()} -> {test_dates[-1].date()} ({len(test_dates)} obs)")

    results = {name: [] for name in list(MODELS.keys()) + ["M2 EGARCH", "M2b Corrected EGARCH", "M3 Combined", "M9 Regime"]}
    actuals = []

    eg_params = None
    comb_coef = None
    eg_alpha, eg_beta = 0.0, 1.0
    m9_c, m9_k = 0.5, 1.0
    M9_K_CAP = 20.0

    for i, tdate in enumerate(test_dates):
        hist = data[data.index < tdate]
        if len(hist) < min_train:
            continue

        do_refit = (i % refit_freq == 0) or (eg_params is None)

        if do_refit:
            log.info(f"Refit at {tdate.date()} (step {i}/{len(test_dates)})...")
            # Fit all OLS models
            for spec in MODELS.values():
                spec.fit(hist)

            # Fit EGARCH
            am = arch_model(hist["return"], vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
            eg_res = am.fit(starting_values=eg_params, disp="off", show_warning=False)
            eg_params = eg_res.params.values

            # Fit Combined (HAR features + EGARCH sigma)
            eg_cv = eg_res.conditional_volatility.values
            n_a = min(len(eg_cv), len(hist))
            
            # MZ Correction for EGARCH in-sample
            X_mz = np.column_stack([np.ones(n_a), eg_cv[-n_a:]])
            y_mz = hist["rv"].values[-n_a:]
            beta_hat_mz = np.linalg.lstsq(X_mz, y_mz, rcond=None)[0]
            eg_alpha, eg_beta = float(beta_hat_mz[0]), float(beta_hat_mz[1])
            eg_cv_corr = eg_alpha + eg_beta * eg_cv[-n_a:]
            
            X_comb = np.column_stack([
                hist[["rv1", "rv2", "rv5", "rv22"]].values[-n_a:],
                eg_cv_corr
            ])
            y_comb = hist["rv"].values[-n_a:]
            comb_mod = LinearRegression().fit(X_comb, y_comb)
            comb_coef = (comb_mod.intercept_, comb_mod.coef_)

            # Task 4 (M9): fit logistic weighting (c, k) on in-sample fitted values,
            # over the SAME expanding hist window as everything else above -- never
            # uses data at or past tdate. har_fitted / eg_cv_corr are in-sample fitted
            # series (same pattern already used for M2b's own MZ correction).
            har_spec = MODELS["M1 HAR+TS"]
            X_har_hist = hist[har_spec.features].values[-n_a:]
            har_fitted = np.maximum(har_spec.coef[0] + X_har_hist @ har_spec.coef[1], 1e-4)
            eg_cv_corr_floor = np.maximum(eg_cv_corr, 1e-4)
            p_hist = hist["rv1_pctile"].values[-n_a:]
            y9 = hist["rv"].values[-n_a:]
            valid9 = np.isfinite(p_hist)

            best_c, best_k, best_loss = 0.5, 1.0, np.inf
            for c_try in np.arange(0.10, 0.901, 0.05):
                for k_try in range(1, int(M9_K_CAP) + 1):
                    w_try = 1.0 / (1.0 + np.exp(-k_try * (p_hist[valid9] - c_try)))
                    pred9_try = np.maximum(
                        w_try * eg_cv_corr_floor[valid9] + (1 - w_try) * har_fitted[valid9], 1e-4)
                    loss_try = qlike(y9[valid9], pred9_try)
                    if loss_try < best_loss:
                        best_loss, best_c, best_k = loss_try, float(c_try), float(k_try)
            m9_c, m9_k = best_c, min(best_k, M9_K_CAP)
            if m9_param_log is not None:
                m9_param_log.append({"refit_date": tdate, "c": m9_c, "k": m9_k, "train_qlike": best_loss})

        # EGARCH filter with fixed params
        am_fix = arch_model(hist["return"], vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
        eg_fixed = am_fix.fix(eg_params)
        eg_sigma = float(eg_fixed.conditional_volatility.iloc[-1])

        # Forecasts
        row = data.loc[tdate]
        for name, spec in MODELS.items():
            results[name].append(spec.predict(row))

        # M2 EGARCH standalone
        try:
            fc = eg_fixed.forecast(horizon=1, method="simulation", simulations=sim_paths, reindex=False)
            eg_pred = float(np.sqrt(fc.variance.iloc[-1, 0]))
        except Exception:
            eg_pred = eg_sigma
        results["M2 EGARCH"].append(max(eg_pred, 1e-4))
        
        # M2b Corrected EGARCH
        eg_pred_corr = eg_alpha + eg_beta * eg_pred
        results["M2b Corrected EGARCH"].append(max(eg_pred_corr, 1e-4))

        # M3 Combined
        x_comb = np.array([row["rv1"], row["rv2"], row["rv5"], row["rv22"], eg_pred_corr])
        comb_pred = float(comb_coef[0] + np.dot(comb_coef[1], x_comb))
        results["M3 Combined"].append(max(comb_pred, 1e-4))

        # M9 Regime-weighted: logistic(p_t; c, k) blend of M1 HAR+TS and M2b Corrected
        # EGARCH. p_t = row["rv1_pctile"], already ex-ante (rv1 shift(1) ranked against
        # its own trailing 252-day past window) -- never touches today's actual RV.
        p_t = row["rv1_pctile"]
        if np.isfinite(p_t):
            w9 = 1.0 / (1.0 + np.exp(-m9_k * (float(p_t) - m9_c)))
        else:
            w9 = 0.5
        m1_today = results["M1 HAR+TS"][-1]
        m9_pred = w9 * max(eg_pred_corr, 1e-4) + (1 - w9) * m1_today
        results["M9 Regime"].append(max(m9_pred, 1e-4))

        actuals.append(float(row["rv"]))

    actuals = np.array(actuals)
    for k in results:
        results[k] = np.array(results[k])
    return actuals, results, test_dates[-len(actuals):]


# ─────────────────────────────────────────────────────────────────
# 5. REPORTING
# ─────────────────────────────────────────────────────────────────

def full_report(actuals, results, test_dates):
    S = "=" * 80
    s = "-" * 80
    N = len(actuals)

    print(f"\n{S}")
    print(f"  EXTENDED MODEL ZOO — SPY Garman-Klass RV  (N={N})")
    print(f"  Test: {test_dates[0].date()} -> {test_dates[-1].date()}")
    print(S)

    # ── A. Accuracy Table ────────────────────────────────────────
    print(f"\n  A. FORECAST ACCURACY")
    print(f"  {s}")
    mets = {}
    for name, fc in sorted(results.items()):
        m = {
            "RMSE": float(np.sqrt(np.mean((actuals - fc)**2))),
            "MAE": float(np.mean(np.abs(actuals - fc))),
            "QLIKE": qlike(actuals, fc),
        }
        mets[name] = m

    print(f"  {'Model':<20} {'RMSE':>10} {'MAE':>10} {'QLIKE':>10} {'Rank(RMSE)':>12}")
    print(f"  {s}")
    ranked = sorted(mets, key=lambda k: mets[k]["RMSE"])
    rank_map = {name: i+1 for i, name in enumerate(ranked)}
    for name in sorted(results.keys()):
        m = mets[name]
        r = rank_map[name]
        marker = " <-- BEST" if r == 1 else ""
        print(f"  {name:<20} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['QLIKE']:>10.4f} {r:>10}{marker}")

    # ── B. DM Tests vs M1 HAR+TS ────────────────────────────────
    m1_fc = results["M1 HAR+TS"]
    print(f"\n  B. DIEBOLD-MARIANO TESTS vs M1 HAR+TS")
    print(f"  {s}")
    print(f"  {'Challenger':<20} {'Loss':>6} {'DM stat':>10} {'p-value':>10} {'Sig':>5} {'Result':<20}")
    print(f"  {s}")
    for name in sorted(results.keys()):
        if name == "M1 HAR+TS":
            continue
        fc = results[name]
        for loss in ("mse", "qlike"):
            stat, pval = dm_test(actuals, m1_fc, fc, loss=loss)
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
            if stat < 0 and pval < 0.05:
                res = f"WORSE than M1"
            elif stat > 0 and pval < 0.05:
                res = f"BETTER than M1"
            else:
                res = "Indistinguishable"
            print(f"  {name:<20} {loss.upper():>6} {stat:>+10.3f} {pval:>10.4f} {sig:>5} {res:<20}")

    # ── C. Mincer-Zarnowitz Efficiency ───────────────────────────
    print(f"\n  C. MINCER-ZARNOWITZ FORECAST EFFICIENCY")
    print(f"     (Efficient forecast: alpha=0, beta=1)")
    print(f"  {s}")
    print(f"  {'Model':<20} {'alpha':>8} {'beta':>8} {'R2':>8} {'F(a=0,b=1)':>12} {'p-val':>8} {'Efficient?':>12}")
    print(f"  {s}")
    for name in sorted(results.keys()):
        mz = mincer_zarnowitz(actuals, results[name])
        eff = "YES" if mz["F_pval"] > 0.05 else "NO"
        print(f"  {name:<20} {mz['alpha']:>+8.4f} {mz['beta']:>8.4f} {mz['R2']:>8.4f} {mz['F_stat']:>12.2f} {mz['F_pval']:>8.4f} {eff:>12}")

    # ── D. Forecast Encompassing ──────────────────────────────────
    print(f"\n  D. FORECAST ENCOMPASSING (Fair-Shiller)")
    print(f"     actual = l1*f1 + l2*f2 + eps")
    print(f"     If l2 p-val > 0.05 => f1 encompasses f2 (f2 adds nothing)")
    print(f"  {s}")
    
    best_name = ranked[0]
    # Test key pairs
    pairs = [
        ("M1 HAR+TS", "M2 EGARCH"),
        ("M1 HAR+TS", "M4 Log-HAR"),
        ("M1 HAR+TS", "M5 Asym-HAR"),
        ("M1 HAR+TS", "M6 HAR+Ret"),
        ("M4 Log-HAR", "M5 Asym-HAR"),
        ("M4 Log-HAR", "M2 EGARCH"),
        (best_name, "M2 EGARCH"),
    ]
    # Deduplicate
    seen = set()
    for f1n, f2n in pairs:
        if (f1n, f2n) in seen or f1n not in results or f2n not in results:
            continue
        seen.add((f1n, f2n))
        enc = forecast_encompassing(actuals, results[f1n], results[f2n])
        print(f"  {f1n:<20} l1={enc['lambda1']:+.3f} (p={enc['p1']:.4f})  |  "
              f"{f2n:<20} l2={enc['lambda2']:+.3f} (p={enc['p2']:.4f})  "
              f"{'=> '+f1n+' encompasses' if enc['p2'] > 0.05 else '=> '+f2n+' adds info' if enc['p2'] < 0.05 else ''}")

    # ── E. Conditional Performance (Crisis vs Calm) ──────────────
    print(f"\n  E. CONDITIONAL PERFORMANCE (Crisis vs Calm)")
    print(f"     Crisis defined as actual RV > 90th percentile of training RV")
    print(f"  {s}")
    rv_90 = np.percentile(actuals, 90)
    crisis_mask = actuals > rv_90
    calm_mask = ~crisis_mask
    n_crisis = np.sum(crisis_mask)
    n_calm = np.sum(calm_mask)
    print(f"  Threshold: {rv_90:.4f}  |  Crisis days: {n_crisis}  |  Calm days: {n_calm}")
    print(f"  {'Model':<20} {'RMSE(Crisis)':>14} {'RMSE(Calm)':>14} {'MAE(Crisis)':>14} {'MAE(Calm)':>14}")
    print(f"  {s}")
    for name in sorted(results.keys()):
        fc = results[name]
        rmse_c = float(np.sqrt(np.mean((actuals[crisis_mask] - fc[crisis_mask])**2)))
        rmse_q = float(np.sqrt(np.mean((actuals[calm_mask] - fc[calm_mask])**2)))
        mae_c = float(np.mean(np.abs(actuals[crisis_mask] - fc[crisis_mask])))
        mae_q = float(np.mean(np.abs(actuals[calm_mask] - fc[calm_mask])))
        print(f"  {name:<20} {rmse_c:>14.4f} {rmse_q:>14.4f} {mae_c:>14.4f} {mae_q:>14.4f}")

    # ── E2. Crisis Subsample DM Tests ────────────────────────────
    print(f"\n  E2. CRISIS SUBSAMPLE DM TESTS vs M1 HAR+TS")
    print(f"  {s}")
    print(f"  {'Model':<20} {'Loss':>6} {'DM stat':>10} {'p-value':>10} {'Result':<20}")
    print(f"  {s}")
    a_c = actuals[crisis_mask]
    m1_c = m1_fc[crisis_mask]
    for name in ["M2 EGARCH", "M2b Corrected EGARCH", "M3 Combined"]:
        fc_c = results[name][crisis_mask]
        for loss in ("mse", "qlike"):
            stat, pval = dm_test(a_c, m1_c, fc_c, loss=loss)
            if stat < 0 and pval < 0.05:
                res = f"WORSE than M1"
            elif stat > 0 and pval < 0.05:
                res = f"BETTER than M1"
            else:
                res = "Indistinguishable"
            print(f"  {name:<20} {loss.upper():>6} {stat:>+10.3f} {pval:>10.4f} {res:<20}")

    # ── F. Residual Diagnostics ──────────────────────────────────
    print(f"\n  F. RESIDUAL DIAGNOSTICS (lag=20)")
    print(f"  {s}")
    print(f"  {'Model':<20} {'LB(r) p':>10} {'LB(r2) p':>10} {'ARCH-LM p':>10} {'SigPACF':>14}")
    print(f"  {s}")
    for name in sorted(results.keys()):
        fc = results[name]
        r = actuals - fc
        r = r[np.isfinite(r)]
        try:
            lb1 = acorr_ljungbox(r, lags=[20], return_df=True)["lb_pvalue"].iloc[0]
            lb2 = acorr_ljungbox(r**2, lags=[20], return_df=True)["lb_pvalue"].iloc[0]
            arch_p = het_arch(r, nlags=min(10, len(r)//4))[1]
            ci = 1.96 / np.sqrt(len(r))
            pv = pacf(r, nlags=min(25, len(r)//4), method="ywm")
            sig = [l for l in range(1, len(pv)) if abs(pv[l]) > ci]
        except Exception:
            lb1 = lb2 = arch_p = -1
            sig = []
        lb1_s = f"{lb1:.4f}" if lb1 >= 0 else "ERR"
        lb2_s = f"{lb2:.4f}" if lb2 >= 0 else "ERR"
        ar_s  = f"{arch_p:.4f}" if arch_p >= 0 else "ERR"
        print(f"  {name:<20} {lb1_s:>10} {lb2_s:>10} {ar_s:>10} {str(sig):>14}")

    # ── G. Final Verdict ─────────────────────────────────────────
    print(f"\n{S}")
    print(f"  FINAL VERDICT")
    print(f"  {s}")
    print(f"  Best RMSE:  {ranked[0]} ({mets[ranked[0]]['RMSE']:.4f})")
    best_ql = min(mets, key=lambda k: mets[k]["QLIKE"])
    print(f"  Best QLIKE: {best_ql} ({mets[best_ql]['QLIKE']:.4f})")

    # Does anything beat M1 HAR+TS significantly?
    m1_fc = results["M1 HAR+TS"]
    any_wins = False
    for name in sorted(results.keys()):
        if name == "M1 HAR+TS":
            continue
        stat, pval = dm_test(actuals, m1_fc, results[name], loss="mse")
        if stat > 0 and pval < 0.05:
            print(f"  [+] {name} SIGNIFICANTLY BEATS M1 HAR+TS (DM stat={stat:+.3f}, p={pval:.4f})")
            any_wins = True
    if not any_wins:
        print(f"  [-] NO model significantly beats M1 HAR+TS at the 5% level.")
    print(S + "\n")

    return mets, ranked


# ─────────────────────────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────────────────────────

def make_plots(actuals, results, test_dates, mets, ranked, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Extended Model Zoo — SPY GK-RV Walk-Forward OOS", fontsize=14, fontweight="bold")

    colors = {
        "M1 HAR+TS": "#3498db", "M2 EGARCH": "#e74c3c", "M3 Combined": "#2ecc71",
        "M4 Log-HAR": "#9b59b6", "M5 Asym-HAR": "#f39c12", "M6 HAR+Ret": "#1abc9c",
        "M7 Roll-HAR": "#e67e22", "M8 KitchSink": "#34495e",
    }

    # ── P1: RMSE bar chart (sorted) ──────────────────────────────
    ax = axes[0, 0]
    names_sorted = ranked
    rmses = [mets[n]["RMSE"] for n in names_sorted]
    cols = [colors.get(n, "#95a5a6") for n in names_sorted]
    bars = ax.barh([n.replace(" ", "\n") for n in names_sorted], rmses, color=cols, edgecolor="white", height=0.6)
    for bar, val in zip(bars, rmses):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
                va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("OOS RMSE — All Models", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

    # ── P2: QLIKE bar chart (sorted) ─────────────────────────────
    ax = axes[0, 1]
    ql_ranked = sorted(results.keys(), key=lambda k: mets[k]["QLIKE"])
    qlikes = [mets[n]["QLIKE"] for n in ql_ranked]
    cols2 = [colors.get(n, "#95a5a6") for n in ql_ranked]
    bars = ax.barh([n.replace(" ", "\n") for n in ql_ranked], qlikes, color=cols2, edgecolor="white", height=0.6)
    for bar, val in zip(bars, qlikes):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
                va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("QLIKE (lower is better)")
    ax.set_title("OOS QLIKE — All Models", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

    # ── P3: Cumulative Squared Error Difference (CSED) ───────────
    ax = axes[1, 0]
    base_name = "M1 HAR+TS"
    base_se = (actuals - results[base_name])**2
    for name in ["M4 Log-HAR", "M5 Asym-HAR", "M6 HAR+Ret", "M3 Combined", "M2 EGARCH"]:
        if name not in results:
            continue
        alt_se = (actuals - results[name])**2
        csed = np.cumsum(base_se - alt_se)
        c = colors.get(name, "#95a5a6")
        ax.plot(test_dates, csed, color=c, lw=1.2, alpha=0.85, label=name)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_title(f"CSED vs {base_name} (upward = alt wins)", fontweight="bold")
    ax.set_ylabel("Cumul. SE difference")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.2)

    # ── P4: Forecasts time series (top 3 + actual) ───────────────
    ax = axes[1, 1]
    ax.plot(test_dates, actuals, color="grey", lw=1.0, alpha=0.7, label="Actual GK-RV")
    for name in ranked[:3]:
        c = colors.get(name, "#95a5a6")
        ax.plot(test_dates, results[name], color=c, lw=1.0, alpha=0.8, label=name)
    ax.set_title("Top 3 Forecasts vs Actual", fontweight="bold")
    ax.set_ylabel("Volatility (% / day)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(out_dir, "extended_model_zoo.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved: {path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    df = fetch_data()
    data = compute_features(df)
    log.info(f"Features built: {len(data)} obs, {list(data.columns)}")

    actuals, results, test_dates = walk_forward(data)
    mets, ranked = full_report(actuals, results, test_dates)
    make_plots(actuals, results, test_dates, mets, ranked)

    log.info("Done.")


if __name__ == "__main__":
    main()
