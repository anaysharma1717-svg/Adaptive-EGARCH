"""
dp_egarch_v3.py  --  Adaptive EGARCH v3 (smoothness + feature clipping)

Unit convention (matching arch library baseline exactly):
    eps = log_return * 100  (percentage returns, e.g. 1% move = 1.0)
    All sigma outputs are in % per day  (e.g. 0.9 = 0.9%/day)
    This matches garch_fc_vol / egarch_fc_vol / rv_yz in forecasts.csv

Parameterization:
    omega_t = -0.05 + 0.10 * sigmoid(w_omega @ X_t)  -> (-0.05, +0.05)
              Covers baseline EGARCH omega range: [-0.042, -0.005]
    alpha_t = -0.10 + 0.40 * sigmoid(w_alpha @ X_t)  -> (-0.10, +0.30)
              Covers baseline EGARCH alpha range: [-0.088, +0.259]
    beta, gamma  fixed from mean of baseline EGARCH MLE

Regularization & Architecture: 
    L2 on w_omega and w_alpha (lambda=0.01)
    Temporal smoothness penalty on omega_t and alpha_t (lambda=100.0)
    Feature clipping at +/- 3.0 standard deviations
No EMA.
Walk-forward: monthly refit, expanding window -- matching baseline protocol exactly.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as sigmoid  # numerically stable sigmoid
import warnings
warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────────
E_ABS_Z      = np.sqrt(2 / np.pi)  # E[|z|] for standard normal
L2_LAMBDA    = 0.01                # L2 regularisation strength
SMOOTH_LAMBDA = 100.0              # Temporal smoothness penalty weight
FEATURE_CLIP  = 3.0                # Clip z-scored features at +/- 3 std
BURN_IN      = 500                 # min obs before first refit
REFIT_EVERY  = 21                  # trading days between refits (monthly)
# Floor: sigma never below 0.1 %/day (baseline EGARCH never goes below ~0.3)
# A floor of 1e-8 lets sigma hit 0.0001%/day which makes z=eps/sigma explode.
SIGMA2_FLOOR = 0.01               # sigma_floor = sqrt(0.01) = 0.1 %/day
SIGMA2_CAP   = 10000.0            # sigma_cap   = 100 %/day  (hard ceiling)
RETURN_SCALE = 100.0              # multiply log returns to match arch library convention
Z_CLIP       = 10.0               # clip standardised residuals to [-10, 10]

# ── Parameterisation helpers ───────────────────────────────────────────────────
def omega_t(raw):
    """
    omega_t = -0.05 + 0.10 * sigmoid(raw)  ->  (-0.050, +0.050)
    Baseline EGARCH omega range: [-0.042, -0.005]. This bounds covers it.
    """
    return -0.05 + 0.10 * sigmoid(raw)

def alpha_t(raw):
    """
    alpha_t = -0.10 + 0.40 * sigmoid(raw)  ->  (-0.100, +0.300)
    Baseline EGARCH alpha range: [-0.088, +0.259]. This bounds covers it.
    """
    return -0.10 + 0.40 * sigmoid(raw)


# ── EGARCH recursion with dynamic omega/alpha ──────────────────────────────────
def egarch_recursion(w_omega, w_alpha, beta, gamma, eps, X):
    """
    Run the EGARCH log-variance recursion forward.

    Parameters
    ----------
    w_omega, w_alpha : arrays of shape (n_features,)
    beta, gamma      : scalars (fixed from baseline MLE)
    eps              : array of shape (T,) -- de-meaned returns
    X                : array of shape (T, n_features)

    Returns
    -------
    log_sigma2 : array of shape (T,)
    """
    T = len(eps)
    log_sigma2 = np.empty(T)

    # Initialise with unconditional log-variance
    log_sigma2[0] = np.log(np.var(eps) + SIGMA2_FLOOR)

    # Pre-compute all raw scores as vectors (one matrix multiply, not T dot products)
    # This is the key speedup: X has shape (T, n_feat), so X @ w gives shape (T,)
    raw_w_vec = X @ w_omega   # shape (T,)
    raw_a_vec = X @ w_alpha   # shape (T,)
    om_vec    = omega_t(raw_w_vec)   # shape (T,), all omegas pre-computed
    al_vec    = alpha_t(raw_a_vec)   # shape (T,), all alphas pre-computed

    log_floor = np.log(SIGMA2_FLOOR)
    log_cap   = np.log(SIGMA2_CAP)

    for t in range(1, T):
        sigma_prev = np.exp(0.5 * log_sigma2[t - 1])
        z_prev     = eps[t - 1] / (sigma_prev + 1e-10)
        # Clip z to prevent explosion when sigma accidentally hits the floor
        z_prev     = max(-Z_CLIP, min(Z_CLIP, z_prev))

        ls = (om_vec[t]
              + al_vec[t] * (abs(z_prev) - E_ABS_Z)
              + gamma * z_prev
              + beta * log_sigma2[t - 1])

        # Apply floor AND ceiling
        log_sigma2[t] = max(log_floor, min(log_cap, ls))

    return log_sigma2


# ── Negative log-likelihood + L2 + Smoothness ──────────────────────────────────
def neg_loglik(params, beta, gamma, eps, X, l2=L2_LAMBDA, smooth_lambda=SMOOTH_LAMBDA):
    n_feat = X.shape[1]
    w_omega = params[:n_feat]
    w_alpha = params[n_feat:]

    log_sigma2 = egarch_recursion(w_omega, w_alpha, beta, gamma, eps, X)
    sigma2 = np.exp(log_sigma2)

    # Gaussian log-likelihood
    ll = -0.5 * np.sum(np.log(2 * np.pi) + log_sigma2 + eps ** 2 / sigma2)

    # L2 penalty
    l2_penalty = l2 * (np.dot(w_omega, w_omega) + np.dot(w_alpha, w_alpha))
    
    # Temporal smoothness penalty: \lambda \sum (\omega_t - \omega_{t-1})^2 + (\alpha_t - \alpha_{t-1})^2
    om_vec = omega_t(X @ w_omega)
    al_vec = alpha_t(X @ w_alpha)
    smooth_penalty = smooth_lambda * (np.sum(np.diff(om_vec)**2) + np.sum(np.diff(al_vec)**2))

    return -ll + l2_penalty + smooth_penalty


# ── Walk-forward evaluation ────────────────────────────────────────────────────
def walk_forward(eps, X, beta_fixed, gamma_fixed, feature_cols):
    """
    Monthly expanding-window walk-forward.
    Returns a DataFrame with columns:
        Date, sigma_adaptive, rv_yz (target), horizon
    """
    T = len(eps)
    n_feat = len(feature_cols)

    records = []
    w_omega = np.zeros(n_feat)   # initialise at zero -> sigmoid(0)=0.5
    w_alpha = np.zeros(n_feat)

    refit_dates = []

    for t in range(BURN_IN, T):
        # Refit at burn-in and then every REFIT_EVERY days
        if (t == BURN_IN) or ((t - BURN_IN) % REFIT_EVERY == 0):
            eps_train = eps[:t]
            X_train   = X[:t]

            # Warm-start from previous refit weights (much faster convergence)
            x0 = np.concatenate([w_omega, w_alpha])
            result = minimize(
                neg_loglik,
                x0,
                args=(beta_fixed, gamma_fixed, eps_train, X_train),
                method='L-BFGS-B',
                # Loosened tolerances: speed >> marginal precision for walk-forward
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4}
            )

            if result.success or result.fun < neg_loglik(x0, beta_fixed, gamma_fixed, eps_train, X_train):
                w_omega = result.x[:n_feat]
                w_alpha = result.x[n_feat:]
                refit_dates.append(t)

        # h=1 forecast: use log_sigma2 at t (already computed on training data)
        # We re-compute the full recursion to get sigma at t
        log_s2 = egarch_recursion(w_omega, w_alpha, beta_fixed, gamma_fixed, eps[:t+1], X[:t+1])
        sigma_1 = np.exp(0.5 * log_s2[-1])

        # h=5 forecast: iterate 4 more steps with eps=0 expectation
        log_s2_h = log_s2[-1]
        for h in range(1, 5):
            log_s2_h = (omega_t(w_omega @ X[min(t + h, T - 1)])
                        + 0.0          # E[alpha*(|z|-E|z|)] = 0 under Normal
                        + 0.0          # E[gamma*z] = 0
                        + beta_fixed * log_s2_h)
        sigma_5 = np.exp(0.5 * log_s2_h)

        # h=21 forecast
        log_s2_h = log_s2[-1]
        for h in range(1, 21):
            log_s2_h = (omega_t(w_omega @ X[min(t + h, T - 1)])
                        + 0.0
                        + 0.0
                        + beta_fixed * log_s2_h)
        sigma_21 = np.exp(0.5 * log_s2_h)

        records.append({
            'idx': t,
            'sigma_v3_h1':  sigma_1,
            'sigma_v3_h5':  sigma_5,
            'sigma_v3_h21': sigma_21,
        })

    print(f"  Total refits performed: {len(refit_dates)}")
    return pd.DataFrame(records)


# ── Metrics ────────────────────────────────────────────────────────────────────
def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))
def mae(a, b):  return np.mean(np.abs(a - b))
def qlike(rv2, sigma2):
    ratio = rv2 / (sigma2 + 1e-12)
    return np.mean(ratio - np.log(ratio) - 1)

def dm_test(e1, e2):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) correction.
    H0: equal predictive accuracy.  Negative stat = model1 more accurate.
    """
    d  = e1 ** 2 - e2 ** 2
    n  = len(d)
    d_bar = d.mean()
    # Newey-West variance estimate (lag = 0 only for simplicity at h=1)
    var_d = np.var(d, ddof=1) / n
    dm_stat = d_bar / np.sqrt(var_d + 1e-12)
    # HLN correction
    hln = dm_stat * np.sqrt((n + 1 - 2 + 1/n) / n)
    from scipy.stats import t as t_dist
    p = 2 * t_dist.sf(np.abs(hln), df=n - 1)
    return hln, p


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    feat_df = pd.read_csv('data/dp_egarch_features.csv', index_col=0, parse_dates=True)
    base_df = pd.read_csv('data/forecasts.csv', index_col='date', parse_dates=True)

    # ── Feature columns (all 11, now all Z-scored) ────────────────────────────
    feature_cols = [
        'VIX_Z', 'VVIX_Z', 'VIX_TERM_Z', 'YZ_VOL_21_Z',
        'VOL_Z_Z', 'CREDIT_SPREAD_Z', 'YIELD_CURVE_Z',
        'OVERNIGHT_GAP_Z', 'VOL_MOMENTUM_Z',
        'DAYS_TO_FOMC_Z', 'DAYS_TO_CPI_NFP_Z'
    ]

    feat_df = feat_df.dropna(subset=feature_cols + ['SPY_RETURN'])
    # UNIT FIX: multiply by 100 to match arch library convention (percentage returns)
    # Baseline EGARCH params (beta, gamma, omega) were all fitted on eps*100.
    eps   = feat_df['SPY_RETURN'].values * RETURN_SCALE
    X     = feat_df[feature_cols].values
    
    # Feature Clipping: prevent extreme events from saturating the sigmoid and killing gradients
    X     = np.clip(X, -FEATURE_CLIP, FEATURE_CLIP)
    
    dates = feat_df.index

    # ── Fixed beta & gamma from baseline EGARCH mean MLE ─────────────────────
    params_egarch = pd.read_csv('data/params_egarch.csv')
    beta_fixed  = params_egarch['beta'].mean()
    gamma_fixed = params_egarch['gamma'].mean()
    print(f"  Fixed beta  = {beta_fixed:.6f}")
    print(f"  Fixed gamma = {gamma_fixed:.6f}")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    print(f"\nRunning walk-forward (burn-in={BURN_IN}, refit every {REFIT_EVERY} days)...")
    wf = walk_forward(eps, X, beta_fixed, gamma_fixed, feature_cols)

    # Align with dates and add realized vol target
    wf['Date'] = dates[wf['idx'].values]
    wf = wf.set_index('Date')

    wf = wf.dropna()
    print(f"  Forecast observations: {len(wf)}")

    # ── Pull baseline GARCH / EGARCH forecasts ────────────────────────────────
    # forecasts.csv columns: garch_fc_vol, egarch_fc_vol, rv_yz
    # All three are in %/day units -- same as our sigma_v1 output (eps*100 scale).
    # UNIT FIX: do NOT use YZ_VOL_21 from features CSV (that is annualized %).
    # Use rv_yz from forecasts.csv which is already in %/day, consistent with
    # garch_fc_vol and egarch_fc_vol.
    base_df.index = pd.to_datetime(base_df.index)
    merged = wf.join(base_df[['garch_fc_vol', 'egarch_fc_vol', 'rv_yz']], how='inner')
    merged = merged.dropna(subset=['rv_yz', 'garch_fc_vol', 'egarch_fc_vol', 'sigma_v3_h1'])
    merged.rename(columns={'rv_yz': 'rv_target'}, inplace=True)

    # ── Compute metrics for all three models ─────────────────────────────────
    def metrics_row(name, pred, rv_target, ref_err=None):
        e_mod = pred - rv_target
        dm_stat, dm_p = (np.nan, np.nan)
        if ref_err is not None:
            dm_stat, dm_p = dm_test(e_mod, ref_err)
        return {
            'Model':    name,
            'Horizon':  'h=1',
            'RMSE':     rmse(pred, rv_target),
            'MAE':      mae(pred, rv_target),
            'QLIKE':    qlike(rv_target**2, pred**2),
            'DM_stat':  round(dm_stat, 4) if not np.isnan(dm_stat) else '--',
            'DM_p':     round(dm_p, 4)    if not np.isnan(dm_p)    else '--',
            'N':        len(rv_target),
        }

    rv   = merged['rv_target'].values
    e_g  = merged['garch_fc_vol'].values - rv   # GARCH errors (reference)

    rows = []
    rows.append(metrics_row('GARCH(1,1)',       merged['garch_fc_vol'].values,  rv))
    rows.append(metrics_row('EGARCH(1,1,1,t)',  merged['egarch_fc_vol'].values, rv, e_g))
    rows.append(metrics_row('Adaptive v3',       merged['sigma_v3_h1'].values,  rv, e_g))

    results = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print("RESULTS -- Model x h=1 (Yang-Zhang RV target)")
    print("=" * 72)
    print(results[['Model', 'RMSE', 'MAE', 'QLIKE', 'DM_stat', 'DM_p', 'N']].to_string(index=False))
    print()

    # ── Stability checks ──────────────────────────────────────────────────────
    print("=" * 72)
    print("STABILITY CHECKS -- Adaptive v3")
    print("=" * 72)
    s = merged['sigma_v3_h1'].values
    print(f"  NaN count        : {np.isnan(s).sum()}")
    print(f"  Inf count        : {np.isinf(s).sum()}")
    print(f"  Min sigma        : {s.min():.6f}")
    print(f"  Max sigma        : {s.max():.4f}")
    print(f"  Mean sigma       : {s.mean():.4f}")
    print(f"  Std sigma        : {s.std():.4f}")
    print(f"  Max daily jump   : {np.max(np.abs(np.diff(s))):.4f}")

    # Save output
    merged[['garch_fc_vol', 'egarch_fc_vol', 'sigma_v3_h1', 'sigma_v3_h5', 'sigma_v3_h21', 'rv_target']].to_csv('data/adaptive_v3_forecasts.csv')
    results.to_csv('data/adaptive_v3_results.csv', index=False)
    print("\nSaved: data/adaptive_v3_forecasts.csv")
    print("Saved: data/adaptive_v3_results.csv")


if __name__ == '__main__':
    main()
