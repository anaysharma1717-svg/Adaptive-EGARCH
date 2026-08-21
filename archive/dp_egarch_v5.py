"""
dp_egarch_v5.py  --  HMM Regime-Switching EGARCH (Adaptive v5)

Unit convention (matching arch library baseline exactly):
    eps = log_return * 100  (percentage returns, e.g. 1% move = 1.0)
    All sigma outputs are in % per day  (e.g. 0.9 = 0.9%/day)
    This matches garch_fc_vol / egarch_fc_vol / rv_yz in forecasts.csv

Parameterization:
    omega_t = -0.05 + 0.10 * sigmoid(w_omega @ X_t)  -> (-0.05, +0.05)
              Covers baseline EGARCH omega range: [-0.042, -0.005]
    alpha_t = FIXED to baseline EGARCH MLE
    beta, gamma fixed to baseline EGARCH MLE

Regularization & Architecture: 
    L2 on w_omega (lambda=0.01)
    Features: HMM 'prob_crisis' + constant bias term
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


# ── Unrolled EGARCH Recursion ────────────────────────────────────────────────
def egarch_recursion(w_omega, fixed_alpha, beta, gamma, eps, X):
    """
    Returns array of log(sigma_t^2) for t=0...T-1.
    alpha is now a fixed scalar.
    """
    T = len(eps)
    log_sigma2 = np.empty(T)

    # Initialise with unconditional log-variance
    log_sigma2[0] = np.log(np.var(eps) + SIGMA2_FLOOR)

    # Pre-compute all raw scores as vectors (one matrix multiply, not T dot products)
    raw_w_vec = X @ w_omega   # shape (T,)
    om_vec    = omega_t(raw_w_vec)   # shape (T,), all omegas pre-computed

    log_floor = np.log(SIGMA2_FLOOR)
    log_cap   = np.log(SIGMA2_CAP)

    for t in range(1, T):
        sigma_prev = np.exp(0.5 * log_sigma2[t - 1])
        z_prev     = eps[t - 1] / (sigma_prev + 1e-10)
        # Clip z to prevent explosion when sigma accidentally hits the floor
        z_prev     = max(-Z_CLIP, min(Z_CLIP, z_prev))

        ls = (om_vec[t]
              + fixed_alpha * (abs(z_prev) - E_ABS_Z)
              + gamma * z_prev
              + beta * log_sigma2[t - 1])

        # Apply floor AND ceiling
        log_sigma2[t] = max(log_floor, min(log_cap, ls))

    return log_sigma2


# ── Negative log-likelihood + L2 ──────────────────────────────────────────────
def neg_loglik(params, fixed_alpha, beta, gamma, eps, X, l2=L2_LAMBDA):
    w_omega = params

    log_sigma2 = egarch_recursion(w_omega, fixed_alpha, beta, gamma, eps, X)
    sigma2 = np.exp(log_sigma2)

    # Gaussian log-likelihood
    ll = -0.5 * np.sum(np.log(2 * np.pi) + log_sigma2 + eps ** 2 / sigma2)

    # L2 penalty (only on w_omega now)
    l2_penalty = l2 * np.dot(w_omega, w_omega)

    return -ll + l2_penalty


# ── Walk-forward evaluation ────────────────────────────────────────────────────
def walk_forward(eps, X, w_omega_init, fixed_alpha, fixed_beta, fixed_gamma, burn_in, refit_every):
    """
    Monthly expanding-window walk-forward.
    """
    T = len(eps)
    records = []
    w_omega_curr = w_omega_init

    for t in range(burn_in, T):
        # Refit at burn-in and then every refit_every days
        if (t == burn_in) or ((t - burn_in) % refit_every == 0):
            eps_train = eps[:t]
            X_train   = X[:t]

            res = minimize(
                neg_loglik,
                w_omega_curr,
                args=(fixed_alpha, fixed_beta, fixed_gamma, eps_train, X_train),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            w_omega_curr = res.x

        # Get the latest sigma_{t-1} via recursion
        log_s2 = egarch_recursion(w_omega_curr, fixed_alpha, fixed_beta, fixed_gamma, eps[:t+1], X[:t+1])
        
        # h=1 forecast
        sigma_1 = np.exp(0.5 * log_s2[-1])
        
        # h=5 and h=21: iterated expectations
        log_s2_h5 = log_s2[-1]
        log_s2_h21 = log_s2[-1]
        for h in range(1, 22):
            sigma_prev_h = np.exp(0.5 * log_s2_h21)
            log_s2_next = (omega_t(w_omega_curr @ X[min(t + h, T - 1)])
                           + fixed_alpha * (E_ABS_Z - E_ABS_Z)
                           + fixed_beta * log_s2_h21)
            log_s2_h21 = log_s2_next
            if h == 5: log_s2_h5 = log_s2_next

        records.append({
            'idx': t,
            'sigma_v5_h1':  sigma_1,
            'sigma_v5_h5':  np.exp(0.5 * log_s2_h5),
            'sigma_v5_h21': np.exp(0.5 * log_s2_h21),
        })

    return pd.DataFrame(records)


# ── Metrics ────────────────────────────────────────────────────────────────────
def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))
def mae(a, b):  return np.mean(np.abs(a - b))
def qlike(rv2, sigma2):
    ratio = rv2 / (sigma2 + 1e-12)
    return np.mean(ratio - np.log(ratio) - 1)

def dm_test(e1, e2):
    d  = e1 ** 2 - e2 ** 2
    n  = len(d)
    d_bar = d.mean()
    var_d = np.var(d, ddof=1) / n
    dm_stat = d_bar / np.sqrt(var_d + 1e-12)
    hln = dm_stat * np.sqrt((n + 1 - 2 + 1/n) / n)
    from scipy.stats import t as t_dist
    p = 2 * t_dist.sf(np.abs(hln), df=n - 1)
    return hln, p


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    feat_df = pd.read_csv('data/dp_egarch_features.csv', index_col=0, parse_dates=True)
    hmm_df = pd.read_csv('data/hmm_features.csv', index_col=0, parse_dates=True)
    
    # Merge HMM features
    feat_df['prob_crisis'] = hmm_df['prob_crisis']

    # UNIT FIX: multiply by 100 to match arch library convention (percentage returns)
    eps   = feat_df['SPY_RETURN'].values * RETURN_SCALE
    
    # Feature X is just an intercept and the HMM state probability
    prob_c = feat_df['prob_crisis'].values
    X = np.column_stack([np.ones(len(prob_c)), prob_c])
    
    dates = feat_df.index

    # ── Fixed baseline parameters from EGARCH mean MLE ─────────────────────────
    params_df = pd.read_csv('params_egarch.csv')
    FIXED_ALPHA = params_df['alpha'].mean()
    FIXED_BETA  = params_df['beta'].mean()
    FIXED_GAMMA = params_df['gamma'].mean()
    print(f"  Fixed alpha = {FIXED_ALPHA:.6f}")
    print(f"  Fixed beta  = {FIXED_BETA:.6f}")
    print(f"  Fixed gamma = {FIXED_GAMMA:.6f}")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    print(f"\nRunning walk-forward (burn-in={BURN_IN}, refit every {REFIT_EVERY} days)...")
    
    n_feat = X.shape[1]
    w_omega_init = np.zeros(n_feat)
    
    wf = walk_forward(
        eps, X, w_omega_init, FIXED_ALPHA, FIXED_BETA, FIXED_GAMMA,
        burn_in=BURN_IN, refit_every=REFIT_EVERY
    )

    # Align with dates and add realized vol target
    wf['Date'] = dates[wf['idx'].values]
    wf = wf.set_index('Date')

    wf = wf.dropna()
    print(f"  Forecast observations: {len(wf)}")

    # ── Pull baseline GARCH / EGARCH forecasts ────────────────────────────────
    # forecasts.csv columns: garch_fc_vol, egarch_fc_vol, rv_yz
    # ── Align with baseline models ─────────────────────────────────────────────
    base_df = pd.read_csv('forecasts.csv', index_col='date', parse_dates=True)
    merged = wf.join(base_df[['garch_fc_vol', 'egarch_fc_vol', 'rv_yz']], how='inner')
    merged = merged.dropna(subset=['rv_yz', 'garch_fc_vol', 'egarch_fc_vol', 'sigma_v5_h1'])
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
    rows.append(metrics_row('Adaptive v5',       merged['sigma_v5_h1'].values,  rv, e_g))

    results = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print("RESULTS -- Model x h=1 (Yang-Zhang RV target)")
    print("=" * 72)
    print(results[['Model', 'RMSE', 'MAE', 'QLIKE', 'DM_stat', 'DM_p', 'N']].to_string(index=False))
    print()

    # ── Stability checks ──────────────────────────────────────────────────────
    print("=" * 72)
    print("STABILITY CHECKS -- Adaptive v5")
    print("=" * 72)
    s = merged['sigma_v5_h1'].values
    print(f"  NaN count        : {np.isnan(s).sum()}")
    print(f"  Inf count        : {np.isinf(s).sum()}")
    print(f"  Min sigma        : {s.min():.6f}")
    print(f"  Max sigma        : {s.max():.4f}")
    print(f"  Mean sigma       : {s.mean():.4f}")
    print(f"  Std sigma        : {s.std():.4f}")
    print(f"  Max daily jump   : {np.max(np.abs(np.diff(s))):.4f}")

    # Save output
    merged[['garch_fc_vol', 'egarch_fc_vol', 'sigma_v5_h1', 'sigma_v5_h5', 'sigma_v5_h21', 'rv_target']].to_csv('data/adaptive_v5_forecasts.csv')
    results.to_csv('data/adaptive_v5_results.csv', index=False)
    print("\nSaved: data/adaptive_v5_forecasts.csv")
    print("Saved: data/adaptive_v5_results.csv")


if __name__ == '__main__':
    main()
