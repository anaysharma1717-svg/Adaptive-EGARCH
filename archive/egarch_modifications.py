"""
egarch_modifications.py
══════════════════════════════════════════════════════════════════════════════
Three structural EGARCH modifications, evaluated with the identical
walk-forward protocol (expanding window, monthly refit, Yang-Zhang RV target)
as the GARCH/EGARCH baselines in forecasts.csv.

Unit convention (matching arch library baseline exactly):
    eps = log_return * 100  (percentage returns, 1% move = 1.0)
    All sigma outputs are in %/day

────────────────────────────────────────────────────────────────────────────
Mod A — VIX-Augmented EGARCH
    log(h_t) = ω + β·log(h_{t-1}) + α·(|z_{t-1}|−E|z|) + γ·z_{t-1} + δ·log(VIX_{t-1})
    Extra parameter δ: freely estimated each refit.

Mod B — Realized EGARCH
    log(h_t) = ω + β·log(h_{t-1}) + α·(log(RV_{t-1}) − μ_rv) + γ·z_{t-1}
    Replaces the |z|-E|z| innovation with the Yang-Zhang RV (lagged 1 day).
    RV is 21d trailing Yang-Zhang, computed from same-day OHLC → fully causal.

Mod C — GJR-GARCH(1,1,t)   [Glosten-Jagannathan-Runkle 1993]
    h_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·𝟙[ε_{t-1}<0] + β·h_{t-1}
    Hard-threshold leverage. Simpler than EGARCH's exponential leverage.
    Student-t distribution retained.

────────────────────────────────────────────────────────────────────────────
Baselines (from forecasts.csv):
    GARCH(1,1)          RMSE=0.266  MAE=0.177
    EGARCH(1,1,1,t)     RMSE=0.295  MAE=0.202
    Adaptive v4 (best)  RMSE=0.299  MAE=0.204
────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.special import gammaln
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')
os.makedirs('data',  exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
BURN_IN       = 500       # minimum obs before first refit
REFIT_EVERY   = 21        # trading days between refits (≈ monthly)
RETURN_SCALE  = 100.0     # log-return × 100 → %/day  (matches arch library)
SIGMA2_FLOOR  = 0.01      # sigma_floor = 0.1 %/day
SIGMA2_CAP    = 10_000.0  # sigma_cap   = 100 %/day
H_FLOOR       = SIGMA2_FLOOR
H_CAP         = SIGMA2_CAP
LOG_H_FLOOR   = np.log(SIGMA2_FLOOR)
LOG_H_CAP     = np.log(SIGMA2_CAP)
Z_CLIP        = 10.0
E_ABS_Z       = np.sqrt(2.0 / np.pi)   # E[|z|] for standard Normal

# ══════════════════════════════════════════════════════════════════════════════
#  STUDENT-t LOG-LIKELIHOOD  (vectorised, numerically stable)
# ══════════════════════════════════════════════════════════════════════════════
def t_logpdf_vec(z, nu):
    """
    Log-PDF of the standardized Student-t with nu degrees of freedom.
    Standardized so that Var(z) = 1  (requires nu > 2).
    """
    nu_2 = nu / 2.0
    log_norm = (gammaln(nu_2 + 0.5)
                - gammaln(nu_2)
                - 0.5 * np.log(np.pi * (nu - 2.0)))
    return log_norm - (nu_2 + 0.5) * np.log(1.0 + z ** 2 / (nu - 2.0))


def total_loglik(eps, sigma2, nu):
    """
    Sum of Student-t log-likelihood over T observations.
    eps    : array (T,)  returns in %/day
    sigma2 : array (T,)  conditional variance in (%/day)^2
    nu     : scalar      degrees of freedom (> 2)
    """
    sigma  = np.sqrt(np.maximum(sigma2, 1e-12))
    z      = np.clip(eps / sigma, -Z_CLIP, Z_CLIP)
    return np.sum(t_logpdf_vec(z, nu) - np.log(sigma))


# ══════════════════════════════════════════════════════════════════════════════
#  MOD A — VIX-AUGMENTED EGARCH
#  params = [ω, α, β, γ, δ, log_nu_m2]
#  log(h_t) = ω + α·(|z_{t-1}|−E|z|) + β·log(h_{t-1}) + γ·z_{t-1} + δ·log(VIX_{t-1})
# ══════════════════════════════════════════════════════════════════════════════
def _mod_a_step(log_h_prev, eps_prev, log_vix_prev, om, al, be, gm, de):
    sig_prev = np.exp(0.5 * log_h_prev)
    z        = np.clip(eps_prev / (sig_prev + 1e-10), -Z_CLIP, Z_CLIP)
    lh       = om + al * (abs(z) - E_ABS_Z) + be * log_h_prev + gm * z + de * log_vix_prev
    return np.clip(lh, LOG_H_FLOOR, LOG_H_CAP)


def _mod_a_recursion(params, eps, log_vix):
    """Full log-variance recursion for Mod A."""
    om, al, be, gm, de = params[:5]
    T      = len(eps)
    log_h2 = np.empty(T)
    log_h2[0] = np.log(np.var(eps) + SIGMA2_FLOOR)
    for t in range(1, T):
        log_h2[t] = _mod_a_step(log_h2[t-1], eps[t-1], log_vix[t-1], om, al, be, gm, de)
    return log_h2


def _neg_ll_mod_a(params, eps, log_vix):
    nu = np.exp(params[5]) + 2.01
    log_h2 = _mod_a_recursion(params, eps, log_vix)
    return -total_loglik(eps, np.exp(log_h2), nu)


def _fit_mod_a(eps, log_vix, x0=None):
    if x0 is None:
        x0 = np.array([-0.05, 0.15, 0.95, -0.15, 0.05, np.log(4.0)])
    bounds = [
        (-2.0,  2.0),                          # ω
        (-1.0,  1.0),                          # α
        ( 0.05, 0.999),                        # β  (persistence)
        (-1.0,  1.0),                          # γ  (leverage)
        (-2.0,  2.0),                          # δ  (VIX loading)
        (np.log(0.01), np.log(98.0)),          # log(ν−2)
    ]
    res = minimize(_neg_ll_mod_a, x0, args=(eps, log_vix),
                   method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 400, 'ftol': 1e-8, 'gtol': 1e-6})
    return res.x, np.exp(res.x[5]) + 2.01


# ══════════════════════════════════════════════════════════════════════════════
#  MOD B — REALIZED EGARCH
#  params = [ω, α, β, γ, log_nu_m2]
#  log(h_t) = ω + α·(log(RV_{t-1}) − μ_rv) + β·log(h_{t-1}) + γ·z_{t-1}
# ══════════════════════════════════════════════════════════════════════════════
def _mod_b_step(log_h_prev, eps_prev, log_rv_centered_prev, om, al, be, gm):
    sig_prev = np.exp(0.5 * log_h_prev)
    z        = np.clip(eps_prev / (sig_prev + 1e-10), -Z_CLIP, Z_CLIP)
    lh       = om + al * log_rv_centered_prev + be * log_h_prev + gm * z
    return np.clip(lh, LOG_H_FLOOR, LOG_H_CAP)


def _mod_b_recursion(params, eps, log_rv_c):
    """Full log-variance recursion for Mod B."""
    om, al, be, gm = params[:4]
    T      = len(eps)
    log_h2 = np.empty(T)
    log_h2[0] = np.log(np.var(eps) + SIGMA2_FLOOR)
    for t in range(1, T):
        log_h2[t] = _mod_b_step(log_h2[t-1], eps[t-1], log_rv_c[t-1], om, al, be, gm)
    return log_h2


def _neg_ll_mod_b(params, eps, log_rv_c):
    nu = np.exp(params[4]) + 2.01
    log_h2 = _mod_b_recursion(params, eps, log_rv_c)
    return -total_loglik(eps, np.exp(log_h2), nu)


def _fit_mod_b(eps, log_rv_c, x0=None):
    if x0 is None:
        x0 = np.array([-0.05, 0.15, 0.95, -0.15, np.log(4.0)])
    bounds = [
        (-2.0,  2.0),
        (-3.0,  3.0),           # α: RV is in log scale, needs wider range
        ( 0.05, 0.999),
        (-1.0,  1.0),
        (np.log(0.01), np.log(98.0)),
    ]
    res = minimize(_neg_ll_mod_b, x0, args=(eps, log_rv_c),
                   method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 400, 'ftol': 1e-8, 'gtol': 1e-6})
    return res.x, np.exp(res.x[4]) + 2.01


# ══════════════════════════════════════════════════════════════════════════════
#  MOD C — GJR-GARCH(1,1,t)
#  params = [ω, α, γ, β, log_nu_m2]
#  h_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·𝟙[ε_{t-1}<0] + β·h_{t-1}
# ══════════════════════════════════════════════════════════════════════════════
def _mod_c_step(h_prev, eps_prev, om, al, gm, be):
    I   = 1.0 if eps_prev < 0 else 0.0
    eps2 = eps_prev ** 2
    h    = om + al * eps2 + gm * eps2 * I + be * h_prev
    return np.clip(h, H_FLOOR, H_CAP)


def _mod_c_recursion(params, eps):
    """Full variance recursion for GJR-GARCH."""
    om, al, gm, be = params[:4]
    T = len(eps)
    h  = np.empty(T)
    h[0] = np.var(eps)
    for t in range(1, T):
        h[t] = _mod_c_step(h[t-1], eps[t-1], om, al, gm, be)
    return h


def _neg_ll_mod_c(params, eps):
    om, al, gm, be = params[:4]
    nu = np.exp(params[4]) + 2.01
    # Stationarity: α + γ/2 + β < 1; positivity constraints
    if om <= 0 or al < 0 or gm < 0 or be < 0:
        return 1e10
    if al + gm / 2.0 + be >= 0.9999:
        return 1e10
    h = _mod_c_recursion(params, eps)
    return -total_loglik(eps, h, nu)


def _fit_mod_c(eps, x0=None):
    if x0 is None:
        var0 = np.var(eps)
        x0 = np.array([var0 * 0.05, 0.05, 0.08, 0.88, np.log(4.0)])
    bounds = [
        (1e-6, None),          # ω > 0
        (1e-6, 0.5),           # α ≥ 0
        (1e-6, 0.5),           # γ ≥ 0
        (1e-6, 0.999),         # β ≥ 0
        (np.log(0.01), np.log(98.0)),
    ]
    res = minimize(_neg_ll_mod_c, x0, args=(eps,),
                   method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 400, 'ftol': 1e-8, 'gtol': 1e-6})
    return res.x, np.exp(res.x[4]) + 2.01


# ══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD ENGINE  (state-stepping between refits — O(T) per model)
# ══════════════════════════════════════════════════════════════════════════════

def wf_mod_a(eps, log_vix, dates):
    """Walk-forward for Mod A (VIX-Augmented EGARCH)."""
    T         = len(eps)
    forecasts = np.full(T, np.nan)
    params    = None   # [ω, α, β, γ, δ, log_nu_m2]
    log_h_state = np.log(np.var(eps[:BURN_IN]) + SIGMA2_FLOOR)
    x0        = None

    for t in range(1, T):
        # ── Refit at burn-in and then monthly ────────────────────────────────
        if t >= BURN_IN and ((t == BURN_IN) or ((t - BURN_IN) % REFIT_EVERY == 0)):
            try:
                new_params, _ = _fit_mod_a(eps[:t], log_vix[:t], x0)
                # After refit: re-anchor state by running full recursion to t
                log_h_full   = _mod_a_recursion(new_params, eps[:t], log_vix[:t])
                log_h_state  = log_h_full[-1]
                params       = new_params
                x0           = new_params.copy()
            except Exception:
                pass

        # ── One-step state update ─────────────────────────────────────────────
        if params is not None:
            om, al, be, gm, de = params[:5]
            log_h_state = _mod_a_step(log_h_state, eps[t-1], log_vix[t-1], om, al, be, gm, de)
            forecasts[t] = np.exp(0.5 * log_h_state)

    return pd.Series(forecasts, index=dates, name='mod_a')


def wf_mod_b(eps, rv_yz_raw, dates):
    """Walk-forward for Mod B (Realized EGARCH).
    rv_yz_raw : raw RV values in %/day, aligned to dates. We use log(RV_{t-1}) centered
                on an expanding mean of log(RV) computed up to each refit."""
    T         = len(eps)
    forecasts = np.full(T, np.nan)
    params    = None
    log_h_state = np.log(np.var(eps[:BURN_IN]) + SIGMA2_FLOOR)
    x0        = None

    # Pre-compute log_rv (replace zeros/negatives with floor)
    log_rv = np.log(np.maximum(rv_yz_raw, 0.01))

    for t in range(1, T):
        # ── Refit ─────────────────────────────────────────────────────────────
        if t >= BURN_IN and ((t == BURN_IN) or ((t - BURN_IN) % REFIT_EVERY == 0)):
            mu_rv    = np.nanmean(log_rv[:t])     # expanding mean of log(RV)
            log_rv_c = log_rv[:t] - mu_rv         # centered
            try:
                new_params, _ = _fit_mod_b(eps[:t], log_rv_c, x0)
                log_h_full    = _mod_b_recursion(new_params, eps[:t], log_rv_c)
                log_h_state   = log_h_full[-1]
                params        = new_params
                x0            = new_params.copy()
                _mu_rv_cache  = mu_rv             # store for step function
            except Exception:
                pass

        # ── One-step update ───────────────────────────────────────────────────
        if params is not None:
            om, al, be, gm = params[:4]
            lrv_c_prev     = log_rv[t-1] - _mu_rv_cache
            log_h_state    = _mod_b_step(log_h_state, eps[t-1], lrv_c_prev, om, al, be, gm)
            forecasts[t]   = np.exp(0.5 * log_h_state)

    return pd.Series(forecasts, index=dates, name='mod_b')


def wf_mod_c(eps, dates):
    """Walk-forward for Mod C (GJR-GARCH)."""
    T         = len(eps)
    forecasts = np.full(T, np.nan)
    params    = None
    h_state   = np.var(eps[:BURN_IN])
    x0        = None

    for t in range(1, T):
        # ── Refit ─────────────────────────────────────────────────────────────
        if t >= BURN_IN and ((t == BURN_IN) or ((t - BURN_IN) % REFIT_EVERY == 0)):
            try:
                new_params, _ = _fit_mod_c(eps[:t], x0)
                h_full        = _mod_c_recursion(new_params, eps[:t])
                h_state       = h_full[-1]
                params        = new_params
                x0            = new_params.copy()
            except Exception:
                pass

        # ── One-step update ───────────────────────────────────────────────────
        if params is not None:
            om, al, gm, be = params[:4]
            h_state      = _mod_c_step(h_state, eps[t-1], om, al, gm, be)
            forecasts[t] = np.sqrt(h_state)

    return pd.Series(forecasts, index=dates, name='mod_c')


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS & DIEBOLD-MARIANO
# ══════════════════════════════════════════════════════════════════════════════
def rmse(a, b):  return np.sqrt(np.mean((a - b) ** 2))
def mae(a, b):   return np.mean(np.abs(a - b))
def qlike(rv, sig):
    ratio = rv ** 2 / (sig ** 2 + 1e-12)
    return np.mean(ratio - np.log(ratio) - 1.0)

def dm_test(e1, e2):
    """Diebold-Mariano (Harvey-Leybourne-Newbold 1997 corrected). Squared loss."""
    from scipy.stats import t as t_dist
    d     = e1 ** 2 - e2 ** 2
    n     = len(d)
    d_bar = d.mean()
    var_d = np.var(d, ddof=1) / n
    dm    = d_bar / np.sqrt(var_d + 1e-12)
    hln   = dm * np.sqrt((n + 1 - 2 + 1.0 / n) / n)
    p     = 2 * t_dist.sf(np.abs(hln), df=n - 1)
    return round(hln, 4), round(p, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  EGARCH STRUCTURAL MODIFICATIONS — Walk-Forward Evaluation")
    print("=" * 72)

    # ── Load SPY returns ───────────────────────────────────────────────────────
    print("\nLoading data...")
    feat_df = pd.read_csv('data/dp_egarch_features.csv', index_col=0, parse_dates=True)
    base_df = pd.read_csv('forecasts.csv', index_col='date', parse_dates=True)
    base_df.index = pd.to_datetime(base_df.index)

    eps_series = (feat_df['SPY_RETURN'].dropna() * RETURN_SCALE)
    dates      = eps_series.index
    eps        = eps_series.values.astype(float)
    print(f"  Returns: {len(eps)} days  ({dates[0].date()} → {dates[-1].date()})")

    # ── VIX for Mod A ─────────────────────────────────────────────────────────
    print("  Downloading VIX...")
    vix_raw = yf.download('^VIX', start=str(dates[0].date()),
                          progress=False, auto_adjust=True)['Close'].squeeze()
    vix_raw.index = pd.to_datetime(vix_raw.index)
    vix_aligned   = vix_raw.reindex(dates).ffill().bfill()
    log_vix       = np.log(vix_aligned.values.astype(float))
    print(f"  VIX: {(~np.isnan(log_vix)).sum()} valid rows")

    # ── RV for Mod B: lagged Yang-Zhang from forecasts.csv ────────────────────
    # rv_yz[t] is the trailing 21-day Yang-Zhang RV for day t (in %/day).
    # We use rv_yz[t-1] (yesterday's RV) as the Realized EGARCH innovation.
    rv_aligned = base_df['rv_yz'].reindex(dates).ffill().bfill()
    rv_arr     = rv_aligned.values.astype(float)
    # Lag by 1: rv used in recursion at step t is rv[t-1]
    rv_lagged  = np.roll(rv_arr, 1)
    rv_lagged[0] = rv_arr[0]  # initialise first obs
    print(f"  RV:  {(~np.isnan(rv_arr)).sum()} valid rows")

    # ── Run walk-forward for each modification ─────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  Walk-forward  burn_in={BURN_IN}  refit_every={REFIT_EVERY} days")
    print(f"{'─'*72}")

    print("\n  [Mod A] VIX-Augmented EGARCH...")
    fc_a = wf_mod_a(eps, log_vix, dates)

    print("\n  [Mod B] Realized EGARCH (log-RV innovation)...")
    fc_b = wf_mod_b(eps, rv_lagged, dates)

    print("\n  [Mod C] GJR-GARCH(1,1,t)...")
    fc_c = wf_mod_c(eps, dates)

    # ── Merge with baselines ───────────────────────────────────────────────────
    print("\n  Merging with baselines...")
    mod_df = pd.DataFrame({'mod_a': fc_a, 'mod_b': fc_b, 'mod_c': fc_c})
    merged = mod_df.join(base_df[['garch_fc_vol', 'egarch_fc_vol', 'rv_yz']], how='inner')
    merged = merged.dropna(subset=['rv_yz', 'garch_fc_vol', 'egarch_fc_vol',
                                   'mod_a', 'mod_b', 'mod_c'])
    rv = merged['rv_yz'].values
    print(f"  Evaluation window: {merged.index[0].date()} → {merged.index[-1].date()}")
    print(f"  N observations   : {len(merged)}")

    # ── Results ────────────────────────────────────────────────────────────────
    e_garch = merged['garch_fc_vol'].values - rv   # GARCH errors = DM reference

    models = [
        ('GARCH(1,1)',            merged['garch_fc_vol'].values,  False),
        ('EGARCH(1,1,1,t)',       merged['egarch_fc_vol'].values, True),
        ('Mod A: VIX-EGARCH',     merged['mod_a'].values,         True),
        ('Mod B: Realized-EGARCH',merged['mod_b'].values,         True),
        ('Mod C: GJR-GARCH',      merged['mod_c'].values,         True),
    ]

    rows = []
    for name, pred, do_dm in models:
        e = pred - rv
        dm_s, dm_p = dm_test(e, e_garch) if do_dm else ('—', '—')
        rows.append({
            'Model':   name,
            'RMSE':    round(rmse(pred, rv), 4),
            'MAE':     round(mae(pred, rv), 4),
            'QLIKE':   round(qlike(rv, pred), 4),
            'DM stat': dm_s,
            'DM p':    dm_p,
        })

    results = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print("  RESULTS — h=1, Yang-Zhang RV target  (lower RMSE/MAE/QLIKE = better)")
    print("=" * 72)
    print(results.to_string(index=False))
    print()

    garch_rmse   = results.loc[results['Model'] == 'GARCH(1,1)', 'RMSE'].values[0]
    egarch_rmse  = results.loc[results['Model'] == 'EGARCH(1,1,1,t)', 'RMSE'].values[0]
    winners_garch  = results[results['RMSE'] < garch_rmse]['Model'].tolist()
    winners_egarch = results[results['RMSE'] < egarch_rmse]['Model'].tolist()
    print(f"  Baseline GARCH  RMSE: {garch_rmse:.4f}")
    print(f"  Baseline EGARCH RMSE: {egarch_rmse:.4f}")
    print(f"  Models beating GARCH : {winners_garch  or ['None']}")
    print(f"  Models beating EGARCH: {winners_egarch or ['None']}")

    # ── Stability check ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  STABILITY CHECKS")
    print("=" * 72)
    for name, col in [('Mod A', 'mod_a'), ('Mod B', 'mod_b'), ('Mod C', 'mod_c')]:
        s = merged[col].values
        print(f"  {name}: NaN={np.isnan(s).sum()}  Inf={np.isinf(s).sum()}"
              f"  min={s.min():.3f}  max={s.max():.3f}"
              f"  mean={s.mean():.3f}  MaxJump={np.max(np.abs(np.diff(s))):.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    results.to_csv('data/egarch_mod_results.csv', index=False)
    merged.to_csv('data/egarch_mod_forecasts.csv')
    print("\n  Saved: data/egarch_mod_results.csv")
    print("  Saved: data/egarch_mod_forecasts.csv")

    # ── Plot ───────────────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    COLS = {
        'GARCH(1,1)':             ('steelblue',   '--', 0.7),
        'EGARCH(1,1,1,t)':        ('slategray',   '--', 0.8),
        'Mod A: VIX-EGARCH':      ('crimson',     '-',  1.0),
        'Mod B: Realized-EGARCH': ('darkorange',  '-',  1.0),
        'Mod C: GJR-GARCH':       ('forestgreen', '-',  1.0),
    }
    MAP_COL = {
        'GARCH(1,1)':             'garch_fc_vol',
        'EGARCH(1,1,1,t)':        'egarch_fc_vol',
        'Mod A: VIX-EGARCH':      'mod_a',
        'Mod B: Realized-EGARCH': 'mod_b',
        'Mod C: GJR-GARCH':       'mod_c',
    }

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    fig.suptitle("EGARCH Structural Modifications — Walk-Forward Comparison\n"
                 "Target: Yang-Zhang Realized Volatility (%/day)",
                 fontsize=13, fontweight='bold')

    # Panel 1: Forecast vs realized
    ax1.fill_between(merged.index, merged['rv_yz'], alpha=0.15, color='black', label='Realized Vol (YZ)')
    ax1.plot(merged.index, merged['rv_yz'], 'k-', linewidth=0.5, alpha=0.5)
    for lbl, (c, ls, lw) in COLS.items():
        ax1.plot(merged.index, merged[MAP_COL[lbl]], color=c, linestyle=ls,
                 linewidth=lw * 0.7, alpha=0.85, label=lbl)
    ax1.set_ylabel('Volatility (%/day)', fontsize=10)
    ax1.legend(fontsize=8, loc='upper right', ncol=3)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(0, None)

    # Panel 2: Absolute error
    for lbl, (c, ls, lw) in COLS.items():
        err = np.abs(merged[MAP_COL[lbl]].values - merged['rv_yz'].values)
        ax2.plot(merged.index, err, color=c, linestyle=ls, linewidth=lw * 0.6, alpha=0.75, label=lbl)
    ax2.set_ylabel('|Forecast − RV| (%/day)', fontsize=10)
    ax2.legend(fontsize=8, loc='upper right', ncol=3)
    ax2.grid(True, alpha=0.2)

    # Panel 3: 63-day rolling RMSE
    window = 63
    for lbl, (c, ls, lw) in COLS.items():
        err2      = (merged[MAP_COL[lbl]] - merged['rv_yz']) ** 2
        roll_rmse = err2.rolling(window).mean().apply(np.sqrt)
        ax3.plot(merged.index, roll_rmse, color=c, linestyle=ls,
                 linewidth=lw * 0.8, alpha=0.9, label=lbl)
    # Annotate final RMSE
    for _, row in results.iterrows():
        lbl = row['Model']
        if lbl in COLS:
            c = COLS[lbl][0]
            ax3.annotate(f"  {lbl}: {row['RMSE']:.4f}",
                         xy=(1.0, 0), xycoords='axes fraction',
                         fontsize=7, color=c,
                         xytext=(0, -15 * list(COLS).index(lbl)),
                         textcoords='offset points', ha='right')
    ax3.set_ylabel(f'Rolling {window}-day RMSE', fontsize=10)
    ax3.legend(fontsize=8, loc='upper right', ncol=3)
    ax3.grid(True, alpha=0.2)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig('plots/egarch_mod_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: plots/egarch_mod_comparison.png")

    print("\n" + "=" * 72)
    print("  DONE — review data/egarch_mod_results.csv for full results")
    print("=" * 72)


if __name__ == '__main__':
    main()
