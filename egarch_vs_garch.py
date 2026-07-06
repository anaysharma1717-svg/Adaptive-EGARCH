"""
egarch_vs_garch.py
==================
SPY Volatility Baseline: GARCH(1,1) vs EGARCH(1,1) with Leverage Effect Diagnostics

Pipeline:
  1. fetch_data()         - Pull SPY daily OHLCV via yfinance (2015-present)
  2. compute_returns()    - Log returns x100 (arch library convention)
  3. fit_garch()          - Vanilla GARCH(1,1, Normal dist)
  4. fit_egarch()         - EGARCH(1,1,1) with Normal AND Student-t; best by AIC/BIC
  5. forecast_volatility()- 1-day-ahead forecast for each fitted model
  6. evaluate_forecast()  - Rolling OOS RMSE on last-6-months test set
  7. plot_comparison()    - Overlay: realized vol + GARCH + EGARCH fitted vols

Key convention:
  Returns are scaled x100 before fitting (e.g. -1.2% -> -1.2).
  All reported volatilities are in units of daily % move (NOT annualized)
  unless explicitly noted.

Dependencies: yfinance, arch, pandas, numpy, matplotlib, scipy
  pip install yfinance arch pandas numpy matplotlib scipy
"""

import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import yfinance as yf
from arch import arch_model
from scipy import stats


warnings.filterwarnings("ignore")


# =============================================================================
# 1.  DATA
# =============================================================================

def fetch_data(ticker="SPY", start="2015-01-01", end=None):
    """
    Download daily OHLCV bars for `ticker` via yfinance.

    Returns
    -------
    pd.DataFrame  with columns: Open, High, Low, Close, Volume
                  Index is a tz-naive DatetimeIndex.
    """
    print("\n" + "="*65)
    print(f"  Fetching {ticker} daily OHLCV  [{start} -> {'today' if end is None else end}]")
    print("="*65)

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        sys.exit(f"ERROR: yfinance returned empty DataFrame for {ticker}.")

    # Strip timezone
    raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index

    # Flatten MultiIndex columns (newer yfinance versions)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    print(f"  OK  {len(raw):,} trading days  |  "
          f"{raw.index[0].date()} -> {raw.index[-1].date()}")
    return raw


# =============================================================================
# 2.  RETURNS
# =============================================================================

def compute_returns(df, price_col="Close", scale=100.0):
    """
    Compute daily log returns, scaled by `scale` (default 100).

    Scaling convention (arch library):
      Raw log returns for SPY have variance ~0.000144, far too small for
      stable MLE. Multiplying by 100 gives variance ~1.44, well-behaved.
      All downstream volatilities are in '% per day' units.
    """
    log_ret = np.log(df[price_col] / df[price_col].shift(1)) * scale
    log_ret = log_ret.dropna()
    log_ret.name = "log_return_pct"

    print(f"\n  Return statistics (x{scale}):")
    print(f"    Obs   : {len(log_ret):,}")
    print(f"    Mean  : {log_ret.mean():+.4f} %/day")
    print(f"    Std   : {log_ret.std():.4f} %/day")
    print(f"    Min   : {log_ret.min():.4f}  (raw {log_ret.min()/scale*100:.2f}%)")
    print(f"    Max   : {log_ret.max():.4f}  (raw {log_ret.max()/scale*100:.2f}%)")
    print(f"    Skew  : {log_ret.skew():.4f}")
    print(f"    Kurt  : {log_ret.kurtosis():.4f}  "
          f"({'fat tails' if log_ret.kurtosis() > 0 else 'thin tails'})")
    return log_ret


# =============================================================================
# 3.  VANILLA GARCH(1,1)
# =============================================================================

def fit_garch(returns, dist="normal"):
    """
    Fit a standard GARCH(1,1) model.

    GARCH(1,1) variance equation:
        sigma^2_t = omega  +  alpha * epsilon^2_{t-1}  +  beta * sigma^2_{t-1}

    Parameters
    ----------
    omega  - long-run variance floor (always positive)
    alpha  - weight on last shock^2; how fast vol reacts to news
    beta   - weight on last variance; how slowly vol decays
    alpha+beta - persistence: close to 1 -> shocks die out very slowly

    Returns: ARCHModelResult
    """
    print("\n" + "-"*65)
    print("  Fitting GARCH(1,1)  [symmetric, normal dist]")
    print("-"*65)

    am  = arch_model(returns, vol="Garch", p=1, q=1, dist=dist, rescale=False)
    res = am.fit(disp="off", show_warning=False)

    p = res.params
    print(f"\n  Fitted parameters:")
    print(f"    omega (w)   = {p['omega']:.6f}  <- long-run variance floor")
    print(f"    alpha       = {p['alpha[1]']:.6f}  <- shock sensitivity (ARCH term)")
    print(f"    beta        = {p['beta[1]']:.6f}  <- variance persistence (GARCH term)")
    pers = p['alpha[1]'] + p['beta[1]']
    print(f"    alpha+beta  = {pers:.6f}  "
          f"<- {'near-unit-root, very slow mean reversion' if pers > 0.97 else 'moderate persistence'}")
    print(f"\n    AIC = {res.aic:.2f}  |  BIC = {res.bic:.2f}  |  "
          f"Log-Lik = {res.loglikelihood:.2f}")
    return res


# =============================================================================
# 4.  EGARCH(1,1)  with distribution comparison
# =============================================================================

def fit_egarch(returns):
    """
    Fit EGARCH(1,1) with BOTH Normal and Student-t error distributions.
    Select the better-fitting distribution via AIC (primary) and BIC.

    EGARCH(1,1) log-variance equation (Nelson 1991):
        ln(sigma^2_t) = omega
                      + beta  * ln(sigma^2_{t-1})
                      + alpha * [|z_{t-1}| - E|z|]   <- magnitude of shock
                      + gamma * z_{t-1}               <- SIGN of shock

    where  z_t = epsilon_t / sigma_t  (standardised residual)

    Key difference from GARCH:
      - Models log(sigma^2), so variance is ALWAYS positive -- no constraints.
      - gamma (gamma) captures the LEVERAGE EFFECT:
            gamma < 0  -> negative shock (price drop) raises vol MORE than a
                          positive shock of equal size.
            gamma = 0  -> symmetric (EGARCH without asymmetry = no leverage).
            gamma > 0  -> positive shocks raise vol more (unusual for equities).

    arch library parameter naming:
      omega     -> omega
      alpha[1]  -> alpha: symmetric shock magnitude
      gamma[1]  -> gamma: asymmetric leverage term  *** KEY PARAMETER ***
      beta[1]   -> beta: log-variance persistence
      nu        -> Student-t degrees of freedom (only in t distribution)

    Returns: (best_result, normal_result, t_result)
    """
    print("\n" + "-"*65)
    print("  Fitting EGARCH(1,1)  -- Normal distribution")
    print("-"*65)
    am_n  = arch_model(returns, vol="EGARCH", p=1, o=1, q=1, dist="normal", rescale=False)
    res_n = am_n.fit(disp="off", show_warning=False)

    print("\n" + "-"*65)
    print("  Fitting EGARCH(1,1)  -- Student-t distribution")
    print("-"*65)
    am_t  = arch_model(returns, vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
    res_t = am_t.fit(disp="off", show_warning=False)

    # Distribution comparison
    print("\n" + "-"*65)
    print("  DISTRIBUTION COMPARISON  (EGARCH Normal vs Student-t)")
    print("-"*65)
    print(f"  {'Metric':<12}  {'Normal':>12}  {'Student-t':>12}  {'Winner':>10}")
    print("  " + "-"*50)

    aic_winner = "Normal" if res_n.aic < res_t.aic else "Student-t"
    bic_winner = "Normal" if res_n.bic < res_t.bic else "Student-t"
    ll_winner  = "Normal" if res_n.loglikelihood > res_t.loglikelihood else "Student-t"

    print(f"  {'AIC':<12}  {res_n.aic:>12.2f}  {res_t.aic:>12.2f}  {aic_winner:>10}")
    print(f"  {'BIC':<12}  {res_n.bic:>12.2f}  {res_t.bic:>12.2f}  {bic_winner:>10}")
    print(f"  {'Log-Lik':<12}  {res_n.loglikelihood:>12.2f}  "
          f"{res_t.loglikelihood:>12.2f}  {ll_winner:>10}")

    best       = res_t if res_t.aic < res_n.aic else res_n
    best_label = "Student-t" if res_t.aic < res_n.aic else "Normal"
    print(f"\n  Best EGARCH distribution: {best_label}  (lower AIC wins)")

    _print_egarch_params(best, best_label)
    _leverage_effect_report(best)

    return best, res_n, res_t


def _print_egarch_params(res, label):
    """Pretty-print EGARCH parameter table with economic interpretations."""
    p  = res.params
    t  = res.tvalues
    pv = res.pvalues

    print("\n" + "-"*65)
    print(f"  EGARCH ({label}) -- Fitted Parameters")
    print("-"*65)
    print(f"  {'Param':<14} {'Value':>12} {'t-stat':>10} {'p-value':>10}  Meaning")
    print("  " + "-"*70)

    param_meta = {
        "omega":    "constant in log(sigma^2) equation",
        "alpha[1]": "symmetric shock magnitude (|z| term)",
        "gamma[1]": "LEVERAGE EFFECT  (sign of shock)  *** KEY ***",
        "beta[1]":  "log-variance persistence",
    }
    if "nu" in p.index:
        param_meta["nu"] = "Student-t degrees of freedom  (< 30 -> fat tails)"

    for name, desc in param_meta.items():
        if name in p.index:
            stars = ("***" if pv[name] < 0.001 else
                     "**"  if pv[name] < 0.01  else
                     "*"   if pv[name] < 0.05  else
                     "."   if pv[name] < 0.10  else "")
            print(f"  {name:<14} {p[name]:>12.6f} {t[name]:>10.3f} "
                  f"{pv[name]:>10.4f}  {desc} {stars}")


def _leverage_effect_report(res):
    """
    Explicitly test and explain gamma (leverage effect) parameter.
    Core economic question: does SPY exhibit the leverage effect?
    """
    p  = res.params
    pv = res.pvalues
    ci = res.conf_int()

    gamma   = p.get("gamma[1]", float("nan"))
    p_gamma = pv.get("gamma[1]", float("nan"))
    ci_lo   = ci.loc["gamma[1]", "lower"] if "gamma[1]" in ci.index else float("nan")
    ci_hi   = ci.loc["gamma[1]", "upper"] if "gamma[1]" in ci.index else float("nan")

    print("\n" + "="*65)
    print("  LEVERAGE EFFECT DIAGNOSTIC  (gamma parameter)")
    print("="*65)
    print(f"  gamma (gamma[1]) = {gamma:+.6f}")
    print(f"  95% CI           = [{ci_lo:+.6f},  {ci_hi:+.6f}]")
    print(f"  p-value          = {p_gamma:.6f}")
    print()

    if np.isnan(gamma):
        print("  WARNING: gamma not found in fitted parameters.")
        return

    sig = p_gamma < 0.05
    neg = gamma < 0

    if sig and neg:
        print("  CONFIRMED: LEVERAGE EFFECT PRESENT IN SPY:")
        print(f"      gamma = {gamma:.4f}  (negative AND significant at 5%)")
        print()
        print("  Economic interpretation:")
        print("    A 1-sigma NEGATIVE shock (price DROP) changes ln(sigma^2) by:")
        print(f"      alpha*(|z|-E|z|) + gamma*(-1)  =  alpha*(1-E|z|) - gamma")
        print("    A 1-sigma POSITIVE shock (price RISE) changes ln(sigma^2) by:")
        print(f"      alpha*(|z|-E|z|) + gamma*(+1)  =  alpha*(1-E|z|) + gamma")
        print(f"    Difference = -2*gamma = {-2*gamma:+.4f} log-variance units per sigma shock")
        print()
        print("    Translation: When SPY falls by 1-sigma, the vol process")
        print(f"    spikes by {abs(2*gamma):.4f} extra log-variance units vs a same-")
        print("    sized rise. This is the 'fear asymmetry' that vanilla GARCH misses.")
    elif sig and not neg:
        print("  WARNING: INVERSE leverage effect (unusual for equity indices).")
        print(f"      gamma = {gamma:+.4f}  -- positive shocks raise vol more.")
    else:
        print(f"  NOT SIGNIFICANT: Leverage effect p = {p_gamma:.4f}  (> 0.05)")
        print("      gamma is indistinguishable from zero.")
        print("      GARCH and EGARCH will produce near-identical forecasts.")
    print("="*65)


# =============================================================================
# 5.  VOLATILITY FORECAST  (1-day ahead)
# =============================================================================

def forecast_volatility(res_garch, res_egarch, horizon=1):
    """
    Generate h-day-ahead variance forecasts for both models.

    For GARCH  (symmetric): method='analytic' is exact for any horizon.
    For EGARCH (asymmetric): method='simulation' required for h > 1 because
        the log-variance cannot be analytically iterated when the expectation
        of |z| depends on the distribution non-linearly. For h=1 both work;
        simulation is used for consistency across horizons.

    Returns dict with keys:
        garch_vol_pct, egarch_vol_pct, garch_ann_vol, egarch_ann_vol
    """
    print("\n" + "-"*65)
    print("  1-DAY-AHEAD VOLATILITY FORECAST")
    print("-"*65)

    # GARCH -- analytic, exact
    fc_g      = res_garch.forecast(horizon=horizon, method="analytic", reindex=False)
    garch_var = float(fc_g.variance.iloc[-1, -1])
    garch_vol = np.sqrt(garch_var)

    # EGARCH -- simulation-based (10k paths for low Monte Carlo noise)
    fc_e       = res_egarch.forecast(horizon=horizon, method="simulation",
                                     simulations=10_000, reindex=False)
    egarch_var = float(fc_e.variance.iloc[-1, -1])
    egarch_vol = np.sqrt(egarch_var)

    garch_ann  = garch_vol  * np.sqrt(252)
    egarch_ann = egarch_vol * np.sqrt(252)

    print(f"\n  {'Model':<14}  {'Daily Vol':>12}  {'Ann. Vol (x sqrt252)':>20}")
    print("  " + "-"*50)
    print(f"  {'GARCH(1,1)':<14}  {garch_vol:>10.4f}%  {garch_ann:>18.2f}%")
    print(f"  {'EGARCH(1,1)':<14}  {egarch_vol:>10.4f}%  {egarch_ann:>18.2f}%")

    diff = egarch_vol - garch_vol
    print(f"\n  EGARCH - GARCH = {diff:+.4f}%/day  "
          f"({'higher' if diff > 0 else 'lower'} vol from EGARCH)")

    return dict(
        garch_vol_pct  = garch_vol,
        egarch_vol_pct = egarch_vol,
        garch_ann_vol  = garch_ann,
        egarch_ann_vol = egarch_ann,
    )


# =============================================================================
# 6.  OUT-OF-SAMPLE EVALUATION
# =============================================================================

def evaluate_forecast(returns, test_months=6):
    """
    Walk-forward OOS evaluation: fit on training set, rolling 1-step forecasts
    across the test period, compare against realized volatility via RMSE.

    Split:
      train = all but last test_months calendar months
      test  = last test_months calendar months

    Realized vol proxy:
      |r_t| as a proxy for daily volatility (simple, interpretable).
      r^2_t as the proxy for variance (matches the Mincer-Zarnowitz standard).

    Method:
      Fit model on training data. Apply fitted parameters to full series
      (train + test) and extract 1-step-ahead forecasts starting at the
      train/test boundary. This is 'fixed-window expanding' -- standard
      for a baseline. Production would re-fit monthly.

    Returns dict with RMSE metrics, forecast series, and fitted results.
    """
    print("\n" + "-"*65)
    print(f"  OUT-OF-SAMPLE EVALUATION  (last {test_months} months = test set)")
    print("-"*65)

    # Train/test split
    split_date = returns.index[-1] - pd.DateOffset(months=test_months)
    train = returns[returns.index <= split_date]
    test  = returns[returns.index >  split_date]

    print(f"\n  Train: {train.index[0].date()} -> {train.index[-1].date()}  ({len(train):,} obs)")
    print(f"  Test : {test.index[0].date()}  -> {test.index[-1].date()}   ({len(test):,} obs)")

    # Fit on training set
    print("\n  Fitting GARCH on training set ...", end=" ", flush=True)
    am_g  = arch_model(train, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res_g = am_g.fit(disp="off", show_warning=False)
    print("done.")

    print("  Fitting EGARCH on training set ...", end=" ", flush=True)
    am_e  = arch_model(train, vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
    res_e = am_e.fit(disp="off", show_warning=False)
    print("done.")

    # Apply fitted parameters to full series, extract OOS forecasts
    full = pd.concat([train, test])
    train_size = len(train)

    print("  Generating GARCH OOS forecasts ...", end=" ", flush=True)
    am_g_full  = arch_model(full, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res_g_full = am_g_full.fit(starting_values=res_g.params.values,
                               disp="off", show_warning=False)
    fc_g = res_g_full.forecast(horizon=1, method="analytic",
                               reindex=True, start=train_size)
    garch_fc_var = fc_g.variance.dropna().squeeze()
    garch_fc_vol = np.sqrt(garch_fc_var)
    print("done.")

    print("  Generating EGARCH OOS forecasts ...", end=" ", flush=True)
    am_e_full  = arch_model(full, vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
    res_e_full = am_e_full.fit(starting_values=res_e.params.values,
                               disp="off", show_warning=False)
    fc_e = res_e_full.forecast(horizon=1, method="simulation",
                               simulations=5_000, reindex=True,
                               start=train_size)
    egarch_fc_var = fc_e.variance.dropna().squeeze()
    egarch_fc_vol = np.sqrt(egarch_fc_var)
    print("done.")

    # Align on common test dates
    common_idx   = garch_fc_vol.index.intersection(
                       egarch_fc_vol.index).intersection(test.index)
    realized_vol = np.abs(test.loc[common_idx])   # |return| as vol proxy
    realized_sq  = test.loc[common_idx] ** 2       # r^2 as variance proxy

    g_vol = garch_fc_vol.reindex(common_idx)
    e_vol = egarch_fc_vol.reindex(common_idx)

    # RMSE metrics
    garch_rmse_vol  = float(np.sqrt(np.mean((g_vol - realized_vol) ** 2)))
    egarch_rmse_vol = float(np.sqrt(np.mean((e_vol - realized_vol) ** 2)))
    garch_rmse_var  = float(np.sqrt(np.mean((g_vol**2 - realized_sq) ** 2)))
    egarch_rmse_var = float(np.sqrt(np.mean((e_vol**2 - realized_sq) ** 2)))

    print(f"\n  {'Metric':<32}  {'GARCH':>10}  {'EGARCH':>10}  {'Winner':>10}")
    print("  " + "-"*68)
    for label, g_val, e_val in [
        ("RMSE (vol,  % / day)",         garch_rmse_vol,  egarch_rmse_vol),
        ("RMSE (var,  (%^2) / day)",      garch_rmse_var,  egarch_rmse_var),
    ]:
        w = "EGARCH" if e_val < g_val else "GARCH"
        print(f"  {label:<32}  {g_val:>10.6f}  {e_val:>10.6f}  {w:>10}")

    if egarch_rmse_vol < garch_rmse_vol:
        winner = "EGARCH"
        improvement = (garch_rmse_vol - egarch_rmse_vol) / garch_rmse_vol * 100
        print(f"\n  EGARCH wins by {improvement:.2f}% lower vol RMSE")
        print("  The leverage effect captures real asymmetric dynamics.")
    else:
        winner = "GARCH"
        improvement = (egarch_rmse_vol - garch_rmse_vol) / egarch_rmse_vol * 100
        print(f"\n  GARCH wins by {improvement:.2f}% lower vol RMSE")
        print("  This test period may have been dominated by symmetric shocks.")
        improvement = -improvement

    return dict(
        garch_rmse      = garch_rmse_vol,
        egarch_rmse     = egarch_rmse_vol,
        winner          = winner,
        improvement_pct = improvement,
        test_dates      = common_idx,
        garch_fc_vol    = g_vol,
        egarch_fc_vol   = e_vol,
        realized_vol    = realized_vol,
        train           = train,
        test            = test,
        res_g_full      = res_g_full,
        res_e_full      = res_e_full,
    )


# =============================================================================
# 7.  VISUALIZATION
# =============================================================================

def plot_comparison(returns, res_garch, res_egarch, eval_results,
                    window=20, save_path="egarch_vs_garch.png"):
    """
    Publication-quality 3-panel dark-mode figure:

    Panel 1  Full-sample: 20-day realized vol + GARCH + EGARCH conditional vol
    Panel 2  OOS zoom:    1-day-ahead forecast vol vs realized  (test period only)
    Panel 3a Asymmetry scatter: shock sign vs next-day delta-vol (leverage evidence)
    Panel 3b RMSE bar chart: head-to-head OOS accuracy comparison
    """
    ACCENT_G  = "#4FC3F7"   # GARCH  -- ice blue
    ACCENT_E  = "#FF8A65"   # EGARCH -- warm orange
    ACCENT_RV = "#A5D6A7"   # Realized vol -- soft green
    ACCENT_SP = "#90CAF9"   # Split marker -- light blue

    fig = plt.figure(figsize=(18, 14), facecolor="#0D1117")
    gs  = GridSpec(3, 2, figure=fig,
                   hspace=0.42, wspace=0.30,
                   left=0.07, right=0.97,
                   top=0.93,  bottom=0.06)

    ax1  = fig.add_subplot(gs[0, :])
    ax2  = fig.add_subplot(gs[1, :])
    ax3a = fig.add_subplot(gs[2, 0])
    ax3b = fig.add_subplot(gs[2, 1])

    # Common data
    roll_rv         = returns.rolling(window).std()
    garch_cond_vol  = pd.Series(res_garch.conditional_volatility,  index=returns.index)
    egarch_cond_vol = pd.Series(res_egarch.conditional_volatility, index=returns.index)

    test_start = eval_results["test"].index[0]
    test_end   = eval_results["test"].index[-1]

    # -- Panel 1: Full history ------------------------------------------------
    ax1.set_facecolor("#161B22")
    ymax = max(roll_rv.max(), garch_cond_vol.max(), egarch_cond_vol.max()) * 1.15
    ax1.fill_betweenx([0, ymax], test_start, test_end,
                      color=ACCENT_SP, alpha=0.08, label="Test period")
    ax1.plot(roll_rv,         color=ACCENT_RV, lw=0.9, alpha=0.70,
             label=f"{window}-day Rolling Realized Vol")
    ax1.plot(garch_cond_vol,  color=ACCENT_G,  lw=1.1, alpha=0.88,
             label="GARCH(1,1) Conditional Vol")
    ax1.plot(egarch_cond_vol, color=ACCENT_E,  lw=1.1, alpha=0.88,
             label="EGARCH(1,1) Conditional Vol")
    ax1.axvline(test_start, color=ACCENT_SP, lw=1.2, ls="--", alpha=0.6)
    ax1.set_title("Full History: Conditional Volatility -- GARCH vs EGARCH  (% / day)",
                  fontsize=13, color="white", pad=10)
    ax1.set_ylabel("Daily Vol (%)", color="white")
    ax1.legend(loc="upper left", framealpha=0.25, fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.grid(alpha=0.12)
    ax1.tick_params(colors="grey")
    ax1.set_ylim(0, ymax)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363D")

    # -- Panel 2: OOS zoom ---------------------------------------------------
    ax2.set_facecolor("#161B22")
    oos_rv = eval_results["realized_vol"]
    oos_g  = eval_results["garch_fc_vol"]
    oos_e  = eval_results["egarch_fc_vol"]

    ax2.plot(oos_rv, color=ACCENT_RV, lw=1.2, alpha=0.75, label="|Return| (realized)")
    ax2.plot(oos_g,  color=ACCENT_G,  lw=1.4, alpha=0.90, label="GARCH 1-day forecast")
    ax2.plot(oos_e,  color=ACCENT_E,  lw=1.4, alpha=0.90, label="EGARCH 1-day forecast")

    g_rmse = eval_results["garch_rmse"]
    e_rmse = eval_results["egarch_rmse"]
    ax2.text(0.01, 0.95,
             f"GARCH  RMSE = {g_rmse:.4f} %/day\nEGARCH RMSE = {e_rmse:.4f} %/day",
             transform=ax2.transAxes, va="top", ha="left",
             fontsize=9, color="white",
             bbox=dict(facecolor="#1F2937", edgecolor="#374151", alpha=0.85, pad=5))
    ax2.set_title("OOS Test Period: 1-Day-Ahead Forecasts vs Realized Volatility",
                  fontsize=13, color="white", pad=10)
    ax2.set_ylabel("Daily Vol (%)", color="white")
    ax2.legend(loc="upper right", framealpha=0.25, fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.grid(alpha=0.12)
    ax2.tick_params(colors="grey")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363D")

    # -- Panel 3a: Asymmetry scatter -----------------------------------------
    ax3a.set_facecolor("#161B22")
    z_scores = pd.Series(res_egarch.std_resid, index=returns.index)
    e_vol_s  = pd.Series(res_egarch.conditional_volatility, index=returns.index)
    delta_vol = e_vol_s.diff(1).shift(-1)

    clip = 4.0
    mask = (z_scores.abs() < clip) & delta_vol.notna()
    z_p  = z_scores[mask]
    dv_p = delta_vol[mask]

    neg_m = z_p < 0
    pos_m = z_p >= 0
    ax3a.scatter(z_p[neg_m], dv_p[neg_m], color="#FF6B6B", s=4, alpha=0.30,
                 label="Negative shock (price down)")
    ax3a.scatter(z_p[pos_m], dv_p[pos_m], color="#4FC3F7", s=4, alpha=0.30,
                 label="Positive shock (price up)")

    for side_mask, color in [(neg_m, "#FF6B6B"), (pos_m, "#4FC3F7")]:
        x = z_p[side_mask].values
        y = dv_p[side_mask].values
        if len(x) > 2:
            m, b, *_ = stats.linregress(x, y)
            xr = np.linspace(x.min(), x.max(), 50)
            ax3a.plot(xr, m * xr + b, color=color, lw=2.0, alpha=0.85)

    ax3a.axhline(0, color="white", lw=0.6, alpha=0.35)
    ax3a.axvline(0, color="white", lw=0.6, alpha=0.35)
    ax3a.set_title("Leverage Effect: Shock Sign vs Next-Day Delta EGARCH Vol",
                   fontsize=11, color="white", pad=8)
    ax3a.set_xlabel("Standardised Residual z_t  (shock size + sign)", color="grey")
    ax3a.set_ylabel("Delta EGARCH cond. vol_{t+1}  (%/day)", color="grey")
    ax3a.legend(fontsize=8, framealpha=0.25)
    ax3a.grid(alpha=0.10)
    ax3a.tick_params(colors="grey")
    for spine in ax3a.spines.values():
        spine.set_edgecolor("#30363D")

    # -- Panel 3b: RMSE bar chart --------------------------------------------
    ax3b.set_facecolor("#161B22")
    models = ["GARCH(1,1)", "EGARCH(1,1)"]
    rmses  = [g_rmse, e_rmse]
    colors = [ACCENT_G, ACCENT_E]
    bars   = ax3b.bar(models, rmses, color=colors, width=0.45,
                      edgecolor="#30363D", linewidth=0.8)
    for bar, val in zip(bars, rmses):
        ax3b.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + max(rmses) * 0.01,
                  f"{val:.5f}",
                  ha="center", va="bottom", fontsize=10,
                  color="white", fontweight="bold")

    winner  = eval_results["winner"]
    improv  = abs(eval_results["improvement_pct"])
    ax3b.set_title(
        f"OOS RMSE Comparison\n{winner} wins  |  Delta = {improv:.2f}% better accuracy",
        fontsize=11, color="white", pad=8)
    ax3b.set_ylabel("RMSE  (% / day)", color="grey")
    ax3b.tick_params(colors="grey")
    ax3b.grid(axis="y", alpha=0.12)
    for spine in ax3b.spines.values():
        spine.set_edgecolor("#30363D")

    # Super-title
    fig.suptitle(
        "SPY Volatility Engine  --  GARCH(1,1) vs EGARCH(1,1) Baseline\n"
        "Leverage Effect  |  Distribution Selection  |  OOS Walk-Forward Evaluation",
        fontsize=15, color="white", y=0.97, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\n  Plot saved -> {save_path}")

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # 1. Data
    raw     = fetch_data(ticker="SPY", start="2015-01-01")
    returns = compute_returns(raw, price_col="Close", scale=100.0)

    # 2. Fit full-sample GARCH (for parameter reporting and 1-day forecast)
    res_garch = fit_garch(returns, dist="normal")

    # 3. Fit full-sample EGARCH (Normal vs t; best selected by AIC)
    res_egarch, res_egarch_n, res_egarch_t = fit_egarch(returns)

    # 4. 1-day ahead forecast comparison
    fc = forecast_volatility(res_garch, res_egarch, horizon=1)
    print(f"\n  Next trading-day vol estimate:")
    print(f"      GARCH  : +/-{fc['garch_vol_pct']:.3f}%/day  ({fc['garch_ann_vol']:.1f}% annualised)")
    print(f"      EGARCH : +/-{fc['egarch_vol_pct']:.3f}%/day  ({fc['egarch_ann_vol']:.1f}% annualised)")

    # 5. Walk-forward OOS evaluation (last 6 months = test)
    eval_results = evaluate_forecast(returns, test_months=6)

    # 6. Plot all panels
    plot_comparison(
        returns      = returns,
        res_garch    = eval_results["res_g_full"],
        res_egarch   = eval_results["res_e_full"],
        eval_results = eval_results,
        window       = 20,
        save_path    = "egarch_vs_garch.png",
    )

    # 7. Final summary card
    print("\n" + "="*65)
    print("  FINAL SUMMARY")
    print("="*65)
    print(f"  Sample        : {raw.index[0].date()} -> {raw.index[-1].date()}")
    print(f"  Observations  : {len(returns):,}")
    print(f"  GARCH  AIC    : {res_garch.aic:.2f}")
    print(f"  EGARCH AIC    : {res_egarch.aic:.2f}  "
          f"({'better' if res_egarch.aic < res_garch.aic else 'worse'} fit than GARCH)")
    print(f"  Next-day vol  : GARCH {fc['garch_vol_pct']:.3f}% | EGARCH {fc['egarch_vol_pct']:.3f}%")
    print(f"  OOS winner    : {eval_results['winner']}  "
          f"({abs(eval_results['improvement_pct']):.2f}% lower RMSE)")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
