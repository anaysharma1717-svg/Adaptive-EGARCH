"""
egarch_time_series_augmented.py
===============================
Testing whether time-series features (HAR, PACF lags) can improve EGARCH OOS.

Note: The Python `arch` library does not natively support external regressors (x) 
in the variance equation for EGARCH models, nor does it have a built-in Realized-EGARCH.
We will test:
1. Baseline EGARCH(1,1)
2. AR-EGARCH (AR lags in the mean equation based on PACF)
"""
import warnings
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

warnings.filterwarnings("ignore")

def fetch_data(ticker="SPY", start="2011-01-01"):
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw

def compute_features(df):
    log_ret = np.log(df['Close'] / df['Close'].shift(1)) * 100.0
    
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    rv = np.sqrt(np.maximum(gk_var, 1e-8)) * 100.0
    
    data = pd.DataFrame({'return': log_ret, 'rv': rv}).dropna()
    return data

def dm_test(actual_lst, pred1_lst, pred2_lst, h=1, crit="MSE", power=2):
    """
    Diebold-Mariano test for predictive accuracy.
    actual_lst: actual values (GK-RV)
    pred1_lst: predictions from model 1 (baseline)
    pred2_lst: predictions from model 2 (alternative)
    """
    import math
    actual = np.array(actual_lst)
    pred1 = np.array(pred1_lst)
    pred2 = np.array(pred2_lst)
    
    if crit == "MSE":
        e1 = (actual - pred1)**2
        e2 = (actual - pred2)**2
    elif crit == "MAE":
        e1 = np.abs(actual - pred1)
        e2 = np.abs(actual - pred2)
    elif crit == "QLIKE":
        e1 = actual**2 / pred1**2 - np.log(actual**2 / pred1**2) - 1
        e2 = actual**2 / pred2**2 - np.log(actual**2 / pred2**2) - 1
    
    d = e1 - e2
    mean_d = np.mean(d)
    
    gamma = []
    for lag in range(0, h):
        cov = np.cov(d[lag:], d[:len(d)-lag])[0,1] if lag > 0 else np.var(d)
        gamma.append(cov)
    
    var_d = gamma[0] + 2 * sum(gamma[1:])
    n = len(d)
    if var_d == 0:
        return 0, 1.0
        
    stat = mean_d / math.sqrt((var_d / n))
    
    # HLN Correction
    k = ((n + 1 - 2*h + (h/n)*(h-1)) / n) ** 0.5
    stat = stat * k
    
    p_value = 2 * (1 - stats.t.cdf(abs(stat), df=n-1))
    return stat, p_value

def evaluate_models():
    raw = fetch_data()
    data = compute_features(raw)
    
    # Match previous test dates
    test_months = 6
    split_date = data.index[-1] - pd.DateOffset(months=test_months)
    test_dates = data[data.index > split_date].index
    
    print(f"Test period: {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")
    
    baseline_fc = []
    ar_fc = []
    actuals = []
    
    # Fit initial parameters on train to use as starting values (speedup)
    train_init = data[data.index <= split_date]
    ret_train = train_init['return']
    
    am_base = arch_model(ret_train, vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
    res_base_init = am_base.fit(disp="off")
    
    am_ar = arch_model(ret_train, mean='AR', lags=[1, 2], vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
    res_ar_init = am_ar.fit(disp="off")

    base_params = res_base_init.params.values
    ar_params = res_ar_init.params.values
    
    print("Running Walk-Forward Out-Of-Sample Testing...")
    
    for i, tdate in enumerate(test_dates):
        if i % 10 == 0:
            print(f"  Processing day {i}/{len(test_dates)}...")
        train = data[data.index < tdate]
        y = train['return']
        
        # We can use the fast expanding window approach with `update` if we wanted, 
        # but refitting every N days or daily is safer. 
        # For speed, we will fit on full data up to tdate using previous params.
        
        am_b = arch_model(y, vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
        res_b = am_b.fit(starting_values=base_params, disp="off", show_warning=False)
        fc_b = res_b.forecast(horizon=1, method="simulation", simulations=2000, reindex=False)
        baseline_fc.append(np.sqrt(fc_b.variance.iloc[-1, 0]))
        base_params = res_b.params.values
        
        am_a = arch_model(y, mean='AR', lags=[1, 2], vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
        res_a = am_a.fit(starting_values=ar_params, disp="off", show_warning=False)
        fc_a = res_a.forecast(horizon=1, method="simulation", simulations=2000, reindex=False)
        ar_fc.append(np.sqrt(fc_a.variance.iloc[-1, 0]))
        ar_params = res_a.params.values
        
        actuals.append(data.loc[tdate, 'rv'])
        
    actuals = np.array(actuals)
    base_fc = np.array(baseline_fc)
    ar_fc = np.array(ar_fc)
    
    def qlike(y, yhat):
        return np.mean(y**2 / yhat**2 - np.log(y**2 / yhat**2) - 1)
        
    print("\nRESULTS")
    print("="*40)
    print("Baseline EGARCH(1,1):")
    print(f"  RMSE  : {np.sqrt(np.mean((actuals - base_fc)**2)):.4f}")
    print(f"  MAE   : {np.mean(np.abs(actuals - base_fc)):.4f}")
    print(f"  QLIKE : {qlike(actuals, base_fc):.4f}")
    
    print("\nAR(2)-EGARCH(1,1):")
    print(f"  RMSE  : {np.sqrt(np.mean((actuals - ar_fc)**2)):.4f}")
    print(f"  MAE   : {np.mean(np.abs(actuals - ar_fc)):.4f}")
    print(f"  QLIKE : {qlike(actuals, ar_fc):.4f}")
    
    dm_stat, dm_p = dm_test(actuals, base_fc, ar_fc, crit="MSE")
    print("\nDiebold-Mariano Test (AR-EGARCH vs Baseline, MSE):")
    print(f"  DM Stat: {dm_stat:.4f}, p-value: {dm_p:.4f}")
    if dm_p < 0.05 and dm_stat < 0:
        print("  -> AR-EGARCH is significantly better")
    elif dm_p < 0.05 and dm_stat > 0:
        print("  -> Baseline is significantly better")
    else:
        print("  -> No significant difference")
        
    # Residual Diagnostics for the final AR-EGARCH model
    print("\nFinal Model Residual Diagnostics (AR-EGARCH):")
    std_resid = res_a.std_resid.dropna()
    lb = acorr_ljungbox(std_resid, lags=[10, 20], return_df=True)
    print("  Ljung-Box (Std Resid, check mean eqn):")
    print(lb)
    
    lb_sq = acorr_ljungbox(std_resid**2, lags=[10, 20], return_df=True)
    print("  Ljung-Box (Std Resid Sq, check var eqn):")
    print(lb_sq)
    
    arch_test = het_arch(std_resid, nlags=10)
    print(f"  ARCH-LM Test: stat={arch_test[0]:.4f}, p={arch_test[1]:.4f}")

if __name__ == "__main__":
    evaluate_models()
