"""
run_pipeline.py
===============
Production Pipeline for EGARCH(1,1) Volatility Forecasting (SPY).
Performs walk-forward out-of-sample evaluation without look-ahead bias,
and generates 21-day annualized volatility forecasts for QuantConnect deployment.

Dependencies: yfinance, arch, pandas, numpy, scipy
"""
import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_data(ticker="SPY", start="2016-01-01"):
    logger.info(f"Fetching {ticker} data from {start}...")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        logger.error("No data fetched.")
        sys.exit(1)
        
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def prepare_features(df):
    logger.info("Computing Garman-Klass RV and Returns...")
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    rv = np.sqrt(np.maximum(gk_var, 1e-8)) * 100.0  # Daily % volatility
    
    returns = np.log(df['Close'] / df['Close'].shift(1)) * 100.0
    
    data = pd.DataFrame({'return': returns, 'rv': rv}).dropna()
    return data

def walk_forward_eval(data, test_months=6, refit_freq=21, h=21):
    """
    Strict out-of-sample walk-forward EGARCH evaluation.
    Refits the model every `refit_freq` days to prevent look-ahead bias,
    and forecasts `h` days ahead.
    """
    logger.info(f"Starting Walk-Forward OOS Evaluation (Test Months: {test_months})...")
    split_date = data.index[-1] - pd.DateOffset(months=test_months)
    
    train_data = data[data.index <= split_date]
    test_data = data[data.index > split_date]
    test_dates = test_data.index
    
    logger.info(f"Train period: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} obs)")
    logger.info(f"Test period:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} obs)")
    
    forecasts = {}
    last_params = None
    
    for i, tdate in enumerate(test_dates):
        # We only fit on data STRICTLY before the current date tdate
        history = data[data.index < tdate]
        y_history = history['return']
        
        # Refit periodically or use last params
        if i % refit_freq == 0 or last_params is None:
            if i % refit_freq == 0:
                logger.info(f"Refitting model at {tdate.date()} (OOS step {i}/{len(test_dates)})...")
            am = arch_model(y_history, vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
            res = am.fit(starting_values=last_params, disp="off", show_warning=False)
            last_params = res.params.values
            model_res = res
        else:
            # Use last fitted parameters to update the filter for the new observation
            am = arch_model(y_history, vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
            model_res = am.fix(last_params)
            
        # Forecast 21 days ahead using simulation (required for EGARCH multi-step)
        fc = model_res.forecast(horizon=h, method="simulation", simulations=2000, reindex=False)
        
        # We want the cumulative variance over the horizon, annualized
        # Sum of variance over h days -> h-day variance. 
        # Annualized volatility = sqrt( (h-day variance / h) * 252 )
        h_day_var = fc.variance.iloc[-1].sum()
        ann_vol = np.sqrt( (h_day_var / h) * 252 )
        
        forecasts[tdate.strftime("%Y-%m-%d")] = float(ann_vol)
        
    return forecasts

def main():
    parser = argparse.ArgumentParser(description="EGARCH Pipeline")
    parser.add_argument("--start", type=str, default="2016-01-01", help="Start date for data fetch")
    parser.add_argument("--test-months", type=int, default=6, help="Number of months for OOS test")
    args = parser.parse_args()
    
    df = fetch_data(start=args.start)
    data = prepare_features(df)
    
    # 21-day forecast horizon for the straddle algorithm
    forecasts = walk_forward_eval(data, test_months=args.test_months, refit_freq=21, h=21)
    
    # Export for QuantConnect
    output_dir = "quantconnect_backtest"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "egarch_forecasts.json")
    
    with open(out_path, "w") as f:
        json.dump(forecasts, f, indent=4)
        
    logger.info(f"Exported {len(forecasts)} forecasts to {out_path}")
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
