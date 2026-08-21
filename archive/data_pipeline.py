import pandas as pd
import numpy as np
import yfinance as yf
from pandas.tseries.offsets import CustomBusinessDay
import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

try:
    import pandas_datareader.data as web
except ImportError:
    print("Please run: pip install pandas_datareader")
    sys.exit(1)

def get_economic_dates(start_year=2005, end_year=2030):
    """
    Generate hardcoded/rule-based dates for NFP, CPI, and FOMC.
    NFP: First Friday of every month.
    CPI: 13th of every month (approximate mid-month release).
    FOMC: Every 6 weeks (8 times a year). Approximated as 3rd Wednesday of specific months.
    """
    start_date = f'{start_year}-01-01'
    end_date = f'{end_year}-12-31'
    
    # NFP: First Friday
    nfp_dates = pd.date_range(start=start_date, end=end_date, freq='WOM-1FRI')
    
    # CPI: Approximate as 13th of the month. If weekend, shift to Monday.
    cpi_dates = pd.date_range(start=start_date, end=end_date, freq='MS') + pd.Timedelta(days=12)
    cpi_dates = cpi_dates + pd.to_timedelta(np.where(cpi_dates.weekday == 5, 2, np.where(cpi_dates.weekday == 6, 1, 0)), unit='D')
    
    # FOMC: 8 times a year (Jan, Mar, May, Jun, Jul, Sep, Nov, Dec). Approx 3rd Wednesday.
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
    fomc_dates = []
    for year in range(start_year, end_year + 1):
        for month in fomc_months:
            # Get 3rd Wednesday
            dr = pd.date_range(start=f'{year}-{month:02d}-01', end=f'{year}-{month:02d}-28', freq='WOM-3WED')
            if len(dr) > 0:
                fomc_dates.append(dr[0])
    
    fomc_dates = pd.DatetimeIndex(fomc_dates)
    return set(nfp_dates), set(cpi_dates), set(fomc_dates)

def days_to_next_event(current_date, event_dates):
    """Returns number of days to the next event date."""
    future_dates = [d for d in event_dates if d >= current_date]
    if not future_dates:
        return 30 # fallback
    return (min(future_dates) - current_date).days

def calculate_yang_zhang_vol(df_ohlc, window=21):
    """
    Calculates Yang-Zhang volatility using Open, High, Low, Close.
    """
    # Assuming df_ohlc has 'Open', 'High', 'Low', 'Close' for SPY
    log_ho = np.log(df_ohlc['High'] / df_ohlc['Open'])
    log_lo = np.log(df_ohlc['Low'] / df_ohlc['Open'])
    log_co = np.log(df_ohlc['Close'] / df_ohlc['Open'])
    
    log_oc = np.log(df_ohlc['Open'] / df_ohlc['Close'].shift(1))
    log_oc_sq = log_oc**2
    
    log_cc = np.log(df_ohlc['Close'] / df_ohlc['Close'].shift(1))
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    sigma_o_sq = log_oc_sq.rolling(window=window).mean()
    sigma_c_sq = log_cc_sq.rolling(window=window).mean()
    sigma_rs_sq = rs.rolling(window=window).mean()
    
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    
    # YZ Variance
    yz_var = sigma_o_sq + k * sigma_c_sq + (1 - k) * sigma_rs_sq
    
    # Annualized Volatility
    yz_vol = np.sqrt(yz_var * 252) * 100
    return yz_vol

def main():
    print("Downloading Market Data (YFinance)...")
    tickers = ['SPY', '^VIX', '^VVIX', '^VIX9D', '^VIX3M', 'HYG', 'LQD']
    # VVIX only goes back to ~2007
    start_date = '2010-01-01'
    data = yf.download(tickers, start=start_date)
    
    print("Downloading Treasury Yields (FRED)...")
    yields = web.DataReader(['DGS2', 'DGS10'], 'fred', start_date)
    
    # Process SPY OHLCV
    spy = data.xs('SPY', axis=1, level=1)
    
    # Combine into feature dataframe
    features = pd.DataFrame(index=spy.index)
    
    print("Calculating Features...")
    # 1 & 2. VIX & VVIX
    features['VIX'] = data['Close']['^VIX']
    features['VVIX'] = data['Close']['^VVIX']
    
    # 3. VIX Term Structure
    features['VIX_TERM'] = data['Close']['^VIX9D'] - data['Close']['^VIX3M']
    
    # 4. Yang-Zhang Vol
    features['YZ_VOL_21'] = calculate_yang_zhang_vol(spy, window=21)
    
    # 5. Volume Z-Score
    vol_mean = spy['Volume'].rolling(21).mean()
    vol_std = spy['Volume'].rolling(21).std()
    features['VOL_Z'] = (spy['Volume'] - vol_mean) / vol_std
    
    # 6. Credit Spread (HYG/LQD Price Ratio)
    features['CREDIT_SPREAD'] = data['Close']['HYG'] / data['Close']['LQD']
    
    # 7. Yield Curve
    yields_aligned = yields.reindex(features.index).ffill()
    features['YIELD_CURVE'] = yields_aligned['DGS10'] - yields_aligned['DGS2']
    
    # 8. Overnight Gap
    features['OVERNIGHT_GAP'] = np.log(spy['Open'] / spy['Close'].shift(1))
    
    # 9. Volatility Momentum
    yz_vol_5 = calculate_yang_zhang_vol(spy, window=5)
    features['VOL_MOMENTUM'] = yz_vol_5 / features['YZ_VOL_21']
    
    # 10 & 11. Days to Events
    nfp_dates, cpi_dates, fomc_dates = get_economic_dates()
    
    # Vectorized countdown
    print("Calculating event countdowns...")
    features['DAYS_TO_FOMC'] = [days_to_next_event(d, fomc_dates) for d in features.index]
    features['DAYS_TO_CPI'] = [days_to_next_event(d, cpi_dates) for d in features.index]
    features['DAYS_TO_NFP'] = [days_to_next_event(d, nfp_dates) for d in features.index]
    # For CPI/NFP we take the min distance
    features['DAYS_TO_CPI_NFP'] = np.minimum(features['DAYS_TO_CPI'], features['DAYS_TO_NFP'])
    features.drop(columns=['DAYS_TO_CPI', 'DAYS_TO_NFP'], inplace=True)
    
    # Target Variable: Returns for models
    features['SPY_RETURN'] = np.log(spy['Close'] / spy['Close'].shift(1))
    
    print("Normalizing continuous features (252-day rolling Z-score)...")
    # BUG FIX: ALL 11 features must be Z-scored, including event countdown features.
    # DAYS_TO_FOMC (0-62) and DAYS_TO_CPI_NFP (0-24) were previously left on their
    # raw integer scale, causing raw scores w^T X to explode to ±120 inside exp().
    features_to_zscore = ['VIX', 'VVIX', 'VIX_TERM', 'YZ_VOL_21', 'VOL_Z', 'CREDIT_SPREAD',
                          'YIELD_CURVE', 'OVERNIGHT_GAP', 'VOL_MOMENTUM',
                          'DAYS_TO_FOMC', 'DAYS_TO_CPI_NFP']
    
    for col in features_to_zscore:
        roll_mean = features[col].rolling(252).mean()
        roll_std = features[col].rolling(252).std()
        features[f'{col}_Z'] = (features[col] - roll_mean) / roll_std
        
    # Drop rows with NaNs (mostly due to the 252-day rolling window)
    features_clean = features.dropna()
    
    output_path = 'data/dp_egarch_features.csv'
    features_clean.to_csv(output_path)
    print(f"Success! Final feature matrix saved to {output_path} with {len(features_clean)} trading days.")

if __name__ == "__main__":
    main()
