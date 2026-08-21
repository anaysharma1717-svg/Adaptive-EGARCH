import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from scipy.stats import t as t_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def fetch_data(ticker="SPY", start="2011-01-01"):
    print(f"Fetching {ticker} data from {start}...")
    df = yf.download(ticker, start=start, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize()
    return df

def garman_klass_vol(df):
    print("Computing Daily Garman-Klass Realized Volatility...")
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    # Convert to daily % volatility
    gk_vol = np.sqrt(np.maximum(gk_var, 1e-8)) * 100.0
    return gk_vol.dropna()

def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))
def mae(a, b): return np.mean(np.abs(a - b))
def dm_test(e1, e2):
    d = e1**2 - e2**2
    n = len(d)
    if n == 0: return np.nan, np.nan
    d_bar = d.mean()
    var_d = np.var(d, ddof=1) / n
    dm = d_bar / np.sqrt(var_d + 1e-12)
    hln = dm * np.sqrt((n + 1 - 2 + 1.0/n) / n)
    p = 2 * t_dist.sf(np.abs(hln), df=n-1)
    return round(hln, 4), round(p, 4)

def main():
    # 1. Prepare full history RV (Daily Garman-Klass)
    df_spy = fetch_data(start="2011-01-01")
    rv_series = garman_klass_vol(df_spy)
    
    # 2. Build HAR features
    har_df = pd.DataFrame({'rv': rv_series})
    har_df['rv_1'] = har_df['rv'].shift(1)
    har_df['rv_5'] = har_df['rv_1'].rolling(5).mean()
    har_df['rv_22'] = har_df['rv_1'].rolling(22).mean()
    har_df = har_df.dropna()
    
    # 3. Load baseline forecasts to align dates and target
    print("Loading baseline forecasts...")
    base_df = pd.read_csv('data/forecasts.csv', index_col='date', parse_dates=True)
    
    # Ensure indices intersect
    common_dates = base_df.index.intersection(har_df.index)
    base_df = base_df.loc[common_dates]
    
    # Extract refit dates
    refit_dates = pd.to_datetime(base_df['refit_date'].unique())
    refit_dates = refit_dates.sort_values()
    
    # 4. Walk-forward OLS estimation
    print("Running HAR-RV Walk-Forward (expanding window)...")
    har_forecasts = pd.Series(index=common_dates, dtype=float)
    
    model = LinearRegression()
    
    for i, rdate in enumerate(refit_dates):
        # Training data: all available before this refit_date
        train_df = har_df[har_df.index < rdate]
        if len(train_df) < 100:
            continue
            
        X_train = train_df[['rv_1', 'rv_5', 'rv_22']].values
        y_train = train_df['rv'].values
        model.fit(X_train, y_train)
        
        # Test dates associated with this refit
        test_mask = (base_df['refit_date'] == str(rdate.date())) | (base_df['refit_date'] == str(rdate))
        test_dates = base_df[test_mask].index
        test_dates = test_dates.intersection(har_df.index)
        
        for tdate in test_dates:
            X_test = har_df.loc[tdate, ['rv_1', 'rv_5', 'rv_22']].values.reshape(1, -1)
            pred = model.predict(X_test)[0]
            # Floor forecast at a small positive number
            har_forecasts.loc[tdate] = max(pred, 0.01)
            
    har_forecasts = har_forecasts.dropna()
    base_df = base_df.loc[har_forecasts.index]
    har_df = har_df.loc[har_forecasts.index]
    
    # Target is the ACTUAL daily Garman-Klass vol
    target = har_df['rv'].values
    garch_fc = base_df['garch_fc_vol'].values
    egarch_fc = base_df['egarch_fc_vol'].values
    har_fc = har_forecasts.values
    
    # 5. Evaluate
    har_rmse = rmse(har_fc, target)
    garch_rmse = rmse(garch_fc, target)
    egarch_rmse = rmse(egarch_fc, target)
    
    har_mae = mae(har_fc, target)
    garch_mae = mae(garch_fc, target)
    egarch_mae = mae(egarch_fc, target)
    
    err_har = har_fc - target
    err_garch = garch_fc - target
    dm_stat, dm_p = dm_test(err_har, err_garch)
    
    print("\n" + "="*65)
    print("  RESULTS — h=1, Daily Garman-Klass Target")
    print("="*65)
    print(f"                 Model   RMSE    MAE ")
    print(f"            GARCH(1,1) {garch_rmse:.4f} {garch_mae:.4f}")
    print(f"       EGARCH(1,1,1,t) {egarch_rmse:.4f} {egarch_mae:.4f}")
    print(f"                HAR-RV {har_rmse:.4f} {har_mae:.4f}")
    print("-"*65)
    print(f"  DM stat (HAR vs GARCH) : {dm_stat:.4f}")
    print(f"  DM p-value             : {dm_p:.4f}")
    if har_rmse < garch_rmse:
        print(f"  Result                 : HAR-RV BEATS GARCH by {(garch_rmse - har_rmse)/garch_rmse*100:.1f}%")
    else:
        print(f"  Result                 : GARCH WINS by {(har_rmse - garch_rmse)/har_rmse*100:.1f}%")
    print("="*65)
    
    # 6. Save results to CSV
    base_df['har_fc_vol'] = har_forecasts
    base_df['target_gk'] = target
    base_df.to_csv('data/har_rv_results.csv')
    
    # 7. Plot
    plt.figure(figsize=(14, 6))
    plt.plot(base_df.index, target, color='lightgrey', label='Target (Daily Garman-Klass)', alpha=0.7)
    plt.plot(base_df.index, garch_fc, color='#1f77b4', lw=1, alpha=0.8, label=f'GARCH(1,1) [RMSE {garch_rmse:.3f}]')
    plt.plot(base_df.index, har_fc, color='#ff7f0e', lw=1, alpha=0.8, label=f'HAR-RV [RMSE {har_rmse:.3f}]')
    plt.title("GARCH vs HAR-RV OOS Forecasts (Target: Daily Garman-Klass RV)")
    plt.ylabel("Volatility (%/day)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('plots/har_rv_vs_garch.png', dpi=150)
    print("\nSaved plot to plots/har_rv_vs_garch.png")

if __name__ == '__main__':
    main()
