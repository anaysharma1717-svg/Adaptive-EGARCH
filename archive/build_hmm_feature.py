import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings
import os

warnings.filterwarnings("ignore")

def main():
    print("Loading data...")
    # Download VIX directly from Yahoo Finance for complete coverage (2010-2026)
    import yfinance as yf
    vix_raw = yf.download('^VIX', start='2010-01-01', progress=False)
    vix_raw.index = pd.to_datetime(vix_raw.index)
    vix_series = vix_raw['Close'].squeeze().dropna().sort_index()
    vix_data   = np.log(vix_series.values).reshape(-1, 1)

    prob_crisis = np.zeros(len(vix_series))

    BURN_IN = 500

    print("Generating expanding-window HMM probabilities...")

    crisis_state = 0  # initialise; will be set at first fit
    model = None

    for t in range(BURN_IN, len(vix_series)):
        if t % 500 == 0:
            print(f"  Processed {t}/{len(vix_series)} days...")

        # Refit HMM every 60 days to balance speed and accuracy
        if t == BURN_IN or t % 60 == 0:
            model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
            model.fit(vix_data[:t])
            # Crisis state = higher mean log(VIX)
            crisis_state = int(np.argmax(model.means_.flatten()))

        # Predict probability for day t
        probs = model.predict_proba(vix_data[:t+1])
        prob_crisis[t] = probs[-1, crisis_state]

    # Fill burn-in period with the first known probability
    prob_crisis[:BURN_IN] = prob_crisis[BURN_IN]

    result = pd.DataFrame({
        'VIX':        vix_series.values,
        'prob_crisis': prob_crisis
    }, index=vix_series.index)

    os.makedirs('data', exist_ok=True)
    out_path = 'data/hmm_features.csv'
    result.to_csv(out_path)
    print(f"\nSaved HMM features to {out_path}")
    print(f"Date range: {result.index[0].date()} -> {result.index[-1].date()}")
    print(f"Years covered: {sorted(result.index.year.unique().tolist())}")

if __name__ == '__main__':
    main()
