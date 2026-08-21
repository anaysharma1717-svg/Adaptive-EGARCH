# QuantConnect Volatility Arbitrage Strategy

This folder contains a fully functional QuantConnect algorithm designed to backtest a **Delta-Neutral Straddle Volatility Arbitrage** strategy using your `EGARCH_H21` volatility forecasts.

## The Strategy
1. **Forecast Horizon:** 21 Trading Days.
2. **Options Contract:** Nearest-to-the-Money (ATM) Straddle expiring in roughly 21 days.
3. **Entry Logic:**
   - Compares the option's Implied Volatility (IV) to your custom EGARCH 21-day forecast.
   - If `IV > Forecast + 2%`: Sell the straddle (Expect IV crush / mean reversion).
   - If `IV < Forecast - 2%`: Buy the straddle (Expect volatility expansion).
4. **Hedging:** Computes the total Delta of the options position daily at 3:45 PM and buys/sells SPY shares to maintain strict Delta Neutrality.
5. **Exit:** Liquidates the straddle 1 day before expiration to avoid physical delivery complications.

## How to Run it (Free Options Data via Lean Cloud)

Since you do not have Docker installed, you cannot run this backtest locally. Instead, you will run it on the QuantConnect Cloud infrastructure, which gives you free access to decades of SPY options data.

### Option 1: Via the QuantConnect Web IDE (Easiest)
1. Go to [QuantConnect.com](https://www.quantconnect.com/) and create a new Python Algorithm.
2. Open `main.py` from this folder, copy all the code, and paste it into the QuantConnect editor's `main.py`.
3. Open `forecast_data.py` from this folder, copy all the code, and create a new file in the QuantConnect editor named `forecast_data.py`, then paste the code.
4. Hit **Backtest**.

### Option 2: Via Lean CLI (Command Line)
If you want to use the Lean CLI you installed:
1. Open a terminal and run `lean login` to authenticate with your QuantConnect account.
2. Run `lean cloud push "QuantConnect_VolArb"` to push this code to a cloud project.
3. Run `lean cloud backtest "QuantConnect_VolArb"` to execute the backtest on their servers.
