# QuantConnect Volatility Arbitrage Algorithm
# Strategy: Delta-Neutral 21-day ATM Straddle
# Engine: Lean CLI (QuantConnect)

from AlgorithmImports import *
import json, gzip, base64
compressed_data = "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"
EGARCH_H21_FORECASTS = json.loads(gzip.decompress(base64.b64decode(compressed_data)).decode('utf-8'))

import math

class VolatilityArbitrageAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2016, 7, 1)
        self.SetEndDate(2026, 5, 22)
        self.SetCash(100000)

        # 1. Add Underlying SPY
        self.spy = self.AddEquity("SPY", Resolution.Minute)
        self.spy.SetDataNormalizationMode(DataNormalizationMode.Raw)

        # 2. Add Options
        option = self.AddOption("SPY", Resolution.Minute)
        
        # Filter: Strikes within +/- 2 strikes (ATM), Expiry 20-30 days
        option.SetFilter(-2, 2, timedelta(20), timedelta(30))
        self.option_symbol = option.Symbol
        
        # We need the Greeks to compute IV and Delta
        self.SetSecurityInitializer(lambda x: x.SetMarketPrice(self.GetLastKnownPrice(x)))
        option.PriceModel = OptionPriceModels.BlackScholes()
        
        # 3. State variables
        self.active_straddle_expiry = None
        self.active_calls = None
        self.active_puts = None
        self.call_delta = 0.0
        self.put_delta = 0.0
        self.edge_threshold = 0.02  # 2% IV vs Forecast premium required to enter
        
        # 4. Schedule Delta Hedging daily at 15:45 (right before close)
        self.Schedule.On(self.DateRules.EveryDay("SPY"),
                         self.TimeRules.BeforeMarketClose("SPY", 15),
                         self.DeltaHedge)

    def OnData(self, slice: Slice):
        # We only trade once a day right after the open (10:00 AM)
        if self.Time.hour != 10 or self.Time.minute != 0:
            return

        # If we have an active straddle expiring today or tomorrow, liquidate and wait for next entry
        if self.active_straddle_expiry is not None:
            time_to_expiry = (self.active_straddle_expiry - self.Time).days
            if time_to_expiry <= 1:
                self.Liquidate()
                self.active_straddle_expiry = None
                self.active_calls = None
                self.active_puts = None
            return

        # Check if we have a forecast for today
        date_str = self.Time.strftime("%Y-%m-%d")
        if date_str not in EGARCH_H21_FORECASTS:
            return
            
        forecast_vol = EGARCH_H21_FORECASTS[date_str]

        # Find option chain
        chain = slice.OptionChains.get(self.option_symbol)
        if not chain:
            return
            
        # Update cached deltas for active positions
        if self.active_calls or self.active_puts:
            for contract in chain:
                if contract.Symbol == self.active_calls:
                    self.call_delta = contract.Greeks.Delta
                elif contract.Symbol == self.active_puts:
                    self.put_delta = contract.Greeks.Delta

        # We only trade once a day right after the open (10:00 AM)
        if self.Time.hour != 10 or self.Time.minute != 0:
            return

        # Filter for the expiry closest to 21 days
        expiries = sorted(set(x.Expiry for x in chain))
        if not expiries:
            return
            
        target_expiry = min(expiries, key=lambda x: abs((x - self.Time).days - 21))
        
        # Filter contracts for this expiry
        contracts = [x for x in chain if x.Expiry == target_expiry]
        if not contracts:
            return
            
        # Find ATM strike
        underlying_price = self.spy.Price
        atm_strike = sorted(set(x.Strike for x in contracts), key=lambda x: abs(x - underlying_price))[0]
        
        # Get Call and Put
        call = next((x for x in contracts if x.Right == OptionRight.Call and x.Strike == atm_strike), None)
        put = next((x for x in contracts if x.Right == OptionRight.Put and x.Strike == atm_strike), None)
        
        if not call or not put:
            return
            
        # Get Implied Volatility (average of Call and Put)
        iv_call = call.ImpliedVolatility
        iv_put = put.ImpliedVolatility
        
        if iv_call == 0 or iv_put == 0:
            return
            
        avg_iv = (iv_call + iv_put) / 2.0
        
        # Entry Logic
        edge = avg_iv - forecast_vol
        
        # Size per straddle (Target ~10% of portfolio margin per trade)
        # Simplified: Buy/Sell 10 straddles
        quantity = 10
        
        if edge > self.edge_threshold:
            # IV is too high -> Sell Straddle
            self.Sell(call.Symbol, quantity)
            self.Sell(put.Symbol, quantity)
            self.active_straddle_expiry = target_expiry
            self.active_calls = call.Symbol
            self.active_puts = put.Symbol
            self.Debug(f"[{self.Time}] SELLING Straddle. IV: {avg_iv:.3f}, Forecast: {forecast_vol:.3f}, Edge: {edge:.3f}")
            
        elif edge < -self.edge_threshold:
            # IV is too low -> Buy Straddle
            self.Buy(call.Symbol, quantity)
            self.Buy(put.Symbol, quantity)
            self.active_straddle_expiry = target_expiry
            self.active_calls = call.Symbol
            self.active_puts = put.Symbol
            self.Debug(f"[{self.Time}] BUYING Straddle. IV: {avg_iv:.3f}, Forecast: {forecast_vol:.3f}, Edge: {edge:.3f}")

    def DeltaHedge(self):
        """
        Calculates the net delta of the options position and buys/sells SPY to neutralize it.
        """
        if self.active_straddle_expiry is None:
            # If no options, ensure no stock
            if self.Portfolio["SPY"].Invested:
                self.Liquidate("SPY")
            return
            
        # Calculate total options delta
        total_delta = 0
        
        if self.active_calls and self.Portfolio[self.active_calls].Invested:
            qty = self.Portfolio[self.active_calls].Quantity
            mult = self.Securities[self.active_calls].SymbolProperties.ContractMultiplier
            total_delta += self.call_delta * qty * mult
            
        if self.active_puts and self.Portfolio[self.active_puts].Invested:
            qty = self.Portfolio[self.active_puts].Quantity
            mult = self.Securities[self.active_puts].SymbolProperties.ContractMultiplier
            total_delta += self.put_delta * qty * mult
                    
        # The stock itself has a delta of 1 per share.
        # We want: Total Options Delta + Stock Shares = 0
        # So: Target Stock Shares = -Total Options Delta
        
        target_shares = -total_delta
        current_shares = self.Portfolio["SPY"].Quantity
        
        # Only hedge if the difference is more than 10 shares to save on fees
        if abs(target_shares - current_shares) > 10:
            self.SetHoldings("SPY", 0) # Clear old hedge
            
            if target_shares != 0:
                self.MarketOrder("SPY", round(target_shares))
