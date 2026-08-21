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
