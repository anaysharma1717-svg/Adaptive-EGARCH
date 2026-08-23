"""
STAGE 7 -- Pairs-trading pilot (shelved, null result).

WHAT THIS SCRIPT ACTUALLY DOES: reprints saved results only. NOTE: the
original task1_*.py/task1b_*.py driver scripts for this pilot were already
removed in an earlier cleanup pass (data + results were kept, intermediate
scripts were not, per an explicit scope decision at the time) -- they are
NOT in research/pipeline/ and cannot be pointed to. See
research/pairs_trading/results/ for the underlying data this reprints.

Question this stage answers: separate from the volatility-forecasting work
above, is there a viable statistical-arbitrage pair among six hand-picked,
economically-linked futures pairs?

  ZB=F/ZN=F   Treasury curve (30Y/10Y)          -- expected strongest candidate
  ES=F/NQ=F   large-cap vs tech
  ES=F/RTY=F  large-cap vs small-cap
  NQ=F/RTY=F  tech vs small-cap
  GC=F/HG=F   precious vs industrial metals
  GC=F/SI=F   gold vs silver

This is a FIXED, pre-specified list -- not a screen. No pairs were added,
no combinations were generated, nothing was selected after seeing results.
A seventh pair, CL=F/NG=F (crude oil vs natural gas), was in the ORIGINAL
three-pair list but was dropped before any pair-level analysis: WTI crude
went genuinely negative on 2020-04-20 (a real market event, not a data
error), which makes log(price) -- and therefore any log-space cointegration
test, hedge ratio, or spread -- undefined across that window. Patching it
(clipping, shifting, dropping just those two days) would have been an
unprincipled judgment call with no way to prove it doesn't also distort the
hedge ratio over the surrounding window, so the pair was dropped entirely
rather than patched. That data file has been removed from this repository.

Methodology per pair: rolling Engle-Granger cointegration test (log-space
OLS hedge ratio, refit every 21 trading days on an expanding window,
Augmented Dickey-Fuller test on the fit residuals), full-sample ADF on the
resulting spread, and an Ornstein-Uhlenbeck half-life (how many days a
mean-reverting spread takes to close half the gap back to its average --
the key number for whether a pair moves fast enough to actually trade).

FIXED, pre-registered tradeability bar (not loosened after seeing results):
  - half-life between 5 and 60 days
  - cointegration pass rate > 30%
  - cointegrated in at least one recent (last-2-year) window
  - full-sample ADF p < 0.05

RESULT: zero of six pairs meet all four criteria. See the table below.
"""
import os
import pandas as pd

SRC = os.path.join("research", "pairs_trading", "results")
OUT_DIR = os.path.join("research", "results", "07_pairs_trading_pilot")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 90)
print("  STAGE 7: PAIRS-TRADING PILOT (six pre-specified pairs)")
print("=" * 90)

summary = pd.read_csv(os.path.join(SRC, "task1b_pair_summary.csv"))
print("\nFixed tradeability criteria: half-life in [5,60]d, coint pass rate > 30%,")
print("cointegrated in >=1 recent window, ADF p < 0.05\n")
print(summary.to_string(index=False))

n_qualify = int(summary["meets_all_4_criteria"].sum())
print(f"\nPairs meeting ALL 4 criteria: {n_qualify} / {len(summary)}")
if n_qualify == 0:
    print("ZERO pairs qualify. Thresholds were not loosened to produce a survivor --")
    print("this is the reported finding, not a failure to find one.")
    closest = summary.assign(
        hl_gap=summary["half_life_days"].apply(lambda h: max(0, h - 60) if h > 60 else max(0, 5 - h))
    ).sort_values("hl_gap").iloc[0]
    print(f"\nClosest pair: {closest['pair']} (half-life={closest['half_life_days']:.1f}d, "
          f"still {closest['half_life_days']-60:.1f}d over the 60-day cap)"
          if closest["half_life_days"] > 60 else "")

summary.to_csv(os.path.join(OUT_DIR, "pair_summary.csv"), index=False)
print(f"\nSaved: {OUT_DIR}/pair_summary.csv")
print(f"\nNote: this stage covers the SIX active pairs only. Full leg-level data quality")
print(f"diagnostics, hedge-ratio histories, and spread series for all pairs remain in")
print(f"research/pairs_trading/results/ (not duplicated here, per the original appendix scope).")
