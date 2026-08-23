"""
STAGE 6 -- Robustness-check appendix: h=5 horizon replication + FOMC dummy.

Question this stage answers: two sanity checks on the headline finding
(stages 3-5), run AFTER the main investigation concluded, to see if it's
fragile to two obvious variations.

CHECK 1 -- does the crisis-QLIKE edge (stage 3, Test B) survive if we
forecast the AVERAGE volatility over the next 5 trading days (h=5, the
Corsi HAR multi-horizon convention) instead of just tomorrow (h=1)?
Answer: partially. The crisis-QLIKE edge for raw EGARCH over HAR DOES
replicate and remains significant (p=0.02) -- but EGARCH's h=5 forecast is
dramatically WORSE than HAR everywhere else (overall and calm both p<0.0002),
a much larger overall degradation than at h=1. Not a clean confirmation.

CHECK 2 -- does adding a single, hypothesis-driven FOMC-announcement-day
dummy to M1 help? Answer: the coefficient is real and highly significant
in-sample (+0.224, p<0.0001, matching the obvious prior that FOMC days are
more volatile) and the out-of-sample error on FOMC days specifically drops
a lot (QLIKE 0.593 -> 0.141) -- but with only 12 FOMC days in the 376-day
test window, no DM test reaches significance. This is a real effect that a
small test window is underpowered to confirm out-of-sample, not a null
result -- an important distinction to keep straight.

Data sourcing note carried over from when this was built: FOMC dates
2021-2027 came from a single direct fetch of the official Federal Reserve
calendar page (high confidence); 2011-2020 were assembled via web search
against individual official Fed press-release pages, not independently
re-verified page-by-page (lower confidence, flagged at the time). CPI/NFP
were dropped entirely rather than fabricated, after BLS.gov and FRED both
blocked automated fetches (HTTP 403) on three separate attempts.
"""
import os
import pandas as pd

SRC = os.path.join("research", "results", "appendix_horizon_calendar")
OUT_DIR = os.path.join("research", "results", "06_horizon_and_calendar_appendix")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 90)
print("  STAGE 6: ROBUSTNESS APPENDIX (h=5 horizon, FOMC dummy)")
print("=" * 90)

print("\n--- CHECK 1: h=5 metrics ---")
h5_metrics = pd.read_csv(os.path.join(SRC, "appendix_h5_metrics.csv"))
print(h5_metrics.to_string(index=False))

print("\n--- CHECK 1: h=5 DM tests (raw EGARCH vs M1 -- does the crisis edge replicate?) ---")
h5_dm = pd.read_csv(os.path.join(SRC, "appendix_h5_dm_tests.csv"))
print(h5_dm.to_string(index=False))

print("\n--- CHECK 2: M1 vs M1-macro metrics ---")
macro_metrics = pd.read_csv(os.path.join(SRC, "appendix_macro_metrics.csv"))
print(macro_metrics.to_string(index=False))

print("\n--- CHECK 2: M1-macro DM tests ---")
macro_dm = pd.read_csv(os.path.join(SRC, "appendix_macro_dm_tests.csv"))
print(macro_dm.to_string(index=False))

print("\n--- CHECK 2: final-refit FOMC-day coefficient ---")
final_ols = pd.read_csv(os.path.join(SRC, "appendix_macro_final_ols_summary.csv"))
print(final_ols.to_string(index=False))

print("\n--- CHECK 2: FOMC dates used (sourcing/confidence documented per-row) ---")
fomc_dates = pd.read_csv(os.path.join(SRC, "appendix_fomc_dates_used.csv"))
print(f"Total dates: {len(fomc_dates)}, sources: {fomc_dates['source'].unique().tolist()}")

for fname in ["appendix_h5_metrics.csv", "appendix_h5_dm_tests.csv", "appendix_macro_metrics.csv",
              "appendix_macro_dm_tests.csv", "appendix_macro_final_ols_summary.csv"]:
    pd.read_csv(os.path.join(SRC, fname)).to_csv(os.path.join(OUT_DIR, fname), index=False)

print(f"\nBOTTOM LINE: both checks are 'real effect, doesn't cleanly confirm the headline story'")
print(f"-- reported as nuanced, not collapsed into pass/fail. All outputs saved to {OUT_DIR}/")
