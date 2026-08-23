"""
STAGE 5 -- The combination-model chain: M9 -> M9c -> M9d -> M9e.

WHAT THIS SCRIPT ACTUALLY DOES: reprints saved results only -- see
research/pipeline/taskA_m9_regime_eval.py and research/pipeline/
stepB2_m2c_m9c.py / stepB3_m2d_m9d.py / stepB4_m2e_m9e.py for the actual
computation. Nothing below is recomputed; this is a read-and-display of
already-saved CSVs.

Question this stage answers: stage 3 found raw EGARCH beats HAR in crisis.
Can a SMOOTH (not hard-switched) combination of the two exploit that
regime-dependent advantage better than the static M3 combination (stage 2)
does?

Each M9-variant combines M1 (HAR+TS) with whichever EGARCH-correction variant
stage 4 had just produced, via the SAME logistic-weighting mechanism every
time (never retuned beyond its own training-only grid search):

    p_t  = ex-ante trailing-252-day percentile rank of yesterday's RV
           (uses data through t-1 only -- never today's own volatility)
    w_t  = 1 / (1 + exp(-k * (p_t - c)))          [a smooth "S-curve", not a
                                                     hard if/else switch]
    M9_t = w_t * (EGARCH-based forecast)_t + (1 - w_t) * M1_t

(c, k) are refit every 21 trading days on the expanding training window only,
minimizing in-sample QLIKE via a small grid search (c in [0.10,0.90], k in
{1..20}) -- identical grid, identical procedure, every single time. Only the
EGARCH-based input series changes across M9/M9c/M9d/M9e, matching stage 4's
M2b/M2c/M2d-regime/M2e respectively.

  M9  (combines M2b): does NOT beat static M3 in crisis (the negative result
      that triggered the whole stage-4 investigation in the first place).
  M9c (combines M2c): still does not beat M3 in crisis with significance.
  M9d (combines M2d-regime): first variant to numerically beat M3's crisis
      QLIKE (0.966 vs 1.079) -- but the DM test against M3 is not
      significant (p=0.282).
  M9e (combines M2e): best crisis QLIKE yet (0.929) -- still not significant
      vs M3 (p=0.18-0.28), and it inherits M2e's calm-day cost.

One specific, quotable finding: M9e's fitted steepness k landed on 1.0 (the
grid's MINIMUM) at every one of 18 refits, and the realized weight on the
EGARCH-based forecast never exceeded 0.646 even in crisis -- i.e. this is
genuinely NOT a disguised hard switch; the data itself never supported a
sharp regime distinction, only a gentle lean.
"""
import os
import pandas as pd

RES = os.path.join("research", "results")
OUT_DIR = os.path.join(RES, "05_regime_weighted_combination")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 90)
print("  STAGE 5: REGIME-WEIGHTED COMBINATION CHAIN (M9 -> M9c -> M9d -> M9e)")
print("=" * 90)

print("\n--- M9 (combines M2b): metrics ---")
m9_metrics = pd.read_csv(os.path.join(RES, "taskA_m9_metrics.csv"))
print(m9_metrics.to_string(index=False))
print("\n--- M9 DM tests vs M3 and M1 ---")
m9_dm = pd.read_csv(os.path.join(RES, "taskA_m9_dm_tests.csv"))
print(m9_dm.to_string(index=False))
print("\n--- M9 Mincer-Zarnowitz ---")
print(pd.read_csv(os.path.join(RES, "taskA_m9_mz.csv")).to_string(index=False))

print("\n--- M9c (combines M2c): metrics ---")
m9c_metrics = pd.read_csv(os.path.join(RES, "stepB2_m9c_metrics.csv"))
print(m9c_metrics.to_string(index=False))
print("\n--- M9c DM tests ---")
dm2 = pd.read_csv(os.path.join(RES, "stepB2_dm_tests.csv"))
m9c_dm = dm2[dm2["comparison"].str.contains("M9c", na=False)]
print(m9c_dm.to_string(index=False))

print("\n--- M9d (combines M2d-regime): metrics ---")
m9d_metrics = pd.read_csv(os.path.join(RES, "stepB3_m9d_metrics.csv"))
print(m9d_metrics.to_string(index=False))
print("\n--- M9d DM tests ---")
dm3 = pd.read_csv(os.path.join(RES, "stepB3_dm_tests.csv"))
m9d_dm = dm3[dm3["comparison"].str.contains("M9d", na=False)]
print(m9d_dm.to_string(index=False))

print("\n--- M9e (combines M2e): metrics ---")
m9e_metrics = pd.read_csv(os.path.join(RES, "stepB4_m9e_metrics.csv"))
print(m9e_metrics.to_string(index=False))
print("\n--- M9e DM tests ---")
dm4 = pd.read_csv(os.path.join(RES, "stepB4_dm_tests.csv"))
m9e_dm = dm4[dm4["comparison"].str.contains("M9e", na=False)]
print(m9e_dm.to_string(index=False))

print("\n--- M9e realized weight distribution by ex-ante regime (is it secretly a hard switch?) ---")
w9e_dist = pd.read_csv(os.path.join(RES, "stepB4_w9e_distribution.csv"))
print(w9e_dist.to_string(index=False))
print("Answer: no. Max realized weight in crisis = "
      f"{w9e_dist.loc[w9e_dist['regime']=='crisis','max'].iloc[0]:.3f} -- well below 1.0.")

m9_metrics.to_csv(os.path.join(OUT_DIR, "m9_metrics.csv"), index=False)
m9_dm.to_csv(os.path.join(OUT_DIR, "m9_dm_tests.csv"), index=False)
m9c_metrics.to_csv(os.path.join(OUT_DIR, "m9c_metrics.csv"), index=False)
m9c_dm.to_csv(os.path.join(OUT_DIR, "m9c_dm_tests.csv"), index=False)
m9d_metrics.to_csv(os.path.join(OUT_DIR, "m9d_metrics.csv"), index=False)
m9d_dm.to_csv(os.path.join(OUT_DIR, "m9d_dm_tests.csv"), index=False)
m9e_metrics.to_csv(os.path.join(OUT_DIR, "m9e_metrics.csv"), index=False)
m9e_dm.to_csv(os.path.join(OUT_DIR, "m9e_dm_tests.csv"), index=False)
w9e_dist.to_csv(os.path.join(OUT_DIR, "m9e_weight_distribution.csv"), index=False)

print(f"\nBOTTOM LINE: none of the four combination variants beat static M3 in crisis with")
print(f"statistical significance. M9d/M9e get numerically closest. All outputs saved to {OUT_DIR}/")
