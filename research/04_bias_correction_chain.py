"""
STAGE 4 -- The bias-correction chain: M2b -> M2c -> M2d -> M2e.

Question this stage answers: raw EGARCH has a real crisis-QLIKE edge over
HAR (stage 3, Test B). The standard fix for EGARCH's known bias -- a
Mincer-Zarnowitz (MZ) correction -- is already part of M2b/M3. Does that
correction preserve the crisis edge, or does "fixing" EGARCH's bias
accidentally erase the one place it was actually earning its keep?

Each step below is a genuine iteration, in order, each one motivated by what
the previous step found -- not four independent experiments:

  M2b (existing, stage 2): pooled MZ correction, corrected = alpha + beta*raw,
    fit once on ALL training days.
    -> Found (this stage, "diagnosis"): the correction is fit on a sample
       that's 90% calm days, so its single beta reflects calm-day dynamics.
       beta_crisis (0.830) != beta_calm (0.617), t~5.85 -- confirmed, large.
       M2b vs raw M2, crisis QLIKE: DM=-3.83, p=0.0005 -- the correction
       measurably destroys the crisis edge.

  M2c: same correction form, but fit TWO separate regressions (crisis-only,
    calm-only training days), applied by each day's own ex-ante regime.
    -> Found: real, significant improvement over M2b, but STILL
       significantly worse than raw EGARCH in crisis (p=0.0002).

  M2d: same regime split, but the (alpha,beta) are fit by numerically
    MINIMIZING QLIKE instead of ordinary least squares (OLS minimizes
    squared error, a different target than QLIKE -- the hypothesis was that
    OLS's squared-error optimum imposes more shrinkage than QLIKE wants).
    -> Found: the hypothesis was WRONG. QLIKE-fit beta came out LOWER than
       OLS beta in every regime, at every refit -- the opposite of the
       prediction. Still, M2d-regime was a large, significant improvement
       over M2c (not for the predicted reason), but still lost to raw
       EGARCH in crisis (p=0.0002).

  M2e: stop correcting in crisis entirely -- use raw EGARCH there, and only
    apply the (QLIKE-fit) correction on calm days.
    -> Found: the best crisis numbers of the whole chain, but still not
       significantly better than raw EGARCH itself in crisis (since crisis
       days ARE mostly just raw EGARCH by construction) -- and a real,
       significant calm-day cost, because the ex-ante regime flag is an
       imperfect predictor (only 19 of 45 ex-ante-"crisis" days were
       actually crisis days -- 26 false alarms use the raw, noisier signal
       on days that turned out calm).

Everything below is loaded from already-saved CSVs; only the M2c-vs-raw-M2
crisis DM test is recomputed here (it was never saved as its own row
elsewhere), from the fixed per-day forecasts already saved by stage 2 of the
original investigation (stepB2_forecasts_weights_losses.csv) -- a
deterministic function of already-saved data, not a new computation.
"""
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extended_model_zoo import dm_test

RES = os.path.join("research", "results")
OUT_DIR = os.path.join(RES, "04_bias_correction_chain")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 90)
print("  STAGE 4: BIAS-CORRECTION CHAIN (M2b -> M2c -> M2d -> M2e)")
print("=" * 90)

# --- Diagnosis: MZ beta differs by regime ---
print("\n--- Diagnosis: MZ-correction beta, pooled vs crisis-only vs calm-only training ---")
mz_regime = pd.read_csv(os.path.join(RES, "stepB1_mz_by_regime.csv"))
print(mz_regime[["regime", "N", "alpha", "beta"]].to_string(index=False))

print("\n--- (a)/(b)/(c): does the correction destroy the crisis edge? ---")
dm_ab_c = pd.read_csv(os.path.join(RES, "stepB1_dm_tests_crisis.csv"))
print(dm_ab_c.to_string(index=False))

# --- M2c: regime-split OLS correction ---
print("\n--- M2c: regime-split OLS correction, metrics ---")
m2c_metrics = pd.read_csv(os.path.join(RES, "stepB2_m2c_metrics.csv"))
print(m2c_metrics.to_string(index=False))

print("\n--- M2c vs raw M2, crisis (recomputed from saved per-day forecasts -- not previously saved as its own row) ---")
step2 = pd.read_csv(os.path.join(RES, "stepB2_forecasts_weights_losses.csv"))
crisis2 = step2["crisis_eval"].values.astype(bool)
m2c_vs_m2_rows = []
for loss in ("qlike", "mse"):
    stat, pval = dm_test(step2["actual"].values[crisis2], step2["M2_raw"].values[crisis2],
                          step2["M2c_regime"].values[crisis2], loss=loss)
    m2c_vs_m2_rows.append({"comparison": "M2c vs raw M2", "subsample": "crisis", "loss": loss,
                            "N": int(crisis2.sum()), "dm_stat": stat, "p_value": pval})
    print(f"  {loss:>6}  N={int(crisis2.sum())}  DM={stat:+.4f}  p={pval:.4f}")

# --- M2d: QLIKE-fit regime-split correction ---
print("\n--- M2d: QLIKE-fit correction, metrics ---")
m2d_metrics = pd.read_csv(os.path.join(RES, "stepB3_m2d_metrics.csv"))
print(m2d_metrics.to_string(index=False))

print("\n--- M2d DM tests (crisis): does QLIKE-fitting change the conclusion? ---")
dm3 = pd.read_csv(os.path.join(RES, "stepB3_dm_tests.csv"))
m2d_dm = dm3[dm3["comparison"].str.contains("M2d", na=False)]
print(m2d_dm.to_string(index=False))

# --- M2e: regime-switched application ---
print("\n--- M2e: regime-switched application (crisis=raw, calm=corrected), metrics ---")
m2e_metrics = pd.read_csv(os.path.join(RES, "stepB4_m2e_metrics.csv"))
print(m2e_metrics.to_string(index=False))

print("\n--- M2e DM tests ---")
dm4 = pd.read_csv(os.path.join(RES, "stepB4_dm_tests.csv"))
m2e_dm = dm4[dm4["comparison"].str.contains("M2e", na=False)]
print(m2e_dm.to_string(index=False))

# --- consolidated save ---
pd.concat([
    mz_regime.assign(section="mz_by_regime"),
], ignore_index=True).to_csv(os.path.join(OUT_DIR, "diagnosis_mz_by_regime.csv"), index=False)
dm_ab_c.to_csv(os.path.join(OUT_DIR, "abc_dm_tests.csv"), index=False)
pd.DataFrame(m2c_vs_m2_rows).to_csv(os.path.join(OUT_DIR, "m2c_vs_raw_m2_crisis.csv"), index=False)
m2c_metrics.to_csv(os.path.join(OUT_DIR, "m2c_metrics.csv"), index=False)
m2d_metrics.to_csv(os.path.join(OUT_DIR, "m2d_metrics.csv"), index=False)
m2d_dm.to_csv(os.path.join(OUT_DIR, "m2d_dm_tests.csv"), index=False)
m2e_metrics.to_csv(os.path.join(OUT_DIR, "m2e_metrics.csv"), index=False)
m2e_dm.to_csv(os.path.join(OUT_DIR, "m2e_dm_tests.csv"), index=False)

print(f"\nBOTTOM LINE: every corrected variant (M2b, M2c, M2d-regime) remains significantly")
print(f"worse than raw EGARCH in crisis (p~0.0002-0.0005). M2e closes the gap the most but")
print(f"crisis days there ARE mostly raw EGARCH by construction; the real cost shows up as a")
print(f"significant calm-day degradation from ex-ante regime false alarms.")
print(f"\nAll outputs saved to: {OUT_DIR}/")
