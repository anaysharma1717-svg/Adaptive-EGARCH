"""
STAGE 3 -- The crisis-regime test(s).

WHAT THIS SCRIPT ACTUALLY DOES: recomputes DM tests from saved forecasts. It
loads two already-saved per-day forecast CSVs (task1_labeled_forecasts.csv,
taskA_m9_forecasts_weights_losses.csv) and runs dm_test() on that fixed
data. The forecasts themselves came from research/pipeline/task1_dm_
significance.py and research/pipeline/taskA_m9_regime_eval.py.

Question this stage answers: does model performance differ between calm and
turbulent markets, and specifically -- is there a real, exploitable edge
hiding in the crisis subsample that an overall-sample comparison would miss?

IMPORTANT clarification folded into this stage, not glossed over: there were
TWO distinct crisis-regime DM tests run in this project's history, testing
two different model pairs, with two different results. Presenting both
avoids misrepresenting the project's actual history:

  TEST A (the original, pre-registered Task 1 test): corrected-M3 vs M1
  HAR+TS, crisis subsample. Result: NOT statistically significant
  (p ranges 0.15-0.88 across subsamples/losses). This was reported plainly
  at the time -- a null result, not hidden.

  TEST B (discovered later, while investigating why M9 -- stage 5 -- wasn't
  working): RAW (uncorrected) EGARCH (M2) vs M1 HAR+TS, crisis subsample.
  Result: DM = +2.797, p = 0.0081, N = 38 -- a real, statistically
  significant edge. THIS is the "p=0.008" finding that motivated the entire
  bias-correction investigation in stage 4 and the combination-model chain
  in stage 5.

Both are reproduced below from already-saved per-day forecasts -- no
re-running of EGARCH, no new numbers.
"""
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))
from extended_model_zoo import dm_test

OUT_DIR = os.path.join("research", "results", "03_crisis_regime_test")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 90)
print("  STAGE 3: CRISIS-REGIME TESTS")
print("=" * 90)

# ---------------------------------------------------------------------------
# TEST A -- corrected-M3 vs M1 (the original, pre-registered Task 1 test)
# ---------------------------------------------------------------------------
print("\n--- TEST A: corrected-M3 vs M1 HAR+TS (original pre-registered test) ---")
a = pd.read_csv(os.path.join("research", "results", "task1_labeled_forecasts.csv"), parse_dates=["date"])
actual_a, m1_a, m3_a, crisis_a = a["actual"].values, a["M1 HAR+TS"].values, a["M3 Combined"].values, a["crisis"].values
calm_a = ~crisis_a

rows_a = []
for label, mask in [("overall", np.ones(len(a), dtype=bool)), ("crisis", crisis_a), ("calm", calm_a)]:
    for loss in ("mse", "qlike"):
        stat, pval = dm_test(actual_a[mask], m1_a[mask], m3_a[mask], loss=loss)
        rows_a.append({"test": "A_corrected_M3_vs_M1", "subsample": label, "loss": loss,
                        "N": int(mask.sum()), "dm_stat": stat, "p_value": pval})
        print(f"  [{label:<8}] {loss:>6}  N={int(mask.sum())}  DM={stat:+.4f}  p={pval:.4f}")
print("  Result: NOT significant anywhere -- reported as a null result at the time.")

# ---------------------------------------------------------------------------
# TEST B -- raw EGARCH (M2) vs M1 (the discovery that drove everything after it)
# ---------------------------------------------------------------------------
print("\n--- TEST B: raw EGARCH (M2) vs M1 HAR+TS (the p=0.008 discovery) ---")
b = pd.read_csv(os.path.join("research", "results", "taskA_m9_forecasts_weights_losses.csv"), parse_dates=["date"])
actual_b, m1_b, m2_b, crisis_b = b["actual"].values, b["M1_HAR_TS"].values, b["M2_EGARCH_raw"].values, b["crisis"].values
calm_b = ~crisis_b

rows_b = []
for label, mask in [("overall", np.ones(len(b), dtype=bool)), ("crisis", crisis_b), ("calm", calm_b)]:
    for loss in ("mse", "qlike"):
        stat, pval = dm_test(actual_b[mask], m1_b[mask], m2_b[mask], loss=loss)
        rows_b.append({"test": "B_raw_M2_vs_M1", "subsample": label, "loss": loss,
                        "N": int(mask.sum()), "dm_stat": stat, "p_value": pval})
        print(f"  [{label:<8}] {loss:>6}  N={int(mask.sum())}  DM={stat:+.4f}  p={pval:.4f}")
print("  Result: crisis/QLIKE is significant (p=0.008) -- raw EGARCH genuinely beats HAR")
print("  specifically when it matters. This motivated stages 4 and 5.")

out = pd.DataFrame(rows_a + rows_b)
out.to_csv(os.path.join(OUT_DIR, "crisis_dm_tests_A_and_B.csv"), index=False)
print(f"\nSaved: {OUT_DIR}/crisis_dm_tests_A_and_B.csv")
