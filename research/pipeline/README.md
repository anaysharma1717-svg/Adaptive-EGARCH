# research/pipeline/ — the lab notebook

This folder holds the original analysis scripts, run in the order below, that
produced every result in this project by actually fitting/refitting models
against real data (as opposed to `research/01-07`, a verification layer that
mostly replays what these scripts already computed — see the main
[README.md](../../README.md) for that distinction). Run any of these directly
from the repo root, e.g. `python research/pipeline/task1_dm_significance.py`.

Actual chronological order:

| order | script | what it did |
|---|---|---|
| 0a | `extended_model_zoo.py` | The shared engine: SPY data, Garman-Klass RV, all 9 models (M1-M8, M2b), the DM test / QLIKE / Mincer-Zarnowitz functions every later script imports. Every other file in this folder depends on this one. |
| 0b | `_confirm_baseline.py` | Confirmed the initial baseline numbers (RMSE, MZ beta, etc.) reproduced from a fresh run, before any further task began — establishes `baseline_confirm_forecasts.csv`. |
| 1 | `task1_dm_significance.py` | **Task 1**: crisis-regime Diebold-Mariano test, corrected-M3 vs M1 HAR+TS. Result: not significant (p=0.15-0.88) — the project's first honestly-reported null result. |
| 2 | `task3_loghar_fix.py` | **Task 3**: found and fixed a Jensen's-inequality retransformation bug in the Log-HAR model (M4); before/after comparison. |
| 3 | `taskA_m9_regime_eval.py` | **Task 4 / Task A**: built M9, the logistic regime-weighted combination of HAR and corrected EGARCH; full evaluation (this supersedes an earlier, incomplete Task-4 script that was removed as a duplicate — this file has the complete, final version, including the discovery that raw EGARCH significantly beats HAR in crisis, DM=+2.797 p=0.008). |
| 4 | `stepB1_diagnostic_regime_mz.py` | **Step 1**: diagnosed *why* M9 wasn't working — the Mincer-Zarnowitz bias correction is fit on a training sample that's 90% calm days, so it over-shrinks EGARCH's signal specifically in crisis. |
| 5 | `stepB2_m2c_m9c.py` | **Step 2**: M2c/M9c — split the correction by regime. Real improvement, still loses to raw EGARCH in crisis. |
| 6 | `stepB3_m2d_m9d.py` | **Step 3**: M2d/M9d — refit the correction to minimize QLIKE instead of squared error. The motivating prediction (QLIKE-fit beta should be higher) was falsified; still an improvement, still not enough. |
| 7 | `stepB4_m2e_m9e.py` | **Step 4**: M2e/M9e — stop correcting in crisis entirely, use raw EGARCH there. Best numbers of the chain, still not statistically significant against the static M3 baseline. |
| 8 | `appendix_h5_and_macro.py` | Post-hoc robustness checks: does the crisis edge replicate at a 5-day horizon? Does a single FOMC-announcement dummy help M1? Both "real effect, doesn't cleanly confirm" results. |

Every script writes its output to `research/results/` (or `research/results/appendix_horizon_calendar/` for stage 8). Those CSVs are what `research/01-06` reads back.
