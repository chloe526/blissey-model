# Start here

Scope: LA Hacks Sessions 03 + 04 only. Development cross-validation, one participant, no frozen test block yet.

## Current

| Path | What |
|---|---|
| `lahacks_v2/` | Current pipeline: gain-normalized windows, evaluation v2 (all-window event scoring), experiments, final model |
| `reports/ml_v2/` | Current experiment results and the final (UNVALIDATED) model |
| `reports/ml_strategy_review/` | Diagnosis (`REPORT.md`) and execution plan (`PLAN.md`) |
| `reports/capture_review/` | Packet-capture root cause, firmware fix, safe gap tolerance (`REPORT.md`) |
| `csi.py`, `csi_quality.py`, `preprocess.py` | Shared packet parser, quality-v2 gates, window transform. Hash-pinned by `lahacks_cv`; do not edit |

## Baseline v1 (kept for comparison; do not modify)

| Path | What |
|---|---|
| `lahacks_cv/` | First LA-only CNN/RF cross-validation (Veri job `d4c10e9207e8`) |
| `data/lahacks_quality_v2_cv_v1/` | Its unnormalized window dataset |
| `reports/lahacks_job_d4c10e9207e8/`, `reports/lahacks_failure_analysis/`, `reports/lahacks_stage1/` | Its results and audits |

## Legacy (not used by the current pipeline)

- MOMENT / Cal Poly experiment: `train.py`, `evaluate.py`, `model_utils.py`, `training_data.py`, `smoke_test_moment.py`, `fetch_dataset.py`, `scripts/`, `configs/moment-architecture.json`, `configs/training.json`, `data/processed/`, `reports/moment_*`, `reports/smoke_*`, `reports/veri_*`. Described by `README.md`.
- Historical CNN: `cnn_baseline/`. Only its `model.py` architecture is reused.
- Unfinished drafts: `labeling_v3.py`, `preprocess_v3.py`, `configs/label_policy_v3.json`.
- Old audits: `audit_quality_v2.py`, `review_preprocessing.py`, `inspect_dataset.py`, `characterize_acquisition.py`, `reports/archive/`.
