# Execution plan: evaluation v2 → gain-normalized baseline → model ablations

Follows `REPORT.md`. Development CV only (S03/S04, one participant). Nothing here is an unbiased test score.

## Ground rules (apply to every step)

- **New code in a new package `lahacks_v2/`.** `lahacks_cv/`, its hash-pinned imports (`csi.py`, `csi_quality.py`, `preprocess.py`), datasets, checkpoints and reports stay untouched. v2 *imports* them and never edits them.
- **Disk (~0.5 GB free):** no new 512-point tensor datasets on disk. Windows are rebuilt in memory from the pinned raw cache. Only small feature caches, predictions and checkpoints (<100 KB each) are written, all under `reports/ml_v2/`.
- **Folds A (03→04) and B (04→03) are fixed.** Every fitted thing (scalers, thresholds, label offsets, model weights) is fit on the training block only. Thresholds: a training-block quantile of normal-window scores, declared before looking at validation.
- **CPU only, local, no Veri.** Seed 42.
- **One change per experiment.** Every run is scored by the same E0 evaluator on the same window population.

## Repository cleanup: lightweight only

A full reorganization is **skipped**. `blissey-model/` has no git of its own (no undo), and `lahacks_cv` hash-pins the top-level modules, so moves would break verification. Instead (~15 min):
1. `blissey-model/START_HERE.md`: what's current (`lahacks_v2/`, `reports/ml_strategy_review/`, `reports/ml_v2/`), what's baseline v1 (`lahacks_cv/`, job `d4c10e9207e8`), and what's legacy (MOMENT top-level scripts, `cnn_baseline/`, `*_v3.py` drafts, `data/processed/`).
2. A one-line banner at the top of `README.md` pointing to `START_HERE.md`.
3. No deletions. Deleting `data/processed/` (233 MB, legacy MOMENT/Cal Poly arrays) would relieve disk pressure, but that's the user's call.

## Action 1: evaluation v2 (E0), no training

`lahacks_v2/windows.py`
- Rebuild **every** grid window (3 s / 0.5 s) for all 92 recordings from the raw cache, using the unchanged quality-v2 gates.
- Keep the windows that pass quality. Tag each with recording, block, activity, event id, and role: `normal_recording`, `pre_action` (ends ≥0.5 s before start), `in_event` (overlaps [start, end]), `post_event`.
- Labels: `label_v2` (the existing rule) and `label_onset` (E3, filled in later).
- Transforms: `raw` (identical to v1) and `gain` (per-sample common-mode division, then per-window channel-median DC removal).

`lahacks_v2/evaluate.py`
- Window AUC / AP on the `label_v2` population (comparable to v1), reported per fold. No pooling across models.
- Alerts: the unchanged `ALERT_POLICY`, run over *all* passing windows of each recording in time order.
- **Detection:** any alert with window end in [start, end + 2 s].
- **False alerts:** alerts in normal recordings, plus alerts in fall recordings before start. Exposure is reported separately for each. Post-event alerts are reported separately and not counted as false alerts.
- Structural ceiling per event (can a 3-window run exist at all?).

Deliverable: re-score the saved Fold B CNN/RF (inference only) under E0, as a v1-vs-E0 table. Fold A cannot be re-scored because its checkpoints were lost.

## Action 2: gain-normalized unfitted baseline (E1)

- Score = mean |Δ| of the 150 ms-smoothed, gain-normalized window. No training.
- Threshold = 95th percentile of training-block normal windows.
- Success: AUC ≥0.85 in both folds, and E0 event recall above RF v1 at a lower or equal false-alert rate.
- Includes an offline/live parity test: the `gain` transform must equal what the live path computes from the same packets.

## Action 3: model ablations (one change each, all scored by E0)

| Step | Change vs previous | Keep if |
|---|---|---|
| E2a | RF (same config, same 575 features) on `gain` windows | AUC better than v1 in both folds |
| E2b | CNN (same config) on `gain` windows | as above; Fold B normal false positives <10% |
| E3 | Best of E1/E2 with onset labels: positives are windows centred in [start + δ₁, start + δ₂], offsets fit on the training block; pre-action windows become negatives | E0 event recall ↑ at ≤ false alerts/h |
| E5 | Small temporal model: logistic regression on E1-style features of the current + previous 2 windows | Beats E1 *and* E2 in both folds |
| E4 (optional) | Gap gate 0.5 s + interpolation-fraction feature | Interpolation fraction alone is not predictive (AUC < 0.6) on training labels |

**Pre-declared selection:**
1. Higher **minimum** over folds of E0 event recall (covered events), subject to false alerts/h no worse than the E1 baseline.
2. Tie-break: higher minimum fold AUC.
3. Second tie-break: the simpler model.

**Final artifact:**
- Refit the selected configuration on S03+S04 with frozen settings and save it under `reports/ml_v2/final/`, labelled **UNVALIDATED: no frozen test block**.
- Include a CPU latency check and a raw-packets → alert replay parity test.

## Not doable with current data (needs `session_05`)

- An unbiased score.
- A credible false-alerts/h figure.
- Breaking the recording-order confound.
- Generalization to other participants.

Recommended collection: fix acquisition first (≥90% of normal windows passing quality), then record interleaved activities with an "impact" marker, and freeze the block as the test.
