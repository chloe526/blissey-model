# ML v2 results: gain normalization + evaluation v2

Status: **development cross-validation only.** One participant, two time-ordered blocks, no frozen test block. Fold A trains on S03 and validates on S04; Fold B trains on S04 and validates on S03.

How these results were produced:
- All models were trained locally on CPU; no Veri.
- Existing datasets, checkpoints and reports are unchanged.
- Thresholds are the pre-declared 95th percentile of training-block normal scores (out-of-fold for trained models). No validation block was used to pick thresholds or models.

## What changed

1. **Evaluation v2 (`lahacks_v2/evaluate.py`).**
   - Scores all 1,929 quality-passing windows, including pre-action and post-event windows in fall recordings.
   - An alert ending in [action_start, action_end + 2 s] counts as a detection.
   - Alerts in normal recordings, and alerts before action_start, count as false alerts.
   - The window AUC still uses the fixed v1 labeled population (1,550 / 52), so it stays comparable.
   - Alertable events (those with at least 3 contiguous windows, as the alert rule needs): 7/30 in S04 and 4/4 in S03.
2. **Gain-normalized input (`lahacks_v2/windows.gain_normalize`).** Each sample is divided by its across-subcarrier mean, then each channel by its window median. The transform is stateless and verified identical offline and live.

## Results (validation block; AUC on the v1 population; events and false alerts at the pre-declared threshold)

| Model | AUC A / B | Events detected A / B | False alerts/h A / B |
|---|---|---|---|
| v1 saved CNN (Fold B only, re-scored) | – / 0.36 | – / 4 of 4 | – / 556 |
| v1 saved RF (Fold B only, re-scored) | – / 0.73 | – / 1 of 4 | – / 16 |
| CNN, raw input (v1 config, local) | 0.45 / 0.34 | 2/7 · 4/4 | 100 / 568 |
| RF, raw input (v1 config, local) | 0.63 / 0.73 | 5/7 · 1/4 | 111 / 49 |
| **E1** unfitted motion score (gain) | **0.94** / 0.86 | 1/7 · 4/4 | 11 / 210 |
| **E2 RF, gain input (selected)** | **0.89 / 0.89** | **4/7 · 3/4** | 45 / 91 |
| E2 CNN, gain input | 0.40 / 0.72 | 0/7 · 1/4 | 0 / 21 |
| E5 context logistic regression (gain) | 0.79 / 0.82 | 4/7 · 3/4 | 56 / 152 |
| E3 RF, gain input, onset labels | 0.91 / 0.84 | 4/7 · 2/4 | 45 / 62 |

Details are in `results.json`. Scores are in `scores_*.npz`. The threshold sweep in `tradeoff_descriptive.json` is descriptive only and was not used for selection.

## Conclusions

- **Gain normalization is the decisive fix.**
  - The same RF config goes from AUC 0.63/0.73 to 0.89/0.89.
  - The CNN's fold B false-positive flood disappears: normal windows flagged at 0.5 drop from 92% to 5%.
- **The RF on gain input is selected under the pre-declared rule:**
  1. It has the best minimum-fold AUC (0.89).
  2. It ties for the best minimum-fold event recall (4/7 and 3/4).
  3. It produces fewer false alerts than the alternative with the same recall (the context logistic regression).
  - Caveat: it does **not** meet the "≤ E1's false-alert rate" condition in fold A (45 vs 11/h).
- **The CNN is not worth pursuing on this data.** Even with gain input it fails fold A, which has only 3 training falls.
- **Onset labels (E3) and temporal context (E5) give no reliable gain.** In fold B, the onset window fit on the training block was degenerate (1 positive).
- **Alert rates are not deployable yet.**
  - At about 20 false alerts/h, roughly 4/7 and 2/4 events are detected (descriptive sweep).
  - Normal exposure is only 3.8 min (S04) and 13.8 min (S03), so each false alert moves the rate by 4–16/h. These rates are rough.

## Final artifact: `final/model.joblib` (UNVALIDATED)

| Property | Value |
|---|---|
| Training data | RF (300 trees, depth ≤6), v1 config, refit on S03+S04: 1,550 normal / 52 fall windows (13 events) |
| Input | Gain-normalized windows, then the existing 575 window-local features |
| Threshold | 0.205 (out-of-fold 95th percentile of normal windows) |
| Alert policy | Unchanged v1: 2-of-3 votes, 5 s cooldown |
| Size / latency | 214 KiB; ~15 ms median CPU per window (Mac) |
| Streaming check | `lahacks_v2.final.Detector` takes raw packets and reproduces offline probabilities (difference ≤1e-16) and alerts on 4 replayed recordings |

The detection threshold is the main knob to adjust after a real test.

**Before claiming performance:**
- Record a fresh interleaved `session_05`, fix acquisition first, and keep it frozen.
- Score this exact artifact once with `lahacks_v2.evaluate`.

Reproduce:
```
.venv311/bin/python -B -m lahacks_v2.test_v2 -v
.venv311/bin/python -B -m lahacks_v2.run          # skips experiments already in results.json
.venv311/bin/python -B -m lahacks_v2.final        # refuses to overwrite final/model.joblib
```
