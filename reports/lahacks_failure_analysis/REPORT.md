# LA Hacks-only CNN/RF failure analysis

Scope: saved quality-v2 arrays, completed Veri job `d4c10e9207e8`, exported predictions, and the existing recording/window audit. No training or threshold optimization was run. Both blocks contain participant p1; this is a two-block, single-participant comparison, not a population generalization estimate.

## Cross-block failures

| Validation block | CNN at fixed 0.50 | Random Forest at fixed 0.50 | Event coverage / detection |
|---|---:|---:|---|
| Session 04 (train 03; 332 windows, 31 fall) | 0 TP, 31 FN; 0 FP | 0 TP, 31 FN; 0 FP | 10/30 covered falls; all covered events missed |
| Session 03 (train 04; 1,270 windows, 21 fall) | 21 TP, 0 FN; 1,130 FP | 3 TP, 18 FN; 24 FP | 3/3 covered by CNN, 3/3 RF; detections 3/3 CNN and 1/3 RF |

Session 03 CNN false positives are spread across activities: lie-down 199/199 windows, sit-down 144/144, stand-up 220/221, walking 189/189, get-up-from-floor 70/75, empty room 133/167, sitting-still 106/106, standing-still 69/69. RF has fewer false positives, but misses 18/21 fall windows. Both models fail on every Session 04 fall window. Recording-level counts and event IDs are in `errors_by_recording.csv` and `fall_events.csv`.

## Why

A strong acquisition shift is visible even among normal windows: raw median CSI amplitude is 16.22 in Session 03 and 18.90 in Session 04 (standardized difference −1.91); first-difference variance differs by −0.44 SD. The windows overlap and are not independent. Session 04 negatives are only sit-down/lie-down and were collected in a different, class-ordered block; the model can learn block/activity cues rather than falls. In Session 03, simple variance/energy distinguish falls from lie-down reasonably (rank AUC 0.88) but only weakly in Session 04 (0.59). Sit-down separation is moderate (0.80–0.85); get-up-from-floor separation is weak in Session 03 (variance AUC 0.62; absolute-difference AUC 0.47). These are univariate window descriptions, not fitted classifiers.

Of 34 annotations, 13 have at least one quality-v2 positive window (Session 03: 3/4; Session 04: 10/30); 21 do not. All 21 have candidate positive windows, so minimum label geometry alone did not erase them. Rejected candidate windows most often breach the 250 ms gap gate (161), then temporal coverage (111), edge support (83), sample span (20), or effective rate (13); reasons overlap. Among the 21 uncovered events, 12 have no passing-quality window overlapping the annotated action at all, while 9 do have such a window but no eligible positive window under the v2 label boundary. The latter indicates possible annotation-boundary mismatch and needs review, not automatic relabeling. Per-event candidate/rejection counts are in `event_coverage.json` and `fall_events.csv`.

## Threshold tradeoff and limits

At the existing 0.50 policy on Session 03, CNN detects all 3 covered events but yields 551 false-alert episodes/hour; RF detects 1/3 with 17.4/hour. In the descriptive sweep, RF recall falls as false alerts fall; CNN keeps high recall only with hundreds of false alerts/hour. This fold has just 829.5 seconds (13.8 minutes) of accepted normal exposure, so hourly rates are unstable. Both folds together have 17.6 minutes. Do not select a threshold from this held-out block.

Fold B score histograms are plotted in `probabilities_by_fold.png`. Veri flattened Fold A artifact paths, leaving no per-window probability arrays/checkpoints in the available export; only its aggregate confusion counts survive. Its panels therefore show the known `<0.50` result, not invented distributions. This prevents a complete two-fold probability-distribution comparison.

## Highest-impact next steps

1. Repair and verify packet capture/coverage first; pilot each staged fall against the unchanged v2 gates and retain uncovered attempts in the audit.
2. Collect across multiple participants and additional physical blocks; interleave falls, sit/lie/get-up, and empty-room trials over time. Keep a new block frozen for final evaluation.
3. Time-sync and review the 9 action-overlap/label-boundary cases; fix artifact path collisions so every fold’s probabilities, event predictions, and checkpoints are retained. Then compare models only on new grouped development blocks.

Plots: `probabilities_by_fold.png`, `session_feature_shift.png`, and `threshold_tradeoffs_fold_b.png`. Detailed CSV/JSON tables accompany this report.
