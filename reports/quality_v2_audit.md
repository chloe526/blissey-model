# CSI quality-filter revision — October 17, 2026

**Revised arrays are ready for review. All 24 tests and full source/label/scaler reconstruction checks passed. No model weights were downloaded and no Veri job was launched.**

| Split | Accepted windows, before → after | Quality-rejected, before → after | Normal / fall windows, before → after | Represented fall events, before → after |
| --- | --- | --- | --- | --- |
| Train | 819 → 835 | 94 → 78 | 770 / 49 → 786 / 49 | 10/10 → 10/10 |
| Validation | 298 → 465 | 795 → 628 | 292 / 6 → 457 / 8 | 3/4 → 4/4 |
| Test | 906 → 1,270 | 1,174 → 810 | 889 / 17 → 1,249 / 21 | 4/4 → 3/4 |

Counts are correlated sliding windows, not independent events. Recording groups remain session_02/session_01/session_03_LAHACKS; four mixed-source training trials remain excluded. Annotation intervals, label policy, channel selection, three-second duration and 0.5-second stride are unchanged. Every retained common window keeps its original label. Training-only normalization was refitted on the 835 accepted training windows and applied consistently to all splits.

## Cause and timestamp evidence

The old filter used `1 / median(interpacket gap)` as an expected rate. Bursts have many short gaps and fewer long pauses, so the median gap describes within-burst cadence rather than wall-clock throughput. Median trial rates in training/development were 54.69/43.39 Hz; the reciprocal-median-gap estimates were 82.86/155.21 Hz.

The firmware emits radio receive timestamp at legacy field 18 and device steady-clock seconds at field 23. The collector records monotonic elapsed seconds and UTC receipt time; all lines within one serial chunk share the latter timestamps. Development radio timestamps were increasing with no duplicates. Median host/radio span disagreement was about 1.3–1.4 ms over a 30-second trial; individual trials reached about 52 ms. Both radio acquisition and serial receipt can be irregular; host bursts cannot be interpreted as a uniform high-frequency sensor.

The host elapsed timeline is retained for interpolation and event alignment. No radio clock retiming or annotation shifts were introduced. Duplicate host timestamps retain the first packet and count once; multiple copies cannot inflate density. The receiver captures one AP/STA link over 921600-baud serial. Source firmware defaults to a 100-packet/s TX setting but contains no saved flashed configuration proving that rate; it is not treated as delivered CSI frequency.

## Frozen quality policy

`configs/packet_quality_v2.json` was locked against **label-blind session_01/session_02 timing only**, before regenerating test data. The development trials delivered 26.37–85.94 packets/s. Test labels/performance did not choose thresholds.

A window must satisfy **all** of these checks:

- At least **20 unique-timestamp packets/s**: at least 60 actual packets in three seconds. This conservative floor is below the slowest observed development trial, while rejecting clearly sparse streams; it is not a clinical adequacy guarantee.
- Actual samples occupy at least **80% of 100 ms bins**: at least 24 of 30 bins. Dense bursts cannot compensate for missing wall time.
- At least **90% first-to-last sample span**, with an actual sample within **100 ms of each endpoint**.
- No interpolation-bracketing gap over **250 ms**, unchanged from v1. No extrapolation, non-finite inputs or all-zero CSI windows are accepted.

This is a coverage requirement, not a relaxed count quota. It newly rejects 25 old validation and 66 old test windows while admitting 192/430 others. Its thresholds remain frozen despite those results. Single-reason and overlapping-reason rejection counts are saved in the JSON audit.

## Sensor coverage and a remaining blind spot

Coverage below merges overlapping accepted input windows. It measures available labelled sensor time, not clinical detection performance. All splits retain **52 feature channels from one radio link**.

| Split | Accepted labelled seconds, before → after | Fraction of eligible recorded time | Fraction of label-candidate time |
| --- | --- | --- | --- |
| Train | 476.5 → 487.5 | 61.2% → 62.6% | 91.4% → 93.5% |
| Validation | 220.5 → 376.0 | 30.7% → 52.4% | 36.4% → 62.0% |
| Test | 665.5 → 847.5 | 53.0% → 67.4% | 58.1% → 74.0% |

Fall-trial baseline and uncertain boundary time remain unlabelled, so full-recording coverage cannot reach 100% under this label policy.

**One test fall is now uncovered:** `20261017_214356_001779Z`. Its previously accepted window had 22 packets/s but only 23/30 occupied bins and a 212 ms unsupported endpoint. It fails the frozen coverage/endpoint rules. The event annotation remains present; it must be reported as uncovered rather than silently removed or called a successful model detection. No test-driven threshold adjustment was made. Validation now represents all four events, but eight positive windows from one session are still insufficient for reliable checkpoint selection.

## Artifacts and verification

- Original artifacts, config, preprocessor source and checksums: `data/processed_legacy_v1/`.
- Updated tensors: `[835,52,512]`, `[465,52,512]`, `[1270,52,512]` under `data/processed/`; inputs remain MOMENT-compatible float32. NPZ files also retain per-window quality diagnostics.
- Evidence: `quality_threshold_lock.json`, `development_acquisition_characterization.json`, `quality_v2_comparison.json`, `quality_v2_tests.txt` and `quality_v2_verification.txt` in this reports directory.
- Tests cover regular and bursty sampling, long gaps, insufficient packets, concentrated bursts, endpoint coverage, duplicate timestamps, non-finite/absent signal, split integrity and preprocessing equivalence.

Reproduce from `blissey-model`: `.venv311/bin/python preprocess.py`, then `.venv311/bin/python audit_quality_v2.py` and `.venv311/bin/python -m unittest discover -s tests -v`. The archived v1 artifacts are not overwritten by these commands. Old v1 checkpoints must use their saved v1 scaler/artifacts; do not mix versions. Full training remains deferred for review.
