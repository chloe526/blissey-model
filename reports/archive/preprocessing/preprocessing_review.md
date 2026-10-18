# Preprocessing review — October 17, 2026

**Ready for an approved one-step GPU smoke test. Not enough validation evidence for reliable checkpoint selection. No GPU job launched. Existing preprocessing config, arrays, final test labels and window manifest remain byte-for-byte unchanged.**

| Split | Collection group | Source / contributing trials | Annotated / represented fall events | Normal / fall windows |
| --- | --- | --- | --- | --- |
| Train | session_02 | 30 / 25 | 10 / 10 | 770 / 49 |
| Validation | session_01 | 24 / 20 | 4 / 3 | 292 / 6 |
| Final test | session_03_LAHACKS | 42 / 35 | 4 / 4 | 889 / 17 |

1. **Tensor compatibility passes.** Float32 `[B,52,512]` and all-valid `[B,512]` masks match the [pinned official classification implementation](https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/momentfm/models/moment.py). Native multichannel encoding produces 64 patches/channel; `concat` gives a 53,248-input classifier and `[B,2]` logits. Construction and forward both select `concat`. Meta-device architecture validation passed; actual pretrained inference awaits Veri.

2. **Recording/session separation passes.** All **2,023 windows** reconstruct from raw packets with matching labels, provenance, source hashes and normalized features. Training-only refitting matches the scaler. No shared recording IDs, duplicate source files, cross-group capture-time overlap or packet identity mismatch was found. There are **three observed collection sessions, one per split**. Four metadata IDs include two within the same contiguous training collection block. Participant p1 appears in validation/test and training metadata for normal trials: this is not participant-disjoint evaluation. Four mixed-source training trials remain excluded.

3. **Distinct events, not windows, limit support.** Eighteen paired action intervals exist: 10 train, 4 validation, 4 test. Validation retains positive-window counts **[0,1,1,4]** per event. Trial `20261012_023117_618940Z` loses all four eligible windows to density rejection. Six overlapping positives therefore represent only three events in one session.

4. **Label policy discards informative boundary candidates.** Full containment plus a 0.25-second inner margin yields **49/13/28** positive candidates before quality filtering, **49/6/17** afterward. Holding current quality rules fixed, removing only the margin would yield **57 train / 9 validation** positives; half-window overlap would yield **117 / 22**. These are unimplemented diagnostics, not verified alternative labels. Review development annotations and predeclare overlap/uncertainty rules before broadening labels. Approximate action intervals do not identify impact. Much excluded time is baseline; excluded windows are not all missed falls. Keep final test labels/windows frozen.

5. **Quality failures combine two different rules.**

   | Split | Gap only | Density only | Both | Rejected |
   | --- | ---: | ---: | ---: | ---: |
   | Train | 1 | 82 | 11 | 94 |
   | Validation | 51 | 230 | 514 | 795 |
   | Test | 55 | 515 | 604 | 1,174 |

   Reciprocal median interarrival overestimates rate under bursty sampling. Median trial wall-clock rates are **54.69/43.39/33.40 Hz**, versus reciprocal-median-gap estimates **82.86/155.21/71.85 Hz**. This inflates the density expectation. Increasing only the gap limit from 0.25 to 0.5 seconds retains **819→820, 298→349, 906→956** windows: it does not solve the main problem.

   Keep **0.25 seconds provisionally**: it already spans about 11 nominal validation or 8 test packet periods; longer interpolation can erase brief dynamics. Compare device/host timing before changing clocks. Investigate a calibrated wall-clock density rule on development data. A per-recording mean-rate diagnostic with unchanged gap/label rules retains **901 train / 525 validation** windows, including 10 validation positives across all four events. This is not proof of signal quality; no artifacts or thresholds were changed based on test retention.

6. **Imbalance needs an effective training-only remedy.** Positive prevalence is **5.98%/2.01%/1.88%**. Existing weights `[0.5318,8.3571]` canceled at batch size one because of [PyTorch weighted-mean reduction](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html). Fixed using the mean of weighted per-example losses; gradient regression tests pass. Clipping can still attenuate scaling. Prefer training-only event/class-balanced sampling with unweighted loss, or larger effective batches accumulating weighted losses before clipping. Never oversample validation/test; report recall/F1 and event/alert metrics alongside accuracy.

7. **Current checkpoint selection is unreliable.** Three represented validation events in one session cannot support stable F1 selection; event recall moves in thirds. Always-normal validation accuracy is already 97.99%. Recovering the fourth event helps but does not resolve this. Collect additional development sessions/participants and annotated falls. Until then, use predeclared short diagnostic runs or explicitly limited two-session cross-validation on train/validation only. Keep final test out of all selection decisions.

## Prepared smoke test

The refreshed allowlist contains frozen NPZ files and current locked dependencies. Explicit `--data-dir data/processed` avoids accidental redirection through `VERI_DATA_DIR`. Veri downloads actual pinned **AutonLab/MOMENT-1-large**, runs one training window/one backward pass/one optimizer update, validates one batch, saves `$VERI_OUTPUT_DIR/best.pt`, and compares logits after reload. Test windows are never used for those model passes. One-batch F1 is diagnostic only.

```bash
cd /Users/chloe/chloe526.github.io/blissey/blissey-model
PYTHON=.venv311/bin/python bash scripts/submit_veri.sh smoke --dry-run
```

Replace `--dry-run` with `--execute` **only after explicit approval**. The exact staged Veri command is in `preprocessing_review_smoke_dry_run.txt`. Prior unmeasured smoke estimate: $0.09–$0.26 at the $1.05/hour snapshot; runtime and current pricing were not measured in this review.

**Evidence:** `preprocessing_review.json`, `preprocessing_review_tests.txt` (15 passed), `preprocessing_review_smoke_manifest_check.json`. Reproduce the read-only audit with `.venv311/bin/python review_preprocessing.py`. Changed only audit tooling/reports, weighted loss, runtime group-membership checks and explicit bundled-data paths. Pretrained GPU execution remains pending.
