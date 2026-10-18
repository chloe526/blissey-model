# Stage 1 gate: LA Hacks only

Audit date: October 17, 2026, America/Los_Angeles. Repository pinned to **06c97032c93a43ddccda5aa8381a95bced54755c** ([recordings at this commit](https://github.com/chloe526/blissey-data/tree/06c97032c93a43ddccda5aa8381a95bced54755c/recordings)). Stage 1 took approximately 8 minutes including data retrieval and verification. No training, new labels, dataset materialization, model scoring, Veri calls or job submissions occurred.

**Gate result: development-only grouped CV is feasible; the requested independent whole-block test of 6–8 falls is not currently feasible.** There are only two LA Hacks collection blocks and one participant. Reserving either entire block for testing leaves one block, which cannot support grouped development CV. Randomly choosing six falls out of the same development block would violate the requested whole-block separation.

The user confirmed that Sessions 03 and 04 are genuinely separate physical collection blocks and that no additional blocks exist yet. Both folders reuse metadata session ID `session_202610172026`; the explicit folder/block assignment and this confirmation take precedence over that shared metadata string. No recording from Sessions 01 or 02 enters this audit or the proposed modeling pool.

**Available data, under unchanged quality-v2 labels and signal gates**

| Block | Participant | Recordings | Annotated fall trials / quality-covered events | Normal recordings / contributing | Dynamic hard-negative recordings / contributing | Accepted normal / fall windows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 03_LAHACKS | p1 | 42 | 4 / 3 | 38 / 36 | 21 / 21 | 1,249 / 21 |
| 04_LAHACKS | p1 | 50 | 30 / 10 | 20 / 15 | 20 / 15 | 301 / 31 |
| Total | **1 unique person** | **92** | **34 / 13** | **58 / 51** | **41 / 36** | **1,550 / 52** |

There are **1,602 accepted windows**, with 3.25% labeled fall, representing only **13 quality-covered annotated events**, not 52 independent falls. All 34 fall trials have one valid paired operator action interval; no trial is excluded for missing/ambiguous annotations or mixed radio sources here. A quality-covered event means at least one positive window under the existing v2 label rule and quality gates. It does not establish the precise moment of descent/impact.

Dynamic hard negatives are explicitly defined as verified walking, stand-up, sit-down, lie-down and get-up-from-floor recordings. Other verified normal/static categories remain in the normal total. Session 04 contributes ten lie-down and ten sit-down trials; retained windows are only **43 lie-down / 258 sit-down**. Five of its normal recordings have no accepted window. Its broader normal/background coverage is therefore substantially narrower than Session 03's.

**Acquisition limits dominate the new block**

| Block | Label-eligible windows | Accepted | Quality rejected | Ignored by label policy | Falls lost to quality | Falls lost to v2 minimum-duration geometry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 03 | 2,080 | 1,270 | 810 (38.9%) | 188 | 1 / 4 | 0 |
| 04 | 1,273 | 332 | 941 (73.9%) | 1,427 | **20 / 30** | **0** |

Session 04's action intervals span **4.93–7.66 s**, median **6.87 s**. Every event has at least one candidate positive under the old containment rule; twenty lose all such candidates to quality. These broad operator intervals do not prove that the physical falls themselves lasted that long.

Session 04 has **869 excessive-gap**, **665 temporal-coverage**, **457 unsupported-edge**, **95 sample-span** and **20 effective-rate** rejection reasons; reasons overlap. Its median recording rate is 41.49 Hz, with individual rates as low as 15.67 Hz. Mean packet rate alone misses these outages. There are 684 host inter-packet gaps over 250 ms, and **681 also exceed 250 ms in device receive timestamps**. Only three corresponding host gaps have positive device gaps at or below 250 ms. Host-time duplicates number just 17. Acquisition/receive gaps, not solely serial timestamp batching, are therefore a material problem; the audit cannot attribute them specifically to interference, firmware capture or transport backpressure.

Eighteen new events have some high-quality window overlapping their broad action interval, but only ten have a quality-passing v2 positive. The other eight are **not automatically recoverable falls**: a passing overlap could contain preparation or recovery instead of descent. Twelve new events have no passing grid window overlapping their action interval at all. A label-v3 rule cannot repair missing measurements. No filters were relaxed and no interval centers were treated as impacts.

Accepted normal monitored exposure is only **13.83 minutes** in Session 03 and **3.79 minutes** in Session 04, **17.62 minutes total**, after merging overlapping windows. Raw-duration totals and overlapping-window counts would overstate evaluated normal exposure. Outages and uncovered events must remain separately reported.

The new block also has class/time confounding: all thirty staged-fall recordings were collected before the lie-down batch, followed by the sit-down batch. Do not invent new independent blocks by splitting these class-ordered runs. Future blocks should interleave activities while preserving genuine block boundaries.

**Feasible assignments for review, not applied to any dataset**

| Development fold | Model-fit block | Training quality-covered events | Validation block | Validation quality-covered events | Frozen test |
| --- | --- | ---: | --- | ---: | --- |
| A | 03 | 3 (21 fall windows) | 04 | 10 (31 fall windows) | None |
| B | 04 | 10 (31 fall windows) | 03 | 3 (21 fall windows) | None |

Both folds have both classes, so identical grouped CV is mechanically possible for CNN, RF and an energy baseline. Fold A trains on only three independent events; fold B's training normal set contains only lie-down and sit-down. Performance and threshold selection will be unstable. Threshold/smoothing settings chosen on these folds are development choices; the selected development score is not an unbiased final-test estimate. Report per-block metrics and pooled out-of-fold event/alert results, not window accuracy alone.

If approved for this fallback, the final development-grade refit would use **03 + 04: 13 represented events, 1,550 normal windows and 52 fall windows** and must be labeled **UNVALIDATED — no frozen independent test**. No existing events have been assigned to a new test set.

**Preferred route to an actual test:** collect a fresh, physically separate LA Hacks block (for example `session_05_LAHACKS`) with approximately **6–8 quality-covered annotated falls**, matching hard negatives and continuous ordinary/background activity. Keep all of this block out of fitting, grouped CV, threshold selection and smoothing selection. Record all attempted falls, including acquisition failures; do not silently discard failed attempts from system-level reporting. Keep Sessions 03 + 04 as the development pool with the folds above. More participants would improve scope, but a new block with only p1 still evaluates only within-person, venue-specific behavior.

Before staging more falls, verify acquisition on a normal pilot with the existing quality gates. A one-second or several-hundred-millisecond capture outage will not be fixed by a larger GPU. Preserve the intended transmitter filter, radio/firmware/cases, board placement and timing diagnostics; investigate the receive gaps rather than changing the quality rule to fit these recordings. Interleave positive and negative activities in the new block, and review timing annotations before scoring models.

**Stage 2/3 constraints to preserve**

- Stage 2 starts with the existing v2 label semantics and unchanged quality gates. Extract inputs from LA-only raw data; fit any scaler, learned PCA or other learned transformation separately on each training fold. Do not reuse the old Cal Poly-fitted array normalization. Per-window median normalization or per-window PCA summaries must use only the current window. None of these operations was fitted here.
- Compare CNN, RF and energy under the same grouped folds and alert policy. Tune thresholds/smoothing only from development predictions. Keep RF's requested variance-by-thirds, peak/end ratio, band-power, PCA-variance and median-normalized features for Stage 2 implementation. Hardware selection on Veri is deferred until approval; faster GPU compute does not automatically accelerate RF/CPU feature work. No MOMENT or Colab.
- The event-centered v3 files from interrupted earlier work remain **unfinished drafts**. Stage 1 does not import or use them. Existing new annotations are still approximate action clicks; a reviewed descent anchor/bounds are needed before claiming a v3 label experiment is valid. Do not fabricate a midpoint or silently activate v3 in the baseline.
- For a reportable frozen-test score, first refit the selected configuration on all non-test development data with frozen settings, then score the fresh test once and deploy those exact weights. Scoring one checkpoint and subsequently refitting a different deployment checkpoint would make that score no longer a test of the deployed weights. If no new block arrives, Stage 3 may produce only an explicitly unvalidated artifact and inference parity checks after approval; it cannot create an independent test retrospectively.

**Verification and reproducibility**

The public tree was pinned before downloading the new block. Only metadata, annotations and measurement JSONL files were downloaded to a temporary cache (approximately 93.3 MB); raw serial binaries and Cal Poly recordings were not downloaded. Existing local Session 03 files were reused after their Git blob hashes matched the same pinned commit. `source_manifest.json` records SHA-256 hashes of all 254 inspected source files. Session 03 exactly reproduces its earlier 1,249/21 accepted class counts.

The audit uses only `csi.py`, v2 `preprocess.label_window`/`recording_features`/`transform_window`, `csi_quality.py`, and the frozen v2 geometry/carrier/quality configuration. Each passing window exercises the real raw-to-finite-float32 `[52,512]` transform without normalization; no tensor arrays are saved. Cal Poly mean/std values are not applied. `window_audit.jsonl` records each window's outcome and annotated event ID. `acquisition_diagnostics.json` separates host and device timing evidence.

Reproduce while the temporary cache exists:

```bash
cd /Users/chloe/chloe526.github.io/blissey/blissey-model
PYTHONDONTWRITEBYTECODE=1 .venv311/bin/python -B reports/lahacks_stage1/audit.py \
  --root /tmp/blissey-stage1-06c97032c93a \
  --revision 06c97032c93a43ddccda5aa8381a95bced54755c
```

Stage 1 ends here for user review. The choice is development-only Stage 2 with explicit unvalidated status, or obtaining a new frozen test block before final evaluation. No later-stage work is authorized by this audit alone.
