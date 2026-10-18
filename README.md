# Blissey Beacon — MOMENT / Veri experiment

> **Legacy document.** This README describes the historical MOMENT pipeline. For the current LA Hacks pipeline, read [START_HERE.md](START_HERE.md).

Implemented for the recordings present on October 17, 2026. This directory contains the ML pipeline only. The application and raw recording tree are unchanged. The approved L4 smoke job `261036b21c27` completed with a real pretrained forward/backward update and verified checkpoint reload. No additional paid job has been launched. Its one-window diagnostic score is not full validation or test performance. See the [full-training preparation report](reports/moment_full_training_preparation.md) for evidence, artifact hashes and commands.

## Current results and limitations

The latest pipeline uses quality policy v2. Read the [quality revision audit](reports/quality_v2_audit.md) before training. Current splits have **835 / 465 / 1,270** windows, with **49 / 8 / 21** positive windows and **10 / 4 / 3** represented fall events. There are still four annotated test events; one lacks a usable window under the frozen quality rules. Test labels and annotation intervals are unchanged. The earlier [preprocessing review](reports/archive/preprocessing/preprocessing_review.md) records the historical v1 baseline. Current arrays and configuration are in `data/processed/`. Current local tests: **38 passed**. Checkpoint selection remains unreliable with this small validation set.

For a read-only current-data audit run `.venv311/bin/python audit_quality_v2.py` or `.venv311/bin/python review_preprocessing.py`. The preprocessing command and general smoke script regenerate current arrays. The class-weight cancellation fix and runtime group-membership guards remain in place.

`reports/dataset_audit.json` audits all 96 recordings, 96 measurement JSONL files, 96 metadata files, 53 event JSONL files and 96 raw serial files. There are four free-text metadata session IDs, but **three physical collection sessions** and two participant IDs. All recordings explicitly declare their category as a recording label. Eighteen staged-fall trials have paired operator action markers. There are 78 explicit normal trials; four contain multiple radio sources and are excluded, leaving 74 eligible normal trials. Missing event files are never used to infer a negative label.

Each measurement contains 128 signed int8 values: 64 complex LLTF pairs in imaginary/real order. This is the legacy 25-column firmware format, not the similarly sized Espressif example CSV format. The firmware's `len` field often says 384 because it describes the full CSI buffer, while the recording prints the first 128 LLTF bytes. Amplitudes are `hypot(imag, real)`. First-four-byte hardware uncertainty, guard bins and the DC bin are excluded. Training-session observations intersected with the physical LLTF active bins yield **52 channels**; the model channel count is derived from that selection, not assumed in the parser.

Firmware source enables LTF merging, disables channel filtering, and uses automatic CSI scaling. No host normalization was found. Packet rates across trials are approximately 21.67–85.94 Hz (median 45.25 Hz), with irregular arrivals. Host timestamps are UTC ISO 8601; packet elapsed time is monotonic seconds. UI action elapsed values sometimes differ from the measurement clock. Events are aligned using event host UTC minus the median measurement `(host UTC - elapsed)` origin. These remain approximate **action intervals**, not precise fall impact times.

The split is fixed before windows are created:

| Split | Physical session | Windows: normal / fall | Independent trials contributing windows |
| --- | --- | --- | --- |
| Train | session_02 | 786 / 49 | 26 (16 normal, 10 fall) |
| Validation | session_01 | 457 / 8 | 23 (19 normal, 4 fall) |
| Test | session_03_LAHACKS | 1,249 / 21 | 39 (36 normal, 3 fall) |

Arrays have shapes `[835,52,512]`, `[465,52,512]`, `[1270,52,512]`. All three include both classes. Validation has very few positive windows; one annotated test fall loses all windows to quality checks. These data support a **small supervised feasibility experiment** with staged falls. They cannot establish participant-independent, population, clinical or deployment performance. Session 01 and session 03 also share participant p1. Overlapping windows are correlated and are not independent trials.

## Preprocessing policy

`csi.py` is the reusable packet/amplitude parser. `preprocess.transform_window` is shared by offline preprocessing and raw-window inference.

* Only complete trials with explicit recognized category metadata are eligible. Mixed-source recordings and ambiguous fall annotations are excluded.
* Duplicate elapsed timestamps keep the first packet. Packets are sorted by elapsed time. Invalid layouts, signed-value ranges, timestamps and RX state are rejected and audited.
* Windows span three wall-clock seconds, stride 0.5 seconds. Expected native sample counts come from measured packet rates (per-trial rates and training reference rate are saved). Each channel is linearly interpolated directly onto 512 equally spaced timestamps, including the endpoints. Upsampling adds no information. There is no padding or extrapolation.
* Quality v2 requires at least 20 unique packets/s, samples in at least 80% of 100 ms bins, at least 90% sample span, an actual sample within 100 ms of each edge, and no interpolation gap above 250 ms. Thresholds were frozen from development acquisition behavior only; no median-gap rate estimate is used. Diagnostics and sensor coverage are in `reports/quality_v2_comparison.json`.
* Explicit normal-category windows are class 0. A staged-fall window is class 1 only if it fits entirely inside a paired action interval with a 0.25-second inner margin. Partial overlaps, boundary uncertainty and baseline portions of fall trials are excluded. No entire fall recording is labeled positive.
* Feature selection and per-channel mean/std fitting use training data only. Validation, test and inference reuse that scaler. MOMENT additionally performs its native per-window RevIN; this can remove absolute-level information and is part of this experiment.
* NPZ files retain recording IDs and window bounds; the config retains session/participant provenance, source measurement hashes, selected channels, event intervals, scaler and artifact hashes. Training verifies the hashes and session separation before loading weights.

The physical-session mapping is deliberately explicit. New session folders require reviewing the split policy rather than silently changing the evaluation.


## Folder layout and cleanup

- Top-level Python files: MOMENT training/evaluation/inference and shared CSI preprocessing/audits.
- `cnn_baseline/`: independent CNN code, configuration, checkpoints and measured results.
- `configs/` and `scripts/`: fixed quality/training settings, pinned data source and Veri packaging/entrypoints.
- `data/processed/`: authoritative quality-v2 train/validation/test arrays and original preprocessing config.
- `reports/`: current audits, quality evidence, smoke artifacts and verification results. `reports/archive/` retains historical diagnostics.
- `.venv311/`: the working local Python environment.

The obsolete `.venv`, generated caches, Finder metadata, empty output placeholders and duplicate hosting staging were removed. The duplicate `preprocessing_review_v2.json` had identical bytes to the retained `reports/quality_v2_comparison.json`; the review command can regenerate it. Source code, model settings, data and checkpoints are unchanged. See `reports/cleanup_manifest.json` for the exact cleanup and preservation checks.

Generated output directories are created by their scripts. Recreate a local hosting bundle only when needed:

```bash
.venv311/bin/python scripts/prepare_hosting.py
```

## Dependencies and safe local checks

Use the working `.venv311` environment with Python 3.11 locally and pinned official MOMENT source revision `38f7310ad594100747ca2a8357e9c7ca7d323e0e` (package version 0.1.5). It supports native multichannel classification. The exact pretrained model revision is `ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc`.

An isolated `.venv311` has already been installed in this workspace. From this directory:

```bash
cd /Users/chloe/chloe526.github.io/blissey/blissey-model
PYTHON=.venv311/bin/python bash scripts/check_environment.sh
PYTHON=.venv311/bin/python bash scripts/smoke_test.sh
```

The local smoke script **does not download weights**. It runs the audit, preprocessing, the weight-free unit/integration tests, official model import, CLI argument checks and shell syntax checks. These passed. Tests include actual session separation, finite tensors, duplicate handling, host-clock marker alignment, partial-overlap rejection, normalization, exposure union and reproduction of an unseen raw test window using the saved transform.

To reproduce the local environment on another compatible Mac with Python 3.11:

```bash
uv venv --python 3.11 .venv311
uv pip install --python .venv311/bin/python -r requirements-local.lock.txt
```

`requirements-local.txt` documents core pins; the lock captures the actual local transitive versions. `requirements-veri.lock.txt` resolves successfully for Linux x86_64 / Python 3.10 without installing GPU packages locally. `veri/base` documents Python 3.10, PyTorch 2.8.0+cu128, CUDA 12.8.1, transformers 4.56.2 and NumPy 2.1.3. The remote lock keeps torch 2.8.0 and layers the tested transformers 4.55.4, HF Hub 0.34.4, NumPy 1.26.4 and MOMENT revision on it. `fsspec==2025.3.0` also respects the base image's datasets constraints. Installed CUDA packages should satisfy the lock. The completed L4 smoke verified these core versions on CUDA; its actual run configuration and logs are saved under `reports/smoke_261036b21c27_*`.

The official model's architecture was constructed on PyTorch's `meta` device without allocating weights. Native head dimensions are 53,248 -> 2, with 106,498 trainable head parameters and 341,240,320 frozen parameters. Architecture/config reconstruction and state-dict shapes matched. The local architecture check did not execute pretrained weights. Actual pretrained execution and checkpoint reload subsequently passed on Veri job `261036b21c27`. The Mac lacks space for the full checkpoint, so weights remain remote. `smoke_test_moment.py` remains available for a future machine with enough storage but is not called by local checks.

## Veri upload packaging and dry run

The previous full-data code upload received HTTP 413 before any GPU job. Packaging is now mode-specific; see the [packaging fix report](reports/archive/packaging/veri_packaging_fix.md).

* **Smoke:** four actual quality-v2 windows (fall + normal from train and validation), original scaler/metadata, model code and requirements. No test array. Latest measured tar.gz: **399,720 bytes (0.381 MiB)** versus the rejected **244,775,769 bytes (233.436 MiB)**.
* **Epoch/full:** code-only upload. The GPU downloads three commit-pinned processed artifacts (train, validation and unchanged config) separately, verifies sizes/SHA-256 and preserves the numerical splits. No full arrays go through `code_artifacts.upload`.

Every submission validates the allowlist and measures the exact SDK tar/gzip stream before network upload. It stops above an **8 MiB local budget** with a 64 KiB multipart margin; the actual server limit is unknown. This budget is configurable through `VERI_MAX_UPLOAD_MIB`. Old staging paths contain full arrays and must not be reused.

```bash
cd /Users/chloe/chloe526.github.io/blissey/blissey-model
PYTHON=.venv311/bin/python bash scripts/submit_veri.sh smoke --dry-run
```

This creates only a tiny temporary package and prints the measured size and exact CLI command. It uploads nothing. The smoke fixtures retain real labels/provenance, source indices/hashes and the full training scaler. They are explicitly diagnostic and cannot be used as a full training dataset.

## Remote smoke — paid, requires approval

After explicit approval:

```bash
PYTHON=.venv311/bin/python bash scripts/submit_veri.sh smoke --execute
```

The remote script runs:

```bash
python train.py --data-dir data/smoke --smoke-data --epochs 1 --max-batches 1 --batch-size 1
```

This downloads the actual pinned **AutonLab/MOMENT-1-large** weights on Veri, runs one training window/one backward pass/**one optimizer update**, validates one batch, saves actual full state to `$VERI_OUTPUT_DIR/best.pt`, then reloads the checkpoint and compares logits. No pretrained weights come from the Mac. Expected final output:

```text
PASS: actual pretrained MOMENT; one optimizer step; checkpoint captured and reloaded.
```

Captured artifacts are `best.pt`, `run_config.json`, `history.json` and `reload_check.json`. The one-batch validation score is diagnostic only. `veri_sdk.log_metrics` charts validation F1 as the generic `reward` field. GPU execution, checkpoint capture and reload passed in the completed approved job; retrieved evidence is linked in the full-training report.

## Full training data source

Use the selected public **chloe526/blissey-data** GitHub repository for **train/validation only**, plus the unchanged preprocessing config. The checksummed hosting bundle can be regenerated as `hosting/quality-v2-trainval-58540c5b7f3d/` (123,091,645 artifact bytes) with `.venv311/bin/python scripts/prepare_hosting.py`. The duplicate staging copy was removed during cleanup; original datasets remain unchanged. Each array is below GitHub's 100 MiB regular-Git limit; train produces the >50 MiB warning. Test arrays, raw recordings, legacy artifacts and checkpoints are unnecessary for hosting/training. The unchanged config retains all split provenance and annotation metadata.

The active source manifest is `configs/dataset_source.json`, configured for commit `4727901ff595c3a79760c1785f540ec753084dc1`. The submission wrapper verifies the remote bytes before any job. This cleanup did not change or revalidate the remote source. The GitHub example is a historical template. The [preparation report](reports/moment_full_training_preparation.md) preserves the earlier preparation evidence and artifact hashes; its pre-publication availability observations describe that earlier snapshot.

To regenerate the source manifest after publishing a new reviewed dataset revision:

```bash
DATASET_COMMIT=REPLACE_WITH_ACTUAL_PUBLISHED_40_CHARACTER_COMMIT
.venv311/bin/python scripts/make_dataset_source.py \
  --backend github --repo-id chloe526/blissey-data \
  --revision "$DATASET_COMMIT" --path-prefix processed/quality-v2 \
  --output configs/dataset_source.json
.venv311/bin/python fetch_dataset.py --source configs/dataset_source.json \
  --verify-only --splits train val
PYTHON=.venv311/bin/python bash scripts/submit_veri.sh full --dry-run
```

Both GitHub and Hugging Face dataset repositories are supported with immutable 40-character revisions and exact SHA-256 checks. The manifest generator defaults to config/train/val only; `--include-test` is an explicit opt-in for a separate future evaluation distribution. Full training fetches and loads train/val only even with an older all-split source manifest. It never requires test arrays on Veri.

Only after successful data verification and separate explicit GPU approval:

```bash
PYTHON=.venv311/bin/python bash scripts/submit_veri.sh full --execute
```

The remote entrypoint is `bash scripts/remote_train.sh`; it fetches pinned train/val, records `dataset_source.json` under `VERI_OUTPUT_DIR`, then runs `python train.py --data-dir data/remote`. Defaults remain frozen-encoder/head fine-tuning: seed 42, batch 1, AdamW lr 0.001, maximum 20 epochs, patience 4. Correct per-example weighted CE uses train counts [786,49] and weights [0.53117048,8.52040816]. Validation loss is unweighted CE on all 465 validation windows; `reward` is validation fall F1. Best F1 selects the checkpoint, with loss breaking ties. No one-batch limit is present in full mode. The eight positive validation windows from four events in one session limit checkpoint selection.

Output artifacts are `best.pt` (full state, approximately 1.37 GB), `run_config.json`, `history.json`, `reload_check.json`, `dataset_source.json` and Veri's `metrics.jsonl`. All are saved beneath `VERI_OUTPUT_DIR`; a full forward comparison verifies the reloaded best checkpoint. The smoke checkpoint is diagnostic and is not resumed for full training.

On a machine/GPU environment holding the exported artifact and processed data:

```bash
python evaluate.py --checkpoint /path/to/best.pt \
  --data-dir data/processed --output reports/evaluation.json
```

The evaluator loads the full trained checkpoint without Hugging Face access and checks preprocessing identity. It reports accuracy, precision, fall recall, F1, confusion matrix, class counts, independent held-out recordings/sessions, covered-event detection rate and false-positive alert episodes/hour. Overlapping window durations are merged rather than added. Contiguous positive decisions merge into one alert; decisions occur at window end. Normal exposure is the union of evaluated windows in explicitly normal trials. Unsupported/gapped time is excluded. Missing eligible events are counted as uncovered. True impact latency is **not estimable**; action-start-to-first-alert is separately labeled as an approximate proxy. No zero-latency or calibrated-probability claims are made. Diagnostic checkpoints require `--allow-diagnostic` and remain unsuitable for performance claims.

An exact raw-window inference acceptance command using a held-out recording is:

```bash
python infer.py --checkpoint /path/to/best.pt \
  --recording ../blissey-data/recordings/session_03_LAHACKS/empty_room/20261017_203126_416953Z \
  --start-seconds 0.017783041985239834
```

This reconstructs the model/scaler, parses raw I/Q, transforms the unseen window identically and prints logits and an experimental class-1 softmax score. The raw preprocessing equivalence test passed locally; full-checkpoint raw inference awaits completion of the separately approved full-training run. For remote inference, supply an authorized unseen recording separately; raw recordings are deliberately absent from the training upload.

MOMENT-1-large is a large PyTorch T5 encoder. Nothing here proves TensorFlow Lite, ONNX, Raspberry Pi or QNX compatibility. Future deployment needs runtime/export/operator checks, memory/latency measurement and likely distillation or a different deployment model. The parser/scaler are reusable; the checkpoint is not presented as QNX-ready.

## Completed smoke and remaining checks

The completed smoke job's API-reported cost is **$0.3521364207**. Its one training window and one validation window do not provide reliable full-epoch timing or cost. The dashboard reports 6.5 minutes including setup and transfers; the train/validation epoch section was 22.7447 seconds. No full-training cost or duration has been measured. Check current GPU pricing before any future approved submission.

Before another paid job, run the remote-data dry run against the configured immutable dataset source and obtain explicit GPU approval. Test evaluation follows selection of a full-training checkpoint, on a machine with sufficient disk/RAM. No further job was launched in this preparation.

References checked: [Veri custom scripts](https://docs.veri.studio/training/custom-script), [base images](https://docs.veri.studio/training/base-images), [official MOMENT source](https://github.com/moment-timeseries-foundation-model/moment/tree/38f7310ad594100747ca2a8357e9c7ca7d323e0e), [pretrained model](https://huggingface.co/AutonLab/MOMENT-1-large), [Espressif CSI layout](https://docs.espressif.com/projects/esp-idf/en/v4.4/esp32/api-guides/wifi.html#wi-fi-channel-state-information). Source/runtime snapshots are in `reports/runtime_sources.json`.
