# MOMENT full-training preparation — 2026-10-17

> Cleanup note: this report records the original preparation snapshot. The active `configs/dataset_source.json` now identifies commit `4727901ff595c3a79760c1785f540ec753084dc1`; older 404/publication observations below refer to the earlier commit. Cleanup did not revalidate remote hosting. The duplicate local hosting bundle was removed and can be recreated exactly with `.venv311/bin/python scripts/prepare_hosting.py`. Original data/config hashes remain unchanged. Historical readiness checks moved to `reports/archive/hosting/`.

Smoke job **261036b21c27 passed** on NVIDIA L4. Read-only Veri queries confirmed `completed` status and downloadable artifacts. Its actual `run_config.json`, `history.json` and `reload_check.json` were retrieved; model weights were not downloaded to this Mac. No new job or public upload was launched during this preparation.

The logs show CUDA PyTorch 2.8.0+cu128, MOMENT 0.1.5, transformers 4.55.4, HF Hub 0.34.4 and NumPy 1.26.4. The actual remote config identifies **AutonLab/MOMENT-1-large**, revision **ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc**. MOMENT's implementation is pinned to **38f7310ad594100747ca2a8357e9c7ca7d323e0e**. The pretrained encoder is loaded remotely from Hugging Face; its classification head is newly initialized, as required by official MOMENT classification. There is no randomly initialized encoder fallback.

Evidence includes 106,498 trainable head parameters / 341,240,320 frozen parameters, one optimizer step on one real training fall window, finite logits and probabilities, and the final smoke PASS. That PASS requires finite nonzero head gradients, a changed head weight, no gradients on frozen parameters, and a saved full-state checkpoint exceeding 1 GB. The retrieved reload report has `passed: true`, identical validation logits `[-1.6093992, 1.6652651]`, and `diagnostic: true`. Reload reconstructs architecture from the saved config and loads the full state offline; its forward output is checked against the saved model. Logs and the artifact manifest confirm upload of `best.pt`, `history.json`, `run_config.json`, `reload_check.json` and `metrics.jsonl`.

The dashboard's **loss 0.0371** is precisely `0.03713152930140495`: **unweighted validation cross-entropy on one fall window**. Its fall probability was 0.9635493755, so loss is approximately `-log(0.9635493755)`. It is separate from the training loss, 0.7617096901, on the single training fall window before that update. The tiny smoke fixture had one normal and one fall per split, so its training class weights were both 1; `--max-batches 1` evaluated only the first validation window, a fall.

The dashboard's **reward 1.0** is **validation fall-class F1**, explicitly passed to `veri_sdk.log_metrics`. Its confusion matrix was `[[0,0],[0,1]]`: one true positive, zero evaluated normal windows. Accuracy also happened to equal 1 on this one window, but reward is not accuracy, and this does not measure specificity, the complete validation split, or held-out test performance. The dashboard's 6.5-minute duration is not a per-epoch timing estimate. The logged train/validation epoch section took 22.7447 seconds; the reload report's measured training/reload section took 31.8788 seconds. Setup, downloads and artifact transfer are additional work. The API reports the completed job cost as $0.3521364207; no full-training cost is inferred from it.

The reproducible bundle path is `hosting/quality-v2-trainval-58540c5b7f3d/`. After regeneration, it contains only these three original numerical/metadata files plus `SHA256SUMS` and `artifact_manifest.json`:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| train.npz | 78,369,259 | `03aaa608201a645e7eb76ee83a55414f8f3c2dad22d9f51e1ae63b725bf2d690` |
| val.npz | 44,662,102 | `8c221423e4fcfc3eccd8bfa027d32757b727b4d3c1069d99e7f3603d648b331b` |
| preprocessing_config.json | 60,284 | `8f83dbdd4e2d1987c12b701f75d655938710ce4797291b1a43bc44dc666c5983` |

The three artifacts total **123,091,645 bytes (117.39 MiB)**. The content revision is **quality-v2-trainval-58540c5b7f3d**; its full manifest-content hash is `58540c5b7f3da44e47ba72957fff2d31cb6ef52f5cc5e803d71fb874d3482fae`. This content identity is distinct from a Git commit pin. APFS clones preserve the originals without a second full physical copy on this low-space Mac.

All original train, validation, test and preprocessing-config SHA-256 values remain unchanged. Train is `[835,52,512]`, class counts `[786,49]`, session_02. Validation is `[465,52,512]`, class counts `[457,8]`, session_01. The untouched local test remains `[1270,52,512]`, counts `[1249,21]`, session_03_LAHACKS, SHA-256 `9ee6b912f86e78adc41270e615b5bd718fac26cd658e998fefdeb16659fd6b90`. No window, scaler, session assignment, quality threshold or label was regenerated or changed. The unchanged config retains all splits' provenance and annotation metadata; **test signal tensors are absent from the hosting bundle and training job**. No raw recordings, environments, credentials, legacy artifacts or generated model checkpoints are bundled. CNN code and checkpoints are separate and unchanged.

Use the selected existing public **chloe526/blissey-data GitHub repository**, path `processed/quality-v2/`. Train and validation are each below GitHub's 100 MiB regular-Git file limit; train exceeds the 50 MiB warning threshold. No LFS is needed for these two arrays. Use Git rather than the browser uploader. Hosting the larger test array is unnecessary. [GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

The read-only tree check found **no processed artifacts** at current commit `87228a1e39a9ba785605c979802dd67e56578cc1`. Consequently there is **no valid remote dataset revision yet**. The included `configs/dataset_source.github.example.json` binds the correct local hashes but still points at that unavailable example path; it is deliberately not installed as `configs/dataset_source.json`. A remote commit SHA cannot be finalized before publication. No publication has been performed or implied. The existing wrapper correctly refuses the full dry run before upload or job submission while the active manifest is absent.

To recreate and verify the local bundle:

```bash
cd /Users/chloe/chloe526.github.io/blissey/blissey-model
.venv311/bin/python scripts/prepare_hosting.py
(cd hosting/quality-v2-trainval-58540c5b7f3d && shasum -a 256 -c SHA256SUMS)
```

When publication is authorized, copy only the five allowlisted bundle files to `processed/quality-v2/` in the chosen repository, review the exact staged paths, and commit/push those files. Do not use `git add .` in the recording repository. The following is the narrowly scoped staging command after copying the files; it does not commit or push:

```bash
git -C ../blissey-data add -f -- \
  processed/quality-v2/train.npz processed/quality-v2/val.npz \
  processed/quality-v2/preprocessing_config.json \
  processed/quality-v2/SHA256SUMS processed/quality-v2/artifact_manifest.json
git -C ../blissey-data diff --cached --name-only
```

After those files are published, replace the value below with the **actual full 40-character commit that contains them**, then generate and remotely verify the pinned source. A moving `main` or a made-up SHA is rejected. Each downloaded artifact is checked for exact length and SHA-256 before it is used.

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

The dry run verifies remote bytes without saving another local dataset, creates a code-only allowlist package and measures the SDK tar/gzip stream. The offline package-size check measured **35,850 raw file bytes / 13,133 compressed tar.gz bytes across 13 files**, well below the 8 MiB guard (see `veri_full_upload_preflight.json`); this does **not** claim that the unpublished remote dataset passed its availability check. No `.npz`, pretrained weights, `.git`, `.venv`, caches, test tensors or legacy artifacts enter the code upload. The code package carries the download manifest, pinned dependency lock, model/training/evaluation code and remote entrypoint.

After a successful dry run **and separate explicit paid-job approval**, the exact full submission command is:

```bash
PYTHON=.venv311/bin/python bash scripts/submit_veri.sh full --execute
```

This uses one L4-24GB with `veri/base` and `requirements-veri.lock.txt`. The remote entrypoint is `bash scripts/remote_train.sh`. It runs:

```bash
python fetch_dataset.py --source configs/dataset_source.json \
  --output data/remote --splits train val
python train.py --data-dir data/remote
```

It downloads original MOMENT pretrained weights directly on Veri. Full training starts afresh from that pretrained encoder and a seeded classification head; the one-step diagnostic checkpoint is not resumed. With the preserved `configs/training.json`, this is **full-dataset head fine-tuning**, with the encoder frozen (`unfreeze_last_blocks: 0`), 106,498 trainable parameters, concat reduction, `[B,52,512]` inputs and an all-valid `[B,512]` input mask. It is not end-to-end encoder fine-tuning. AdamW uses learning rate 0.001, weight decay 0.01, clipping 1.0, seed 42, batch size 1, maximum 20 epochs and early-stopping patience 4.

Training cross-entropy weights are `N/(2*n_class)` using **training only**: normal **0.53117048**, fall **8.52040816**. The implementation computes `mean(weight[target] * CE_per_example)`, preserving class-weight effects at batch size one. Standard PyTorch weighted-mean CE would cancel those weights for a one-example batch. All train windows are visited once per epoch with deterministic shuffling; there is no validation/test oversampling. Gradient clipping remains part of the fixed optimization configuration.

Each complete validation pass evaluates **all 465 windows** in evaluation mode, with no gradients or shuffling. It reports unweighted mean cross-entropy, precision, fall recall, F1, accuracy, class counts and a confusion matrix with true rows / predicted columns, class order `[normal, fall]`. Veri loss is this validation CE; Veri reward is this validation fall F1. The best checkpoint maximizes validation F1, breaking ties by lower validation CE. Four epochs without a lexicographic improvement stop training. Test tensors are neither downloaded nor loaded by training. The data loader verifies selected artifact hashes, recording assignments and train/validation session separation.

Validation has only **eight positive windows from four events in one session**. Correlated windows do not increase the independent event/session count. This supports a feasibility run, with unstable checkpoint selection; it does not support a robust hyperparameter search. Keep this fixed configuration, report overfitting if observed, and collect additional independent development sessions before broader model selection. Do not tune with the final test labels or results.

Artifacts saved beneath **VERI_OUTPUT_DIR** and captured by Veri:

| Artifact | Contents |
| --- | --- |
| best.pt | Full encoder + best trained head state, architecture, preprocessing, training config, validation metrics, epoch, package versions; approximately 1.37 GB |
| run_config.json | Fixed hyperparameters, exact preprocessing/scaler, installed versions, device, actual train-only class weights and metric definitions |
| history.json | Complete per-epoch train/validation metrics and times |
| reload_check.json | Offline checkpoint-reconstruction forward comparison and elapsed time |
| dataset_source.json | Exact processed-data repository, immutable revision, sizes and hashes |
| metrics.jsonl | Veri's logged validation loss/F1 values |

`best.pt` remains portable without Hugging Face access. The Mac currently lacks sufficient free disk for the full checkpoint. Run download/evaluation on a machine with adequate storage and RAM, using the same environment and the unchanged local quality-v2 test/config files. No model or test download was attempted here.

```bash
# On that adequately provisioned machine, after the approved full job completes:
FULL_JOB_ID=REPLACE_WITH_COMPLETED_FULL_JOB_ID
veri jobs logs "$FULL_JOB_ID" --quiet
veri jobs download "$FULL_JOB_ID" --output-dir checkpoints/moment-full
python evaluate.py \
  --checkpoint "checkpoints/moment-full/$FULL_JOB_ID/best.pt" \
  --data-dir data/processed --batch-size 1 --device auto \
  --output reports/moment_full_test_evaluation.json
```

Evaluation loads only the **unchanged local test array and original preprocessing config** and verifies metadata identity with the checkpoint. Keep the decision threshold fixed at 0.5. Evaluate the chosen full checkpoint once for final reporting; do not pass `--allow-diagnostic`. Outputs are JSON, a text summary and a window-predictions NPZ. Metrics include fall recall, precision, F1, confusion matrix, covered-event detection, uncovered events and false-alert episodes per normal monitored hour. Three of four annotated test events have usable windows under the existing frozen policy; the fourth stays uncovered. Actual impact latency is unavailable because the annotations mark approximate action intervals.

Validation performed here: **38 passing weight-free unit/integration tests**, real-array integrity and split checks, streaming hash verification against the prepared bundle, code-only archive measurement/allowlist checks, CLI argument checks and shell syntax checks. Tests cover correct batch-one weighting, the exact Veri metric mapping, no default test download even with a legacy all-split manifest, immutable commit requirements, hash mismatch/oversize rejection, a real-data train/val hosting bundle, and remote-entrypoint argument/output propagation using a stub interpreter. These tests do not substitute random weights for a pretrained-model test; actual pretrained CUDA execution was verified from the completed job's artifacts above.

Remaining prerequisites: publish the reviewed processed-only bundle, set its actual immutable GitHub revision, pass the remote-data dry run, and obtain approval before a new paid submission. All three splits and the preprocessing config are unchanged.
