# Independent 1D CNN baseline

This is a **randomly initialized CNN**, separate from the pretrained MOMENT/Veri experiment. It imports no MOMENT, Hugging Face or Veri modules and runs locally. The only shared input is the read-only quality-v2 dataset in `../data/processed/`.

| | CNN baseline | Existing MOMENT experiment |
| --- | --- | --- |
| Code | `cnn_baseline/` | Top-level model scripts and Veri submission scripts |
| Initialization | Random | Pretrained AutonLab/MOMENT-1-large |
| Dependencies | PyTorch + NumPy | Existing MOMENT/Veri stack |
| Checkpoints | `cnn_baseline/checkpoints/<run>/best.pt` | Existing checkpoint/output directories |
| Results | `cnn_baseline/reports/<run>/` | Existing reports |

No existing MOMENT/Veri code, configuration, requirements, checkpoints or dataset files were modified. Checkpoint files are ignored by Git. CNN checkpoints use `model_type=blissey_cnn_1d_v1`; the CNN loader explicitly rejects other checkpoint formats.

## Architecture and fixed experiment

Input `[B,52,512]` → Conv1D(52→16,k9,stride2) → Conv1D(16→32,k7,stride2) → Conv1D(32→64,k5,stride2) → global average pooling → dropout(0.2) → Linear(64→2).

Each convolution block includes GroupNorm and ReLU. GroupNorm supports training at batch size one. The model has **21,666 parameters**. No extra RevIN, channel flattening, resampling or normalization is applied: the already normalized quality-v2 arrays are used directly.

`config.json` predeclares seed 42, batch size 32, AdamW lr 0.001 / weight decay 0.001, maximum 30 epochs, six-epoch patience, gradient clipping at 5, and threshold 0.5. Early stopping selects the lowest **unweighted validation cross entropy**, with minimum improvement 0.0001. This is the sole checkpoint-selection rule; neither architecture nor threshold was tuned on test results.

Class weights come only from the training counts: `N / (2 * count_per_class)`. Training uses `cross_entropy(..., weight=..., reduction='none').mean()`, so class weights do not cancel for batch size one. Validation/test are not oversampled. Their loss is unweighted and their original labels are preserved.

The train/validation/test groups remain session_02/session_01/session_03_LAHACKS. The trainer opens only `train.npz` and `val.npz`; the test array is opened only by the evaluation command after checkpoint selection. File hashes, quality version, tensor shapes and recording/group assignments are checked. The original training-fitted preprocessing configuration is saved in the CNN checkpoint.

## Exact local commands

Run from `blissey-model`. The existing `.venv311` already supplies PyTorch/NumPy; no environment changes or downloads are needed. A standalone environment elsewhere can install only `cnn_baseline/requirements.txt`.

```bash
cd /Users/chloe/chloe526.github.io/blissey/blissey-model

# Start a new fixed baseline run. Use another name if this one already exists.
.venv311/bin/python -m cnn_baseline.train \
  --data-dir data/processed --run-name local-cnn --device cpu

# Evaluate the selected CNN checkpoint on the untouched test split.
.venv311/bin/python -m cnn_baseline.evaluate \
  --checkpoint cnn_baseline/checkpoints/local-cnn/best.pt \
  --data-dir data/processed --split test --device cpu \
  --output cnn_baseline/reports/local-cnn/test_evaluation.json

# CNN-only unit/integration tests; no MOMENT, downloads or paid jobs.
.venv311/bin/python -m unittest discover -s cnn_baseline/tests -v
```

Optional `--epochs` and `--batch-size` control training. Existing run checkpoints are never overwritten; choose a new `--run-name`. The measured run below is already saved as `seed42`. To inspect/re-evaluate it, substitute `seed42` for `local-cnn`; do not retrain that existing run name. CPU is the tested default. Optional local CUDA support has not been exercised on this Mac; there are no remote submission commands in this directory.

## Measured result: seed42

The fixed CPU run completed seven epochs; early stopping selected **epoch 1**. One test evaluation then used those frozen weights and the predeclared 0.5 threshold.

| Metric | Selected checkpoint: train | Validation | Test |
| --- | ---: | ---: | ---: |
| Fall recall | 100% | 100% | **100%** |
| Precision | 20.68% | 3.13% | **2.04%** |
| F1 | 0.3427 | 0.0606 | **0.0400** |
| Positive windows | 49 | 8 | 21 |

Test confusion matrix, true rows / predicted columns, classes `[normal, fall]`:

```text
[[ 242, 1007],
 [   0,   21]]
```

The selected model detects **3/3 covered test action events**. The fourth annotated test event has no usable quality-v2 window and remains explicitly uncovered. Including that data-coverage limitation, detected events / all annotated events is 3/4. These are approximate staged-action intervals, not precisely timed impacts. Test has 39 contributing recordings from one collection session; windows overlap and are not independent trials.

**This baseline has poor discrimination and excessive false positives.** Later training improves training F1 to 0.873 while validation F1 is about 0.050 and validation loss worsens. This is consistent with overfitting and/or session shift. Early stopping retains epoch 1, which still generalizes poorly. Architecture, class weighting, checkpoint rule and threshold were not changed after seeing test results. High event recall alone is not a useful success claim when 1,007 normal windows are flagged.

Under the documented contiguous-positive-window episode policy, there are 126 false-positive episodes over 829.5 seconds of evaluated normal exposure (546.8/hour extrapolated from this short recording set). This is a simple offline policy, not the application's alert logic. Impact detection latency is not estimable from the action annotations.

## Size and timing

Measured on the current Apple Silicon Mac, **CPU / two PyTorch threads / float32**:

| Measurement | Result |
| --- | ---: |
| Parameters | 21,666 |
| Saved checkpoint, including preprocessing metadata | 126,031 bytes (123.1 KiB) |
| Training + validation epoch loop | 2.917 seconds |
| Data load/verification | 0.364 seconds |
| Script work after configuration, including final verification | 4.088 seconds |
| Batch-one forward latency, median | **0.190 ms** |
| Batch-one forward latency, p95 | **0.200 ms** |

Latency uses 20 warmup passes and 200 timed forward passes on an already resident `[1,52,512]` tensor. It excludes three-second acquisition, preprocessing, file I/O, transfer and alert logic. Timings vary with machine load; this is not a QNX/device measurement or a measured MOMENT speed comparison. No pretrained model was downloaded and no paid GPU was used.

## Artifacts

- `checkpoints/seed42/best.pt`: actual selected CNN weights, architecture/training config, source hashes and saved preprocessing metadata. Optimizer resume state is not included.
- `reports/seed42/history.json`: all seven epochs, losses and train/validation metrics.
- `reports/seed42/training_summary.json`: selected-checkpoint train/validation metrics, duration and reload verification.
- `reports/seed42/test_evaluation.json` / `.txt`: held-out metrics, events, class distribution, size and timing.
- `reports/seed42/test_evaluation_predictions.npz`: probabilities, original labels and recording/time provenance.
- `reports/tests.txt`: six passing CNN tests covering shape/backprop, weighted B=1 gradients, checkpoint roundtrip, event coverage, held-out access and dependency isolation.
- `reports/protected_inputs_verification.json`: hash check that all pre-existing pipeline/data files remain unchanged.
