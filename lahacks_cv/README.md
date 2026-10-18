# LA Hacks-only CNN + Random Forest CV v1

This is separate from MOMENT and the historical CNN experiment. It never imports
MOMENT, reads historical processed arrays, loads old weights or uses Cal Poly
normalization. Reusing the small CNN's architecture **code** is intentional.

All results must say **development cross-validation only; no independent final
test**. Both physical blocks involve p1. Do not infer generalization to new
participants from these results.

## Data and preservation

`source.json` pins public `chloe526/blissey-data` commit
`06c97032c93a43ddccda5aa8381a95bced54755c`, with SHA-256 and sizes for 254
Session 03/04 metadata, events and measurement files. It also hashes the unchanged
quality-v2 parser, quality check, interpolation, labels, CNN architecture and rules.
Raw serial binaries are excluded. No publication, git commit or source mutation
is needed. There are no newly uploaded participant recordings.

The remote worker downloads only those already public, pinned LA files directly
from GitHub and checks every byte size/hash. The lightweight **code** archive has
no data, checkpoints, virtualenvs, git directories, caches or old experiments.
The source download is 166.3 MB and stays in a disposable `/tmp` cache. The
dataset is rebuilt on the worker rather than needing a new processed-data hosting
commit. Thus no existing Cal Poly dataset source manifest is used.

Quality-v2 is unchanged: 52 physical CSI amplitude channels, 3 seconds, 0.5-second
stride, 512 interpolated points; >=20 Hz, >=80% occupied 100ms bins, >=90% span,
<=100ms edge distance and <=250ms gap. Explicit normal labels are zero; positives
are wholly inside the existing host-aligned operator interval with 250ms inner
margin. Other fall-trial windows remain ignored. No label-v3 imports or outputs.

Reproduced counts:

| Block | Recordings | Normal / fall windows | Annotated / covered events |
|---|---:|---:|---:|
| 03 | 42 | 1,249 / 21 | 4 / 3 |
| 04 | 50 | 301 / 31 | 30 / 10 |
| Total | 92 | 1,550 / 52 | 34 / 13 |

`data/lahacks_quality_v2_cv_v1/windows.npz` stores one **unnormalized** float32
`[1602,52,512]` array and aligned labels, block/recording/event IDs and timestamps.
`dataset.json` stores sources, annotation intervals, quality counts, tensor hash
and separate fold scaler fits. This avoids two additional copies of overlapping
data. Every output directory is new; existing datasets/runs are refused.

## Training and evaluation

Fold A fits 03 and evaluates 04; B fits 04 and evaluates 03. Recording and event
IDs cannot cross blocks. Both models consume precisely the same eligible rows.
CNN normalization is fitted independently on the training block, never the outer
validation block. RF features normalize each channel by its own window median;
PCA variance summaries are also window-local. No dataset-fitted RF transform.

CNN: random initialization, 3 Conv1D/GroupNorm/ReLU blocks, global average pool,
dropout 0.2, 21,666 parameters; AdamW, weight decay and clipping. Whole recordings
within the training block are stratified by class into inner fit/development
partitions. The inner scaler uses inner fit only. Unweighted inner development
loss selects epochs with early stopping; another randomly initialized model
then trains on the complete outer training block for that many epochs. Outer
validation predictions select neither epochs nor weights. The small number of
inner fall events makes epoch selection unstable; inspect this explicitly.

Both models use a predeclared **training-only** fall/normal weight ratio
`min(4,sqrt(n_normal/n_fall))`, versus full inverse-frequency ratios up to 59.5.
CNN loss averages weighted per-example cross-entropy, retaining weights at batch
size one. RF has 300 trees, depth <=6 and >=5 samples/leaf. This is one fixed
configuration, not a validation-driven weighting search. Reports include training
and validation predicted-fall fractions/precision/recall to diagnose excessive
false positives or missed falls; poor results must be reported, not auto-retuned.

RF's 575 features include per-channel variance, absolute first difference,
first-difference variance, variance in each temporal third, peak/end motion
ratio, energy, 0–1/1–3/3–10 Hz power fractions, and three window-local PCA variance
fractions. The 512 interpolated samples do not imply a 170 Hz physical capture
rate; spectral features stop at 10 Hz.

Threshold is fixed at 0.5 for both models. The same causal alert rule requires
2 positives in the last 3 contiguous windows (three-window warmup), resets after
>0.75s missing decisions, and enforces a 5s cooldown. Nothing is tuned on held-out
fold predictions. Packet gaps and ignored windows break vote continuity; each
recording resets state. Sustained positives can generate another cooldown-defined
episode after 5s, explicitly matching the operational false-alert definition.

Window precision/recall/F1, confusion matrices, ROC-AUC and average precision
(the stated PR-AUC convention) use raw thresholded window probabilities. Event
recall uses **aggregated alerts** associated with eligible event windows, both
among 13 covered events and all 34 annotated attempts. Uncovered events are
reported separately. Operator intervals are not verified impact timestamps.
False-alert episodes/hour uses the union of accepted windows on verified normal
recordings (~17.6 minutes), including warmup; no overlapping-window double count.
No claims about an unobserved continuous deployment stream are warranted.

Reports include per-fold, macro and pooled results; CPU batch-one latency includes
normalization/features/model, excludes packet parsing/interpolation. Training
duration, parameter count, feature-extraction duration, serialized sizes, hashes,
seed, package versions, input identity hashes and reload checks are saved.
RF probabilities permit reduction-order roundoff (`rtol=1e-12`, `atol=1e-15`),
while threshold decisions must remain exactly identical; the maximum difference
is recorded. CNN reload logits still use exact comparison on the same CPU path.
CUDA determinism is requested; bitwise identity across platforms/library versions
is not promised. Joblib artifacts are trusted project outputs, not arbitrary
downloaded pickle files.

## Local preparation and checks

From `blissey-model/`, using the already verified LA-only temporary cache:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv311/bin/python -B -m lahacks_cv.data \
  --raw-root /tmp/blissey-stage1-06c97032c93a \
  --output data/lahacks_quality_v2_cv_v1
```

That output is already prepared; **do not rerun into the same directory**. To
reproduce later with a different, new output, add `--fetch` and give a new
`--raw-root` cache. No full data duplication is needed for each fold.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv311/bin/python -B -m lahacks_cv.data \
  --check-dataset data/lahacks_quality_v2_cv_v1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv311/bin/python -B -m unittest lahacks_cv.test_pipeline -v
```

Tests train tiny **synthetic** fixtures on CPU to verify saving/reloading and
inference, never fit a real-data model. The new local dependencies are the pinned
scikit-learn/SciPy/joblib/threadpoolctl versions in `requirements.txt`.

## Veri: one job, separate approval

```bash
PYTHONDONTWRITEBYTECODE=1 .venv311/bin/python -B -m lahacks_cv.submit --dry-run --verify-remote
```

This verifies source SHA-256 checks, validates the installed CLI, creates a curated
code package, measures the SDK-equivalent compressed archive and enforces the
8 MiB local policy including a 64 KiB multipart margin. It does **not** upload
code or start a job. The policy is not a claim about an undocumented server limit.
`reports/lahacks_cv_upload_preflight.json` records measured sizes and the temporary
code directory. It excludes `.git`, `.venv*`, caches, all NPZ/checkpoint files,
`processed_legacy_v1` and unnecessary tests. An offline packaging-only dry run
omits `--verify-remote`; `--execute` always verifies source integrity, rehashing
verified local files rather than re-downloading them. `--verify-remote` now also
reuses local copies and downloads only missing files. Logs explicitly distinguish
local verification from network verification; a local hit is not a fresh check
of current GitHub availability.

The submitter searches the earlier audit cache and local `blissey-data/recordings`
only for manifest-listed LA files. `--source-root PATH` adds an explicit read-only
source root. Missing files download to the revision-specific, git-ignored
`data/lahacks_source_cache/`; verified existing files are not copied there.
Wrong hashes fail without overwriting originals. No cached receipt is trusted
instead of actually rehashing the file.

Network operations use 10s socket and 45s per-file checks; an isolated subprocess
enforces a **120s hard whole-verification deadline**, including DNS and thread
shutdown. Progress reports every 25 files and every 5s while downloading; failures
identify the current path. CLI discovery has a separate 15s deadline. Increase
the whole deadline only explicitly with `--verification-timeout SECONDS`.

Safe, network-free validation using the verified cache:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv311/bin/python -B -m lahacks_cv.submit \
  --dry-run --verify-remote --offline \
  --source-root /tmp/blissey-stage1-06c97032c93a --verification-timeout 60
```

This performs no paid submission, data rebuild or real-data training. If a local
source file is missing, it stops instead of fetching it.

Only after explicit approval:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv311/bin/python -B -m lahacks_cv.submit --execute
```

One `L4-24GB` worker runs CNN on CUDA and standard sklearn RF on its CPU with four
threads. No MOMENT dependency/download. The installed Veri CLI requires GPU type;
CPU-only submission is not assumed. The base image provides CUDA torch 2.8.0;
the requirements pin it without replacing it with a CPU wheel. Linux dependency
resolution is checked before approval; actual GPU execution awaits this job.

All retained outputs are under `$VERI_OUTPUT_DIR/lahacks_cv_v1/`: `dataset.json`,
`run_config.json`, `benchmark.json`, `artifacts.json`, and `fold_A/`, `fold_B/`
with `cnn.pt`, `inner_cnn_best.pt`, `rf.joblib`, preprocessing/selection/metrics
JSON and aligned predictions. Source SHA-256 metadata and fold scalers persist;
temporary raw files and the reproducible intermediate NPZ do not consume artifact
storage. Artifacts include no participant raw measurements. Expected memory is
well within one L4; setup/download overhead may dominate this small experiment.

The job prints a proposed ranking for **human review**, not automatic deployment.
No command performs the all-LA deployment refit. That separate step is deliberately
gated until the benchmark finishes and results are reviewed. Fold artifacts can
be read by `lahacks_cv.infer.load_fold()` for diagnostics; `live_window()` shares
the identical quality/interpolation code tested offline. They are not approved
final deployment artifacts.
