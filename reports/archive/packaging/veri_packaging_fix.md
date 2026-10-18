# Veri 413 packaging fix

The failed snapshot was `/var/folders/fg/rszyw6kx7qvc9w17ddjvp2y40000gn/T/blissey-veri-u_0acff2`.

| Package | File bytes, including manifest | tar.gz bytes |
| --- | ---: | ---: |
| Rejected full-data upload | 244,788,205 (233.448 MiB) | 244,775,769 (233.436 MiB) |
| Revised smoke upload | 422,118 (0.403 MiB) | **399,556 (0.381 MiB)** |
| Full-training code-only example | small code/manifest only | 12,800 |

The old archive was measured using Veri 0.2.44's exact sorted tar/gzip order and filter, streaming into a byte counter rather than filling the Mac's disk. `.npz` is already compressed; another gzip pass barely reduced it. The three arrays dominate:

| File | Bytes |
| --- | ---: |
| test.npz | 121,638,288 |
| train.npz | 78,369,259 |
| val.npz | 44,662,102 |

No endpoint limit was documented or established from the 413 response. The new **8 MiB local budget**, including a conservative 64 KiB multipart margin, is a project safeguard, not a claimed Veri limit. `VERI_MAX_UPLOAD_MIB` can change that local budget. The preflight runs before the actual CLI upload and exits nonzero when over budget or when the manifest/allowlist is violated.

## Smoke package

Exactly **four actual quality-v2 windows** are included: one fall and one normal from training, and one fall and one normal from validation. No test windows or full arrays are uploaded. Selected source indices, full-source hashes and original training scaler are recorded in the subset config. The selected tensors/labels are byte-for-byte equal to their source rows. The subset is never renormalized or relabeled.

The package contains only model/training modules, pinned requirements/config, `remote_smoke.sh`, two tiny NPZ files, preprocessing metadata and manifests. There is no `.venv`, `.git`, cache, legacy data, raw recording, credentials or pretrained/checkpoint weight. Full-data loaders remain the default; tiny data requires explicit `--smoke-data --epochs 1 --max-batches 1 --batch-size 1`, preventing accidental full training on the fixture.

On Veri, the unchanged `model_utils.load_pretrained` downloads actual pinned **AutonLab/MOMENT-1-large** from Hugging Face. One actual training example produces a forward pass, finite logits/softmax/loss, backward pass and exactly one optimizer update. One validation example and checkpoint reload checks follow; `$VERI_OUTPUT_DIR/best.pt` contains actual full model weights. No local MOMENT download or remote GPU test was performed here.

Dry run completed:

```bash
cd /Users/chloe/chloe526.github.io/blissey/blissey-model
PYTHON=.venv311/bin/python bash scripts/submit_veri.sh smoke --dry-run
```

**Exact retry, after approval only:**

```bash
cd /Users/chloe/chloe526.github.io/blissey/blissey-model
PYTHON=.venv311/bin/python bash scripts/submit_veri.sh smoke --execute
```

The wrapper creates a fresh small staging directory and invokes the validated `veri run-script` flags. Do not reuse older staging directories: they still contain full arrays. The latest expanded command is in `packaging_fix_smoke_dry_run.txt`. Fresh tar headers can change the exact compressed byte count slightly; the preflight prints it each time.

## Full training: external data, code-only upload

Read-only inspection of public [chloe526/blissey-data at 87228a1e39a9ba785605c979802dd67e56578cc1](https://github.com/chloe526/blissey-data/tree/87228a1e39a9ba785605c979802dd67e56578cc1) found **no `.npz` files or preprocessing_config.json**. The proposed pinned processed path returned HTTP 404. Full training is therefore blocked pending separately publishing only these existing files:

```text
processed/quality-v2/preprocessing_config.json
processed/quality-v2/train.npz
processed/quality-v2/val.npz
processed/quality-v2/test.npz
```

Do not upload raw recordings, environments, credentials, legacy arrays or checkpoints. Nothing has been published or committed by this task. A Hugging Face dataset repo is the practical option for these numerical binaries: [commit-pinned Hub downloads](https://huggingface.co/docs/huggingface_hub/guides/download) preserve arbitrary files without converting tensors to JSONL. GitHub also works through Git LFS; the 116 MiB test file exceeds [GitHub's 100 MiB regular-file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

After the processed artifacts are hosted and you know their full commit SHA:

```bash
# Replace OWNER/REPO and the immutable 40-character revision.
.venv311/bin/python scripts/make_dataset_source.py \
  --backend huggingface --repo-id OWNER/REPO \
  --revision FULL_40_CHARACTER_COMMIT_SHA \
  --path-prefix processed/quality-v2 --output configs/dataset_source.json

# Read-only verification; streams/hashes all four files without saving a copy.
.venv311/bin/python fetch_dataset.py --source configs/dataset_source.json --verify-only

PYTHON=.venv311/bin/python bash scripts/submit_veri.sh full --dry-run
```

For GitHub use `--backend github --repo-id chloe526/blissey-data` with the **future commit that actually contains the processed artifacts**. `configs/dataset_source.github.example.json` binds to today's local hashes but points at a currently missing path; it is an example, not a ready dataset source.

Full/epoch submission verifies remote bytes locally before submission and uploads **no NPZ arrays**. The code-only job downloads exactly the four specified files at the pinned revision, supports Git LFS media pointers, checks file sizes and SHA-256, and validates quality-v2 provenance/splits before model loading. It never clones the repository or downloads raw recordings. The original numerical split files and saved scaler remain unchanged.

## Verification

- **33 tests passed**, including genuine-row/scaler equivalence, absence of test/full arrays, code-only full packaging, archive measurement, hash failure, Git LFS resolution, and shell rejection before upload. The rejection test uses a local fake CLI; no paid command was sent.
- Actual installed Veri SDK 0.2.44 generated the smoke archive with a fake HTTP transport: **399,556 bytes**, exactly matching preflight. Its contents/hashes matched the allowlist. No network upload occurred.
- Current quality-v2 config and all three full dataset hashes are unchanged.
- No GPU job launched, no credits redeemed, and no credentials printed.

Machine-readable evidence: `rejected_upload_sizes.json`, `veri_smoke_upload_preflight.json`, `veri_smoke_sdk_archive_check.json`, `veri_packaging_fix.json`, `veri_packaging_fix_tests.txt` and `pinned_dataset_availability.txt`.
