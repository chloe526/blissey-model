#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-smoke}
ACTION=${2:---dry-run}
case "$MODE" in
  smoke) ENTRYPOINT='bash scripts/remote_smoke.sh' ;;
  epoch) ENTRYPOINT='bash scripts/remote_train.sh --epochs 1' ;;
  full) ENTRYPOINT='bash scripts/remote_train.sh' ;;
  *) echo 'Usage: scripts/submit_veri.sh [smoke|epoch|full] [--dry-run|--execute]' >&2; exit 2 ;;
esac
case "$ACTION" in --dry-run|--execute) ;; *) echo 'Expected --dry-run or --execute' >&2;exit 2 ;; esac
PYTHON=${PYTHON:-python}
HELP=$(veri run-script --help)
for FLAG in --entrypoint --gpu-type --gpu-count --base-image --requirements; do
  if [[ "$HELP" != *"$FLAG"* ]]; then echo "CLI does not support $FLAG" >&2; exit 1;fi
done
PACKAGE_ARGS=(--mode "$MODE")
if [[ "$MODE" != smoke ]]; then
  SOURCE=${DATASET_SOURCE_CONFIG:-configs/dataset_source.json}
  if [[ ! -f "$SOURCE" ]]; then
    echo 'Full training needs a published, commit-pinned processed dataset manifest. See README; no upload/job submitted.' >&2
    exit 1
  fi
  # Verify the remote bytes BEFORE any paid job; no local dataset copy created.
  "$PYTHON" fetch_dataset.py --source "$SOURCE" --verify-only --splits train val
  PACKAGE_ARGS+=(--dataset-source "$SOURCE")
fi
STAGE=$("$PYTHON" scripts/package_veri.py "${PACKAGE_ARGS[@]}")
REPORT="reports/veri_${MODE}_upload_preflight.json"
# Always check BEFORE invoking the CLI's code_artifacts.upload call.
"$PYTHON" scripts/archive_preflight.py "$STAGE" --max-mib "${VERI_MAX_UPLOAD_MIB:-8}" --report "$REPORT"
COMMAND=(veri run-script "$STAGE" --entrypoint "$ENTRYPOINT" --base-image veri/base --gpu-type L4-24GB --gpu-count 1 --requirements "$STAGE/requirements-veri.lock.txt")
echo "Upload manifest: $STAGE/upload_manifest.json"
echo "Preflight report: $REPORT"
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "$ACTION" == --execute ]]; then
  echo 'Submitting paid L4 job; use only after explicit approval.' >&2
  "${COMMAND[@]}"
else
  echo 'DRY RUN: no upload and no GPU job submitted.'
fi
