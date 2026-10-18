# Historical diagnostics

These snapshots are retained for provenance, rather than used by current training:

- `preprocessing/`: the original quality-v1 review and its smoke/verification outputs.
- `packaging/`: the rejected full-data upload, packaging fix and older dry runs.
- `setup/`: superseded environment and test logs.
- `hosting/`: availability/readiness checks from before the active dataset source was configured.

Current quality-v2 audits, development evidence, threshold locks, upload preflights and successful remote smoke artifacts remain in `reports/`. CNN results remain separately under `cnn_baseline/reports/`.

Commands and temporary paths inside archived logs describe their original run. Use the top-level README and current scripts for new work. `reports/cleanup_manifest.json` records the original and archived locations.
