# LA Hacks job d4c10e9207e8 comparison

Both folds completed. Development cross-validation only; one participant, two blocks, no independent final test.

| Model / fold | Precision | Recall | F1 | Covered events | False alerts/h | CPU median ms | Size KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN A | 0.00% | 0.00% | 0.00% | 0/10 | 0.00 | 1.30 | 89.7 |
| RF A | 0.00% | 0.00% | 0.00% | 0/10 | 0.00 | 57.68 | 145.1 |
| CNN B | 1.82% | 100.00% | 3.58% | 3/3 | 551.18 | 1.63 | 89.7 |
| RF B | 11.11% | 14.29% | 12.50% | 1/3 | 17.36 | 57.79 | 138.2 |

A trains 03 and validates 04; B trains 04 and validates 03. Event detection uses fixed 2-of-3 voting and 5-second cooldown. Fold A predicts no falls; its stored precision uses zero-division=0. CPU latency includes window features/normalization and inference, excludes packet parsing/resampling.

Pooled CNN: 3/13 covered events (3/34 all annotated), 432.54 false alerts/hour. Pooled RF: 1/13 covered events (1/34 all annotated), 13.62 false alerts/hour. Only 17.62 minutes of accepted normal exposure; 21 events are quality-uncovered. Macro metrics are in comparison.json and the original benchmark.json.

Artifacts: worker logs show 18 uploads, but API exposes 11 flattened filenames. Hashes prove seven Fold A files have been replaced by Fold B basenames in the available final set. Both folds passed remote reload checks. Available Fold B CNN/RF checkpoints were independently reloaded on CPU and checked against saved predictions; see local_reload_verification.json.

Fix artifact retention by preserving relative paths or exporting a single archive from retained worker output. Without retained worker files/object versions, Fold A checkpoints cannot be recovered from these downloads. No retraining was performed.

Demo recommendation: RF as a supervised development prototype, with clear missed-fall limitations. CNN alerts excessively; automated recall-first ranking does not make it suitable. Next: CPU replay/live feature-parity check of the saved Fold B RF using its exact preprocessing and fixed alert policy before any demo deployment.
