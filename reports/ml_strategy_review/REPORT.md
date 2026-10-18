# Blissey Beacon: ML strategy review (LA Hacks Sessions 03 + 04 only)

Date: 2026-10-17. Scope: Sessions 03/04 raw data (pinned cache `/private/tmp/blissey-stage1-06c97032c93a`), `data/lahacks_quality_v2_cv_v1`, and job `d4c10e9207e8`. I did **no training, Veri jobs or dataset/checkpoint/report changes**. The only model execution was inference with the saved **Fold B** CNN/RF (Fold A checkpoints were lost to path flattening). All AUCs below are descriptive: windows overlap, and there are only 13 covered events from one participant.

Reproduce (from `blissey-model/`):
`.venv311/bin/python -B reports/ml_strategy_review/diagnose.py` → `analysis.json`
`.venv311/bin/python -B reports/ml_strategy_review/gain_check.py` → `gain_check.json`

## Bottom line

Thresholds and class weights aren't the main problem, and neither is architecture size. **Both models get an input representation dominated by ESP32 gain fluctuation and absolute amplitude level.** In Session 04, absolute level is confounded with recording order. When common-mode gain is removed, a single **unfitted** motion feature separates falls from normals in **both** sessions (AUC 0.86 / 0.93), whereas the trained models reach 0.36–0.73 cross-session. Fix the representation and the evaluation harness first, then compare models against that feature baseline. Data collection runs in parallel and is required for any unbiased claim.

## 1. Root-cause ranking

| # | Cause | Evidence | Type | Confidence |
|---|---|---|---|---|
| 1 | **The CNN learned absolute amplitude level, not motion.** | Fold B CNN on S03: replacing each window's per-channel DC with the training mean drops predicted-fall fraction from **90.6% to 0.6%**, and AUC rises from 0.36 to 0.62. A global gain change alone gives 49%. CNN probability vs window level, Spearman −0.37. | Confirmed (inference-only probe) | High |
| 2 | **Recording-order confound in S04.** All 30 falls came first, then lie_down, then sit_down. Amplitude drifts over time: Spearman 0.48 vs order within the fall recordings alone. | In S04, absolute level alone separates fall from normal at **AUC 0.09**, an inverted but near-perfect cue. In S03 it gives 0.62. A fold trained on S04 learns "lower level = fall", and S03 is lower overall, which explains the 1,130 false positives. S04 fall windows also carry more interpolation (interp-fraction AUC 0.71; S03 0.46). | Confirmed confound | High |
| 3 | **Common-mode gain (AGC) noise hides motion.** | 90% of empty-room within-window variance is shared by all 52 subcarriers (coherence 0.90). Dividing each sample by its across-subcarrier mean moves walking-vs-empty AUC from 0.79 to **0.98**, and fall-vs-normal from 0.79 to **0.86** (S03) and 0.77 to **0.93** (S04). Neither model's features remove this: RF normalizes per channel only, and the CNN uses dataset z-scores. | Confirmed signal effect; the specific feature was picked after trying about 8 variants on this data | High (effect); medium (exact magnitude) |
| 4 | **Too few independent positives for 575 features or 21k parameters.** | Fold A trains on **3 events** (21 windows). RF training AUC 0.99996 vs validation 0.62. CNN early stopping selects epochs from 1–2 inner fall recordings. | Data limitation | High |
| 5 | **Event metric has a structural ceiling.** | Only eligible (strictly contained) windows are scored, and the alert rule needs a 3-window run. **6/13 covered events (6/10 in S04) cannot fire under any model**: they have 1–2 eligible windows, or a quality gap splits the run. Fall-recording windows outside the interval (pre-fall, get-up) are never scored, so their false alerts are never counted either. | Confirmed evaluation bug | High |
| 6 | **Positive windows include post-fall stillness.** | After gain removal, marker-aligned motion peaks at **+3.0–3.5 s** after `action_start` in S03 (all 4 events, mean z≈3). In S04 it is elevated from **0 to 4 s**, then falls below baseline. Late "contained" windows are mostly lying still. Only 16/52 positive windows contain the in-interval motion peak. | Design tradeoff (label noise) | Medium |
| 7 | **Acquisition loss.** | 21/34 events have no positive. Packet coverage inside the interval equals coverage before and after it (≈0.65–0.99 in all three), so the **falls don't cause the gaps**. The gaps are acquisition-wide and time-dependent. | Confirmed data limitation | High |
| 8 | **CNN can't represent temporal order.** | 3 stride-2 convolutions give a receptive field of about 37/512 samples (≈0.22 s), followed by global average pooling. It cannot encode "burst, then stillness". | Design; secondary | Medium |

**Ruled out.** No block or recording leakage (`fold_indices` enforces it; scalers are fit on training only). Class mapping and loss are correct (softmax[:,1] / `classes_.index(1)` = fall; per-example weighted CE). Offline and live share the same transform (`live_window` → `transform_window`). **Annotation clock misalignment is not supported:** gain-normalized motion is time-locked to the markers. The raw-amplitude profile is flat, which is why earlier work suspected misalignment.

## 2. Labeling integrity

- **Positive labels are aligned but imprecise.** Fall motion sits roughly in [start, start+4.5 s] of a ~7 s operator interval (the notes say "2 s buffer before/after"). Windows starting after about start+2.5 s are mostly post-fall stillness. This is label noise, not a bug.
- **The 9 "overlap" uncovered events are not safely relabelable.** None has a quality-passing window covering the interval midpoint ±0.75 s (0/9). Their passing windows sit at the interval edges (preparation or recovery). An overlap ≥1.5 s rule gives 7/9 events 1–5 windows, but only 2 reach a 3-window run. **Do not relabel these automatically.**
- **Ignored windows (1,615; 1,427 in S04)** are unlabeled pre-fall and post-fall context, not mislabeled data. Pre-action windows that end ≥0.5 s before `action_start` are defensible hard negatives. Post-interval windows (lying, get-up) should stay excluded from training but **must be scored in evaluation**.
- **Quality-rejected windows (1,751)** fail for acquisition reasons, not falls. Relaxing only the gap gate from 0.25 s to 0.5 s would raise S04 covered events from **10 to 21** (v2 rule) or 25 (overlap rule), and alertable events from 4 to 7–13. That recovery is only legitimate with an explicit missing-data mask, plus a check that interpolation amount doesn't predict the label (it already does in S04, AUC 0.71).
- **Normal labels** describe the participant's activity, not the room. "Still" classes show motion comparable to walking in raw amplitude (crowded hackathon room). This is an unverified noise source, and future recordings should log it.

## 3. Evaluation validity

| Result | Trust? | Note |
|---|---|---|
| Per-fold window ROC-AUC (CNN 0.60/0.36, RF 0.62/0.73) | **Yes**: genuine failure | Threshold-free, block-separated. The models really don't transfer. |
| Fold B CNN false positives / Fold A "all normal" | Yes (mechanism now explained) | Rows 1–2 above. Fold A's all-normal output also stacks the 0.5 threshold on uncalibrated, weight-capped scores. |
| Covered-event recall (3/13, 1/13) | **No**: re-evaluate | Ceiling is 7/13 (4/10 in Fold A) because of row 5. |
| False alerts/hour (432.5, 13.6) | **No** | Based on 17.6 min, and excludes fall-recording time outside the event. |
| Pooled window F1 | **No** | Concatenates probabilities from two different models. Not a model property. |

Verdict: the models genuinely fail cross-session, and the event metrics additionally understate recall. Both need fixing, and the fixes are independent.

## 4. Experiment priority table

Shared rules for every experiment:
- Folds A/B are fixed.
- Transforms are window-local or fit on the training block only.
- Thresholds come from training-block predictions only (for example, the 95th percentile of training normals).
- The evaluation population is fixed (E0).
- The existing `d4c10e9207e8` results stay as baseline v1.
- Every experiment except E0 involves training and needs your approval. All of them fit on CPU locally; no Veri.

| ID | Experiment (one change) | Hypothesis | Expected benefit | Effort | Success criterion | Existing data enough? |
|---|---|---|---|---|---|---|
| **E0** | Evaluation v2: score **every** quality-passing window in all 92 recordings. An alert inside [start, end+2 s] counts as a detection; any other alert counts as a false alert, with exposure reported separately for normal and fall recordings. Report per-fold AUC and the structural ceiling. Re-score saved Fold B models with inference only. | Event recall is under-measured | Honest, label-independent yardstick for E1–E5 | 0.5–1 day | Ceiling reported; Fold B v1 vs v2 numbers side by side | Yes |
| **E1** | Unfitted baseline: common-mode gain removal + per-window channel-median DC removal + smoothed motion energy, with a training-block threshold | Motion physics transfers across sessions | Floor every model must beat; possibly deployable on a Pi | 0.5 day | AUC ≥0.85 in both folds; event recall > RF v1 at ≤ RF v1 false alerts/hour | Yes |
| **E2** | Same CNN/RF, input = E1 representation (DC removal first, then gain removal as a second ablation) | Rows 1–3 cause the failure | Removes the level shortcut; CNN false positives collapse | 1 day | Both folds AUC ≥0.80; Fold B CNN normal-FP <10% | Yes |
| **E3** | Labels: positives = windows centred in [start+0.5, start+4.0 s], with the offset **fit on the training block only**. Pre-action windows become negatives. | Late contained windows are mislabeled stillness | Cleaner positives; +3–6 usable windows per event | 0.5 day | E0 event recall up at equal false alerts/hour vs E2 | Yes (weak: 13–16 events) |
| **E4** | Quality: gap gate 0.5 s, plus an interpolation mask channel/feature | Recover 11+ events without a shortcut | Training events 13 → about 21–25 | 1 day | Interp-fraction AUC on training labels ~0.5–0.6; E0 metrics don't drop | Yes (but the confound risk is real) |
| **E5** | Small temporal-context model: logistic regression / small GBM over E1 features from the current + previous 2 windows ("burst then still") | The temporal pattern separates falls from sit/lie/get-up | Better fall vs get-up separation (currently 0.81–0.87) | 1 day | Beats E1 and E2 on E0 metrics in both folds | Yes (exploratory only) |
| **E6** | New block `session_05` (interleaved activities, impact-click marker, acquisition fixed first) used as a **frozen test** | Generalization beyond 2 time-ordered blocks | The only unbiased estimate | Collection time | 6–8 covered falls; ≥30 min of continuous normal activity | **No: needs new data** |

**Testable now:** E0–E5 (representation, level/time shortcut, label geometry, quality masking, evaluation ceiling).
**Needs new recordings:**
- breaking the recording-order confound (interleaved collection)
- a credible false alerts/hour figure (17.6 min is too little)
- cross-participant and room generalization
- any final unbiased score
- verifying the impact timing that E3 assumes

## 5. Recommended strategy

**Change the preprocessing and the evaluation first. Keep a lightweight feature model (E1/E5, with RF as the comparison) as the primary candidate. Demote the CNN to an ablation arm. Collect a new block in parallel.**

Reasoning:
- The best-supported signal is a physics-level motion feature that works across sessions without fitting.
- With 13 events, any learned model mostly learns block identity unless its input can't see level or gain.
- Switching architecture isn't justified yet. Rows 1–3 explain the failure without invoking capacity, and a temporal CNN would still see the level confound.
- Revisit the CNN only if E2 shows it beating E1/E5 in both folds.

## 6. Immediate implementation plan (in order)

1. **Build evaluation v2 (E0) and fix artifact retention.** Add a scorer over all quality-passing windows with event and false-alert rules. Keep fold-prefixed artifact paths (or a single archive) so Fold A checkpoints and probabilities survive. Re-score the saved Fold B CNN/RF with inference only, and publish the v1 vs v2 table. No training required.
2. **Add a versioned gain-normalized transform and the E1 baseline.** New function next to `transform_window`, without changing v2: per-sample common-mode division, then per-window channel-median DC removal. Add unit tests for offline/live parity, plus the unfitted motion-energy detector with a threshold from the training block only. Run it through E0 on Folds A/B.
3. **With approval, run E2 → E3 → E4 locally on CPU, one change at a time.** Use the fixed seed and folds, and compare against E1 under E0. At the same time, do an acquisition pilot (normal recordings must pass v2 gates at ≥90%) and record `session_05` interleaved, with an "impact" marker button. Freeze it untouched as the test block.

## Decision

**Next: fix the input representation (gain/DC removal) and the evaluation harness, then show that a simple gain-normalized motion detector beats the current models across both sessions, before spending any effort on architecture.** In parallel, repair acquisition and record one interleaved, frozen `session_05` block. Without it, no result from Sessions 03/04 can be called validated.
