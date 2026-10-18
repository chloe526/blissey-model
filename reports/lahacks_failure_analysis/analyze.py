"""Descriptive analysis of saved LA Hacks CV outputs; never trains or tunes."""
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from threadpoolctl import threadpool_limits

MODEL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL))
from lahacks_cv.common import sha, write_json
from lahacks_cv.data import load_dataset
from lahacks_cv.features import extract
from lahacks_cv.metrics import ALERT_POLICY, alert_decisions, union_seconds

OUT = Path(__file__).resolve().parent
JOB = MODEL / 'reports/lahacks_job_d4c10e9207e8'
DATA = MODEL / 'data/lahacks_quality_v2_cv_v1'


def save_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open('w', newline='') as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, metadata = load_dataset(DATA)
    raw_level = np.median(np.median(data['X'], axis=2), axis=1)
    raw_temporal_sd = np.mean(np.std(data['X'], axis=2), axis=1)
    base = JOB / 'artifacts'
    benchmark = json.loads((base / 'benchmark.json').read_text())
    p = base / 'predictions.npz'
    with np.load(p, allow_pickle=False) as z:
        predfile = {k: z[k] for k in z.files}
    mask03 = data['group'] == 'session_03_LAHACKS'
    rows03 = np.flatnonzero(mask03)
    np.testing.assert_array_equal(data['recording_id'][rows03], predfile['recording_id'])
    np.testing.assert_array_equal(data['start'][rows03], predfile['start'])
    np.testing.assert_array_equal(data['y'][rows03], predfile['y'])
    if len(rows03) != 1270:
        raise ValueError('Fold B predictions must align with all 1,270 Session 03 windows')

    # Read immutable CV predictions, then align them to canonical source windows.
    probability = {name: predfile[name + '_probability'] for name in ('cnn', 'rf')}
    all_rows = []
    for fold, group in (('A', 'session_04_LAHACKS'), ('B', 'session_03_LAHACKS')):
        indices = np.flatnonzero(data['group'] == group)
        for model in ('cnn', 'rf'):
            saved = benchmark['folds'][fold][model]
            y = data['y'][indices]
            if fold == 'A':
                # Persisted aggregate confusion matrix establishes every score < 0.5,
                # but the scores themselves were flattened/lost by Veri's export API.
                if saved['window']['confusion_matrix'] != [[301, 0], [31, 0]]:
                    raise ValueError('Unexpected fold-A saved confusion matrix')
                pred = np.zeros(len(indices), dtype=np.int8)
            else:
                pred = (probability[model] >= ALERT_POLICY['threshold']).astype(np.int8)
            for local, j in enumerate(indices):
                rid = str(data['recording_id'][j]); rec = metadata['records'][rid]
                if fold == 'B':
                    fall_window = int(y[local] == 1); fp = int(y[local] == 0 and pred[local] == 1)
                    fn = int(y[local] == 1 and pred[local] == 0); tp = int(y[local] == 1 and pred[local] == 1)
                else:
                    fall_window = int(y[local] == 1); fp = 0; fn = fall_window; tp = 0
                all_rows.append(dict(fold=fold, session=group, model=model, recording_id=rid,
                    activity=rec['activity'], label=int(y[local]), predicted=int(pred[local]),
                    false_positive=fp, false_negative=fn, true_positive=tp, window_index=int(j)))

    confusion_rows = []
    for fold, model, activity in sorted({(r['fold'], r['model'], r['activity']) for r in all_rows}):
        rr = [r for r in all_rows if (r['fold'], r['model'], r['activity']) == (fold, model, activity)]
        confusion_rows.append(dict(fold=fold, session=rr[0]['session'], model=model, activity=activity,
            recordings=len({r['recording_id'] for r in rr}), windows=len(rr),
            true_positive=sum(r['true_positive'] for r in rr), false_positive=sum(r['false_positive'] for r in rr),
            false_negative=sum(r['false_negative'] for r in rr),
            true_negative=sum(r['label'] == 0 and r['predicted'] == 0 for r in rr)))
    save_csv('errors_by_activity.csv', confusion_rows)
    save_csv('errors_by_recording.csv', all_rows)

    event_rows = []
    audit = json.loads((MODEL / 'reports/lahacks_stage1/audit.json').read_text())
    audit_events = {e['event_id']: e for block in audit['blocks'].values() for e in block['events']}
    positive_audit = {}
    with (MODEL / 'reports/lahacks_stage1/window_audit.jsonl').open() as f:
        for line in f:
            w = json.loads(line)
            if w['event_id']:
                q = positive_audit.setdefault(w['event_id'], dict(candidates=0, quality_pass=0,
                    label_rejected=0, gap=0, coverage=0, span=0, edge=0, rate=0))
                q['candidates'] += 1
                q['quality_pass'] += int(w['outcome'] == 'fall')
                q['label_rejected'] += int(w['outcome'] == 'ignore')
                reasons = set(w['quality_reasons'])
                q['gap'] += int('excessive_packet_gap' in reasons)
                q['coverage'] += int('insufficient_temporal_coverage' in reasons)
                q['span'] += int('insufficient_sample_span' in reasons)
                q['edge'] += int('unsupported_window_edge' in reasons)
                q['rate'] += int('insufficient_effective_rate' in reasons)
    for group, block in audit['blocks'].items():
        fold = 'A' if group == 'session_04_LAHACKS' else 'B'
        model_scores = benchmark['folds'][fold]
        for event in block['events']:
            rid, eid = event['event_id'].split(':fall:')
            rec = metadata['records'][rid]
            q = positive_audit.get(event['event_id'], dict(candidates=0, quality_pass=0, label_rejected=0,
                gap=0, coverage=0, span=0, edge=0, rate=0))
            for model in ('cnn', 'rf'):
                stat = next(e for e in model_scores[model]['events']['per_event']
                            if e['event_id'] == event['event_id'])
                windows = [r for r in all_rows if r['model'] == model and r['recording_id'] == rid and r['label'] == 1]
                detected = stat['detected'] if stat['covered'] else None
                event_rows.append(dict(fold=fold, session=group, model=model, recording_id=rid,
                    activity=rec['activity'], event_id=event['event_id'], quality_covered=bool(q['quality_pass']),
                    positive_candidates=q['candidates'], passing_positive_windows=q['quality_pass'],
                    quality_pass_windows_overlapping_action=audit_events[event['event_id']]['quality_pass_windows_overlapping_action'],
                    labeling_excluded=q['label_rejected'], rejected_gap=q['gap'], rejected_coverage=q['coverage'],
                    rejected_span=q['span'], rejected_edge=q['edge'], rejected_rate=q['rate'],
                    window_false_negatives=sum(r['false_negative'] for r in windows),
                    alert_detected=(None if fold == 'A' else detected)))
    save_csv('fall_events.csv', event_rows)

    # Score-distribution figure: the Fold A scores cannot be reconstructed from
    # its saved predictions or remote checkpoint manifest, so display that gap.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    colors = {0: '#3274a1', 1: '#c44e52'}
    for col, model in enumerate(('cnn', 'rf')):
        ax = axes[0, col]
        ax.text(.5, .56, 'Per-window scores unavailable\nVeri flattened Fold A paths', ha='center', va='center', transform=ax.transAxes)
        ax.text(.5, .38, 'Saved aggregate: 0/332 predicted falls', ha='center', va='center', transform=ax.transAxes)
        ax.set(title=f'{model.upper()} · Fold A (train 03 → validate 04)', xlabel='Fall probability', ylabel='Density', xlim=(0, 1))
        ax = axes[1, col]
        for label, name in ((0, 'normal'), (1, 'fall')):
            values = probability[model][predfile['y'] == label]
            ax.hist(values, bins=np.linspace(0, 1, 31), density=True, histtype='step', linewidth=1.8,
                    color=colors[label], label=f'{name} (n={len(values)})')
        ax.axvline(.5, color='black', linestyle='--', linewidth=1, label='fixed 0.50 threshold')
        ax.set(title=f'{model.upper()} · Fold B (train 04 → validate 03)', xlabel='Fall probability', ylabel='Density', xlim=(0, 1))
        ax.legend(fontsize=8)
    fig.suptitle('Saved out-of-block probability distributions (Fold A scores missing)')
    fig.savefig(OUT / 'probabilities_by_fold.png', dpi=160); plt.close(fig)

    # Descriptive threshold sweep of Fold B only. No threshold is selected.
    threshold_rows = []
    for model in ('cnn', 'rf'):
        scores = probability[model]; yy = predfile['y']
        for threshold in np.round(np.arange(.05, 1., .05), 2):
            pr = (scores >= threshold).astype(np.int8)
            tp = int(((yy == 1) & (pr == 1)).sum()); fn = int(((yy == 1) & (pr == 0)).sum())
            fp = int(((yy == 0) & (pr == 1)).sum()); tn = int(((yy == 0) & (pr == 0)).sum())
            alert_count = 0; exposure = 0.; covered_events = 0; detected_events = 0
            for rid, rec in metadata['records'].items():
                if rec['group'] != 'session_03_LAHACKS':
                    continue
                local = np.flatnonzero(predfile['recording_id'] == rid)
                if not len(local): continue
                order = local[np.argsort(predfile['end'][local])]
                alerts = alert_decisions(predfile['end'][order], scores[order], dict(ALERT_POLICY, threshold=float(threshold)))
                if rec['status'] == 'verified_non_fall':
                    alert_count += int(alerts.sum())
                    exposure += union_seconds(list(zip(predfile['start'][order], predfile['end'][order])))
                else:
                    for number, (a, b) in enumerate(rec['intervals']):
                        eligible = data['event_id'][rows03[order]] == f'{rid}:fall:{number}'
                        if eligible.any():
                            covered_events += 1
                            detected_events += int(np.any(alerts & eligible & (predfile['end'][order] >= a) & (predfile['end'][order] <= b)))
            threshold_rows.append(dict(model=model, fold='B', threshold=float(threshold), tp=tp, fp=fp, fn=fn, tn=tn,
                window_recall=tp/(tp+fn) if tp+fn else None, window_precision=tp/(tp+fp) if tp+fp else 0.,
                quality_covered_events=covered_events, detected_covered_events=detected_events,
                covered_event_recall=detected_events/covered_events if covered_events else None,
                false_alert_episodes=alert_count, false_alerts_per_hour=alert_count/(exposure/3600)))
    save_csv('threshold_sweep_descriptive_only.csv', threshold_rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for model, color in (('cnn', '#c44e52'), ('rf', '#3274a1')):
        rr = [r for r in threshold_rows if r['model'] == model]
        axes[0].plot([r['threshold'] for r in rr], [r['window_recall'] for r in rr], marker='o', ms=3, color=color, label=model.upper())
        axes[1].plot([r['false_alerts_per_hour'] for r in rr], [r['window_recall'] for r in rr], marker='o', ms=3, color=color, label=model.upper())
        axes[2].plot([r['false_alerts_per_hour'] for r in rr], [r['covered_event_recall'] for r in rr], marker='o', ms=3, color=color, label=model.upper())
        fixed = next(r for r in rr if r['threshold'] == .5)
        axes[1].annotate('0.50 fixed', (fixed['false_alerts_per_hour'], fixed['window_recall']), fontsize=8)
        axes[2].annotate('0.50 fixed', (fixed['false_alerts_per_hour'], fixed['covered_event_recall']), fontsize=8)
    axes[0].set(xlabel='Descriptive threshold (not selected)', ylabel='Window fall recall', ylim=(-.03, 1.03))
    axes[1].set(xlabel='False alert episodes/hour', ylabel='Window fall recall')
    axes[2].set(xlabel='False alert episodes/hour', ylabel='Covered-event recall', ylim=(-.03, 1.03))
    for ax in axes: ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Fold B threshold tradeoffs · descriptive only, no validation tuning')
    fig.savefig(OUT / 'threshold_tradeoffs_fold_b.png', dpi=160); plt.close(fig)

    # Window-local features are unsupervised descriptive summaries, not fitted transforms.
    with threadpool_limits(4): features = extract(data['X'])
    names = [f'variance ch-{i}' for i in range(52)] + [f'mean |diff| ch-{i}' for i in range(52)] + \
            [f'diff variance ch-{i}' for i in range(52)] + [f'third {part} variance ch-{i}' for part in ('early','middle','late') for i in range(52)] + \
            [f'peak/end ratio ch-{i}' for i in range(52)] + [f'energy ch-{i}' for i in range(52)] + \
            [f'{band}Hz power ch-{i}' for band in ('0-1','1-3','3-10') for i in range(52)] + [f'PCA fraction {i}' for i in range(1,4)]
    if len(names) != features.shape[1]: raise ValueError('Feature map mismatch')
    thirds = features[:, 156:312].reshape(-1, 3, 52)
    descriptors = {'mean_channel_variance': np.mean(features[:, :52], axis=1),
        'raw_median_amplitude': raw_level, 'raw_temporal_standard_deviation': raw_temporal_sd,
        'mean_absolute_first_difference': np.mean(features[:, 52:104], axis=1),
        'mean_first_difference_variance': np.mean(features[:, 104:156], axis=1),
        'early_middle_late_variance_ratio': thirds[:,0].mean(axis=1) /
             np.maximum(thirds[:,2].mean(axis=1), 1e-9),
        'mean_peak_end_motion_ratio': np.mean(features[:, 312:364], axis=1),
        'mean_normalized_energy': np.mean(features[:, 364:416], axis=1),
        'mean_0_1hz_power': np.mean(features[:, 416:468], axis=1),
        'mean_1_3hz_power': np.mean(features[:, 468:520], axis=1),
        'mean_3_10hz_power': np.mean(features[:, 520:572], axis=1),
        'pca_top1_fraction': features[:,572], 'pca_top3_fraction_sum': features[:,572:].sum(axis=1)}
    shift_rows=[]
    for descriptor, values in descriptors.items():
        if values.shape != data['y'].shape:
            raise ValueError(f'Descriptor alignment mismatch: {descriptor} {values.shape} vs {data["y"].shape}')
        for label, cname in ((0,'normal'),(1,'fall')):
            one = values[(data['group']=='session_03_LAHACKS') & (data['y']==label)]
            two = values[(data['group']=='session_04_LAHACKS') & (data['y']==label)]
            if len(one) and len(two):
                sd=np.sqrt((one.var(ddof=1)+two.var(ddof=1))/2)
                shift_rows.append(dict(comparison='session03 vs session04', activity_or_class=cname,
                    feature=descriptor, n03=len(one), n04=len(two), median03=float(np.median(one)), median04=float(np.median(two)),
                    standardized_mean_difference=float((one.mean()-two.mean())/sd) if sd else 0.,
                    mann_whitney_p_descriptive=float(mannwhitneyu(one,two,alternative='two-sided').pvalue)))
    confounds=('sit_down','lie_down','get_up_from_floor')
    for group in ('session_03_LAHACKS','session_04_LAHACKS'):
        for activity in confounds:
            normal_idx=np.flatnonzero((data['group']==group)&(data['y']==0)&np.array([metadata['records'][str(r)]['activity']==activity for r in data['recording_id']]))
            fall_idx=np.flatnonzero((data['group']==group)&(data['y']==1))
            for descriptor,values in descriptors.items():
                if len(normal_idx) and len(fall_idx):
                    u=mannwhitneyu(values[fall_idx],values[normal_idx],alternative='two-sided')
                    auc=float(u.statistic/(len(fall_idx)*len(normal_idx)))
                    shift_rows.append(dict(comparison=f'falls vs {activity}',activity_or_class=group,feature=descriptor,
                        n03=len(fall_idx),n04=len(normal_idx),median03=float(np.median(values[fall_idx])),
                        median04=float(np.median(values[normal_idx])),standardized_mean_difference=None,
                        mann_whitney_p_descriptive=float(u.pvalue),rank_auc_fall_higher=auc))
    save_csv('feature_comparisons.csv',shift_rows)
    # Plot standardized session shifts for normal/fall and effect-size-like AUC for hard negatives.
    chosen=('mean_channel_variance','mean_absolute_first_difference','mean_first_difference_variance',
            'early_middle_late_variance_ratio','mean_peak_end_motion_ratio','mean_normalized_energy',
            'mean_0_1hz_power','mean_1_3hz_power','mean_3_10hz_power','pca_top1_fraction',
            'raw_median_amplitude','raw_temporal_standard_deviation')
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,label in zip(axes,('normal','fall')):
        rr=[r for r in shift_rows if r['comparison']=='session03 vs session04' and r['activity_or_class']==label and r['feature'] in chosen]
        ax.barh([r['feature'] for r in rr],[r['standardized_mean_difference'] for r in rr],color='#4c72b0')
        ax.axvline(0,color='black',lw=.7);ax.axvline(1,color='gray',lw=.7,ls='--');ax.axvline(-1,color='gray',lw=.7,ls='--')
        ax.set_title('Session 03 minus 04 · '+label);ax.set_xlabel('Standardized mean difference (window level)')
    fig.suptitle('Window-local CSI feature shifts; windows are correlated')
    fig.savefig(OUT / 'session_feature_shift.png',dpi=160);plt.close(fig)

    # Event coverage constraints for every annotated event, sourced from existing v2 audit.
    constraints={}
    for row in event_rows:
        if row['model']=='cnn':constraints[row['event_id']]=row
    covered=sum(r['quality_covered'] for r in constraints.values())
    lost_reasons={k:sum(r['rejected_'+k] for r in constraints.values()) for k in ('gap','coverage','span','edge','rate')}
    event_summary=dict(annotated_events=34,usable=sum(r['quality_covered'] for r in event_rows if r['model']=='cnn'),
        unusable=34-covered,by_session={},loss_reason_candidate_window_counts=lost_reasons,
        events=[dict(event_id=r['event_id'],session=r['session'],activity=r['activity'],covered=r['quality_covered'],
                     quality_passing_positive_windows=r['passing_positive_windows'],candidate_positive_windows=r['positive_candidates'],
                     quality_pass_windows_overlapping_action=r['quality_pass_windows_overlapping_action'],
                     label_excluded=r['labeling_excluded'],rejected={k:r['rejected_'+k] for k in ('gap','coverage','span','edge','rate')})
                for r in constraints.values()])
    for group in ('session_03_LAHACKS','session_04_LAHACKS'):
        rr=[e for e in event_summary['events'] if e['session']==group]
        event_summary['by_session'][group]=dict(events=len(rr),usable=sum(e['covered'] for e in rr),unusable=sum(not e['covered'] for e in rr))
    write_json(OUT/'event_coverage.json',event_summary)
    # Identify top shifts/confoundings without significance-based feature selection.
    normal_shift=sorted([r for r in shift_rows if r['comparison']=='session03 vs session04' and r['activity_or_class']=='normal'],
                        key=lambda r:abs(r['standardized_mean_difference']),reverse=True)
    confound_summary={a:[r for r in shift_rows if r['comparison']==f'falls vs {a}' and r['feature'] in chosen] for a in confounds}
    counts_by_model={m:{f:{'FP':int(benchmark['folds'][f][m]['window']['confusion_matrix'][0][1]),
                           'FN':int(benchmark['folds'][f][m]['window']['confusion_matrix'][1][0]),
                           'TP':int(benchmark['folds'][f][m]['window']['confusion_matrix'][1][1]),
                           'TN':int(benchmark['folds'][f][m]['window']['confusion_matrix'][0][0])} for f in ('A','B')}
                     for m in ('cnn','rf')}
    report=dict(title='Why LA Hacks-only CNN/RF failed across collection blocks',
        scope='Saved quality-v2 windows and completed Veri job d4c10e9207e8; no model training, threshold selection, or data changes.',
        prediction_availability='Fold B per-window outputs are available. Fold A scores, event list, and checkpoints are absent from Veri final artifacts because its manifest flattened paths and collisions retained Fold B basenames. Fold A error totals and the fact all scores <0.5 are available only from benchmark.json; activity-level FP=0 and all covered fold-A events are missed can be stated, but score distributions and event-by-event predictions cannot be reconstructed.',
        overall=dict(blocks=2,participants=1,quality_covered_fall_events=13,annotated_events=34,windows=1602,normal=1550,fall=52),
        fold_confusion_counts=counts_by_model,events=event_summary,
        fold_B_activity_errors=confusion_rows,
        feature_shift_top_normal=normal_shift[:6],
        confound_feature_comparisons=confound_summary,
        threshold_curve='threshold_sweep_descriptive_only.csv; Fold B only; descriptive sweep, fixed policy, threshold 0.5 marked, no selected optimum.',
        recommendations=[
            'Acquisition first: Session 04 has high genuine device/host packet gaps; 20 of 30 attempts lose every v2 positive window to quality. Pilot and repair packet capture, then interleave fall/hard-negative trials across time.',
            'Collect independent variation: both blocks are p1, one venue/setup, and Session 04 is class/time ordered; add participants and interleaved blocks, reserving a genuinely new block as frozen final test.',
            'Improve labels/coverage before model search: time-sync and review descent onset/end; preserve all attempts and uncovered-event accounting. Do not loosen v2 gates to manufacture positives.'
        ],
        caveats=['Two blocks cannot estimate participant or site generalization.',
                 'Window comparisons are correlated by overlapping 0.5s stride; p-values are descriptive only.',
                 'RF spectra use interpolated 512-point 3s windows and are not native-rate spectral measurements.',
                 'Only 17.6 minutes of accepted normal exposure; false-alert/hour estimates are unstable.',
                 'No raw probabilities for Fold A; those plot panels are intentionally marked unavailable.'])
    write_json(OUT/'analysis.json',report)
    print(json.dumps(dict(output=str(OUT), windows=len(all_rows), fall_events=len(event_rows)//2,
        covered=covered, foldB_feature_comparisons=len(shift_rows), top_normal_shift=normal_shift[:3],
        foldB_false_positives={m:counts_by_model[m]['B']['FP'] for m in ('cnn','rf')}),indent=2))


if __name__=='__main__': main()
