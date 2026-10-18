"""Verify regenerated arrays against source labels and the preserved v1 baseline."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from csi import load_recording
from csi_quality import window_quality
from evaluate import union_duration
from preprocess import fingerprint,label_window,recording_features,transform_window
from training_data import load_data


def identities(dataset):
    return {(str(rid),round(float(t),8)):i for i,(rid,t) in enumerate(zip(dataset.recording_id,dataset.start))}


def split_summary(dataset,config,split):
    records={rid:r for rid,r in config['recordings'].items() if r['split']==split}
    observed_seconds=0.;events=0;retained=0;event_counts={}
    for rid,r in records.items():
        mask=dataset.recording_id==rid
        observed_seconds+=union_duration(list(zip(dataset.start[mask],dataset.end[mask])))
        for i,(a,b) in enumerate(r['fall_intervals']):
            count=int((mask&(dataset.y==1)&(dataset.start>=a)&(dataset.end<=b)).sum())
            event_counts[f'{rid}:{i}']=count;events+=1;retained+=bool(count)
    duration=sum(r['exposure_seconds'] for r in records.values())
    return dict(accepted_windows=len(dataset),rejected_quality_windows=config['quality'][split]['rejected_quality_windows'],
        class_counts=np.bincount(dataset.y,minlength=2).tolist(),
        retained_fall_events=retained,annotated_fall_events=events,positive_windows_per_event=event_counts,
        contributing_recordings=len(set(dataset.recording_id)),eligible_recordings=len(records),
        collection_sessions=len(config['group_splits'][split]),
        sensor_coverage=dict(csi_channels=len(config['channel_indices']),radio_links=1,
            eligible_recording_seconds=duration,accepted_labelled_window_union_seconds=observed_seconds,
            accepted_labelled_recording_fraction=observed_seconds/duration))


def audit_revised(root,old_root,new_root):
    old,oc=load_data(old_root);new,nc=load_data(new_root)
    assert nc['preprocessing_version']==2
    for key in ('seed','group_splits','window_seconds','stride_seconds','sequence_length','label_policy','channel_indices','recordings'):
        assert oc[key]==nc[key],f'Changed frozen provenance/label field: {key}'
    lock=json.loads(Path('reports/quality_threshold_lock.json').read_text())
    assert lock['policy']==nc['packet_quality'] and lock['policy_sha256']==nc['quality_config_sha256']
    assert fingerprint('configs/packet_quality_v2.json')==lock['policy_sha256']
    assert fingerprint(nc['packet_quality']['development_evidence'])==nc['packet_quality']['development_evidence_sha256']
    manifest=json.loads((Path(old_root)/'baseline_manifest.json').read_text())
    assert all(fingerprint(Path(old_root)/name)==h for name,h in manifest.items()),'Baseline was changed'
    before_hash={p.name:fingerprint(p) for p in Path(new_root).iterdir() if p.suffix in ('.npz','.json')}
    result=dict(preprocessing_version=2,baseline=str(old_root),revised=str(new_root),checks={},splits={})
    paths={json.loads(p.read_text())['recording_id']:p.parent for p in Path(root).rglob('metadata.json')}
    for split in ('train','val','test'):
        od,nd=old[split],new[split];before=split_summary(od,oc,split);after=split_summary(nd,nc,split)
        assert nd.X.shape[1:]==(52,512) and nd.X.dtype==np.float32
        oi,ni=identities(od),identities(nd);assert len(ni)==len(nd)
        metric_names=('effective_rate_hz','occupied_bin_fraction','sample_span_fraction','max_edge_distance_seconds','max_gap_seconds')
        with np.load(Path(new_root)/f'{split}.npz',allow_pickle=False) as archive:
            saved_metrics={name:archive[name].copy() for name in metric_names}
        for key in oi.keys()&ni.keys():assert od.y[oi[key]]==nd.y[ni[key]],'Shared window changed label'
        reasons=Counter();combinations=Counter();candidates=0;verified=0;candidate_seconds=0.
        candidate_class=Counter();accepted_metrics=[];sums=np.zeros(len(nc['channel_indices']));squares=sums.copy();points=0
        for rid,r in nc['recordings'].items():
            if r['split']!=split:continue
            folder=paths[rid]
            assert folder.relative_to(Path(root)).parts[0] in nc['group_splits'][split]
            metadata,rows,status,intervals,issues,errors=load_recording(folder)
            assert not errors and fingerprint(folder/'measurements.jsonl')==r['measurement_sha256']
            assert status==r['label_status'] and intervals==r['fall_intervals']
            times,features,dupes=recording_features(rows,nc['channel_indices']);spans=[]
            for start in np.arange(times[0],times[-1]-nc['window_seconds']+1e-9,nc['stride_seconds']):
                end=start+nc['window_seconds'];label=label_window(start,end,status,intervals)
                if label is None:continue
                candidates+=1;candidate_class[int(label)]+=1;spans.append((float(start),float(end)))
                key=(rid,round(float(start),8));diagnostic=window_quality(times,start,nc)
                failure=diagnostic['reasons']
                if not failure:
                    try:actual=transform_window(times,features,start,nc)
                    except ValueError as exc:failure=[str(exc)]
                if failure:
                    assert key not in ni,'Rejected window was stored'
                    reasons.update(failure);combinations[','.join(failure)]+=1;continue
                assert key in ni,'Passing source window is missing'
                i=ni[key];assert nd.y[i]==label and np.isclose(nd.end[i],end)
                np.testing.assert_allclose(actual,nd.X[i],rtol=2e-5,atol=3e-6)
                for name in metric_names:assert np.isclose(saved_metrics[name][i],diagnostic[name])
                accepted_metrics.append(diagnostic);verified+=1
                if split=='train':
                    raw=transform_window(times,features,start,nc,normalize=False).astype(np.float64)
                    sums+=raw.sum(axis=1);squares+=(raw*raw).sum(axis=1);points+=raw.shape[1]
            candidate_seconds+=union_duration(spans)
        assert verified==len(nd) and candidates==len(od)+oc['quality'][split]['rejected_quality_windows']
        assert candidates==len(nd)+nc['quality'][split]['rejected_quality_windows']
        assert dict(combinations)==nc['quality'][split]['rejection_reasons']
        if split=='train':
            mean=sums/points;std=np.maximum(np.sqrt(squares/points-mean**2),1e-6)
            np.testing.assert_allclose(mean,nc['mean'],rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(std,nc['std'],rtol=1e-6,atol=1e-6)
        for summary in (before,after):
            summary['sensor_coverage']['label_candidate_window_union_seconds']=candidate_seconds
            summary['sensor_coverage']['accepted_fraction_of_label_candidate_exposure']=summary['sensor_coverage']['accepted_labelled_window_union_seconds']/candidate_seconds
        for name,values in saved_metrics.items():
            assert len(values)==len(nd) and np.isfinite(values).all()
        result['splits'][split]=dict(before=before,after=after,label_candidate_windows=candidates,
            candidate_class_counts=[candidate_class[0],candidate_class[1]],
            newly_accepted_windows=len(ni.keys()-oi.keys()),newly_rejected_windows=len(oi.keys()-ni.keys()),
            rejection_flags=dict(reasons),rejection_combinations=dict(combinations),
            accepted_quality_ranges={name:dict(min=float(min(q[name] for q in accepted_metrics)),median=float(np.median([q[name] for q in accepted_metrics])),max=float(max(q[name] for q in accepted_metrics))) for name in ('effective_rate_hz','occupied_bin_fraction','max_gap_seconds')},
            tensor_shape=list(nd.X.shape))
    after_hash={p.name:fingerprint(p) for p in Path(new_root).iterdir() if p.suffix in ('.npz','.json')}
    assert before_hash==after_hash
    result['checks']=dict(baseline_hashes_preserved=True,source_hashes_verified=True,
        original_session_splits_unchanged=True,annotation_intervals_and_label_policy_unchanged=True,
        common_window_labels_unchanged=True,all_revised_windows_reconstructed=True,
        train_only_scaler_refit_matches=True,independent_split_checks_passed=True,
        quality_thresholds_match_development_only_lock=True,test_used_for_threshold_selection=False,
        finite_moment_shape_B_52_512=True,revised_files_unchanged_during_audit=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',default='../blissey-data/recordings')
    p.add_argument('--baseline',default='data/processed_legacy_v1');p.add_argument('--processed',default='data/processed')
    p.add_argument('--output',default='reports/quality_v2_comparison.json');a=p.parse_args()
    r=audit_revised(Path(a.data_root),Path(a.baseline),Path(a.processed))
    Path(a.output).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    for split,s in r['splits'].items():
        b,c=s['before'],s['after']
        print(f"{split}: accepted {b['accepted_windows']} -> {c['accepted_windows']}; rejected {b['rejected_quality_windows']} -> {c['rejected_quality_windows']}; classes {b['class_counts']} -> {c['class_counts']}; events {b['retained_fall_events']} -> {c['retained_fall_events']}")
    print('All provenance, labels, scaler, sensor coverage and MOMENT tensor checks passed.')

if __name__=='__main__':main()
