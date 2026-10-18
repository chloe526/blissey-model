"""Read-only reconstruction audit. Writes reports only; never changes split arrays."""
import argparse
from collections import Counter,defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from csi import load_recording,seconds
from preprocess import fingerprint,label_window,recording_features,transform_window
from training_data import load_data


def quantiles(values):
    return dict(zip(('min','p25','median','p75','p95','max'),map(float,np.quantile(values,[0,.25,.5,.75,.95,1])))) if values else None


def snapshot(root):
    return {p.name:fingerprint(p) for p in sorted(root.iterdir()) if p.suffix in ('.json','.npz')}


def review(data_root,processed):
    data_root=Path(data_root);processed=Path(processed)
    before=snapshot(processed);data,config=load_data(processed)
    result=dict(schema_version=1,processed_file_hashes_before=before,checks={},splits={},records=[],warnings=[])
    seen_ids={};raw_spans=[];hash_groups=defaultdict(list);meta_labels=Counter();packet_identity_mismatches=[]
    arrays={}
    for split,d in data.items():
        arrays[split]={(str(rid),round(float(t),8)):i for i,(rid,t) in enumerate(zip(d.recording_id,d.start))}
        assert len(arrays[split])==len(d.y),'Duplicate window identity'
        result['splits'][split]=dict(shape=list(d.X.shape),class_counts=np.bincount(d.y,minlength=2).tolist(),
            positive_percent=100*float((d.y==1).mean()),stored_recordings=len(set(d.recording_id)),
            annotated_events=0,events_with_retained_positive_windows=0,events_without_retained_positive_windows=[],
            collection_groups=config['group_splits'][split],metadata_session_ids=[],participants=[],
            quality_breakdown=dict(gap_only=0,density_only=0,both=0,pass_windows=0),
            quality_by_class={str(y):dict(gap_only=0,density_only=0,both=0,pass_windows=0) for y in (0,1)},
            label_candidates=0,fully_contained_no_margin=0,overlap_at_least_half_window=0,any_event_overlap=0,
            total_sliding_windows=0,label_excluded=0,
            gap_sensitivity_same_density={str(t):0 for t in (.1,.25,.5,1.)},
            pass_if_only_gap_guard=0,pass_if_recording_mean_rate_density_and_gap=0,
            mean_rate_density_diagnostic_by_class={'0':0,'1':0},
            dev_label_sensitivity_same_quality=({'no_margin':0,'half_window_overlap':0} if split!='test' else None),
            window_counts_per_event=[])
    training_sum=np.zeros(len(config['channel_indices']));training_squares=training_sum.copy();training_n=0
    reconstructed=Counter();group_stats=defaultdict(lambda:dict(starts=[],ends=[],sessions=set(),participants=set(),trials=0))
    for metadata_path in sorted(data_root.rglob('metadata.json')):
        folder=metadata_path.parent
        if not (folder/'measurements.jsonl').exists():continue
        metadata,rows,status,intervals,issues,errors=load_recording(folder)
        rid=metadata['recording_id'];relative=folder.relative_to(data_root);group=relative.parts[0]
        assert rid not in seen_ids,'Duplicate recording ID';seen_ids[rid]=group
        g=group_stats[group];g['trials']+=1;g['sessions'].add(metadata['session']);g['participants'].add(metadata['participant'])
        g['starts'].append(metadata['started_at']);g['ends'].append(metadata['ended_at'])
        raw_spans.append((seconds(rows[0]['host_time']),seconds(rows[-1]['host_time']),group,rid))
        for p in rows:
            if any(p.get(key)!=metadata[key] for key in ('recording_id','session','participant','activity')):
                packet_identity_mismatches.append(rid);break
        h=fingerprint(folder/'measurements.jsonl');hash_groups[h].append(rid)
        split=next(s for s,groups in config['group_splits'].items() if group in groups)
        meta_labels[(split,status)]+=1
        if len({p.get('source_mac') for p in rows})!=1:
            assert rid not in config['recordings'],'Mixed-source trial unexpectedly included'
            continue
        r=config['recordings'][rid]
        assert r['group']==group and r['split']==split and r['session']==metadata['session']
        assert r['measurement_sha256']==h,'Raw source changed'
        assert r['label_status']==status and np.allclose(r['fall_intervals'],intervals),'Source annotation changed'
        d=data[split];s=result['splits'][split]
        s['metadata_session_ids'].append(metadata['session']);s['participants'].append(metadata['participant'])
        times,features,duplicates=recording_features(rows,config['channel_indices'])
        dt=np.diff(times);mean_rate=(len(times)-1)/(times[-1]-times[0]);median_rate=1/np.median(dt)
        s['annotated_events']+=len(intervals)
        record=dict(recording_id=rid,group=group,split=split,activity=metadata['activity'],
            mean_rate_hz=float(mean_rate),median_interarrival_rate_hz=float(median_rate),
            packet_gap_quantiles_seconds=quantiles(dt.tolist()),
            fall_intervals=intervals,positive_windows=0,quality=Counter(),event_window_counts=[],
            dev_mean_rate_positive_windows=0)
        for start in np.arange(times[0],times[-1]-config['window_seconds']+1e-9,config['stride_seconds']):
            end=start+config['window_seconds'];s['total_sliding_windows']+=1
            overlap=max((max(0,min(end,b)-max(start,a)) for a,b in intervals),default=0)
            if intervals:
                s['any_event_overlap']+=int(overlap>0)
                s['overlap_at_least_half_window']+=int(overlap>=config['window_seconds']/2)
                s['fully_contained_no_margin']+=int(any(start>=a and end<=b for a,b in intervals))
                if split!='test':
                    left=max(0,np.searchsorted(times,start,side='right')-1)
                    right=min(len(times),np.searchsorted(times,end,side='left')+1)
                    tt=times[left:right];gg=np.diff(tt)
                    quality_ok=np.max(gg)<=config['max_gap_seconds'] and len(tt)>=max(16,.5*config['window_seconds']/np.median(gg))
                    if quality_ok:
                        s['dev_label_sensitivity_same_quality']['no_margin']+=int(any(start>=a and end<=b for a,b in intervals))
                        s['dev_label_sensitivity_same_quality']['half_window_overlap']+=int(overlap>=config['window_seconds']/2)
            label=label_window(start,end,status,intervals)
            if label is None:s['label_excluded']+=1;continue
            s['label_candidates']+=1
            left=max(0,np.searchsorted(times,start,side='right')-1)
            right=min(len(times),np.searchsorted(times,end,side='left')+1)
            tt=times[left:right];gaps=np.diff(tt)
            maximum=float(np.max(gaps));rate=1/float(np.median(gaps));count=len(tt)
            gap_bad=maximum>config['max_gap_seconds']
            density_bad=count<max(16,.5*rate*config['window_seconds'])
            reason='both' if gap_bad and density_bad else 'gap_only' if gap_bad else 'density_only' if density_bad else 'pass_windows'
            s['quality_breakdown'][reason]+=1;s['quality_by_class'][str(label)][reason]+=1;record['quality'][reason]+=1
            s['pass_if_only_gap_guard']+=int(not gap_bad)
            alternate_ok=not gap_bad and count>=max(16,.5*mean_rate*config['window_seconds'])
            s['pass_if_recording_mean_rate_density_and_gap']+=int(alternate_ok)
            s['mean_rate_density_diagnostic_by_class'][str(label)]+=int(alternate_ok)
            record['dev_mean_rate_positive_windows']+=int(split!='test' and alternate_ok and label==1)
            for threshold in s['gap_sensitivity_same_density']:
                s['gap_sensitivity_same_density'][threshold]+=int(not density_bad and maximum<=float(threshold))
            key=(rid,round(float(start),8))
            if reason!='pass_windows':assert key not in arrays[split],'Rejected window was stored';continue
            assert key in arrays[split],'Passing window missing from artifact'
            index=arrays[split][key];assert d.y[index]==label and np.isclose(d.end[index],end)
            x=transform_window(times,features,start,config)
            np.testing.assert_allclose(x,d.X[index],rtol=2e-5,atol=3e-6)
            reconstructed[split]+=1
            if split=='train':
                raw=transform_window(times,features,start,config,normalize=False).astype(np.float64)
                training_sum+=raw.sum(axis=1);training_squares+=(raw*raw).sum(axis=1);training_n+=raw.shape[1]
            record['positive_windows']+=int(label==1)
        for event_index,(a,b) in enumerate(intervals):
            selected=(d.recording_id==rid)&(d.y==1)&(d.start>=a)&(d.end<=b)
            count=int(selected.sum());record['event_window_counts'].append(count);s['window_counts_per_event'].append(count)
            if count:s['events_with_retained_positive_windows']+=1
            else:s['events_without_retained_positive_windows'].append(dict(recording_id=rid,event_index=event_index))
        record['quality']=dict(record['quality']);result['records'].append(record)
    for split,s in result['splits'].items():
        assert reconstructed[split]==len(data[split]),'Not all stored windows verified'
        quality=s['quality_breakdown'];assert sum(quality[k] for k in ('gap_only','density_only','both'))==config['quality'][split]['rejected_quality_windows']
        s['metadata_session_ids']=sorted(set(s['metadata_session_ids']));s['participants']=sorted(set(s['participants']))
        s['source_label_recording_counts']={status:n for (part,status),n in meta_labels.items() if part==split}
        s['mean_packet_rate_hz']=quantiles([r['mean_rate_hz'] for r in result['records'] if r['split']==split])
        s['median_interarrival_rate_hz']=quantiles([r['median_interarrival_rate_hz'] for r in result['records'] if r['split']==split])
    actual_mean=training_sum/training_n;actual_std=np.sqrt(training_squares/training_n-actual_mean**2)
    np.testing.assert_allclose(actual_mean,config['mean'],rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(actual_std,config['std'],rtol=1e-6,atol=1e-6)
    overlaps=[]
    for i,(a,b,group,rid) in enumerate(raw_spans):
        for c,d,other,rid2 in raw_spans[i+1:]:
            if group!=other and min(b,d)>max(a,c):overlaps.append([rid,rid2])
    duplicates=[ids for ids in hash_groups.values() if len(ids)>1]
    result['checks'].update(all_source_and_artifact_hashes_match=True,all_windows_reconstructed=True,
        labels_rederived_without_changes=True,normalization_refit_train_only_matches=True,
        source_group_assignment_matches=True,cross_split_recording_and_group_overlap=False,
        cross_group_capture_time_overlaps=overlaps,duplicate_source_files=duplicates,
        packet_identity_mismatches=packet_identity_mismatches,participant_overlap_val_test=sorted(set(result['splits']['val']['participants'])&set(result['splits']['test']['participants'])))
    result['collection_groups']={g:dict(recordings=x['trials'],metadata_ids=sorted(x['sessions']),participants=sorted(x['participants']),first_utc=min(x['starts']),last_utc=max(x['ends'])) for g,x in group_stats.items()}
    # Architecture-only meta validation does not allocate or download pretrained weights.
    import torch
    from momentfm import MOMENTPipeline
    from momentfm.models.moment import MOMENT
    architecture=json.loads(Path('configs/moment-architecture.json').read_text())
    with torch.device('meta'):
        model=MOMENTPipeline(config=architecture,model_kwargs=dict(task_name='classification',n_channels=len(config['channel_indices']),num_class=2,reduction='concat'));model.init()
    result['moment']=dict(input_shape=['B',52,512],mask_shape=['B',512],patches=512//model.patch_len,
        head_input=model.head.linear.in_features,head_output=model.head.linear.out_features,
        trainable_head_parameters=sum(p.numel() for p in model.head.parameters()),
        native_multichannel=True,pretrained_forward_run=False,
        installed_classify_source_sha256=hashlib.sha256(inspect.getsource(MOMENT.classify).encode()).hexdigest(),
        official_source='https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/momentfm/models/moment.py')
    result['processed_file_hashes_after']=snapshot(processed)
    assert before==result['processed_file_hashes_after'],'Audit mutated input artifacts'
    result['checks']['processed_files_unchanged']=True
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',default='../blissey-data/recordings');p.add_argument('--processed',default='data/processed');p.add_argument('--output');a=p.parse_args()
    config=json.loads((Path(a.processed)/'preprocessing_config.json').read_text())
    if config.get('preprocessing_version',1)>=2:
        from audit_quality_v2 import audit_revised
        result=audit_revised(Path(a.data_root),Path('data/processed_legacy_v1'),Path(a.processed))
        output=Path(a.output or 'reports/preprocessing_review_v2.json')
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('Quality v2 comparison and reconstruction passed:',output)
        return
    result=review(a.data_root,a.processed)
    Path(a.output or 'reports/preprocessing_review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({s:{k:v for k,v in stats.items() if k not in ('events_without_retained_positive_windows',)} for s,stats in result['splits'].items()},indent=2))
    print('All stored windows and labels reconstructed; processed files unchanged.')

if __name__=='__main__':main()
