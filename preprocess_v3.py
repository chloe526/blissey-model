"""Reviewed descent-onset supervision; audit by default, materialize only if ready."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from csi import load_recording
from csi_quality import window_quality
from inspect_dataset import audit
from preprocess import fingerprint,label_window,recording_features,transform_window
from labeling_v3 import decision,timing_usable,validate_event,validate_groups,validate_policy,validate_config

ROOT=Path(__file__).resolve().parent


def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def source_events(record):
    return [dict(event_id=f"{record['recording_id']}:fall:{i}",source_event_index=i,status='needs_review',
        source_action_interval_seconds=span,onset_bounds_seconds=None,motion_end_bounds_seconds=None,
        reason='Operator action clicks do not verify descent onset, motion end or impact')
        for i,span in enumerate(record['fall_intervals'])]


def resolve_events(record,entry,source_hashes):
    events=source_events(record)
    if entry is None:return events
    if entry.get('source_sha256')!=source_hashes:raise ValueError(f"Stale annotation source hashes: {record['recording_id']}")
    indices=set();ids=set()
    for supplied in entry.get('events',[]):
        event=copy.deepcopy(supplied);index=event.get('source_event_index')
        if index is not None:
            if type(index) is not int or not 0<=index<len(events) or index in indices:raise ValueError('Invalid/duplicate source event index')
            indices.add(index)
            # Preserve the original operator interval solely as review provenance.
            event['source_action_interval_seconds']=events[index]['source_action_interval_seconds']
        validate_event(event,[record['start_seconds'],record['end_seconds']])
        if event['event_id'] in ids:raise ValueError('Duplicate physical event in one recording')
        ids.add(event['event_id'])
        if index is None:events.append(event)
        else:events[index]=event
    if len({e['event_id'] for e in events})!=len(events):raise ValueError('Duplicate physical event in one recording')
    return events


def counts_template():
    return dict(positive_windows=0,normal_windows=0,ignored_windows=0,quality_rejected_windows=0,
                label_candidate_positive_windows=0,per_event_positive_windows={},per_event_label_candidate_windows={})


def count_window(summary,logical,quality,event_id=''):
    if logical is None:summary['ignored_windows']+=1;return
    if logical==1:
        summary['label_candidate_positive_windows']+=1
        summary['per_event_label_candidate_windows'][event_id]=summary['per_event_label_candidate_windows'].get(event_id,0)+1
    if quality['reasons']:summary['quality_rejected_windows']+=1;return
    summary['positive_windows' if logical else 'normal_windows']+=1
    if logical:summary['per_event_positive_windows'][event_id]=summary['per_event_positive_windows'].get(event_id,0)+1


def build(data_root,base_dir,output,annotations=None,split_manifest=None,policy_path=None,materialize=False):
    data_root=Path(data_root).resolve();base_dir=Path(base_dir).resolve();output=Path(output).resolve()
    if output==base_dir or output.is_relative_to(base_dir) or output.is_relative_to(data_root) or base_dir.is_relative_to(output):
        raise ValueError('Output must be a separate new dataset directory, outside raw and existing artifacts')
    if output.exists() and any(output.glob('*.npz')):raise ValueError('Refusing to overwrite a materialized dataset; choose a new output version')
    base=read_json(base_dir/'preprocessing_config.json')
    if base.get('preprocessing_version')!=2 or len(base['channel_indices'])!=52:raise ValueError('Use the unchanged quality-v2 source configuration')
    policy_path=Path(policy_path or ROOT/'configs/label_policy_v3.json')
    policy=validate_policy(read_json(policy_path))
    ann=read_json(annotations) if annotations else dict(schema_version=1,time_reference='measurement_elapsed_seconds',recordings={})
    if ann.get('schema_version')!=1 or ann.get('time_reference')!='measurement_elapsed_seconds' or not isinstance(ann.get('recordings'),dict):raise ValueError('Unsupported annotation schema/time reference')
    inventory=audit(data_root)
    split_doc=read_json(split_manifest) if split_manifest else dict(schema_version=1,group_splits=base['group_splits'],excluded_groups=[])
    if split_doc.get('schema_version')!=1:raise ValueError('Unsupported split manifest')
    groups=split_doc['group_splits'];excluded=split_doc.get('excluded_groups',[])
    validate_groups(groups,excluded,inventory['groups'])
    by_id={r['recording_id']:r for r in inventory['recordings']}
    if len(by_id)!=len(inventory['recordings']):raise ValueError('Duplicate recording ID')
    if set(ann['recordings'])-set(by_id):raise ValueError('Annotations reference unknown recordings')
    config=copy.deepcopy(base)
    config.update(preprocessing_version=3,dataset_ready=False,artifact_sha256={},recordings={},quality={},mean=None,std=None,
        group_splits=groups,excluded_groups=excluded,labeling_config=policy,labeling_config_sha256=fingerprint(policy_path),
        label_policy='Reviewed descent-onset context with guaranteed motion overlap; uncertain fall windows ignored.',
        normalization='Per-channel statistics refitted on this version training windows only; no validation/test fitting',
        normalization_status='not_fitted',source_v2_config_sha256=fingerprint(base_dir/'preprocessing_config.json'))
    config['annotation_sha256']=fingerprint(Path(annotations)) if annotations else None
    template=dict(schema_version=1,time_reference='measurement_elapsed_seconds',recordings={})
    comparison={};provenance=[];accepted={s:[] for s in groups};owners={};review_queue=[];skipped=[]
    for record in inventory['recordings']:
        rid=record['recording_id'];group=record['group'];folder=data_root/record['path']
        split=next((s for s,gg in groups.items() if group in gg),None)
        if split is None:skipped.append(rid);continue
        metadata,rows,raw_status,intervals,issues,errors=load_recording(folder)
        hashes={name:fingerprint(folder/name) for name in ('measurements.jsonl','metadata.json','events.jsonl') if (folder/name).exists()}
        source_eligible=(metadata.get('status')=='complete' and metadata.get('category_is_recording_label') is True and record['sources']==1)
        if not source_eligible or metadata['activity'] not in ('staged_fall',) and raw_status!='verified_non_fall':
            skipped.append(rid);continue
        if raw_status=='verified_non_fall' and rid in ann['recordings']:raise ValueError('Fall sidecar contradicts a verified normal recording')
        events=resolve_events(record,ann['recordings'].get(rid),hashes) if metadata['activity']=='staged_fall' else []
        status='annotated_fall' if metadata['activity']=='staged_fall' else 'verified_non_fall'
        if status=='annotated_fall':
            template['recordings'][rid]=dict(source_sha256=hashes,events=copy.deepcopy(events))
            if not events:review_queue.append(dict(recording_id=rid,event_id=None,reason='No located/reviewed physical fall annotation'))
            for e in events:
                previous=owners.setdefault(e['event_id'],(split,group))
                if previous!=(split,group):raise ValueError('Physical fall event crosses split or collection-session boundaries')
                if not timing_usable(e,policy):review_queue.append(dict(recording_id=rid,event_id=e['event_id'],reason='Verified descent bounds required or uncertainty exceeds policy'))
        saved={k:record[k] for k in ('group','session','participant','activity','start_seconds','end_seconds','exposure_seconds','measured_rate_hz')}
        saved.update(split=split,label_status=status,fall_intervals=intervals,fall_events=events,measurement_sha256=hashes['measurements.jsonl'],source_sha256=hashes)
        config['recordings'][rid]=saved
        summary=comparison.setdefault(group,dict(split=split,annotated_physical_falls=0,verified_motion_events=0,
            fall_interval_durations_seconds=[],v2=counts_template(),v3=counts_template(),events=[]))
        summary['annotated_physical_falls']+=len(events);summary['verified_motion_events']+=sum(timing_usable(e,policy) for e in events)
        summary['fall_interval_durations_seconds'].extend(b-a for a,b in intervals)
        summary['events'].extend(dict(event_id=e['event_id'],recording_id=rid,status=e['status'],timing_usable=timing_usable(e,policy)) for e in events)
        times,features,_=recording_features(rows,config['channel_indices'])
        for start in np.arange(times[0],times[-1]-3+1e-9,.5):
            start=float(start);end=start+3
            old=label_window(start,end,record['label_status'],intervals)
            old_index=next((i for i,(a,b) in enumerate(intervals) if old==1 and start>=a+.25 and end<=b-.25),None)
            old_id=next((e['event_id'] for e in events if e.get('source_event_index')==old_index),f'{rid}:fall:{old_index}') if old==1 else ''
            old_quality=window_quality(times,start,config) if old is not None else dict(reasons=[])
            count_window(summary['v2'],old,old_quality,old_id)
            labeled=decision(start,end,status,events,policy)
            quality=window_quality(times,start,config) if labeled['label'] is not None else dict(reasons=[])
            count_window(summary['v3'],labeled['label'],quality,labeled['event_id'])
            item=dict(recording_id=rid,split=split,group=group,start_seconds=start,end_seconds=end,
                **labeled,quality_reasons=quality['reasons'],quality_assessed=labeled['label'] is not None)
            provenance.append(item)
            if labeled['label'] is not None and not quality['reasons']:accepted[split].append(item)
    for group,summary in comparison.items():
        summary['annotated_physical_falls']=len({e['event_id'] for e in summary['events']})
        summary['pending_review_events']=len({e['event_id'] for e in summary['events'] if not e['timing_usable']})
        n=summary['annotated_physical_falls']
        for version in ('v2','v3'):
            z=summary[version];represented=len(z['per_event_positive_windows']);candidate=len(z['per_event_label_candidate_windows'])
            z.update(represented_falls=represented,falls_without_label_candidates=n-candidate,
                percent_falls_without_label_candidates=100*(n-candidate)/n if n else None,
                falls_with_candidates_lost_to_quality=candidate-represented)
            z['positive_windows_per_annotated_event']={e['event_id']:z['per_event_positive_windows'].get(e['event_id'],0) for e in summary['events']}
        summary['newly_recovered_verified_events']=len(set(summary['v3']['per_event_positive_windows'])-set(summary['v2']['per_event_positive_windows']))
        summary['newly_recovered_verified_short_events_under_3_5_seconds']=sum(e['event_id'] in summary['v3']['per_event_positive_windows'] and e['event_id'] not in summary['v2']['per_event_positive_windows'] and
            any(v['event_id']==e['event_id'] and timing_usable(v,policy) and v['motion_end_bounds_seconds'][1]-v['onset_bounds_seconds'][0]<3.5 for v in config['recordings'][e['recording_id']]['fall_events']) for e in summary['events'])
    ready=all({w['label'] for w in windows}=={0,1} for windows in accepted.values())
    config['quality']={s:dict(windows=len(w),class_counts=[sum(x['label']==i for x in w) for i in (0,1)],
         independent_recordings=len({x['recording_id'] for x in w}),independent_sessions=len(groups[s])) for s,w in accepted.items()}
    summary=dict(preprocessing_version=3,signal_quality_version=2,dataset_ready=False,labels_have_both_classes_in_every_split=ready,
        materialization_requested=materialize,manual_review=review_queue,sessions=comparison,
        skipped_or_explicitly_excluded_recordings=skipped,
        quality_count_policy='Disjoint counts: logical ignore first; quality evaluated only on fall/normal candidates. Unreviewed falls are pending supervision, not measured model false negatives.',
        threshold_selection='Prospective context/uncertainty contract; no model scores, test-performance optimization, or inferred impact times used.')
    output.parent.mkdir(parents=True,exist_ok=True)
    if materialize and ready:
        required=sum(len(w) for w in accepted.values())*52*512*4+64*1024**2
        if shutil.disk_usage(output.parent).free<required:raise RuntimeError(f'Need approximately {required} free bytes; audit output creates no duplicate arrays')
    with tempfile.TemporaryDirectory(prefix='label-v3-',dir=output.parent) as tmp:
        stage=Path(tmp)
        if materialize and ready:
            materialize_arrays(data_root,inventory,accepted,config,stage)
            config['dataset_ready']=True;summary['dataset_ready']=True;validate_config(config)
        write_json(stage/'preprocessing_config.json',config)
        write_json(stage/'comparison.json',summary)
        write_json(stage/'annotation_review_template.json',template)
        with (stage/'window_provenance.jsonl').open('w') as stream:
            for item in provenance:stream.write(json.dumps(item,allow_nan=False)+'\n')
        manifest=dict(schema_version=1,preprocessing_version=3,dataset_ready=config['dataset_ready'],source_config_sha256=config['source_v2_config_sha256'],
            files={f.name:dict(bytes=f.stat().st_size,sha256=fingerprint(f)) for f in stage.iterdir()},
            note='Audit-only if not ready. No model may train on an unresolved or nonmaterialized manifest.')
        write_json(stage/'manifest.json',manifest)
        output.mkdir(parents=True,exist_ok=True)
        for file in stage.iterdir():file.replace(output/file.name)
    return summary


def materialize_arrays(root,inventory,accepted,config,stage):
    """Two passes: bounded per-split tensor allocation; scaler fitted on train only."""
    by_id={r['recording_id']:r for r in inventory['recordings']};total=np.zeros(52);squares=np.zeros(52);n=0
    cache=None;times=features=None
    def raw(item):
        nonlocal cache,times,features
        if item['recording_id']!=cache:
            cache=item['recording_id'];_,rows,*_=load_recording(root/by_id[cache]['path']);times,features,_=recording_features(rows,config['channel_indices'])
        return transform_window(times,features,item['start_seconds'],config,normalize=False)
    for item in accepted['train']:
        x=raw(item).astype(np.float64);total+=x.sum(axis=1);squares+=(x*x).sum(axis=1);n+=512
    mean=total/n;std=np.maximum(np.sqrt(np.maximum(squares/n-mean**2,0)),1e-6)
    config.update(mean=mean.tolist(),std=std.tolist(),normalization_status='fitted_on_train_only')
    for split,items in accepted.items():
        X=np.empty((len(items),52,512),dtype=np.float32)
        for i,item in enumerate(items):X[i]=((raw(item).astype(np.float64)-mean[:,None])/std[:,None]).astype(np.float32)
        np.savez_compressed(stage/f'{split}.npz',X=X,y=np.asarray([w['label'] for w in items],dtype=np.int64),
            recording_id=np.asarray([w['recording_id'] for w in items]),event_id=np.asarray([w['event_id'] for w in items]),
            start_seconds=np.asarray([w['start_seconds'] for w in items]),end_seconds=np.asarray([w['end_seconds'] for w in items]))
        config['artifact_sha256'][split]=fingerprint(stage/f'{split}.npz');del X


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='../blissey-data/recordings');p.add_argument('--base-dir',default='data/processed')
    p.add_argument('--output',default='data/processed_v3_audit');p.add_argument('--annotations');p.add_argument('--splits');p.add_argument('--label-policy')
    p.add_argument('--materialize',action='store_true',help='Write separate NPZ files only when every split has verified positives and negatives')
    a=p.parse_args();s=build(a.data_root,a.base_dir,a.output,a.annotations,a.splits,a.label_policy,a.materialize)
    print(json.dumps(dict(dataset_ready=s['dataset_ready'],pending_manual_review=len(s['manual_review']),output=a.output),indent=2))
    if not s['dataset_ready']:print('No training-ready NPZ dataset emitted. Complete timing review, then use --materialize with a new output directory.')


if __name__=='__main__':main()
