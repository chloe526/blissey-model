"""Stage-1-only LA Hacks quality-v2 audit. No fitting, relabeling or NPZ writes."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

MODEL_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(MODEL_ROOT))
from csi import load_recording
from preprocess import label_window,recording_features,transform_window
from csi_quality import window_quality

GROUPS=('session_03_LAHACKS','session_04_LAHACKS')
DYNAMIC_NORMAL={'walking','stand_up','sit_down','lie_down','get_up_from_floor'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def quant(values):
    if not values:return None
    return dict(zip(('min','p25','median','p75','max'),map(float,np.quantile(values,[0,.25,.5,.75,1]))))
def union(spans):
    total=0.;right=None
    for a,b in sorted(spans):
        total+=max(0.,b-max(a,a if right is None else right));right=max(b,b if right is None else right)
    return total
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(root,revision,out):
    root=Path(root).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    base=json.loads((MODEL_ROOT/'data/processed/preprocessing_config.json').read_text())
    # Geometry/physical carriers/acquisition gates only. No Cal Poly scaler or X.
    cfg={k:base[k] for k in ('window_seconds','stride_seconds','sequence_length','channel_indices','packet_quality','max_gap_seconds')}
    assert len(cfg['channel_indices'])==52 and cfg['sequence_length']==512
    assert cfg['packet_quality']['version']==2
    records=[];windows=[];source_files=[]
    for group in GROUPS:
        folder=root/group
        if not folder.is_dir():raise ValueError('Required LA Hacks block missing: '+group)
        for metadata_path in sorted(folder.rglob('metadata.json')):
            m,rows,status,intervals,issues,errors=load_recording(metadata_path.parent)
            rid=m['recording_id'];source_count=len({p.get('source_mac') for p in rows})
            if source_count!=1:status='ambiguous';issues.append('Multiple/missing radio sources')
            for name in ('metadata.json','events.jsonl','measurements.jsonl'):
                p=metadata_path.with_name(name)
                if p.exists():source_files.append(dict(path=f"{group}/{metadata_path.parent.relative_to(folder)}/{name}",bytes=p.stat().st_size,sha256=sha(p)))
            r=dict(recording_id=rid,group=group,participant=m.get('participant'),activity=m['activity'],
                metadata_session=m.get('session'),label_status=status,source_count=source_count,
                valid_packets=len(rows),invalid_packets=len(errors),issues=issues,
                created_at=m.get('created_at'),started_at=m.get('started_at'),ended_at=m.get('ended_at'),
                all_candidate_windows=0,normal_windows=0,positive_windows=0,ignored_windows=0,quality_rejected_windows=0,
                quality_rejection_reasons={},normal_accepted_union_seconds=0.,fall_intervals=intervals,
                fall_interval_durations_seconds=[b-a for a,b in intervals],
                events=[dict(event_id=f'{rid}:fall:{i}',interval_seconds=[a,b],v2_positive_candidates=0,
                    v2_positive_windows=0,quality_pass_windows_overlapping_action=0) for i,(a,b) in enumerate(intervals)])
            if len(rows)<2:
                r['label_status']='ambiguous';records.append(r);continue
            t,x,dupes=recording_features(rows,cfg['channel_indices']);duration=float(t[-1]-t[0]);gaps=np.diff(t)
            r.update(duration_seconds=duration,effective_rate_hz=(len(t)-1)/duration,
                duplicate_host_timestamps=dupes,max_gap_seconds=float(gaps.max()),
                gaps_over_250ms=int((gaps>.25).sum()))
            normal_spans=[];reasons=Counter()
            for start in np.arange(t[0],t[-1]-3+1e-9,.5):
                start=float(start);end=start+3;r['all_candidate_windows']+=1
                quality=window_quality(t,start,cfg)
                signal_ok=not quality['reasons']
                if signal_ok:
                    # Exercise the actual raw -> [52,512] path; retain no tensor.
                    transformed=transform_window(t,x,start,cfg,normalize=False)
                    assert transformed.shape==(52,512) and transformed.dtype==np.float32
                label=label_window(start,end,status,intervals)
                event_id=''
                for event in r['events']:
                    a,b=event['interval_seconds']
                    if signal_ok and start<b and end>a:event['quality_pass_windows_overlapping_action']+=1
                    if label==1 and start>=a+.25 and end<=b-.25:
                        event['v2_positive_candidates']+=1;event['v2_positive_windows']+=int(signal_ok);event_id=event['event_id']
                outcome='ignore'
                if label is None:r['ignored_windows']+=1
                elif not signal_ok:
                    outcome='quality_rejected';r['quality_rejected_windows']+=1;reasons.update(quality['reasons'])
                elif label==1:outcome='fall';r['positive_windows']+=1
                else:outcome='normal';r['normal_windows']+=1;normal_spans.append((start,end))
                windows.append(dict(recording_id=rid,group=group,start_seconds=start,end_seconds=end,
                    outcome=outcome,label=label,event_id=event_id,quality_reasons=quality['reasons']))
            r['quality_rejection_reasons']=dict(reasons);r['normal_accepted_union_seconds']=union(normal_spans)
            assert r['all_candidate_windows']==sum(r[k] for k in ('normal_windows','positive_windows','ignored_windows','quality_rejected_windows'))
            records.append(r)
        print('Audited block:',group,flush=True)
    assert len({r['recording_id'] for r in records})==len(records),'Recording ID collision'
    blocks={}
    for group in GROUPS:
        rr=[r for r in records if r['group']==group];normal=[r for r in rr if r['label_status']=='verified_non_fall'];events=[e for r in rr for e in r['events']]
        s=dict(recordings=len(rr),participants=dict(Counter(r['participant'] for r in rr)),metadata_sessions=dict(Counter(r['metadata_session'] for r in rr)),
            activity_recordings=dict(Counter(r['activity'] for r in rr)),fall_recordings=sum(r['activity']=='staged_fall' for r in rr),
            annotated_action_events=len(events),v2_label_candidate_fall_events=sum(e['v2_positive_candidates']>0 for e in events),
            quality_covered_fall_events=sum(e['v2_positive_windows']>0 for e in events),
            signal_covered_action_events_any_overlap=sum(e['quality_pass_windows_overlapping_action']>0 for e in events),
            normal_recordings=len(normal),normal_contributing_recordings=sum(r['normal_windows']>0 for r in normal),
            dynamic_hard_negative_recordings=sum(r['activity'] in DYNAMIC_NORMAL for r in normal),
            dynamic_hard_negative_contributing_recordings=sum(r['activity'] in DYNAMIC_NORMAL and r['normal_windows']>0 for r in normal),
            ambiguous_recordings=[dict(recording_id=r['recording_id'],activity=r['activity'],issues=r['issues']) for r in rr if r['label_status']=='ambiguous'],
            invalid_packets=sum(r['invalid_packets'] for r in rr),fall_interval_durations_seconds=quant([d for r in rr for d in r['fall_interval_durations_seconds']]),
            effective_packet_rate_hz=quant([r['effective_rate_hz'] for r in rr if 'effective_rate_hz' in r]),
            monitored_normal_seconds=sum(r['normal_accepted_union_seconds'] for r in rr),
            source_recording_seconds=sum(r.get('duration_seconds',0) for r in rr),
            events=events,
            activity_windows={a:dict(normal=sum(r['normal_windows'] for r in rr if r['activity']==a),fall=sum(r['positive_windows'] for r in rr if r['activity']==a)) for a in sorted({r['activity'] for r in rr})})
        for key in ('all_candidate_windows','normal_windows','positive_windows','ignored_windows','quality_rejected_windows'):s[key]=sum(r[key] for r in rr)
        s['quality_rejection_fraction_of_label_candidates']=s['quality_rejected_windows']/max(1,s['normal_windows']+s['positive_windows']+s['quality_rejected_windows'])
        s['fall_events_lost_to_label_geometry']=sum(e['v2_positive_candidates']==0 for e in events)
        s['fall_events_lost_to_quality']=sum(e['v2_positive_candidates']>0 and e['v2_positive_windows']==0 for e in events)
        s['class_counts_normal_fall']=[s['normal_windows'],s['positive_windows']]
        blocks[group]=s
    result=dict(stage=1,repository='chloe526/blissey-data',revision=revision,
        audited_utc=datetime.now(timezone.utc).isoformat(),lahacks_groups=list(GROUPS),
        physical_blocks_confirmed_by_user=True,participant_count=len({r['participant'] for r in records}),blocks=blocks,recordings=records,
        definitions=dict(quality_covered_fall_event='At least one positive window under unchanged quality-v2 labels and gates; an operator action event is not a verified impact.',
            dynamic_hard_negative='Verified normal walking/stand_up/sit_down/lie_down/get_up_from_floor recordings; static negatives counted separately within all normals.',
            quality_rejected='Disjoint count after v2 labeling; ignored windows are never negatives. No threshold changed.'),
        audit_transform=cfg,normalization_fitted=False,calpoly_recordings_read=False,training_performed=False,
        preserved_original_v2_config_sha256=sha(MODEL_ROOT/'data/processed/preprocessing_config.json'))
    write(out/'audit.json',result)
    write(out/'source_manifest.json',dict(revision=revision,files=source_files,source_file_count=len(source_files),bytes=sum(p['bytes'] for p in source_files),
        reused_local_block='session_03_LAHACKS',downloaded_temporary_block='session_04_LAHACKS',raw_serial_downloaded=False))
    with (out/'window_audit.jsonl').open('w') as stream:
        for w in windows:stream.write(json.dumps(w)+'\n')
    print(json.dumps({g:{k:s[k] for k in ('recordings','participants','fall_recordings','annotated_action_events','quality_covered_fall_events','normal_recordings','normal_contributing_recordings','dynamic_hard_negative_recordings','dynamic_hard_negative_contributing_recordings','class_counts_normal_fall','ignored_windows','quality_rejected_windows','ambiguous_recordings','fall_interval_durations_seconds','effective_packet_rate_hz','monitored_normal_seconds','fall_events_lost_to_label_geometry','fall_events_lost_to_quality')} for g,s in blocks.items()},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--output',default=str(Path(__file__).parent));a=p.parse_args();run(a.root,a.revision,a.output)
