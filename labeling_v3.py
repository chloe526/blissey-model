"""Conservative reviewed-event labels. No model imports, impact guesses or scores."""
import math
import re


def validate_policy(policy):
    if policy.get('version')!=3 or policy.get('strategy')!='verified_descent_onset_on_streaming_grid':raise ValueError('Unsupported label policy')
    keys=('min_seconds_after_latest_onset','max_seconds_after_earliest_onset',
          'min_guaranteed_motion_overlap_seconds','max_seconds_after_earliest_motion_end','max_boundary_uncertainty_seconds')
    if any(type(policy.get(k)) not in (int,float) or not math.isfinite(policy[k]) or policy[k]<0 for k in keys):raise ValueError('Invalid label thresholds')
    if policy.get('window_seconds')!=3 or policy.get('stride_seconds')!=.5:raise ValueError('Keep 3-second windows and 0.5-second stride')
    if not 0<policy[keys[2]]<=policy[keys[0]]<policy[keys[1]]<3:raise ValueError('Invalid onset context/overlap limits')
    if policy[keys[3]]>=1.5 or policy[keys[4]]>=policy[keys[1]]-policy[keys[0]]:raise ValueError('Excessive stationary context or uncertainty')
    return policy


def bounds(value,name):
    if not isinstance(value,list) or len(value)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in value) or value[0]>value[1]:
        raise ValueError(f'{name}: need finite [earliest, latest] seconds')
    return value


def validate_event(event,exposure):
    if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',event.get('event_id','')):raise ValueError('Invalid physical event_id')
    if event.get('status') not in ('needs_review','verified'):raise ValueError('Invalid event review status')
    if event['status']=='needs_review':
        if event.get('onset_bounds_seconds') is not None or event.get('motion_end_bounds_seconds') is not None:
            raise ValueError('Unreviewed events must not carry guessed motion boundaries')
        return event
    if event.get('anchor_type')!='descent_onset':raise ValueError('Need a reviewed descent onset; impact/action midpoint is not assumed')
    onset=bounds(event.get('onset_bounds_seconds'),'onset');end=bounds(event.get('motion_end_bounds_seconds'),'motion end')
    if not exposure[0]<=onset[0]<=onset[1]<end[0]<=end[1]<=exposure[1]:raise ValueError('Motion bounds reversed/overlapping or outside recorded exposure')
    review=event.get('review',{})
    if review.get('method') not in ('synchronized_video_review','direct_observer_review') or any(not isinstance(review.get(k),str) or not review[k].strip() for k in ('reviewer','reviewed_at','notes')):
        raise ValueError('Verified event needs reviewer, review method, timestamp and evidence notes')
    return event


def timing_usable(event,policy):
    return event.get('status')=='verified' and all(event[k][1]-event[k][0]<=policy['max_boundary_uncertainty_seconds']+1e-9 for k in ('onset_bounds_seconds','motion_end_bounds_seconds'))


def decision(start,end,status,events,policy):
    """Return logical fall/normal/ignore plus the real reviewed physical event ID."""
    if not math.isfinite(start) or not math.isfinite(end) or abs(end-start-3)>1e-8:raise ValueError('Expected a finite 3-second window')
    if status=='verified_non_fall':return dict(outcome='normal',label=0,event_id='',reason='verified_normal_recording')
    if status!='annotated_fall':return dict(outcome='ignore',label=None,event_id='',reason='unverified_recording')
    intersect=[];candidates=[]
    for event in events:
        if event.get('status')=='verified':a,b=event['onset_bounds_seconds'][0],event['motion_end_bounds_seconds'][1]
        else:
            span=event.get('source_action_interval_seconds')
            if span is None:return dict(outcome='ignore',label=None,event_id='',reason='unlocated_unreviewed_event')
            a,b=span
        if start<b and end>a:intersect.append(event)
        if not timing_usable(event,policy):continue
        onset=event['onset_bounds_seconds'];finish=event['motion_end_bounds_seconds']
        overlap=max(0.,min(end,finish[0])-max(start,onset[1]))
        if (start<=onset[0] and end>=onset[1]+policy['min_seconds_after_latest_onset']-1e-9
            and end<=onset[0]+policy['max_seconds_after_earliest_onset']+1e-9
            and end<=finish[0]+policy['max_seconds_after_earliest_motion_end']+1e-9
            and overlap+1e-9>=policy['min_guaranteed_motion_overlap_seconds']):candidates.append(event)
    if len(candidates)==1 and len(intersect)==1:
        return dict(outcome='fall',label=1,event_id=candidates[0]['event_id'],reason='verified_descent_onset_context')
    reason='multiple_or_ambiguous_events' if len(intersect)>1 else 'no_confident_onset_context'
    return dict(outcome='ignore',label=None,event_id='',reason=reason)


def validate_groups(groups,excluded=(),observed=None):
    if set(groups)!= {'train','val','test'} or any(not isinstance(v,list) or not v or len(v)!=len(set(v)) for v in groups.values()):raise ValueError('Explicit nonempty train/val/test collection groups required')
    sets=[set(v) for v in groups.values()]
    if any(a&b for i,a in enumerate(sets) for b in sets[i+1:]) or set.union(*sets)&set(excluded):raise ValueError('Collection-session leakage')
    if observed is not None and set.union(*sets)|set(excluded)!=set(observed):raise ValueError('Observed groups differ from explicit split/exclusion manifest')


def validate_config(config):
    validate_groups(config['group_splits'])
    if config.get('preprocessing_version')!=3:return
    if not config.get('dataset_ready'):raise ValueError('quality-v3 is not training-ready: review fall motion timing and materialize a new dataset')
    validate_policy(config['labeling_config'])
    owners={}
    for rid,record in config['recordings'].items():
        split=record['split']
        if record['group'] not in config['group_splits'][split]:raise ValueError('Incorrect group assignment')
        for event in record.get('fall_events',[]):
            validate_event(event,[record['start_seconds'],record['end_seconds']])
            previous=owners.setdefault(event['event_id'],(split,record['group']))
            if previous!=(split,record['group']):raise ValueError('Physical fall event crosses split or collection-session boundaries')


def validate_provenance(config,dataset,split):
    if config.get('preprocessing_version')!=3:return
    import numpy as np
    if dataset.event_id is None or len(dataset.event_id)!=len(dataset.y):raise ValueError('v3 requires per-window physical event provenance')
    for i,rid in enumerate(dataset.recording_id):
        record=config['recordings'][str(rid)]
        if record['split']!=split:raise ValueError('Recording leakage')
        expected=decision(float(dataset.start[i]),float(dataset.end[i]),record['label_status'],record.get('fall_events',[]),config['labeling_config'])
        if expected['label']!=int(dataset.y[i]) or expected['event_id']!=str(dataset.event_id[i]):raise ValueError('Window label/event provenance differs from reviewed timing')
    if np.any((dataset.y==0)&(dataset.event_id!='')):raise ValueError('Normal window cannot claim a fall event')


def event_metrics(dataset,pred,config,split):
    """Event hits use recorded IDs, not old full-interval containment semantics."""
    import numpy as np
    validate_provenance(config,dataset,split)
    pred=np.asarray(pred)
    if pred.shape!=dataset.y.shape or not set(pred.tolist())<={0,1}:raise ValueError('Invalid event predictions')
    grouped={};alerts=0;exposure=0.
    for rid,record in config['recordings'].items():
        if record['split']!=split:continue
        indices=np.flatnonzero(dataset.recording_id==rid);indices=indices[np.argsort(dataset.end[indices])]
        if record['label_status']=='verified_non_fall':
            right=None;previous=None
            for i in indices:
                a,b=float(dataset.start[i]),float(dataset.end[i]);exposure+=max(0.,b-max(a,a if right is None else right));right=max(b,b if right is None else right)
                if pred[i]:
                    if previous is None or b-previous>config['stride_seconds']*1.5:alerts+=1
                    previous=b
        for event in record.get('fall_events',[]):
            if not timing_usable(event,config['labeling_config']):continue
            eligible=indices[(dataset.y[indices]==1)&(dataset.event_id[indices]==event['event_id'])]
            item=grouped.setdefault(event['event_id'],dict(event_id=event['event_id'],eligible_windows=0,detected=False))
            item['eligible_windows']+=len(eligible);item['detected']|=bool(np.any(pred[eligible]==1))
    covered=sum(v['eligible_windows']>0 for v in grouped.values());detected=sum(v['detected'] for v in grouped.values());total=len(grouped)
    return dict(covered_event_detection_rate=detected/covered if covered else None,detected_events=detected,
        covered_events=covered,uncovered_events=total-covered,annotated_events=total,
        all_annotated_event_detection_fraction=detected/total if total else None,per_event=list(grouped.values()),
        false_alert_episodes=alerts,normal_monitored_seconds=exposure,false_alerts_per_hour=alerts/(exposure/3600) if exposure else None,
        impact_detection_latency_seconds=None,limitation='Reviewed descent-onset supervision; no impact times inferred. Unreviewed events are not verified evaluation ground truth.')
