"""CNN-only metrics; independent of the MOMENT evaluation module."""
import numpy as np


def classification(y,pred):
    y=np.asarray(y);pred=np.asarray(pred)
    if y.ndim!=1 or y.shape!=pred.shape or not len(y):raise ValueError('Need equally sized, nonempty labels/predictions')
    if not set(y.tolist())<={0,1} or not set(pred.tolist())<={0,1}:raise ValueError('Binary labels/predictions required')
    tn=int(((y==0)&(pred==0)).sum());fp=int(((y==0)&(pred==1)).sum())
    fn=int(((y==1)&(pred==0)).sum());tp=int(((y==1)&(pred==1)).sum())
    return dict(accuracy=(tn+tp)/len(y),precision=tp/(tp+fp) if tp+fp else 0.,
        fall_recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        precision_defined=bool(tp+fp),confusion_matrix=[[tn,fp],[fn,tp]],
        class_distribution=[int((y==0).sum()),int((y==1).sum())])


def union_seconds(spans):
    total=0.;right=None
    for start,end in sorted(spans):
        total+=max(0.,end-max(start,right if right is not None else start))
        right=max(end,right if right is not None else end)
    return total


def event_metrics(dataset,pred,config,split):
    if config.get('preprocessing_version')==3:
        from labeling_v3 import event_metrics as reviewed_event_metrics
        return reviewed_event_metrics(dataset,pred,config,split)
    pred=np.asarray(pred)
    covered=detected=uncovered=alerts=0;exposure=0.;per_event=[]
    for rid,record in sorted(config['recordings'].items()):
        if record['split']!=split:continue
        indices=np.flatnonzero(dataset.recording_id==rid)
        indices=indices[np.argsort(dataset.end[indices])]
        positives=indices[pred[indices]==1]
        if record['label_status']=='verified_non_fall':
            exposure+=union_seconds(list(zip(dataset.start[indices],dataset.end[indices])))
            previous=None
            for i in positives:
                end=float(dataset.end[i])
                if previous is None or end-previous>config['stride_seconds']*1.5:alerts+=1
                previous=end
        for event_index,(start,end) in enumerate(record['fall_intervals']):
            windows=[i for i in indices if dataset.y[i]==1 and dataset.start[i]>=start and dataset.end[i]<=end]
            hit=any(pred[i]==1 for i in windows)
            if windows:covered+=1;detected+=int(hit)
            else:uncovered+=1
            per_event.append(dict(recording_id=rid,event_index=event_index,eligible_windows=len(windows),
                detected=bool(hit) if windows else None))
    all_events=covered+uncovered
    return dict(covered_event_detection_rate=detected/covered if covered else None,
        detected_events=detected,covered_events=covered,uncovered_events=uncovered,annotated_events=all_events,
        all_annotated_event_detection_fraction=detected/all_events if all_events else None,
        per_event=per_event,false_alert_episodes=alerts,normal_monitored_seconds=exposure,
        false_alerts_per_hour=alerts/(exposure/3600) if exposure else None,
        alert_policy='One episode per contiguous run of positive decisions; merge gaps <=1.5 strides; decision at window end.',
        exposure_policy='Union of evaluated windows on verified normal trials; exclude unobserved gaps.',
        impact_detection_latency_seconds=None,
        limitation='Approximate staged-action annotations; uncovered events are reported explicitly, not scored as measured model misses.')
