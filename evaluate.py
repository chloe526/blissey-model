"""Held-session evaluation; window scores are experimental and uncalibrated."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from metrics import classification_metrics
from model_utils import device_for,load_checkpoint,logits_for,write_json
from training_data import load_data


def union_duration(spans):
    total=0.;end=None
    for a,b in sorted(spans):
        total+=max(0,b-max(a,end if end is not None else a));end=max(b,end if end is not None else b)
    return total


def temporal_metrics(dataset,pred,preprocessing):
    if preprocessing.get('preprocessing_version')==3:
        from labeling_v3 import event_metrics
        result=event_metrics(dataset,pred,preprocessing,'test')
        return dict(event_detection_rate=result['covered_event_detection_rate'],
            covered_annotated_events=result['covered_events'],detected_events=result['detected_events'],
            uncovered_events=result['uncovered_events'],all_annotated_event_detection_fraction=result['all_annotated_event_detection_fraction'],
            false_positive_alerts=result['false_alert_episodes'],normal_monitored_seconds=result['normal_monitored_seconds'],
            false_positive_alerts_per_hour=result['false_alerts_per_hour'],detection_latency_seconds=None,
            detection_latency_unavailable_reason='Reviewed descent onset is not an impact timestamp.',
            per_event=result['per_event'],limitation=result['limitation'])
    normal_seconds=0.;false_alerts=0;events=0;detected=0;uncovered=0;proxy=[]
    ids={rid for rid,r in preprocessing['recordings'].items() if r['split']=='test'}
    for rid in ids:
        selected=np.flatnonzero(dataset.recording_id==rid)
        selected=selected[np.argsort(dataset.end[selected])]
        record=preprocessing['recordings'][str(rid)]
        positive=selected[np.asarray(pred)[selected]==1]
        if record['label_status']=='verified_non_fall':
            normal_seconds+=union_duration([(dataset.start[i],dataset.end[i]) for i in selected])
            # One alert per contiguous run of positive decisions, using window END.
            last=None
            for i in positive:
                t=dataset.end[i]
                if last is None or t-last>preprocessing['stride_seconds']*1.5:false_alerts+=1
                last=t
        else:
            for a,b in record['fall_intervals']:
                covered=[i for i in selected if a<=dataset.start[i] and dataset.end[i]<=b]
                if not covered:uncovered+=1;continue
                events+=1
                hits=[dataset.end[i] for i in positive if a<=dataset.start[i] and dataset.end[i]<=b]
                if hits:detected+=1;proxy.append(float(min(hits)-a))
    return dict(event_detection_rate=detected/events if events else None,
        covered_annotated_events=events,detected_events=detected,uncovered_events=uncovered,
        false_positive_alerts=false_alerts,normal_monitored_seconds=normal_seconds,
        false_positive_alerts_per_hour=false_alerts/(normal_seconds/3600) if normal_seconds else None,
        detection_latency_seconds=None,
        detection_latency_unavailable_reason='No reference impact times; operator action intervals cannot measure impact latency.',
        action_start_to_first_alert_seconds=proxy,
        mean_action_start_to_first_alert_seconds=float(np.mean(proxy)) if proxy else None,
        alert_policy='Merge consecutive positive decisions separated by <=1.5 strides; decisions at window end.',
        exposure_policy='Union of evaluated input windows from verified normal trials only; excluded packet gaps are not monitored exposure.',
        limitation='Event rate covers eligible windows inside approximate action intervals only; partial/boundary windows excluded.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--device',default='auto');p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--output',default='reports/evaluation.json');p.add_argument('--allow-diagnostic',action='store_true');a=p.parse_args()
    device=device_for(a.device);model,ckpt=load_checkpoint(a.checkpoint,device)
    if ckpt['diagnostic'] and not a.allow_diagnostic:raise ValueError('One-batch diagnostic checkpoint: pass --allow-diagnostic for integration testing only')
    data,pre=load_data(a.data_dir,splits=('test',))
    if pre!=ckpt['preprocessing']:raise ValueError('Evaluation preprocessing differs from checkpoint')
    probabilities=[]
    with torch.no_grad():
        for x,y in DataLoader(data['test'],batch_size=a.batch_size):
            probabilities.extend(logits_for(model,x.to(device),ckpt['training_config']['reduction']).softmax(-1)[:,1].cpu().tolist())
    pred=(np.asarray(probabilities)>=0.5).astype(int)
    result=dict(**classification_metrics(data['test'].y,pred),**temporal_metrics(data['test'],pred,pre),
        independent_test_recordings=len(set(data['test'].recording_id)),
        independent_test_sessions=len(pre['group_splits']['test']),threshold=0.5,diagnostic=ckpt['diagnostic'],
        score='Uncalibrated softmax(class 1), not a clinical probability',
        limitations=['Staged falls, two participants; one held-out session. No population/clinical generalization claim.'])
    write_json(a.output,result)
    summary=f"Accuracy {result['accuracy']:.4f}; precision {result['precision']:.4f}; fall recall {result['recall_fall']}; F1 {result['f1']:.4f}\nConfusion matrix [true rows, predicted columns]: {result['confusion_matrix']}\nIndependent test recordings: {result['independent_test_recordings']}; sessions: {result['independent_test_sessions']}\nEvent detection rate: {result['event_detection_rate']}; false alerts/hour: {result['false_positive_alerts_per_hour']}\nImpact latency: not estimable. Scores are experimental. Diagnostic: {result['diagnostic']}\n"
    Path(a.output).with_suffix('.txt').write_text(summary);print(summary)
    # Predictions retain only trial IDs/times, not source MACs or raw samples.
    np.savez_compressed(Path(a.output).with_name(Path(a.output).stem+'_predictions.npz'),
        probability=np.asarray(probabilities),prediction=pred,y=data['test'].y,
        recording_id=data['test'].recording_id,start_seconds=data['test'].start,end_seconds=data['test'].end)

if __name__=='__main__':main()
