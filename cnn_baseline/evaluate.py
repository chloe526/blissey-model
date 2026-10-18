"""Evaluate a frozen CNN checkpoint at its saved threshold; no threshold tuning."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import load_splits,sha256
from .metrics import classification,event_metrics
from .model import weighted_cross_entropy
from .runtime import setup,load_cnn,write_json,benchmark

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/processed')
    p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--output',type=Path)
    a=p.parse_args()
    # Reading the checkpoint does not invoke any pretrained loader or network.
    raw=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if raw.get('model_type')!='blissey_cnn_1d_v1':raise ValueError('Expected independent CNN checkpoint')
    config=raw['config'];device=setup(config['seed'],config['threads'],a.device)
    model,checkpoint=load_cnn(a.checkpoint,device);del raw
    data,pre=load_splits(a.data_dir,(a.split,));dataset=data[a.split]
    if pre!=checkpoint['preprocessing']:raise ValueError('Dataset/preprocessing differs from saved CNN checkpoint')
    output=a.output or BASE/'reports'/a.checkpoint.parent.name/f'{a.split}_evaluation.json'
    probabilities=[];loss_sum=0.;started=time.perf_counter()
    with torch.inference_mode():
        for x,y in DataLoader(dataset,batch_size=config['batch_size'],shuffle=False,num_workers=0):
            logits=model(x.to(device))
            if not torch.isfinite(logits).all():raise ValueError('Non-finite CNN predictions')
            loss_sum+=float(weighted_cross_entropy(logits,y.to(device)))*len(y)
            probabilities.extend(logits.softmax(-1)[:,1].cpu().tolist())
    evaluation_seconds=time.perf_counter()-started
    probability=np.asarray(probabilities);pred=(probability>=config['decision_threshold']).astype(np.int64)
    latency=benchmark(model,torch.from_numpy(dataset.X[:1]).to(device),device)
    metrics=classification(dataset.y,pred);events=event_metrics(dataset,pred,pre,a.split)
    result=dict(model_type=checkpoint['model_type'],split=a.split,decision_threshold=config['decision_threshold'],
        threshold_source='Saved predeclared training config; no test tuning',best_epoch=checkpoint['epoch'],
        unweighted_loss=loss_sum/len(dataset),**metrics,events=events,
        contributing_recordings=len(set(dataset.recording_id)),collection_sessions=len(pre['group_splits'][a.split]),
        parameter_count=sum(p.numel() for p in model.parameters()),checkpoint_bytes=a.checkpoint.stat().st_size,
        checkpoint_sha256=sha256(a.checkpoint),inference_latency=latency,
        training_duration_seconds=checkpoint['training_duration_seconds'],evaluation_seconds=evaluation_seconds,
        dataset_sha256=pre['artifact_sha256'][a.split],
        limitations=['Staged-fall proof of concept; one collection session in this split.',
            'Overlapping windows are correlated; tiny event count cannot establish generalization.',
            'MOMENT has not been benchmarked here; this is a separate random-initialized CNN.'])
    write_json(output,result)
    np.savez_compressed(output.with_name(output.stem+'_predictions.npz'),
        probability=probability,prediction=pred,y=dataset.y,recording_id=dataset.recording_id,
        start_seconds=dataset.start,end_seconds=dataset.end)
    text=(f"CNN baseline ({a.split}), checkpoint epoch {checkpoint['epoch']}, threshold {config['decision_threshold']}\n"
        f"Fall recall {metrics['fall_recall']:.4f}; precision {metrics['precision']:.4f}; F1 {metrics['f1']:.4f}\n"
        f"Confusion matrix [true rows, predicted columns]: {metrics['confusion_matrix']}\n"
        f"Events: {events['detected_events']}/{events['covered_events']} covered; {events['uncovered_events']} uncovered of {events['annotated_events']} annotated\n"
        f"Parameters {result['parameter_count']}; checkpoint {result['checkpoint_bytes']} bytes; training {result['training_duration_seconds']:.2f}s\n"
        f"{device} B=1 latency median {latency['median_ms']:.3f}ms, p95 {latency['p95_ms']:.3f}ms ({latency['threads']} threads)\n")
    output.with_suffix('.txt').write_text(text);print(text)

if __name__=='__main__':main()
