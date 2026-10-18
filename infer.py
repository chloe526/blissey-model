"""Reload trained weights and transform one previously unseen raw CSI window."""
import argparse
import json
import torch
from csi import load_recording
from preprocess import recording_features,transform_window
from model_utils import load_checkpoint,device_for,logits_for


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--recording',required=True)
    p.add_argument('--start-seconds',type=float,required=True);p.add_argument('--device',default='auto');a=p.parse_args()
    device=device_for(a.device);model,ckpt=load_checkpoint(a.checkpoint,device);pre=ckpt['preprocessing']
    metadata,rows,status,intervals,issues,errors=load_recording(a.recording)
    if len({p['source_mac'] for p in rows})!=1:raise ValueError('Mixed radio sources are unsupported')
    rid=metadata['recording_id']
    if pre['recordings'].get(rid,{}).get('split')=='train':raise ValueError('Inference acceptance test requires an unseen recording')
    times,features,duplicates=recording_features(rows,pre['channel_indices'])
    window=transform_window(times,features,a.start_seconds,pre)
    with torch.no_grad():
        logits=logits_for(model,torch.from_numpy(window).unsqueeze(0).to(device),ckpt['training_config']['reduction'])
        probability=logits.softmax(-1)[0,1].item()
    print(json.dumps(dict(recording_id=rid,input_shape=list(window.shape),logits=logits.cpu().tolist(),
        experimental_fall_score=probability,predicted_class=int(probability>=0.5),
        diagnostic_checkpoint=ckpt['diagnostic'],malformed_packets=len(errors),duplicates=duplicates,
        note='Experimental uncalibrated score; no QNX runtime/export implied'),indent=2))

if __name__=='__main__':main()
