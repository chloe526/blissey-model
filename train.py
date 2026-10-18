"""Custom PyTorch head fine-tuning; safe locally, CUDA via Veri run-script."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from metrics import classification_metrics
from model_utils import device_for,load_pretrained,load_checkpoint,logits_for,seed_all,training_mode,write_json
from training_data import load_data


class SampleWeightedCrossEntropy(torch.nn.Module):
    """Mean over examples, so class weights also work for one-example batches.

    CrossEntropyLoss(weight=..., reduction='mean') divides by the sum of
    target weights, which cancels weighting entirely when batch_size == 1.
    """
    def __init__(self, weight=None):
        super().__init__()
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        return torch.nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction='none').mean()


def run_epoch(model,loader,criterion,device,reduction,optimizer=None,max_batches=None,clip=1.0):
    if optimizer:training_mode(model)
    else:model.eval()
    total=0.;seen=0;ys=[];predictions=[];steps=0;first_logits=None;first_probabilities=None
    for batch,(x,y) in enumerate(loader):
        if max_batches and batch>=max_batches:break
        x=x.to(device);y=y.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            if optimizer:optimizer.zero_grad(set_to_none=True)
            logits=logits_for(model,x,reduction);loss=criterion(logits,y)
            probabilities=logits.detach().softmax(-1)
            if not torch.isfinite(probabilities).all():raise ValueError('Non-finite softmax probabilities')
            if first_logits is None:
                first_logits=logits.detach().cpu().tolist();first_probabilities=probabilities.cpu().tolist()
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            if optimizer:
                loss.backward()
                trainable=[p for p in model.parameters() if p.requires_grad]
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):raise ValueError('Missing/non-finite trainable gradient')
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen parameters received gradients')
                if not any(bool(torch.any(p.grad!=0)) for p in trainable):raise ValueError('All trainable gradients are zero')
                before=model.head.linear.weight.detach().clone() if max_batches else None
                torch.nn.utils.clip_grad_norm_(trainable,clip,error_if_nonfinite=True);optimizer.step();steps+=1
                if before is not None and torch.equal(before,model.head.linear.weight):raise ValueError('Diagnostic step did not update classification head')
        total+=loss.item()*len(y);seen+=len(y);ys.extend(y.cpu().tolist());predictions.extend(logits.argmax(-1).detach().cpu().tolist())
    if not seen:raise ValueError('Empty epoch')
    return dict(loss=total/seen,**classification_metrics(ys,predictions),windows=seen,optimizer_steps=steps,first_logits=first_logits,first_probabilities=first_probabilities)


def log_veri(epoch,validation):
    try:import veri_sdk
    except ImportError:
        if os.environ.get('VERI_METRICS_FILE'):raise RuntimeError('Veri job requires veri-sdk dependency')
        return
    veri_sdk.log_metrics(step=epoch,reward=validation['f1'],loss=validation['loss'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/training.json');p.add_argument('--data-dir',default=os.environ.get('VERI_DATA_DIR') or 'data/processed')
    p.add_argument('--epochs',type=int);p.add_argument('--max-batches',type=int)
    p.add_argument('--batch-size',type=int);p.add_argument('--device',default='auto')
    p.add_argument('--smoke-data',action='store_true',help='Permit tiny diagnostic train/val fixture; requires exactly one optimizer step')
    p.add_argument('--unfreeze-last-blocks',type=int);a=p.parse_args()
    config=json.loads(Path(a.config).read_text())
    for key in ('epochs','batch_size','unfreeze_last_blocks'):
        if getattr(a,key) is not None:config[key]=getattr(a,key)
    if config['epochs']<1 or config['batch_size']<1 or (a.max_batches is not None and a.max_batches<1):p.error('epochs, batch-size and max-batches must be positive')
    if a.smoke_data and (config['epochs']!=1 or config['batch_size']!=1 or a.max_batches!=1):
        p.error('--smoke-data requires --epochs 1 --max-batches 1 --batch-size 1')
    seed_all(config['seed']);device=device_for(a.device)
    if os.environ.get('VERI_OUTPUT_DIR') and device.type!='cuda':raise RuntimeError('Veri training requires a working CUDA device')
    output_dir=Path(os.environ.get('VERI_OUTPUT_DIR','./checkpoints'));output_dir.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output_dir).free<1_600_000_000:raise RuntimeError('Need >=1.6 GB free in output directory for full trained checkpoint')
    # Full training also keeps test arrays off the GPU and out of model selection.
    data,preprocessing=load_data(a.data_dir,splits=('train','val'))
    if bool(preprocessing.get('diagnostic_subset',False))!=a.smoke_data:
        raise ValueError('Diagnostic subset requires --smoke-data; full training requires complete data')
    for split in ('train','val'):
        if set(data[split].y)!={0,1}:raise ValueError(f'{split} must contain both classes')
    counts=np.bincount(data['train'].y,minlength=2);print('Training class counts:',counts.tolist(),flush=True)
    print('Pretrained model:',config['model_id'],'revision:',config['model_revision'],flush=True)
    model=load_pretrained(len(preprocessing['channel_indices']),config).to(device)
    weights=torch.tensor(len(data['train'])/(2*counts),dtype=torch.float32,device=device) if config['class_weighting'] else None
    class_weights=weights.detach().cpu().tolist() if weights is not None else None
    print('Training class weights:',class_weights,flush=True)
    criterion=SampleWeightedCrossEntropy(weight=weights)
    # Validation loss is unweighted and comparable between experiments.
    validation_criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    generator=torch.Generator().manual_seed(config['seed'])
    loaders={s:DataLoader(data[s],batch_size=config['batch_size'],shuffle=s=='train',num_workers=0,generator=generator) for s in ('train','val')}
    history=[];best=(-1.,float('-inf'));stale=0;started=time.monotonic()
    versions={package:importlib.metadata.version(package) for package in ('torch','numpy','momentfm','transformers','huggingface-hub')}
    write_json(output_dir/'run_config.json',dict(training=config,preprocessing=preprocessing,versions=versions,device=str(device),diagnostic=a.max_batches is not None,
        class_weights=class_weights,training_loss='Mean of per-example class-weighted cross entropy; weights fitted on train only',
        validation_loss='Unweighted cross entropy over validation windows',reward_meaning='Validation fall-class F1'))
    for epoch in range(1,config['epochs']+1):
        t=time.monotonic()
        training=run_epoch(model,loaders['train'],criterion,device,config['reduction'],optimizer,a.max_batches,config['gradient_clip'])
        validation=run_epoch(model,loaders['val'],validation_criterion,device,config['reduction'],max_batches=a.max_batches)
        record=dict(epoch=epoch,train=training,validation=validation,seconds=time.monotonic()-t,reward_meaning='validation fall-class F1',diagnostic=a.max_batches is not None)
        history.append(record);print(json.dumps(record),flush=True);log_veri(epoch,validation)
        write_json(output_dir/'history.json',history)
        score=(validation['f1'],-validation['loss'])
        if score>best:
            best=score;stale=0
            checkpoint=dict(schema_version=1,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                model_config=vars(model.config),training_config=config,preprocessing=preprocessing,
                epoch=epoch,validation=validation,versions=versions,diagnostic=a.max_batches is not None,
                classes=['normal','possible_fall'],score_description='Uncalibrated softmax probability of class 1')
            torch.save(checkpoint,output_dir/'best.pt')
        else:stale+=1
        if stale>=config['patience']:break
    # Actually reload full weights without contacting HF and compare a forward pass.
    model.eval();x=data['val'][0][0].unsqueeze(0).to(device)
    del optimizer
    # Compare against best state, which can differ from the last epoch.
    best_state=torch.load(output_dir/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(best_state['state_dict']);del best_state
    with torch.no_grad():expected=logits_for(model,x,config['reduction']).cpu()
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    restored,metadata=load_checkpoint(output_dir/'best.pt',device)
    with torch.no_grad():actual=logits_for(restored,x,config['reduction']).cpu()
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
    write_json(output_dir/'reload_check.json',dict(passed=True,logits=actual.tolist(),elapsed_seconds=time.monotonic()-started,diagnostic=a.max_batches is not None))
    print('Checkpoint reload/forward verified:',output_dir/'best.pt',flush=True)

if __name__=='__main__':main()
