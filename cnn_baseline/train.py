"""Local, random-initialized CNN baseline. Imports no MOMENT/Veri modules."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import load_splits,sha256
from .metrics import classification,event_metrics
from .model import MODEL_TYPE,build_model,weighted_cross_entropy
from .runtime import setup,write_json,load_cnn

ROOT=Path(__file__).resolve().parents[1]
BASE=Path(__file__).resolve().parent


def epoch(model,loader,device,weights=None,optimizer=None,clip=5.,threshold=.5):
    model.train(optimizer is not None);loss_sum=0.;labels=[];predictions=[]
    for x,y in loader:
        x=x.to(device);y=y.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            if not torch.isfinite(logits).all():raise ValueError('Non-finite CNN logits')
            loss=weighted_cross_entropy(logits,y,weights)
            if not torch.isfinite(loss):raise ValueError('Non-finite CNN loss')
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
                optimizer.step()
        loss_sum+=float(loss.detach())*len(y)
        labels.extend(y.cpu().tolist());predictions.extend((logits.detach().softmax(-1)[:,1]>=threshold).long().cpu().tolist())
    if not labels:raise ValueError('Empty loader')
    return dict(loss=loss_sum/len(labels),windows=len(labels),**classification(labels,predictions))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/processed')
    p.add_argument('--config',type=Path,default=BASE/'config.json')
    p.add_argument('--run-name',default='seed42');p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int);a=p.parse_args()
    if not a.run_name or '/' in a.run_name or '\\' in a.run_name or a.run_name in ('.','..'):p.error('Use a simple run name')
    config=json.loads(a.config.read_text())
    for key in ('epochs','batch_size'):
        if getattr(a,key) is not None:config[key]=getattr(a,key)
    if config['epochs']<1 or config['batch_size']<1 or config['patience']<1:p.error('Positive epochs, batch size and patience required')
    if config['selection_metric']!='unweighted_validation_loss':p.error('Only predeclared validation-loss selection is supported')
    checkpoint_dir=BASE/'checkpoints'/a.run_name;report_dir=BASE/'reports'/a.run_name
    path=checkpoint_dir/'best.pt'
    if path.exists():raise FileExistsError(f'{path} already exists; choose a new --run-name')
    device=setup(config['seed'],config['threads'],a.device)
    started=time.perf_counter();data,preprocessing=load_splits(a.data_dir,('train','val'))
    data_seconds=time.perf_counter()-started
    for split in ('train','val'):
        if set(data[split].y)!={0,1}:raise ValueError(f'{split} needs both classes')
    checkpoint_dir.mkdir(parents=True,exist_ok=True);report_dir.mkdir(parents=True,exist_ok=True)
    model=build_model(config).to(device);parameters=sum(p.numel() for p in model.parameters())
    counts=np.bincount(data['train'].y,minlength=2)
    weights=torch.tensor(len(data['train'])/(2*counts),dtype=torch.float32,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    generator=torch.Generator().manual_seed(config['seed'])
    train_loader=DataLoader(data['train'],batch_size=config['batch_size'],shuffle=True,num_workers=0,generator=generator)
    val_loader=DataLoader(data['val'],batch_size=config['batch_size'],shuffle=False,num_workers=0)
    run=dict(model_type=MODEL_TYPE,initialization='random; no pretrained weights',config=config,device=str(device),
        parameter_count=parameters,class_weights=weights.cpu().tolist(),training_class_distribution=counts.tolist(),
        preprocessing_config_sha256=sha256(a.data_dir/'preprocessing_config.json'),
        artifact_sha256=preprocessing['artifact_sha256'],training_inputs_opened=['train.npz','val.npz'],
        versions={name:importlib.metadata.version(name) for name in ('torch','numpy')},
        python=platform.python_version(),platform=platform.platform(),processor=platform.processor())
    write_json(report_dir/'run_config.json',run)
    print(json.dumps(dict(model_type=MODEL_TYPE,parameters=parameters,device=str(device),class_counts=counts.tolist())),flush=True)
    best=float('inf');stale=0;history=[];loop_start=time.perf_counter()
    for number in range(1,config['epochs']+1):
        tick=time.perf_counter()
        training=epoch(model,train_loader,device,weights,optimizer,config['gradient_clip'],config['decision_threshold'])
        validation=epoch(model,val_loader,device,threshold=config['decision_threshold'])
        improved=validation['loss']<best-config['min_delta']
        record=dict(epoch=number,training=training,validation=validation,seconds=time.perf_counter()-tick,selected=improved)
        history.append(record);write_json(report_dir/'history.json',history)
        print(json.dumps(dict(epoch=number,train_weighted_loss=training['loss'],validation_loss=validation['loss'],
            train_f1=training['f1'],validation_f1=validation['f1'],validation_fall_recall=validation['fall_recall'],seconds=record['seconds'],selected=improved)),flush=True)
        if improved:
            best=validation['loss'];stale=0
            checkpoint=dict(format_version=1,model_type=MODEL_TYPE,config=config,
                state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                epoch=number,validation=validation,preprocessing=preprocessing,run=run)
            temporary=path.with_suffix('.tmp');torch.save(checkpoint,temporary);temporary.replace(path)
        else:stale+=1
        if stale>=config['patience']:break
    loop_seconds=time.perf_counter()-loop_start
    del optimizer,model
    best_model,saved=load_cnn(path,device)
    saved['training_duration_seconds']=loop_seconds
    saved['data_loading_seconds']=data_seconds
    saved['epochs_completed']=len(history)
    temporary=path.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(path)
    train_eval_loader=DataLoader(data['train'],batch_size=config['batch_size'],shuffle=False,num_workers=0)
    train_metrics=epoch(best_model,train_eval_loader,device,threshold=config['decision_threshold'])
    val_metrics=epoch(best_model,val_loader,device,threshold=config['decision_threshold'])
    # Save comparable train/validation metrics on exactly the selected weights.
    with torch.inference_mode():
        val_pred=np.concatenate([(best_model(x.to(device)).softmax(-1)[:,1]>=config['decision_threshold']).cpu().numpy() for x,_ in val_loader])
    val_events=event_metrics(data['val'],val_pred,preprocessing,'val')
    reloaded,_=load_cnn(path,device)
    sample=data['val'][0][0].unsqueeze(0).to(device)
    with torch.inference_mode():torch.testing.assert_close(best_model(sample),reloaded(sample),rtol=0,atol=0)
    after=sha256(a.data_dir/'preprocessing_config.json')
    if after!=run['preprocessing_config_sha256']:raise ValueError('Preprocessing config changed while training')
    summary=dict(**run,best_epoch=saved['epoch'],epochs_completed=len(history),stopped_early=len(history)<config['epochs'],
        early_stopping_metric='unweighted validation cross entropy; test never opened',
        training_duration_seconds=loop_seconds,data_loading_seconds=data_seconds,total_command_seconds=time.perf_counter()-started,
        checkpoint_bytes=path.stat().st_size,checkpoint_sha256=sha256(path),
        selected_model_train=train_metrics,selected_model_validation=val_metrics,
        validation_events=val_events,checkpoint_reload_verified=True,
        limitations=['Validation has only eight correlated positive windows from four events in one session.',
            'Single fixed baseline run; threshold 0.5 and hyperparameters set before test evaluation.'])
    write_json(report_dir/'training_summary.json',summary)
    print(json.dumps(dict(best_epoch=saved['epoch'],epochs=len(history),training_seconds=loop_seconds,
        checkpoint_bytes=path.stat().st_size,train_f1=train_metrics['f1'],validation_f1=val_metrics['f1'],checkpoint=str(path))),flush=True)

if __name__=='__main__':main()
