"""Real official pretrained forward pass and one gradient step (no GPU job)."""
import argparse
import json
from pathlib import Path
import torch
from model_utils import device_for,load_pretrained,logits_for,seed_all,training_mode,write_json
from training_data import load_data


def main():
    p=argparse.ArgumentParser();p.add_argument('--import-only',action='store_true')
    p.add_argument('--data-dir',default='data/processed');p.add_argument('--device',default='auto')
    p.add_argument('--report',default='reports/moment_smoke.json');a=p.parse_args()
    from momentfm import MOMENTPipeline
    print('Official MOMENTPipeline import OK',flush=True)
    if a.import_only:return
    data,pre=load_data(a.data_dir);seed_all(42);device=device_for(a.device)
    config=json.loads(Path('configs/training.json').read_text());model=load_pretrained(len(pre['channel_indices']),config).to(device)
    # Take an actual FALL window to exercise positive-class gradients.
    index=int((data['train'].y==1).nonzero()[0][0]);x,y=data['train'][index]
    x=x.unsqueeze(0).to(device);y=y.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():logits=logits_for(model,x,config['reduction']);prob=logits.softmax(-1)
    torch.testing.assert_close(prob.sum(-1),torch.ones(1,device=device))
    training_mode(model);optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-3)
    before=model.head.linear.weight.detach().clone();optimizer.zero_grad()
    loss=torch.nn.functional.cross_entropy(logits_for(model,x,config['reduction']),y);loss.backward()
    gradients={n:p.grad is not None and bool(torch.isfinite(p.grad).all()) for n,p in model.named_parameters() if p.requires_grad}
    if not gradients or not all(gradients.values()):raise RuntimeError('Trainable gradients invalid')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise RuntimeError('Frozen parameters received gradients')
    optimizer.step()
    if torch.equal(before,model.head.linear.weight):raise RuntimeError('Head weights did not update')
    result=dict(passed=True,model_id=config['model_id'],model_revision=config['model_revision'],device=str(device),
        input_shape=list(x.shape),mask_shape=[1,512],logits=logits.cpu().tolist(),probabilities=prob.cpu().tolist(),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        gradients=gradients,loss=float(loss.detach().cpu()),head_updated=True)
    write_json(a.report,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
