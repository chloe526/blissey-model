"""Official pretrained MOMENT model construction and portable checkpoint reload."""
import json
import random
from pathlib import Path
import numpy as np
import torch

MODEL_ID='AutonLab/MOMENT-1-large'
MODEL_REVISION='ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc'


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def device_for(name='auto'):
    # CPU is the portable local default; CUDA is the remote default. MPS opt-in.
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if name=='auto' else name)


def load_pretrained(channels, config):
    from momentfm import MOMENTPipeline
    from huggingface_hub import try_to_load_from_cache, constants
    import shutil
    cache=Path(constants.HF_HUB_CACHE)
    existing=[try_to_load_from_cache(config.get('model_id',MODEL_ID),name,revision=config.get('model_revision',MODEL_REVISION)) for name in ('model.safetensors','pytorch_model.bin')]
    if not any(isinstance(path,str) and Path(path).is_file() for path in existing):
        location=cache
        while not location.exists():location=location.parent
        free=shutil.disk_usage(location).free
        if free<2_000_000_000:
            raise RuntimeError(f'Pretrained checkpoint download needs >=2 GB free in HF cache; available {free/1e9:.2f} GB. Set HF_HOME to a larger disk.')
    try:
        model=MOMENTPipeline.from_pretrained(config.get('model_id',MODEL_ID),
            revision=config.get('model_revision',MODEL_REVISION),
            model_kwargs=dict(task_name='classification',n_channels=channels,num_class=2,
                reduction=config.get('reduction','concat'),freeze_encoder=True,freeze_embedder=True))
        model.init()
    except Exception as exc:
        raise RuntimeError('Cannot load official pretrained MOMENT. Check pinned requirements, network access and HF cache; no random fallback.') from exc
    configure_trainable(model,config.get('unfreeze_last_blocks',0))
    return model


def configure_trainable(model, unfreeze=0):
    for p in model.parameters():p.requires_grad=False
    for p in model.head.parameters():p.requires_grad=True
    if unfreeze:
        if not 0 < unfreeze <= len(model.encoder.block):raise ValueError('Invalid encoder block count')
        for block in model.encoder.block[-unfreeze:]:
            for p in block.parameters():p.requires_grad=True
    counts=dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad))
    print('Parameters:',json.dumps(counts),flush=True)
    return counts


def training_mode(model):
    model.train()
    # Frozen backbone dropout would otherwise add noise and change embeddings.
    for name in ('encoder','patch_embedding','normalizer'):
        module=getattr(model,name,None)
        if module is not None and not any(p.requires_grad for p in module.parameters()):module.eval()


def logits_for(model,x,reduction='concat'):
    if x.ndim!=3 or x.shape[-1]!=512 or not torch.isfinite(x).all():raise ValueError('Input must be finite [B,C,512]')
    mask=torch.ones((len(x),512),dtype=torch.long,device=x.device)
    logits=model(x_enc=x,input_mask=mask,reduction=reduction).logits
    if logits.shape!=(len(x),2) or not torch.isfinite(logits).all():raise ValueError('Invalid binary logits')
    return logits


def load_checkpoint(path,device):
    from momentfm import MOMENTPipeline
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    if ckpt.get('schema_version')!=1:raise ValueError('Unsupported checkpoint schema')
    n=len(ckpt['preprocessing']['channel_indices']);config=ckpt['training_config']
    # Construct from saved architecture and full trained state: no HF download.
    model=MOMENTPipeline(config=ckpt['model_config'],model_kwargs=dict(
        task_name='classification',n_channels=n,num_class=2,reduction=config['reduction']))
    model.init();model.load_state_dict(ckpt['state_dict'],strict=True);model.to(device).eval()
    return model,ckpt


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
