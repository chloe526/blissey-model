"""CNN-specific reproducibility, safe checkpoint I/O and local timing."""
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from .model import MODEL_TYPE,build_model


def setup(seed,threads,device):
    if threads<1:raise ValueError('threads must be positive')
    torch.set_num_threads(threads);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('Requested local CUDA is unavailable')
        torch.cuda.manual_seed_all(seed)
    elif device!='cpu':raise ValueError('Use cpu or an available local cuda device')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return torch.device(device)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_cnn(path,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('model_type')!=MODEL_TYPE or checkpoint.get('format_version')!=1:
        raise ValueError('This loader accepts only CNN-baseline checkpoints, not MOMENT weights')
    model=build_model(checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.to(device).eval()
    return model,checkpoint


def benchmark(model,sample,device,warmup=20,repetitions=200):
    if len(sample)!=1:raise ValueError('Benchmark is batch size one')
    def synchronize():
        if device.type=='cuda':torch.cuda.synchronize(device)
    model.eval();durations=[]
    with torch.inference_mode():
        for _ in range(warmup):model(sample)
        synchronize()
        for _ in range(repetitions):
            synchronize();start=time.perf_counter_ns();model(sample);synchronize()
            durations.append((time.perf_counter_ns()-start)/1e6)
    return dict(batch_size=1,input_shape=list(sample.shape),device=str(device),threads=torch.get_num_threads(),
        warmup=warmup,repetitions=repetitions,median_ms=float(np.median(durations)),
        mean_ms=float(np.mean(durations)),p95_ms=float(np.quantile(durations,.95)),
        scope='Model forward only, tensor already on device. Excludes CSI acquisition, preprocessing, file I/O and transfer.')
