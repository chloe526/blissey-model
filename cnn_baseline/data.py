"""Independent read-only loader for the shared corrected CSI dataset."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from labeling_v3 import validate_config,validate_provenance


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


class CSIWindows(Dataset):
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.X=data['X'].copy();self.y=data['y'].copy()
            self.recording_id=data['recording_id'].copy()
            self.start=data['start_seconds'].copy();self.end=data['end_seconds'].copy()
            self.event_id=data['event_id'].copy() if 'event_id' in data else None
        n=len(self.y)
        if self.X.shape!=(n,52,512) or n==0 or self.X.dtype!=np.float32:raise ValueError('Expected nonempty float32 [N,52,512]')
        if self.y.shape!=(n,) or self.y.dtype!=np.int64 or not set(self.y.tolist())<={0,1}:raise ValueError('Invalid binary int64 labels')
        if not np.isfinite(self.X).all():raise ValueError('Non-finite CSI inputs')
        if not (len(self.recording_id)==len(self.start)==len(self.end)==n):raise ValueError('Missing window provenance')
        if not np.isfinite(self.start).all() or not np.isfinite(self.end).all() or np.any(self.end<=self.start):raise ValueError('Invalid window timestamps')

    def __len__(self):return len(self.y)
    def __getitem__(self,index):return torch.from_numpy(self.X[index]),torch.tensor(self.y[index],dtype=torch.long)


def load_splits(root,splits=('train','val')):
    root=Path(root)
    if not splits or len(set(splits))!=len(splits) or not set(splits)<={'train','val','test'}:raise ValueError('Invalid splits')
    config=json.loads((root/'preprocessing_config.json').read_text())
    if config.get('preprocessing_version') not in (2,3) or config.get('packet_quality',{}).get('version')!=2 or config.get('diagnostic_subset'):raise ValueError('CNN requires quality-v2 signals with v2/v3 supervision')
    validate_config(config)
    if len(config['channel_indices'])!=52 or config['sequence_length']!=512:raise ValueError('Incompatible preprocessing')
    # Validate the full recording manifest without opening test.npz in training.
    declared={split:set() for split in ('train','val','test')}
    for rid,record in config['recordings'].items():
        split=record['split']
        if split not in declared or record['group'] not in config['group_splits'][split]:raise ValueError('Invalid recording/group assignment')
        declared[split].add(record['group'])
    for a,b in (('train','val'),('train','test'),('val','test')):
        if declared[a]&declared[b]:raise ValueError('Collection-session leakage')
    datasets={}
    for split in splits:
        path=root/f'{split}.npz'
        if sha256(path)!=config['artifact_sha256'][split]:raise ValueError(f'{split} artifact hash mismatch')
        dataset=CSIWindows(path)
        validate_provenance(config,dataset,split)
        for rid in set(dataset.recording_id):
            if config['recordings'][str(rid)]['split']!=split:raise ValueError('Recording assigned to wrong split')
        datasets[split]=dataset
    for i,a in enumerate(splits):
        for b in splits[i+1:]:
            if set(datasets[a].recording_id)&set(datasets[b].recording_id):raise ValueError('Recording leakage')
    return datasets,config
