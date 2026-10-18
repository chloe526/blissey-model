import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from labeling_v3 import validate_config,validate_provenance


class CSIDataset(Dataset):
    def __init__(self,path,channels=None):
        with np.load(path,allow_pickle=False) as data:
            self.X=data['X'].copy();self.y=data['y'].copy()
            self.recording_id=data['recording_id'].copy()
            self.start=data['start_seconds'].copy();self.end=data['end_seconds'].copy()
            self.event_id=data['event_id'].copy() if 'event_id' in data else None
        if self.X.ndim!=3 or self.X.shape[2]!=512 or len(self.X)!=len(self.y) or not len(self.y):raise ValueError(f'{path}: invalid/empty tensors')
        if self.X.dtype!=np.float32 or self.y.dtype!=np.int64:raise ValueError(f'{path}: expected float32 X, int64 y')
        if channels is not None and self.X.shape[1]!=channels:raise ValueError('Channel mismatch')
        if not np.isfinite(self.X).all() or not set(self.y.tolist())<={0,1}:raise ValueError('Non-finite inputs or invalid labels')
        if not (len(self.start)==len(self.end)==len(self.recording_id)==len(self.y)):raise ValueError('Provenance length mismatch')
        if not np.isfinite(self.start).all() or not np.isfinite(self.end).all() or np.any(self.end<=self.start):raise ValueError('Invalid window times')

    def __len__(self):return len(self.y)
    def __getitem__(self,index):return torch.from_numpy(self.X[index]),torch.tensor(self.y[index],dtype=torch.long)


def load_data(root,splits=('train','val','test')):
    root=Path(root)
    if not splits or len(set(splits))!=len(splits) or not set(splits)<={'train','val','test'}:
        raise ValueError('Invalid split selection')
    try:
        config=json.loads((root/'preprocessing_config.json').read_text())
        validate_config(config)
        data={}
        for split in splits:
            path=root/f'{split}.npz'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=config['artifact_sha256'][split]:raise ValueError(f'{split}: artifact hash mismatch')
            data[split]=CSIDataset(path,len(config['channel_indices']))
            validate_provenance(config,data[split],split)
            for rid in set(data[split].recording_id):
                if config['recordings'][rid]['split']!=split:raise ValueError('Incorrect recording provenance')
                if config['recordings'][rid]['group'] not in config['group_splits'][split]:
                    raise ValueError('Recording does not belong to its assigned collection group')
        for a,b in ((a,b) for i,a in enumerate(splits) for b in splits[i+1:]):
            if set(data[a].recording_id)&set(data[b].recording_id):raise ValueError('Recording leakage')
            groups_a={config['recordings'][rid]['group'] for rid in data[a].recording_id}
            groups_b={config['recordings'][rid]['group'] for rid in data[b].recording_id}
            if groups_a&groups_b:raise ValueError('Session leakage')
        return data,config
    except (OSError,KeyError) as exc:raise ValueError(f'Cannot load prepared data in {root}. Run preprocess.py first.') from exc
