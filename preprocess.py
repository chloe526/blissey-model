"""Session-disjoint, timestamp-based CSI windows. Never writes to raw data."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from csi import amplitudes, load_recording
from csi_quality import window_quality
from inspect_dataset import audit


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_window(start, end, status, intervals, margin=0.25):
    if status == 'verified_non_fall': return 0
    if status == 'annotated_fall' and any(start >= a+margin and end <= b-margin for a,b in intervals):
        return 1
    # Partial overlap, boundary uncertainty, and fall-trial baseline are excluded.
    return None


def transform_window(times, features, start, config, normalize=True):
    """Same operation for offline and streaming inference; no arbitrary reshaping."""
    times=np.asarray(times,dtype=np.float64); features=np.asarray(features,dtype=np.float64)
    end=start+config['window_seconds']
    if features.shape != (len(times),len(config['channel_indices'])) or not np.isfinite(features).all():
        raise ValueError('Invalid/non-finite CSI feature matrix')
    quality=window_quality(times,start,config)
    if quality['reasons']:raise ValueError(','.join(quality['reasons']))
    left,right=quality['left'],quality['right']
    tt=times[left:right]; xx=features[left:right]
    if not np.any(xx):raise ValueError('No CSI signal')
    # Uniform wall-clock grid directly to 512. No fabricated temporal resolution;
    # Quality uses fixed wall-clock coverage requirements, not tensor length.
    grid=np.linspace(start,end,config['sequence_length'])
    result=np.stack([np.interp(grid,tt,xx[:,i]) for i in range(xx.shape[1])])
    if normalize:
        result=(result-np.asarray(config['mean'])[:,None])/np.asarray(config['std'])[:,None]
    if not np.isfinite(result).all(): raise ValueError('Non-finite model input')
    return result.astype(np.float32)


def recording_features(rows, indices):
    unique={}; duplicates=0
    for packet in rows:
        t=float(packet['elapsed_seconds'])
        if t in unique: duplicates+=1; continue
        unique[t]=amplitudes(packet,indices)
    times=np.array(sorted(unique))
    return times,np.stack([unique[t] for t in times]),duplicates


def prepare(root, output, seed=42, quality_config=None):
    root=Path(root); output=Path(output)
    report=audit(root)
    if not report['supervised_ready']: raise ValueError('Verified fall intervals and explicit non-fall trials required')
    # A physical session is the independent unit, including its inconsistent metadata IDs.
    groups=report['groups']
    if groups != ['session_01','session_02','session_03_LAHACKS']:
        raise ValueError('Collection groups changed: review and explicitly update session split policy')
    split={'train':['session_02'],'val':['session_01'],'test':['session_03_LAHACKS']}
    train_records=[r for r in report['recordings'] if r['group'] in split['train'] and r['label_status']!='ambiguous']
    # Feature selection uses TRAIN ONLY. Physical LLTF null/guard bins and first
    # four unreliable bytes are excluded even if noise makes them nonzero.
    observed=set.intersection(*(set(r['observed_nonzero_pair_indices']) for r in train_records))
    physical=set(range(6,32))|set(range(33,59))
    indices=sorted(observed & physical)
    if not indices: raise ValueError('No valid training channels')
    rate=float(np.median([r['measured_rate_hz'] for r in train_records]))
    quality_path=Path(quality_config) if quality_config else Path(__file__).parent/'configs/packet_quality_v2.json'
    packet_quality=json.loads(quality_path.read_text())
    if packet_quality['version']!=2:raise ValueError('New preprocessing requires quality version 2')
    config=dict(schema_version=1,seed=seed,window_seconds=3.0,stride_seconds=0.5,
        sequence_length=512,channel_indices=indices,reference_rate_hz=rate,
        approximate_native_samples_per_window=round(rate*3),max_gap_seconds=packet_quality['max_gap_seconds'],
        preprocessing_version=2,packet_quality=packet_quality,quality_config_sha256=fingerprint(quality_path),
        resampling='linear interpolation on a uniform wall-clock grid including both endpoints; no padding',
        label_policy='Explicit normal category -> 0. Staged-fall window entirely within host-aligned action interval with 0.25s inner margin -> 1. All other fall windows excluded.',
        group_splits=split,normalization='per-channel training windows only; MOMENT also applies its native per-window RevIN',
        independent_unit='physical collection session',recordings={},quality={})
    arrays={}
    for part, selected in split.items():
        windows=[]; labels=[]; ids=[]; starts=[]; ends=[]; diagnostics=[]
        quality=dict(rejection_reasons={},excluded_recordings=0,excluded_label_windows=0,rejected_quality_windows=0,duplicate_packets=0)
        for r in report['recordings']:
            if r['group'] not in selected: continue
            if r['label_status']=='ambiguous': quality['excluded_recordings']+=1;continue
            folder=root/r['path']; m,rows,status,intervals,issues,errors=load_recording(folder)
            times,features,dupes=recording_features(rows,indices); quality['duplicate_packets']+=dupes
            rid=r['recording_id']
            config['recordings'][rid]={k:r[k] for k in ('group','session','participant','activity','label_status','fall_intervals','start_seconds','end_seconds','exposure_seconds','measured_rate_hz')}
            config['recordings'][rid].update(split=part,measurement_sha256=fingerprint(folder/'measurements.jsonl'))
            for start in np.arange(times[0],times[-1]-config['window_seconds']+1e-9,config['stride_seconds']):
                end=start+config['window_seconds'];label=label_window(start,end,status,intervals)
                if label is None:quality['excluded_label_windows']+=1;continue
                try: x=transform_window(times,features,start,config,normalize=False)
                except ValueError as exc:
                    quality['rejected_quality_windows']+=1
                    reason=str(exc);quality['rejection_reasons'][reason]=quality['rejection_reasons'].get(reason,0)+1
                    continue
                windows.append(x); labels.append(label);ids.append(rid);starts.append(start);ends.append(end)
                diagnostics.append(window_quality(times,start,config))
        if not windows or set(labels)!={0,1}:
            raise ValueError(f'{part}: insufficient valid windows of BOTH classes; no artifacts written')
        arrays[part]=dict(X=np.stack(windows),y=np.asarray(labels,dtype=np.int64),
            recording_id=np.asarray(ids),start_seconds=np.asarray(starts),end_seconds=np.asarray(ends))
        for key in ('effective_rate_hz','occupied_bin_fraction','sample_span_fraction','max_edge_distance_seconds','max_gap_seconds'):
            arrays[part][key]=np.asarray([d[key] for d in diagnostics],dtype=np.float64)
        quality.update(windows=len(labels),class_counts=np.bincount(labels,minlength=2).tolist(),
            independent_recordings=len(set(ids)),independent_sessions=len(selected))
        config['quality'][part]=quality
    train=arrays['train']['X'].astype(np.float64)
    config['mean']=train.mean(axis=(0,2)).tolist()
    config['std']=np.maximum(train.std(axis=(0,2)),1e-6).tolist()
    output.parent.mkdir(parents=True,exist_ok=True)
    for part,arr in arrays.items():
        arr['X']=((arr['X']-np.asarray(config['mean'])[None,:,None])/np.asarray(config['std'])[None,:,None]).astype(np.float32)
        if not np.isfinite(arr['X']).all():raise ValueError('Non-finite normalized windows')
    # Finish every artifact before replacing outputs; config is committed last.
    with tempfile.TemporaryDirectory(prefix='preprocess-',dir=output.parent) as staging:
        staging=Path(staging)
        for part,arr in arrays.items():np.savez_compressed(staging/f'{part}.npz',**arr)
        config['artifact_sha256']={part:fingerprint(staging/f'{part}.npz') for part in arrays}
        (staging/'preprocessing_config.json').write_text(json.dumps(config,indent=2)+'\n')
        output.mkdir(parents=True,exist_ok=True)
        for name in ('train.npz','val.npz','test.npz','preprocessing_config.json'):
            (staging/name).replace(output/name)
    print(json.dumps(config['quality'],indent=2));print('Prepared:',output)
    return config


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',default='../blissey-data/recordings')
    p.add_argument('--output',default='data/processed');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--quality-config',default=None);a=p.parse_args()
    prepare(a.data_root,a.output,a.seed,a.quality_config)

if __name__=='__main__':main()
