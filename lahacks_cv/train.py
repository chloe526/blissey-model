"""Two outer block folds; fresh CNN/RF; no test set, tuning or deployment refit."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, Dataset

from cnn_baseline.model import build_model, weighted_cross_entropy
from .common import (FOLDS, ROOT, STATUS, canonical_hash, class_weights, fit_normalization,
                     fold_indices, inner_recording_split, normalize, sha, write_json)
from .data import load_dataset
from .features import FEATURE_CONFIG, extract, extract_one
from .metrics import ALERT_POLICY, classification, event_summary, events, verify_rf_reload


def seed_everything(seed, threads):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class Windows(Dataset):
    def __init__(self, data, indices, scaler):
        self.data, self.indices, self.scaler = data, indices, scaler

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return torch.from_numpy(normalize(self.data['X'][j], self.scaler)), int(self.data['y'][j])


def loader(data, indices, scaler, config, seed, shuffle=False):
    return DataLoader(Windows(data, indices, scaler), batch_size=config['batch_size'], shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def epoch(model, batches, device, optimizer=None, weights=None, clip=5.):
    model.train(optimizer is not None); total = 0.; count = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = weighted_cross_entropy(model(x), y, weights)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite CNN loss')
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
                optimizer.step()
        total += float(loss.detach()) * len(y); count += len(y)
    return total / count


def predict_cnn(model, data, indices, scaler, config, device):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(x.to(device)).softmax(-1)[:, 1].cpu().numpy()
                               for x, _ in loader(data, indices, scaler, config, 0)])


def state_digest(model):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fit_cnn(data, train, scaler, run, device, output):
    config = run['cnn']; inner_train, inner_dev = inner_recording_split(data, train, run['seed'])
    inner_scaler = fit_normalization(data['X'], inner_train)
    seed_everything(run['seed'], run['threads'])
    model = build_model(config).to(device)
    weights = torch.tensor(class_weights(data['y'][inner_train]), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    fit_batches = loader(data, inner_train, inner_scaler, config, run['seed'], True)
    dev_batches = loader(data, inner_dev, inner_scaler, config, run['seed'])
    best = float('inf'); stale = 0; best_epoch = 0; history = []; best_state = None
    started = time.perf_counter()
    for number in range(1, config['epochs'] + 1):
        training_loss = epoch(model, fit_batches, device, optimizer, weights, config['gradient_clip'])
        validation_loss = epoch(model, dev_batches, device)
        history.append(dict(epoch=number, weighted_training_loss=training_loss, inner_validation_loss=validation_loss))
        print(json.dumps(dict(stage='inner_cnn', **history[-1])), flush=True)
        if validation_loss < best - config['min_delta']:
            best = validation_loss; best_epoch = number; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config['patience']:
            break
    torch.save(dict(state_dict=best_state, config=config, normalization=inner_scaler,
                    best_epoch=best_epoch, initialization='fresh random'), output / 'inner_cnn_best.pt')
    write_json(output / 'inner_selection.json', dict(history=history, selected_epochs=best_epoch,
        fit_recordings=sorted(set(data['recording_id'][inner_train].tolist())),
        development_recordings=sorted(set(data['recording_id'][inner_dev].tolist())),
        normalization=inner_scaler, class_weights=weights.cpu().tolist(),
        selection='unweighted inner recording-level loss; outer validation never read'))
    # Outer-fold model starts from scratch and fits ALL outer training recordings.
    # This is a fold fit, not the separately gated all-LA deployment refit.
    del model, optimizer, best_state
    seed_everything(run['seed'] + 1000, run['threads'])
    model = build_model(config).to(device)
    initial_digest = state_digest(model)
    weights = torch.tensor(class_weights(data['y'][train]), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    batches = loader(data, train, scaler, config, run['seed'] + 1000, True)
    for number in range(best_epoch):
        loss = epoch(model, batches, device, optimizer, weights, config['gradient_clip'])
        print(json.dumps(dict(stage='outer_cnn_fit', epoch=number + 1, weighted_training_loss=loss)), flush=True)
    seconds = time.perf_counter() - started
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(dict(format_version=1, state_dict=state, architecture=config, normalization=scaler,
                    selected_epochs=best_epoch, seed=run['seed'] + 1000,
                    status=STATUS, initialization='fresh random; no checkpoint resume',
                    initial_state_sha256=initial_digest, state_sha256=state_digest(model)), output / 'cnn.pt')
    return model, seconds, best_epoch


def latency(fn, x, repeats=50):
    for _ in range(5):
        fn(x)
    durations = []
    for _ in range(repeats):
        tick = time.perf_counter(); fn(x); durations.append((time.perf_counter() - tick) * 1000)
    return dict(batch_size=1, device='cpu', samples=repeats, median_ms=float(np.median(durations)),
                p95_ms=float(np.quantile(durations, .95)), includes='window normalization/features and model; excludes packet parsing/resampling')


def benchmark(data, metadata, run, output, device):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite a run or reuse its checkpoints')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing silent CPU fallback')
    output.mkdir(parents=True)
    run = dict(run, status=STATUS, alert_policy=ALERT_POLICY, feature_config=FEATURE_CONFIG,
               source_manifest_sha256=metadata['source_manifest_sha256'],
               dataset_sha256=metadata['artifact_sha256'], device=device,
               code_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'lahacks_cv').glob('*.py'))},
               versions={p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scipy', 'scikit-learn', 'joblib')},
               refit_performed=False, threshold_tuned=False)
    write_json(output / 'run_config.json', run)
    write_json(output / 'dataset.json', metadata)
    seed_everything(run['seed'], run['threads'])
    started = time.perf_counter(); features = extract(data['X']); feature_seconds = time.perf_counter() - started
    results = {}; predictions = {'cnn': [], 'rf': []}; truths = []; indices_list = []
    for fold in FOLDS:
        folder = output / ('fold_' + fold); folder.mkdir()
        train, val = fold_indices(data, fold)
        scaler = fit_normalization(data['X'], train)
        if not np.allclose(scaler['mean'], metadata['fold_normalization'][fold]['mean'], rtol=0, atol=0):
            raise ValueError('Training-fold normalization is not reproducible')
        ids = dict(training_windows=canonical_hash([(str(data['recording_id'][i]), float(data['start'][i]), int(data['y'][i])) for i in train]),
                   validation_windows=canonical_hash([(str(data['recording_id'][i]), float(data['start'][i]), int(data['y'][i])) for i in val]))
        write_json(folder / 'preprocessing.json', dict(geometry=metadata['geometry'], normalization=scaler,
            feature_config=FEATURE_CONFIG, alert_policy=ALERT_POLICY, class_weights=class_weights(data['y'][train]).tolist(),
            train_group=FOLDS[fold][0], validation_group=FOLDS[fold][1], equivalent_input_hashes=ids))
        cnn, cnn_seconds, selected_epochs = fit_cnn(data, train, scaler, run, device, folder)
        cp = predict_cnn(cnn, data, val, scaler, run['cnn'], device)
        cnn_train_metrics = classification(data['y'][train], predict_cnn(cnn, data, train, scaler, run['cnn'], device))
        cnn.cpu().eval()
        saved = torch.load(folder / 'cnn.pt', map_location='cpu', weights_only=True)
        reloaded = build_model(saved['architecture']); reloaded.load_state_dict(saved['state_dict']); reloaded.eval()
        sample = torch.from_numpy(normalize(data['X'][val[0]], scaler)).unsqueeze(0)
        with torch.inference_mode():
            torch.testing.assert_close(cnn(sample), reloaded(sample), rtol=0, atol=0)
            cnn_latency = latency(lambda x: reloaded(torch.from_numpy(normalize(x, scaler)).unsqueeze(0)), data['X'][val[0]])
        weight = class_weights(data['y'][train]); tick = time.perf_counter()
        rf = RandomForestClassifier(**run['rf'], class_weight={0: float(weight[0]), 1: float(weight[1])},
                                    random_state=run['seed'], n_jobs=run['threads'])
        rf.fit(features[train], data['y'][train]); rf_seconds = time.perf_counter() - tick
        rp = rf.predict_proba(features[val])[:, list(rf.classes_).index(1)]
        joblib.dump(dict(model=rf, features=FEATURE_CONFIG, status=STATUS,
                         source_manifest_sha256=metadata['source_manifest_sha256']), folder / 'rf.joblib', compress=3)
        rf_loaded = joblib.load(folder / 'rf.joblib')['model']
        rf_reload = verify_rf_reload(rp, rf_loaded.predict_proba(features[val])[:, 1])
        rf_latency = latency(lambda x: rf_loaded.predict_proba(extract_one(x)[None]), data['X'][val[0]])
        results[fold] = {}
        for name, prob, seconds, timings, artifact in (
                ('cnn', cp, cnn_seconds, cnn_latency, folder / 'cnn.pt'),
                ('rf', rp, rf_seconds, rf_latency, folder / 'rf.joblib')):
            train_metrics = cnn_train_metrics if name == 'cnn' else classification(data['y'][train], rf.predict_proba(features[train])[:, 1])
            results[fold][name] = dict(window=classification(data['y'][val], prob),
                events=events(data, val, prob, metadata), training_window=train_metrics,
                inference_latency=timings, training_seconds=seconds, artifact_bytes=artifact.stat().st_size,
                artifact_sha256=sha(artifact), equivalent_input_hashes=ids,
                checkpoint_reload_verified=True,
                excessive_false_positive_diagnostic='Compare fixed-cap training/validation predicted-fall fractions; no outer-fold reweighting or retuning.')
            predictions[name].append(prob)
        results[fold]['cnn']['parameter_count'] = sum(p.numel() for p in cnn.parameters())
        results[fold]['cnn']['selected_epochs'] = selected_epochs
        results[fold]['rf']['reload_verification'] = rf_reload
        np.savez_compressed(folder / 'predictions.npz', y=data['y'][val], recording_id=data['recording_id'][val],
                            start=data['start'][val], end=data['end'][val], cnn_probability=cp, rf_probability=rp)
        write_json(folder / 'metrics.json', results[fold])
        truths.append(data['y'][val]); indices_list.append(val)
        del cnn, reloaded, rf, rf_loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pooled = {}; macro = {}
    for name in ('cnn', 'rf'):
        em = [results[f][name]['events'] for f in FOLDS]
        pooled[name] = dict(window=classification(np.concatenate(truths), np.concatenate(predictions[name])),
            events=event_summary([e for m in em for e in m['per_event']],
                                 sum(m['false_alert_episodes'] for m in em), sum(m['accepted_normal_seconds'] for m in em)))
        macro[name] = {key: float(np.mean([results[f][name]['window'][key] for f in FOLDS]))
                       for key in ('fall_precision', 'fall_recall', 'fall_f1', 'roc_auc', 'pr_auc_average_precision')}
        macro[name].update({key: float(np.mean([m[key] for m in em]))
                            for key in ('covered_event_recall', 'all_annotated_event_recall', 'false_alerts_per_hour')})
    ranking = sorted(('cnn', 'rf'), key=lambda n: (-macro[n]['covered_event_recall'],
                     macro[n]['false_alerts_per_hour'], -macro[n]['fall_f1'], n))
    summary = dict(status=STATUS, folds=results, macro=macro, pooled=pooled, proposed_ranking=ranking,
        ranking_rule='macro covered-event recall descending, false alerts/hour ascending, window F1 descending; review required',
        feature_extraction_seconds=feature_seconds, feature_count=features.shape[1],
        deployment_refit_performed=False, review_required_before_refit=True,
        limitations=['Two blocks, one participant; no independent final test.',
                     'Only 13 quality-covered events and ~17.6 minutes accepted normal exposure.',
                     'CV model selection is exploratory development, not unbiased final performance.',
                     'Fixed thresholds/voting; no held-out threshold search. Acquisition failures reported separately.',
                     'Weighted probabilities are decision scores, not calibrated fall risks.'])
    write_json(output / 'benchmark.json', summary)
    artifacts = {str(p.relative_to(output)): dict(bytes=p.stat().st_size, sha256=sha(p))
                 for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'artifacts.json', artifacts)
    print(json.dumps(dict(status=STATUS, proposed_ranking=ranking, results=str(output / 'benchmark.json'),
                          review_required_before_refit=True)), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    a = p.parse_args()
    output = a.output or (Path(os.environ['VERI_OUTPUT_DIR']) / 'lahacks_cv_v1')
    config = json.loads((ROOT / 'lahacks_cv/config.json').read_text())
    data, meta = load_dataset(a.data_dir)
    with threadpool_limits(limits=config['threads']):
        benchmark(data, meta, config, output, a.device)


if __name__ == '__main__':
    main()
