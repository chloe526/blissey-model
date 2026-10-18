"""Evaluate an exported model (model.json).

Backends:
- tflite: when model.json has a "tflite" entry (export_tflite.py), the score comes from model.tflite run by
  tflite_runtime (QNX: python3-tflite-runtime from oss.qnx.com; falls back to ai_edge_litert). The file's
  SHA-256 is checked on load. A missing runtime is an error, never a silent switch to the stdlib scorer.
- python: the standard-library scorer below, for bundles without TFLite or when asked explicitly (backend='python').

Supported model types (exported by pipeline_v3/portable.py from scikit-learn):
- forest:   random forest; probability = mean over trees of the leaf's fall fraction
- boosting: gradient boosted trees; probability = sigmoid(init + learning_rate * sum(leaf values))
- logistic: logistic regression on standardized log features
Trees compare float32(x) <= threshold, exactly as scikit-learn does internally.
"""
import hashlib
import json
import math
import os
import struct

FORMAT = 'blissey_edge_model_v1'
_F32 = struct.Struct('f')


def _f32(v):
    return _F32.unpack(_F32.pack(v))[0]


def _leaf(tree, x):
    left, right, feat, thr = tree['left'], tree['right'], tree['feature'], tree['threshold']
    node = 0
    while left[node] != -1:
        node = left[node] if x[feat[node]] <= thr[node] else right[node]
    return tree['value'][node]


def _interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            raise ImportError('model.json declares a TFLite model but neither tflite_runtime nor ai_edge_litert is '
                              'installed (or run with backend python)')
    return Interpreter


class _TFLite:
    def __init__(self, spec, base_dir, n_features):
        import numpy as np
        with open(os.path.join(base_dir, spec['file']), 'rb') as f:
            blob = f.read()
        if hashlib.sha256(blob).hexdigest() != spec['sha256']:
            raise ValueError('%s does not match the SHA-256 in model.json' % spec['file'])
        Interpreter = _interpreter_class()
        try:
            self.interp = Interpreter(model_content=blob, num_threads=1)
        except TypeError:                     # older tflite_runtime without num_threads
            self.interp = Interpreter(model_content=blob)
        self.interp.allocate_tensors()
        inp, out = self.interp.get_input_details()[0], self.interp.get_output_details()[0]
        if list(inp['shape']) != [1, n_features] or inp['dtype'] != np.float32:
            raise ValueError('model.tflite input is %s %s, expected [1, %d] float32' % (list(inp['shape']), inp['dtype'], n_features))
        self.inp, self.out = inp['index'], out['index']
        self.x = np.zeros((1, n_features), dtype=np.float32)

    def score(self, x):
        self.x[0, :] = x
        self.interp.set_tensor(self.inp, self.x)
        self.interp.invoke()
        return float(self.interp.get_tensor(self.out)[0, 0])


class Model:
    def __init__(self, bundle, base_dir='.', backend=None):
        if bundle.get('format') != FORMAT:
            raise ValueError('Unsupported model format: %r' % bundle.get('format'))
        self.bundle = bundle
        self.spec = bundle['model']
        self.kind = self.spec['type']
        self.n_features = bundle['n_features']
        if self.kind not in ('forest', 'boosting', 'logistic'):
            raise ValueError('Unsupported model type: %r' % self.kind)
        self.backend = backend or ('tflite' if 'tflite' in bundle else 'python')
        if self.backend not in ('tflite', 'python'):
            raise ValueError('Unknown backend: %r' % self.backend)
        if self.backend == 'tflite' and 'tflite' not in bundle:
            raise ValueError('backend tflite requested but model.json has no "tflite" entry (run export_tflite.py)')
        self._tflite = _TFLite(bundle['tflite'], base_dir, self.n_features) if self.backend == 'tflite' else None
        # float32 interpreter vs float64 training scores: allowed absolute difference in golden checks
        self.score_tolerance = bundle['tflite']['tolerance'] if self._tflite else 0.0

    @classmethod
    def load(cls, path, backend=None):
        with open(path) as f:
            return cls(json.load(f), os.path.dirname(os.path.abspath(path)), backend)

    def fall_score(self, x):
        """x: feature list -> uncalibrated fall score in [0, 1]."""
        if len(x) != self.n_features:
            raise ValueError('Expected %d features, got %d' % (self.n_features, len(x)))
        if self._tflite:
            return self._tflite.score(x)
        if self.kind == 'logistic':
            s = self.spec
            z = s['intercept']
            for v, mu, sd, w in zip(x, s['mean'], s['scale'], s['coef']):
                z += w * ((math.log(max(v, s['log_floor'])) if s['log'] else v) - mu) / sd
            return 1.0 / (1.0 + math.exp(-z)) if z >= -700 else 0.0
        x32 = [_f32(v) for v in x]
        if self.kind == 'forest':
            trees = self.spec['trees']
            return sum(_leaf(t, x32) for t in trees) / len(trees)
        z = self.spec['init'] + self.spec['learning_rate'] * sum(_leaf(t, x32) for t in self.spec['trees'])
        return 1.0 / (1.0 + math.exp(-z)) if z >= -700 else 0.0
