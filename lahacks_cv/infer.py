"""CPU window inference for NEW fold artifacts, not a reviewed final deployment model."""
import json
from pathlib import Path

import numpy as np
from preprocess import transform_window
from .common import normalize
from .features import FEATURE_CONFIG, extract_one


def live_window(times, amplitude, start, geometry):
    """Exactly the same raw-amplitude interpolation/quality path as offline data."""
    return transform_window(times, amplitude, start, geometry, normalize=False)


def load_fold(folder, model_type):
    folder = Path(folder)
    config = json.loads((folder / 'preprocessing.json').read_text())
    if model_type == 'cnn':
        import torch
        from cnn_baseline.model import build_model
        saved = torch.load(folder / 'cnn.pt', map_location='cpu', weights_only=True)
        if saved['normalization'] != config['normalization']:
            raise ValueError('CNN normalization mismatch')
        model = build_model(saved['architecture']); model.load_state_dict(saved['state_dict']); model.eval()
        def probability(window):
            with torch.inference_mode():
                x = torch.from_numpy(normalize(window, saved['normalization'])).unsqueeze(0)
                return float(model(x).softmax(-1)[0, 1])
    elif model_type == 'rf':
        import joblib
        saved = joblib.load(folder / 'rf.joblib')
        if saved['features'] != FEATURE_CONFIG or config['feature_config'] != FEATURE_CONFIG:
            raise ValueError('RF feature version mismatch')
        def probability(window):
            return float(saved['model'].predict_proba(extract_one(window)[None])[0, 1])
    else:
        raise ValueError('Choose cnn or rf')
    return probability, config
