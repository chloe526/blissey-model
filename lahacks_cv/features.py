"""Stateless window-local RF features; no globally fitted PCA or scaler."""
import numpy as np

FEATURE_CONFIG = {
    'version': 'median_movement_spectrum_pca_v1', 'input_shape': [52, 512],
    'window_seconds': 3., 'median_floor': 1e-6, 'ratio_floor': 1e-8,
    'ratio_clip': 100., 'bands_hz': [[0., 1.], [1., 3.], [3., 10.]],
    'pca': 'top 3 channel covariance eigenvalue fractions, fitted within each window only',
    'normalization': 'each channel divided by its own window median, minus one',
    'spectrum': 'Hann taper; fractions of non-DC power up to 10Hz; interpolated signal, not native 170Hz acquisition',
}


def extract_one(x):
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (52, 512) or not np.isfinite(x).all():
        raise ValueError('RF expects finite [52,512] amplitude window')
    x = x / np.maximum(np.median(np.abs(x), axis=1, keepdims=True), 1e-6) - 1.
    diff = np.diff(x, axis=1)
    thirds = np.array_split(x, 3, axis=1)
    variances = [s.var(axis=1) for s in thirds]
    movement = np.square(diff)
    ratio = np.minimum(movement.max(axis=1) / np.maximum(movement[:, -170:].mean(axis=1), 1e-8), 100.)
    centered = x - x.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered * np.hanning(512), axis=1)) ** 2
    frequency = np.fft.rfftfreq(512, d=3. / 511.)
    denominator = np.maximum(power[:, (frequency > 0) & (frequency <= 10)].sum(axis=1), 1e-12)
    bands = [power[:, (frequency > lo) & (frequency <= hi)].sum(axis=1) / denominator
             for lo, hi in FEATURE_CONFIG['bands_hz']]
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered @ centered.T / 511.), 0.)
    pca = eigenvalues[-3:][::-1] / max(eigenvalues.sum(), 1e-12)
    # 11 per-channel summaries + 3 window-local PCA summaries = 575 features.
    result = np.concatenate([x.var(axis=1), np.abs(diff).mean(axis=1), diff.var(axis=1),
                             *variances, ratio, np.square(x).mean(axis=1), *bands, pca]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite RF feature')
    return result


def extract(x):
    return np.stack([extract_one(window) for window in x])
