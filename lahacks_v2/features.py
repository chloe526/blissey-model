"""Compact, window-local motion features on gain-normalized windows (stateless; offline == live)."""
import numpy as np

SMOOTH = 25          # ~150 ms moving average on the 512-point / 3 s grid
TRIM = 20            # drop smoothing edge effects
BURST = 85           # ~0.5 s
NAMES = ('absdiff', 'smooth_std', 'absdiff_t1', 'absdiff_t2', 'absdiff_t3',
         'burst_max', 'tail_over_burst', 'pc1_fraction', 'interp_free_std')


def _smooth(r):
    c = np.cumsum(np.pad(r, [(0, 0)] * (r.ndim - 1) + [(SMOOTH, 0)]), axis=-1, dtype=np.float64)
    return ((c[..., SMOOTH:] - c[..., :-SMOOTH]) / SMOOTH)[..., TRIM:-TRIM]


def motion_score(g):
    """E1 detector score: mean |first difference| of the smoothed gain-normalized window."""
    sm = _smooth(np.asarray(g, np.float64) - 1)
    return np.abs(np.diff(sm, axis=-1)).mean(axis=(-2, -1))


def extract(G, batch=256):
    """G: gain-normalized [N,52,512] -> [N, len(NAMES)] float32."""
    out = []
    for i in range(0, len(G), batch):
        sm = _smooth(np.asarray(G[i:i + batch], np.float64) - 1)
        d = np.abs(np.diff(sm, axis=-1)).mean(axis=1)                # [n, T] channel-mean movement
        thirds = np.array_split(d, 3, axis=-1)
        c = np.cumsum(np.pad(d, [(0, 0), (BURST, 0)]), axis=-1)
        burst = ((c[:, BURST:] - c[:, :-BURST]) / BURST).max(axis=-1)
        centred = sm - sm.mean(axis=-1, keepdims=True)
        cov = np.einsum('nct,ndt->ncd', centred, centred)
        eig = np.linalg.eigvalsh(cov)
        f = np.stack([d.mean(-1), sm.std(-1).mean(-1), *[t.mean(-1) for t in thirds], burst,
                      thirds[2].mean(-1) / np.maximum(burst, 1e-9), eig[:, -1] / np.maximum(eig.sum(-1), 1e-12),
                      np.median(sm.std(-1), axis=-1)], axis=1)
        out.append(f)
    f = np.concatenate(out)
    scale = [0, 1, 2, 3, 4, 5, 8]                                   # log-transform scale features
    f[:, scale] = np.log(np.maximum(f[:, scale], 1e-9))
    return f.astype(np.float32)


def with_context(F, data):
    """Concatenate features of the current window and the two previous windows (causal).
    A missing previous window (quality gap) repeats the nearest available one."""
    key = {(r, round(float(e), 3)): i for i, (r, e) in enumerate(zip(data['recording_id'], data['end']))}
    p1 = np.array([key.get((r, round(float(e) - .5, 3)), i)
                   for i, (r, e) in enumerate(zip(data['recording_id'], data['end']))])
    return np.concatenate([F, F[p1], F[p1[p1]]], axis=1)
