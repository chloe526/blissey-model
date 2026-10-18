"""Window quality, interpolation, gain normalization and portable features (pv1). Stdlib only.

Quality gate: a port of the frozen quality-v2 rules (csi_quality.window_quality) plus the
gap-tolerant gate (longest gap <= 0.5 s, unsupported edge <= 0.25 s) used for live inference.

pv1 features (424 values) from a 3 s window on a 150-point (50 Hz) grid:
- gain normalization: divide each time sample by its across-subcarrier mean (removes common
  AGC gain), then each subcarrier by its window median (removes absolute level). r = g - 1.
- per subcarrier (52 x 8): variance of r; mean |diff| of the 7-point smoothed signal, overall and per
  third; variance of the smoothed signal (slow motion); variance of the residual r - smooth
  (fast motion); burst ratio = max 0.5 s mean |diff| / last-third mean |diff| (clipped at 100).
- global (8): channel-mean movement curve: mean, per third, 0.5 s burst max, tail/burst ratio;
  median subcarrier std of the smoothed signal; common-motion share (variance of the subcarrier-mean
  smoothed signal over mean subcarrier variance).
"""
from bisect import bisect_left, bisect_right
import math

WINDOW_SECONDS = 3.0
GRID = 150
SMOOTH_HALF = 3          # 7-point moving average (~140 ms at 50 Hz)
BURST = 25               # 0.5 s at 50 Hz
V2 = dict(min_rate_hz=20.0, bin_seconds=0.1, min_occupied=0.8, min_span=0.9, max_edge=0.1, max_gap=0.25)
TOLERANT = dict(max_gap=0.5, max_edge=0.25)
N_PER_CHANNEL = 8
N_GLOBAL = 8


def quality(times, start, duration=WINDOW_SECONDS):
    """Same metrics as csi_quality.window_quality (v2). times: sorted, strictly increasing list."""
    end = start + duration
    if start < times[0] or end > times[-1]:
        raise ValueError('Window lacks endpoint coverage')
    left = max(0, bisect_right(times, start) - 1)
    right = min(len(times), bisect_left(times, end) + 1)
    maximum = max(times[i + 1] - times[i] for i in range(left, right - 1))
    lo, hi = bisect_left(times, start), bisect_right(times, end)
    inside = times[lo:hi]
    n_bins = int(math.ceil(duration / V2['bin_seconds'] - 1e-10))
    if inside:
        bins = {min(max(int(math.floor((t - start + 1e-10) / V2['bin_seconds'])), 0), n_bins - 1) for t in inside}
        occupied = len(bins) / n_bins
        span = (inside[-1] - inside[0]) / duration
        edge = max(inside[0] - start, end - inside[-1])
    else:
        occupied = span = 0.0
        edge = duration
    rate = len(inside) / duration
    reasons = []
    if rate + 1e-9 < V2['min_rate_hz']:
        reasons.append('insufficient_effective_rate')
    if occupied + 1e-9 < V2['min_occupied']:
        reasons.append('insufficient_temporal_coverage')
    if span + 1e-9 < V2['min_span']:
        reasons.append('insufficient_sample_span')
    if edge > V2['max_edge'] + 1e-9:
        reasons.append('unsupported_window_edge')
    if maximum > V2['max_gap'] + 1e-9:
        reasons.append('excessive_packet_gap')
    return dict(left=left, right=right, max_gap_seconds=maximum, max_edge_distance_seconds=edge,
                occupied_bin_fraction=occupied, sample_span_fraction=span, effective_rate_hz=rate,
                reasons=reasons, pass_v2=not reasons,
                pass_tolerant=maximum <= TOLERANT['max_gap'] + 1e-9 and edge <= TOLERANT['max_edge'] + 1e-9)


def interpolate(times, amps, start, left, right, grid=GRID, duration=WINDOW_SECONDS):
    """Linear interpolation (np.interp semantics) of packets[left:right] onto `grid` points.
    Returns 52 lists (one per subcarrier) of length `grid`."""
    n_ch = len(amps[left])
    out = [[0.0] * grid for _ in range(n_ch)]
    step = duration / (grid - 1)
    j = left
    last = right - 1
    for k in range(grid):
        g = start + k * step
        while j < last - 1 and times[j + 1] <= g:
            j += 1
        t0, t1 = times[j], times[j + 1]
        if g <= t0:
            row = amps[j]
            for c in range(n_ch):
                out[c][k] = row[c]
        elif g >= t1:
            row = amps[j + 1]
            for c in range(n_ch):
                out[c][k] = row[c]
        else:
            w = (g - t0) / (t1 - t0)
            a, b = amps[j], amps[j + 1]
            for c in range(n_ch):
                out[c][k] = a[c] + w * (b[c] - a[c])
    return out


def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _var(values):
    n = len(values)
    m = sum(values) / n
    return sum((v - m) * (v - m) for v in values) / n


def _thirds(values):
    """Split like numpy.array_split(values, 3)."""
    n = len(values)
    q, r = divmod(n, 3)
    sizes = [q + (1 if i < r else 0) for i in range(3)]
    out, i = [], 0
    for s in sizes:
        out.append(values[i:i + s])
        i += s
    return out


def _burst_max(d):
    best = run = sum(d[:BURST])
    for i in range(BURST, len(d)):
        run += d[i] - d[i - BURST]
        if run > best:
            best = run
    return best / BURST


def features(x):
    """x: 52 lists of interpolated raw amplitudes (length GRID) -> pv1 feature list (476 floats)."""
    n_ch, n = len(x), len(x[0])
    # 1) common-mode (per time sample) gain removal
    common = [0.0] * n
    for k in range(n):
        s = 0.0
        for c in range(n_ch):
            s += x[c][k]
        common[k] = max(s / n_ch, 1e-6)
    per_channel, d_curves, sm_all, sm_std = [], [], [], []
    for c in range(n_ch):
        row = [x[c][k] / common[k] for k in range(n)]
        med = max(_median(row), 1e-6)
        r = [v / med - 1.0 for v in row]
        # 2) 7-point centred moving average, valid region only
        w = 2 * SMOOTH_HALF + 1
        run = sum(r[:w])
        sm = [run / w]
        for k in range(w, n):
            run += r[k] - r[k - w]
            sm.append(run / w)
        res = [r[k + SMOOTH_HALF] - sm[k] for k in range(len(sm))]
        d = [abs(sm[k + 1] - sm[k]) for k in range(len(sm) - 1)]
        t1, t2, t3 = _thirds(d)
        mean_d = sum(d) / len(d)
        tail = sum(t3) / len(t3)
        burst = _burst_max(d)
        var_sm = _var(sm)
        per_channel.append((_var(r), mean_d, sum(t1) / len(t1), sum(t2) / len(t2), tail,
                            var_sm, _var(res), min(burst / max(tail, 1e-8), 100.0)))
        d_curves.append(d)
        sm_all.append(sm)
        sm_std.append(math.sqrt(var_sm))
    # fixed order: 8 blocks of 52 per-subcarrier values, then 8 global values
    out = []
    for f in range(N_PER_CHANNEL):
        out.extend(pc[f] for pc in per_channel)
    # 3) global features
    m = len(d_curves[0])
    D = [sum(d_curves[c][k] for c in range(n_ch)) / n_ch for k in range(m)]
    g1, g2, g3 = _thirds(D)
    burst = _burst_max(D)
    tail = sum(g3) / len(g3)
    msm = [sum(sm_all[c][k] for c in range(n_ch)) / n_ch for k in range(len(sm_all[0]))]
    mean_var = sum(s * s for s in sm_std) / n_ch
    out.extend([sum(D) / m, sum(g1) / len(g1), sum(g2) / len(g2), tail, burst, tail / max(burst, 1e-12),
                _median(sm_std), _var(msm) / max(mean_var, 1e-12)])
    return out


def window_features(times, amps, start, gate='tolerant'):
    """Full path for one window: quality gate -> interpolation -> pv1 features.
    Returns (features or None, quality dict). gate: 'tolerant' (live) or 'v2'."""
    q = quality(times, start)
    ok = q['pass_tolerant'] if gate == 'tolerant' else q['pass_v2']
    if not ok:
        return None, q
    return features(interpolate(times, amps, start, q['left'], q['right'])), q


N_FEATURES = 52 * N_PER_CHANNEL + N_GLOBAL
