"""Versioned CSI acquisition gates; never infer a rate from short burst gaps."""
import numpy as np


def window_quality(times, start, config):
    """Return metrics/reasons on unique timestamps, including interpolation brackets.

    v1 remains supported for reconstructing archived artifacts/checkpoints.
    v2 has fixed, development-justified thresholds. No per-test fitting occurs.
    """
    times=np.asarray(times,dtype=np.float64)
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Need finite strictly increasing packet times; deduplicate before assessing quality')
    duration=float(config['window_seconds']);end=float(start)+duration
    if not np.isfinite(start) or not np.isfinite(duration) or duration<=0:
        raise ValueError('Invalid window bounds')
    if start<times[0] or end>times[-1]:raise ValueError('Window lacks endpoint coverage')
    left=max(0,int(np.searchsorted(times,start,side='right'))-1)
    right=min(len(times),int(np.searchsorted(times,end,side='left'))+1)
    bracket=times[left:right];gaps=np.diff(bracket)
    inside=times[(times>=start)&(times<=end)]
    maximum=float(np.max(gaps));quality=config.get('packet_quality',{})
    if quality.get('version',1)==1:
        rate=1/float(np.median(gaps))
        bad=maximum>config['max_gap_seconds'] or len(bracket)<max(16,.5*rate*duration)
        return dict(left=left,right=right,max_gap_seconds=maximum,
            effective_rate_hz=len(inside)/duration,
            reasons=['Insufficient packets or excessive gap'] if bad else [])
    if quality['version']!=2:raise ValueError('Unsupported packet quality version')
    n_bins=int(np.ceil(duration/quality['coverage_bin_seconds']-1e-10))
    if len(inside):
        bins=np.clip(np.floor((inside-start+1e-10)/quality['coverage_bin_seconds']).astype(int),0,n_bins-1)
        occupied=len(np.unique(bins))/n_bins
        span=float((inside[-1]-inside[0])/duration)
        edge=max(float(inside[0]-start),float(end-inside[-1]))
    else:occupied=span=0.;edge=duration
    rate=len(inside)/duration
    metrics=dict(left=left,right=right,unique_packets=len(inside),effective_rate_hz=rate,
        occupied_bin_fraction=occupied,sample_span_fraction=span,
        max_edge_distance_seconds=edge,max_gap_seconds=maximum)
    # Every test is independent: bursts cannot compensate for missing wall time.
    checks=(('insufficient_effective_rate',rate+1e-9<quality['min_effective_rate_hz']),
        ('insufficient_temporal_coverage',occupied+1e-9<quality['min_occupied_bin_fraction']),
        ('insufficient_sample_span',span+1e-9<quality['min_sample_span_fraction']),
        ('unsupported_window_edge',edge>quality['max_edge_distance_seconds']+1e-9),
        ('excessive_packet_gap',maximum>quality['max_gap_seconds']+1e-9))
    metrics['reasons']=[name for name,failed in checks if failed]
    return metrics
