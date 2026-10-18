"""Read-only: packet types, device-vs-host gaps, per-type amplitude scale, for S03/S04 raw cache."""
import json, sys, collections
from pathlib import Path
import numpy as np
RAW = Path(sys.argv[1] if len(sys.argv) > 1 else '/private/tmp/blissey-stage1-06c97032c93a')
IDX = list(range(6, 32)) + list(range(33, 59))
rows = []; types_all = collections.Counter(); amp_by_type = collections.defaultdict(list)
for g in ('session_03_LAHACKS', 'session_04_LAHACKS'):
    for f in sorted((RAW / g).glob('*/*/measurements.jsonl')):
        P = [json.loads(l) for l in f.open()]
        ff = [p['firmware_fields'] for p in P]
        dev = np.array([int(x[18]) for x in ff]) / 1e6; host = np.array([p['elapsed_seconds'] for p in P])
        typ = [(x[5], x[7], x[20]) for x in ff]          # sig_mode, bandwidth, sig_len
        c = collections.Counter(typ); types_all.update(c)
        a = np.array([np.hypot(*np.array(p['csi'], float).reshape(64, 2).T)[IDX].mean() for p in P])
        for t, v in zip(typ, a): amp_by_type[t].append(v)
        dd = np.diff(dev); hd = np.diff(host)
        top = c.most_common(1)[0]
        rows.append(dict(group=g[:10], activity=f.parent.parent.name, n=len(P), dur=host[-1] - host[0],
            rate=len(P) / (host[-1] - host[0]), dev_gaps250=int((dd > .25).sum()), host_gaps250=int((hd > .25).sum()),
            dev_gap_time=float(dd[dd > .25].sum()), max_dev_gap=float(dd.max()),
            n_types=len(c), top_type=str(top[0]), top_frac=top[1] / len(P),
            rssi=float(np.median([int(x[3]) for x in ff])), noise=float(np.median([int(x[14]) for x in ff])),
            amp_cv_between_types=float(np.std([np.median(a[[t == k for t in typ]]) for k in c]) / np.median(a)) if len(c) > 1 else 0.,
            frac_beacon_like=sum(v for k, v in c.items() if k[2] in ('0',)) / len(P)))
print('ALL packet types (sig_mode,bandwidth,sig_len) top 12:')
tot = sum(types_all.values())
for k, v in types_all.most_common(12):
    print('  ', k, v, f'{v/tot:.1%}', 'median amp %.1f' % np.median(amp_by_type[k]))
import statistics as st
for g in ('session_03', 'session_04'):
    R = [r for r in rows if r['group'] == g]
    print(g, 'recordings', len(R), 'median rate %.1f' % st.median(r['rate'] for r in R),
          'dev gaps>250ms/rec %.1f' % st.mean(r['dev_gaps250'] for r in R), 'host %.1f' % st.mean(r['host_gaps250'] for r in R),
          'gap time frac %.1f%%' % (100 * sum(r['dev_gap_time'] for r in R) / sum(r['dur'] for r in R)),
          'rssi %.0f' % st.median(r['rssi'] for r in R))
r_ = np.array([r['rate'] for r in rows]); gaps = np.array([r['dev_gap_time'] / r['dur'] for r in rows]); rs = np.array([r['rssi'] for r in rows])
print('corr(rssi, gap-time frac) %.2f; corr(rate, gap frac) %.2f' % (np.corrcoef(rs, gaps)[0, 1], np.corrcoef(r_, gaps)[0, 1]))
json.dump(rows, open(Path(__file__).parent / 'packet_audit.json', 'w'), indent=1)
