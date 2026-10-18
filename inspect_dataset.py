"""Audit every recording without changing the source or requiring ML dependencies."""
import argparse
import collections
import json
from pathlib import Path
from statistics import median
from csi import load_recording, seconds


def audit(root):
    root = Path(root).resolve()
    recordings = []
    for path in sorted(root.rglob('metadata.json')):
        if '.git' in path.parts or not path.with_name('measurements.jsonl').exists():
            continue
        m, rows, label, intervals, issues, errors = load_recording(path.parent)
        times = [float(p['elapsed_seconds']) for p in rows]
        diffs = [b-a for a,b in zip(times,times[1:])]
        positive = [d for d in diffs if d > 0]
        nonzero = [i for i in range(2,64) if any(p['csi'][2*i] or p['csi'][2*i+1] for p in rows)]
        host_offsets = [seconds(p['host_time'])-p['elapsed_seconds'] for p in rows]
        rel = path.parent.relative_to(root)
        # Physical collection folder takes precedence over inconsistent free-text session metadata.
        group = rel.parts[0] if len(rel.parts) >= 3 else m.get('session')
        sources = len({p.get('source_mac') for p in rows})
        if sources != 1:
            label = 'ambiguous'; issues.append('Multiple/missing radio sources')
        radio = sorted({(p['firmware_fields'][5],p['firmware_fields'][7],p['firmware_fields'][16],p['firmware_fields'][17]) for p in rows})
        recordings.append(dict(recording_id=m['recording_id'], path=str(rel), group=group,
            session=m.get('session'), participant=m.get('participant'), activity=m['activity'],
            label_status=label, fall_intervals=intervals, issues=issues,
            valid_packets=len(rows), malformed_packets=len(errors), error_examples=errors[:5],
            start_seconds=min(times) if times else None, end_seconds=max(times) if times else None,
            exposure_seconds=max(times)-min(times) if times else 0,
            measured_rate_hz=(len(set(times))-1)/(max(times)-min(times)) if positive else None,
            median_instantaneous_rate_hz=1/median(positive) if positive else None,
            max_gap_seconds=max(positive) if positive else None,
            duplicate_timestamps=len(times)-len(set(times)), out_of_order=sum(d<0 for d in diffs),
            host_clock_offset_range_seconds=max(host_offsets)-min(host_offsets) if rows else None,
            csi_lengths=dict(collections.Counter(len(p['csi']) for p in rows)),
            observed_nonzero_pair_indices=nonzero, radio_modes=radio,
            sources=sources, metadata_saved_measurements=m.get('saved_measurements')))
    labels = collections.Counter(r['label_status'] for r in recordings)
    common = sorted(set.intersection(*(set(r['observed_nonzero_pair_indices']) for r in recordings if r['valid_packets']))) if recordings else []
    inventory=collections.Counter(p.name for p in root.rglob('*') if p.is_file() and '.git' not in p.parts)
    orphan_measurements=[str(p.relative_to(root)) for p in root.rglob('measurements.jsonl') if not p.with_name('metadata.json').exists()]
    return dict(file_inventory=dict(inventory),orphan_measurements=orphan_measurements,
        explicit_category_non_fall_count=sum(r['activity']!='staged_fall' for r in recordings),
        schema_version=1, source='blissey-data (read only)', total_recordings=len(recordings),
        groups=sorted({r['group'] for r in recordings}), sessions=sorted({r['session'] for r in recordings}),
        participants=sorted({r['participant'] for r in recordings}),
        activity_counts=dict(collections.Counter(r['activity'] for r in recordings)),
        label_counts=dict(labels), common_valid_pair_indices=common, available_complex_pairs=64,
        encoding='128 signed raw int8 values: imaginary, real pairs; first 4 bytes excluded',
        filtering='Firmware ltf_merge_en=1, channel_filter_en=0, manu_scale=0; no host normalization found',
        timestamp_policy='Measurement elapsed seconds; UI events mapped by host UTC minus median measurement host/elapsed origin',
        supervised_ready=labels['annotated_fall']>0 and labels['verified_non_fall']>0,
        limitations=['Staged falls only; two participants; operator action intervals are approximate.',
            'Three physical collection groups; held-session experiment cannot establish population generalization.',
            'Unknown/malformed trials are excluded, never assigned negative labels.'], recordings=recordings)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',default='../blissey-data/recordings')
    p.add_argument('--output',default='reports/dataset_audit.json'); args=p.parse_args()
    result=audit(args.data_root)
    if not result['total_recordings']: raise SystemExit('No recordings found; check --data-root')
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n')
    rates=[r['measured_rate_hz'] for r in result['recordings'] if r['measured_rate_hz']]
    print(f"Recordings: {result['total_recordings']}; physical groups: {len(result['groups'])}; participants: {result['participants']}")
    print(f"Labels: {result['label_counts']}; usable channels: {len(result['common_valid_pair_indices'])}/64")
    print(f'Packet rate min/median/max: {min(rates):.2f}/{median(rates):.2f}/{max(rates):.2f} Hz')
    print('Supervised staged-fall experiment:', 'READY (limitations apply)' if result['supervised_ready'] else 'BLOCKED; verified intervals/categories required')
    print('Audit:',target)

if __name__=='__main__': main()
