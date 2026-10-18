"""Label-blind calibration: fixed development groups only; never reads test/events."""
import json,sys
from pathlib import Path
from collections import Counter
import numpy as np
MODEL_ROOT=Path(__file__).resolve().parent
from csi import packets
result=[]
for group in ('session_01','session_02'):
 for p in sorted((MODEL_ROOT.parent/'blissey-data/recordings'/group).rglob('measurements.jsonl')):
  rows=[r for _,r,e in packets(p) if not e]
  if len({r['source_mac'] for r in rows})!=1:continue
  host=np.array([float(r['elapsed_seconds']) for r in rows]);radio=np.array([int(r['firmware_fields'][18]) for r in rows],dtype=np.int64)/1e6
  fw=np.array([float(r['firmware_fields'][23]) for r in rows])
  ht=np.unique(host);windows=[]
  for start in np.arange(ht[0],ht[-1]-3+1e-9,.5):
   inside=ht[(ht>=start)&(ht<=start+3)]
   gaps=np.diff(ht[max(0,np.searchsorted(ht,start,side='right')-1):min(len(ht),np.searchsorted(ht,start+3)+1)])
   bins=np.clip(np.floor((inside-start)/.1).astype(int),0,29)
   windows.append([len(inside)/3,(inside[-1]-inside[0])/3,len(np.unique(bins))/30,float(max(gaps))])
  result.append(dict(recording_id=p.parent.name,group=group,packets=len(rows),host_rate=(len(ht)-1)/(ht[-1]-ht[0]),median_gap_rate=1/np.median(np.diff(ht)),
   radio_nonpositive=int((np.diff(radio)<=0).sum()),host_duplicates=len(host)-len(ht),radio_span=float(radio[-1]-radio[0]),host_span=float(host[-1]-host[0]),
   radio_host_offset_range_s=float(np.ptp(host-radio)),radio_host_offset_p95_p05_s=float(np.quantile(host-radio,.95)-np.quantile(host-radio,.05)),
   max_host_gap=float(np.max(np.diff(host))),max_radio_gap=float(np.max(np.diff(radio))),
   firmware_clock_monotonic=bool(np.all(np.diff(fw)>0)),window_diagnostics=windows))
rates=[r['host_rate'] for r in result]
print('Recordings',len(result),'rates q',np.quantile(rates,[0,.1,.5,.9,1]).tolist())
for group in ('session_01','session_02'):
 rr=[r for r in result if r['group']==group];w=np.concatenate([r['window_diagnostics'] for r in rr])
 print(group,'window metrics rate/span/occupied_bins/gap quantiles',np.quantile(w,[0,.1,.5,.9,1],axis=0).tolist())
 print('clock diagnostics',dict(host_duplicates=sum(r['host_duplicates'] for r in rr),radio_nonpositive=sum(r['radio_nonpositive'] for r in rr),offset95range=np.quantile([r['radio_host_offset_p95_p05_s'] for r in rr],[0,.5,1]).tolist(),span_diff_range=np.quantile([abs(r['host_span']-r['radio_span']) for r in rr],[0,.5,1]).tolist()))
(MODEL_ROOT/'reports/development_acquisition_characterization.json').write_text(json.dumps(dict(groups_used=['session_01','session_02'],test_read=False,recordings=result),indent=2)+'\n')
