"""Describe paired response and coverage; scene labels are not contact truth."""
import json
import numpy as np
from fusion_refinement_cache_dev_20261013 import OUT
from fusion_temporal_candidates_dev_20261013 import run_candidates

def metrics(frames, out):
    anyq=out.any(axis=1); n=len(frames)
    tt=np.array([r['t_ns'] for r in frames],dtype=np.int64)/1e9
    widths=np.r_[np.minimum(np.maximum(np.diff(tt),0),.5), .2] if n else np.zeros(0)
    runs=[]; current=0.
    for i,v in enumerate(anyq):
        if v:
            if i and tt[i]-tt[i-1]>.5:
                runs.append(current); current=0.
            current+=float(widths[i])
        elif current: runs.append(current); current=0.
    if current:runs.append(current)
    return dict(n=n, H=int(out[:,0].sum()),B=int(out[:,1].sum()),any=int(anyq.sum()),
                end_n=min(15,n),end_any=int(anyq[-15:].sum()),end_H=int(out[-15:,0].sum()),
                first_s=float(tt[np.flatnonzero(anyq)[0]]-tt[0]) if anyq.any() else None,
                alert_s=float(widths[anyq].sum()),longest_s=max(runs,default=0.),episodes=len(runs),
                rgb_available=sum(r['available'] for r in frames),
                pose_invalid=sum(not r['pose_ok'] for r in frames),
                matched=sum(r['matched'] for r in frames))

def main(input_name='readout.json', output_name='comparison.json'):
    data=json.loads((OUT/input_name).read_text()); report=dict(sessions={})
    for key,s in data['sessions'].items():
        arms={}; preds={}
        for anchor,fr in s['arms'].items():
            outputs=run_candidates(fr)
            for temporal,o in outputs.items():
                name=anchor+'/'+temporal
                preds[name]=o; arms[name]=metrics(fr,o)
        base=preds['baseline/baseline_2of3']
        basefr=s['arms']['baseline']
        for name,o in preds.items():
            anchor=name.split('/')[0]; fr=s['arms'][anchor]
            common=np.array([a['available'] and b['available'] and a['pose_ok'] for a,b in zip(basefr,fr)])
            m=arms[name]
            m['response_added_query_frames']=int((o&~base).sum())
            m['response_removed_query_frames']=int((base&~o).sum())
            m['common_covered_removed_query_frames']=int(((base&~o)&common[:,None]).sum())
            m['common_covered_added_query_frames']=int(((o&~base)&common[:,None]).sum())
            m['coverage_lost_frames']=sum(a['available'] and not b['available'] for a,b in zip(basefr,fr))
            m['first_shift_s']=m['first_s']-arms['baseline/baseline_2of3']['first_s'] if m['first_s'] is not None and arms['baseline/baseline_2of3']['first_s'] is not None else None
            c=s['label']['class']
            m['descriptive_pass']=(m['any']/m['n']<=.10 if c=='quiet' else m['any']/m['n']<=.30 if c=='side' else m['end_any']/m['end_n']>=.50 if c=='ahead' else m['end_H']/m['end_n']>=.50 if c=='head' else None)
        report['sessions'][key]=dict(label=s['label'],arms=arms)
    (OUT/output_name).write_text(json.dumps(report,indent=1),encoding='utf8')
    names=list(next(iter(report['sessions'].values()))['arms'])
    for name in names:
        print('\n'+name)
        for key,s in report['sessions'].items():
            m=s['arms'][name]
            if key.startswith('073') or key in ('070601','072853','072939'):
                print(key,s['label']['class'],f"any {m['any']}/{m['n']} end {m['end_any']}/{m['end_n']} H-end {m['end_H']} RGBcov {m['rgb_available']} +/- {m['response_added_query_frames']}/{m['response_removed_query_frames']} shift {m['first_shift_s']}")

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='readout.json'); ap.add_argument('--output',default='comparison.json')
    a=ap.parse_args();main(a.input,a.output)
