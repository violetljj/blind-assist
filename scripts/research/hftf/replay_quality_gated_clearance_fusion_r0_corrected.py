#!/usr/bin/env python3
"""Single deterministic A/B replay over a precomputed raw/truth stream."""
from __future__ import annotations
import argparse, hashlib, json, math, os
from collections import defaultdict
from pathlib import Path
from quality_gated_clearance_fusion_r0 import Evidence, Filter

SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_corrected_replay_result"

def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest().upper()

def mae(values):
    return sum(values)/len(values) if values else None

def replay(stream_path: Path):
    rows=[json.loads(x) for x in stream_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    by_parent=defaultdict(list)
    for r in rows: by_parent[str(r['parent_id'])].append(r)
    result={}
    for parent, items in sorted(by_parent.items()):
        items.sort(key=lambda r:int(r['timestamp_ns']))
        filt=Filter(); arms={'raw_per_frame_geometry':[], 'quality_gated_fusion':[]}
        for r in items:
            raw=tuple(r['raw_clearance_m'])
            truth=tuple(r['truth_clearance_m'])
            raw_state=tuple(r['raw_geometry_state'])
            for band in range(3):
                if r['raw_geometry_valid'][band] and r['truth_geometry_valid'][band]:
                    arms['raw_per_frame_geometry'].append((band,raw[band],truth[band],raw_state[band]))
            out=filt.update(Evidence(int(r['timestamp_ns']),raw,tuple(r['raw_geometry_valid']),bool(r['tof_valid']),float(r['teacher_age_s']),float(r['frozen_a2_disagreement'])))
            for band in range(3):
                if out.clearance_m[band] is not None and r['truth_geometry_valid'][band]:
                    arms['quality_gated_fusion'].append((band,out.clearance_m[band],truth[band],out.state[band].value))
        metrics={}
        for name, vals in arms.items():
            errors=[abs(float(v)-float(t)) for _,v,t,_ in vals]
            transitions=[]; prev_truth=None; prev_pred=None
            for _,v,t,s in vals:
                truth_state='OCCUPIED' if float(t)<=1.5 else 'CLEAR'
                if prev_truth is not None: transitions.append(truth_state==prev_truth and s==prev_pred or truth_state!=prev_truth and s!=prev_pred)
                prev_truth,prev_pred=truth_state,s
            metrics[name]={'clearance_mae_m':mae(errors),'known_pairs':len(vals),'transition_agreement':sum(transitions)/len(transitions) if transitions else None}
        result[parent]=metrics
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo-root',type=Path,required=True); ap.add_argument('--protocol',type=Path,required=True); ap.add_argument('--stream',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args();
    if a.output.exists(): raise ValueError('overwrite forbidden')
    protocol=json.loads(a.protocol.read_text(encoding='utf-8')); assert protocol['schema'].endswith('corrected_replay_protocol')
    assert sha256_file(a.stream)==protocol['stream']['sha256']
    result={'schema':SCHEMA,'protocol_sha256':sha256_file(a.protocol),'stream_sha256':sha256_file(a.stream),'parentwise':replay(a.stream),'model_loaded':False,'optimizer_constructed':False,'training_started':False,'holdout_outcomes_opened':False,'terminal':'QUALITY_GATED_CLEARANCE_FUSION_R0_REPLAY_COMPLETE_DEVELOPMENT_SIGNAL_ONLY'}
    a.output.parent.mkdir(parents=True,exist_ok=True); fd=os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(result,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
