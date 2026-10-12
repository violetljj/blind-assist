"""Keep visual observations while explicitly abstaining from unsupported distance.

Engineering provenance states, NOT calibrated confidence or a safety decision.
Reads only label-blind diagnostics. No candidates are removed or moved.
"""
import json
from collections import Counter
from pathlib import Path

OUT=Path(__file__).resolve().parents[4]/'artifacts.local/work/thin-confidence-dev-20261013'


def state(c, support):
    g=c['geometry']['robust_replace'];q=c['confidence']['frame_fit']
    if g['status']!='AVAILABLE':return 'UNKNOWN_GEOMETRY'
    if g['domain_status']=='BELOW_NEAR_MODEL_RANGE':return 'BELOW_MODEL_RANGE'
    if not c['confidence']['raw_rules']['local_consistent']:return 'LOCAL_DEPTH_SPREAD'
    if q['anchor'] not in ('tof','robust_tof'):return 'UNVERIFIED_ANCHOR'
    if not q['accepted']:return 'FIT_REJECTED'
    if support.get('gate') is not True:return 'OUTSIDE_ANCHOR_SUPPORT'
    return 'ANCHOR_SUPPORTED_UNVALIDATED'


def main():
    counts=Counter();frames=0
    with (OUT/'frames.jsonl').open() as src,(OUT/'scale_sensitivity.jsonl').open() as aux,(OUT/'distance_states.jsonl').open('w',encoding='utf8') as output:
        for a,b in zip(src,aux,strict=True):
            r=json.loads(a);s=json.loads(b);assert (r['key'],r['i'])==(s['key'],s['i'])
            cs=[]
            for c,d in zip(r['candidates'],s['candidates'],strict=True):
                assert all(c[k]==d[k] for k in ('x1','y1','x2','y2'))
                status=state(c,d['scale_support']);counts[status]+=1
                cs.append(dict(visual_candidate={k:c[k] for k in ('x1','y1','x2','y2','rank','track_id','hits','stable')},distance_state=status,unvalidated_camera_z_estimate_m=c['geometry']['robust_replace']['camera_z_median_m'],distance_verified=False,safety_decision=None))
            output.write(json.dumps(dict(key=r['key'],i=r['i'],candidates=cs),allow_nan=False)+'\n');frames+=1
    assert frames==1677 and sum(counts.values())==18483
    summary=dict(frames=frames,candidates=sum(counts.values()),counts=dict(counts),evidence='visual observations retained; no metric verification or safety decisions; priority-ordered diagnostic states')
    (OUT/'distance_states_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
