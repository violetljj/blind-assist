"""Assistant weak-label image localization, never obstacle/safety accuracy."""
import json
import hashlib
import argparse
from pathlib import Path
import cv2
import numpy as np
from frozen_fusion_eval_v2 import confirm

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/thin-structure-dev-20261013'
OLD = ROOT / 'artifacts.local/work/fusion-refinement-dev-20261013'


def matches(segment, c, tolerance=10):
    a, b = np.asarray(segment, float).reshape(2, 2)
    p, q = np.array([[c['x1'], c['y1']], [c['x2'], c['y2']]])
    length = np.linalg.norm(b-a); size = np.linalg.norm(q-p)
    if min(length, size) < 1: return False
    u = (b-a)/length; v = (q-p)/size
    if abs(u@v) < np.cos(np.deg2rad(15)): return False
    normal = np.array([-v[1], v[0]])
    if max(abs((a-p)@normal), abs((b-p)@normal)) > tolerance: return False
    lo, hi = sorted([(p-a)@u, (q-a)@u])
    return max(0, min(length, hi)-max(0, lo)) >= .5*length


FILTERS = {
    'top40': lambda c: True,
    'top20': lambda c: c['rank'] <= 20,
    'paired': lambda c: c['kind'] == 'pair',
    'stable': lambda c: c['stable'],
    'positive_depth': lambda c: c['positive_depth'],
    'stable_positive': lambda c: c['stable'] and c['positive_depth'],
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='detection_frames.jsonl')
    parser.add_argument('--suffix',default='')
    args=parser.parse_args()
    # Narrow geometry check: parallel overlap passes; crossing and tiny touches fail.
    assert matches([10, 0, 10, 100], dict(x1=13,y1=0,x2=13,y2=80))
    assert not matches([10,0,10,100], dict(x1=0,y1=50,x2=100,y2=50))
    assert not matches([10,0,10,100], dict(x1=10,y1=95,x2=10,y2=125))
    labels = json.loads((OUT/'annotations.json').read_text())
    detections = {(r['key'],r['i']): r for r in map(json.loads,(OUT/args.input).read_text().splitlines())}
    old = json.loads((OLD/'robust_readout.json').read_text())
    alerts = {}
    for key, s in old['sessions'].items():
        for arm in ('baseline', 'robust_replace'):
            fr = s['arms'][arm]
            h = confirm([r['tof_strong'][0] or r['rgb'][0] for r in fr])
            b = confirm([r['tof_strong'][1] or r['rgb'][1] for r in fr])
            for r, yes in zip(fr, h|b): alerts[key,r['i'],arm] = bool(yes)
    rows=[]; panels={}
    for s in labels['samples']:
        key,i=s['session'],s['i']; d=detections[key,i]; cs=d['candidates']
        r=dict(session=key,i=i,label_status=s['label_status'],matches={},counts={},matched_candidates=[],alerts={arm:alerts[key,i,arm] for arm in ('baseline','robust_replace')})
        for name, keep in FILTERS.items():
            selected=[c for c in cs if keep(c)]
            r['counts'][name]=len(selected)
            r['matches'][name]=any(matches(s['segment'],c) for c in selected) if s['segment'] else None
        r['matched_candidates']=[c for c in cs if s['segment'] and matches(s['segment'],c)]
        r['strict3px']=any(matches(s['segment'],c,3) for c in cs) if s['segment'] else None
        rows.append(r)
        im=cv2.imread(str(OUT/f'raw_{key}_{i:04d}.jpg'))
        for c in cs:
            color=(100,100,100)
            if c['positive_depth']: color=(0,180,255)
            if c in r['matched_candidates']:color=(0,255,0)
            cv2.line(im,(round(c['x1']),round(c['y1'])),(round(c['x2']),round(c['y2'])),color,1)
        if s['segment']:
            x,y,xx,yy=s['segment'];cv2.line(im,(x,y),(xx,yy),(255,0,255),2)
        header=np.zeros((40,480,3),np.uint8)
        cv2.putText(header,f'{key}/{i} hit={r["matches"]["top40"]} depth={r["matches"]["positive_depth"]}',(5,24),0,.5,(255,255,255),1)
        panels.setdefault(key,[]).append(np.vstack((header,im)))
    summary={}
    for key in ['all']+list(panels):
        rr=[r for r in rows if key=='all' or r['session']==key]
        target=[r for r in rr if r['label_status']=='visible_segment']
        negative=[r for r in rr if r['label_status']=='no_target_rod']
        summary[key]=dict(evaluable=len(target),not_evaluable=sum(r['label_status']=='NOT_EVALUABLE' for r in rr),hits={f:sum(r['matches'][f] for r in target) for f in FILTERS},distractor_frames=len(negative),distractor_candidate_counts={f:sum(r['counts'][f] for r in negative) for f in FILTERS},old_nonalert={arm:dict(n=sum(not r['alerts'][arm] for r in target),hits=sum(r['matches']['top40'] and not r['alerts'][arm] for r in target)) for arm in ('baseline','robust_replace')})
        summary[key]['strict3px_hits']=sum(r['strict3px'] for r in target)
    for key, pp in panels.items():cv2.imwrite(str(OUT/f'result{args.suffix}_{key}.jpg'),np.vstack((np.hstack(pp[:3]),np.hstack(pp[3:]))))
    result=dict(evidence=labels['provenance'],match_rule=labels['match_rule'],annotation_sha256=hashlib.sha256((OUT/'annotations.json').read_bytes()).hexdigest(),summary=summary,rows=rows)
    (OUT/f'evaluation{args.suffix}.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
