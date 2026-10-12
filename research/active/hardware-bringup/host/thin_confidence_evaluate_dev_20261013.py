"""Evaluate distance-quality abstention costs; no metric-distance truth exists here."""
import json
import hashlib
from pathlib import Path
from thin_structure_evaluate_dev_20261013 import matches

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/thin-confidence-dev-20261013'
OLD=ROOT/'artifacts.local/work/thin-structure-dev-20261013'
PREV=ROOT/'artifacts.local/work/thin-local-geometry-dev-20261013'


def read(path):
    return {(r['key'],r['i']):r for r in map(json.loads,path.read_text().splitlines())}


def main():
    data=read(OUT/'frames.jsonl');previous=read(PREV/'camera_chain_frames.jsonl')
    support=read(OUT/'scale_sensitivity.jsonl')
    assert data.keys()==previous.keys() and len(data)==1677
    for key,r in data.items():
        assert len(r['candidates'])==len(previous[key]['candidates'])
        for c,d in zip(r['candidates'],previous[key]['candidates']):
            assert all(c[k]==d[k] for k in ('x1','y1','x2','y2','score','track_id','hits','stable'))
        for c,d in zip(r['candidates'],support[key]['candidates']):
            assert all(c[k]==d[k] for k in ('x1','y1','x2','y2'))
            c['scale_support']=d['scale_support']
            c['confidence']['arms']['anchor_support']=c['confidence']['arms']['range_only'] and d['scale_support'].get('gate') is True
    names=['unfiltered','range_only','fit_gate','tof_agreement','local_consistency','anchor_support']
    def passed(c,name):return True if name=='unfiltered' else c['confidence']['arms'][name] is True
    def inside(c):return any(c['geometry']['robust_replace']['query'])
    labels=json.loads((OLD/'annotations.json').read_text());rows=[]
    for s in labels['samples']:
        cs=data[s['session'],s['i']]['candidates']
        hit=[c for c in cs if s['segment'] and matches(s['segment'],c)]
        row=dict(session=s['session'],i=s['i'],label_status=s['label_status'],matched_candidates=hit,arms={})
        for name in names:
            kept=[c for c in cs if passed(c,name)];target=[c for c in hit if passed(c,name)]
            row['arms'][name]=dict(target=bool(target),stable_target=any(c['stable'] for c in target),stable_inside_target=any(c['stable'] and inside(c) for c in target),candidates=len(kept),stable=sum(c['stable'] for c in kept))
        rows.append(row)
    summary={}
    for key in ['all']+list(dict.fromkeys(s['session'] for s in labels['samples'])):
        rr=[r for r in rows if key=='all' or key==r['session']]
        targets=[r for r in rr if r['label_status']=='visible_segment'];neg=[r for r in rr if r['label_status']=='no_target_rod']
        summary[key]=dict(n=len(targets),negative_n=len(neg),arms={name:dict(target=sum(r['arms'][name]['target'] for r in targets),stable_target=sum(r['arms'][name]['stable_target'] for r in targets),stable_inside_target=sum(r['arms'][name]['stable_inside_target'] for r in targets),negative_candidates=sum(r['arms'][name]['candidates'] for r in neg),negative_stable=sum(r['arms'][name]['stable'] for r in neg)) for name in names})
    totals={name:dict(candidates=0,stable_inside=0,stable_inside_frames=0) for name in names}
    for r in data.values():
        for name in names:
            cs=[c for c in r['candidates'] if passed(c,name)]
            eligible=sum(c['stable'] and inside(c) for c in cs)
            totals[name]['candidates']+=len(cs);totals[name]['stable_inside']+=eligible;totals[name]['stable_inside_frames']+=eligible>0
    failures=[]
    for key,i in [('073554',59),('073635',39),('073821',81)]:
        cs=data[key,i]['candidates']
        if key=='073635':cs=[c for c in cs if c['x1']==253 and c['y1']==305 and c['x2']==317]
        else:
            label=next(s for s in labels['samples'] if s['session']==key and s['i']==i)
            cs=[c for c in cs if matches(label['segment'],c)]
        failures.append(dict(session=key,i=i,candidates=cs))
    result=dict(summary=summary,totals=totals,failures=failures,rows=rows,annotation_sha256=hashlib.sha256((OLD/'annotations.json').read_bytes()).hexdigest(),evidence='consumed Development weak localization only; threshold pass is not distance accuracy')
    (OUT/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(summary['all'],indent=2))


if __name__=='__main__':main()
