"""Reproduce evaluator-only proposal ceiling and centerline depth diagnosis."""
import json
import numpy as np
from thin_structure_evaluate_dev_20261013 import OUT, OLD, matches


def main():
    labels=json.loads((OUT/'annotations.json').read_text())
    selected={(s['session'],s['i']):s for s in labels['samples'] if s['segment']}
    ceiling=[]
    for line in (OUT/'detection_all_proposals.jsonl').open():
        r=json.loads(line);key=(r['key'],r['i'])
        if key not in selected:continue
        cs=[c for c in r['candidates'] if matches(selected[key]['segment'],c)]
        ceiling.append(dict(session=key[0],i=key[1],hit=bool(cs),best_rank=min((c['rank_before_dedup'] for c in cs),default=None),positive=any(c['positive_depth'] for c in cs),strict3=any(matches(selected[key]['segment'],c,3) for c in cs)))
    result=dict(rows=ceiling,hits=sum(r['hit'] for r in ceiling),positive=sum(r['positive'] for r in ceiling),strict3=sum(r['strict3'] for r in ceiling),n=len(ceiling))
    (OUT/'all_proposals_evaluation.json').write_text(json.dumps(result,indent=2))
    manifest=json.loads((OLD/'cache_manifest.json').read_text());rows=[]
    for (key,i),s in selected.items():
        frame=next(r for r in manifest['sessions'][key]['frames'] if r['i']==i)
        row=dict(session=key,i=i,available=bool(frame.get('cache')))
        if frame.get('cache'):
            with np.load(OLD/frame['cache']) as z:d=z['disp'].copy()
            pts=np.array(s['segment']).reshape(2,2);u=pts[1]-pts[0];u=u/np.linalg.norm(u);n=np.array([-u[1],u[0]])
            q=pts[0]+np.linspace(.1,.9,16)[:,None]*(pts[1]-pts[0])
            iqr=max(float(np.percentile(d,75)-np.percentile(d,25)),1e-5)
            xy=np.rint(np.stack([q,q+6*n,q-6*n])).astype(int)
            valid=((xy[:,:,0]>=0)&(xy[:,:,0]<480)&(xy[:,:,1]>=0)&(xy[:,:,1]<640)).all(axis=0)
            if valid.sum()>=8:
                values=d[xy[:,valid,1],xy[:,valid,0]]
                row['center_prominence']=float(np.median(values[0]-np.maximum(values[1],values[2]))/iqr)
        rows.append(row)
    result=dict(evidence='annotation-initialized oracle diagnostic, never candidate inference',rows=rows,available=sum('center_prominence' in r for r in rows),positive=sum(r.get('center_prominence',-1)>=.02 for r in rows))
    (OUT/'oracle_center_depth.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
