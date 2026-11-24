"""Score immutable geometry predictions against label-only frozen walk episodes."""
import json
import sys

from replay import OUT,MATH,PACK,HERE,sha
import numpy as np

M3=MATH/'m3_online_conformal_safety'
M9=MATH/'m9_pgl4_frobenius_integrability'
sys.path.insert(0,str(M3))
from eval_alert_operating_point import Clip,summarize
from eval_on_v4 import make_episodes,BW_SPEEDS_PROXY


def evaluate(arm,meta):
    all_clips=[]
    counts={}
    for walk in meta['walks']:
        for name,reg in meta['walks'][walk]['clips'].items():
            path=OUT/(name+'.npz')
            if reg['prediction_sha256']!=sha(path):
                raise ValueError('Changed predictions')
            with np.load(path) as saved:
                score,clear_proxy,tiles,npoints,ms=saved[arm].T
            with np.load(PACK/path.name) as p:
                rows=p['eval_rows']
            eps=make_episodes(name,rows,BW_SPEEDS_PROXY,kind='blind')
            valid=eps[0]['valid']
            known_clear=valid&(rows[:,0]>4.)&(rows[:,2]<=.5)
            distance=np.cumsum(np.where(valid,rows[:,3],0.),dtype=float)
            c=Clip({'name':name},'aria',walk,int(walk=='walk_1'),score,score,score,eps,known_clear,distance)
            c.clear_proxy,c.tiles,c.npoints,c.ms=clear_proxy.astype(bool),tiles,npoints,ms
            all_clips.append(c)
    def display(cs,cut):
        results=summarize(sum(c.count(c.raw>=cut) for c in cs))
        results['timely']=results['avoidable']-results['late']
        conf=np.zeros((3,3),int)
        for c in cs:
            rows=c.episodes[0]['rows']
            valid=c.episodes[0]['valid']
            row=np.where(valid&(rows[:,0]<=4.),0,np.where(c.clear,1,2))
            col=np.where(c.raw>=cut,0,np.where(c.clear_proxy,1,2))
            np.add.at(conf,(row,col),1)
        results['reference_by_predicted_hazard_clear_unknown']=conf.tolist()
        return results
    cut_sets={}
    sensor={}; actions={}
    for walk in meta['walks']:
        own=[c for c in all_clips if c.source==walk]; train=[c for c in all_clips if c.source!=walk]
        # Enumerate all distinct positive integer support cuts in training only.
        choices=[np.inf,*sorted({x for c in train for x in c.raw if x>0})]
        pool=[]
        for k in choices:
            r=display(train,k)
            if r['fa_per_km']<=5. and r['clear_warnings_per_km']<=5.:
                pool.append(((-r['timely'],r['repeated_clear_warnings'],r['clear_alert_frames'],r['alert_frames'],-k),k,r))
        _,k,training=min(pool,key=lambda o:o[0])
        cut_sets[walk]=dict(cut=None if np.isinf(k) else int(k),training=training,heldout=display(own,k))
        for c in own:
            actions[c.obs['name']]=c.raw>=k
        p=np.concatenate([c.npoints for c in own]); ms=np.concatenate([c.ms for c in own])
        sensor[walk]=dict(frames=len(p),any_points=int((p>0).sum()),triangulated=int(p.sum()),peak_points=int(p.max()),
                          points_p50_p95=np.percentile(p,[50,95]).tolist(),step_ms_p50_p95=np.percentile(ms,[50,95]).tolist(),
                          support_at_least_one=int(sum((c.raw>0).sum() for c in own)))
    pooled=summarize(sum(c.count(actions[c.obs['name']]) for c in all_clips))
    pooled['timely']=pooled['avoidable']-pooled['late']
    pooled['reference_by_predicted_hazard_clear_unknown']=np.sum([a['heldout']['reference_by_predicted_hazard_clear_unknown']
                                                       for a in cut_sets.values()],axis=0).tolist()
    aligned=OUT.parent/'aria_alert_support/v5_aria_alert_support.json'
    events=json.loads(aligned.read_text())['results']['v4']['events']
    cs={c.obs['name']:c for c in all_clips}
    fixed_hits=[]; active_hits=[]
    for e in events:
        if e['timely_hazard_frames'] and not e['diagnostics']['any_now_truth_ray18']:
            ix=e['timely_hazard_frames']; c=cs[e['clip']]
            fixed_hits.append(bool((c.raw[ix]>=3).any()))
            active_hits.append(bool(actions[e['clip']][ix].any()))
    return dict(sensor_counts=sensor, fixed_three_voxel=display(all_clips,3),cross_walk=cut_sets,pooled=pooled,
                targeted_13=dict(n=len(fixed_hits),fixed_three_hits=sum(fixed_hits),cross_walk_hits=sum(active_hits),
                                 evaluator_alignment_sha256=sha(aligned)))


def main():
    meta=json.loads((OUT/'observation_inventory.json').read_text())
    records={arm:evaluate(arm,meta) for arm in ('provided','open')}
    result=dict(scores=records,code_sha256={p.name:sha(p) for p in (Path(__file__),M3/'eval_alert_operating_point.py',
            M3/'eval_on_v4.py')},inventory_sha256=sha(OUT/'observation_inventory.json'))
    (OUT/'sensor_witness.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    for k,r in records.items():
        print(k,'pooled',r['pooled'],'fixed',r['fixed_three_voxel'],'targeted',r['targeted_13'])


from pathlib import Path
if __name__=='__main__':
    main()
