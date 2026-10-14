"""Equivalent cached profile-bank comparison; no observation or label inputs."""
from functools import lru_cache
import numpy as np
import thin_appearance_features_dev_20261014 as F

SCALES=(4,6,8,10,12,14)


def normalized_rows(rows):
    a=np.asarray(rows,float).reshape(len(rows),-1)
    norms=np.linalg.norm(a,axis=1)
    return a/np.maximum(norms[:,None],1e-30),norms


@lru_cache(maxsize=1024)
def bank(profile_tuple):
    x=np.asarray(profile_tuple,float).reshape(33,3)
    background=np.linspace(x[0],x[-1],len(x))
    weights=np.linalg.norm(x-background,axis=1)*(np.abs(F.OFFSETS)<=8)
    center=float(weights@F.OFFSETS/max(float(weights.sum()),1e-8))
    profiles=[]
    for scale in SCALES:
        grid=np.linspace(-1,1,21)
        p=np.stack([np.interp(grid*scale+center,F.OFFSETS,x[:,k]) for k in range(3)],1)
        profiles.append(p)
    full=np.asarray(profiles)
    shape=full-np.asarray([np.linspace(p[0],p[-1],len(p)) for p in full])
    gradient=np.diff(full,axis=1)
    def shifts(a):return [normalized_rows(a[:,:-1]),normalized_rows(a),normalized_rows(a[:,1:])]
    return center,shifts(shape),shifts(gradient)


def pair_max(a,b):
    out=np.full((6,6),-2.,float)
    # F.shifted_cosine: shift<0 x[:-1]/y[1:], 0 x/y, >0 x[1:]/y[:-1].
    for i,j in ((0,2),(1,1),(2,0)):
        xa,na=a[i];yb,nb=b[j]
        valid=na[:,None]*nb[None,:]>=1e-8
        values=np.clip(xa@yb.T,-1.,1.)
        out=np.maximum(out,np.where(valid,values,-2.))
    index=np.unravel_index(int(np.argmax(out)),out.shape)
    value=float(out[index])
    return (None,None,None) if value==-2. else (value,SCALES[index[0]],SCALES[index[1]])


def compare(a,b):
    if a.get('status')!='AVAILABLE' or b.get('status')!='AVAILABLE':
        return dict(status='UNKNOWN',reason='MISSING_COLOR_CROSS_SECTION')
    ca,sa,ga=bank(tuple(np.asarray(a['lab_profile']).ravel()))
    cb,sb,gb=bank(tuple(np.asarray(b['lab_profile']).ravel()))
    best=pair_max(sa,sb);gradient=pair_max(ga,gb)[0]
    return dict(status='AVAILABLE',lab_core_delta=float(np.linalg.norm(np.asarray(a['core_lab'])-b['core_lab'])),
        lab_contrast_delta=float(np.linalg.norm(np.asarray(a['contrast_lab'])-b['contrast_lab'])),
        normalized_shape_similarity=best[0],best_shape_scale_a=best[1],best_shape_scale_b=best[2],
        gradient_shape_similarity=gradient,automatic_profile_center_a=ca,automatic_profile_center_b=cb,
        shape_definition='multiscale background-subtracted Lab profile; automatic energy-center alignment',
        old_gray_gradient_similarity=F.shifted_cosine(a['old_gray_gradient'],b['old_gray_gradient']) if a['old_gray_profile_valid'] and b['old_gray_profile_valid'] else None,
        strip_state_a=a['quality']['strip_state'],strip_state_b=b['quality']['strip_state'],
        stripe_reliable_a=a['quality']['stripe_reliable'],stripe_reliable_b=b['quality']['stripe_reliable'])


def controls(path):
    import json,time
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    ds=[f['feature'] for r in rows for f in r['features'] if f['feature']['status']=='AVAILABLE']
    chosen=[(ds[i],ds[(i*31+17)%len(ds)]) for i in range(80)]
    chosen.extend((ds[i],ds[i]) for i in range(10))
    start=time.perf_counter();maxerror=0.;scale_ties=0
    for a,b in chosen:
        slow=F.compare_appearance(a,b);fast=compare(a,b)
        for k in ('lab_core_delta','lab_contrast_delta','normalized_shape_similarity','gradient_shape_similarity',
                  'automatic_profile_center_a','automatic_profile_center_b','old_gray_gradient_similarity'):
            assert (slow[k] is None)==(fast[k] is None),(k,slow[k],fast[k])
            if slow[k] is not None:
                e=abs(slow[k]-fast[k]);maxerror=max(maxerror,e);assert e<1e-10,(k,e)
        scale_ties+=bool((slow['best_shape_scale_a'],slow['best_shape_scale_b'])!=(fast['best_shape_scale_a'],fast['best_shape_scale_b']))
        assert (slow['normalized_shape_similarity'] is None or slow['normalized_shape_similarity']>=.65)==(fast['normalized_shape_similarity'] is None or fast['normalized_shape_similarity']>=.65)
    return dict(pairs=len(chosen),max_numeric_error=maxerror,equivalent_maximum_scale_ties=scale_ties,
                decision_parity=True,wall_s=time.perf_counter()-start,cache=str(bank.cache_info()))


def matching_controls(path):
    import json
    import thin_appearance_track_dev_20261014 as T
    ds=[f['feature'] for r in [json.loads(s) for s in path.read_text().splitlines()] for f in r['features']]
    checks=0;maxcosterror=0.;fast_function=compare
    for i in range(100):
        a,b=ds[i],ds[(i*31+17)%len(ds)]
        for arm in ('local_lab','local_strip','anchor_strip'):
            T.Fast.compare=F.compare_appearance
            slow=T.compatible(a,b,arm)
            T.Fast.compare=fast_function
            fast=T.compatible(a,b,arm)
            assert all(slow[k]==fast[k] for k in ('accepted','status','reason'))
            if slow['accepted']:
                def cost(q):return .3*q['lab_core_delta']+.2*q['lab_contrast_delta']+(0. if arm=='local_lab' else 3*(1-q['normalized_shape_similarity']))
                error=abs(cost(slow)-cost(fast));assert error<1e-10;maxcosterror=max(maxcosterror,error)
            checks+=1
    return dict(status='PASS',matching_reason_decision_checks=checks,max_association_cost_error=maxcosterror,
                geometry_and_gates_unchanged=True,labels_read=False)
