"""Recover contaminated affine fits rather than reject RGB wholesale."""
import json
import time
import copy
import numpy as np
from atom_camera import AtomCamera
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_small_model_dev_20261012 as S
import real_rgb_query_feasibility_dev_20261012 as F
from fusion_refinement_cache_dev_20261013 import OUT
from fusion_refinement_readout_dev_20261013 import project
from fusion_anchor_candidates_dev_20261013 import robust_affine

def main():
    start=time.perf_counter()
    data=json.loads((OUT/'readout.json').read_text())
    cache=json.loads((OUT/'cache_manifest.json').read_text())
    cam=AtomCamera();K=cam.K_upright;w,h=cam.upright_size
    vv,uu=np.mgrid[:h,:w]
    dirs=np.stack(((uu-K[0,2])/K[0,0],(vv-K[1,2])/K[1,1],np.ones_like(uu)),-1).reshape(-1,3)
    Rct=np.array(json.loads((X.OUT/'camera_tof_extrinsic.json').read_text())['rotation']);rays=X.zone_rays()
    for key,s in data['sessions'].items():
        replacement=[];strict=[];attempts=0;recovered=0
        for original,quality,c in zip(s['arms']['baseline'],s['arms']['quality_floor'],cache['sessions'][key]['frames']):
            a=copy.deepcopy(original);b=copy.deepcopy(quality)
            if 'cache' in c and not original['quality'].get('accepted',False):
                attempts+=1
                with np.load(OUT/c['cache']) as p:disp,valid=p['disp'],p['valid']
                dz,zfac=S.zone_disparity(disp,K,Rct,rays)
                dist=np.array(c['dist']);keep=np.isfinite(dist)&(dist>.3)&(dist<4)
                inv=np.full(64,np.nan);inv[keep]=1/(dist[keep]*zfac[keep])
                fit=robust_affine(dz,inv)
                a['robust_diagnostic']=b['robust_diagnostic']=fit['diagnostics']
                if fit['fit'] is not None:
                    recovered+=1
                    cr=dirs@(F.rot(c['pitch'],c['roll'])@Rct.T).T
                    projected=project(disp,valid,fit['fit'],cr)
                    for row in (a,b):row.update(projected,fit=fit['fit'],anchor='robust_tof')
            replacement.append(a);strict.append(b)
        s['arms']['robust_replace']=replacement
        s['arms']['robust_strict']=strict
        print(key,'attempts',attempts,'recovered',recovered,'RGB',np.sum([r['rgb'] for r in replacement],axis=0).tolist(),flush=True)
    data['robust_wall_s']=time.perf_counter()-start
    (OUT/'robust_readout.json').write_text(json.dumps(data),encoding='utf8')
    print('wall',data['robust_wall_s'])

if __name__=='__main__':main()
