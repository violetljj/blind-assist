"""Fixed fit component and profile-endpoint diagnostics, no fitting."""
import time
import numpy as np
import real_twin_joint_calibration as J
from cnh_sim2real_frozen import FrozenReplay

def main():
    started=time.monotonic();fit=J.read('geometry_fit.json');g=np.array([fit['parameters'][n] for n in J.GNAMES]);params,bg=J.instrument();bg=bg.reshape(64,16)
    profile=fit['profiles']['curtain_distance_m'];admitted=[q for q in profile['grid'] if q['objective']<=1.05*fit['objective']]
    endpoints=[min(admitted,key=lambda q:q['value']),max(admitted,key=lambda q:q['value'])];e=None;rows=[]
    try:
        e=FrozenReplay()
        for sid in ('4','4b','9'):
            d=J.recording(sid);a=d['A'][d['stable']].mean(0)
            cases=[]
            out=g.copy();out[1]=4.8;cases.append(('curtain_out_4p8',out,params,False))
            p=params.copy();p['offset_m']=0;cases.append(('offset_zero',g,p,False))
            p=params.copy();p.update(sigma_m=2*J.I.WIDTH/8,tail_fraction=.1,tail_scale_m=4*J.I.WIDTH/8);cases.append(('nominal_pulse_proxy',g,p,False))
            cases.append(('mean_only',g,params,True))
            cases.append(('curtain_out_mean_only',out,params,True))
            cases.append(('background_only_mean',g,params,True))
            cases.extend((f'profile_curtain_{endpoint["value"]:.1f}',np.array(endpoint['parameters']),params,False) for endpoint in endpoints)
            for name,gg,pp,mean_only in cases:
                rays,w=J.I.plane_rays(16);hit=J.D.S.raycast_boxes(np.zeros(3),rays@J.D.rx(gg[0]).T,J.boxes(gg))
                mean=J.I.predict(hit['distance'],hit['rho'],hit['cos'],w,pp,bias=bg).reshape(64,16);var=J.variance(mean-bg,a)
                if name=='background_only_mean':mean=bg.copy()
                h=np.broadcast_to(mean,(48,64,16)).copy() if mean_only else np.concatenate([mean+np.random.default_rng(seed).normal(size=(16,64,16))*np.sqrt(var) for seed in J.SEEDS])
                synthetic=dict(H=h,A=np.broadcast_to(a,(48,64)),seq=np.tile(np.arange(16),3));synthetic['ms']=synthetic['seq']*200
                key=f'{sid}_{name}';np.savez_compressed(J.OUT/(key+'_component_input.npz'),**synthetic,expected=mean)
                r=e.infer(J.normalized(synthetic,'v2').reshape(-1,8,8,16),synthetic['seq'],synthetic['ms']);np.savez_compressed(J.OUT/(key+'_component_scores.npz'),**r)
                rows.append(dict(condition=sid,component=name,geometry=gg.tolist(),instrument=pp,counts=J.D.counts(r)))
        J.save('component_report.json',dict(status='COMPLETE',rows=rows,seconds=time.monotonic()-started,resources_released=True,diagnostic_only=True,plan_sha256=J.I.sha(J.PLAN)))
    finally:
        if e:e.close()
    print('COMPONENTS',len(rows),'seconds',time.monotonic()-started)
if __name__=='__main__':main()
