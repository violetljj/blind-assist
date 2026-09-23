"""Frozen public temporal scores, then evaluator-only return attribution."""
import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
from scipy.stats import rankdata
from query_occupancy_data import read, write, sha, observation_tokens
from tof_fov45_core import simulate, boxes45
from ba_camera_corridor import sample_native, rays
from range_pair_observations import evaluator_target_mask

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
ROOT=REPO/'artifacts.local/evidence/ba-motion-return-20260923'
COHORTS={'stability':('ba-local-stability-20260923','local-stability/'),
         'rescue':('ba-local-rescue-fresh-20260923','local-rescue-fresh/')}
SCORES=('residual','static_residual','wrong_residual')


def auc(y, residual):
    y=np.asarray(y,bool); x=-np.asarray(residual,float)
    assert np.isfinite(x).all()
    n=int(y.sum()); m=len(y)-n
    if not n or not m:return None
    return float((rankdata(x)[y].sum()-n*(n+1)/2)/(n*m))


def aggregate(rows):
    out={}
    for label in ('contributor','corridor_target'):
        mixed=[r for r in rows if r[label]['residual'] is not None]
        out[label]=dict(evaluable_zones=len(mixed),total_zones=len(rows),
            geometry_groups=len({r['group'] for r in mixed}),
            auc={s:float(np.mean([r[label][s] for r in mixed])) if mixed else None for s in SCORES},
            common_points=sum(r['common_points'] for r in mixed))
    return out


def prepare():
    files=[Path(__file__),HERE/'motion_return_public.py',HERE/'MOTION_RETURN_PROTOCOL_20260923.md',
           HERE/'range_pair_observations.py',HERE/'inherit_spatial_model.py',HERE/'tof_fov45_core.py',
           HERE/'ba_camera_corridor.py',HERE/'query_occupancy_data.py']
    write(ROOT/'plan/seal.json',dict(files={p.relative_to(REPO).as_posix():sha(p) for p in files}))
    inputs=[dict(alias='plan',path=str(ROOT/'plan'),role='configuration',purpose='frozen-public-motion-test')]
    for name,(stem,_) in COHORTS.items():
        base=ROOT.parent/stem
        for alias,path,role in [('spec',base/'plan/spec.json','configuration'),
          ('observations',base.with_name(stem+'-prepared')/'observations','observation'),
          ('manifest',base.with_name(stem+'-prepared')/'materialization.json','configuration'),
          ('native',base.with_name(stem+'-capture')/'evaluator','evaluator')]:
            inputs.append(dict(alias=name+'_'+alias,path=str(path),role=role,purpose='consumed-motion-return-diagnostic'))
    write(ROOT/'run-spec.json',dict(schema='blindassist-asset-run-v1',id='motion-return-20260923-v1',
      route='ue-motion-return',question='Does causal public optical motion add within-zone return attribution evidence?',
      evaluator='research/active/dtr-r0/nearfield/motion_return_diagnostic.py',
      evidence_boundary='Consumed controlled pure translation; no alert or learned ownership claim',
      reuse=dict(mode='diagnostic',query='LOCAL stability rescue RGB ToF motion return attribution'),inputs=inputs,
      outputs=[dict(alias='result',path=str(ROOT.with_name(ROOT.name+'-run')/'result.json'),role='result',required=True)],
      result_output='result',command=[sys.executable,str(Path(__file__)),'execute','--result','{{output:result}}']))


def execute(result):
    from motion_return_public import extract
    assert read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])['state']=='running'
    start=time.perf_counter(); out=result.parent; seal=read(ROOT/'plan/seal.json')
    for f,h in seal['files'].items():assert sha(REPO/f)==h
    sys.path.insert(0,str(REPO))
    from tools.research_backend import BackendCandidate,DeviceObservation,select_backend
    select_backend('scalar-scoring',cpu=BackendCandidate('opencv-lk-numpy','cpu',lambda:sum(range(10)),
        lambda _:DeviceObservation('cpu',platform.processor(),'OpenCV CPU LK and NumPy '+np.__version__)),
        cpu_reason='TASK_NOT_GPU_SUITABLE',record_path=out/'backend.json')
    summaries={}; all_rows=[]; parity=0
    for cohort,(stem,prefix) in COHORTS.items():
        base=ROOT.parent/stem; prep=base.with_name(stem+'-prepared'); cap=base.with_name(stem+'-capture')
        manifest=read(prep/'materialization.json')
        for f in ('tof.npy','rgb.npy','identities.json'):
            assert sha(prep/'observations'/f)==manifest['hashes']['observations/'+f]
        ids=read(prep/'observations/identities.json')
        rgb=np.load(prep/'observations/rgb.npy',mmap_mode='r');tof=np.load(prep/'observations/tof.npy')
        assert np.array_equal(tof[:,:,2:],np.broadcast_to(tof[0,:,2:],tof[:,:,2:].shape))
        frames=[]
        for i in range(len(tof)):
            # Only reset/order identities, never geometry or evaluator labels.
            if ids[i]['frame_in_clip']==0:
                frames.append(None); continue
            assert ids[i]['clip_id']==ids[i-1]['clip_id'] and ids[i]['frame_in_clip']==ids[i-1]['frame_in_clip']+1
            frames.append(extract(rgb[i],rgb[i-1],tof[i],tof[i-1]))
            if i%96==1:print('PUBLIC',cohort,i,flush=True)
        template=next(f for f in frames if f is not None)
        # First frames retain the fixed candidate grid but have no causal score.
        for i,f in enumerate(frames):
            if f is None:
                f={k:np.array(v,copy=True) for k,v in template.items()}
                for k in ('tracking_raw','tracked','matched','wrong_matched'):f[k][:]=False
                for k in ('fb_error',*SCORES):f[k][:]=np.nan
                f['missing_reason'][:]=8
                f['previous_zone'][:]=-1;f['wrong_previous_zone'][:]=-1
                f['previous_xy'][:]=np.nan;f['previous_return_valid'][:]=False
                from inherit_spatial_model import canonical_tof
                valid=canonical_tof(tof[i])[1]
                f['current_return_valid']=f['candidate'] & valid[np.maximum(f['current_zone'],0)]
                frames[i]=f
        arrays={k:np.stack([f[k] for f in frames]) for k in template}
        np.savez_compressed(out/(cohort+'-public.npz'),**arrays)
        write(out/(cohort+'-public-seal.json'),dict(sha256=sha(out/(cohort+'-public.npz')),
            evaluator_opened=False,frames=len(frames),rgb_sha256=sha(prep/'observations/rgb.npy'),
            tof_sha256=sha(prep/'observations/tof.npy')))
        # Evaluator starts only after public predictions are sealed.
        spec=read(base/'plan/spec.json'); geo=read(cap/'evaluator/geometry.json')
        rows=[]; count=Counter(); strata=defaultdict(Counter); labels=[]
        ax,ay=rays();grid=template['low_flat_index']; zones=template['current_zone']
        native_hashes={}
        for i,(case,g,f) in enumerate(zip(spec['cases'],geo,frames,strict=True)):
            assert all(case['camera'][k]==0 for k in ('pitch','yaw','roll'))
            if case['frame_in_clip']:
                prior=spec['cases'][i-1]
                assert case['objects']==prior['objects']
                assert all(case['camera'][k]==prior['camera'][k] for k in ('y','z'))
            path=cap/'evaluator'/g['native_path']; assert sha(path)==g['native_sha256'];native_hashes[str(i)]=sha(path)
            native=np.load(path); depth=sample_native(native)
            values,traces=simulate(depth,prefix+case['sensor_noise_key'],boxes45())
            assert np.array_equal(observation_tokens(values,boxes45()),tof[i]); parity+=1
            contrib=np.zeros(depth.size,bool)
            for trace in traces:
                if trace['observed']:contrib[trace['pixel_indices']]=True
            target=evaluator_target_mask(native,g)
            inside=(depth>=.3)&(depth<=3)&(np.abs(ax*depth)<=.3)&(ay*depth>=-.2)&(ay*depth<=.9)
            y=contrib[grid]; relevant=(contrib & target.ravel() & inside.ravel())[grid]
            common=f['matched'] & f['wrong_matched']
            for s in SCORES:common &= np.isfinite(f[s])
            assert not np.any(y & ~f['current_return_valid'])
            c=Counter(points=int(f['candidate'].sum()),current_valid=int(f['current_return_valid'].sum()),
              tracked=int((f['tracked'] & f['candidate']).sum()),matched=int(f['matched'].sum()),common=int(common.sum()),
              contributors=int(y.sum()),matched_contributors=int((y & f['matched']).sum()),
              common_contributors=int((y & common).sum()),relevant_contributors=int(relevant.sum()),
              matched_relevant=int((relevant & f['matched']).sum()),frames=1,first_frames=int(case['frame_in_clip']==0))
            for reason,n in zip(*np.unique(f['missing_reason'][f['candidate']],return_counts=True)):
                c['reason_'+str(reason)]+=int(n)
            count.update(c)
            for axis in ('shape','layout_relation','phase','base_group_id'):
                strata[axis+'/'+str(case[axis])].update(c)
            for zone in range(64):
                take=common & (zones==zone)
                rows.append(dict(cohort=cohort,frame=i,zone=zone,group=case['base_group_id'],
                  phase=case['phase'],shape=case['shape'],relation=case['layout_relation'],common_points=int(take.sum()),
                  contributor={s:auc(y[take],f[s][take]) for s in SCORES},
                  corridor_target={s:auc(relevant[take],f[s][take]) for s in SCORES}))
            labels.append(dict(contributor=y,corridor_target=relevant))
        np.savez_compressed(out/(cohort+'-labels.npz'),**{k:np.stack([v[k] for v in labels]) for k in labels[0]})
        stats=aggregate(rows); primary=stats['contributor']; a=primary['auc']
        coverage=count['matched_contributors']/count['contributors'] if count['contributors'] else 0
        passed=(a['residual'] is not None and a['residual']>=.7 and
            all(a['residual']-a[s]>=.05 for s in SCORES[1:]) and coverage>=.5 and primary['geometry_groups']>=4)
        report=dict(summary=stats,coverage=dict(count),matched_contributor_fraction=coverage,pass_gate=passed,
          strata_coverage={k:dict(v) for k,v in strata.items()},
          strata_auc={axis:{v:aggregate([r for r in rows if r[axis]==v]) for v in sorted({r[axis] for r in rows})}
             for axis in ('phase','shape','relation','group')},native_hashes=native_hashes)
        write(out/(cohort+'-report.json'),report); summaries[cohort]=report;all_rows.extend(rows)
        print('EVALUATED',cohort,primary,'coverage',coverage,flush=True)
    write(out/'zone-rows.json',all_rows)
    write(result,dict(status='PASS',decision='MOTION_RETURN_ATTRIBUTION_COMPONENT' if all(r['pass_gate'] for r in summaries.values())
          else 'MOTION_RETURN_ATTRIBUTION_NOT_ESTABLISHED',
          cohorts={k:{x:r[x] for x in ('summary','coverage','matched_contributor_fraction','pass_gate')} for k,r in summaries.items()},
          saved_ToF_parity_frames=parity,fits=0,cutoff_selections=0,frames=1152,elapsed_s=time.perf_counter()-start))
    for f,h in seal['files'].items():assert sha(REPO/f)==h
    write(out/'output-seal.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute']);p.add_argument('--result',type=Path)
    a=p.parse_args();prepare() if a.command=='prepare' else execute(a.result)
