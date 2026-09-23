"""Matched fixed-scene range crossings, public readouts and evaluator attribution."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
from query_occupancy_data import read,write,sha,observation_tokens
from query_occupancy_spec import relative_signature
from tof_fov45_core import simulate,boxes45
from ba_camera_corridor import sample_native
from range_pair_observations import public_readouts,evaluator_target_mask,summarize_trace_support

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
ROOT=REPO/'artifacts.local/evidence/ba-range-pair-20260923'
COHORTS={'stability':('ba-local-stability-20260923','local-stability/'),
         'rescue':('ba-local-rescue-fresh-20260923','local-rescue-fresh/')}
RULES=('global_min','corridor_min','corridor_median3')
DRAWS=16


def pairs_from_spec(spec):
    clips=defaultdict(list)
    for i,c in enumerate(spec['cases']):clips[c['clip_id']].append((i,c))
    pairs=[];seen=set()
    for clip,seq in clips.items():
        assert [c['frame_in_clip'] for i,c in seq]==list(range(12))
        c0=seq[0][1]
        for i,c in seq:
            assert c['objects']==c0['objects']
            assert all(c['camera'][k]==c0['camera'][k] for k in ('y','z','pitch','yaw','roll'))
        for phase,near,far in [('entry',3,2),('exit',9,10)]:
            ni,n=seq[near];fi,f=seq[far]
            assert n['nominal_front_m']<=3<f['nominal_front_m']
            key=(relative_signature(n),relative_signature(f),relative_signature(seq[near-1][1]),relative_signature(seq[far-1][1]))
            pairs.append(dict(clip_id=clip,phase=phase,near=ni,far=fi,group=c0['base_group_id'],
                shape=c0['shape'],relation=c0['layout_relation'],unique=key not in seen,
                near_truth_m=n['nominal_front_m'],far_truth_m=f['nominal_front_m']))
            seen.add(key)
    return pairs


def measure(readouts,previous):
    result={}
    public=public_readouts(readouts)
    for rule in RULES:
        r=public[rule];zones=r['selected_zone_ids']
        both=[j for j in zones if readouts[j,1]==1 and previous[j,1]==1]
        delta=float(np.median((readouts[both,0]-previous[both,0])*8)) if both else None
        result[rule]=dict(**r,causal_delta_m=delta,history_common_zones=len(both))
    return result


def summary(records):
    intended=[r for r in records if r['relation']!='OUTSIDE']
    outside=[r for r in records if r['relation']=='OUTSIDE']
    out={}
    for rule in RULES:
        rows=[r['readouts'][rule] for r in intended]
        valid=[r for r in rows if r['near_m'] is not None and r['far_m'] is not None]
        both=lambda r:bool(r['near_m'] is not None and r['far_m'] is not None and .3<=r['near_m']<=3<r['far_m'])
        endpoints=[r['readouts'][rule][key] for r in outside for key in ('near_m','far_m')]
        witnessed=[r for r in rows if r['near_target'] and r['far_target']]
        nonwitness=[r for r in rows if not(r['near_target'] and r['far_target'])]
        out[rule]=dict(pairs=len(rows),available_pairs=len(valid),missing_pairs=len(rows)-len(valid),
            correct_crossings=sum(both(r) for r in rows),joint_rate=sum(both(r) for r in rows)/len(rows),
            near_ranked_closer=sum(r['near_m']<r['far_m'] for r in valid),
            ranking_all_pairs=sum(r['near_m']<r['far_m'] for r in valid)/len(rows),
            target_witness_both=len(witnessed),target_witness_joint=sum(both(r) for r in witnessed),
            missing_target_witness=len(nonwitness),nonwitness_joint=sum(both(r) for r in nonwitness),
            outside_endpoints=len(endpoints),outside_missing=sum(x is None for x in endpoints),
            outside_alerts=sum(x is not None and .3<=x<=3 for x in endpoints),
            outside_FPR=sum(x is not None and .3<=x<=3 for x in endpoints)/len(endpoints))
    return out


def prepare():
    sources=[Path(__file__),HERE/'range_pair_observations.py',HERE/'RANGE_PAIR_PROTOCOL_20260923.md',
        HERE/'tof_fov45_core.py',HERE/'ba_camera_corridor.py',HERE/'query_occupancy_data.py',HERE/'query_occupancy_spec.py']
    write(ROOT/'plan/seal.json',dict(files={p.relative_to(REPO).as_posix():sha(p) for p in sources}))
    inputs=[dict(alias='plan',path=str(ROOT/'plan'),role='configuration',purpose='fixed-pair-and-noise-diagnostic')]
    for name,(stem,prefix) in COHORTS.items():
        base=ROOT.parent/stem
        for alias,path,role in [('spec',base/'plan/spec.json','configuration'),
            ('observations',base.with_name(stem+'-prepared')/'observations','observation'),
            ('manifest',base.with_name(stem+'-prepared')/'materialization.json','configuration'),
            ('native',base.with_name(stem+'-capture')/'evaluator','evaluator'),
            ('labels',base.with_name(stem+'-prepared')/'labels/evaluation.npz','evaluator')]:
            inputs.append(dict(alias=name+'_'+alias,path=str(path),role=role,purpose='consumed-paired-range-observability'))
    write(ROOT/'run-spec.json',dict(schema='blindassist-asset-run-v1',id='range-pair-20260923-v1',route='ue-range-pair',
        question='Are matched3m crossings distinguishable from public RGB ToF and causal history under fixed sensor noise?',
        evaluator='research/active/dtr-r0/nearfield/range_pair_diagnostic.py',evidence_boundary='Consumed controlled Development; repeated noise is not new scenes',
        reuse=dict(mode='diagnostic',query='LOCAL stability rescue fixed-scene crossing3m entry exit native sensor noise'),
        inputs=inputs,outputs=[dict(alias='result',path=str(ROOT.with_name(ROOT.name+'-run')/'result.json'),role='result',required=True)],
        result_output='result',command=[sys.executable,str(Path(__file__)),'execute','--result','{{output:result}}']))


def execute(result):
    assert read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])['state']=='running'
    start=time.perf_counter();out=result.parent;seal=read(ROOT/'plan/seal.json')
    for f,h in seal['files'].items():assert sha(REPO/f)==h,f
    sys.path.insert(0,str(REPO))
    from tools.research_backend import BackendCandidate,DeviceObservation,select_backend
    select_backend('scalar-scoring',cpu=BackendCandidate('fixed-numpy-sensor','cpu',lambda:sum(range(10)),
        lambda _:DeviceObservation('cpu',platform.processor(),'NumPy '+np.__version__)),
        cpu_reason='TASK_NOT_GPU_SUITABLE',record_path=out/'backend.json')
    all_reports={};all_records=[];parity=0
    for cohort,(stem,prefix) in COHORTS.items():
        root=ROOT.parent/stem;prep=root.with_name(stem+'-prepared');cap=root.with_name(stem+'-capture')
        spec=read(root/'plan/spec.json');geo=read(cap/'evaluator/geometry.json');manifest=read(prep/'materialization.json')
        for f in ('tof.npy','rgb.npy','identities.json'):
            assert sha(prep/'observations'/f)==manifest['hashes']['observations/'+f]
        tof=np.load(prep/'observations/tof.npy');rgb=np.load(prep/'observations/rgb.npy',mmap_mode='r')
        pairs=pairs_from_spec(spec);needed=sorted({i for p in pairs for k in ('near','far') for i in (p[k],p[k]-1)})
        observations={};traces={};depths={};hashes={}
        for count,i in enumerate(needed):
            case=spec['cases'][i];g=geo[i];path=cap/'evaluator'/g['native_path']
            assert sha(path)==g['native_sha256'];hashes[str(i)]=sha(path)
            native=np.load(path);depths[i]=native
            values,trace=simulate(sample_native(native),prefix+case['sensor_noise_key'],boxes45())
            public=observation_tokens(values,boxes45())
            assert np.array_equal(public,tof[i]),(cohort,i,'saved ToF mismatch');parity+=1
            observations[i]={-1:public};traces[i]={-1:trace}
            for draw in range(DRAWS):
                values,trace=simulate(sample_native(native),'range-pair/'+cohort+'/'+case['sensor_noise_key']+'/'+str(draw),boxes45())
                observations[i][draw]=observation_tokens(values,boxes45());traces[i][draw]=trace
            if count%72==0:print('RESAMPLE',cohort,count,'/',len(needed),flush=True)
        public={str(i):{str(draw):measure(observations[i][draw],observations[i-1][draw])
            for draw in range(-1,DRAWS)} for i in sorted({p[k] for p in pairs for k in ('near','far')})}
        write(out/(cohort+'-public.json'),public)
        write(out/(cohort+'-public-seal.json'),dict(public_sha256=sha(out/(cohort+'-public.json')),
            native_hashes=hashes,truth_or_attribution_joined=False,frames=len(public),draws_per_frame=17))
        # Only this evaluator block joins world labels and target-contributor masks.
        labpath=prep/'labels/evaluation.npz';assert sha(labpath)==manifest['hashes']['labels/evaluation.npz']
        lab=np.load(labpath)
        masks={}
        for i in map(int,public):masks[i]=evaluator_target_mask(depths[i],geo[i])
        records=[]
        for p in pairs:
            ni,fi=p['near'],p['far']
            assert lab['valid'][[ni,fi]][:,[1,4]].all()
            truth=[bool((lab['classes'][i,[1,4]]<6).any()) for i in (ni,fi)]
            assert truth==([True,False] if p['relation']!='OUTSIDE' else [False,False])
            l1=float(np.abs(rgb[ni].astype(np.float32)-rgb[fi].astype(np.float32)).mean()/255)
            for draw in range(-1,DRAWS):
                r=dict(cohort=cohort,**p,draw=draw,rgb_L1=l1,readouts={})
                for rule in RULES:
                    n,f=(public[str(i)][str(draw)][rule] for i in (ni,fi))
                    ns,fs=(summarize_trace_support(traces[i][draw],v['selected_zone_ids'],masks[i]) for i,v in ((ni,n),(fi,f)))
                    r['readouts'][rule]=dict(near_m=n['estimate_m'],far_m=f['estimate_m'],
                        near_target=ns['target_witnessed'],far_target=fs['target_witnessed'],
                        near_support=ns,far_support=fs,near_delta=n['causal_delta_m'],far_delta=f['causal_delta_m'])
                records.append(r)
        dwell=[]
        for i,c in enumerate(spec['cases']):
            if c['trajectory']=='approach_dwell_return' and c['frame_in_clip'] in (5,6,7,8):
                assert relative_signature(c)==relative_signature(spec['cases'][i-1])
                dwell.append(float(np.abs(rgb[i].astype(np.float32)-rgb[i-1].astype(np.float32)).mean()/255))
        strata={part:summary([r for r in records if r['unique'] and (r['draw']==-1 if part=='saved' else r['draw']>=0)]) for part in ('saved','noise')}
        stratified={axis:{value:{part:summary([r for r in records if r['unique'] and r[axis]==value and
            (r['draw']==-1 if part=='saved' else r['draw']>=0)]) for part in ('saved','noise')}
            for value in sorted({r[axis] for r in records})} for axis in ('phase','shape')}
        # Each shape/phase includes both positive and OUTSIDE controls.
        all_reports[cohort]=dict(summary=strata,strata=stratified,pairs=len(pairs),unique_pairs=sum(p['unique'] for p in pairs),
            geometry_groups=len({p['group'] for p in pairs}),saved_rgb_pair_L1=[r['rgb_L1'] for r in records if r['draw']==-1 and r['unique']],
            dwell_rgb_L1=dwell,source_spec_sha256=sha(root/'plan/spec.json'))
        all_records.extend(records)
        write(out/(cohort+'-report.json'),all_reports[cohort])
        del observations,traces,depths,rgb
    passing=[rule for rule in RULES if rule!='global_min' and all(
        rep['summary'][part][rule]['joint_rate']>=.8 and rep['summary'][part][rule]['outside_FPR']<=.1
        for rep in all_reports.values() for part in ('saved','noise'))]
    write(out/'pairs.json',all_records)
    write(result,dict(status='PASS',decision='PUBLIC_RANGE_COMPONENT' if passing else 'PUBLIC_RANGE_ADMISSION_NOT_ESTABLISHED',
        passing=passing,summaries={k:v['summary'] for k,v in all_reports.items()},original_ToF_parity_frames=parity,
        recorded_pairs=192,unique_geometry_pairs=sum(r['unique_pairs'] for r in all_reports.values()),
        fits=0,cutoff_selections=0,new_capture_frames=0,noise_draws=16,elapsed_s=time.perf_counter()-start))
    for f,h in seal['files'].items():assert sha(REPO/f)==h,f
    write(out/'output-seal.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(read(result)['decision'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute']);p.add_argument('--result',type=Path)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    else:execute(a.result)
