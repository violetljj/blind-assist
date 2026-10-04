"""Frozen signed observations with query-anchored noisy-pose oracle templates.

Known scene and current true sensor anchor remain privileged. Only causal
relative pose in past candidate expectation templates changes; photon draws,
quadrature, template priors, scene selection and M3 outputs remain frozen.
CPU ray tracing is TASK_NOT_GPU_SUITABLE; no model inference or training.
"""
import os
for _threads in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_threads]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import cnh_displacement_ceiling_render as R
import cnh_displacement_ceiling_evaluate as E

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'artifacts.local/work/cnh-displacement-ceiling-20261003'
OUT=ROOT/'artifacts.local/work/cnh-memory-reality-20261004/pose_oracle'
RUN='CNH_MEMORY_POSE_ORACLE_20261004'
CANDIDATES=[0,1,4,5,6]
WINDOWS=(8,12,16)
ARMS=('noisy8','noisy12','noisy16','true16','M3_smooth')
BINS={'primary1.2-2.1m':(1.2,2.1),'near0.9-1.2m':(.9,1.2),'combined0.9-2.1m':(.9,2.1)}
BUDGET=3600
BOOT_SEED=2026100402
N_BOOT=1000
read=E.read
sha=E.sha
save=E.save


def anchor_history(sensor,noisy,frame):
    """At t<=f: S_f @ inv(N_f) @ N_t. Never access future poses."""
    if frame<0 or frame>=len(sensor) or frame>=len(noisy):
        raise ValueError('Anchor frame out of range')
    return sensor[frame]@np.linalg.inv(noisy[frame])@noisy[:frame+1]


def bind_inputs():
    previous=read(SOURCE/'result.json')
    if previous.get('status')!='COMPLETE' or len(previous['units'])!=48:
        raise ValueError('Complete48scene displacement result required')
    hashes={str(SOURCE/'result.json'):sha(SOURCE/'result.json'),str(SOURCE/'PLAN.json'):sha(SOURCE/'PLAN.json')}
    for unit in previous['units']:
        for folder,suffix in (('observations','.npz'),('templates','.npz'),('truth','.json'),('scores','.npz')):
            path=SOURCE/folder/f'unit{unit}{suffix}'
            digest=sha(path)
            if previous['provenance']['input_sha256'].get(str(path))!=digest:
                raise ValueError('Frozen displacement payload changed: '+str(path))
            hashes[str(path)]=digest
    return previous,hashes


def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'PLAN.json'
    if path.exists():
        record=read(path)
        if record['run']!=RUN:
            raise ValueError('Existing pose oracle PLAN belongs to another run')
        return record
    previous,hashes=bind_inputs()
    sources=R.source_sha256()
    sources.update({str(Path(__file__).resolve()):sha(__file__),str(Path(E.__file__).resolve()):sha(E.__file__)})
    record=dict(run=RUN,units=previous['units'],input_sha256=hashes,source_sha256=sources,
        role='Consumed synthetic Development; known-scene conditional diagnostic; no training',
        candidate_delta_indices=CANDIDATES,candidate_intrusion_cm=[1,2,-10,-15,-20],
        primary_positive_indices=[0,1],primary_negative_indices=[2,3,4],class_prior='Equal class mass; uniform mixture within class',
        K=4,quadrature_subdivision=16,windows=list(WINDOWS),truepose_reference_window=16,
        bins=BINS,decision_frames='Only original frame3..15 whose target front is >=.9 and<2.1; all available t<=f remain potential history',
        pose_rule='Current-query sensor anchor: sensor[f] @ inv(noisy[k,f]) @ noisy[k,t], t<=f',
        observation_rule='Original signed hist and ambient frozen; never counts or ambient_estimate',
        visibility_rule='Prior target-return fraction over five primary branches in1.2-2.1m >=.5;36 high/12 low; posthoc original subgroup retained',
        condition=dict(stratum='low_visible',bin='combined0.9-2.1m',criterion='noisy12 OR noisy16 scene macro AUC>=.85; report each separately',
            automatic_training=False,interpretation='Known-scene noisy-pose ceiling only; not evidence a trainable memory model or real sensor reaches it'),
        bootstrap=dict(replicates=N_BOOT,seed=BOOT_SEED,cluster='whole base scene'),
        compute=dict(classification='TASK_NOT_GPU_SUITABLE',workers=4,BLAS_threads_per_worker=1,backend='frozen NumPy CPU expected renderer; exact Skellam likelihood'),
        budget_seconds=BUDGET,scheduling_stop_before_seconds=3300,
        limits=['Current true sensor anchor and full scene boxes are privileged; only relative historical pose is perturbed',
            'Changing pose templates is conditional model mismatch, not adding observations or physical data',
            'Low-visible12scenes all mode0; scene/context/FOV confounds remain',
            'Outside10cm is pass0-10cm; primary task is nominal branch discrimination',
            'No baseline promotion, physical safety, device calibration or unknown-target deployment claim'])
    save(path,record)
    print('PLAN',sha(path),flush=True)
    return record


def validate_sources(record):
    for path,digest in record['source_sha256'].items():
        if sha(path)!=digest:
            raise ValueError('Pose oracle source changed: '+path)


def scene_metadata(previous,unit):
    return next(s for s in previous['per_scene'] if s['unit']==unit)


def mixtures(loglike):
    return E.logsumexp(loglike[:2],axis=0)-np.log(2)-E.logsumexp(loglike[2:],axis=0)+np.log(3)


def run_scene(unit,smoke=False):
    started=time.monotonic();record=read(OUT/'PLAN.json');validate_sources(record)
    digest=sha(OUT/'PLAN.json')
    output=OUT/'units'/f'unit{unit}.json'
    if output.exists():
        receipt=read(output)
        if receipt['plan_sha256']!=digest or receipt['score_sha256']!=sha(OUT/receipt['score_file']):
            raise ValueError('Existing unit identity differs')
        return receipt
    previous=read(SOURCE/'result.json');meta=scene_metadata(previous,unit)
    with np.load(SOURCE/'observations'/f'unit{unit}.npz',allow_pickle=False) as obs:
        hist,ambient,sensor,noisy=obs['hist'],obs['ambient'],obs['sensor'],obs['noisy']
    with np.load(SOURCE/'templates'/f'unit{unit}.npz',allow_pickle=False) as templates:
        expected=templates['expected']
    with np.load(SOURCE/'scores'/f'unit{unit}.npz',allow_pickle=False) as z:
        raw=z['raw']
    truth=read(SOURCE/'truth'/f'unit{unit}.json')
    if hist.shape!=(7,4,16,8,8,16) or noisy.shape!=(4,16,4,4) or expected.shape!=(7,16,8,8,16):
        raise ValueError('Frozen scene axes differ')
    ranges=np.array(truth['front_range_m'])
    decision=E.FRAMES[(ranges[E.FRAMES]>=.9)&(ranges[E.FRAMES]<2.1)]
    if not len(decision):
        raise ValueError('Scene lacks declared decision distances')
    visibility=meta['target_visibility']['primary1.2-2.1m']
    visibility_fraction=sum(visibility[str(d)]['frames_with_target_return'] for d in (1,2,-10,-15,-20))/sum(visibility[str(d)]['frames'] for d in (1,2,-10,-15,-20))
    scores={arm:np.empty((5,4,len(decision)),float) for arm in ARMS}
    scores['M3_smooth']=E.m3_smooth(raw)[CANDIDATES,:,:,truth['group']][:,:,decision-3]
    # One fixed per-frame likelihood pass for the true-pose16 reference.
    true_ll=np.empty((5,5,4,16),float)
    for ci,candidate in enumerate(CANDIDATES):
        value=E.skellam_logpmf_signed(hist[CANDIDATES],expected[candidate][None,None],8*ambient[None,None,...,None])
        true_ll[ci]=value.sum(axis=(-3,-2,-1))
    true_frame=mixtures(true_ll)
    # Mix after window likelihood totals, never independently per exposure.
    true_total=E.rolling_sum(true_ll,16)
    scores['true16']=mixtures(true_total)[...,decision]
    smoke_checks=None
    if smoke:
        error=0.
        for frame in (3,15):
            anchored=anchor_history(sensor,sensor,frame)
            np.testing.assert_allclose(anchored,sensor[:frame+1],atol=2e-14,rtol=0)
            value=R.expected(dict(poses=anchored,boxes=truth['boxes'][0]))
            error=max(error,float(np.max(np.abs(value['expectation']-expected[0,:frame+1]))))
        if error>1e-9:
            raise ValueError('True-pose anchor expectation parity failed: '+str(error))
        changed=noisy[0].copy();changed[14:,:3,3]+=100
        np.testing.assert_array_equal(anchor_history(sensor,noisy[0],13),anchor_history(sensor,changed,13))
        # The unchanged true-pose12 kernel reproduces the frozen original macro.
        true12=mixtures(E.rolling_sum(true_ll,12))[...,decision]
        keep=(ranges[decision]>=1.2)&(ranges[decision]<2.1)
        auc=E.binary_auc(true12[:2,:,keep].ravel(),true12[2:,:,keep].ravel())
        np.testing.assert_allclose(auc,meta['cells']['primary1.2-2.1m']['oracle12']['auc'],atol=1e-13)
        smoke_checks=dict(status='PASS',true_anchor_expectation_max_error=error,no_future_pose_influence=True,true12_original_AUC_parity=True)
    for k in range(4):
        for fi,frame in enumerate(decision):
            anchored=anchor_history(sensor,noisy[k],int(frame))
            ll=np.empty((5,5,frame+1),float)
            for ci,candidate in enumerate(CANDIDATES):
                mean=R.expected(dict(poses=anchored,boxes=truth['boxes'][candidate]))
                if not np.array_equal(mean['ambient'],ambient[:frame+1]):
                    raise ValueError('Pose template changed the photon/ambient budget')
                value=E.skellam_logpmf_signed(hist[CANDIDATES,k,:frame+1],mean['expectation'][None],8*ambient[None,:frame+1,...,None])
                ll[ci]=value.sum(axis=(-3,-2,-1))
            for window in WINDOWS:
                totals=ll[...,max(0,frame-window+1):].sum(-1)
                scores[f'noisy{window}'][:,k,fi]=mixtures(totals)
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError('Nonfinite pose-oracle score')
    cells={}
    for name,(low,high) in BINS.items():
        keep=(ranges[decision]>=low)&(ranges[decision]<high)
        cells[name]={}
        for arm,value in scores.items():
            pos,neg=value[:2,:,keep].ravel(),value[2:,:,keep].ravel()
            cells[name][arm]=dict(auc=E.binary_auc(pos,neg),positive_n=len(pos),negative_n=len(neg))
    score=OUT/'scores'/f'unit{unit}.npz';score.parent.mkdir(exist_ok=True,parents=True)
    with score.open('xb') as stream:
        np.savez_compressed(stream,frames=decision,**scores)
    receipt=dict(status='COMPLETE',unit=unit,plan_sha256=digest,score_file=str(score.relative_to(OUT)).replace('\\','/'),score_sha256=sha(score),
        group=truth['group'],mode=truth['mode'],context=truth['context'],visibility_fraction=visibility_fraction,
        stratum='high_visible' if visibility_fraction>=.5 else 'low_visible',frames=decision.tolist(),cells=cells,
        elapsed_s=time.monotonic()-started,smoke_checks=smoke_checks)
    output.parent.mkdir(exist_ok=True,parents=True);save(output,receipt)
    print('completed pose scene',unit,'seconds',round(receipt['elapsed_s'],2),flush=True)
    return receipt


def aggregate(units):
    if len(units)!=48:
        raise ValueError('Full48scene aggregation required')
    rng=np.random.default_rng(BOOT_SEED)
    boot=np.array([np.bincount(rng.integers(48,size=48),minlength=48) for _ in range(N_BOOT)])
    strata={'all':np.ones(48,bool),'high_visible':np.array([u['stratum']=='high_visible' for u in units]),
            'low_visible':np.array([u['stratum']=='low_visible' for u in units])}
    if strata['high_visible'].sum()!=36 or strata['low_visible'].sum()!=12:
        raise ValueError('Prior visibility stratum identity differs')
    cells={}
    pooled={label:{b:{a:[[],[]] for a in ARMS} for b in BINS} for label in strata}
    for ui,unit in enumerate(units):
        meta=scene_metadata(read(SOURCE/'result.json'),unit['unit'])
        front=np.array(meta['front_range_m'])[np.array(unit['frames'])-3]
        with np.load(OUT/unit['score_file'],allow_pickle=False) as z:
            for name,(low,high) in BINS.items():
                keep=(front>=low)&(front<high)
                for arm in ARMS:
                    value=z[arm]
                    for label,selection in strata.items():
                        if selection[ui]:
                            pooled[label][name][arm][0].append(value[:2,:,keep].ravel())
                            pooled[label][name][arm][1].append(value[2:,:,keep].ravel())
    for label,selection in strata.items():
        cells[label]={}
        for name in BINS:
            cells[label][name]={}
            for arm in ARMS:
                values=np.array([u['cells'][name][arm]['auc'] for u in units])
                if not np.isfinite(values).all():
                    raise ValueError('Unknown/missing scene AUC; preserve denominator')
                weighted=boot@np.where(selection,values,0)
                denominator=boot@selection.astype(float)
                sampled=np.divide(weighted,denominator,out=np.full(N_BOOT,np.nan),where=denominator>0)
                pos,neg=(np.concatenate(v) for v in pooled[label][name][arm])
                cells[label][name][arm]=dict(scene_macro_AUC=float(values[selection].mean()),ci95=np.nanpercentile(sampled,[2.5,97.5]).tolist(),
                    pooled_AUC=E.binary_auc(pos,neg),scenes=int(selection.sum()),positive_n=len(pos),negative_n=len(neg),unknown_scenes=0,
                    per_scene_min=float(values[selection].min()),per_scene_max=float(values[selection].max()))
    combined=cells['low_visible']['combined0.9-2.1m']
    passed={arm:combined[arm]['scene_macro_AUC']>=.85 for arm in ('noisy12','noisy16')}
    return dict(status='COMPLETE',verdict='B_CONDITION_MET_KNOWN_SCENE' if any(passed.values()) else 'B_CONDITION_NOT_MET',
        cells=cells,units=[u['unit'] for u in units],per_scene=units,
        condition=dict(stratum='low_visible12scenes',bin='combined0.9-2.1m',cut=.85,checks=passed,
            automatic_training=False,interpretation='Conditional feasibility evidence only; unknown scene/target and hardware effectiveness remain open'),
        bootstrap=dict(replicates=N_BOOT,seed=BOOT_SEED,cluster='whole base scenes'),
        n=dict(base_scenes=48,low_visible_scenes=12,high_visible_scenes=36,photon_replicas=4,candidates=5))


def report(result):
    lines=[result['verdict'],'','# 已知场景：真实当前锚点与含噪历史姿态的记忆oracle','',
        '原48场景signed CNH观测冻结；只把候选期望的历史sensor姿态换为S_f @ inv(N_f) @ N_t（t≤f）。当前sensor锚点和完整场景仍是特权信息；8/12/16窗均因果。原M3推理结果保留，不训练、不重新推理。','',
        '主候选为内1/2cm与外10/15/20cm，类别先验各1/2，类内均匀混合；外10cm仍是pass0–10cm。本任务是名义横向分支辨识，不等于必须报与真正clear的完整报警任务。','',
        '| 分层/距离 | 含噪8宏AUC [95%CI] | 含噪12 | 含噪16 | 真姿态16 | M3平滑 | 正/负样本 |','| --- | --- | --- | --- | --- | --- | --- |']
    for label,bins in result['cells'].items():
        for name,arms in bins.items():
            texts=[]
            for arm in ARMS:
                c=arms[arm];texts.append(f"{c['scene_macro_AUC']:.4f} [{c['ci95'][0]:.4f},{c['ci95'][1]:.4f}]")
            c=arms['noisy12']
            lines.append('| '+label+'/'+name+' | '+' | '.join(texts)+f" | {c['positive_n']}/{c['negative_n']} (s={c['scenes']}) |")
    lines+=['','| 低可见12场景pooled AUC | 含噪8 | 含噪12 | 含噪16 | 真姿态16 | M3平滑 |','| --- | --- | --- | --- | --- | --- |']
    for name,arms in result['cells']['low_visible'].items():
        lines.append('| '+name+' | '+' | '.join(f"{arms[a]['pooled_AUC']:.4f}" for a in ARMS)+' |')
    lines+=['','预设B条件：'+json.dumps(result['condition'],ensure_ascii=False),'',
        '宏AUC先同场景排序再等场景平均；pooled AUC含跨场景比较，均不是冻结单阈值的及时率/误停率。低可见为12个基础场景，4个光子复本和多个分支/帧不是独立场景；该组全部mode0、8panel/4none，运动和context混淆仍在。','',
        '真实姿态16不是所有部署算法的保证；含噪模板也仍知道盒形、反射率、背景、当前真锚点及候选集合。B≥.85只能支持在本条件下继续问方法可行性，不能证明训练模型达到该数值，不自动启动训练。','',
        '16×16 angular quadrature、光子噪声、脉冲/增益和代理场景保持原样，未新增硬件、RGB或真实输入证据。1000次共同整场景bootstrap仅条件于既有受控Development、固定模板与噪声样本。','',
        '运行耗时：'+json.dumps(result['runtime'],ensure_ascii=False),'',
        '输入、代码及PLAN SHA见result.json；逐场景score缓存用于恢复，不重采样观测或更换子集。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')


def smoke():
    record=plan();validate_sources(record)
    unit=record['units'][0]
    scene=run_scene(unit,smoke=True)
    if scene['smoke_checks'] is None or scene['smoke_checks']['status']!='PASS':
        raise ValueError('First-scene smoke receipt missing')
    estimate=scene['elapsed_s']*47/4
    output=dict(status='PASS',unit=unit,scene_elapsed_s=scene['elapsed_s'],remaining47_fourworker_estimate_s=estimate,
        checks=scene['smoke_checks'],plan_sha256=sha(OUT/'PLAN.json'))
    save(OUT/'smoke.json',output);print(json.dumps(output),flush=True)
    return output


def run():
    started=time.monotonic();record=plan();validate_sources(record)
    smoke_record=read(OUT/'smoke.json')
    if smoke_record['status']!='PASS' or smoke_record['plan_sha256']!=sha(OUT/'PLAN.json'):
        raise ValueError('Matching first-scene smoke required')
    result_path=OUT/'result.json'
    if result_path.exists():
        previous=read(result_path)
        if previous['status']!='COMPLETE':raise ValueError('Inspect incomplete result')
        return previous
    pending_units=[];units={}
    for unit in record['units']:
        path=OUT/'units'/f'unit{unit}.json'
        if path.exists():units[unit]=run_scene(unit)
        else:pending_units.append(unit)
    budget_hit=False
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={}
        while pending_units or futures:
            if time.monotonic()-started>3300:
                budget_hit=True;pending_units=[]
            while pending_units and len(futures)<4:
                unit=pending_units.pop(0);futures[pool.submit(run_scene,unit)]=unit
            if not futures:break
            ready,_=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
            for future in ready:
                unit=futures.pop(future);units[unit]=future.result()
                print('progress',len(units),'/48','wall_s',round(time.monotonic()-started,1),flush=True)
    if budget_hit or len(units)!=48:
        save(OUT/'budget_status.json',dict(status='PARTIAL_BUDGET',completed_units=sorted(units),elapsed_s=time.monotonic()-started))
        raise TimeoutError('Budget reached; retain partial unit evidence, no full-cohort result')
    validate_sources(record)
    for path,digest in record['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen input changed during oracle run: '+path)
    result=aggregate([units[u] for u in record['units']])
    result['provenance']=dict(plan=record,plan_sha256=sha(OUT/'PLAN.json'),smoke=smoke_record,evaluator_sha256=sha(__file__))
    result['runtime']=dict(run_wall_s=time.monotonic()-started,first_scene_smoke_s=smoke_record['scene_elapsed_s'],workers=4,
        summed_scene_cpu_wall_s=sum(u['elapsed_s'] for u in units.values()),classification='TASK_NOT_GPU_SUITABLE',budget_seconds=BUDGET)
    save(result_path,result);report(result)
    print(result['verdict'],json.dumps(result['condition']),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=('plan','smoke','run'))
    args=parser.parse_args();{'plan':plan,'smoke':smoke,'run':run}[args.stage]()
