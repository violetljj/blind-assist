"""Posthoc ensemble diagnostic on frozen pilot2 scores; no promotion/training.

New M3 per-seed inference caches are task-owned; original scores stay immutable.
Three-model means average raw logits before the frozen causal five-score readout.
"""
import argparse
from pathlib import Path
import numpy as np
import cnh_unknown_target_reference_evaluate as U

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'artifacts.local/work/cnh-readout-pilot2-20261005'
OUT=ROOT/'artifacts.local/work/cnh-t2-fit-diagnostic-20261004'
SEED=2026100601
PRIMARY='primary1.2-2.1m'


def inspect(out=OUT):
    paths=[out/'predictions/m3_original_batch'/f'M3_seed{s}.npz' for s in range(5)]
    return dict(existing_M3_mean5=str(SOURCE/'predictions/M3_fresh_evaluation.npz'),
        available_seed_scores=[str(p) for p in paths if p.exists()],missing_seed_scores=[str(p) for p in paths if not p.exists()],
        required_source=dict(rows=str(SOURCE/'inputs/fresh_evaluation/rows.npz'),
            voxels=str(SOURCE/'inputs/fresh_evaluation/voxels.npy'),
            checkpoints=[str(ROOT/'artifacts.local/work/cnh-margin-labels-20261002/models/M3'/f'model_seed{s}.pt') for s in range(5)]),
        rule='Missing per-seed scores require explicit frozen-weight inference; this module never launches GPU or reconstructs them from a mean')


def ordered_smooth(raw,rows,units):
    order=np.lexsort((rows['frame'],rows['replica'],rows['variant'],rows['unit']))
    ids=np.column_stack([rows[k][order] for k in ('unit','variant','replica')]).reshape(-1,13,3)
    expected=np.array([(u,v,k) for u in units for v in range(7) for k in range(4)])
    if raw.shape!=(17472,2) or not np.isfinite(raw).all() or not np.array_equal(ids[:,0],expected) or not np.all(ids==ids[:,:1]):
        raise ValueError('Raw score/episode identity mismatch')
    if not np.array_equal(rows['frame'][order].reshape(-1,13),np.tile(np.arange(3,16),(1344,1))):raise ValueError('Expected exactly retained frames3..15')
    x=raw[order].reshape(-1,13,2);smoothed=np.empty(x.shape,np.float64)
    for t in range(13):
        start=max(0,t-4);w=np.array([1.,2.,4.,8.,16.])[-(t-start+1):]
        smoothed[:,t]=(x[:,start:t+1]*w[None,:,None]).sum(1)/w.sum()
    return smoothed.reshape(48,7,4,13,2)


def evaluate(out=OUT):
    if (out/'ensemble.json').exists():raise FileExistsError('Completed diagnostic remains immutable')
    plan=U.read(out/'PLAN.json')
    if plan['bootstrap']['n']!=1000 or plan['bootstrap']['seed']!=SEED:raise ValueError('Diagnostic PLAN must freeze1000 paired-scene draws and seed2026100601')
    state=inspect(out)
    if state['missing_seed_scores']:raise FileNotFoundError(state['missing_seed_scores'])
    hashes={str(out/'PLAN.json'):U.sha(out/'PLAN.json')}
    def read(path):hashes[str(path)]=U.sha(path);return U.read(path)
    def npz(path):
        hashes[str(path)]=U.sha(path)
        with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def npy(path):hashes[str(path)]=U.sha(path);return np.load(path,allow_pickle=False).astype(np.float64)
    original_plan=read(SOURCE/'PLAN.json');units=original_plan['eval_units']
    if len(units)!=48 or len(set(units))!=48:raise ValueError('Require original full48-unit cohort')
    rows=npz(SOURCE/'inputs/fresh_evaluation/rows.npz')
    truth=[read(SOURCE/'truth/evaluation'/f'unit{u}.json') for u in units]
    if [t['unit'] for t in truth]!=units:raise ValueError('Truth unit order mismatch')
    group=np.array([t['group'] for t in truth]);ranges=np.array([t['front_range_m'][3:16] for t in truth])
    mraw=[];vraw=[]
    for seed in range(5):
        path=out/'predictions/m3_original_batch'/f'M3_seed{seed}.npz';z=npz(path);r=read(path.with_suffix('.json'))
        checkpoint=ROOT/'artifacts.local/work/cnh-margin-labels-20261002/models/M3'/f'model_seed{seed}.pt'
        if r['status']!='COMPLETE' or r['model_sha256']!=U.sha(checkpoint):raise ValueError('Per-seed M3 checkpoint receipt mismatch')
        score_hash=r.get('raw_sha256',r.get('score_sha256'))
        if score_hash!=U.sha(path):raise ValueError('Per-seed M3 cache hash mismatch')
        if r.get('plan_sha256')!=hashes[str(out/'PLAN.json')]:raise ValueError('Per-seed M3 belongs to another diagnostic PLAN')
        if r.get('row_sha256')!=hashes[str(SOURCE/'inputs/fresh_evaluation/rows.npz')]:raise ValueError('Per-seed M3 row identity receipt mismatch')
        mraw.append(np.asarray(z['raw'],np.float64))
    for seed in range(3):
        vraw.append(npy(SOURCE/'scores/fresh_evaluation'/f'V_retest_seed{seed}.npy'))
    mean5=npz(SOURCE/'predictions/M3_fresh_evaluation.npz')['raw'].astype(np.float64)
    gpu_mean5=npz(out/'predictions/m3_original_batch/original_GPU_mean5.npz')['raw'].astype(np.float64)
    parity=read(out/'predictions/m3_original_batch/parity.json')
    if parity['status']!='PASS' or parity['plan_sha256']!=hashes[str(out/'PLAN.json')]:raise ValueError('Original batch parity receipt missing or wrong PLAN')
    gpu_mean5_error=float(np.abs(gpu_mean5-mean5).max())
    if gpu_mean5_error>=1e-6:raise ValueError(f'Original batch64 GPU five-model cache not reproduced:{gpu_mean5_error}')
    # The validation repeats the original GPU float32 reduction. Diagnostic
    # means deliberately use real-arithmetic float64; rounding is not a recipe.
    mean5_error=float(np.abs(np.mean(mraw,axis=0)-mean5).max())
    rawarms={f'M3_seed{s}':x for s,x in enumerate(mraw)}
    rawarms.update({f'V_seed{s}':x for s,x in enumerate(vraw)})
    rawarms.update(M3_mean012=np.mean(mraw[:3],axis=0),V_mean012=np.mean(vraw,axis=0),M3_frozen_mean5=mean5,
        M3_mathematical_mean5=np.mean(mraw,axis=0))
    values={}
    for arm,raw in rawarms.items():
        score=ordered_smooth(raw,rows,units);v=[]
        for i in range(48):
            keep=(ranges[i]>=1.2)&(ranges[i]<2.1);q=score[i,...,group[i]]
            v.append(U.binary_auc(q[:2][...,keep].ravel(),q[[4,5,6]][...,keep].ravel()))
        values[arm]=np.array(v)
    rng=np.random.default_rng(SEED)
    boot=np.stack([np.bincount(rng.integers(48,size=48),minlength=48) for _ in range(1000)])
    def summary(v):
        return dict(value=float(v.mean()),ci95=np.quantile(boot@v/48,[.025,.975]).tolist(),scenes=48)
    comparisons={}
    pairs=[(f'V_seed{s}',f'M3_seed{s}') for s in range(3)]
    pairs += [('V_mean012','M3_mean012')]+[(f'V_seed{s}','M3_frozen_mean5') for s in range(3)]
    pairs += [('V_mean012','M3_frozen_mean5'),('M3_mean012','M3_frozen_mean5')]
    for a,b in pairs:comparisons[a+'_minus_'+b]=summary(values[a]-values[b])
    mean5_auc_error=float(np.abs(values['M3_mathematical_mean5']-values['M3_frozen_mean5']).max())
    old=read(SOURCE/'result.json');nat=old['natural']['cells'];height_costs={}
    for height in ('all','same_height','other_height'):
        v=nat['V_retest']['subgroups'][height]['metrics']['clear'];vd=nat['VD']['subgroups'][height]['metrics']['clear']
        height_costs[height]=dict(V_stops=v['expected_stops'],VD_stops=vd['expected_stops'],difference=vd['expected_stops']-v['expected_stops'],
            n=v['expected_episodes'],proxy_minutes=v['clear_minutes'])
    if height_costs['same_height']['difference']!=-5 or height_costs['other_height']['difference']!=-14:raise ValueError('Frozen natural cost decomposition changed')
    for p,h in hashes.items():
        if U.sha(p)!=h:raise ValueError('Diagnostic input changed')
    result=dict(status='COMPLETE_POSTHOC_DIAGNOSTIC',macroauc={a:summary(v) for a,v in values.items()},paired_differences=comparisons,
        per_scene=[dict(unit=u,auc={a:float(v[i]) for a,v in values.items()}) for i,u in enumerate(units)],
        M3_original_batch_GPU_mean5_raw_max_difference=gpu_mean5_error,
        M3_mathematical_mean5_raw_max_difference=mean5_error,M3_mathematical_mean5_per_scene_auc_max_difference=mean5_auc_error,
        natural_VD_minus_V_clear_height_decomposition=height_costs,
        bootstrap=dict(n=1000,seed=SEED,cluster='whole48scenes,common paired weights',condition='frozen models and raw caches'),
        interpretation=[
            'Matched-seed V_s−M3_s and V3−M3{0,1,2} isolate the comparison from a3-vs5 ensemble-count mismatch. They retain the same generator/training-family limitations.',
            'AUC of mean logits differs from mean of per-seed AUCs. Ensemble uplift alone does not prove a better single-model representation.',
            'VD teacher labels only directly supervise valid displacement target-height queries. Natural/other-height queries keep hard labels; other-height clear−14 cannot establish a direct teacher mechanism.',
            'Shared parameters, changed target objective and correlated outputs can change unsupervised-other-query behavior. Cost counts are descriptive, not causal attribution.',
            'Posthoc fit-bottleneck diagnosis; no candidate gate, threshold change, M3 replacement or hardware claim.'],
        provenance=dict(input_sha256=hashes,source_sha256=U.sha(__file__),original_pilot_plan_sha256=U.sha(SOURCE/'PLAN.json')))
    lines=['# T2拟合瓶颈：集成拆分事后诊断','',
        '| 臂 | 主48场景macroAUC | 95%区间 |','|---|---:|---:|']
    for a,m in result['macroauc'].items():lines.append(f"| {a} | {m['value']:.6f} | [{m['ci95'][0]:.6f},{m['ci95'][1]:.6f}] |")
    lines+=['','| 配对比较 | 差 | 95%区间 |','|---|---:|---:|']
    for a,m in comparisons.items():lines.append(f"| {a} | {m['value']:+.6f} | [{m['ci95'][0]:+.6f},{m['ci95'][1]:+.6f}] |")
    lines+=['',f'原batch64/GPU float32五种子mean复现原raw最大差{gpu_mean5_error:.9g}（要求<1e−6）。独立float64数学平均raw差{mean5_error:.9g}，其逐场景AUC最大差{mean5_auc_error:.9g}，单独描述浮点规约误差。所有集成先平均raw logits再用五帧因果平滑；AUC不能按单seed AUC线性平均。','',
        'VD−V自然清晰首停分解：同高度76→71（−5），其他高度136→122（−14）；全部212→193（−19），原冻结阈值不变。教师没有直接对其他高度或自然查询加软目标，这组变化不能作为教师直接改善另一高度的机制证据。','',
        '同seed与同三seed集成比较减少集成数量不齐造成的混淆；仍条件于冻结已训练模型、同一模拟器和既有消费批。集成增益不能证明单模型表示更好。',
        '这些是事后诊断，不跑候选门槛，不改变M3，不是实机结果；共享参数及目标查询训练变化可能间接影响另一高度输出，计数不能单独归因。','']
    (out/'ENSEMBLE.md').write_text('\n'.join(lines),encoding='utf8');U.save(out/'ensemble.json',result)
    print('ENSEMBLE COMPLETE',comparisons['V_mean012_minus_M3_mean012'],flush=True)
    return result


def check():
    rows=dict(unit=np.repeat(np.arange(48),7*4*13),variant=np.tile(np.repeat(np.arange(7),4*13),48),
        replica=np.tile(np.repeat(np.arange(4),13),48*7),frame=np.tile(np.arange(3,16),48*7*4))
    raw=np.arange(17472*2.).reshape(17472,2)
    x=ordered_smooth(raw,rows,list(range(48)))
    assert x.shape==(48,7,4,13,2)
    np.testing.assert_allclose(x[0,0,0,-1],(raw[8:13]*np.array([1,2,4,8,16])[:,None]).sum(0)/31)
    print('PASS ensemble diagnostic identities and frozen causal weights')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('inspect','check','evaluate'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if a.stage=='inspect':print(inspect(a.out))
    elif a.stage=='check':check()
    else:evaluate(a.out)
