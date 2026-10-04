"""Pilot2 frozen-score evaluation; no training, model selection or rendering.

V_retest retains the old natural threshold/scores. VD and fitted T2 calibrate
once on the separate natural calibration cohort. All fresh decisions use the
same48 scene macroAUC and whole-scene paired bootstrap as pilot1.
"""
import argparse
from pathlib import Path
import time
import numpy as np

import cnh_temporal_readout_evaluate as E
import cnh_unknown_target_reference_evaluate as U
import cnh_sequence_observed_evaluate as OE
import cnh_three_level_sequence as SE

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-readout-pilot2-20261005'
OLD=E.OUT
FRAMES=np.arange(3,16)
PRIMARY=E.PRIMARY
DELTA=[1.,2.,5.,-5.,-10.,-15.,-20.]
BOOT_SEED=2026100503


def boot_weights(n,seed):
    rng=np.random.default_rng(seed)
    return np.asarray([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(1000)])


def t2_status(out,hashes):
    path=out/'t2_fit_acceptance.json'
    receipt=U.read(path);hashes[str(path)]=U.sha(path)
    if receipt.get('status') not in ('T2_FIT','T2_UNFIT'):
        raise ValueError('T2 fit acceptance must explicitly be T2_FIT or T2_UNFIT')
    return receipt


def fresh_geometry(out,units,hashes):
    scenes,visible=[],[]
    for unit in units:
        tp=out/'truth/evaluation'/f'unit{unit}.json'
        ep=out/'templates/evaluation'/f'unit{unit}.npz'
        truth=U.read(tp)
        if truth['unit']!=unit or truth['intrusion_cm']!=DELTA:
            raise ValueError('Fresh unit/delta identities differ from frozen PLAN')
        with np.load(ep,allow_pickle=False) as z:
            ids=z['object_id']
            flag=np.any(ids.reshape(7,16,-1)==0,axis=-1)[:,FRAMES]
        vf=float(flag[:2].mean())
        scenes.append(dict(unit=unit,group=int(truth['group']),context=truth['context'],
            front_range_m=np.asarray(truth['front_range_m'])[FRAMES].tolist(),
            visibility_fraction=vf,fov_in=vf>=.5))
        visible.append(flag)
        hashes.update({str(p):U.sha(p) for p in (tp,ep)})
    return scenes,np.stack(visible)


def fresh_scores(raw,rows,units,scenes):
    ordered,keys=E.ordered_episodes(raw,rows,fresh=True)
    expected=[(u,v,k) for u in units for v in range(7) for k in range(4)]
    if keys!=expected:raise ValueError('Incomplete/misaligned new48x7xK4 row identities')
    full=ordered.reshape(48,7,4,13,2)
    return np.stack([full[i,...,s['group']] for i,s in enumerate(scenes)])


def auc_metrics(scores,scenes,arms,boot):
    ranges=np.asarray([s['front_range_m'] for s in scenes])
    domains={f'{lo:g}-{hi:g}m':(ranges>=lo)&(ranges<hi) for lo,hi in U.BINS}
    domains.update({PRIMARY:(ranges>=1.2)&(ranges<2.1),'1.6-2.6m':(ranges>=1.6)&(ranges<2.6)})
    groups=E.strata(scenes)
    cells={};per_scene=[dict(**s,auc={}) for s in scenes]
    for domain,masks in domains.items():
        values={}
        for arm,score in scores.items():
            value=[]
            for i,mask in enumerate(masks):
                positive=score[i,[0,1]][...,mask].ravel()
                negative=score[i,[4,5,6]][...,mask].ravel()
                auc=U.binary_auc(positive,negative)
                value.append(np.nan if auc is None else auc)
                per_scene[i]['auc'].setdefault(domain,{})[arm]=dict(value=auc,positive_n=len(positive),negative_n=len(negative))
            values[arm]=np.asarray(value)
        cells[domain]={}
        for group,keep in groups.items():
            cell=dict(scenes=int(keep.sum()),arms={},paired_minus_M3={},paired_minus_V={},gap_closed={})
            for arm,val in values.items():
                cell['arms'][arm]=U.macro_summary(val,keep,boot)[0]
                if arm!='M3':cell['paired_minus_M3'][arm]=U.macro_summary(val-values['M3'],keep,boot)[0]
            for arm in arms:
                cell['gap_closed'][arm]=E.gap_fraction(values[arm]-values['M3'],values['D']-values['M3'],keep,boot)
                if arm!='V_retest':cell['paired_minus_V'][arm]=U.macro_summary(values[arm]-values['V_retest'],keep,boot)[0]
            cells[domain][group]=cell
    return cells,per_scene


def natural(out,arms,seed,hashes):
    import cnh_readout_pilot2 as C
    geometry,old_scores,source_receipt=OE.load_inputs()
    baseline=U.read(OE.OUT/'result.json');old=U.read(OLD/'result.json')
    hashes.update({str(p):U.sha(p) for p in (OE.OUT/'result.json',OE.OUT/'rows.json',OLD/'result.json')})
    cal=geometry['split']=='calib';ev=geometry['split']=='evaluation'
    candidate={}
    for split,selector in (('calibration',cal),('evaluation',ev)):
        for arm in arms:
            origin=OLD if arm=='V_retest' else out
            score_arm='V' if arm=='V_retest' else arm
            # All natural input rows remain at the original cache; new scores
            # share those identities without copied or linked input trees.
            path=C.input_folder(split)/'rows.npz'
            with np.load(path,allow_pickle=False) as z:rows={k:z[k] for k in z.files}
            hashes[str(path)]=U.sha(path)
            raw=[]
            for s in range(3):
                value,path=E.prediction(origin,split,score_arm,s);raw.append(value);hashes[str(path)]=U.sha(path)
            ordered,keys=E.ordered_episodes(np.mean(raw,axis=0),rows)
            expected=sorted(set(zip(geometry['unit'][selector].astype(int),geometry['config'][selector].astype(int))))
            if keys!=expected:raise ValueError('Natural score unit/config ordering differs from frozen OE geometry')
            actual=[(u,c,q) for u,c in keys for q in (0,1)]
            if actual!=list(zip(geometry['unit'][selector],geometry['config'][selector],geometry['query'][selector])):
                raise ValueError('Natural HEAD/BODY ordering differs from frozen OE geometry')
            candidate.setdefault(arm,{})[split]=ordered.transpose(0,2,1).reshape(-1,13)
    fields=('unit','query','frame_ranges','frame_category','ref_category','clear_all','covered','last_category')
    g={key:geometry[key][ev] for key in fields}
    unique,ui=np.unique(g['unit'],return_inverse=True);boot=boot_weights(len(unique),seed)
    weights={category:(g['covered']&(g['ref_category']==category)).astype(float) for category in OE.CONTACTS+('pass0-10cm',)}
    weights['clear']=g['clear_all'].astype(float)
    rows_geom=U.read(OE.OUT/'rows.json')
    target_groups=np.asarray([r['target_group'] for r,keep in zip(rows_geom,ev) if keep])
    subgroups={'all':np.ones(len(ui),bool),'same_height':g['query']==target_groups,'other_height':g['query']!=target_groups}
    scores={'M3':old_scores['M3'][ev]}
    thresholds={'M3':float(baseline['cells']['M3']['threshold']),'V_retest':float(old['natural']['cells']['V']['threshold'])}
    calibration={'M3':baseline['cells']['M3']['calibration'],'V_retest':old['natural']['cells']['V']['calibration']}
    for arm in arms:
        scores[arm]=candidate[arm]['evaluation']
        if arm!='V_retest':thresholds[arm],calibration[arm]=SE.calibrate(candidate[arm]['calibration'],geometry['clear_all'][cal],1.)
    if thresholds['M3']!=.8557642486787612:raise ValueError('Frozen M3 threshold changed')
    cells,draws={},{}
    for arm,score in scores.items():
        stopped,timely,lead=SE.first_stops(score,thresholds[arm],g['frame_ranges'])
        subgroup_cells={}
        for name,keep in subgroups.items():
            selected={category:w*keep for category,w in weights.items()}
            metrics,sampled=SE.summarize(selected,g['covered'],stopped,timely,lead,ui,boot)
            for metric in metrics.values():metric.pop('episode_draw_pairs',None)
            clear_num=SE.unit_totals(selected['clear']*stopped,ui,len(unique))
            clear_den=SE.unit_totals(selected['clear'],ui,len(unique))*13*.2/60
            _,sampled['clear']=SE.rates(clear_num,clear_den,boot)
            subgroup_cells[name]=dict(n=int(keep.sum()),covered_n=int((keep&g['covered']).sum()),metrics=metrics,
                censored=dict(n=int((keep&~g['covered']).sum()),already_alarm=int((keep&~g['covered']&stopped).sum()),unalarmed=int((keep&~g['covered']&~stopped).sum())))
            if name=='all':draws[arm]=sampled
        cells[arm]=dict(threshold=thresholds[arm],calibration=calibration[arm],
            threshold_origin='Old pilot1 frozen natural threshold' if arm in ('M3','V_retest') else 'Calibrated once on95000-95047 whole-query clear first stops <=1/proxy min',
            metrics=subgroup_cells['all']['metrics'],subgroups=subgroup_cells)
        if arm in ('M3','V_retest'):
            original=baseline['cells']['M3'] if arm=='M3' else old['natural']['cells']['V']
            for category in ('contact0-2cm','contact>5cm','clear'):
                metric=cells[arm]['metrics'][category];prior=original['metrics'][category]
                field='expected_stops' if category=='clear' else 'expected_timely_stops'
                if metric[field]!=prior[field] or metric['expected_episodes']!=prior['expected_episodes']:
                    raise ValueError('Old natural M3/V_retest count/denominator parity failed')
    comparisons={}
    for arm in arms:
        m,b=cells[arm]['metrics'],cells['M3']['metrics']
        shallow=m['contact0-2cm']['timely_rate']['value']-b['contact0-2cm']['timely_rate']['value']
        deep=m['contact>5cm']['timely_rate']['value']-b['contact>5cm']['timely_rate']['value']
        clear=m['clear']['false_stops_per_min']['value']-b['clear']['false_stops_per_min']['value']
        comparisons[arm]=dict(shallow_delta=shallow,shallow_ci95=SE.interval(draws[arm]['contact0-2cm']-draws['M3']['contact0-2cm']),
            deep_delta=deep,deep_ci95=SE.interval(draws[arm]['contact>5cm']-draws['M3']['contact>5cm']),
            clear_delta_per_min=clear,clear_ci95=SE.interval(draws[arm]['clear']-draws['M3']['clear']),guardrail_pass=clear<=.1+1e-12 and deep>=-.02-1e-12)
    return dict(cells=cells,comparisons=comparisons,source_provenance=source_receipt,
        definition='Frozen all-physical-surface deadline labels; covered contacts only; one first stop/query,13x.2sec exposure including post-alarm time; proxy minutes are not real walking'),thresholds


def fresh_sequences(scores,scenes,visible,thresholds,arms,boot):
    ranges=np.asarray([s['front_range_m'] for s in scenes]);groups=E.strata(scenes);result={}
    for arm in ('M3',)+tuple(arms):
        hit=scores[arm]>=thresholds[arm];stopped=hit.any(-1);first=hit.argmax(-1)
        stop_range=ranges[np.arange(48)[:,None,None],first];timely=stopped&(stop_range>=.9)
        result[arm]={}
        for group,keep in groups.items():
            cell=dict(threshold=thresholds[arm],shallow={},outside={})
            for name,ids in (('inside1/2cm',[0,1]),('inside5cm',[2])):
                n=np.full(48,len(ids)*4);counts=timely[:,ids].sum((1,2))
                metric,_=U.pooled_rate(counts,n,keep,boot)
                metric['all_first_stops']=int(stopped[:,ids].sum((1,2))[keep].sum())
                lead=(stop_range[:,ids]-.5)/.8;chosen=keep[:,None,None]&timely[:,ids]
                metric['median_lead_conditional_timely_s']=float(np.median(lead[chosen])) if chosen.any() else None
                vf=visible[np.arange(48)[:,None,None],np.asarray(ids)[None,:,None],first[:,ids]]
                early=chosen&(stop_range[:,ids]>=2.1)
                metric['early_first_front_ge2p1']=dict(n=int(early.sum()),target_visible=int((early&vf).sum()),no_target_visible_ray=int((early&~vf).sum()))
                cell['shallow'][name]=metric
            for name,ids in (('outside15cm',[5]),('outside20cm',[6]),('outside15/20cm',[5,6])):
                cell['outside'][name]=U.pooled_rate(stopped[:,ids].sum((1,2)),np.full(48,len(ids)*4),keep,boot)[0]
            result[arm][group]=cell
    return result


def decide(main,guards,arms,t2):
    v=main['paired_minus_M3']['V_retest']
    replicated=v['value']>=.015 and v['ci95'][0] is not None and v['ci95'][0]>0
    candidates={}
    for arm in arms:
        delta=main['paired_minus_M3'][arm]
        passed=delta['value']>=.045 and delta['ci95'][0] is not None and delta['ci95'][0]>0 and guards[arm]['guardrail_pass']
        candidates[arm]=dict(candidate=passed,auc_delta=delta['value'],paired_ci95=delta['ci95'],guardrail_pass=guards[arm]['guardrail_pass'])
    effects={}
    for arm in ('VD','T2'):
        if arm not in arms:
            effects[arm]=dict(status='T2_UNFIT',established=False,value=None,ci95=[None,None]);continue
        delta=main['paired_minus_V'][arm]
        effects[arm]=dict(status='ESTABLISHED_FOR_FROZEN_SIMULATED_RECIPE' if delta['value']>=.02 and delta['ci95'][0] is not None and delta['ci95'][0]>0 else 'NOT_ESTABLISHED',
            established=delta['value']>=.02 and delta['ci95'][0] is not None and delta['ci95'][0]>0,value=delta['value'],ci95=delta['ci95'])
    return dict(replication='V_REPLICATED' if replicated else 'V_NOT_REPLICATED',
        branch='CANDIDATE' if any(c['candidate'] for c in candidates.values()) else 'NO_CANDIDATE',
        candidates=candidates,effects_minus_V=effects,t2_status=t2['status'],
        scope='Simulation Development; candidate only proposes confirmation, never promotes or replaces M3. T2 initialization/epochs/capacity differ from V.')


def report(result,out):
    main=result['auc'][PRIMARY]['all'];arms=result['available_arms'];decision=result['decision']
    fmt=lambda x:'—' if x is None else f'{x:.4f}'
    lines=[decision['replication']+' / '+decision['branch'],'','# 读出试点2','',
        '新48场景；三seed raw logits先平均，再用冻结五分数因果平滑。整场景配对bootstrap；自然V/M3阈值复用，VD/T2仅在独立自然校准批校准一次。','',
        '| 臂 | 主macroAUC | 对M3差及95%区间 | 补上比例 |','|---|---:|---:|---:|']
    for arm in ('M3',)+tuple(arms)+('D',):
        delta=main['paired_minus_M3'].get(arm);gap=main['gap_closed'].get(arm)
        d='—' if delta is None else f"{delta['value']:+.4f} [{fmt(delta['ci95'][0])},{fmt(delta['ci95'][1])}]"
        g='—' if gap is None or gap['value'] is None else f"{gap['value']:.1%} ({gap['status']})"
        lines.append(f"| {arm} | {fmt(main['arms'][arm]['value'])} | {d} | {g} |")
    if 'T2' not in arms:lines+=['','T2_UNFIT：按训练拟合验收条件停止，未计算单seed主候选，不混入三seed表。']
    lines+=['','| 单seed | 主AUC | 对M3差及95%区间 |','|---|---:|---:|']
    for arm in arms:
        for seed in range(3):
            key=f'{arm}_seed{seed}';delta=main['paired_minus_M3'][key]
            lines.append(f"| {key} | {fmt(main['arms'][key]['value'])} | {delta['value']:+.4f} [{fmt(delta['ci95'][0])},{fmt(delta['ci95'][1])}] |")
    lines+=['','| 对V差 | 差及95%区间 | 预设判断 |','|---|---:|---|']
    for arm,effect in decision['effects_minus_V'].items():
        lines.append(f"| {arm} | {fmt(effect['value'])} [{fmt(effect['ci95'][0])},{fmt(effect['ci95'][1])}] | {effect['status']} |")
    lines+=['','| 自然评价/高度 | 臂 | 浅及时 | 深及时 | 清晰首停/暴露 |','|---|---|---:|---:|---:|']
    for group in ('all','same_height','other_height'):
        for arm in ('M3',)+tuple(arms):
            m=result['natural']['cells'][arm]['subgroups'][group]['metrics'];c=m['clear'];a=m['contact0-2cm'];b=m['contact>5cm']
            lines.append(f"| {group} | {arm} | {a['expected_timely_stops']:g}/{a['expected_episodes']:g} | {b['expected_timely_stops']:g}/{b['expected_episodes']:g} | {c['expected_stops']:g}/{c['clear_minutes']:.3f}min={fmt(c['false_stops_per_min']['value'])}/min |")
    lines+=['','| 距离/分层 | 臂 | macroAUC | 对M3差 | 补上比例 |','|---|---|---:|---:|---:|']
    for domain in (PRIMARY,'1.6-2.6m','1.6-2.1m','2.1-2.6m'):
        for group in ('all','HEAD/none','HEAD/panel','BODY/none','BODY/panel','FOV_IN','FOV_OUT'):
            cell=result['auc'][domain][group]
            for arm in arms:
                gap=cell['gap_closed'][arm];g='—' if gap['value'] is None else f"{gap['value']:.1%}"+('*' if gap['status'].startswith('UNSTABLE') else '')
                lines.append(f"| {domain}/{group} | {arm} | {fmt(cell['arms'][arm]['value'])} | {fmt(cell['paired_minus_M3'][arm]['value'])} | {g} |")
    lines+=['','| 新序列/分层 | 臂 | 浅及时 | 外15首停 | 外20首停 |','|---|---|---:|---:|---:|']
    for group in ('all','FOV_IN','FOV_OUT'):
        for arm in ('M3',)+tuple(arms):
            c=result['fresh_sequences'][arm][group];s=c['shallow']['inside1/2cm'];o=c['outside']
            lines.append(f"| {group} | {arm} | {s['stops']}/{s['n']} | {o['outside15cm']['stops']}/{o['outside15cm']['n']} | {o['outside20cm']['stops']}/{o['outside20cm']['n']} |")
    lines+=['',f"FOV_IN/OUT={result['n']['fov_in']}/{result['n']['fov_out']}场景；全13帧内1/2cm共26分支帧中有至少一条first-visible目标ray的比例≥.5为IN。不能把OUT等同无物理信息。",'',
        '比例*表示D−M3<.01或区间跨零，不能强行解释。D仍掌握目标与背景特权，比例分母偏乐观；教师将这些知识带入监督，学生评价输入只用观测及查询几何。',
        '所有自然接触指标只计覆盖0.9m参考的序列；未报警右删失不算漏报。整段13帧清晰首停暴露包含报警后时间，代理分钟不等于实际连续行走误停率。',
        'T2与V的初始化、训练轮数和容量不同，不能单独归因于几何特征。区间条件于冻结模型与阈值，不包含模拟器、训练或硬件不确定性。',
        f"CPU评价耗时{result['elapsed_s']:.2f}秒；完整阶段计时由运行收据另报。全部模拟Development，不按分层或单seed择优救回，不自动替换M3。",'']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')


def evaluate(out=OUT):
    import cnh_readout_pilot2 as C
    if (out/'result.json').exists():raise FileExistsError('Completed pilot2 result is immutable')
    if out.resolve()!=C.OUT.resolve():raise ValueError('Pilot2 common lifecycle uses its fixed canonical root')
    started=time.monotonic();plan=C.load_plan();C.check_budget();units=plan['eval_units']
    if len(units)!=48 or len(set(units))!=48 or units!=sorted(units) or plan['seeds']!=[0,1,2]:raise ValueError('Require48unique sorted fresh units and all3seeds')
    if plan.get('frames',FRAMES.tolist())!=FRAMES.tolist():raise ValueError('Frozen retained frames must be3..15')
    if plan['bootstrap']['n']!=1000 or int(plan['bootstrap']['seed'])!=BOOT_SEED:raise ValueError('Require1000 bootstraps with frozen pilot2 seed2026100503')
    seed=int(plan['bootstrap']['seed']);hashes={str(out/'PLAN.json'):U.sha(out/'PLAN.json')}
    t2=t2_status(out,hashes);arms=['V_retest','VD']+(['T2'] if t2['status']=='T2_FIT' else [])
    scenes,visible=fresh_geometry(out,units,hashes);rows,path=E.rows_for(out,'fresh_evaluation');hashes[str(path)]=U.sha(path)
    scores={}
    for arm in arms:
        raw=[]
        for s in range(3):
            value,path=E.prediction(out,'fresh_evaluation',arm,s);raw.append(value);hashes[str(path)]=U.sha(path)
            scores[f'{arm}_seed{s}']=fresh_scores(value,rows,units,scenes)
        scores[arm]=fresh_scores(np.mean(raw,axis=0),rows,units,scenes)
    value,path=E.prediction(out,'fresh_evaluation','M3');hashes[str(path)]=U.sha(path);scores['M3']=fresh_scores(value,rows,units,scenes)
    refpath=out/'reference/all_scores.npz';rp=out/'reference/scores_receipt.json';receipt=U.read(rp)
    if receipt['status']!='COMPLETE' or receipt['score_sha256']!=U.sha(refpath) or receipt['plan_sha256']!=hashes[str(out/'PLAN.json')]:raise ValueError('ReferenceD receipt/PLAN mismatch')
    lp=out/'reference/lambda0_check.json';parity=U.read(lp)
    if parity['status']!='PASS' or receipt['lambda0_check_sha256']!=U.sha(lp):raise ValueError('Frozen lambda0 parity receipt mismatch')
    with np.load(refpath,allow_pickle=False) as z:
        if z['units'].tolist()!=units or not np.array_equal(z['frames'],FRAMES):raise ValueError('D axis identity mismatch')
        scores['D']=z['D'].copy()
    hashes.update({str(p):U.sha(p) for p in (refpath,rp,lp)})
    if any(v.shape!=(48,7,4,13) or not np.isfinite(v).all() for v in scores.values()):raise ValueError('All fresh arms require finite48x7x4x13 scores')
    boot=boot_weights(48,seed);auc,per_scene=auc_metrics(scores,scenes,arms,boot)
    nat,thresholds=natural(out,arms,seed,hashes)
    result=dict(status='COMPLETE',available_arms=arms,unavailable_arms={'T2':'T2_UNFIT'} if 'T2' not in arms else {},
        auc=auc,per_scene=per_scene,natural=nat,fresh_sequences=fresh_sequences(scores,scenes,visible,thresholds,arms,boot),
        decision=decide(auc[PRIMARY]['all'],nat['comparisons'],arms,t2),t2_fit_acceptance=t2,
        n=dict(fresh_scenes=48,fov_in=int(E.strata(scenes)['FOV_IN'].sum()),fov_out=int(E.strata(scenes)['FOV_OUT'].sum())),
        bootstrap=dict(n=1000,seed=seed,cluster='whole fresh scene or natural unit;common paired resampling',condition='fixed3seed mean scores and frozen natural thresholds'),
        elapsed_s=time.monotonic()-started,provenance=dict(input_sha256=hashes,evaluator_sha256=U.sha(__file__),
            reused_helpers_sha256={str(Path(m.__file__)):U.sha(m.__file__) for m in (C,E,U,OE,SE)}))
    if any(U.sha(path)!=digest for path,digest in hashes.items()):raise ValueError('Frozen pilot2 evaluation input changed')
    report(result,out);U.save(out/'result.json',result)
    print('PILOT2',result['decision']['replication'],result['decision']['branch'],result['elapsed_s'],flush=True)
    return result


def check():
    # No new score/data reads. Boundary checks include conditional-unfit arm exclusion.
    guards={a:dict(guardrail_pass=True) for a in ('V_retest','VD','T2')}
    main=dict(paired_minus_M3={a:dict(value=.045,ci95=[.001,.07]) for a in guards},
        paired_minus_V={a:dict(value=.02,ci95=[.001,.04]) for a in ('VD','T2')})
    d=decide(main,guards,list(guards),dict(status='T2_FIT'))
    assert d['replication']=='V_REPLICATED' and d['branch']=='CANDIDATE' and d['effects_minus_V']['T2']['established']
    main['paired_minus_M3']['V_retest']['value']=.014999
    main['paired_minus_M3']['VD']['value']=.044999
    d=decide(main,guards,['V_retest','VD'],dict(status='T2_UNFIT'))
    assert d['replication']=='V_NOT_REPLICATED' and d['branch']=='NO_CANDIDATE' and d['effects_minus_V']['T2']['status']=='T2_UNFIT'
    assert 'T2' not in d['candidates']
    rows=dict(unit=np.ones(13,int),config=np.zeros(13,int),variant=np.zeros(13,int),replica=np.zeros(13,int),frame=FRAMES)
    raw=np.arange(26.).reshape(13,2);smooth,_=E.ordered_episodes(raw,rows)
    np.testing.assert_allclose(smooth[0,-1],np.sum(raw[-5:]*np.array([1,2,4,8,16])[:,None],axis=0)/31)
    print('PASS pilot2 replication/candidate/effect boundaries,T2_UNFIT exclusion,causal raw-logit smoothing',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('check','evaluate'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    check() if a.stage=='check' else evaluate(a.out)
