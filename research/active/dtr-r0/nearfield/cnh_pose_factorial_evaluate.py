"""Frozen-score pose/history factorial evaluation on48 existing scenes.

Reuses only pure statistics from the unknown-target reference evaluator.
All matched-cost thresholds are descriptive evaluation-batch thresholds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import cnh_unknown_target_reference_evaluate as U

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-pose-factorial-20261004'
OLD=ROOT/'artifacts.local/work/cnh-displacement-ceiling-20261003'
REFERENCE=ROOT/'artifacts.local/work/cnh-unknown-target-reference-20261004'
BASE='A_M3_noisy'
MAIN='D_oracle8_belief'
REQUIRED=('A_M3_noisy','B_M3_true','M3_quarter','C_oracle8_true','oracle1_true','oracle2_true','oracle4_true',
          'D_oracle8_belief','oracle2_belief','oracle4_belief')
OPTIONAL=('oracle8_quarter',)
PRIMARY=U.PRIMARY
SEED=2026100402
N_BOOT=1000


def visibility_from_old_result(old):
    """Whole13frame target visibility, NOT primary-bin ray fraction.

    An inside1/2 branch-frame is visible iff first-visible object_id==0
    on at least one geometric quadrature ray. No intensity/SNR condition.
    """
    output={}
    for scene in old['per_scene']:
        numerator=denominator=0
        for low,high in U.BINS:
            domain=f'{low:g}-{high:g}m'
            for delta in ('1','2'):
                cell=scene['target_visibility'][domain][delta]
                numerator+=cell['frames_with_target_return']
                denominator+=cell['frames']
        if denominator!=26:
            raise ValueError('Visibility definition requires2branches x13frames, without primary double counting')
        vf=numerator/denominator
        output[scene['unit']]=dict(visible_branch_frames=numerator,total_branch_frames=denominator,
            vf=vf,fov_in=vf>=.5,
            definition='fraction of all13 decision branch-frames inside1/2cm with any first-visible target ray; >=.5 IN')
    return output


def strata_for(scenes):
    strata={'all':np.ones(48,bool)}
    for q,title in ((0,'HEAD'),(1,'BODY')):
        strata[title]=np.array([s['group']==q for s in scenes])
    for context in ('none','panel'):
        strata[context]=np.array([s['context']==context for s in scenes])
        for q,title in ((0,'HEAD'),(1,'BODY')):
            strata[title+'/'+context]=np.array([s['group']==q and s['context']==context for s in scenes])
    for inside,title in ((True,'FOV_IN'),(False,'FOV_OUT')):
        mask=np.array([s['visibility']['fov_in']==inside for s in scenes])
        strata[title]=mask
        for q,group in ((0,'HEAD'),(1,'BODY')):
            strata[title+'/'+group]=mask&strata[group]
        for context in ('none','panel'):
            strata[title+'/'+context]=mask&strata[context]
    return strata


def branch(auc_difference,recall_differences):
    if len(recall_differences)!=4 or any(v is None or not np.isfinite(v) for v in (auc_difference,*recall_differences)):
        return 'NOT_EVALUABLE'
    if auc_difference<=.03 and all(v<=.05 for v in recall_differences):
        return 'POSE_LIMITED'
    if auc_difference>=.06 or any(v>=.10 for v in recall_differences):
        return 'HEADROOM_SURVIVES_POSE'
    return 'INTERMEDIATE'


def cost_workpoint(scores,masks,strata,boot,outside_index):
    neg_keep=np.broadcast_to(masks[:,None,:],(48,4,13))
    base=scores[BASE][:,outside_index][neg_keep]
    if not len(base):
        return dict(status='NOT_EVALUABLE',reason='Empty outside cost domain')
    target=int((base>=U.M3_THRESHOLD).sum())
    thresholds={BASE:dict(threshold=U.M3_THRESHOLD,operator='>=',n=len(base),allowed_stops=target,
        actual_stops=target,target_rate=target/len(base),actual_rate=target/len(base),unattained_stops=0,
        rule='Original frozen A threshold; no fitting')}
    for arm in scores:
        if arm!=BASE:
            thresholds[arm]=U.matched_threshold(scores[arm][:,outside_index][neg_keep],target)
    cells={s:{} for s in strata}
    draws={s:{} for s in strata}
    for arm,score in scores.items():
        t=thresholds[arm]['threshold']
        alarm=score>=t if arm==BASE else score>t
        counts={title:((alarm[:,index]&masks[:,None,:]).sum(axis=(1,2)),masks.sum(1)*4)
            for index,title in ((0,'inside1cm'),(1,'inside2cm'),(2,'inside5cm'),(5,'outside15cm'),(6,'outside20cm'))}
        counts['outside15_20cm']=(counts['outside15cm'][0]+counts['outside20cm'][0],
                                  counts['outside15cm'][1]+counts['outside20cm'][1])
        for stratum,keep in strata.items():
            cells[stratum][arm]={}
            draws[stratum][arm]={}
            for title,(stops,n) in counts.items():
                metric,sampled=U.pooled_rate(stops,n,keep,boot)
                cells[stratum][arm][title]=metric
                draws[stratum][arm][title]=sampled
    changes={s:{} for s in strata}
    for stratum in strata:
        for arm in scores:
            if arm==BASE:
                continue
            changes[stratum][arm+'_minus_A']={}
            for title in cells[stratum][arm]:
                a,b=cells[stratum][arm][title]['rate'],cells[stratum][BASE][title]['rate']
                changes[stratum][arm+'_minus_A'][title]=dict(delta_rate=a-b if a is not None and b is not None else None,
                    paired_scene_ci95=U.interval(draws[stratum][arm][title]-draws[stratum][BASE][title]))
    return dict(status='COMPLETE',target_outside_cm=-U.DELTA[outside_index],thresholds=thresholds,cells=cells,paired_changes=changes,
        role='One pooled threshold per arm/domain/workpoint on evaluation negative scores; no subgroup refit or deployable calibration')


def sequence_cell(score,threshold,operator,ranges,visible,indices,keep,boot):
    """Apply one primary-bin matched-cost threshold to all13 decision frames."""
    selected=score[:,indices]
    alarm=selected>=threshold if operator=='>=' else selected>threshold
    stopped=alarm.any(-1)
    first=alarm.argmax(-1)
    scene_index=np.arange(48)[:,None,None]
    first_range=ranges[scene_index,first]
    first_visible=np.take_along_axis(np.broadcast_to(visible[:,indices,None,:],selected.shape),first[...,None],axis=-1)[...,0]
    covered=(ranges[:,0]>=.9)&(ranges[:,-1]<=.9)
    timely=stopped&(first_range>=.9)&covered[:,None,None]
    early=stopped&(first_range>=2.1)
    per_scene_n=np.full(48,len(indices)*4,int)
    first_metric,_=U.pooled_rate(stopped.sum(axis=(1,2)),per_scene_n,keep,boot)
    timely_metric,_=U.pooled_rate(timely.sum(axis=(1,2)),per_scene_n*covered,keep,boot)
    selected_stops=stopped[keep]
    leads=((first_range-.5)/.8)[keep][selected_stops]
    timely_leads=((first_range-.5)/.8)[keep][timely[keep]]
    n=int(per_scene_n[keep].sum())
    exposure=n*13*.2/60
    return dict(n=n,covered_n=int((per_scene_n*covered)[keep].sum()),right_censored_n=int((per_scene_n*~covered)[keep].sum()),
        first_stops=first_metric,timely_stops=timely_metric,
        no_alarm=n-first_metric['stops'],
        only_late=int((stopped&~timely&covered[:,None,None])[keep].sum()),
        proxy_minutes=exposure,first_stops_per_proxy_min=first_metric['stops']/exposure if exposure else None,
        median_lead_given_first_stop_s=float(np.median(leads)) if len(leads) else None,
        median_lead_given_timely_stop_s=float(np.median(timely_leads)) if len(timely_leads) else None,
        early_first_front_ge2p1=dict(n=int(early[keep].sum()),target_visible=int((early&first_visible)[keep].sum()),
            no_target_visible_ray=int((early&~first_visible)[keep].sum())),
        timely_first_target_visible=int((timely&first_visible)[keep].sum()),
        timely_first_no_target_visible_ray=int((timely&~first_visible)[keep].sum()),
        definition='First alarm in full13frame window; timely front>=.9, early front>=2.1; first-visible flag is geometry, not score attribution')


def sequence_summary(scores,thresholds,ranges,visible,strata,boot):
    output={s:{} for s in strata}
    categories={'inside1_2cm':[0,1],'inside1cm':[0],'inside2cm':[1],
                'outside15cm':[5],'outside20cm':[6],'outside15_20cm':[5,6]}
    for stratum,keep in strata.items():
        for arm,score in scores.items():
            spec=thresholds[arm]
            output[stratum][arm]={name:sequence_cell(score,spec['threshold'],spec['operator'],ranges,visible,indices,keep,boot)
                for name,indices in categories.items()}
    return output


def analyze(scores,scenes,visible,*,seed=SEED):
    if len(scenes)!=48 or len({s['unit'] for s in scenes})!=48 or not set(REQUIRED)<=set(scores) or set(scores)-set(REQUIRED+OPTIONAL):
        raise ValueError('Exactly48 scenes and declared score arms required')
    for arm,value in scores.items():
        if value.shape!=(48,7,4,13) or not np.isfinite(value).all():
            raise ValueError('Invalid score axes or values: '+arm)
    ranges=np.array([s['front_range_m'] for s in scenes])
    if ranges.shape!=(48,13) or visible.shape!=(48,7,13):
        raise ValueError('Distance/visibility axes differ')
    strata=strata_for(scenes)
    rng=np.random.default_rng(seed)
    boot=np.asarray([np.bincount(rng.integers(48,size=48),minlength=48) for _ in range(N_BOOT)])
    domains={f'{low:g}-{high:g}m':(ranges>=low)&(ranges<high) for low,high in U.BINS}
    domains[PRIMARY]=(ranges>=1.2)&(ranges<2.1)
    cells={name:{} for name in strata}
    differences={name:{} for name in strata}
    cost={}
    per_scene=[dict(unit=s['unit'],group=s['group'],context=s['context'],visibility=s['visibility'],auc={}) for s in scenes]
    for domain,masks in domains.items():
        aucs={}
        for arm,score in scores.items():
            values=[]
            for i,mask in enumerate(masks):
                pos,neg=score[i,[0,1]][...,mask].ravel(),score[i,[4,5,6]][...,mask].ravel()
                value=U.binary_auc(pos,neg)
                values.append(np.nan if value is None else value)
                per_scene[i]['auc'].setdefault(domain,{})[arm]=dict(auc=value,positive_n=len(pos),negative_n=len(neg))
            aucs[arm]=np.array(values)
        for stratum,keep in strata.items():
            cells[stratum][domain]={}
            differences[stratum][domain]={}
            for arm in scores:
                metric,_=U.macro_summary(aucs[arm],keep,boot)
                metric.update(positive_n=int((masks[keep].sum()*8)),negative_n=int(masks[keep].sum()*12))
                cells[stratum][domain][arm]=metric
                if arm!=BASE:
                    differences[stratum][domain][arm+'_minus_A']=U.macro_summary(aucs[arm]-aucs[BASE],keep,boot)[0]
        cost[domain]={name:cost_workpoint(scores,masks,strata,boot,index) for name,index in (('outside15cm',5),('outside20cm',6))}
    sequence={}
    for workpoint,cell in cost[PRIMARY].items():
        sequence[workpoint]=sequence_summary(scores,cell['thresholds'],ranges,visible,strata,boot) if cell['status']=='COMPLETE' else dict(status='NOT_EVALUABLE')
    main_auc=differences['all'][PRIMARY][MAIN+'_minus_A']['value']
    recalls={}
    for workpoint,cell in cost[PRIMARY].items():
        change=cell['paired_changes']['all'][MAIN+'_minus_A'] if cell['status']=='COMPLETE' else {}
        recalls[workpoint]={name:change.get(name,{}).get('delta_rate') for name in ('inside1cm','inside2cm')}
    flat=[v for row in recalls.values() for v in row.values()]
    return dict(status='COMPLETE',verdict=branch(main_auc,flat),units=[s['unit'] for s in scenes],
        cells=cells,paired_auc_differences=differences,matched_cost=cost,sequences_at_primary_matched_cost=sequence,
        per_scene=per_scene,decision=dict(primary=PRIMARY,comparison=MAIN+' minus '+BASE,auc_difference=main_auc,
            inner_recall_differences=recalls,rule='D-A signed AUC<=.03 AND four recall differences<=.05 => POSE_LIMITED; AUC>=.06 OR any recall>=.10 => HEADROOM_SURVIVES_POSE; otherwise INTERMEDIATE',
            limits='Descriptive point branch, not equivalence or information impossibility; 1/2/4-frame and quarter-pose arms do not choose branch'),
        n=dict(scenes=48,shallow_episodes=384,fov_in_scenes=int(strata['FOV_IN'].sum()),fov_out_scenes=int(strata['FOV_OUT'].sum()),
            fov_in_shallow=int(strata['FOV_IN'].sum()*8),fov_out_shallow=int(strata['FOV_OUT'].sum()*8)),
        bootstrap=dict(replicates=N_BOOT,seed=seed,cluster='48 whole scenes',pairing='Common matrix for every arm/domain/FOV group',
            limits='Conditional on frozen scores and evaluation-derived thresholds; not pose-belief, threshold, model or sensor uncertainty'),
        missing_optional_arms=[a for a in OPTIONAL if a not in scores],
        definitions=dict(visibility='Full13frame inside1/2 first-visible target presence fraction; >=.5 IN. No target ray is not physical unobservability proof.',
            costs='A frozen >= threshold frame cost, separately outside15 and20; references strict > whole ties. Evaluation-only thresholds applied to whole13frame sequences.',
            time='Lead=(first-front-.5)/.8; main lead conditional on timely alarm, all-first-alarm median also described. Fixed2.6sec/query exposure includes after-alarm time; neither human braking nor real walking burden'))


def write_report(result,out=OUT,with_quarter=False):
    decision=result['decision']
    number=U.number
    core=(BASE,'B_M3_true','C_oracle8_true',MAIN)
    lines=['# 姿态与历史长度的固定因子测量', '',
        '**主描述分支：'+result['verdict']+'。** 主域D−A macroAUC差'+number(decision['auc_difference'])+'；两个成本工作点的内1/2cm四项召回点差共同决定分支。', '',
        '复用48个直行场景、7个位移、K4、帧3–15；本批保持已消费仿真评价用途。A为原noisy-pose M3，B为true-pose M3，C为true-pose已知模板8帧参考，D为估计相对姿态plug-in的8帧候选位移混合参考。它不在姿态分布上边缘化。1/2/4帧与quarter-pose臂仅作机制描述，不用于择优修改主分支。', '',
        '|主1.2–2.1m臂|macroAUC及95%场景区间|减A及配对95%区间|', '|---|---|---|']
    for arm,cell in result['cells']['all'][PRIMARY].items():
        difference=result['paired_auc_differences']['all'][PRIMARY].get(arm+'_minus_A')
        text=number(difference['value'])+' ['+', '.join(number(v) for v in difference['ci95'])+']' if difference else '基线'
        lines.append('|'+arm+'|'+number(cell['value'])+' ['+', '.join(number(v) for v in cell['ci95'])+']|'+text+'|')
    lines+=['', '|主成本工作点|臂|外15cm帧报警率|外20cm帧报警率|内1cm召回|内2cm召回|内1/2cm减A，pp|',
        '|---|---|---|---|---|---|---|']
    for wp,cell in result['matched_cost'][PRIMARY].items():
        if cell['status']!='COMPLETE':
            lines.append('|'+wp+'|NOT_EVALUABLE|--|--|--|--|--|')
            continue
        for arm in core:
            c=cell['cells']['all'][arm]
            d=cell['paired_changes']['all'].get(arm+'_minus_A')
            difference='/'.join(number(d[t]['delta_rate'],100) for t in ('inside1cm','inside2cm')) if d else '0/0'
            lines.append('|'+wp+'|'+arm+'|'+'|'.join(number(c[t]['rate'],100)+'%' for t in ('outside15cm','outside20cm','inside1cm','inside2cm'))+'|'+difference+'|')
    lines+=['', '两个成本点分别匹配A冻结>=0.8557642486787612下外15cm与外20cm的实际帧率。其他臂strict score>threshold，整ties不拆，选最大不超过A整数预算的集合；阈值只在每距域/成本点全48场景选择一次，各子组继承。达到率可能低于预算，原整数预算、实际率和未达到数量均保留。阈值来自评价批，只描述成本前沿，不能部署或当独立校准。', '',
        '|距离域|A AUC|B AUC|C AUC|D AUC|D−A及95%区间|', '|---|---|---|---|---|---|']
    for domain,cells in result['cells']['all'].items():
        d=result['paired_auc_differences']['all'][domain][MAIN+'_minus_A']
        lines.append('|'+domain+'|'+'|'.join(number(cells[a]['value']) for a in core)+'|'+number(d['value'])+
            ' ['+', '.join(number(v) for v in d['ci95'])+']|')
    lines+=['', '|主域分层|场景数|A AUC|D AUC|D−A及95%区间|', '|---|---|---|---|---|']
    for stratum in ('HEAD','BODY','none','panel','HEAD/none','HEAD/panel','BODY/none','BODY/panel','FOV_IN','FOV_OUT'):
        c=result['cells'][stratum][PRIMARY]
        d=result['paired_auc_differences'][stratum][PRIMARY][MAIN+'_minus_A']
        lines.append('|'+stratum+'|'+str(c[BASE]['scenes'])+'|'+number(c[BASE]['value'])+'|'+number(c[MAIN]['value'])+
            '|'+number(d['value'])+' ['+', '.join(number(v) for v in d['ci95'])+']|')
    lines+=['', '|主成本点|FOV层|臂|浅及时/n|浅首次报警|外15/20cm首停/n|及时者提前量中位秒|≥2.1m浅首报：可见/无target ray|',
        '|---|---|---|---|---|---|---|---|']
    for wp,output in result['sequences_at_primary_matched_cost'].items():
        if output.get('status')=='NOT_EVALUABLE':
            continue
        for stratum in ('all','FOV_IN','FOV_OUT'):
            for arm in core:
                shallow=output[stratum][arm]['inside1_2cm']
                clear=output[stratum][arm]['outside15_20cm']
                early=shallow['early_first_front_ge2p1']
                lines.append('|'+wp+'|'+stratum+'|'+arm+'|'+str(shallow['timely_stops']['stops'])+'/'+str(shallow['covered_n'])+
                    '|'+str(shallow['first_stops']['stops'])+'|'+str(clear['first_stops']['stops'])+'/'+str(clear['n'])+
                    '|'+number(shallow['median_lead_given_timely_stop_s'])+'|'+str(early['target_visible'])+'/'+str(early['no_target_visible_ray'])+'|')
    lines+=['', '序列把主1.2–2.1m成本阈值作用于全13帧，再找第一次报警；不是先截断主域找首报。及时要求首报前距≥0.9m；早报≥2.1m。首报时“target可见”只指至少一条first-visible几何ray命中目标，不证明模型报警由目标引起。每query最多一首停；2.6秒代理暴露包含报警后时间，主提前量=(首报前距−0.5)/0.8条件于及时报警，所有首报者中位值另存result，不表示人已安全停止。', '',
        'FOV层用全13帧内1/2cm两个分支中有任何first-visible target ray的比例vf：26个分支帧，vf≥0.5为IN。原36/12场景、288/96浅序列；不是主距离六帧的vf，也不是target rays占全部rays的面积比例或实际SNR。只有这些几何样本支持分层，不能把OUT所有报警当误报或把没有target ray等同物理不可观测。', '',
        '主分支为固定有符号D−A点差：AUC≤.03且两cost点内1/2cm四差均≤.05为POSE_LIMITED；AUC≥.06或任一四差≥.10为HEADROOM_SURVIVES_POSE；其余INTERMEDIATE。POSE_LIMITED不是等价检验或所有算法无余量证明，也不能否定真正对姿态分布边缘化的推断；plug-in失配似然可过度自信。参考仍有真当前anchor、目标尺寸/rho与背景等特权。', '',
        '共同1000整场景bootstrap，seed2026100402，位移/K/帧不当独立样本；阈值在抽样中固定，区间不含阈值、belief/先验、模型或真实传感器不确定性。本结果不能变成实机效果、安全结果或新训练授权。更细分层、各臂内5cm召回、所有距离成本、窗口臂及序列首报可见分类见result.json。', '',
        '数字复现：A输入与上一轮冻结M3逐值一致，C与冻结oracle8逐值一致；旧A序列原阈值及时277/288(IN)、32/96(OUT)及309/384(all)。旧U8两个主成本点各为288/288(IN)、70/96(OUT)，单独保留为原参考复核，不与本轮D混淆。', '',
        '复现：`cnh_pose_factorial_evaluate.py --stage evaluate`；评价只读取封存score与旧几何，不渲染、不做模型推理、似然评分或真实回放。旧结果与协议不覆盖。']
    if with_quarter:
        lines+=['', '## 条件quarter参考', '',
            '只因原D分支为POSE_LIMITED或INTERMEDIATE才运行。原主分数/result完全保留；本补充仅增加oracle8_quarter，主D−A点差、主分支和两原cost阈值核对相同，不从quarter重新选择或改判。', '',
            '|距离域|quarter oracle8 AUC|quarter−A及95%区间|', '|---|---|---|']
        for domain,cells in result['cells']['all'].items():
            c=cells['oracle8_quarter']
            d=result['paired_auc_differences']['all'][domain]['oracle8_quarter_minus_A']
            lines.append('|'+domain+'|'+number(c['value'])+'|'+number(d['value'])+' ['+', '.join(number(v) for v in d['ci95'])+']|')
        lines+=['', '|主成本工作点|FOV层|quarter浅及时/n|quarter外15/20首停/n|内1/2cm减A，pp（all）|',
            '|---|---|---|---|---|']
        for wp,output in result['sequences_at_primary_matched_cost'].items():
            if output.get('status')=='NOT_EVALUABLE':
                continue
            change=result['matched_cost'][PRIMARY][wp]['paired_changes']['all']['oracle8_quarter_minus_A']
            difference='/'.join(number(change[t]['delta_rate'],100) for t in ('inside1cm','inside2cm'))
            for stratum in ('all','FOV_IN','FOV_OUT'):
                s=output[stratum]['oracle8_quarter']['inside1_2cm']
                c=output[stratum]['oracle8_quarter']['outside15_20cm']
                lines.append('|'+wp+'|'+stratum+'|'+str(s['timely_stops']['stops'])+'/'+str(s['covered_n'])+
                    '|'+str(c['first_stops']['stops'])+'/'+str(c['n'])+'|'+(difference if stratum=='all' else '见result分层')+'|')
    (out/('REPORT_WITH_QUARTER.md' if with_quarter else 'REPORT.md')).write_text('\n'.join(lines)+'\n',encoding='utf8')


def evaluate(out=OUT,with_quarter=False):
    result_path=out/('result_with_quarter.json' if with_quarter else 'result.json')
    if result_path.exists():
        previous=U.read(result_path)
        if previous.get('status')!='COMPLETE':
            raise ValueError('Inspect existing incomplete result')
        for path,digest in previous['provenance']['input_sha256'].items():
            if U.sha(path)!=digest:
                raise ValueError('Previous evaluation input changed: '+path)
        print('Existing COMPLETE result inputs verified; no repeated evaluation')
        return previous
    plan_path,scene_path=out/'PLAN.json',out/'scenes.json'
    score_path=out/('scores/all_scores_with_quarter.npz' if with_quarter else 'scores/all_scores.npz')
    receipt_path=out/('scores_lambda0.25_receipt.json' if with_quarter else 'scores_receipt.json')
    plan,receipt=U.read(plan_path),U.read(receipt_path)
    if plan.get('run')!='CNH_POSE_FACTORIAL_20261004' or receipt.get('status')!='COMPLETE' or receipt.get('plan_sha256')!=U.sha(plan_path):
        raise ValueError('Frozen PLAN and complete sealed scores required before scientific evaluation')
    config=plan.get('bootstrap',{})
    if config.get('seed')!=SEED or config.get('replicates',config.get('n'))!=N_BOOT:
        raise ValueError('Frozen bootstrap differs')
    for name,path in ((str(score_path.relative_to(out)).replace('\\','/'),score_path),('scenes.json',scene_path)):
        if receipt.get('output_sha256',{}).get(name,receipt.get('output_sha256',{}).get(str(path)))!=U.sha(path):
            raise ValueError('Score receipt output hash differs: '+name)
    hashes={str(path):U.sha(path) for path in (plan_path,score_path,scene_path,receipt_path,Path(__file__),Path(U.__file__),OLD/'result.json',REFERENCE/'result.json',REFERENCE/'scores/all_scores.npz')}
    with np.load(score_path,allow_pickle=False) as cache:
        if 'units' not in cache.files:
            raise ValueError('Unit index required')
        units=cache['units'].tolist()
        scores={name:cache[name] for name in cache.files if name!='units'}
    primary=None
    if with_quarter:
        primary_path=out/'result.json'
        primary=U.read(primary_path)
        if primary.get('status')!='COMPLETE' or primary.get('verdict') not in ('POSE_LIMITED','INTERMEDIATE'):
            raise ValueError('Quarter oracle permitted only after completed POSE_LIMITED/INTERMEDIATE primary')
        hashes[str(primary_path)]=U.sha(primary_path)
        base_path=out/'scores/all_scores.npz'
        hashes[str(base_path)]=U.sha(base_path)
        with np.load(base_path,allow_pickle=False) as base:
            if set(scores)!=set(REQUIRED+OPTIONAL) or base['units'].tolist()!=units:
                raise ValueError('Quarter extension must add exactlyoracle8_quarter to original score arms')
            for arm in REQUIRED:
                np.testing.assert_array_equal(scores[arm],base[arm])
        for path,digest in primary['provenance']['input_sha256'].items():
            if U.sha(path)!=digest:
                raise ValueError('Primary evidence changed before quarter extension: '+path)
    elif set(scores)!=set(REQUIRED):
        raise ValueError('Primary evaluation must contain onlythe10frozen arms')
    scenes=U.read(scene_path)
    old=U.read(OLD/'result.json')
    if units!=old['units'] or [s['unit'] for s in scenes]!=units:
        raise ValueError('Original48 scene identities and ordering must be retained')
    legacy_visibility=visibility_from_old_result(old)
    visible=[]
    for scene,previous in zip(scenes,old['per_scene']):
        for field in ('unit','group','context'):
            if scene[field]!=previous[field]:
                raise ValueError('Scene identity changed: '+field)
        np.testing.assert_allclose(scene['front_range_m'],previous['front_range_m'],atol=1e-12,rtol=0)
        path=OLD/'templates'/f"unit{scene['unit']}.npz"
        hashes[str(path)]=U.sha(path)
        with np.load(path,allow_pickle=False) as cache:
            target=np.any(cache['object_id'][:,np.arange(3,16)]==0,axis=(-3,-2,-1))
        vf=float(target[:2].mean())
        if abs(vf-legacy_visibility[scene['unit']]['vf'])>1e-12:
            raise ValueError('Raw first-visible flags differ from original visibility receipt')
        if 'target_visible_frame' in scene:
            np.testing.assert_array_equal(target,np.asarray(scene['target_visible_frame'],bool))
        scene['visibility']=legacy_visibility[scene['unit']]
        visible.append(target)
    with np.load(REFERENCE/'scores/all_scores.npz',allow_pickle=False) as cache:
        np.testing.assert_array_equal(scores[BASE],cache['M3_smooth'])
        np.testing.assert_array_equal(scores['C_oracle8_true'],cache['oracle8'])
        legacy_U8=cache['U8']
    visible=np.stack(visible)
    result=analyze(scores,scenes,visible)
    if with_quarter:
        if result['decision']!=primary['decision'] or result['verdict']!=primary['verdict']:
            raise ValueError('Quarter changed primary D branch or its point differences')
        for domain in result['matched_cost']:
            for wp in result['matched_cost'][domain]:
                for arm in REQUIRED:
                    if result['matched_cost'][domain][wp]['thresholds'][arm]!=primary['matched_cost'][domain][wp]['thresholds'][arm]:
                        raise ValueError('Quarter changed an original matched-cost threshold')
        result['conditional_extension']=dict(primary_result_sha256=hashes[str(out/'result.json')],
            added_arm='oracle8_quarter',primary_scores_exact=True,primary_decision_exact=True,primary_thresholds_exact=True,
            role='Conditional quarter description only; original primary score/result files preserved')
    for name,digest in receipt.get('source_sha256',{}).items():
        if U.sha(name)!=digest:
            raise ValueError('Scoring source changed since score sealing')
        hashes[name]=digest
    # Frozen A thresholds stay identical at both descriptive workpoints.
    counts={}
    for stratum in ('all','FOV_IN','FOV_OUT'):
        cell=result['sequences_at_primary_matched_cost']['outside15cm'][stratum][BASE]['inside1_2cm']
        counts[stratum]=dict(timely=cell['timely_stops']['stops'],n=cell['covered_n'])
    if counts!={'all':{'timely':309,'n':384},'FOV_IN':{'timely':277,'n':288},'FOV_OUT':{'timely':32,'n':96}}:
        raise ValueError('Original M3 sequence denominator/timely counts did not reproduce')
    reference=U.read(REFERENCE/'result.json')
    ranges=np.array([s['front_range_m'] for s in scenes])
    legacy_U8_counts={}
    for workpoint,cell in reference['matched_cost'][PRIMARY].items():
        threshold=cell['thresholds']['U8']['threshold']
        alarms=legacy_U8[:,:2]>threshold
        timely=alarms.any(-1)&(ranges[np.arange(48)[:,None,None],alarms.argmax(-1)]>=.9)
        inside=np.array([s['visibility']['fov_in'] for s in scenes])
        legacy_U8_counts[workpoint]=dict(IN=int(timely[inside].sum()),IN_n=int(inside.sum()*8),OUT=int(timely[~inside].sum()),OUT_n=int((~inside).sum()*8))
        if legacy_U8_counts[workpoint]!=dict(IN=288,IN_n=288,OUT=70,OUT_n=96):
            raise ValueError('Frozen U8 sequence parity failed')
    result['legacy_parity']=dict(status='PASS',A_and_C_scores_exact=True,A_sequences=counts,U8_sequences=legacy_U8_counts,
        visibility='Raw target first-visible flags equal old receipt all13frame inside1/2 fractions;36IN/12OUT')
    result['provenance']=dict(input_sha256=hashes,plan=plan,score_receipt=receipt,
        operations=dict(training=False,inference=False,rendering=False,likelihood_scoring=False,real_replay=False,
            evaluation_thresholds_descriptive_only=True),original_evidence_preserved=True)
    for path,digest in hashes.items():
        if U.sha(path)!=digest:
            raise ValueError('Evaluation input changed during read-only computation: '+path)
    U.save(result_path,result)
    write_report(result,out,with_quarter=with_quarter)
    print(result['verdict'],json.dumps(result['decision']))
    return result


def check():
    assert branch(.03,[.05]*4)=='POSE_LIMITED'
    assert branch(.06,[0]*4)=='HEADROOM_SURVIVES_POSE'
    assert branch(.04,[.10,0,0,0])=='HEADROOM_SURVIVES_POSE'
    assert branch(.04,[.06,0,0,0])=='INTERMEDIATE'
    scenes=[dict(unit=i,group=i%2,context='none' if i%4<2 else 'panel',front_range_m=np.linspace(2.57,.65,13).tolist(),
        visibility=dict(vf=1. if i<36 else .2,fov_in=i<36)) for i in range(48)]
    raw=np.broadcast_to(np.array([2.,1.5,3.,0.,-.5,-1.,-2.])[None,:,None,None],(48,7,4,13)).copy()
    visible=np.ones((48,7,13),bool)
    visible[36:]=False
    result=analyze({arm:raw.copy() for arm in REQUIRED},scenes,visible)
    assert result['verdict']=='POSE_LIMITED' and result['n']['fov_in_shallow']==288
    seq=result['sequences_at_primary_matched_cost']['outside15cm']['FOV_OUT'][MAIN]['inside1_2cm']
    assert seq['timely_stops']['stops']==96 and seq['early_first_front_ge2p1']['no_target_visible_ray']==96
    assert seq['early_first_front_ge2p1']['target_visible']==0
    json.dumps(result,allow_nan=False)
    print('PASS point-branch boundaries, common-scene metrics, full-window first alarm, FOV visibility and early no-target-ray accounting')


def check_quarter():
    # New conditional path only: an extremequarter arm may not change D's branch.
    scenes=[dict(unit=i,group=i%2,context='none' if i%4<2 else 'panel',front_range_m=np.linspace(2.57,.65,13).tolist(),
        visibility=dict(vf=1. if i<36 else .2,fov_in=i<36)) for i in range(48)]
    raw=np.broadcast_to(np.array([2.,1.5,3.,0.,-.5,-1.,-2.])[None,:,None,None],(48,7,4,13)).copy()
    scores={arm:raw.copy() for arm in REQUIRED}
    visible=np.ones((48,7,13),bool)
    primary=analyze(scores,scenes,visible)
    scores['oracle8_quarter']=raw.copy()
    scores['oracle8_quarter'][:,:3]+=100
    extension=analyze(scores,scenes,visible)
    assert extension['decision']==primary['decision'] and extension['verdict']==primary['verdict']
    assert extension['missing_optional_arms']==[] and primary['missing_optional_arms']==['oracle8_quarter']
    for domain in primary['matched_cost']:
        for wp in primary['matched_cost'][domain]:
            for arm in REQUIRED:
                assert extension['matched_cost'][domain][wp]['thresholds'][arm]==primary['matched_cost'][domain][wp]['thresholds'][arm]
    cell=extension['sequences_at_primary_matched_cost']['outside15cm']['all']['oracle8_quarter']['inside1_2cm']
    assert cell['median_lead_given_timely_stop_s'] is not None
    print('PASS conditionalquarter adds one descriptive arm, preserves Dbranch/old thresholds, and retains timely-conditioned lead')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=('check','check-quarter','evaluate','finalize'))
    parser.add_argument('--with-quarter',action='store_true')
    args=parser.parse_args()
    {'check':check,'check-quarter':check_quarter,'evaluate':lambda:evaluate(with_quarter=args.with_quarter),
     'finalize':lambda:evaluate(with_quarter=args.with_quarter)}[args.stage]()
