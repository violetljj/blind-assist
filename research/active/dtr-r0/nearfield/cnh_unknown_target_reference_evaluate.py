"""Unknown-target fixed-prior reference: evaluation metrics only.

Scene is the independent unit. Cost matching is a descriptive threshold
selected on this evaluation batch, never a deployable calibration threshold.
No model, renderer or likelihood computation is called by this module.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT/'artifacts.local/work/cnh-displacement-ceiling-20261003'
OUT = ROOT/'artifacts.local/work/cnh-unknown-target-reference-20261004'
PRIMARY = 'primary1.2-2.1m'
DELTA = [1, 2, 5, -5, -10, -15, -20]
BINS = [(.6, .9), (.9, 1.2), (1.2, 1.6), (1.6, 2.1), (2.1, 2.6)]
ARMS = ('M3_smooth', 'oracle8', 'oracle12', 'U8', 'U12')
M3_THRESHOLD = .8557642486787612


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def binary_auc(positive, negative):
    pos, neg = np.asarray(positive, float).ravel(), np.asarray(negative, float).ravel()
    if not len(pos) or not len(neg):
        return None
    if not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError('Nonfinite AUC scores cannot be silently removed')
    rank = rankdata(np.concatenate((pos, neg)), method='average')
    return float((rank[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))


def matched_threshold(negative_scores, allowed_stops):
    """Largest attainable strict->score alarm set not exceeding integer cost.

    Every tied score group is entirely retained or excluded. Score==threshold
    never alarms. No random tie split or positive-score access occurs here.
    """
    value = np.asarray(negative_scores, float).ravel()
    if not len(value) or not np.isfinite(value).all():
        raise ValueError('Nonempty finite negative score domain required')
    if not isinstance(allowed_stops, (int, np.integer)) or not 0 <= allowed_stops <= len(value):
        raise ValueError('An integer admissible false-alarm count is required')
    unique, sizes = np.unique(value, return_counts=True)
    unique, sizes = unique[::-1], sizes[::-1]
    cumulative = np.cumsum(sizes)
    groups = int(np.sum(cumulative <= allowed_stops))
    threshold = float(unique[groups]) if groups < len(unique) else float(np.nextafter(unique[-1], -np.inf))
    if not np.isfinite(threshold):
        raise FloatingPointError('Strict threshold below minimum is not finite')
    actual = int((value > threshold).sum())
    assert actual <= allowed_stops
    assert groups == len(unique) or actual + int(sizes[groups]) > allowed_stops
    return dict(threshold=threshold, operator='>', n=len(value), allowed_stops=int(allowed_stops), actual_stops=actual,
        target_rate=float(allowed_stops/len(value)), actual_rate=float(actual/len(value)),
        unattained_stops=int(allowed_stops-actual),
        threshold_ties=int((value == threshold).sum()), all_score_groups=int(len(unique)),
        rule='Take largest whole tied-score groups with strictly greater scores under the M3 integer cost budget; never split ties')


def interval(value):
    finite = np.asarray(value)[np.isfinite(value)]
    return np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else [None, None]


def macro_summary(values, keep, boot):
    values = np.asarray(values, float)
    eligible = np.asarray(keep, bool) & np.isfinite(values)
    den = boot@eligible.astype(float)
    num = boot@np.where(eligible, np.nan_to_num(values), 0.)
    draws = np.divide(num, den, out=np.full(len(boot), np.nan), where=den>0)
    selected = values[eligible]
    result = dict(value=float(selected.mean()) if len(selected) else None, ci95=interval(draws),
        scenes=int(eligible.sum()), empty_scenes=int(np.sum(keep)-eligible.sum()),
        per_scene_min=float(selected.min()) if len(selected) else None,
        per_scene_median=float(np.median(selected)) if len(selected) else None,
        per_scene_max=float(selected.max()) if len(selected) else None)
    return result, draws


def pooled_rate(stops, n, keep, boot):
    stops, n = np.asarray(stops, int), np.asarray(n, int)
    keep = np.asarray(keep, bool)
    total, numerator = int(n[keep].sum()), int(stops[keep].sum())
    denominator = boot@(n*keep)
    draws = np.divide(boot@(stops*keep), denominator, out=np.full(len(boot), np.nan), where=denominator>0)
    scene_rates = np.divide(stops, n, out=np.full(len(n), np.nan), where=n>0)
    finite = scene_rates[keep & np.isfinite(scene_rates)]
    return dict(stops=numerator, n=total, rate=numerator/total if total else None,
        ci95=interval(draws), scenes=int((keep & (n>0)).sum()),
        scene_macro_rate=float(finite.mean()) if len(finite) else None), draws


def branch_decision(auc_difference, recall_differences):
    if len(recall_differences) != 4 or any(v is None or not np.isfinite(v) for v in (auc_difference, *recall_differences)):
        return 'NOT_EVALUABLE'
    if auc_difference <= .03 and all(v <= .05 for v in recall_differences):
        return 'NEAR'
    if auc_difference >= .06 or any(v >= .10 for v in recall_differences):
        return 'HEADROOM'
    return 'INTERMEDIATE'


def cost_workpoint(scores, masks, strata, boot, outside_index):
    """One domain, one outside branch, pooled cost over all48 scenes."""
    neg_mask=np.broadcast_to(masks[:,None,:],(48,4,13))
    m3_negative=scores['M3_smooth'][:,outside_index][neg_mask]
    if not len(m3_negative):
        return dict(status='NOT_EVALUABLE',reason='No outside branch frames')
    target_count=int((m3_negative>=M3_THRESHOLD).sum())
    thresholds={'M3_smooth':dict(threshold=M3_THRESHOLD,operator='>=',n=len(m3_negative),
        allowed_stops=target_count,actual_stops=target_count,target_rate=target_count/len(m3_negative),
        actual_rate=target_count/len(m3_negative),unattained_stops=0,rule='Original frozen M3 threshold; not fitted here')}
    for arm in ARMS[1:]:
        thresholds[arm]=matched_threshold(scores[arm][:,outside_index][neg_mask],target_count)
    rate_cells={name:{} for name in strata}
    rate_changes={name:{} for name in strata}
    sampled_rates={}
    for arm in ARMS:
        threshold=thresholds[arm]['threshold']
        alarm=scores[arm]>=threshold if arm=='M3_smooth' else scores[arm]>threshold
        sampled_rates[arm]={}
        branch_counts={}
        for index,title in ((0,'inside1cm'),(1,'inside2cm'),(2,'inside5cm'),(5,'outside15cm'),(6,'outside20cm')):
            active=alarm[:,index]&masks[:,None,:]
            branch_counts[title]=(active.sum(axis=(1,2)),masks.sum(1)*4)
        branch_counts['outside15_20cm']=(branch_counts['outside15cm'][0]+branch_counts['outside20cm'][0],
            branch_counts['outside15cm'][1]+branch_counts['outside20cm'][1])
        for name,keep in strata.items():
            rate_cells[name][arm]={}
            sampled_rates[arm][name]={}
            for title,(stops,n) in branch_counts.items():
                metric,draws=pooled_rate(stops,n,keep,boot)
                rate_cells[name][arm][title]=metric
                sampled_rates[arm][name][title]=draws
    for name in strata:
        for arm in ARMS[1:]:
            rate_changes[name][arm+'_minus_M3']={}
            for title in rate_cells[name][arm]:
                a,b=rate_cells[name][arm][title]['rate'],rate_cells[name]['M3_smooth'][title]['rate']
                rate_changes[name][arm+'_minus_M3'][title]=dict(delta_rate=a-b if a is not None and b is not None else None,
                    paired_scene_ci95=interval(sampled_rates[arm][name][title]-sampled_rates['M3_smooth'][name][title]))
    return dict(status='COMPLETE',thresholds=thresholds,cells=rate_cells,paired_changes=rate_changes,
        target_outside_cm=-DELTA[outside_index],
        threshold_fit_role='Evaluation-batch descriptive cost frontier; no deployable calibration or fresh validation',
        target_cost='Pooled frozen M3 frame alarm rate on this single outside branch in this domain; one threshold per arm/domain/workpoint across all48scenes',
        bootstrap_role='Conditional on selected descriptive thresholds; no threshold refitting in draws')


def analyze(scores, scenes, *, seed, n_boot=1000):
    """Five already sealed score arms [scene,delta,K,decision_frame]."""
    if len(scenes) != 48 or len({s['unit'] for s in scenes}) != 48 or set(scores) != set(ARMS):
        raise ValueError('Exactly48 distinct scenes and all five fixed arms required')
    for arm, value in scores.items():
        if np.asarray(value).shape != (48, 7, 4, 13) or not np.isfinite(value).all():
            raise ValueError('Finite sealed score axes differ: '+arm)
    ranges = np.stack([s['front_range_m'] for s in scenes])
    if ranges.shape != (48, 13) or not np.isfinite(ranges).all():
        raise ValueError('13 decision-frame distances per scene required')
    rng = np.random.default_rng(seed)
    boot = np.asarray([np.bincount(rng.integers(48, size=48), minlength=48) for _ in range(n_boot)])
    strata = {'all': np.ones(48, bool)}
    for group, name in ((0, 'HEAD'), (1, 'BODY')):
        strata[name] = np.array([s['group'] == group for s in scenes])
    for context in ('none', 'panel'):
        strata[context] = np.array([s['context'] == context for s in scenes])
        for group, title in ((0, 'HEAD'), (1, 'BODY')):
            strata[title+'/'+context] = np.array([s['group']==group and s['context']==context for s in scenes])
    domains = {f'{low:g}-{high:g}m': (ranges>=low)&(ranges<high) for low, high in BINS}
    domains[PRIMARY] = (ranges>=1.2)&(ranges<2.1)
    all_cells = {name:{} for name in strata}
    paired = {name:{} for name in strata}
    cost = {}
    per_scene = [dict(unit=s['unit'], group=s['group'], context=s['context'], auc={}) for s in scenes]
    for domain, masks in domains.items():
        auc_values = {}
        denominators = []
        for arm in ARMS:
            values = []
            for i, mask in enumerate(masks):
                positive = scores[arm][i, [0, 1]][..., mask].ravel()
                negative = scores[arm][i, [4, 5, 6]][..., mask].ravel()
                value = binary_auc(positive, negative)
                values.append(np.nan if value is None else value)
                per_scene[i]['auc'].setdefault(domain, {})[arm] = dict(auc=value, positive_n=len(positive), negative_n=len(negative))
                if arm == 'M3_smooth':
                    denominators.append((len(positive), len(negative)))
            auc_values[arm] = np.array(values)
        for name, keep in strata.items():
            all_cells[name][domain] = {}
            paired[name][domain] = {}
            for arm in ARMS:
                metric, _ = macro_summary(auc_values[arm], keep, boot)
                metric.update(positive_n=sum(denominators[i][0] for i in np.flatnonzero(keep)),
                              negative_n=sum(denominators[i][1] for i in np.flatnonzero(keep)))
                all_cells[name][domain][arm] = metric
                if arm != 'M3_smooth':
                    metric, _ = macro_summary(auc_values[arm]-auc_values['M3_smooth'], keep, boot)
                    paired[name][domain][arm+'_minus_M3'] = metric
        # Separate outside15/20 workpoints, each set on ALL scenes, not subgroups.
        cost[domain]={title:cost_workpoint(scores,masks,strata,boot,index)
                      for title,index in (('outside15cm',5),('outside20cm',6))}
    auc = paired['all'][PRIMARY]['U8_minus_M3']['value']
    recall_differences={}
    for workpoint,cell in cost[PRIMARY].items():
        change=cell['paired_changes']['all']['U8_minus_M3'] if cell['status']=='COMPLETE' else {}
        recall_differences[workpoint]={title:change.get(title,{}).get('delta_rate') for title in ('inside1cm','inside2cm')}
    recall_values=[v for row in recall_differences.values() for v in row.values()]
    return dict(status='COMPLETE', verdict=branch_decision(auc,recall_values), units=[s['unit'] for s in scenes],
        cells=all_cells, paired_auc_differences=paired, matched_cost=cost, per_scene=per_scene,
        decision=dict(primary=PRIMARY, arm='U8', auc_difference=auc, inner_recall_differences=recall_differences,
            rule='NEAR if AUCdiff<=.03 AND all four inner1/2cm differences at BOTH cost workpoints<=.05; HEADROOM if AUCdiff>=.06 OR any of those differences>=.10; else INTERMEDIATE',
            role='Frozen descriptive signed point branch; NEAR includes inferior U, not equivalence. U12 and all subgroups only describe, cannot select a better branch'),
        bootstrap=dict(replicates=n_boot,seed=seed,cluster='48 whole scenes',pairing='Common scene weight matrix for every arm/domain/stratum',
            conditioning='Frozen scores and evaluation-derived cost thresholds; not model, prior, threshold or domain uncertainty'),
        n=dict(scenes=48,branches=7,replicas=4,decision_frames=13,frame_score_samples=48*7*4*13),
        definitions=dict(AUC='Within-scene inside1/2 versus outside10/15/20 then scene-equal macro mean; half credit ties',
            cost='Frame alarms; outside10 is acceptable pass and excluded from false-alarm matching',
            recall='Pooled frame alarm rate on nominal inside1/2/5cm; not event timeliness, human stopping or real device performance'))


def number(value, factor=1.):
    return '--' if value is None else f'{value*factor:.4f}'


def write_report(result, out=OUT):
    decision=result['decision']
    lines=['# 未知目标尺寸/反射率的固定先验参考', '',
        '**主点分支：'+result['verdict']+'。** U8−M3主macroAUC差'+number(decision['auc_difference'])+'；两独立cost工作点的内1/2cm点差共同决定分支。原阈值不改，U12和分层不用于择优改判。', '',
        '复用旧48场景、7个位移、4个噪声实现、帧3–15；本批永久保持已消费仿真评价用途。没有训练、新噪声、M3推理或真实回放分析。oracle逐帧分数由已获用户授权的旧冻结定义重建后封存；本评价只读取分数，不执行渲染或似然评分。', '',
        '主AUC：每场景在1.2–2.1m内先比较伸入1/2cm与外10/15/20cm，再48场景等权macro平均；同分计半。外10cm属于可接受擦身，不计入匹配误警成本。', '',
        '|臂|主macroAUC及场景95%区间|配对减M3及95%区间|',
        '|---|---|---|']
    for arm in ARMS:
        c=result['cells']['all'][PRIMARY][arm]
        d=result['paired_auc_differences']['all'][PRIMARY].get(arm+'_minus_M3')
        lines.append('|'+arm+'|'+number(c['value'])+' ['+', '.join(number(v) for v in c['ci95'])+']|'+
            (number(d['value'])+' ['+', '.join(number(v) for v in d['ci95'])+']' if d else '基线')+'|')
    lines+=['','|主成本工作点|臂|该成本外侧实际率|另侧实际率|内1cm召回|内2cm召回|内5cm召回|内1/2cm减M3，pp|',
        '|---|---|---|---|---|---|---|---|']
    for workpoint,cell in result['matched_cost'][PRIMARY].items():
        if cell['status']!='COMPLETE':
            lines.append('|'+workpoint+'|NOT_EVALUABLE|--|--|--|--|--|--|')
            continue
        other='outside20cm' if workpoint=='outside15cm' else 'outside15cm'
        for arm in ARMS:
            c=cell['cells']['all'][arm]
            d=cell['paired_changes']['all'].get(arm+'_minus_M3')
            difference='/'.join(number(d[t]['delta_rate'],100) for t in ('inside1cm','inside2cm')) if d else '0/0'
            lines.append('|'+workpoint+'|'+arm+'|'+number(c[workpoint]['rate'],100)+'%|'+number(c[other]['rate'],100)+'%|'+
                '|'.join(number(c[t]['rate'],100)+'%' for t in ('inside1cm','inside2cm','inside5cm'))+'|'+difference+'|')
    lines+=['', '成本按外15cm和外20cm分别建立两个工作点：每距离域分别取M3冻结阈值>=0.855764的实际帧率作目标；各参考臂仅用该外侧分支全48场景分数选择唯一strict score>threshold阈值。整tie组不拆，选最大不超M3整数误警预算的集合。阈值、整数预算、实际达到率与未达到数量全部保留在result；实际成本可能稍低，不能写成严格等成本。', '',
        '这些阈值来自本评价批，只有描述成本前沿的用途，不能部署或当作独立校准。HEAD/BODY、none/panel与交叉子组沿用全体阈值，不对子组重调。召回是名义内1/2/5cm逐帧报警率，不是序列及时停步、用户负担或实机安全效果。', '',
        '|距离域|M3 AUC|U8 AUC|U8−M3及95%区间|', '|---|---|---|---|']
    for domain in result['cells']['all']:
        c=result['cells']['all'][domain]
        d=result['paired_auc_differences']['all'][domain]['U8_minus_M3']
        lines.append('|'+domain+'|'+number(c['M3_smooth']['value'])+'|'+number(c['U8']['value'])+'|'+number(d['value'])+
            ' ['+', '.join(number(v) for v in d['ci95'])+']|')
    lines+=['','|主域分层|场景数|M3 AUC|U8 AUC|U8−M3及95%区间|', '|---|---|---|---|---|']
    for stratum in result['cells']:
        if stratum=='all':
            continue
        c=result['cells'][stratum][PRIMARY]
        d=result['paired_auc_differences'][stratum][PRIMARY]['U8_minus_M3']
        lines.append('|'+stratum+'|'+str(c['U8']['scenes'])+'|'+number(c['M3_smooth']['value'])+'|'+number(c['U8']['value'])+
            '|'+number(d['value'])+' ['+', '.join(number(v) for v in d['ci95'])+']|')
    lines+=['', '分支采用跑前固定有符号U8−M3差：AUC差≤.03且两个成本点的内1/2cm四项召回差均≤.05为NEAR；AUC差≥.06或任一四项差≥.10为HEADROOM；其余INTERMEDIATE。NEAR也包含U更差的情况，不能称统计等价或证明无剩余信息。区间只描述，不替代该点分支，不选择U12或有利子组。', '',
        '未知尺寸/反射率使用扩展离散先验：12个regular（宽.09/.11×深.09/.15×三rho）总mass.9，HEAD/BODY高固定.36/.34；三个tiny（.02×.04×.04×三rho）总mass.1。各rho为.2916666667/.435/.5783333333，regular每项.075、tiny每项1/30。它不是原生成器严格连续先验；有限网格失配会影响U能力，NEAR不能否定其他未知目标推断或证明零余量。背景、真实姿态等仍有特权；known-scene oracle更乐观；所有结果只在当前代理观测模型下成立。', '',
        '1000次共同整场景bootstrap，seed2026100401；位移、K、帧不作为独立样本。召回区间条件于这批选出的阈值，不含阈值重估、先验、真实噪声或跨域不确定性。完整逐距离/分层/各臂配对差、两cost点及原AUC逐场景复现检查见result.json。', '',
        '复现：`cnh_unknown_target_reference_evaluate.py --stage evaluate`。已完成结果只核对输入哈希，不覆盖旧结果。评价输入和源码SHA由result绑定。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def evaluate(out=OUT):
    result_path=out/'result.json'
    if result_path.exists():
        prior=read(result_path)
        if prior.get('status')!='COMPLETE':
            raise ValueError('Inspect existing incomplete result before retry')
        for path,digest in prior['provenance']['input_sha256'].items():
            if sha(path)!=digest:
                raise ValueError('Existing result input changed: '+path)
        print('Existing COMPLETE evaluation inputs verified; no repeated calculation')
        return prior
    plan_path,score_path,scene_path=out/'PLAN.json',out/'scores/all_scores.npz',out/'scenes.json'
    receipt_path=out/'scores_receipt.json'
    plan,receipt=read(plan_path),read(receipt_path)
    if receipt.get('status')!='COMPLETE' or receipt.get('plan_sha256')!=sha(plan_path):
        raise ValueError('All score arms must be COMPLETE and sealed against PLAN before evaluation')
    expected_outputs={'scores/all_scores.npz':score_path,'scenes.json':scene_path}
    for name,path in expected_outputs.items():
        if receipt.get('output_sha256',{}).get(name)!=sha(path):
            raise ValueError('Score receipt does not bind evaluation input: '+name)
    hashes={str(p):sha(p) for p in (plan_path,score_path,scene_path,receipt_path,Path(__file__),OLD/'result.json')}
    with np.load(score_path,allow_pickle=False) as cache:
        if set(cache.files)!=set(ARMS)|{'units'}:
            raise ValueError('Score bundle needs exact five arms and units')
        units=cache['units'].tolist()
        scores={a:cache[a] for a in ARMS}
    scenes=read(scene_path)
    if not isinstance(scenes,list) or [s['unit'] for s in scenes]!=units:
        raise ValueError('Score/scene ordering differs')
    old=read(OLD/'result.json')
    if units!=old['units']:
        raise ValueError('Original48 measurement units must be reused in original order')
    for scene,prior in zip(scenes,old['per_scene']):
        for field in ('unit','group','context'):
            if scene[field]!=prior[field]:
                raise ValueError('Original scene identity differs: '+field)
        np.testing.assert_allclose(scene['front_range_m'],prior['front_range_m'],rtol=0,atol=1e-12)
    bootstrap=plan.get('bootstrap',{})
    if bootstrap.get('seed')!=2026100401 or bootstrap.get('replicates',bootstrap.get('n'))!=1000:
        raise ValueError('Frozen scene-bootstrap configuration differs')
    if plan.get('old_result_sha256')!=hashes[str(OLD/'result.json')]:
        raise ValueError('Original result changed relative to new frozen PLAN')
    for path,digest in receipt.get('source_sha256',{}).items():
        if sha(path)!=digest:
            raise ValueError('Sealed scoring source changed: '+path)
        hashes[path]=digest
    result=analyze(scores,scenes,seed=2026100401,n_boot=1000)
    parity,threshold_parity=0,0
    for i,(current,prior) in enumerate(zip(result['per_scene'],old['per_scene'])):
        for domain in current['auc']:
            for arm in ('M3_smooth','oracle8','oracle12'):
                a,b=current['auc'][domain][arm],prior['cells'][domain][arm]
                if a['positive_n']!=b['positive_n'] or a['negative_n']!=b['negative_n']:
                    raise ValueError('Original AUC denominator changed')
                if (a['auc'] is None)!=(b['auc'] is None) or (a['auc'] is not None and abs(a['auc']-b['auc'])>1e-12):
                    raise ValueError('Frozen oracle/M3 AUC does not reproduce original result')
                parity+=1
            low,high=(1.2,2.1) if domain==PRIMARY else map(float,domain[:-1].split('-'))
            ranges=np.array(scenes[i]['front_range_m'])
            keep=(ranges>=low)&(ranges<high)
            for index,delta in enumerate(DELTA):
                values=scores['M3_smooth'][i,index][:,keep]
                expected=prior['M3_frozen_threshold_curves'][domain][str(delta)]
                if values.size!=expected['n'] or int((values>=M3_THRESHOLD).sum())!=expected['stops']:
                    raise ValueError('Original frozen M3 threshold counts changed')
                threshold_parity+=1
    result['original_auc_point_parity']=dict(status='PASS',per_scene_fields=parity,
        arms=['M3_smooth','oracle8','oracle12'],role='Point values only; new shared bootstrap seed does not rewrite original CIs')
    result['original_M3_threshold_parity']=dict(status='PASS',per_scene_branch_domain_fields=threshold_parity)
    result['branch_class']=result['verdict']
    result['verdict']={'NEAR':'NEAR_REALISTIC_REFERENCE','HEADROOM':'REALISTIC_HEADROOM',
        'INTERMEDIATE':'INTERMEDIATE','NOT_EVALUABLE':'NOT_EVALUABLE'}[result['branch_class']]
    result['provenance']=dict(input_sha256=hashes,plan=plan,score_receipt=receipt,
        operations=dict(training=False,model_inference=False,likelihood_scoring=False,real_replay=False,
            evaluation_thresholds_descriptive_only=True),original_results_preserved=True)
    for path,digest in hashes.items():
        if sha(path)!=digest:
            raise ValueError('Input changed during evaluation: '+path)
    save(result_path,result)
    write_report(result,out)
    print(result['verdict'],json.dumps(result['decision']))
    return result


def check():
    assert binary_auc([0.,1.], [0.,-1.]) == .875
    assert matched_threshold([1,1,2,3], 2)['actual_stops'] == 2
    assert matched_threshold([1,2,2,3], 2)['actual_stops'] == 1
    assert matched_threshold([1,2,2,3], 0)['actual_stops'] == 0
    assert matched_threshold([1,2,2,3], 4)['actual_stops'] == 4
    assert branch_decision(.03,[.05]*4) == 'NEAR'
    assert branch_decision(.06,[0]*4) == 'HEADROOM'
    assert branch_decision(.04,[.10,0,0,0]) == 'HEADROOM'
    assert branch_decision(.04,[.06,0,0,0]) == 'INTERMEDIATE'
    assert branch_decision(None,[0]*4) == 'NOT_EVALUABLE'
    scenes=[dict(unit=i,group=i%2,context='none' if i%4<2 else 'panel',front_range_m=np.linspace(.65,2.57,13).tolist()) for i in range(48)]
    raw=np.broadcast_to(np.array([2.,1.5,3.,0.,-.5,-1.,-2.])[None,:,None,None],(48,7,4,13)).copy()
    result=analyze({arm:raw.copy() for arm in ARMS}, scenes, seed=2026100401, n_boot=1000)
    assert result['verdict']=='NEAR'
    assert result['decision']['auc_difference']==0
    for workpoints in result['matched_cost'].values():
        for cell in workpoints.values():
            for arm in ARMS[1:]:
                assert cell['thresholds'][arm]['actual_stops']<=cell['thresholds'][arm]['allowed_stops']
            assert cell['cells']['all']['U8']['inside1cm']['rate']==1
        # Separate operating points must not silently collapse to pooled cost:
        # at the zero outside20 budget, the same synthetic arm alarms all15.
        assert workpoints['outside20cm']['cells']['all']['U8']['outside15cm']['rate']==1
        assert workpoints['outside15cm']['cells']['all']['U8']['outside15cm']['rate']==0
    json.dumps(result,allow_nan=False)
    print('PASS half-credit AUC, strict-tie cost matching, zero/all budgets, point-branch boundaries and paired-scene aggregate')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=('check','evaluate','finalize'))
    args=parser.parse_args()
    {'check':check,'evaluate':evaluate,'finalize':evaluate}[args.stage]()
