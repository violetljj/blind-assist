"""Independent height-factorial predictions, budgets, grades and accounting.

Only exact-hash locally generated estimators are unpickled. Raw/feature/metric
reconstruction comes from the prior independent audit, not production helpers.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import audit_cnh_graded_corridor_eval_dev as A

OUT = A.ROOT/'artifacts.local/work/cnh-graded-height-cal-dev-20261010'
OLD = A.OUT/'interaction'
MODES = ['shared_shared','shared_height','separate_shared','separate_height']
PARAMS = dict(learning_rate=.1,max_iter=100,max_leaf_nodes=7,max_depth=3,
    min_samples_leaf=50,l2_regularization=1.,early_stopping=False,random_state=20261010)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allocation(total,weights):
    # Independent two-category Hamilton allocation, with explicit HEAD tie.
    h,b = map(int,weights)
    if not h+b:
        assert total==0
        return [0,0]
    numerators = [total*h,total*b]
    quotients = [v//(h+b) for v in numerators]
    missing = total-sum(quotients)
    assert missing in (0,1)
    if missing:
        remainders = [v%(h+b) for v in numerators]
        quotients[0 if remainders[0]>=remainders[1] else 1] += 1
    assert sum(quotients)==total
    return quotients


def cutoff(score,eligible,category,original,fraction,height):
    passed = (category=='pass').any(1)&~(category=='contact').any(1)
    clear = (category=='clear').all(1)
    bp = int(original[passed].any((-1,-2)).sum())
    bc = int(original[clear].any(-1).sum())
    tp,tc = int(fraction*bp),int(fraction*bc)
    candidate = np.where(eligible,score,-np.inf)
    extra = {}
    if height:
        wp = original[passed].any(-2).sum((0,1)).astype(int).tolist()
        wc = original[clear].sum((0,1,2)).astype(int).tolist()
        hp,hc = allocation(tp,wp),allocation(tc,wc)
        theta = [max(A.threshold(candidate[passed,...,q].max(-1),hp[q]),
                     A.threshold(candidate[clear,...,q],hc[q])) for q in range(2)]
    else:
        theta = max(A.threshold(candidate[passed].max((-1,-2)),tp),
                    A.threshold(candidate[clear].max(-1),tc))
    flags = eligible&(score>=np.asarray(theta))
    ap = int(flags[passed].any((-1,-2)).sum())
    ac = int(flags[clear].any(-1).sum())
    if height:
        ahp = flags[passed].any(-2).sum((0,1)).astype(int).tolist()
        ahc = flags[clear].sum((0,1,2)).astype(int).tolist()
        assert all(v<=t for v,t in zip(ahp,hp)) and all(v<=t for v,t in zip(ahc,hc))
        assert ap<=sum(ahp)<=tp and ac<=sum(ahc)<=tc
        extra = dict(baseline_height_pass_clips=wp,baseline_height_clear_slots=wc,
            height_target_pass_clips=hp,height_target_clear_slots=hc,
            height_actual_pass_clips=ahp,height_actual_clear_slots=ahc,
            height_sum_actual_pass_clips=sum(ahp),height_sum_actual_clear_slots=sum(ahc),
            pass_union_overlap=sum(ahp)-ap,clear_union_overlap=sum(ahc)-ac)
    assert ap<=tp and ac<=tc
    return dict(theta=theta,fraction=fraction,baseline_light_pass_clips=bp,baseline_light_clear_slots=bc,
        target_light_pass_clips=tp,target_light_clear_slots=tc,actual_light_pass_clips=ap,
        actual_light_clear_slots=ac,pass_residual=ap-tp,clear_residual=ac-tc,**extra)


def run():
    began = time.monotonic();audit = OUT/'audit';audit.mkdir(exist_ok=True)
    if (audit/'result.json').exists(): raise FileExistsError('Preserve completed audit')
    plan = A.read(OUT/'PLAN.json');records = A.read(OUT/'models.json');cuts = A.read(OUT/'calibrations.json')
    metrics = A.read(OUT/'metrics.json');old_records = A.read(OLD/'models.json')
    producer = Path(__file__).with_name('cnh_graded_height_cal_dev.py')
    A.eq(digest(producer),plan['source_sha256'],'producer hash')
    for path,value in plan['inputs_sha256'].items(): A.eq(digest(A.ROOT/path),value,'input hash')
    A.eq(plan['modes'],MODES,'factorial modes');A.eq(plan['banks'],['score_only','score_spatial'],'banks')
    A.eq(plan['policies'],['filter','replace'],'policies');A.eq(plan['fractions'],[.25,.5,.75],'fractions')
    A.eq(plan['params'],PARAMS,'fixed HGB params')
    tree = ast.parse(producer.read_text(encoding='utf8'))
    fits = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='fit']
    assert len(fits)==1
    A.eq([ast.unparse(n) for n in fits[0].args],
        ["tensor['cal'][bank][..., q:q + 1, :]","data['cal']['category'][:, q:q + 1]","refs['cal']['strong'][..., q:q + 1]",'fit_mask'],'height fit cal-only inputs')
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='calibrator']
    assert len(calls)==1
    A.eq([ast.unparse(n) for n in calls[0].args],
        ["score['cal']","eligible['cal']","data['cal']['category']",'~fit_mask',"refs['cal']['light']",'fraction'],'height cutoff cal-only inputs')
    ds = A.load();families = {}
    for row in ds['cal']['rows']: families.setdefault(row['background_family'],set()).add(row['background_id'])
    assert all(len(v)==2 for v in families.values())
    fit_ids = {min(v) for v in families.values()};cut_ids = {r['background_id'] for r in ds['cal']['rows']}-fit_ids
    validation_ids = {r['background_id'] for r in ds['validation']['rows']}
    assert fit_ids.isdisjoint(cut_ids) and validation_ids.isdisjoint(fit_ids|cut_ids)
    partition = dict(fit_background_ids=sorted(fit_ids),calibrate_background_ids=sorted(cut_ids))
    A.eq(A.read(OUT/'cal_partition.json'),partition,'partition');A.eq(A.read(OLD/'cal_partition.json'),partition,'old partition')
    fit = np.array([r['background_id'] in fit_ids for r in ds['cal']['rows']]);cut = ~fit
    contract = A.read(A.OUT/'feature_contract.json');spatial = {};shared_saved = {};separate_saved = {};grades_saved = {};old_grades = {}
    for split,d in ds.items():
        spatial[split],ids = A.spatial(split,contract);np.testing.assert_array_equal(ids,d['ids'])
        with np.load(OLD/f'{split}_scores.npz',allow_pickle=False) as a: shared_saved[split] = dict(zip(a['keys'].tolist(),a['scores']))
        with np.load(OUT/f'{split}_separate_scores.npz',allow_pickle=False) as a: separate_saved[split] = dict(zip(a['keys'].tolist(),a['scores']))
        with np.load(OUT/f'{split}_grades.npz',allow_pickle=False) as a: grades_saved[split] = dict(zip(a['keys'].tolist(),a['grades']))
        with np.load(OLD/f'{split}_grades.npz',allow_pickle=False) as a: old_grades[split] = dict(zip(a['keys'].tolist(),a['grades']))
    ths = A.read(A.ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
    replayed = {};firsts = {};model_hashes = {};fit_checks = {};budget_rows = [];comparisons = []
    old_cuts = A.read(OLD/'calibrations.json')
    for seed in A.SEEDS:
        th = ths[str(seed)];tensor = {};refs = {}
        for split,d in ds.items():
            ordinary = A.smooth(d['raw'][seed]);base = A.score_features(d['raw'][seed],th['single'])
            tensor[split] = dict(score_only=base,score_spatial=np.concatenate((base,spatial[split]),axis=-1))
            old = (d['m3']>=.9404184587540165)|(d['local']>=4.625390338985158)
            strong = old|(ordinary>=th['addition']);light = ~strong&(ordinary>=th['single'])
            refs[split] = dict(ordinary_OR=strong,prior_light=strong|light,M3=d['m3']>=.8557642486787612,old_fusion=old)
        for bank in plan['banks']:
            scores = {kind:{split:np.empty_like(refs[split]['ordinary_OR'],dtype=float) for split in ds} for kind in ('shared','separate')}
            for height in ('SHARED','HEAD','BODY'):
                if time.monotonic()-began>60: raise TimeoutError('Height audit60s cap')
                shared = height=='SHARED';q = ('HEAD','BODY').index(height) if not shared else None
                key = f'{seed}/{bank}'+('' if shared else '/'+height)
                record = old_records[key] if shared else records[key];root = OLD if shared else OUT
                path = (root/record['path']).resolve();assert path.parent==(root/'models').resolve()
                h = digest(path);A.eq(h,record['sha256'],'model hash/'+key);model_hashes[key] = h
                model = pickle.loads(path.read_bytes());assert type(model) is HistGradientBoostingClassifier
                A.eq({name:model.get_params()[name] for name in PARAMS},PARAMS,'params/'+key)
                A.eq(model.n_features_in_,3 if bank=='score_only' else 37,'runtime dimensions')
                A.eq(model.n_iter_,100,'iterations');np.testing.assert_array_equal(model.classes_,[0,1])
                strong = refs['cal']['ordinary_OR'][fit]
                cat = ds['cal']['category'][fit]
                if not shared: strong,cat = strong[...,q:q+1],cat[:,q:q+1]
                eligible = ~strong;count = eligible.sum(-2,keepdims=True)
                weight = np.broadcast_to(np.divide(1.,count,out=np.zeros_like(count,dtype=float),where=count>0),eligible.shape)[eligible]
                weight *= len(weight)/weight.sum()
                y = np.broadcast_to((cat=='contact')[:,None,None,:],eligible.shape)[eligible]
                A.eq(len(y),record['rows'],'fit rows');A.eq(int(y.sum()),record['contact_rows'],'fit labels')
                assert abs(weight.sum()-record['weight_sum'])<1e-8
                prior = np.sum(weight*y)/weight.sum();baseline = np.log(prior/(1-prior))
                assert abs(model._baseline_prediction.item()-baseline)<1e-12
                fit_checks[key] = dict(rows=len(y),contacts=int(y.sum()),weighted_contact_fraction=float(prior),baseline_logit=float(baseline))
                for split in ds:
                    x = tensor[split][bank] if shared else tensor[split][bank][...,q,:]
                    score = model.decision_function(x.reshape(-1,x.shape[-1])).reshape(x.shape[:-1])
                    if shared:
                        scores['shared'][split] = score;np.testing.assert_array_equal(score,shared_saved[split][f'{seed}/{bank}'])
                    else: scores['separate'][split][...,q] = score
            for split in ds: np.testing.assert_array_equal(scores['separate'][split],separate_saved[split][f'{seed}/{bank}/separate'])
            for mode in MODES:
                kind = mode.split('_')[0];height_cut = mode.endswith('_height')
                for policy in plan['policies']:
                    for fraction in plan['fractions']:
                        key = f'{seed}/{bank}/{mode}/{policy}/{fraction}'
                        original = refs['cal']['prior_light']&~refs['cal']['ordinary_OR']
                        eligible = original if policy=='filter' else ~refs['cal']['ordinary_OR']
                        record = cutoff(scores[kind]['cal'][cut],eligible[cut],ds['cal']['category'][cut],original[cut],fraction,height_cut)
                        A.eq(record,cuts[key],'calibration/'+key)
                        if mode=='shared_shared': A.eq(record,old_cuts[f'{seed}/{bank}/{policy}/{fraction}'],'old cutoff exact')
                        budget_rows.append(dict(key=key,**record))
                        for split,d in ds.items():
                            strong = refs[split]['ordinary_OR'];ep = refs[split]['prior_light']&~strong if policy=='filter' else ~strong
                            grade = np.where(strong,2,np.where(ep&(scores[kind][split]>=np.asarray(record['theta'])),1,0)).astype(np.int8)
                            np.testing.assert_array_equal(grade,grades_saved[split][key]);np.testing.assert_array_equal(grade==2,strong)
                            if policy=='filter': assert np.all((grade==0)|refs[split]['prior_light'])
                            if mode=='shared_shared': np.testing.assert_array_equal(grade,old_grades[split][f'{seed}/{bank}/{policy}/{fraction}'])
                            report = A.describe(grade,d,refs[split]);A.eq(report,metrics[split+'/'+key],'metrics/'+split+'/'+key)
                            replayed[(split,key)] = report
                            if split=='validation': firsts[key] = (A.first(refs[split]['prior_light']),A.first(grade>0))
    rows = 0;seen = set();lookup = {int(v):i for i,v in enumerate(ds['validation']['ids'])}
    with (OUT/'event_ledger.csv').open(encoding='utf8',newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['key'];i = lookup[int(row['scene'])];k = int(row['replica']);q = ('HEAD','BODY').index(row['height'])
            identity = (key,i,k,q);assert identity not in seen;seen.add(identity)
            before,after = firsts[key]
            A.eq(int(row['before_first']),int(before[i,k,q]),'ledger/before');A.eq(int(row['after_first']),int(after[i,k,q]),'ledger/after')
            A.eq(row['category'],ds['validation']['category'][i,q],'ledger/category')
            for col,field in (('family','shape_family'),('background_family','background_family')): A.eq(row[col],str(ds['validation']['rows'][i][field]),'ledger/'+col)
            rows += 1
    A.eq(rows,len(grades_saved['validation'])*len(ds['validation']['rows'])*4*2,'complete event ledger')
    seen = set()
    with (OUT/'summary.csv').open(encoding='utf8',newline='') as stream:
        for row in csv.DictReader(stream):
            key = '/'.join(row[f] for f in ('seed','bank','mode','policy','fraction'));identity=(row['split'],key);assert identity not in seen;seen.add(identity);r=replayed[identity];p=r['paired_any']['prior_light']
            expected = dict(HEAD=r['any']['counts'][0],BODY=r['any']['counts'][1],HEAD_gain_prior=p[0]['rescue'],HEAD_loss_prior=p[0]['loss'],BODY_gain_prior=p[1]['rescue'],BODY_loss_prior=p[1]['loss'],HEAD_delay=p[0]['later'],BODY_delay=p[1]['later'],clear_slots=r['any']['clear_slots'],pass_clips=r['any']['pass_clips'],light_clear_slots=r['joint_costs']['clear']['light']['slots'],light_pass_slots=r['joint_costs']['pass']['light']['slots'],light_pass_clips=r['joint_costs']['pass']['light']['clips'])
            for field,value in expected.items(): A.eq(int(row[field]),value,'summary/'+field)
    A.eq(len(seen),288,'summary cells');A.eq(len(replayed),288,'replayed cells');A.eq(len(cuts),144,'calibrations')
    for seed in A.SEEDS:
        for bank in plan['banks']:
            for policy in plan['policies']:
                for fraction in plan['fractions']:
                    base=f'{seed}/{bank}/shared_shared/{policy}/{fraction}'
                    for mode in MODES[1:]:
                        key=f'{seed}/{bank}/{mode}/{policy}/{fraction}';a,b=replayed[('validation',base)],replayed[('validation',key)]
                        comparisons.append(dict(key=key,HEAD_delta=b['any']['counts'][0]-a['any']['counts'][0],BODY_delta=b['any']['counts'][1]-a['any']['counts'][1],pass_clips_delta=b['any']['pass_clips']-a['any']['pass_clips'],clear_slots_delta=b['any']['clear_slots']-a['any']['clear_slots'],paired=A.paired(grades_saved['validation'][base]>0,grades_saved['validation'][key]>0,ds['validation']['category'])))
    result = dict(status='PASS',cells=288,calibrations=144,exact_hash_models=18,new_height_models=12,score_arrays_exact=24,
        old_shared_grade_cells_exact=72,event_rows=rows,scalar_checks=A.CHECKS,partition=partition,
        validation_background_ids=sorted(validation_ids),model_hashes=model_hashes,fit_checks=fit_checks,
        accounting='All candidate-light union budgets, height allocations/sums/overlap and max-grade actual metrics independently replayed; full paired timing and ledger; no producer policy/metric functions.',
        fit_scope='Prediction replay with exact-hash task-owned sklearn estimators; cal-only source call arguments, rows, weight scale and weighted baseline checked. No estimator retraining.',
        seconds=time.monotonic()-began,audit_source_sha256=digest(Path(__file__)))
    assert result['seconds']<60
    (audit/'budget_checks.json').write_text(json.dumps(budget_rows,indent=2)+'\n',encoding='utf8')
    (audit/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n',encoding='utf8')
    (audit/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':
    with threadpool_limits(limits=2): run()
