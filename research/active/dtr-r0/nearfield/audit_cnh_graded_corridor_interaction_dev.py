"""Replay locally produced, hash-verified shallow interaction models.

Uses independent raw-cache/features/accounting functions from the linear audit.
No training, production policy helpers, new observations or GPU allocation.
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

OUT = A.OUT/'interaction'
PARAMS = dict(learning_rate=.1,max_iter=100,max_leaf_nodes=7,max_depth=3,
    min_samples_leaf=50,l2_regularization=1.,early_stopping=False,random_state=20261010)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    began = time.monotonic();audit = OUT/'audit';audit.mkdir(exist_ok=True)
    if (audit/'result.json').exists(): raise FileExistsError('Preserve completed audit')
    plan = A.read(OUT/'PLAN.json');records = A.read(OUT/'models.json');cuts = A.read(OUT/'calibrations.json');metrics = A.read(OUT/'metrics.json')
    producer = Path(__file__).with_name('cnh_graded_corridor_interaction_dev.py')
    A.eq(digest(producer),plan['source_sha256'],'producer hash')
    A.eq(digest(A.OUT/'PLAN.json'),plan['parent_plan_sha256'],'parent plan hash')
    A.eq(plan['params'],PARAMS,'fixed parameters')
    A.eq(plan['models'],['score_only','score_spatial'],'model banks')
    A.eq(plan['policies'],['filter','replace'],'policies');A.eq(plan['fractions'],[.25,.5,.75],'fractions')
    # Static call arguments verify the producing fit and cutoff consume cal only.
    tree = ast.parse(producer.read_text(encoding='utf8'))
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    fits = [n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='fit']
    assert len(fits)==1
    fit_args = [ast.unparse(n) for n in fits[0].args]
    A.eq(fit_args,["tensor['cal'][bank]","data['cal']['category']","refs['cal']['strong']",'fit_mask'],'fit inputs')
    calibrations = [n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='calibrate']
    assert len(calibrations)==1
    A.eq([ast.unparse(n) for n in calibrations[0].args],
        ["score['cal']","eligible['cal']","data['cal']['category']",'~fit_mask',"refs['cal']['light']",'fraction'],'cutoff inputs')
    ds = A.load();families = {}
    for row in ds['cal']['rows']: families.setdefault(row['background_family'],set()).add(row['background_id'])
    assert all(len(v)==2 for v in families.values())
    fit_ids = {min(v) for v in families.values()};cut_ids = {r['background_id'] for r in ds['cal']['rows']}-fit_ids
    validation_ids = {r['background_id'] for r in ds['validation']['rows']}
    assert fit_ids.isdisjoint(cut_ids) and validation_ids.isdisjoint(fit_ids|cut_ids)
    partition = dict(fit_background_ids=sorted(fit_ids),calibrate_background_ids=sorted(cut_ids))
    A.eq(A.read(OUT/'cal_partition.json'),partition,'interaction partition');A.eq(A.read(A.OUT/'cal_partition.json'),partition,'parent partition')
    fit = np.array([r['background_id'] in fit_ids for r in ds['cal']['rows']]);cut = ~fit
    contract = A.read(A.OUT/'feature_contract.json');spatial = {};scores_saved = {};grades_saved = {}
    forbidden = ('truth','category','background_id','scene_id','rho','placement','target')
    assert not any(any(word in name for word in forbidden) for name in contract['score_names']+contract['spatial_names'])
    for split,d in ds.items():
        spatial[split],ids = A.spatial(split,contract);np.testing.assert_array_equal(ids,d['ids'])
        with np.load(OUT/f'{split}_scores.npz',allow_pickle=False) as a: scores_saved[split] = dict(zip(a['keys'].tolist(),a['scores']))
        with np.load(OUT/f'{split}_grades.npz',allow_pickle=False) as a: grades_saved[split] = dict(zip(a['keys'].tolist(),a['grades']))
    ths = A.read(A.ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
    replayed = {};comparisons = [];model_hashes = {};fit_baselines = {}
    for seed in A.SEEDS:
        th = ths[str(seed)];tensor = {};refs = {}
        for split,d in ds.items():
            ordinary = A.smooth(d['raw'][seed]);base = A.score_features(d['raw'][seed],th['single'])
            tensor[split] = dict(score_only=base,score_spatial=np.concatenate((base,spatial[split]),axis=-1))
            old = (d['m3']>=.9404184587540165)|(d['local']>=4.625390338985158)
            strong = old|(ordinary>=th['addition']);light = ~strong&(ordinary>=th['single'])
            refs[split] = dict(ordinary_OR=strong,prior_light=strong|light,M3=d['m3']>=.8557642486787612,old_fusion=old)
        for bank in plan['models']:
            if time.monotonic()-began>60: raise TimeoutError('Interaction audit60s cap')
            model_key = f'{seed}/{bank}';record = records[model_key];path = (OUT/record['path']).resolve()
            assert path.parent == (OUT/'models').resolve(),('unexpected pickle location',path)
            observed_hash = digest(path);A.eq(observed_hash,record['sha256'],'model hash/'+model_key);model_hashes[model_key] = observed_hash
            # Only the current task's locally generated exact-hash pickle is loaded.
            model = pickle.loads(path.read_bytes());assert type(model) is HistGradientBoostingClassifier
            A.eq({name:model.get_params()[name] for name in PARAMS},PARAMS,'params/'+model_key)
            A.eq(model.n_features_in_,3 if bank=='score_only' else 37,'runtime input dimensions')
            np.testing.assert_array_equal(model.classes_,[0,1]);A.eq(model.n_iter_,100,'fixed iterations')
            eligible = ~refs['cal']['ordinary_OR'][fit];count = eligible.sum(-2,keepdims=True)
            weight = np.broadcast_to(np.divide(1.,count,out=np.zeros_like(count,dtype=float),where=count>0),eligible.shape)[eligible]
            weight *= len(weight)/weight.sum()
            y = np.broadcast_to((ds['cal']['category'][fit]=='contact')[:,None,None,:],eligible.shape)[eligible]
            A.eq(len(y),record['rows'],'fit rows');A.eq(int(y.sum()),record['contact_rows'],'fit contact rows')
            assert abs(weight.sum()-record['weight_sum'])<1e-8
            prior = np.sum(weight*y)/weight.sum();expected_baseline = np.log(prior/(1-prior))
            assert abs(model._baseline_prediction.item()-expected_baseline)<1e-12
            fit_baselines[model_key] = dict(rows=len(y),contact_rows=int(y.sum()),weighted_contact_fraction=float(prior),baseline_logit=float(expected_baseline))
            scores = {}
            for split in ds:
                x = tensor[split][bank]
                scores[split] = model.decision_function(x.reshape(-1,x.shape[-1])).reshape(x.shape[:-1])
                np.testing.assert_array_equal(scores[split],scores_saved[split][model_key])
            for policy in plan['policies']:
                for fraction in plan['fractions']:
                    key = f'{seed}/{bank}/{policy}/{fraction}';record = cuts[key];cat = ds['cal']['category'][cut]
                    passed = (cat=='pass').any(1)&~(cat=='contact').any(1);clear = (cat=='clear').all(1)
                    original = refs['cal']['prior_light']&~refs['cal']['ordinary_OR']
                    ep = original if policy=='filter' else ~refs['cal']['ordinary_OR']
                    candidate = np.where(ep[cut],scores['cal'][cut],-np.inf)
                    bp,bc = int(original[cut][passed].any((-1,-2)).sum()),int(original[cut][clear].any(-1).sum())
                    tp,tc = int(fraction*bp),int(fraction*bc)
                    theta = max(A.threshold(candidate[passed].max((-1,-2)),tp),A.threshold(candidate[clear].max(-1),tc))
                    A.eq(theta,record['theta'],'whole-tie cutoff/'+key)
                    flags = ep[cut]&(scores['cal'][cut]>=theta)
                    ap,ac = int(flags[passed].any((-1,-2)).sum()),int(flags[clear].any(-1).sum())
                    A.eq(record,dict(theta=theta,fraction=fraction,baseline_light_pass_clips=bp,baseline_light_clear_slots=bc,target_light_pass_clips=tp,target_light_clear_slots=tc,actual_light_pass_clips=ap,actual_light_clear_slots=ac,pass_residual=ap-tp,clear_residual=ac-tc),'cal/'+key)
                    assert ap<=tp and ac<=tc
                    for split,d in ds.items():
                        strong = refs[split]['ordinary_OR'];eligible = (refs[split]['prior_light']&~strong) if policy=='filter' else ~strong
                        grade = np.where(strong,2,np.where(eligible&(scores[split]>=theta),1,0)).astype(np.int8)
                        np.testing.assert_array_equal(grade,grades_saved[split][key]);np.testing.assert_array_equal(grade==2,strong)
                        if policy=='filter': assert np.all((grade==0)|refs[split]['prior_light'])
                        report = A.describe(grade,d,refs[split]);A.eq(report,metrics[split+'/'+key],'metrics/'+split+'/'+key)
                        replayed[(split,key)] = report
        for policy in plan['policies']:
            for fraction in plan['fractions']:
                ka,kb = f'{seed}/score_only/{policy}/{fraction}',f'{seed}/score_spatial/{policy}/{fraction}'
                aa,bb = replayed[('validation',ka)],replayed[('validation',kb)]
                comparisons.append(dict(seed=seed,policy=policy,fraction=fraction,
                    HEAD_delta=bb['any']['counts'][0]-aa['any']['counts'][0],BODY_delta=bb['any']['counts'][1]-aa['any']['counts'][1],
                    pass_clips_delta=bb['any']['pass_clips']-aa['any']['pass_clips'],clear_slots_delta=bb['any']['clear_slots']-aa['any']['clear_slots'],
                    paired_contacts=A.paired(grades_saved['validation'][ka]>0,grades_saved['validation'][kb]>0,ds['validation']['category']),
                    cal_score_only=cuts[ka],cal_score_spatial=cuts[kb]))
    seen = set()
    with (OUT/'summary.csv').open(encoding='utf8',newline='') as stream:
        for row in csv.DictReader(stream):
            key = '/'.join((row['seed'],row['model'],row['policy'],row['fraction']));identity=(row['split'],key);assert identity not in seen;seen.add(identity);r=replayed[identity]
            expected = dict(HEAD=r['any']['counts'][0],BODY=r['any']['counts'][1],HEAD_gain_prior=r['paired_any']['prior_light'][0]['rescue'],HEAD_loss_prior=r['paired_any']['prior_light'][0]['loss'],BODY_gain_prior=r['paired_any']['prior_light'][1]['rescue'],BODY_loss_prior=r['paired_any']['prior_light'][1]['loss'],HEAD_extra_strong=r['paired_any']['ordinary_OR'][0]['rescue'],BODY_extra_strong=r['paired_any']['ordinary_OR'][1]['rescue'],clear_slots=r['any']['clear_slots'],pass_clips=r['any']['pass_clips'],light_clear_slots=r['joint_costs']['clear']['light']['slots'],light_pass_slots=r['joint_costs']['pass']['light']['slots'],light_pass_clips=r['joint_costs']['pass']['light']['clips'])
            for field,value in expected.items(): A.eq(int(row[field]),value,'summary/'+field)
    A.eq(len(seen),72,'summary cells');A.eq(len(replayed),72,'replayed cells')
    result = dict(status='PASS',cells=72,hash_verified_models=6,score_arrays_exact=12,calibrations=36,scalar_checks=A.CHECKS,
        partition=partition,validation_background_ids=sorted(validation_ids),model_hashes=model_hashes,fit_weighted_baseline_checks=fit_baselines,
        independent_accounting='Reuses linear audit raw/feature/tie/metrics functions; prediction uses exact-hash task-owned sklearn estimator; producing fit and cal-only call arguments checked against source hash; no estimator retraining',
        scope='No event ledger exists; earliest contact timing and full paired histogram independently checked via saved grades and metrics.',
        seconds=time.monotonic()-began,audit_source_sha256=digest(Path(__file__)))
    (audit/'comparisons.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (audit/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))
    print(json.dumps([dict(seed=r['seed'],policy=r['policy'],fraction=r['fraction'],HEAD_delta=r['HEAD_delta'],BODY_delta=r['BODY_delta'],pass_clips_delta=r['pass_clips_delta'],clear_slots_delta=r['clear_slots_delta']) for r in comparisons],ensure_ascii=False))

if __name__=='__main__':
    with threadpool_limits(limits=2): run()
