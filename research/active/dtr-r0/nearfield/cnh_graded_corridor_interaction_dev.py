"""Fixed shallow nonlinear support interactions, after consumed linear pilot.

Keeps original strong slots and the same authoring-instance fit/cutoff split.
This exploratory comparison is not validation-independent or a new backbone.
"""
import json
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E

OUT = C.OUT/'interaction'
PARAMS = dict(learning_rate=.1,max_iter=100,max_leaf_nodes=7,max_depth=3,
    min_samples_leaf=50,l2_regularization=1.,early_stopping=False,random_state=20261010)
MODELS = ('score_only','score_spatial')


def fit(x,category,strong,mask):
    eligible = ~strong[mask]
    count = eligible.sum(-2,keepdims=True)
    weight = np.broadcast_to(np.divide(1.,count,out=np.zeros_like(count,dtype=float),where=count>0),eligible.shape)[eligible]
    # Mean weight one gives fixed regularization a stable effective sample scale.
    weight *= len(weight)/weight.sum()
    y = np.broadcast_to((category[mask]=='contact')[:,None,None,:],eligible.shape)[eligible].astype(int)
    model = HistGradientBoostingClassifier(**PARAMS).fit(x[mask][eligible],y,sample_weight=weight)
    return model,dict(rows=len(y),contact_rows=int(y.sum()),weight_sum=float(weight.sum()),iterations=int(model.n_iter_))


def run():
    began = time.monotonic()
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Preserve interaction exploration')
    C.save(OUT/'PLAN.json',dict(task='CNH_GRADED_CORRIDOR_INTERACTION_DEV_20261010',
        motivation='Linear spatial evidence has mixed incremental effect; test fixed nonlinear peak/support/depth interactions against matched nonlinear score-only.',
        status='Exploratory selected after consumed linear validation, no independence claim',
        parent_plan_sha256=C.sha(C.OUT/'PLAN.json'),source_sha256=C.sha(Path(__file__)),
        budget='Within parent900s analysis CPU including audits; this command cap300s, no GPU/backbone/newdraws.',
        stop='One fixed parameter set, both banks, all3seeds, declared policies/fractions; no validation retuning.',
        params=PARAMS,models=list(MODELS),policies=list(C.POLICIES),fractions=list(C.FRACTIONS),
        fit='Same cal-fit instances/equal eligible sceneKquery weights as linear, normalized mean1; contact label only in fit/evaluation.',
        strong='Exactly prior ordinary_OR, unchanged slot mask.',
        calibration='Same separated cal-cutoff instances and complete ties light passclip/clearjointslot budgets as linear.',
        features='Identical score3/spatial34 descriptors and missing indicators; no new features chosen from validation outcomes.'))
    try:
        data = C.load()
        fit_mask,partitions = C.split_cal(data['cal']['rows'])
        C.save(OUT/'cal_partition.json',partitions)
        thresholds = C.read(C.PARENT/'thresholds.json')
        spatial = {}
        for split,ds in data.items():
            with np.load(C.OUT/'features'/f'{split}_features.npz',allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['scene_ids'],ds['scene_ids'])
                spatial[split],_ = C.spatial_features(archive)
        summaries,metrics,calibrations,models = [],{},{},{}
        saved = {split:[] for split in data}
        scores_saved = {split:[] for split in data}
        for si,seed in enumerate(G.SEEDS):
            if time.monotonic()-began>300:
                raise TimeoutError('Interaction CPU cap300s')
            th = thresholds[str(seed)]
            tensor,refs = {},{}
            for split,ds in data.items():
                ordinary = ds['candidates'][0,si]
                raw = E.read_job(C.SOURCE/f'scores/ordinary_seed{seed}_{split}.npz','ordinary',seed,split,'ideal')
                base = C.build_score_features(raw,ordinary,th['single'])
                tensor[split] = dict(score_only=base,score_spatial=np.concatenate((base,spatial[split]),-1))
                strong = E.old_fusion(ds['m3'],ds['local'])|(ordinary>=th['addition'])
                light = ~strong&(ordinary>=th['single'])
                refs[split] = dict(strong=strong,light=light,prior=strong|light)
            for bank in MODELS:
                model,fit_record = fit(tensor['cal'][bank],data['cal']['category'],refs['cal']['strong'],fit_mask)
                path = OUT/f'models/seed{seed}_{bank}.pickle'
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(pickle.dumps(model,protocol=5))
                models[f'{seed}/{bank}'] = dict(**fit_record,path=str(path.relative_to(OUT)),sha256=C.sha(path))
                score = {split:model.decision_function(x.reshape(-1,x.shape[-1])).reshape(x.shape[:-1]) for split,x in ((s,tensor[s][bank]) for s in data)}
                for split in data:
                    scores_saved[split].append((f'{seed}/{bank}',score[split]))
                for policy in C.POLICIES:
                    eligible = {s:r['light'] if policy=='filter' else ~r['strong'] for s,r in refs.items()}
                    for fraction in C.FRACTIONS:
                        key = f'{seed}/{bank}/{policy}/{fraction}'
                        record = C.calibrate(score['cal'],eligible['cal'],data['cal']['category'],~fit_mask,refs['cal']['light'],fraction)
                        calibrations[key] = record
                        for split,ds in data.items():
                            r = refs[split]
                            flag = eligible[split]&(score[split]>=record['theta'])
                            grade = np.where(r['strong'],2,np.where(flag,1,0)).astype(np.int8)
                            np.testing.assert_array_equal(grade==2,r['strong'])
                            comparison = dict(ordinary_OR=r['strong'],prior_light=r['prior'],M3=ds['m3']>=E.M3_THETA,
                                old_fusion=E.old_fusion(ds['m3'],ds['local']))
                            result = G.describe(grade,ds,comparison)
                            metrics[f'{split}/{key}'] = result
                            summaries.append(dict(split=split,seed=seed,model=bank,policy=policy,fraction=fraction,
                                HEAD=result['any']['counts'][0],BODY=result['any']['counts'][1],
                                HEAD_gain_prior=result['paired_any']['prior_light'][0]['rescue'],HEAD_loss_prior=result['paired_any']['prior_light'][0]['loss'],
                                BODY_gain_prior=result['paired_any']['prior_light'][1]['rescue'],BODY_loss_prior=result['paired_any']['prior_light'][1]['loss'],
                                HEAD_extra_strong=result['paired_any']['ordinary_OR'][0]['rescue'],BODY_extra_strong=result['paired_any']['ordinary_OR'][1]['rescue'],
                                clear_slots=result['any']['clear_slots'],pass_clips=result['any']['pass_clips'],
                                light_clear_slots=result['joint_costs']['clear']['light']['slots'],light_pass_slots=result['joint_costs']['pass']['light']['slots'],
                                light_pass_clips=result['joint_costs']['pass']['light']['clips']))
                            saved[split].append((key,grade))
        for split in data:
            np.savez_compressed(OUT/f'{split}_grades.npz',keys=np.asarray([k for k,v in saved[split]]),grades=np.asarray([v for k,v in saved[split]]))
            np.savez_compressed(OUT/f'{split}_scores.npz',keys=np.asarray([k for k,v in scores_saved[split]]),scores=np.asarray([v for k,v in scores_saved[split]]))
        G.write_csv(OUT/'summary.csv',summaries)
        C.save(OUT/'metrics.json',metrics);C.save(OUT/'models.json',models);C.save(OUT/'calibrations.json',calibrations)
        C.save(OUT/'receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,cells=len(summaries),model_fits=len(models),GPU_seconds=0))
        print(json.dumps(dict(status='COMPLETE',seconds=time.monotonic()-began,cells=len(summaries))))
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-began))
        raise


if __name__=='__main__':
    with threadpool_limits(limits=2):
        run()
