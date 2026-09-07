"""G12 representation evaluation with common-VAL recall-first primary thresholds."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from evaluate_whisker import normalize_predictions
from factorial_evaluate import primary_thresholds as compatibility_thresholds
from diversity_evaluate import evaluation_scopes,score_scopes
from grounding_evaluate import read,sha
from detached_evaluate import (fresh_scopes,load_maps,select_mask_thresholds,score_localization,
    baseline_reproduction,labels)

ARMS=('original_gate','repvit','repvit_detail')
SEEDS=('17','29','43','ensemble')


def normalize(payload,samples):
    predictions=normalize_predictions(payload,samples)
    if set(predictions)!={(arm,seed,'normal') for arm in ARMS for seed in SEEDS}:
        raise ValueError('Expected A/B/C representation arms, three seeds plus ensembles, normal only')
    if any(any(row[h]!=-1 for h in (2,3)) for rows in predictions.values() for row in rows.values()):
        raise ValueError('Closing predictions must remain UNKNOWN')
    return predictions


def recall_thresholds(predictions,samples,targets,variants):
    """Largest feasible [0,1] threshold minimizes FP then maximizes threshold.

    Inclusive >= alerts. Strict full known coverage of reference VAL near truth
    and predictions; closing UNKNOWN is outside this operating-point selection.
    Only positive score constraints bound the threshold; all VAL negatives are
    retained for explicit FP accounting, including an all-alarm outcome.
    """
    val=[s['sample_id'] for s in samples if s['split']=='val']
    if not val:
        raise ValueError('Reference VAL is required')
    selected={}
    for key,rows in predictions.items():
        heads=[]
        for h in range(2):
            if any(targets[sid][h] not in (0,1) for sid in val):
                raise ValueError('Full known VAL near truth coverage is required')
            if any(not math.isfinite(float(rows[sid][h])) or not 0<=float(rows[sid][h])<=1 for sid in val):
                raise ValueError('Full known VAL near prediction coverage is required')
            positives=[sid for sid in val if targets[sid][h]==1]
            constraints={'overall_positive':positives}
            if h==1:
                constraints['bar_only_positive']=[sid for sid in positives if variants[sid]=='bar_only']
            details={}
            bounds=[]
            for name,ids in constraints.items():
                if not ids:
                    raise ValueError('Missing VAL positive coverage: '+name)
                required=(95*len(ids)+99)//100
                ordered=sorted(float(rows[sid][h]) for sid in ids)
                bound=ordered[len(ids)-required]
                bounds.append(bound)
                details[name]=dict(positives=len(ids),required_TP=required,allowed_FN=len(ids)-required,
                                   maximum_feasible_threshold=bound)
            value=min(bounds)
            for name,ids in constraints.items():
                tp=sum(float(rows[sid][h])>=value for sid in ids)
                details[name].update(actual_TP=tp,actual_recall=tp/len(ids))
                if tp<details[name]['required_TP']:
                    raise AssertionError('Selected threshold violated recall constraint')
            negatives=[sid for sid in val if targets[sid][h]==0]
            fp=sum(float(rows[sid][h])>=value for sid in negatives)
            heads.append(dict(value=value,status='REFERENCE_VAL_RECALL_FIRST_SELECTED',
                validation_samples=len(val),known_truth_coverage=1.,known_prediction_coverage=1.,
                constraints=details,negatives=len(negatives),actual_FP=fp,
                actual_FPR=fp/len(negatives) if negatives else None,
                rule='Maximal feasible threshold with >=95% overall recall and HEAD bar_only >=95%; minimizes FP then maximizes threshold, >= alerts'))
        heads.extend([dict(value=1.,status='CLOSING_UNKNOWN_UNSCORED') for _ in range(2)])
        selected[key]=heads
    return selected


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('capture','learned','reference-capture','threshold-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference-baseline-predictions',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    fresh_prediction=args.learned/'predictions.json';reference_prediction=args.learned/'reference/predictions.json'
    paths={'fresh_predictions':fresh_prediction,'reference_predictions':reference_prediction,
        'threshold_source':args.threshold_source,'fresh_maps':args.learned/'support_predictions.npz',
        'reference_maps':args.learned/'reference/support_predictions.npz'}
    for prefix,capture in (('fresh',args.capture),('reference',args.reference_capture)):
        for name,relative in (('dataset','model/dataset.json'),('spec','evaluator/spec.json'),('labels','evaluator/labels.json'),
                              ('support','evaluator/support.npz'),('objects','evaluator/object_support.npz')):
            paths[prefix+'_'+name]=capture/relative
    if args.reference_baseline_predictions:
        paths['reference_baseline_predictions']=args.reference_baseline_predictions
    hashes={key:sha(path) for key,path in paths.items()}
    fresh=read(paths['fresh_dataset']);reference=read(paths['reference_dataset']);reference_spec=read(paths['reference_spec'])
    scopes=fresh_scopes(fresh,read(paths['fresh_spec']));splits,old_scopes=evaluation_scopes(reference,reference_spec)
    if {s['group_id'] for s in fresh['samples']} & {s['group_id'] for s in reference['samples']}:
        raise ValueError('Fresh and reference groups overlap')
    fresh_targets=labels(paths['fresh_labels'],fresh['samples']);old_targets=labels(paths['reference_labels'],reference['samples'])
    predictions=normalize(read(fresh_prediction),fresh['samples']);old_predictions=normalize(read(reference_prediction),reference['samples'])
    clips={c['clip_id']:c for c in reference_spec['clips']}
    variants={s['sample_id']:clips[s['clip_id']]['variant'] for s in splits['val']}
    # Restricted truth mapping and sample subset prevent TRAIN/TEST calibration reads.
    val_targets={s['sample_id']:old_targets[s['sample_id']] for s in splits['val']}
    primary=recall_thresholds(old_predictions,splits['val'],val_targets,variants)
    secondary=compatibility_thresholds(read(args.threshold_source),predictions)
    fresh_maps=load_maps(paths['fresh_maps'],[s['sample_id'] for s in fresh['samples']],predictions,'test_sample_ids')
    old_maps=load_maps(paths['reference_maps'],[s['sample_id'] for s in reference['samples'] if s['split'] in ('val','test')],old_predictions,'sample_ids')
    support=dict(np.load(paths['fresh_support'],allow_pickle=False));old_support=dict(np.load(paths['reference_support'],allow_pickle=False))
    objects=dict(np.load(paths['fresh_objects'],allow_pickle=False));old_objects=dict(np.load(paths['reference_objects'],allow_pickle=False))
    mask_selection=select_mask_thresholds(old_maps,splits['val'],val_targets,old_support)
    common=dict(schema='nf-g12-representation-evaluation-v1',
        threshold_policy='PRIMARY: same G10 VAL recall-first rule per arm/seed/head: >=95% overall positive recall, HEAD also >=95% bar_only positive recall; among feasible [0,1] thresholds minimize FP then maximize threshold. Full known VAL near truth/prediction coverage required. SECONDARY: unchanged G8 ordinary per-seed/ensemble compatibility thresholds.',
        mask_policy='Fixed .5 localization retained. Additional per-arm/seed/head threshold maximizes reference VAL positive mean IoU on .05,.10,...,.95, smallest first tie; no TEST tuning.',
        mask_selection={'/'.join(k):v for k,v in mask_selection.items()},calibration_sample_ids=[s['sample_id'] for s in splits['val']],
        query_scope='Visible near-object query masks are not full-object segmentation; evidence mass fraction and GT coverage have different denominators.',
        UNKNOWN_policy='Closing remains UNKNOWN. Fresh unknown truth and prediction gaps stay explicit; no UNKNOWN-as-negative conversion. VAL near unknowns violate the full-coverage calibration contract.',
        input_sha256=hashes,evaluator_sha256=sha(Path(__file__)),helper_sha256={name:sha(Path(__file__).with_name(name)) for name in (
            'detached_evaluate.py','diversity_evaluate.py','factorial_evaluate.py','grounding_evaluate.py','evaluate_whisker.py')},
        backend='CPU: TASK_NOT_GPU_SUITABLE')
    def result_for(predictions,scopes,targets,maps,support,objects,regression):
        return dict(common,primary=dict(headline=not regression,evaluations=score_scopes(predictions,primary,scopes,targets)),
            secondary=dict(headline=False,evaluations=score_scopes(predictions,secondary,scopes,targets)),
            localization=score_localization(maps,scopes,targets,support,objects,mask_selection),
            scored_scopes={name:dict(samples=[s['sample_id'] for s in samples],groups=sorted(groups)) for name,(samples,groups) in scopes.items()},
            claim_scope='Consumed G10 TEST regression, not fresh confirmation; original results preserved.' if regression else
                'Fresh controlled Development on one Willow background. Representation comparison under matched training protocol does not establish natural generalization or whole-object segmentation.')
    result=result_for(predictions,scopes,fresh_targets,fresh_maps,support,objects,False)
    regression=result_for(old_predictions,old_scopes,old_targets,old_maps,old_support,old_objects,True)
    if args.reference_baseline_predictions:
        baseline=normalize_predictions(read(args.reference_baseline_predictions),reference['samples'])
        regression['baseline_reproduction']=baseline_reproduction(old_predictions,baseline,primary,old_scopes['all_TEST'][0])
    for key,path in paths.items():
        if sha(path)!=hashes[key]:
            raise ValueError('Input changed during evaluation: '+key)
    args.output.mkdir(parents=True)
    for name,value in (('result.json',result),('regression-result.json',regression)):
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),fresh_test=len(fresh['samples']),reference_val=len(splits['val']),reference_test=len(splits['test']))))

if __name__=='__main__':
    main()
