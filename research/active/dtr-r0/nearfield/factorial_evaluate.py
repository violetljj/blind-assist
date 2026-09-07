"""Frozen-primary and explicitly secondary VAL-calibrated factorial data probe.

CPU: TASK_NOT_GPU_SUITABLE. The primary headline always preserves the G8 ordinary
model's operating thresholds. This evaluator never fits or selects a model.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from evaluate_whisker import HEADS, confusion, normalize_predictions, thresholds, validate_samples
from grounding_evaluate import (VARIANTS, frozen_thresholds, group_metadata,
    paired_metrics, read, sha, support_metrics)

ARMS = ('single_data','factorial_data','frozen_ordinary')
SEEDS = ('17','29','43','ensemble')

def partitions(dataset,spec):
    """Validate disjoint complete quartets; only TEST reaches paired scoring."""
    samples=dataset['samples']; validate_samples(samples)
    clips={c['clip_id']:c for c in spec['clips']}
    groups={}
    for sample in samples:
        clip=clips[sample['clip_id']]
        if clip['group_id']!=sample['group_id'] or clip.get('split',sample['split'])!=sample['split']:
            raise ValueError('Clip/sample group or split mismatch')
        role=clip['variant']
        group=groups.setdefault(sample['group_id'],set())
        if role not in VARIANTS or role in group:
            raise ValueError('Duplicate or invalid quartet variant')
        group.add(role)
    if any(values!=set(VARIANTS) for values in groups.values()):
        raise ValueError('Every group must contain the complete common quartet')
    split_samples={split:[s for s in samples if s['split']==split] for split in ('train','val','test')}
    test,metadata,excluded=group_metadata(dict(dataset,samples=split_samples['test']),spec)
    if excluded:
        raise ValueError('Preflight samples must not be mixed into this main factorial dataset')
    return split_samples,test,metadata

def primary_thresholds(source,keys):
    """Every arm uses the SAME G8 ordinary thresholds for its matching seed."""
    return {key:copy.deepcopy(frozen_thresholds(source,f'ordinary_video/{key[1]}/normal')) for key in keys}

def secondary_thresholds(predictions,samples,targets):
    val=[s for s in samples if s['split']=='val']
    val_targets={s['sample_id']:targets[s['sample_id']] for s in val}
    # Neither train/test scores nor train/test labels enter threshold selection.
    return {key:thresholds(rows,val,val_targets,max_fpr=.05) for key,rows in predictions.items()}

def score_policy(predictions,selected,test,targets,groups):
    by_variant={variant:[s for s in test if s['sample_id'] in {
        group['members'][variant]['sample_id'] for group in groups.values()}] for variant in VARIANTS}
    return {'/'.join(key):dict(thresholds=selected[key],cells=confusion(rows,test,targets,selected[key]),
        by_variant={variant:confusion(rows,subset,targets,selected[key]) for variant,subset in by_variant.items()},
        counterfactual=paired_metrics(rows,targets,selected[key],groups)) for key,rows in predictions.items()}

def reproduction_comparison(predictions,reference,selected,samples):
    comparisons={}
    for seed in SEEDS:
        key=('frozen_ordinary',seed,'normal');old_key=('ordinary_video',seed,'normal')
        fresh=np.asarray([predictions[key][s['sample_id']][:2] for s in samples],float)
        old=np.asarray([reference[old_key][s['sample_id']][:2] for s in samples],float)
        valid=(fresh>=0)&(old>=0)
        limit=np.asarray([t['value'] for t in selected[key][:2]])
        difference=np.abs(fresh-old)
        comparisons[seed]=dict(known_near_cells=int(valid.sum()),UNKNOWN_cells=int((~valid).sum()),
            max_abs_probability_difference=float(difference[valid].max()) if valid.any() else None,
            all_known_scores_within_tolerance=bool(np.allclose(fresh[valid],old[valid],atol=1e-4,rtol=1e-5)) if valid.any() else None,
            near_alert_mismatches=int((((fresh>=limit)!=(old>=limit))&valid).sum()))
    return dict(status='DESCRIPTIVE_REPRODUCTION_CHECK',atol=1e-4,rtol=1e-5,comparisons=comparisons,
        scope='Frozen ordinary near scores versus saved original G9 predictions; numerical and operating-threshold differences are reported without retuning.')

def regression_evaluation(capture,learned,threshold_source,reference_path=None):
    prediction_path=learned/'predictions.json' if learned.is_dir() else learned
    paths=dict(dataset=capture/'model/dataset.json',labels=capture/'evaluator/labels.json',
        spec=capture/'evaluator/spec.json',support=capture/'evaluator/support.npz',
        object_support=capture/'evaluator/object_support.npz',predictions=prediction_path,
        threshold_source=threshold_source)
    hashes={key:sha(path) for key,path in paths.items()}
    dataset=read(paths['dataset']);test,groups,excluded=group_metadata(dataset,read(paths['spec']))
    label_payload=read(paths['labels']);targets=label_payload.get('targets',label_payload)
    if set(targets)!={s['sample_id'] for s in dataset['samples']} or any(len(t)!=4 or any(v not in (-1,0,1) for v in t) for t in targets.values()):
        raise ValueError('Regression label coverage/value mismatch')
    predictions=normalize_predictions(read(prediction_path),dataset['samples'])
    if set(predictions)!={(arm,seed,'normal') for arm in ARMS for seed in SEEDS}:
        raise ValueError('Regression predictions require the same three arms and three seeds plus ensembles')
    selected=primary_thresholds(read(threshold_source),predictions)
    evaluations=score_policy(predictions,selected,test,targets,groups)
    truth_support=dict(np.load(paths['support'],allow_pickle=False))
    object_support=dict(np.load(paths['object_support'],allow_pickle=False))
    map_path=prediction_path.parent/'support_predictions.npz'
    maps_cache=dict(np.load(map_path,allow_pickle=False)) if map_path.exists() else {}
    if maps_cache:
        paths['predicted_support']=map_path;hashes['predicted_support']=sha(map_path)
    map_ids=[str(x) for x in maps_cache.get('test_sample_ids',[])]
    if maps_cache and (len(map_ids)!=len(set(map_ids)) or set(map_ids)!={s['sample_id'] for s in dataset['samples']}):
        raise ValueError('Regression predicted support coverage mismatch')
    localization={}
    for key in predictions:
        arm,seed,mode=key
        map_key=f'{arm}__'+('ensemble' if seed=='ensemble' else 'seed'+seed)+f'__{mode}'
        value=maps_cache.get(map_key)
        if value is not None and value.shape!=(len(map_ids),2,18,32):
            raise ValueError('Regression predicted support shape mismatch')
        maps={sid:value[i] for i,sid in enumerate(map_ids)} if value is not None else None
        localization['/'.join(key)]=support_metrics(test,targets,truth_support,object_support,maps)
    reproduction=dict(status='REFERENCE_PREDICTIONS_NOT_PROVIDED')
    if reference_path:
        paths['reference_predictions']=reference_path;hashes['reference_predictions']=sha(reference_path)
        reference=normalize_predictions(read(reference_path),dataset['samples'])
        reproduction=reproduction_comparison(predictions,reference,selected,test)
    for key,path in paths.items():
        if sha(path)!=hashes[key]:
            raise ValueError('Regression input changed during scoring: '+key)
    return dict(schema='nf-factorial-g9-regression-v1',primary=dict(headline=False,
        threshold_rule='G8 ordinary seed/ensemble thresholds unchanged; no secondary calibration on consumed G9 regression.',evaluations=evaluations),
        localization=localization,reproduction=reproduction,scored_samples=[s['sample_id'] for s in test],
        scored_groups=sorted(groups),excluded_preflight_samples=excluded,input_sha256=hashes,
        claim_scope='Consumed G9 Development regression, not fresh confirmation. Original G9 outputs are preserved.')

def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('capture','learned','threshold-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('regression-capture','regression-learned','regression-reference-predictions'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if bool(args.regression_capture)!=bool(args.regression_learned) or (args.regression_reference_predictions and not args.regression_capture):
        parser.error('Regression capture and learned paths must be supplied together; reference requires regression')
    if args.output.exists():
        parser.error('Use a fresh evaluation output; preserve prior results')
    prediction_path=args.learned/'predictions.json' if args.learned.is_dir() else args.learned
    paths=dict(dataset=args.capture/'model/dataset.json',labels=args.capture/'evaluator/labels.json',
        spec=args.capture/'evaluator/spec.json',support=args.capture/'evaluator/support.npz',
        object_support=args.capture/'evaluator/object_support.npz',predictions=prediction_path,
        threshold_source=args.threshold_source)
    hashes={key:sha(path) for key,path in paths.items()}
    dataset=read(paths['dataset']);spec=read(paths['spec'])
    split_samples,test,groups=partitions(dataset,spec)
    label_payload=read(paths['labels']);targets=label_payload.get('targets',label_payload)
    ids={s['sample_id'] for s in dataset['samples']}
    if set(targets)!=ids or any(len(t)!=4 or any(v not in (-1,0,1) for v in t) or t[2:]!=[-1,-1] for t in targets.values()):
        raise ValueError('Expected complete near labels and explicitly UNKNOWN closing truth')
    predictions=normalize_predictions(read(prediction_path),dataset['samples'])
    expected={(arm,seed,'normal') for arm in ARMS for seed in SEEDS}
    if set(predictions)!=expected:
        raise ValueError('Expected the three fixed arms, three seeds and seed-average ensembles, normal only')
    if any(any(float(score[h])!=-1 for h in (2,3)) for rows in predictions.values() for score in rows.values()):
        raise ValueError('This static probe must not supply closing predictions')
    primary=primary_thresholds(read(args.threshold_source),predictions)
    secondary=secondary_thresholds(predictions,dataset['samples'],targets)
    primary_evaluations=score_policy(predictions,primary,test,targets,groups)
    secondary_evaluations=score_policy(predictions,secondary,test,targets,groups)
    truth_support=dict(np.load(paths['support'],allow_pickle=False))
    object_support=dict(np.load(paths['object_support'],allow_pickle=False))
    map_path=prediction_path.parent/'support_predictions.npz'
    maps_cache=dict(np.load(map_path,allow_pickle=False)) if map_path.exists() else {}
    if maps_cache:
        paths['predicted_support']=map_path;hashes['predicted_support']=sha(map_path)
    map_ids=[str(x) for x in maps_cache.get('test_sample_ids',[])]
    if maps_cache and (len(map_ids)!=len(set(map_ids)) or set(map_ids)!={s['sample_id'] for s in test}):
        raise ValueError('Predicted maps must cover exactly TEST sample IDs')
    localization={}
    for key in predictions:
        arm,seed,mode=key
        map_key=f'{arm}__'+('ensemble' if seed=='ensemble' else 'seed'+seed)+f'__{mode}'
        value=maps_cache.get(map_key)
        if value is not None and value.shape!=(len(map_ids),2,18,32):
            raise ValueError('Predicted support map shape mismatch')
        maps={sid:value[i] for i,sid in enumerate(map_ids)} if value is not None else None
        localization['/'.join(key)]=support_metrics(test,targets,truth_support,object_support,maps)
    for key,path in paths.items():
        if sha(path)!=hashes[key]:
            raise ValueError('Input changed during scoring: '+key)
    helpers={name:sha(Path(__file__).with_name(name)) for name in ('evaluate_whisker.py','grounding_evaluate.py')}
    result=dict(schema='nf-factorial-data-evaluation-v1',
        primary=dict(headline=True,threshold_rule='All arms preserve G8 ordinary_video corresponding seed/ensemble normal thresholds; no new-data threshold tuning.',
            threshold_source_keys={'/'.join(k):f'ordinary_video/{k[1]}/normal' for k in predictions},evaluations=primary_evaluations),
        secondary=dict(headline=False,threshold_rule='Per-arm/seed per-head smallest threshold satisfying <=5% known-prediction known-negative FPR on the SAME new VAL quartets only. Actual TEST FPR is reported and never matched. This secondary operating-point result cannot replace the primary headline.',
            validation_samples=[s['sample_id'] for s in split_samples['val']],evaluations=secondary_evaluations),
        localization=localization,scored_test_samples=[s['sample_id'] for s in test],scored_test_groups=sorted(groups),
        partition_counts={split:dict(samples=len(values),groups=len({s['group_id'] for s in values})) for split,values in split_samples.items()},
        input_sha256=hashes,evaluator_sha256=sha(Path(__file__)),helper_sha256=helpers,
        claim_scope='Descriptive controlled Development on one shared Willow background. The data intervention changes joint-state coverage and unique images; an effect is not evidence for a pair-loss mechanism or natural generalization.',
        UNKNOWN_policy='Closing remains UNKNOWN and unscored. Near unknown truth/predictions and paired denominators are explicit; unknown predictions are not negative. Support coverage and evidence fraction use different denominators.',
        backend='CPU: TASK_NOT_GPU_SUITABLE')
    regression=regression_evaluation(args.regression_capture,args.regression_learned,args.threshold_source,
        args.regression_reference_predictions) if args.regression_capture else None
    args.output.mkdir(parents=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if regression is not None:
        (args.output/'regression-result.json').write_text(json.dumps(regression,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# Factorial data-only evaluation','',result['claim_scope'],'',result['primary']['threshold_rule'],'',
        '| Primary arm / seed | BODY TP / FP / FN | HEAD TP / FP / FN | Both-head all-four correct |',
        '| --- | --- | --- | --- |']
    for key,ev in primary_evaluations.items():
        body,head=ev['cells']['BODYnear'],ev['cells']['HEADnear']
        joint=ev['counterfactual']['all_four_correct']['BODY_HEAD_joint']
        lines.append(f"| {key} | {body['TP']} / {body['FP']} / {body['FN']} | {head['TP']} / {head['FP']} / {head['FN']} | {joint['correct']} / {joint['truth_evaluable_groups']} |")
    lines.extend(['',result['secondary']['threshold_rule'],'','Secondary counts, UNKNOWN, selective-removal conditional/unconditional outcomes, all group states and localization are retained separately in result.json.'])
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),test_samples=len(test),test_groups=len(groups),primary_evaluations=len(primary_evaluations))))

if __name__=='__main__':
    main()
