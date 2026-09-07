"""G11 detached-gate evaluator: frozen primary, reference-VAL calibration only."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_whisker import normalize_predictions
from factorial_evaluate import primary_thresholds,secondary_thresholds
from diversity_evaluate import evaluation_scopes,score_scopes,strata_for_samples
from grounding_evaluate import group_metadata,read,sha,support_metrics,mean

ARMS=('original_gate','detached_gate')
SEEDS=('17','29','43','ensemble')
MASK_GRID=tuple(i/20 for i in range(1,20))


def normalize(payload,samples):
    predictions=normalize_predictions(payload,samples)
    if set(predictions)!={(arm,seed,'normal') for arm in ARMS for seed in SEEDS}:
        raise ValueError('Expected both gate arms, three seeds and ensembles, normal only')
    if any(any(row[h]!=-1 for h in (2,3)) for rows in predictions.values() for row in rows.values()):
        raise ValueError('Closing predictions must be UNKNOWN')
    return predictions


def fresh_scopes(dataset,spec):
    samples,groups,excluded=group_metadata(dataset,spec)
    if excluded:
        raise ValueError('Do not mix preflight samples into fresh main scoring')
    strata=strata_for_samples(samples,spec)
    if set(strata.values())!={'narrow','diverse'}:
        raise ValueError('Fresh TEST requires both narrow and diverse strata')
    scopes={'all_TEST':(samples,groups)}
    for name in ('narrow','diverse'):
        subset=[s for s in samples if strata[s['group_id']]==name]
        scopes[name]=(subset,{gid:g for gid,g in groups.items() if strata[gid]==name})
    return scopes


def load_maps(path,expected_ids,keys,id_key):
    cache=dict(np.load(path,allow_pickle=False))
    ids=[str(x) for x in cache.pop(id_key)]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected_ids):
        raise ValueError('Support map sample coverage mismatch: '+id_key)
    maps={}
    for key in keys:
        arm,seed,mode=key
        name=f'{arm}__'+('ensemble' if seed=='ensemble' else 'seed'+seed)+f'__{mode}'
        array=cache[name]
        if array.shape!=(len(ids),2,18,32) or not np.isfinite(array).all() or ((array<0)|(array>1)).any():
            raise ValueError('Invalid support prediction shape/range')
        maps[key]={sid:array[i] for i,sid in enumerate(ids)}
    return maps


def select_mask_thresholds(maps,samples,targets,support):
    """Only reference VAL known-positive query masks affect this secondary choice."""
    val=[s for s in samples if s['split']=='val'];selected={}
    for key,rows in maps.items():
        heads=[]
        for h in range(2):
            positives=[s['sample_id'] for s in val if targets[s['sample_id']][h]==1]
            ids=[sid for sid in positives if np.asarray(support[sid][h],bool).any()]
            values=[]
            for threshold in MASK_GRID:
                ious=[]
                for sid in ids:
                    truth=np.asarray(support[sid][h],bool);binary=np.asarray(rows[sid][h])>=threshold
                    ious.append(float((binary&truth).sum()/(binary|truth).sum()))
                values.append(dict(threshold=threshold,mean_IoU=float(np.mean(ious)) if ious else None))
            if ids:
                # Ascending grid and strict improvement preserve the first/smallest tie.
                best=max(values,key=lambda x:x['mean_IoU'])
                heads.append(dict(status='REFERENCE_VAL_POSITIVE_MEAN_IOU_SELECTED',value=best['threshold'],
                    mean_validation_IoU=best['mean_IoU'],positive_samples=len(positives),evaluable=len(ids),
                    positive_support_NOT_EVALUABLE=len(positives)-len(ids),grid=values))
            else:
                heads.append(dict(status='NOT_EVALUABLE_NO_VAL_POSITIVE_SUPPORT',value=None,
                    positive_samples=len(positives),evaluable=0,positive_support_NOT_EVALUABLE=len(positives),grid=values))
        selected[key]=heads
    return selected


def calibrated_mask_metrics(rows,samples,targets,support,selected):
    result={}
    for h,part in enumerate(('BODY','HEAD')):
        threshold=selected[h]['value'];positives=[];negatives=[]
        positive_total=sum(targets[s['sample_id']][h]==1 for s in samples)
        negative_total=sum(targets[s['sample_id']][h]==0 for s in samples)
        unknown=sum(targets[s['sample_id']][h]==-1 for s in samples)
        if threshold is not None:
            for sample in samples:
                sid=sample['sample_id'];label=targets[sid][h]
                if label==-1:
                    continue
                probability=np.asarray(rows[sid][h],float);binary=probability>=threshold
                truth=np.asarray(support[sid][h],bool)
                if label==1 and truth.any():
                    positives.append(dict(sample_id=sid,IoU=float((binary&truth).sum()/(binary|truth).sum()),
                        GT_query_coverage=float((binary&truth).sum()/truth.sum()),
                        pointing_hit=bool(truth.flat[int(probability.argmax())]) if probability.sum()>0 else None))
                elif label==0:
                    negatives.append(dict(sample_id=sid,activated=bool(binary.any()),
                        active_pixels=int(binary.sum()),activation_pixel_fraction=float(binary.mean())))
        activated=sum(row['activated'] for row in negatives)
        result[part]=dict(selected_threshold=selected[h],positive_samples=positive_total,
            positive_evaluable=len(positives),positive_NOT_EVALUABLE=positive_total-len(positives),
            positive_mean_IoU=mean([r['IoU'] for r in positives]),positive_query_coverage=mean([r['GT_query_coverage'] for r in positives]),
            positive_pointing=mean([r['pointing_hit'] for r in positives]),
            known_negative_samples=negative_total,negative_evaluable=len(negatives),negative_NOT_EVALUABLE=negative_total-len(negatives),
            negative_activated_samples=activated,negative_activation_sample_rate=activated/len(negatives) if negatives else None,
            negative_activation_pixel_fraction=mean([r['activation_pixel_fraction'] for r in negatives]),
            truth_UNKNOWN=unknown,positive_details=positives,negative_details=negatives)
    return result


def score_localization(maps,scopes,targets,support,objects,selected):
    return {'/'.join(key):{name:dict(fixed_0_5=support_metrics(samples,targets,support,objects,rows),
                reference_VAL_selected=calibrated_mask_metrics(rows,samples,targets,support,selected[key]))
            for name,(samples,groups) in scopes.items()} for key,rows in maps.items()}


def baseline_reproduction(predictions,baseline,primary,samples):
    result={}
    for seed in SEEDS:
        key=('original_gate',seed,'normal');old_key=('expanded_region',seed,'normal')
        fresh=np.asarray([predictions[key][s['sample_id']][:2] for s in samples],float)
        old=np.asarray([baseline[old_key][s['sample_id']][:2] for s in samples],float)
        valid=(fresh>=0)&(old>=0);limit=np.asarray([t['value'] for t in primary[key][:2]])
        result[seed]=dict(known_near_cells=int(valid.sum()),UNKNOWN_cells=int((~valid).sum()),
            max_abs_difference=float(np.abs(fresh-old)[valid].max()) if valid.any() else None,
            within_tolerance=bool(np.allclose(fresh[valid],old[valid],atol=1e-4,rtol=1e-5)) if valid.any() else None,
            primary_alert_flips=int((((fresh>=limit)!=(old>=limit))&valid).sum()))
    return dict(atol=1e-4,rtol=1e-5,comparisons=result,
        scope='Original gate versus consumed G10 expanded_region near scores on old TEST; no retuning.')


def labels(path,samples):
    payload=read(path);targets=payload.get('targets',payload)
    if set(targets)!={s['sample_id'] for s in samples} or any(len(t)!=4 or any(x not in (-1,0,1) for x in t) or t[2:]!=[-1,-1] for t in targets.values()):
        raise ValueError('Near truth coverage/value mismatch or closing truth not UNKNOWN')
    return targets


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('capture','learned','reference-capture','threshold-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference-baseline-predictions',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    fresh_prediction=args.learned/'predictions.json'
    reference_prediction=args.learned/'reference/predictions.json'
    paths={'fresh_predictions':fresh_prediction,'reference_predictions':reference_prediction,'threshold_source':args.threshold_source,
           'fresh_maps':args.learned/'support_predictions.npz','reference_maps':args.learned/'reference/support_predictions.npz'}
    for prefix,capture in (('fresh',args.capture),('reference',args.reference_capture)):
        for name,relative in (('dataset','model/dataset.json'),('spec','evaluator/spec.json'),('labels','evaluator/labels.json'),
                              ('support','evaluator/support.npz'),('objects','evaluator/object_support.npz')):
            paths[prefix+'_'+name]=capture/relative
    if args.reference_baseline_predictions:
        paths['reference_baseline_predictions']=args.reference_baseline_predictions
    hashes={key:sha(path) for key,path in paths.items()}
    fresh=read(paths['fresh_dataset']);reference=read(paths['reference_dataset'])
    scopes=fresh_scopes(fresh,read(paths['fresh_spec']))
    splits,old_scopes=evaluation_scopes(reference,read(paths['reference_spec']))
    if {s['group_id'] for s in fresh['samples']} & {s['group_id'] for s in reference['samples']}:
        raise ValueError('Fresh and reference groups overlap')
    fresh_targets=labels(paths['fresh_labels'],fresh['samples']);old_targets=labels(paths['reference_labels'],reference['samples'])
    predictions=normalize(read(fresh_prediction),fresh['samples']);old_predictions=normalize(read(reference_prediction),reference['samples'])
    primary=primary_thresholds(read(args.threshold_source),predictions)
    secondary=secondary_thresholds(old_predictions,reference['samples'],old_targets)
    fresh_maps=load_maps(paths['fresh_maps'],[s['sample_id'] for s in fresh['samples']],predictions,'test_sample_ids')
    old_maps=load_maps(paths['reference_maps'],[s['sample_id'] for s in reference['samples'] if s['split'] in ('val','test')],old_predictions,'sample_ids')
    support=dict(np.load(paths['fresh_support'],allow_pickle=False));old_support=dict(np.load(paths['reference_support'],allow_pickle=False))
    objects=dict(np.load(paths['fresh_objects'],allow_pickle=False));old_objects=dict(np.load(paths['reference_objects'],allow_pickle=False))
    mask_selection=select_mask_thresholds(old_maps,reference['samples'],old_targets,old_support)
    common=dict(schema='nf-g11-detached-evaluation-v1',
        threshold_policy='PRIMARY: exact G8 ordinary_video corresponding seed/ensemble thresholds for both arms. SECONDARY: scalar thresholds <=5% known-negative FPR selected ONLY on common G10 VAL, then applied unchanged.',
        mask_policy='Fixed .5 localization remains primary. Additional per-arm/seed/head mask threshold maximizes reference G10 VAL known-positive mean IoU over .05,.10,...,.95; smallest threshold wins exact ties. No TEST threshold selection.',
        mask_selection={'/'.join(k):v for k,v in mask_selection.items()},
        calibration_sample_ids=[s['sample_id'] for s in splits['val']],
        query_scope='Near-object query masks are not full-object segmentation. Predicted evidence fractions and GT coverage have different denominators; attribution is descriptive, not causal.',
        UNKNOWN_policy='Unknown truth/predictions remain explicit; zero mass has undefined pointing/evidence fraction. Missing positive support cannot tune mask thresholds. Negative-mask activation is not a false-object-alert count.',
        input_sha256=hashes,evaluator_sha256=sha(Path(__file__)),
        helper_sha256={name:sha(Path(__file__).with_name(name)) for name in ('diversity_evaluate.py','factorial_evaluate.py','grounding_evaluate.py','evaluate_whisker.py')},
        backend='CPU: TASK_NOT_GPU_SUITABLE')
    def result_for(predictions,scopes,targets,maps,support,objects,regression):
        return dict(common,primary=dict(headline=not regression,evaluations=score_scopes(predictions,primary,scopes,targets)),
            secondary=dict(headline=False,evaluations=score_scopes(predictions,secondary,scopes,targets)),
            localization=score_localization(maps,scopes,targets,support,objects,mask_selection),
            scored_scopes={name:dict(samples=[s['sample_id'] for s in samples],groups=sorted(groups)) for name,(samples,groups) in scopes.items()},
            claim_scope='Consumed G10 TEST regression, not fresh confirmation; original results preserved.' if regression else
                'Fresh controlled Development on one Willow background; detached-gate comparison does not establish natural generalization or whole-object segmentation.')
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
