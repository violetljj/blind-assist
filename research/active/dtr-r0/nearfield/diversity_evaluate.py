"""G10 frozen-primary 2x2 diversity/head evaluation; descriptive Development only."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_whisker import HEADS,normalize_predictions
from factorial_evaluate import partitions,primary_thresholds,secondary_thresholds,score_policy
from grounding_evaluate import read,sha,support_metrics

ARMS=('existing_plain','existing_region','expanded_plain','expanded_region')
SEEDS=('17','29','43','ensemble')


def strata_for_samples(samples,spec):
    clips={clip['clip_id']:clip for clip in spec['clips']};group_strata={}
    for sample in samples:
        value=clips[sample['clip_id']]['stratum']
        if value not in ('narrow','diverse'):
            raise ValueError('Invalid stratum')
        gid=sample['group_id']
        if gid in group_strata and group_strata[gid]!=value:
            raise ValueError('A quartet crosses strata')
        group_strata[gid]=value
    return group_strata


def evaluation_scopes(dataset,spec,with_strata=True):
    splits,test,groups=partitions(dataset,spec)
    scopes={'all_TEST':(test,groups)}
    if with_strata:
        strata=strata_for_samples(dataset['samples'],spec)
        for split in ('val','test'):
            if {strata[s['group_id']] for s in splits[split]}!={'narrow','diverse'}:
                raise ValueError('Both common VAL and TEST require narrow and diverse strata')
        for value in ('narrow','diverse'):
            subset=[s for s in test if strata[s['group_id']]==value]
            ids={s['group_id'] for s in subset}
            scopes[value]=(subset,{gid:groups[gid] for gid in ids})
    return splits,scopes


def normalize_for_scope(payload,samples,test_only=None):
    # Regression can cache only TEST; reject all other partial sample coverage.
    expected=samples
    if test_only is not None and set(payload['sample_ids'])=={s['sample_id'] for s in test_only}:
        expected=test_only
    predictions=normalize_predictions(payload,expected)
    if set(predictions)!={(arm,seed,'normal') for arm in ARMS for seed in SEEDS}:
        raise ValueError('Expected four 2x2 arms, three seeds plus ensembles, normal only')
    if any(any(float(score[h])!=-1 for h in (2,3)) for rows in predictions.values() for score in rows.values()):
        raise ValueError('Closing predictions must remain UNKNOWN in this current-frame probe')
    return predictions


def score_scopes(predictions,selected,scopes,targets):
    result={'/'.join(key):{} for key in predictions}
    for name,(samples,groups) in scopes.items():
        values=score_policy(predictions,selected,samples,targets,groups)
        for key,ev in values.items():
            result[key][name]=ev
    return result


def subtraction(value,reference):
    return value-reference if value is not None and reference is not None else None


def contrast_metrics(ev):
    joint=ev['counterfactual']['all_four_correct']['BODY_HEAD_joint']
    result=dict(joint_correct_count=joint['correct'],joint_evaluable_groups=joint['truth_evaluable_groups'],
                joint_correct_rate=joint['all_four_correct_rate'])
    for head in HEADS[:2]:
        cell=ev['cells'][head]
        for metric in ('TP','FP','FN','TN','UNKNOWN','all_opportunity_recall','actual_FPR'):
            result[head+'_'+metric]=cell[metric]
    return result


def factorial_contrasts(evaluations):
    """Paired descriptive differences on identical TEST groups, not significance."""
    result={}
    names=next(iter(evaluations.values())).keys()
    contrasts={'data_at_plain':('expanded_plain','existing_plain'),
               'data_at_region':('expanded_region','existing_region'),
               'head_at_existing':('existing_region','existing_plain'),
               'head_at_expanded':('expanded_region','expanded_plain')}
    for seed in SEEDS:
        result[seed]={}
        for scope in names:
            values={arm:contrast_metrics(evaluations[f'{arm}/{seed}/normal'][scope]) for arm in ARMS}
            if len({v['joint_evaluable_groups'] for v in values.values()})!=1:
                raise ValueError('2x2 arms have unequal paired truth denominators')
            pairs={name:dict(value_arm=arms[0],reference_arm=arms[1],
                differences={metric:subtraction(values[arms[0]][metric],values[arms[1]][metric])
                             for metric in values[arms[0]]}) for name,arms in contrasts.items()}
            interaction={}
            for metric in values['existing_plain']:
                expanded=subtraction(values['expanded_region'][metric],values['expanded_plain'][metric])
                existing=subtraction(values['existing_region'][metric],values['existing_plain'][metric])
                interaction[metric]=subtraction(expanded,existing)
            result[seed][scope]=dict(arm_metrics=values,paired_contrasts=pairs,
                interaction=dict(formula='expanded_region - expanded_plain - existing_region + existing_plain',differences=interaction))
    return dict(scope='Descriptive paired counts and rates only; no significance claim or model selection. Counts share exact truth denominators; higher TP/joint correctness and lower FP favor an arm.',by_seed=result)


def score_capture(capture,learned,threshold_source,regression=False):
    prediction_path=learned/'predictions.json' if learned.is_dir() else learned
    paths=dict(dataset=capture/'model/dataset.json',spec=capture/'evaluator/spec.json',
        labels=capture/'evaluator/labels.json',support=capture/'evaluator/support.npz',
        object_support=capture/'evaluator/object_support.npz',predictions=prediction_path,
        threshold_source=threshold_source)
    hashes={key:sha(path) for key,path in paths.items()}
    dataset=read(paths['dataset']);spec=read(paths['spec'])
    splits,scopes=evaluation_scopes(dataset,spec,with_strata=not regression)
    test=scopes['all_TEST'][0]
    payload=read(paths['labels']);targets=payload.get('targets',payload)
    if set(targets)!={s['sample_id'] for s in dataset['samples']} or any(
        len(v)!=4 or any(x not in (-1,0,1) for x in v) or v[2:]!=[-1,-1] for v in targets.values()):
        raise ValueError('Near label coverage mismatch or closing truth not UNKNOWN')
    predictions=normalize_for_scope(read(prediction_path),dataset['samples'],test_only=test if regression else None)
    primary=primary_thresholds(read(threshold_source),predictions)
    primary_evaluations=score_scopes(predictions,primary,scopes,targets)
    result=dict(schema='nf-g10-diversity-evaluation-v1',
        primary=dict(headline=not regression,
            threshold_rule='Every arm preserves G8 ordinary_video corresponding seed/ensemble normal thresholds unchanged.',
            threshold_source_keys={'/'.join(k):f'ordinary_video/{k[1]}/normal' for k in predictions},
            evaluations=primary_evaluations,contrasts=factorial_contrasts(primary_evaluations)))
    if not regression:
        secondary=secondary_thresholds(predictions,dataset['samples'],targets)
        secondary_evaluations=score_scopes(predictions,secondary,scopes,targets)
        result['secondary']=dict(headline=False,
            threshold_rule='Common new VAL quartets across BOTH strata; per-arm/seed per-head smallest threshold meeting <=5% known-negative known-prediction FPR. Actual TEST FPR is reported, never matched. Secondary cannot replace the primary headline.',
            validation_samples=[s['sample_id'] for s in splits['val']],evaluations=secondary_evaluations,
            contrasts=factorial_contrasts(secondary_evaluations))
    truth_support=dict(np.load(paths['support'],allow_pickle=False))
    object_support=dict(np.load(paths['object_support'],allow_pickle=False))
    map_path=prediction_path.parent/'support_predictions.npz'
    maps_cache=dict(np.load(map_path,allow_pickle=False)) if map_path.exists() else {}
    if maps_cache:
        paths['predicted_support']=map_path;hashes['predicted_support']=sha(map_path)
    map_ids=[str(x) for x in maps_cache.get('test_sample_ids',[])]
    if maps_cache and (len(map_ids)!=len(set(map_ids)) or set(map_ids)!={s['sample_id'] for s in test}):
        raise ValueError('Support predictions must cover exactly TEST samples')
    localization={}
    for key in predictions:
        arm,seed,mode=key
        map_key=f'{arm}__'+('ensemble' if seed=='ensemble' else 'seed'+seed)+f'__{mode}'
        value=maps_cache.get(map_key)
        if value is not None and value.shape!=(len(map_ids),2,18,32):
            raise ValueError('Support prediction shape mismatch')
        maps={sid:value[i] for i,sid in enumerate(map_ids)} if value is not None else None
        localization['/'.join(key)]={name:support_metrics(samples,targets,truth_support,object_support,maps)
                                     for name,(samples,groups) in scopes.items()}
    for key,path in paths.items():
        if sha(path)!=hashes[key]:
            raise ValueError('Input changed during evaluation: '+key)
    result.update(localization=localization,
        scored_scopes={name:dict(samples=[s['sample_id'] for s in samples],groups=sorted(groups)) for name,(samples,groups) in scopes.items()},
        partition_counts={split:dict(samples=len(samples),groups=len({s['group_id'] for s in samples})) for split,samples in splits.items()},
        input_sha256=hashes,evaluator_sha256=sha(Path(__file__)),helper_sha256={name:sha(Path(__file__).with_name(name)) for name in (
            'factorial_evaluate.py','grounding_evaluate.py','evaluate_whisker.py')},
        claim_scope='Consumed old G9B TEST regression only, not fresh confirmation; no recalibration or original-result overwrite.' if regression else
            'Descriptive controlled Development on one Willow background. Data expansion changes image diversity and count; the 2x2 compares fixed-budget implementations, not isolated causal mechanisms or natural generalization.',
        UNKNOWN_policy='Near UNKNOWN truth and prediction coverage, all-opportunity recall and paired eligible denominators are explicit. Closing remains UNKNOWN. Support evidence fractions, query coverage and pointing retain separate denominators.',
        backend='CPU: TASK_NOT_GPU_SUITABLE')
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('capture','learned','threshold-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('regression-capture','regression-learned'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    if bool(args.regression_capture)!=bool(args.regression_learned):
        parser.error('Regression capture and learned paths must be supplied together')
    result=score_capture(args.capture,args.learned,args.threshold_source)
    regression=score_capture(args.regression_capture,args.regression_learned,args.threshold_source,True) if args.regression_capture else None
    args.output.mkdir(parents=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if regression is not None:
        (args.output/'regression-result.json').write_text(json.dumps(regression,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# G10 diversity and head comparison','',result['claim_scope'],'',result['primary']['threshold_rule'],'',
        '| Primary arm / seed | Stratum | BODY TP / FP / FN | HEAD TP / FP / FN | All-four joint correct |',
        '| --- | --- | --- | --- | --- |']
    for key,scopes in result['primary']['evaluations'].items():
        for name,ev in scopes.items():
            body,head=ev['cells']['BODYnear'],ev['cells']['HEADnear'];joint=ev['counterfactual']['all_four_correct']['BODY_HEAD_joint']
            lines.append(f"| {key} | {name} | {body['TP']} / {body['FP']} / {body['FN']} | {head['TP']} / {head['FP']} / {head['FN']} | {joint['correct']} / {joint['truth_evaluable_groups']} |")
    lines.extend(['',result['secondary']['threshold_rule'],'','Primary/secondary paired contrasts, interaction, selective removals, UNKNOWN and support attribution are separate in result.json.'])
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),scopes={k:len(v['groups']) for k,v in result['scored_scopes'].items()})))

if __name__=='__main__':
    main()
