"""G9-A frozen-threshold compound-object counterfactual evaluation.

CPU: TASK_NOT_GPU_SUITABLE. No fitting, threshold selection or model selection.
Fresh compound scenes are out of distribution: effects do not establish a
shortcut's cause. Predicted support mass is descriptive, not causal attribution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_whisker import HEADS, confusion, normalize_predictions

VARIANTS = ('both','bar_only','box_only','neither')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ratio(a,b):
    return a/b if b else None

def mean(values):
    values = [x for x in values if x is not None]
    return dict(evaluable=len(values),mean=float(np.mean(values)) if values else None)

def frozen_thresholds(source, key):
    # Exact identity only. No fallback to another seed/arm or fresh-data threshold.
    threshold = source['evaluations'][key]['thresholds']
    if len(threshold)!=4 or any(not np.isfinite(x['value']) or x['value']<0 for x in threshold):
        raise ValueError('Invalid frozen four-head thresholds')
    return threshold

def group_metadata(dataset,spec):
    clips={c['clip_id']:c for c in spec['clips']}
    cases=spec.get('cases',[])
    groups={}; samples=[]; excluded=[]
    seen=set()
    for sample in dataset['samples']:
        sid=sample['sample_id']; clip=clips[sample['clip_id']]
        if sid in seen or sample['split']!='test' or sample['group_id']!=clip['group_id']:
            raise ValueError('Duplicate sample or split/group mismatch')
        seen.add(sid)
        if sample.get('preflight') or clip.get('preflight') or clip.get('stage')=='preflight':
            excluded.append(sid); continue
        role=clip['variant']
        if role not in VARIANTS:
            raise ValueError('Unknown counterfactual variant')
        index=sample['frame_indices'][-1]
        case=cases[index] if isinstance(cases,list) and cases else cases.get(str(index),cases.get(index,{})) if isinstance(cases,dict) else {}
        geometry={k:case.get(k,clip.get(k)) for k in ('camera','wearer')}
        objects=case.get('objects',clip.get('objects'))
        if not isinstance(objects,list) or any('name' not in o for o in objects):
            raise ValueError('Object metadata missing')
        geometry['objects']={o['name']:o for o in objects}
        member=dict(sample_id=sid,geometry=geometry)
        group=groups.setdefault(sample['group_id'],dict(members={}))
        if role in group['members']:
            raise ValueError('Multiple outcomes per group/variant')
        group['members'][role]=member
        samples.append(sample)
    if not samples:
        raise ValueError('No main samples to score')
    for group in groups.values():
        members=group['members']
        if set(members)!=set(VARIANTS):
            raise ValueError('Each main group requires all four variants')
        base=members['both']['geometry']
        matches=all(base[k] is not None and all(members[v]['geometry'][k]==base[k] for v in VARIANTS) for k in ('camera','wearer'))
        wanted={'both':{'bar','box'},'bar_only':{'bar'},'box_only':{'box'},'neither':set()}
        object_match=all(set(members[v]['geometry']['objects'])==wanted[v] and all(
            obj==base['objects'].get(name) for name,obj in members[v]['geometry']['objects'].items()) for v in VARIANTS)
        group['counterfactual_geometry_verified']=matches and object_match
    return samples,groups,excluded

def paired_metrics(rows,targets,thresholds,groups):
    outcomes={}
    for gid,group in groups.items():
        variants={}
        for variant,member in group['members'].items():
            sid=member['sample_id']; score=rows[sid]; truth=targets[sid]
            alerts=[bool(score[h]>=thresholds[h]['value']) if score[h]>=0 else None for h in range(4)]
            variants[variant]=dict(sample_id=sid,truth=list(truth),scores=[float(x) for x in score],alerts=alerts)
        outcomes[gid]=dict(geometry_verified=group['counterfactual_geometry_verified'],variants=variants)
    results={}
    for name,removed,target_head,retained_head in (('bar_removal','box_only',1,0),('box_removal','bar_only',0,1)):
        records=[]; unknown_truth=0; other_truth=0; geometry_unknown=0
        for gid,outcome in outcomes.items():
            if not outcome['geometry_verified']:
                geometry_unknown+=1;continue
            before=outcome['variants']['both']; after=outcome['variants'][removed]
            truth=[before['truth'][target_head],after['truth'][target_head],before['truth'][retained_head],after['truth'][retained_head]]
            if -1 in truth:
                unknown_truth+=1;continue
            if truth!=[1,0,1,1]:
                other_truth+=1;continue
            before_known=all(before['alerts'][h] is not None for h in (0,1))
            both_correct=before_known and all(before['alerts'][h] for h in (0,1))
            target_flip=before['alerts'][target_head] is True and after['alerts'][target_head] is False
            retained_correct=before['alerts'][retained_head] is True and after['alerts'][retained_head] is True
            deltas={HEADS[h]:after['scores'][h]-before['scores'][h]
                    if min(after['scores'][h],before['scores'][h])>=0 else None for h in (0,1)}
            records.append(dict(group_id=gid,both_correct=both_correct,both_prediction_UNKNOWN=not before_known,
                after_prediction_UNKNOWN=any(after['alerts'][h] is None for h in (0,1)),
                target_flip_correct=target_flip,retained_head_correct=retained_correct,
                joint_success=target_flip and retained_correct,probability_delta_after_minus_both=deltas))
        n=len(records); base=sum(r['both_correct'] for r in records)
        success=sum(r['joint_success'] for r in records)
        results[name]=dict(truth_eligible_groups=n,truth_UNKNOWN_groups=unknown_truth,other_truth_groups=other_truth,
            geometry_NOT_EVALUABLE=geometry_unknown,both_initially_correct=base,
            both_initially_incorrect=sum(not r['both_correct'] and not r['both_prediction_UNKNOWN'] for r in records),
            both_prediction_UNKNOWN=sum(r['both_prediction_UNKNOWN'] for r in records),
            after_prediction_UNKNOWN=sum(r['after_prediction_UNKNOWN'] for r in records),
            target_flip_correct=sum(r['target_flip_correct'] for r in records),
            retained_head_correct=sum(r['retained_head_correct'] for r in records),joint_success=success,
            unconditional_joint_success_rate=ratio(success,n),
            conditional_joint_success_given_both_correct=ratio(success,base),
            probability_deltas={h:mean([r['probability_delta_after_minus_both'][h] for r in records]) for h in HEADS[:2]},groups=records)
    all_four={}
    for heads,name in (((0,),'BODY'),((1,),'HEAD'),((0,1),'BODY_HEAD_joint')):
        eligible=correct=unknown_truth=unknown_prediction=geometry_unknown=0
        for outcome in outcomes.values():
            if not outcome['geometry_verified']:
                geometry_unknown+=1;continue
            values=[(v['truth'][h],v['alerts'][h]) for v in outcome['variants'].values() for h in heads]
            if any(t==-1 for t,a in values):
                unknown_truth+=1;continue
            eligible+=1
            if any(a is None for t,a in values):
                unknown_prediction+=1;continue
            correct+=int(all(a==bool(t) for t,a in values))
        all_four[name]=dict(truth_evaluable_groups=eligible,correct=correct,truth_UNKNOWN_groups=unknown_truth,
            prediction_UNKNOWN_groups=unknown_prediction,geometry_NOT_EVALUABLE=geometry_unknown,
            all_four_correct_rate=ratio(correct,eligible))
    return dict(removal=results,all_four_correct=all_four,group_outcomes=outcomes)

def support_metrics(samples,targets,truth_support,object_support,maps):
    result={}
    for h,part in enumerate(('BODY','HEAD')):
        observations=[]
        for sample in samples:
            sid=sample['sample_id']; truth=targets[sid][h]
            if maps is None or sid not in maps:
                continue
            prediction=np.asarray(maps[sid][h],float)
            query=np.asarray(truth_support[sid][h],bool)
            objects={name:np.asarray(object_support[sid+'__'+name],bool) for name in ('bar','box')}
            if prediction.shape!=(18,32) or query.shape!=prediction.shape or any(m.shape!=query.shape for m in objects.values()):
                raise ValueError('Support shape mismatch')
            if not np.isfinite(prediction).all() or ((prediction<0)|(prediction>1)).any():
                raise ValueError('Invalid support probabilities')
            binary=prediction>=.5; mass=float(prediction.sum())
            relevant=objects['box' if part=='BODY' else 'bar']
            distractor=objects['bar' if part=='BODY' else 'box'] & ~relevant
            union=objects['bar']|objects['box']
            evaluable=truth==1 and bool(query.any())
            observations.append(dict(sample_id=sid,truth=truth,total_predicted_mass=mass,
                mass_in_GT_query=float(prediction[query].sum()),mass_off_GT_query=float(prediction[~query].sum()),
                evidence_fraction_in_GT_query=ratio(float(prediction[query].sum()),mass) if truth!=-1 else None,
                evidence_fraction_off_GT_query=ratio(float(prediction[~query].sum()),mass) if truth!=-1 else None,
                evidence_fraction_on_relevant_object=ratio(float(prediction[relevant].sum()),mass),
                evidence_fraction_on_other_object_only=ratio(float(prediction[distractor].sum()),mass),
                evidence_fraction_background=ratio(float(prediction[~union].sum()),mass),
                GT_query_coverage_at_0_5=ratio(int((binary&query).sum()),int(query.sum())) if evaluable else None,
                IoU_at_0_5=ratio(int((binary&query).sum()),int((binary|query).sum())) if evaluable else None,
                pointing_hit=bool(query.flat[int(prediction.argmax())]) if evaluable and mass>0 else None,
                visible_positive_support_evaluable=evaluable))
        positives=sum(targets[s['sample_id']][h]==1 for s in samples)
        positive_rows=[r for r in observations if r['visible_positive_support_evaluable']]
        result[part]=dict(samples=len(samples),maps_available=len(observations),positive_samples=positives,
            positive_visible_support_evaluable=len(positive_rows),positive_support_NOT_EVALUABLE=positives-len(positive_rows),
            zero_mass_samples=sum(r['total_predicted_mass']==0 for r in observations),
            positive_summary={key:mean([r[key] for r in positive_rows]) for key in (
                'evidence_fraction_in_GT_query','evidence_fraction_off_GT_query','evidence_fraction_on_relevant_object',
                'evidence_fraction_on_other_object_only','evidence_fraction_background','GT_query_coverage_at_0_5','IoU_at_0_5','pointing_hit')},
            samples_detail=observations)
    return result

def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('capture','learned','threshold-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    prediction_path=args.learned/'predictions.json' if args.learned.is_dir() else args.learned
    paths=dict(dataset=args.capture/'model/dataset.json',labels=args.capture/'evaluator/labels.json',
        spec=args.capture/'evaluator/spec.json',support=args.capture/'evaluator/support.npz',
        object_support=args.capture/'evaluator/object_support.npz',predictions=prediction_path,
        threshold_source=args.threshold_source)
    hashes={k:sha(p) for k,p in paths.items()}
    dataset=read(paths['dataset']); samples,groups,excluded=group_metadata(dataset,read(paths['spec']))
    truth=read(paths['labels']); targets=truth.get('targets',truth)
    if set(targets)!={s['sample_id'] for s in dataset['samples']} or any(len(v)!=4 or any(x not in (-1,0,1) for x in v) for v in targets.values()):
        raise ValueError('Truth coverage/value mismatch')
    predictions=normalize_predictions(read(prediction_path),dataset['samples'])
    frozen=read(args.threshold_source)
    selected={key:frozen_thresholds(frozen,'/'.join(key)) for key in predictions}
    support=dict(np.load(paths['support'],allow_pickle=False))
    object_support=dict(np.load(paths['object_support'],allow_pickle=False))
    map_path=prediction_path.parent/'support_predictions.npz'
    maps_cache=dict(np.load(map_path,allow_pickle=False)) if map_path.exists() else {}
    if maps_cache:
        paths['predicted_support']=map_path;hashes['predicted_support']=sha(map_path)
    map_ids=[str(x) for x in maps_cache.get('test_sample_ids',[])]
    if maps_cache and (len(map_ids)!=len(set(map_ids)) or set(map_ids)!={s['sample_id'] for s in dataset['samples']}):
        raise ValueError('Prediction support coverage mismatch')
    evaluations={}
    for key,rows in predictions.items():
        arm,seed,mode=key
        map_key=f'{arm}__'+('ensemble' if seed=='ensemble' else 'seed'+seed)+f'__{mode}'
        array=maps_cache.get(map_key)
        if array is not None and array.shape!=(len(map_ids),2,18,32):
            raise ValueError('Prediction support shape mismatch')
        maps={sid:array[i] for i,sid in enumerate(map_ids)} if array is not None else None
        evaluations['/'.join(key)]=dict(thresholds=selected[key],cells=confusion(rows,samples,targets,selected[key]),
            by_variant={v:confusion(rows,[s for s in samples if any(m['sample_id']==s['sample_id'] for g in groups.values() for role,m in g['members'].items() if role==v)],targets,selected[key]) for v in VARIANTS},
            counterfactual=paired_metrics(rows,targets,selected[key],groups),
            localization=support_metrics(samples,targets,support,object_support,maps))
    for key,path in paths.items():
        if sha(path)!=hashes[key]:
            raise ValueError('Input changed during evaluation: '+key)
    result=dict(schema='nf-g9a-grounding-evaluation-v1',evaluations=evaluations,main_samples=len(samples),main_groups=len(groups),
        excluded_preflight_samples=excluded,input_sha256=hashes,evaluator_sha256=sha(Path(__file__)),
        threshold_policy='Exact G8 arm/seed/mode threshold reuse; no fitting or threshold selection on G9.',
        claim_scope='Fresh compound-scene OOD controlled Development counterfactuals; cannot identify a shortcut cause or establish natural-world grounding.',
        attribution_scope='Descriptive predicted support mass, not causal attribution. GT query coverage measures recovered GT pixels at threshold .5; evidence fraction measures the share of total predicted probability mass inside the GT query. These denominators are different.',
        UNKNOWN_policy='Unknown truth excluded from confusion and paired eligible denominators. Unknown predictions count as unavailable, never negative; all-opportunity recall and conditional pairing denominators explicit. Zero support mass has undefined evidence fraction and pointing.',
        backend='CPU: TASK_NOT_GPU_SUITABLE')
    args.output.mkdir(parents=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),samples=len(samples),groups=len(groups),evaluations=len(evaluations))))

if __name__=='__main__':
    main()
