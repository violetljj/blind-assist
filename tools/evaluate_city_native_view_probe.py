"""One RGB-only paired view probe; existing thresholds, no fitting or selection."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import adapt_city_native as adapt
import audit_city_native_scores as audit
import numpy as np
import torch

route = adapt.route
CONTROL_TRAIN_IDS = [6, 8, 11, 13]


def threshold_pair(control, treatment, threshold):
    if threshold is None:
        return dict(status='NOT_EVALUABLE', threshold=None)
    before, after = float(control) >= threshold, float(treatment) >= threshold
    return dict(threshold=threshold, control_alert=before, treatment_alert=after,
        alert_flip=before != after, direction='GAIN' if not before and after else 'LOSS' if before and not after else 'UNCHANGED')


def support_evidence(probability, mask):
    if probability.shape != mask.shape or not np.isin(mask, [-1, 0, 1]).all():
        raise ValueError('Support dimensions/UNKNOWN labels invalid')
    positive = mask == 1
    if not positive.any():
        return dict(status='UNKNOWN', reason='No reliable positive target support; excluded from target miss counts',
            overlap=None, peak_hit=None, joint_eligible=False)
    intersection = int(((probability >= .5) & positive).sum())
    union = int((((probability >= .5) | positive) & (mask >= 0)).sum())
    return dict(status='EVALUABLE', positive_cells=int(positive.sum()), overlap=intersection > 0,
        target_iou=intersection/union, peak_hit=int(mask.ravel()[int(probability.argmax())]) == 1,
        joint_eligible=True)


def target_scene_composition(scene, target):
    """Count pooled support without silently assigning UNKNOWN target cells."""
    if scene.shape != target.shape or not np.isin(scene,[-1,0,1]).all() or not np.isin(target,[-1,0,1]).all():
        raise ValueError('Invalid scene/target pooled labels')
    positive = scene == 1
    return dict(scene_positive_cells=int(positive.sum()), target_positive_cells=int((target==1).sum()),
        scene_positive_on_target_cells=int((positive & (target==1)).sum()),
        scene_positive_outside_recorded_target_cells=int((positive & (target!=1)).sum()),
        outside_target_known_zero_cells=int((positive & (target==0)).sum()),
        outside_target_unknown_cells=int((positive & (target==-1)).sum()),
        target_unknown_cells=int((target==-1).sum()), scene_unknown_cells=int((scene==-1).sum()))


def run(args):
    started = time.perf_counter()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(route.ARTIFACTS.resolve()) or out == route.ARTIFACTS.resolve():
        raise ValueError('Fresh canonical artifact output required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for new probe inference')
    ids, paths, capture_info = adapt.capture(args.capture)
    if ids != list(range(8)):
        raise ValueError('Expected exactly control0..3 and treatment4..7')
    native = route.NF / 'city-native-adapt-20260908'
    replay = route.NF / 'city-native-replay-20260908'
    train = native / 'train-v1'
    train_ids, train_paths, train_info = adapt.capture(train)
    spec = route.read(args.capture / 'source/spec.json')
    original_spec = route.read(train / 'source/spec.json')
    for i, train_id in enumerate(CONTROL_TRAIN_IDS):
        camera = spec['cases'][i]['camera']
        original_camera = original_spec['cases'][train_id]['camera']
        if any(abs(float(camera[k])-float(original_camera[k])) > 1e-6 for k in ('x','y','z','pitch','yaw','roll')):
            raise ValueError('Control camera does not reproduce declared original TRAIN pose')
    source = route.NF / 'decoupled-20260908/main/learned/decoupled_seed17.pt'
    configurations = dict(original=(native/'fit-v1', 'initial', source),
        native_only=(native/'fit-v1', 'adapted', native/'fit-v1/native_seed17_step300.pt'),
        replay_thin=(replay/'fit-v1', 'adapted', replay/'fit-v1/replay_thin_seed17_step300.pt'))
    history_root = args.score_audit
    history_receipt = route.read(history_root / 'receipt.json')
    if history_receipt.get('status') != 'PASS':
        raise ValueError('Completed original TRAIN score audit required')
    history_manifest = route.read(history_root / 'input-manifest.json')
    if route.sha(history_root / 'input-manifest.json') != history_receipt['input_manifest_sha256']:
        raise ValueError('Historical TRAIN audit input identity changed')
    for key in ('ids','dataset_sha256','rgb_sha256','spec_sha256'):
        if history_manifest['train_capture'][key] != train_info[key]:
            raise ValueError('Original TRAIN control source differs from cached audit')
    manifest = dict(schema='city-native-view-probe-inference-v1', optimizer_steps=0,
        intent='Paired control versus rotated view of the same declared original TRAIN bollard; consumed Development',
        selection=dict(control_indices=[0,1,2,3], treatment_indices=[4,5,6,7], original_train_ids=CONTROL_TRAIN_IDS),
        label_access='New labels are opened only after cached RGB predictions for all three models',
        thresholds='Previously locked DEV/historical only; no calibration',
        capture=capture_info, historical_train_capture=train_info,
        source_sha256={str(p):route.sha(p) for p in [Path(__file__),Path(adapt.__file__),Path(route.__file__),
            route.SOURCE/'decoupled_model.py',route.SOURCE/'representation_model.py']},
        models={}, input_sha256={str(history_root/'receipt.json'):route.sha(history_root/'receipt.json')})
    history, configs = {}, {}
    for name, (fit, arm, checkpoint) in configurations.items():
        protocol, receipt, result, lock = [route.read(fit/n) for n in ('protocol.json','receipt.json','result.json','locked-dev-choice.json')]
        if receipt.get('status') != 'PASS' or route.sha(fit/'result.json') != receipt['result_sha256']:
            raise ValueError('Fit result/receipt identity mismatch')
        if route.sha(fit/'locked-dev-choice.json') != result['dev_choice_sha256']:
            raise ValueError('DEV threshold lock changed')
        expected = protocol['initial_checkpoint_sha256'] if arm == 'initial' else receipt['fit']['checkpoint_sha256']
        if route.sha(checkpoint) != expected:
            raise ValueError('Frozen checkpoint changed')
        history_checkpoint = history_manifest['checkpoints'][name]
        if history_manifest['input_sha256'][history_checkpoint] != expected:
            raise ValueError('Historical TRAIN cache used different weights')
        for model_source in ('decoupled_model.py','representation_model.py'):
            if route.sha(route.SOURCE/model_source) != protocol['source_sha256'][model_source]:
                raise ValueError('Historical model source changed')
        history_path = history_root/f'{name}-train.npz'
        if route.sha(history_path) != history_receipt['prediction_sha256'][name]:
            raise ValueError('Historical TRAIN prediction cache changed')
        history[name] = audit.loaded(history_path)
        configs[name] = dict(historical=protocol['historical_thresholds'], dev_selected=lock['choices'][arm])
        manifest['models'][name] = dict(checkpoint=str(checkpoint), sha256=expected, thresholds=configs[name])
        for path in [fit/n for n in ('protocol.json','receipt.json','result.json','locked-dev-choice.json')] + [history_path]:
            manifest['input_sha256'][str(path)] = route.sha(path)
    out.mkdir(parents=True)
    route.write(out/'input-manifest.json',manifest)
    manifest_hash = route.sha(out/'input-manifest.json')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    predictions, timings = {}, {}
    try:
        x=adapt.rgb(paths)
        for name, (_, _, checkpoint) in configurations.items():
            tick=time.perf_counter()
            model=route.DecoupledModel(route.NF/'representation-20260908/pretrained').cuda()
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
            n,s=adapt.predict(model,x)
            torch.cuda.synchronize()
            predictions[name]=dict(near=n,support=s)
            np.savez_compressed(out/f'{name}-predictions.npz',near=n,support=s)
            timings[name]=time.perf_counter()-tick
            backend=asdict(adapt.torch_observation(model=model))
            del model
        # All new evaluator access starts after the three prediction caches exist.
        label_path=adapt.label_file(args.labels)
        near, support, info=adapt.native_labels(label_path,ids,capture_info)
        label_manifest=route.read(label_path)
        target=next((t for t in label_manifest.get('targets',[]) if t['target_id']==args.target_id),None)
        target_mask=np.full((8,2,18,32),-1,np.int8)
        target_info=dict(target_id=args.target_id,status='UNKNOWN')
        if target is not None and target.get('status')=='EVALUABLE' and target.get('authority'):
            if target.get('frame_indices')!=ids:
                raise ValueError('Target identity/order mismatch')
            mask_path=route.within(label_path.parent,target['masks'])
            raw=np.load(mask_path,allow_pickle=False)
            if raw.shape not in ((8,2,360,640),(8,2,18,32)) or not np.isin(raw,[-1,0,1]).all():
                raise ValueError('Invalid target masks')
            target_mask=route.pooled(raw)
            target_info.update(status='SUPPLIED',authority=target['authority'],mask_sha256=route.sha(mask_path))
        drift=[]
        for i,train_id in enumerate(CONTROL_TRAIN_IDS):
            old=train_paths[train_ids.index(train_id)]
            with adapt.Image.open(old) as im:
                a=np.asarray(im.convert('RGB'),dtype=np.int16)
                ar=np.asarray(im.convert('RGB').resize((256,144),adapt.Image.Resampling.BOX),dtype=np.int16)
            with adapt.Image.open(paths[i]) as im:
                b=np.asarray(im.convert('RGB'),dtype=np.int16)
                br=np.asarray(im.convert('RGB').resize((256,144),adapt.Image.Resampling.BOX),dtype=np.int16)
            drift.append(dict(control_index=i,original_train_id=train_id,old_rgb_sha256=route.sha(old),
                new_rgb_sha256=route.sha(paths[i]),rgb_mean_absolute_difference=float(np.abs(a-b).mean()),
                model_rgb_mean_absolute_difference=float(np.abs(ar-br).mean()),difference_scale='uint8 [0,255]',
                changed_pixel_fraction=float((a!=b).any(axis=-1).mean())))
        results={}
        for name,prediction in predictions.items():
            pairs=[]
            for control,train_id in enumerate(CONTROL_TRAIN_IDS):
                treatment=control+4
                c,t=map(float,prediction['near'][[control,treatment],0])
                prior=float(history[name]['near'][train_ids.index(train_id),0])
                cm,tm=[support_evidence(prediction['support'][i,0],target_mask[i,0]) for i in (control,treatment)]
                for i,evidence in ((control,cm),(treatment,tm)):
                    if near[i,0] < 0:
                        evidence.update(status='UNKNOWN',reason='Scene near geometry UNKNOWN',
                            overlap=None,peak_hit=None,joint_eligible=False)
                policies={policy:threshold_pair(c,t,values[0]['value']) for policy,values in configs[name].items()}
                for policy,row in policies.items():
                    if row.get('threshold') is not None:
                        row.update(control_joint=None if not cm['joint_eligible'] else row['control_alert'] and cm['overlap'],
                            treatment_joint=None if not tm['joint_eligible'] else row['treatment_alert'] and tm['overlap'],
                            historical_control_alert=prior>=row['threshold'],
                            control_alert_agrees_with_historical=(prior>=row['threshold'])==row['control_alert'])
                pairs.append(dict(control_index=control,treatment_index=treatment,original_train_id=train_id,
                    control_near_probability=c,treatment_near_probability=t,paired_delta=t-c,
                    historical_train_probability=prior,control_replay_delta=c-prior,
                    control_scene_near_truth=int(near[control,0]),treatment_scene_near_truth=int(near[treatment,0]),
                    control_target=cm,treatment_target=tm,policies=policies))
            eligible=[p for p in pairs if p['control_target']['joint_eligible'] and p['treatment_target']['joint_eligible']]
            results[name]=dict(pairs=pairs,all_pair_mean_probability_delta=float(np.mean([p['paired_delta'] for p in pairs])),
                control_alert_agreement={policy:sum(p['policies'][policy]['control_alert_agrees_with_historical'] for p in pairs)
                    for policy in ('historical','dev_selected') if configs[name][policy][0]['value'] is not None},
                geometry_evaluable_pairs=len(eligible),geometry_unknown_or_no_positive_pairs=4-len(eligible),
                paired_target_summary={policy:dict(control_alerts=sum(p['policies'][policy]['control_alert'] for p in eligible),
                    treatment_alerts=sum(p['policies'][policy]['treatment_alert'] for p in eligible),
                    control_joint=sum(bool(p['policies'][policy]['control_joint']) for p in eligible),
                    treatment_joint=sum(bool(p['policies'][policy]['treatment_joint']) for p in eligible))
                    for policy in ('historical','dev_selected') if configs[name][policy][0]['value'] is not None})
        if route.sha(out/'input-manifest.json')!=manifest_hash:
            raise ValueError('Frozen input manifest changed')
        composition=[dict(sample_index=i,head='BODY',**target_scene_composition(support[i,0],target_mask[i,0])) for i in ids]
        route.write(out/'result.json',dict(methods=results,labels=info,target=target_info,control_render_drift=drift,
            scene_target_composition=composition,
            interpretation='Paired scores are descriptive global near alerts, not target-specific recognition. Joint alert/target-overlap means co-occurrence only when other query support is present. Outside recorded target support is untargeted, with UNKNOWN portions reported separately; unknown target geometry excluded from opportunity counts. Render drift reported, not asserted zero.'))
        route.write(out/'receipt.json',dict(status='PASS',optimizer_steps=0,actual_backend=backend,
            per_model_load_inference_seconds=timings,total_seconds=time.perf_counter()-started,
            input_manifest_sha256=manifest_hash,result_sha256=route.sha(out/'result.json'),
            predictions_sha256={name:route.sha(out/f'{name}-predictions.npz') for name in predictions}))
    except BaseException as error:
        route.write(out/'receipt.json',dict(status='FAIL',optimizer_steps=0,error=repr(error)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--target-id',default='train_bollard')
    p.add_argument('--score-audit',type=Path,default=route.NF/'city-native-score-audit-20260908/scores-v1')
    p.add_argument('--output',type=Path,default=route.NF/'city-native-view-probe-20260908/evaluation-v1')
    run(p.parse_args())
