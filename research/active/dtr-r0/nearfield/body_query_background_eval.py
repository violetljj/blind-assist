"""Separate native admission and frozen RGB-only background-only inference."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from PIL import Image
from body_query_collection_labels import read, write, sha, verify_capture, floor_acceptance
from body_query_fresh_native_labels import labels_batch, endpoint_reasons


FROZEN = {
    'NEW-step2000.pt':'db39ccfcb9f3d1c8f3722711bab314ff0cf2da708e00d91f7fb263e6fec0ece0',
    'JOINT.pt':'6458b02269a79b278bd639d74c9fbcf8cf9af1a3c926409c4743d85eeec46cde',
    'normalization.npz':'81434e980c19206ee94327e67f2a39181fd798e4819f7a64fb928eab17ae4452',
    'selection.json':'f7989b0befdc69d472ab900aa5522bac65e3103a7afc2b4d0b684da3379e5671'}


def box(mask):
    y,x = np.nonzero(mask)
    return None if not len(x) else [int(x.min()),int(y.min()),int(x.max()),int(y.max())]


def admit(a):
    rows,counts,bindings = [],[],{}
    manifest = read(a.specs/'manifest.json')
    captures = read(a.captures)
    assert set(captures)=={r['region_id'] for r in manifest['regions']}
    for region,location in captures.items():
        root = Path(location)
        spec,world,binding,probes = verify_capture(root)
        planned = next(r for r in manifest['regions'] if r['region_id']==region)
        original_path = a.specs/planned['spec_file']
        assert sha(original_path)==planned['spec_sha256']
        original, mapped = read(original_path),dict(spec)
        # Worker-local map path is the only permitted source rewrite.
        original.pop('map_file'); mapped.pop('map_file')
        assert original==mapped, 'Captured source differs from frozen geometry'
        bindings[region] = binding
        checks = read(root/'evaluator/target-raycheck.json')['rows']
        capture_receipt = read(root/'receipt.json')
        for i,c in enumerate(spec['cases']):
            poses = [r for r in capture_receipt['capture_poses'] if r['sample_index']==i]
            assert len(poses)==1
            for channel in ('rgb','depth'):
                for key in ('x','y','z','yaw','pitch','roll'):
                    difference = poses[0][channel][key]-c['camera'][key]
                    if key in ('yaw','pitch','roll'): difference=(difference+180)%360-180
                    assert abs(difference)<1e-4, 'Actual capture pose differs from declared calibration'
            native = np.load(root/f'evaluator/native/{i:04d}.npy',allow_pickle=False)
            isolated = np.load(root/f'evaluator/isolated/adjustable_cross_member/{i:04d}.npy',allow_pickle=False)
            label = labels_batch(torch.from_numpy(native[None]).cuda(),[c['camera']],[c['floor_z_m']])
            count = label['counts'][0].cpu().numpy()
            raw = label['raw_counts'][0].cpu().numpy()
            reasons = endpoint_reasons(label['near'][0].cpu().tolist(),count.tolist(),raw.tolist(),
                                       c['declared_range'],floor_acceptance(c,probes)['accepted'])
            ray = [r for r in checks if r['sample_index']==i and r['target_id']=='adjustable_cross_member']
            targets = [r for r in capture_receipt['controlled_target_bindings'] if r['sample_index']==i]
            assert len(targets)==1 and len(targets[0]['targets'])==1
            target_binding = targets[0]['targets'][0]
            assert target_binding['target_id']=='adjustable_cross_member'
            if len(ray)==1:
                assert ray[0]['source_component']==target_binding['component_path']
                assert ray[0]['source_mesh']==target_binding['mesh_asset']
            if len(ray)!=1 or ray[0]['status']!='PASS': reasons.append('TARGET_NATIVE_IDENTITY_UNKNOWN')
            valid = np.isfinite(isolated)&(isolated>0)&(isolated<15)
            visible = valid & np.isfinite(native)&(native>0)&(np.abs(native-isolated)<=.03)
            if visible.sum()<3: reasons.append('TARGET_VISIBILITY_GAP')
            bbox,visible_bbox = box(valid),box(visible)
            if bbox is None or visible_bbox is None:
                reasons.append('MISSING_TARGET_RASTER')
            elif any(abs(x-y)>1 for x,y in zip(bbox,visible_bbox)):
                reasons.append('TARGET_VISIBLE_EXTENT_OCCLUDED')
            image = root/f'model/sample/{i:04d}.png'
            rows.append(dict(sample_index=len(rows),capture_index=i,region_id=region,
                 pair_id=c['pair_id'],site_id=c['site_id'],arm=c['arm'],endpoint=c['declared_range'],original_index=c['original_index'],original_pair_id=c['original_pair_id'],
                 rgb_path=str(image),rgb_sha256=sha(image),
                 native_sha256=sha(root/f'evaluator/native/{i:04d}.npy'),
                 isolated_sha256=sha(root/f'evaluator/isolated/adjustable_cross_member/{i:04d}.npy'),
                 target_bbox=bbox,visible_target_bbox=visible_bbox,
                 unknown_pixels=int((~(np.isfinite(native)&(native>0)&(native<100))).sum()),
                 reasons=reasons))
            counts.append(count)
    assert len(rows)==120
    pairs = {}
    for row in rows: pairs.setdefault(row['pair_id'],[]).append(row)
    assert len(pairs)==60
    for pid,members in pairs.items():
        assert len(members)==2 and {r['endpoint'] for r in members}=={'near','far'}
        reasons = sorted(set(x for r in members for x in r['reasons']))
        if members[0]['arm']=='matched' and not reasons:
            sizes = [[b[2]-b[0]+1,b[3]-b[1]+1] for b in [r['visible_target_bbox'] for r in members]]
            if any(abs(n-f)>max(1.,.05*max(n,f)) for n,f in zip(*sizes)):
                reasons.append('TARGET_APPARENT_SIZE_NOT_MATCHED')
        for row in members: row.update(pair_reasons=reasons,admitted=not reasons)
    coverage = {arm:sum(r['admitted'] and r['endpoint']=='near' and r['arm']==arm for r in rows)
                for arm in ('background',)}
    region_coverage = {region:sum(r['admitted'] and r['endpoint']=='near' and r['region_id']==region for r in rows) for region in captures}
    status = 'PASS' if all(v>=24 for v in region_coverage.values()) else 'NOT_EVALUABLE'
    write(a.output/'frame_metadata.json',rows)
    np.savez_compressed(a.output/'evaluator_truth.npz',counts=np.stack(counts))
    write(a.output/'admission.json',dict(status=status,planned_pairs=60,planned_per_arm=60,region_coverage=region_coverage,
          admitted_pairs=coverage,training_steps=0,inference_frames=0,bindings=bindings,
          frame_metadata_sha256=sha(a.output/'frame_metadata.json'),
          truth_sha256=sha(a.output/'evaluator_truth.npz'),source_sha256=sha(__file__),
          device=torch.cuda.get_device_name(0)))
    print(json.dumps(dict(status=status,admitted_pairs=coverage)),flush=True)


def infer(a):
    from body_query_context_evidence import ContextEvidence
    from body_query_range import range_from_counts
    receipt = read(a.admission/'admission.json')
    diagnostic = False
    assert not a.diagnostic_source_shortfall, 'This experiment has no source-shortfall override'
    if diagnostic:
        assert receipt['status']=='NOT_EVALUABLE'
        assert sum(receipt['admitted_pairs'].values())>=200
    else:
        assert receipt['status']=='PASS', 'No confirmatory inference on a source-not-evaluable cohort'
    assert sha(a.admission/'frame_metadata.json')==receipt['frame_metadata_sha256']
    assert sha(a.admission/'evaluator_truth.npz')==receipt['truth_sha256']
    for name,digest in FROZEN.items():
        assert sha((a.baseline if name=='NEW-step2000.pt' else a.decoder)/name)==digest
    rows = read(a.admission/'frame_metadata.json')
    counts = np.load(a.admission/'evaluator_truth.npz',allow_pickle=False)['counts']
    model = ContextEvidence(a.baseline,a.decoder,a.pretrained).cuda().eval()
    arrays = {k:[] for k in ('BASE_counts','JOINT_counts','BASE_ranges','JOINT_ranges','BASE_alerts','retained_alerts')}
    start=time.perf_counter()
    with torch.inference_mode():
        for begin in range(0,len(rows),16):
            images=[]
            for row in rows[begin:begin+16]:
                path = (a.rgb_root/row['region_id']/f"{row['capture_index']:04d}.png") if a.rgb_root else Path(row['rgb_path'])
                assert sha(path)==row['rgb_sha256']
                with Image.open(path) as im:
                    images.append(np.array(im.convert('RGB').resize((256,144),Image.Resampling.BOX)))
            x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).cuda().float()/255
            result=model(x); baseline=model.base(x)
            values=dict(BASE_counts=baseline[2].softmax(-1),
                        JOINT_counts=result['geometry_count_logits'].softmax(-1),
                        BASE_ranges=range_from_counts(baseline[2]).sigmoid(),
                        JOINT_ranges=result['range_probabilities'],
                        BASE_alerts=baseline[0].sigmoid()>=model.alert_thresholds,
                        retained_alerts=result['alerts'])
            for key,value in values.items(): arrays[key].append(value.cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    parity=(arrays['BASE_alerts']==arrays['retained_alerts']).all(1)
    results={}
    truth=counts.reshape(-1,2,2,3).sum(-1)>=3
    for arm in ('background','combined'):
        ids=np.array([i for i,r in enumerate(rows) if r['admitted'] and (arm=='combined' or r['arm']==arm)])
        near=ids[[rows[i]['endpoint']=='near' for i in ids]]
        pairids=sorted({rows[i]['pair_id'] for i in ids})
        results[arm]={}
        for method in ('BASE','JOINT'):
            pred=arrays[method+'_ranges']>=.5
            positive=counts[ids,6:9]>0
            hit=(1-arrays[method+'_counts'][ids,6:9,0])>=.5
            exact=(pred[:,1]==truth[:,1]).all(1)
            correct=sum(all(exact[i] for i in ids if rows[i]['pair_id']==p) for p in pairids)
            results[arm][method]=dict(
                near_hit=dict(numerator=int((hit&positive).sum()),denominator=int(positive.sum())),
                near_to_far_confusion=dict(numerator=int(pred[near,1,1].sum()),denominator=len(near)),
                pair_correct=dict(numerator=int(correct),denominator=len(pairids)),
                original_alert_parity=dict(numerator=int(parity[ids].sum()),denominator=len(ids)))
    spatial_gate=all(results[arm]['JOINT']['pair_correct']['numerator']/results[arm]['JOINT']['pair_correct']['denominator']>=.8
                     for arm in ('background',))
    gate=False  # Attribution diagnostic never authorizes promotion
    np.savez_compressed(a.output/'predictions.npz',**arrays)
    write(a.output/'result.json',dict(status='PASS',promotion_gate=bool(gate),metrics=results,
          evaluation_mode='DIAGNOSTIC_AFTER_SOURCE_SHORTFALL' if diagnostic else 'CONSUMED_BACKGROUND_ATTRIBUTION',
          source_gate=receipt['status'],spatial_gate=bool(spatial_gate),
          all_captured_alert_parity=dict(numerator=int(parity.sum()),denominator=len(rows)),
          admission_sha256=sha(a.admission/'admission.json'),predictions_sha256=sha(a.output/'predictions.npz'),
          frozen_hashes=FROZEN,source_sha256=sha(__file__),training_steps=0,
          device=torch.cuda.get_device_name(0),seconds=time.perf_counter()-start))
    print(json.dumps(read(a.output/'result.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['admit','infer'])
    for key in ('captures','specs','admission','baseline','decoder','pretrained','rgb-root'):
        p.add_argument('--'+key,type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostic-source-shortfall',action='store_true',
                   help='Report admitted subset only; never overrides the failed source/promotion gate')
    a=p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available(), 'CUDA required for this bounded evaluation'
    (admit if a.mode=='admit' else infer)(a)
