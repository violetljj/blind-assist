"""MZ67 native labels: frozen MZ55 kernels with bound geometry-role/shard admission."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch


from mz67_source_contract import admit_spec, geometry_metadata, source_binding_fields, verify_map_file


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8', newline='\n')


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def rotation(degrees):
    p, y, r = (math.radians(degrees[k]) for k in ('pitch', 'yaw', 'roll'))
    sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
    return np.array([[cp*cy, sr*sp*cy-cr*sy, -cr*sp*cy-sr*sy],
                     [cp*sy, sr*sp*sy+cr*cy, -cr*sp*sy+sr*cy], [sp, -sr*cp, cr*cp]])


def actual_bounds(actor, camera, floor):
    bounds = actor['mesh_local_bounds_cm']
    corners = np.array(list(itertools.product(*zip(bounds['min'], bounds['max']))))*.01*np.array(actor['scale'])
    points = corners @ rotation(actor['rotation_deg']).T + np.array(actor['actor_origin_m'])
    points -= np.array([camera['x'], camera['y'], floor])
    points = points @ rotation(dict(pitch=0., yaw=camera['yaw'], roll=0.))
    return points.min(0), points.max(0)


def event_masks(depth):
    yy, xx = np.indices((360, 640)); f = 320./np.tan(np.deg2rad(50.))
    right, up = (xx-319.5)/f, -(yy-179.5)/f
    d = depth.astype(np.float64); y, z = d*right, 1.7+d*up
    good = np.isfinite(d) & (d>0) & (d<100) & (d*np.sqrt(1+right**2+up**2)<=4)
    masks = []
    for front, width, low, high in ((.18, .28, .65, 1.4), (.13, .18, 1.4, 1.85)):
        for half in range(2):
            masks.append(good & (d>=front+half*1.5) & ((d<front+1.5) if half==0 else (d<=front+3.))
                         & (abs(y)<=width) & (z>=low) & (z<=high))
    return np.stack(masks)


def build(capture, output, runtime, dependencies, manifest, specs):
    sys.path.insert(0, str(runtime/'research/active/dtr-r0/nearfield'))
    sys.path.insert(0, str(dependencies))
    from body_query_collection_labels import verify_capture, floor_acceptance
    from body_query_labels import labels
    from multizone64_observation import native_events
    from mz9_contributors import reconstruct
    spec, verified_world, inputs, probes = verify_capture(capture)
    source_receipt = read(capture/'receipt.json')
    cases = spec['cases']
    contract = admit_spec(manifest, specs, spec)
    contract['inputs'].update(verify_map_file(spec))
    binding = source_binding_fields(contract, capture/'source/spec.json')
    output.mkdir(parents=True, exist_ok=False)
    package = output/'training-source'; package.mkdir()
    torch.set_num_threads(1); assert torch.cuda.is_available()
    records, packets, truths, knowns, sources, queries, local_known = [], [], [], [], [], [], []
    pairs, previews, previous = [], [], None
    preview_ids = [i for i,c in enumerate(cases) if c['canary'] or c['native_audit_sample']]
    board = Image.new('RGB', (1280, math.ceil(len(preview_ids)/4)*410), '#171c20')
    draw = ImageDraw.Draw(board)
    with torch.inference_mode():
        for i, (case, row) in enumerate(zip(cases, verified_world['rows'])):
            path = capture/f'evaluator/native/{i:04d}.npy'
            depth = np.load(path, allow_pickle=False); tensor = torch.from_numpy(depth).cuda()
            events = native_events(tensor[None], crop=False)
            geometry = labels(tensor, case['camera'], case['floor_z_m'])
            support = geometry['support'].cpu().numpy()
            np.testing.assert_array_equal(support, np.load(capture/row['mask_path'], allow_pickle=False))
            masks = event_masks(depth); counts = masks.sum((1,2)).tolist()
            assert counts == events['counts'][0].cpu().tolist()
            truth = (np.array(counts)>=3).tolist()
            floor = floor_acceptance(case, probes)
            source_valid = bool(floor['accepted'] and events['observation_valid'][0])
            actors = [a for a in source_receipt['controlled_objects'] if a['case']==case['name']]
            assert len(actors)==len(case['objects'])
            targets, support_bounds = [], []
            for obj, actor in zip(case['objects'], actors):
                assert obj['name']==actor['name'] and obj['center_m']==actor['actor_origin_m']
                assert obj['rotation_deg']==actor['rotation_deg']
                scale = obj.get('scale', obj.get('size_m'))
                if isinstance(scale, (float, int)):
                    scale = [scale]*3
                assert actor['scale']==scale
                low, high = actual_bounds(actor, case['camera'], case['floor_z_m'])
                if obj['target_part']:
                    intended = next(b for b in case['placement_bounds'] if b['name']==obj['name'])
                    np.testing.assert_allclose(low, intended['camera_aligned_min_m'], atol=1e-5, rtol=0)
                    np.testing.assert_allclose(high, intended['camera_aligned_max_m'], atol=1e-5, rtol=0)
                    targets.append({k:v for k,v in actor.items() if k!='case'})
                else:
                    outside = bool(high[1]<-.28 or low[1]>.28)
                    assert outside, (case['name'], obj['name'], low, high)
                    support_bounds.append(dict(name=obj['name'], min_m=low.tolist(), max_m=high.tolist(), outside_queries=True))
            packet = reconstruct(tensor[None])
            ranges = torch.nan_to_num(packet['range_m'][0]).float().cpu().numpy()
            valid = packet['valid'][0].cpu().numpy()
            packets.append((ranges, valid)); truths.append(truth); knowns.append([source_valid]*4)
            sources.append(packet['source_counts'][0].cpu().numpy()>0)
            queries.append(packet['query_counts'][0].cpu().numpy()>0)
            local_known.append(packet['cell_known_counts'][0].cpu().numpy()>0)
            role = case['training_role']
            assert role in ('TRAIN_CANDIDATE', 'CALIBRATION', 'HELDOUT_GEOMETRY')
            record = dict(index=i, frame_id=case['name'], pair_id=case['pair_id'], site_id=case['site_id'],
                region_id=case['region_id'], family=case['condition']['family'], relation=case['variant_id'],
                range=case['declared_range'], support_context=case['support_context'], setting=case['profile_id'], role=role,
                camera=case['camera'], wearer=case['wearer'], floor_z_m=case['floor_z_m'],
                coordinate_contract='Camera forward/right/up transformed by camera yaw to world; eye1.7m; axial native depth',
                rgb=f'model/rgb/{i:04d}.png', rgb_sha256=row['rgb_sha256'], native_sha256=row['native_sha256'],
                worker_native=str(path), source_valid=source_valid, floor=floor, event_truth=truth, event_counts=counts,
                expected_events=case['expected_events'], expected_collapsed_relation=case['expected_collapsed_relation'],
                collapsed_relation=[bool(truth[0] or truth[1]), bool(truth[2] or truth[3])],
                intent_matches=[bool(truth[0] or truth[1]), bool(truth[2] or truth[3])]==[bool(v) for v in case['expected_collapsed_relation']],
                exclusive_anchor_bits_match=truth==[bool(v) for v in case['expected_events']],
                extra_range_bits=[bool(t and not e) for t,e in zip(truth,case['expected_events'])],
                canary=case['canary'], native_audit_sample=case['native_audit_sample'],
                target_actors=targets, ancillary_bounds=support_bounds, **geometry_metadata(case))
            link(capture/f'model/sample/{i:04d}.png', package/record['rgb'])
            if case['native_audit_sample']:
                link(path, package/f'evaluator/native/{i:04d}.npy')
            if case['support_context']=='unsupported':
                previous = (case, depth, masks, targets)
            else:
                before, olddepth, oldmasks, oldtargets = previous
                assert before['pair_id']==case['pair_id'] and oldtargets==targets
                assert before['objects']==[o for o in case['objects'] if o['target_part']]
                difference = (oldmasks!=masks).sum((1,2)).tolist()
                union = oldmasks.any(0)|masks.any(0)
                error = float(np.abs(olddepth[union]-depth[union]).max()) if union.any() else 0.
                pairs.append(dict(pair_id=case['pair_id'], indices=[i-1,i], role=role, target_dictionary_equal=True,
                    actual_target_receipt_equal=True, changed_event_mask_pixels=difference,
                    event_depth_max_abs_m=error, native_pair_invariant=not any(difference) and error==0.))
                previous = None
            if i in preview_ids:
                position = len(previews); x, y = position%4*320, position//4*410
                with Image.open(capture/f'model/sample/{i:04d}.png') as im:
                    board.paste(im.convert('RGB').resize((320,180)), (x,y+30))
                shade=np.uint8(np.clip(1-depth/5.,0,1)*255);color=np.stack([shade]*3,-1)
                color[support[0]==1]=[80,180,255];color[support[1]==1]=[255,120,80]
                board.paste(Image.fromarray(color).resize((320,180)),(x,y+215))
                draw.text((x+4,y+2), f"{case['site_id'][-3:]} {record['family']} {record['relation']} {record['range']}",fill='white')
                draw.text((x+4,y+15), f"p{record['setting']} {record['support_context']}",fill='white')
                draw.text((x+4,y+395), str([int(v) for v in truth])+f" intent={record['intent_matches']}",fill='white')
                previews.append(case['name'])
            records.append(record)
    assert previous is None and len(pairs)*2==len(cases)
    np.savez_compressed(package/'model/packets.npz', ranges=np.stack([p[0] for p in packets]), valid=np.stack([p[1] for p in packets]))
    (package/'evaluator').mkdir(exist_ok=True)
    np.savez_compressed(package/'evaluator/labels.npz', truth=np.array(truths,bool), known=np.array(knowns,bool),
                        source_presence=np.array(sources,bool), query_presence=np.array(queries,bool), cell_known=np.array(local_known,bool))
    write(package/'model/predictor.json', dict(schema='mz67-observable-input-v1', frames=len(cases),
        rgb_files=[r['rgb'] for r in records], packets='model/packets.npz',
        calibration=dict(native_size=[640,360], horizontal_fov_degrees=100, eye_height_m=1.7, zone_crop_degrees=[45,45]),
        forbidden_inputs=['evaluator/*', 'site/setting/family/context/instance IDs', 'native depth or contributor labels']))
    write(package/'evaluator/metadata.json', dict(schema='mz67-training-source-v1', records=records, pairs=pairs,
        label_semantics='Actual IDEAL first/last supported0.1m bin contributors; mask droppedslots only; never inherit to MERGE midpoint',
        expected_relation_authority='Intent only; not labels', role_authority='Bound camera-geometry recipe split; all site/range/relation/support replicas grouped; evaluator only'))
    if previews:
        board.save(output/'contact-sheet.jpg',quality=94)
    for page in range(math.ceil(len(previews)/8)):
        board.crop((0,page*820,1280,min(board.height,(page+1)*820))).save(output/f'preview-{page:02d}.jpg',quality=94)
    native_count=sum(r['native_audit_sample'] for r in records)
    result=dict(status='PASS_SOURCE_CHECKS',frames=len(cases),source_valid_frames=sum(r['source_valid'] for r in records),
        unknown_bits=int((~np.array(knowns)).sum()),intent_matching_frames=sum(r['intent_matches'] for r in records),
        positive_frames_by_query=np.array(truths).sum(0).tolist(),pairs=len(pairs),
        extra_range_bits=np.array([r['extra_range_bits'] for r in records]).sum(0).tolist(),
        exclusive_anchor_matching_frames=sum(r['exclusive_anchor_bits_match'] for r in records),
        invariant_pairs=sum(p['native_pair_invariant'] for p in pairs),native_audit_frames=native_count,
        roles={role:sum(r['role']==role for r in records) for role in ('TRAIN_CANDIDATE','CALIBRATION','HELDOUT_GEOMETRY')},
        preview_frame_ids=previews,visual_review='PENDING',training_steps=0,model_inference_frames=0,
        worker_raw=str(capture),raw_bytes=sum(p.stat().st_size for p in capture.rglob('*') if p.is_file()))
    write(output/'result.json',result)
    for name in ('receipt.json','completion.json','source-integrity.json','process-release.json','render-resource-health.json','world-verification.json'):
        shutil.copyfile(capture/name,package/'evaluator'/('capture-'+name))
    write(package/'evaluator/source-bindings.json',dict(inputs=inputs,source_spec_sha256=sha(capture/'source/spec.json'),
        code_sha256=sha(__file__),geometry_contract=contract,**binding,dependencies={str(p):sha(p) for p in dependencies.glob('*.py')},
        worker_raw=str(capture),native_files=[dict(index=r['index'],sha256=r['native_sha256'],worker_path=r['worker_native']) for r in records]))
    write(output/'receipt.json',dict(status='PASS',inputs=inputs,code_sha256=sha(__file__),geometry_contract=contract,**binding,
        package_hashes={str(p.relative_to(package)):sha(p) for p in (package/'model/predictor.json',package/'model/packets.npz',package/'evaluator/labels.npz',package/'evaluator/metadata.json',package/'evaluator/source-bindings.json')},
        outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        source_spec_sha256=sha(capture/'source/spec.json'),package_directory=str(package),
        native_audit_frames=native_count,training_steps=0,model_inference_frames=0))
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','output','runtime','dependencies','manifest','specs'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    build(args.capture.resolve(),args.output.resolve(),args.runtime.resolve(),args.dependencies.resolve(),args.manifest.resolve(),args.specs.resolve())
