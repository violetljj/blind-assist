"""New AABB intervention coverage with immutable geometric-background splits.

An absent target is a newly rendered background-only world, never rho=0 of an
opaque target and never a subtraction from sampled histograms. Geometry and
labels are evaluator/authoring data; only normalized observations and public
transforms enter downstream models. No work runs at import time.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import cnh_counterfactual_common_dev as C

OUT = C.OUT
FRAMES = np.arange(3, 16)
SPLITS = ('ordinary_train', 'cf_train', 'cal', 'validation')
FAMILIES = {
    'train': ('flat_backwall', 'single_side_slab', 'paired_side_slabs', 'staggered_side_blocks'),
    'cal': ('segmented_backwall', 'side_gate_posts'),
    'validation': ('L_sidewall', 'stacked_side_shelves'),
}
SHAPES = ('horizontal', 'vertical', 'protrusion', 'sign_edge')
PLACEMENTS = ('in1cm', 'in4cm', 'in12cm', 'pass5cm', 'clear15cm')
INNER = dict(zip(PLACEMENTS, (.29, .26, .18, .35, .45)))
DESIGN_SEED, PHOTON_SEED = 2026100961, 2026100962
EPS = 1e-8


def frozen_imports():
    """Frozen sensor/mesh imports reject an already-loaded live geometry WIP."""
    import cnh_proposal_attribution_scenes as S
    import cnh_displacement_ceiling_render as R
    import cnh_location_reference_gpu as G
    import cnh_aligned_boundary_dev as B
    for name in ('cnh_route_sensor', 'cnh_track_a_geometry', 'cnh_track_a_fov', 'cnh_track_a_v13_sensor'):
        if Path(sys.modules[name].__file__).resolve().parent != S.SOURCE.resolve():
            raise ImportError('Frozen source boundary violated: ' + name)
    return S, R, G, B


def box(lo, hi, rho):
    return dict(lo=list(map(float, lo)), hi=list(map(float, hi)), rho=float(rho))


def mirror_boxes(boxes, sign):
    if sign == 1:
        return boxes
    return [dict(b, lo=[-b['hi'][0], *b['lo'][1:]], hi=[-b['lo'][0], *b['hi'][1:]]) for b in boxes]


def backgrounds():
    """Eight geometry families, entirely assigned to train/cal/validation."""
    result = []
    for partition, families in FAMILIES.items():
        for family in families:
            for instance in (0, 1):
                rho = .25 if instance == 0 else .65
                sign = 1 if instance == 0 else -1
                floor = box([-8, 1.65, -8], [8, 1.80, 9], .30 if instance == 0 else .45)
                wall = box([-8, -3, 4.2+.3*instance], [8, 1.65, 4.4+.3*instance], .35)
                near = []
                if family == 'flat_backwall':
                    wall = box([-8, -3, 3.65+.7*instance], [8, 1.65, 3.85+.7*instance], .35)
                elif family == 'single_side_slab':
                    near = mirror_boxes([box([.52, -.35, 1.25], [.92, 1.35, 2.25+.3*instance], rho)], sign)
                elif family == 'paired_side_slabs':
                    near = [box([.58, -.30, 1.35], [1.05, 1.30, 2.95], rho),
                            box([-1.15, -.30, 1.55], [-.58, 1.30, 3.15], .65 if instance == 0 else .25)]
                elif family == 'staggered_side_blocks':
                    near = mirror_boxes([box([.50, -.30, 1.25], [.85, .22, 1.55], rho),
                                         box([-.95, .40, 1.75], [-.50, .95, 2.10], .35),
                                         box([.55, .98, 2.45], [.90, 1.35, 2.80], rho)], sign)
                elif family == 'segmented_backwall':
                    wall = None
                    near = [box([-8, -3, 3.35+.2*instance], [-.72, 1.65, 3.55+.2*instance], rho),
                            box([-.72, -3, 3.75+.3*instance], [.72, 1.65, 3.95+.3*instance], .35),
                            box([.72, -3, 4.15+.2*instance], [8, 1.65, 4.35+.2*instance], rho)]
                elif family == 'side_gate_posts':
                    near = [box([.46, -.50, 1.30], [.58, 1.40, 1.48], rho),
                            box([-.58, -.50, 1.55], [-.46, 1.40, 1.73], rho),
                            box([-1.10, -.52, 1.85], [1.10, -.30, 2.05], .35)]
                elif family == 'L_sidewall':
                    near = mirror_boxes([box([.48, -.35, 1.25], [.70, 1.45, 3.15], rho),
                                         box([.70, -.35, 1.25], [1.40, 1.45, 1.48], rho)], sign)
                elif family == 'stacked_side_shelves':
                    near = mirror_boxes([
                        box([.46, -.38, 1.30], [1.15, -.30, 2.85], rho),
                        box([.46, .94, 1.45], [1.15, 1.02, 2.95], rho),
                        box([.46, 1.25, 1.65], [1.15, 1.33, 3.05], rho)], sign)
                else:
                    raise ValueError(family)
                result.append(dict(background_id=len(result), background_family=family,
                    partition=partition, instance=instance, boxes=[floor]+([] if wall is None else [wall])+near))
    return result


def target(shape, group, side, placement, draw, rho):
    inner = INNER[placement]
    y = .10 if group == 0 else .65
    if shape == 'horizontal':
        width, thickness = (.30 if draw % 2 == 0 else .90), (.04 if draw % 2 == 0 else .10)
        yl, yh, depth = y-thickness/2, y+thickness/2, thickness
    elif shape == 'vertical':
        width = .04 if draw % 2 == 0 else .10
        yl, yh = (-.05, .35) if group == 0 else (.45, .85)
        depth = width
    elif shape == 'protrusion':
        width, yl, yh, depth = .22, y-.10, y+.10, .08
    elif shape == 'sign_edge':
        width, depth = (.30, .02) if draw % 2 == 0 else (.02, .30)
        yl, yh = y-.10, y+.10
    else:
        raise ValueError(shape)
    xl, xh = (inner, inner+width) if side == 1 else (-inner-width, -inner)
    return box([xl, yl, .65], [xh, yh, .65+depth], rho)


def targets(partition):
    result = []
    base = {'train': 410000, 'cal': 420000, 'validation': 430000}[partition]
    for si, shape in enumerate(SHAPES):
        for group in (0, 1):
            for side_index, side in enumerate((-1, 1)):
                draw_list = range(3) if partition == 'train' else range(6)
                for draw in draw_list:
                    pi = (si+group+side_index+draw) % 5 if partition == 'train' else draw
                    placement = PLACEMENTS[pi] if pi < 5 else 'absent'
                    rho = .25 if (si+group+side_index+draw) % 2 == 0 else .65
                    geometry = target(shape, group, side, placement if placement != 'absent' else 'in1cm', draw, rho)
                    result.append(dict(target_id=base+len(result), shape_family=shape, group=group,
                        side=side, placement=placement, draw=draw, rho=rho, target_box=geometry,
                        presence=placement != 'absent'))
    return result


def physical_key(boxes):
    encoded = json.dumps(boxes, sort_keys=True, separators=(',', ':')).encode('utf8')
    return hashlib.sha256(encoded).hexdigest()


def scene_spec():
    bg = backgrounds()
    train_bg = [b for b in bg if b['partition'] == 'train']
    train_targets = targets('train')
    result = {s: [] for s in SPLITS}
    def append(split, t, b, present):
        boxes = ([t['target_box']] if present else []) + b['boxes']
        row = {k: t[k] for k in ('target_id', 'shape_family', 'group', 'side', 'placement', 'rho', 'target_box')}
        row.update(scene_id=len(result[split]), scene_uid=split+':'+str(len(result[split])),
            split=split, presence=bool(present), background_id=b['background_id'],
            background_family=b['background_family'], background_boxes=b['boxes'], boxes=boxes,
            physical_key=physical_key(boxes))
        result[split].append(row)
    for t in train_targets:
        for b in train_bg:
            for present in (False, True):
                append('cf_train', t, b, present)
    rng = np.random.default_rng(DESIGN_SEED)
    assignments = {p: rng.permutation(np.tile(np.arange(8), len(train_targets))) for p in (False, True)}
    for ti, t in enumerate(train_targets):
        for present in (False, True):
            for j in range(8):
                append('ordinary_train', t, train_bg[int(assignments[present][ti*8+j])], present)
    for split in ('cal', 'validation'):
        for t in targets(split):
            for b in (b for b in bg if b['partition'] == split):
                append(split, t, b, t['presence'])
    for split, rows in result.items():
        occurrences = Counter()
        for row in rows:
            key = row['physical_key']
            row['occurrence_draw_id'] = occurrences[key]
            occurrences[key] += 1
    return result


def sampling_seed(row, replica):
    key = bytes.fromhex(row['physical_key'])
    words = np.frombuffer(key, dtype='<u4').tolist()
    return int(np.random.SeedSequence([PHOTON_SEED, *words, row['occurrence_draw_id'], int(replica)]).generate_state(1)[0])


def category_boxes(boxes, shift):
    """Analytic axis-aligned surface-box intersections, all physical boxes.

    A box containing the whole query has no face in its interior; volume overlap
    alone is insufficient. Tangent interior conventions match frozen triangles.
    """
    lo = np.asarray([b['lo'] for b in boxes], float)-shift
    hi = np.asarray([b['hi'] for b in boxes], float)-shift
    result = []
    for yl, yh in ((-.20, .42), (.42, .90)):
        def hits(width, expanded=False):
            qlo = np.array([-width+EPS, yl+EPS, .30+EPS])
            qhi = np.array([width-EPS, yh-EPS, 3.-EPS])
            if expanded:
                qlo[0], qhi[0] = -width, width
            extent = np.minimum(hi, qhi)-np.maximum(lo, qlo)
            overlap = ((extent[:,0] >= 0) if expanded else (extent[:,0] > 0)) & (extent[:,1:] > 0).all(1)
            faces = (((lo >= qlo)&(lo <= qhi))|((hi >= qlo)&(hi <= qhi))).any(1)
            return bool(np.any(overlap & faces))
        result.append('contact' if hits(.30) else 'pass' if hits(.40, True) else 'clear')
    return result


def validate_spec(rows_by_split):
    if set(rows_by_split) != set(SPLITS):
        raise ValueError('Exactly ordinary/CF/cal/validation splits required')
    expected = {'ordinary_train': 768, 'cf_train': 768, 'cal': 384, 'validation': 384}
    marginal_fields = ('target_id', 'shape_family', 'group', 'side', 'placement', 'rho', 'presence', 'background_id')
    for split, rows in rows_by_split.items():
        if len(rows) != expected[split] or [r['scene_id'] for r in rows] != list(range(expected[split])):
            raise ValueError('Scene count/order changed: ' + split)
        assigned = 'train' if split.endswith('train') else split
        if any(r['background_family'] not in FAMILIES[assigned] for r in rows):
            raise ValueError('Intervention crosses background family split')
        for row in rows:
            if physical_key(row['boxes']) != row['physical_key']:
                raise ValueError('Physical key differs from actual boxes')
            expected_boxes = ([row['target_box']] if row['presence'] else []) + row['background_boxes']
            if row['boxes'] != expected_boxes:
                raise ValueError('Target removal must change actual physical boxes')
    a, b = rows_by_split['ordinary_train'], rows_by_split['cf_train']
    for field in marginal_fields:
        if Counter(r[field] for r in a) != Counter(r[field] for r in b):
            raise ValueError('Unmatched augmentation marginal: ' + field)
    compound = lambda rows: Counter((r['target_id'], r['background_id'], r['presence']) for r in rows)
    if len(compound(b)) != 768 or len(compound(a)) >= 768:
        raise ValueError('Ordinary random and CF fullcross coverage must differ')
    return dict(status='PASS', scenes={s: len(v) for s,v in rows_by_split.items()},
        equal_training_marginals=list(marginal_fields),
        train_joint_cells={'ordinary': len(compound(a)), 'counterfactual': len(compound(b))},
        physical_unique={s: len({r['physical_key'] for r in rows}) for s,rows in rows_by_split.items()},
        family_partitions={k: list(v) for k,v in FAMILIES.items()},
        eval_geometry='Cal and validation target IDs are disjoint; fixed same target fixture geometry across unseen background families, not novel-shape transfer',
        noise='Actual-boxes SHA256 + physical-world occurrence draw ID + replica, shared across augmentation recipes when identical. Repeated worlds receive independent occurrence draws.')


def scene_categories(rows, sensor):
    output = np.empty((len(rows), len(FRAMES), 2), dtype='<U7')
    for i,row in enumerate(rows):
        for j,f in enumerate(FRAMES):
            output[i,j] = category_boxes(row['boxes'], sensor[f,:3,3])
            if category_boxes(row['background_boxes'], sensor[f,:3,3]) != ['clear', 'clear']:
                raise ValueError('Background-only actual world is not clear')
    if not np.all(output == output[:, :1]):
        raise ValueError('This fixed fixture requires stable whole-episode categories')
    return output


def source_manifest():
    S,R,G,B = frozen_imports()
    modules = [sys.modules[__name__], C,S,R,G,B,
        *[sys.modules[name] for name in ('cnh_route_sensor','cnh_track_a_geometry','cnh_track_a_fov','cnh_track_a_v13_sensor')]]
    result = {}
    snapshot = OUT/'source_snapshot/data'
    snapshot.mkdir(parents=True, exist_ok=True)
    for module in modules:
        path = Path(module.__file__).resolve()
        name = path.name
        dest = snapshot/name
        if dest.exists() and dest.read_bytes() != path.read_bytes():
            raise ValueError('Preserve previous data source snapshot: ' + name)
        if not dest.exists():
            dest.write_bytes(path.read_bytes())
        result[str(path)] = dict(sha256=C.sha(path), snapshot=str(dest.relative_to(OUT)))
    return result


def prepare():
    began = time.monotonic()
    cfg = C.read(OUT/'PLAN.json')
    if (OUT/'scene_rows.json').exists():
        raise FileExistsError('Preserve prepared scene specification')
    if (cfg['train_scenes_per_aug'],cfg['eval_scenes_per_split'],cfg['train_K'],cfg['eval_K'],cfg['frames']) != (768,384,2,4,FRAMES.tolist()):
        raise ValueError('Scene contract differs from shared PLAN')
    rows = scene_spec()
    audit = validate_spec(rows)
    _,_,_,B = frozen_imports()
    sensor,query = B.poses(-10.)
    # Fixed query/sensor orientations cancel; evaluator labels use world axes.
    for f in FRAMES:
        expected=np.eye(4);expected[:3,3]=-sensor[f,:3,3]
        np.testing.assert_allclose(query[f]@np.linalg.inv(sensor[f]),expected,atol=1e-12,rtol=0)
    C.save(OUT/'scene_rows.json',rows)
    C.save(OUT/'background_rows.json',backgrounds())
    C.save(OUT/'data_design_audit.json',audit)
    categories = {}
    for split, split_rows in rows.items():
        folder = OUT/'data'/split
        folder.mkdir(parents=True, exist_ok=True)
        cats = scene_categories(split_rows,sensor)
        np.savez_compressed(folder/'geometry.npz', category=cats[:,0],frame_category=cats,
            scene_ids=np.arange(len(split_rows)),scene_uids=np.asarray([r['scene_uid'] for r in split_rows]),
            sensor=sensor,public_query=query)
        k=2 if split.endswith('train') else 4
        categories[split]=dict(category_scenes=dict(Counter('/'.join(c) for c in cats[:,0])),
            contact_events_per_height=((cats[:,0]=='contact').sum(0)*k).tolist(),
            joint_clear_slots=int((cats[:,0]=='clear').all(1).sum()*k*13),
            pass_clips=int(((cats[:,0]=='pass').any(1)&~(cats[:,0]=='contact').any(1)).sum()*k),
            physical_contact_events=int((cats[:,0]=='contact').any(1).sum()*k),replicas=k,rows=len(split_rows)*k*13)
    C.save(OUT/'data_source_manifest.json',source_manifest())
    C.save(OUT/'data_prepare_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,
        cpu_only=True,plan_sha256=C.sha(OUT/'PLAN.json'),scene_rows_sha256=C.sha(OUT/'scene_rows.json'),
        categories=categories,network_inputs='histories/transforms/length only; geometry,rows,labels,mask,weights never model features'))
    print(json.dumps(dict(status='PREPARED',audit=audit,categories=categories),ensure_ascii=False),flush=True)


def materialize(hist,ambient,sensor,query,categories,folder,check):
    import cnh_temporal_readout_data as OLD
    scenes,k=hist.shape[:2]
    n=scenes*k*13
    history=np.lib.format.open_memmap(folder/'histories.npy',mode='w+',dtype=np.float16,shape=(n,8,8,8,16))
    transforms=np.lib.format.open_memmap(folder/'transforms.npy',mode='w+',dtype=np.float64,shape=(n,8,4,4))
    length=np.lib.format.open_memmap(folder/'length.npy',mode='w+',dtype=np.int64,shape=(n,))
    yaw=np.lib.format.open_memmap(folder/'transforms_yaw3.npy',mode='w+',dtype=np.float64,shape=(n,8,4,4)) if folder.name in ('cal','validation') else None
    labels=np.empty((n,2),np.float32);mask=np.empty((n,2),np.float32);weights=np.empty((n,2),np.float32)
    row_scene=[];replica=[];frame=[]
    angle=np.deg2rad(3);rotation=np.eye(4);rotation[:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
    offset=0
    for i in range(scenes):
        check()
        z=OLD.normalized_z(hist[i],ambient)
        for rep in range(k):
            for j,f in enumerate(FRAMES):
                begin=max(0,int(f)-7);le=int(f)-begin+1;pad=8-le
                history[offset]=0;history[offset,pad:]=z[rep,begin:f+1]
                transforms[offset]=np.eye(4);transforms[offset,pad:]=query[f]@np.linalg.inv(sensor[f])@sensor[begin:f+1]
                if yaw is not None:
                    yaw[offset]=np.eye(4);yaw[offset,pad:]=rotation@query[f]@np.linalg.inv(sensor[f])@sensor[begin:f+1]
                length[offset]=le
                labels[offset]=(categories[i,j]=='contact')
                mask[offset]=(categories[i,j]!='pass')
                front=.65-sensor[f,2,3]
                weights[offset]=mask[offset]*(2. if 1.6<=front<2.6 else 1.)
                row_scene.append(i);replica.append(rep);frame.append(int(f));offset+=1
    raw_mass=float(weights.sum(dtype=np.float64))
    weights*=np.float32(n/raw_mass)
    for name,array in (('labels',labels),('mask',mask),('weights',weights)):
        np.save(folder/(name+'.npy'),array)
    for array in (history,transforms,length,yaw):
        if array is not None:array.flush()
    del history,transforms,length,yaw
    np.savez_compressed(folder/'rows.npz',scene_id=np.asarray(row_scene),replica=np.asarray(replica),frame=np.asarray(frame))
    return dict(rows=n,scene_count=scenes,replicas=k,raw_weight_mass=raw_mass,
        normalized_weight_mass=float(weights.sum(dtype=np.float64)),weight_rule='maskpass0,far1.6..2.6m multiplier2; every augmentation train totalweight=N like inherited oldtrain totalN; no class balancing')


def render():
    """Called only after controller authorizes the shared scientific queue."""
    stage=C.Stage('data_render')
    engines=[]
    histmaps={}
    cuda_dll_handle=None
    try:
        cfg=C.read(OUT/'PLAN.json')
        prep=C.read(OUT/'data_prepare_receipt.json')
        if prep['status']!='COMPLETE' or prep['plan_sha256']!=C.sha(OUT/'PLAN.json') or prep['scene_rows_sha256']!=C.sha(OUT/'scene_rows.json'):
            raise ValueError('Prepared data identity changed')
        for path,record in C.read(OUT/'data_source_manifest.json').items():
            if C.sha(path)!=record['sha256']:raise ValueError('Frozen data dependency changed: '+path)
        if (OUT/'data_render_receipt.json').exists():
            raise FileExistsError('Preserve completed physical observations')
        # Match the existing A.Engine Windows runtime setup. Importing torch
        # initializes its CUDA DLLs; CuPy's CUDA12 cublas package also needs its
        # directory registered for the whole renderer lifetime, not a temporary
        # handle discarded before the first background pulse matrix multiply.
        import torch
        cuda_dll=Path(torch.__file__).parent.parent/'nvidia/cublas/bin'
        if os.name=='nt' and cuda_dll.is_dir():
            cuda_dll_handle=os.add_dll_directory(str(cuda_dll))
            os.environ['PATH']=str(cuda_dll)+os.pathsep+os.environ.get('PATH','')
        S,R,G,B=frozen_imports()
        rows=C.read(OUT/'scene_rows.json');validate_spec(rows)
        sensor,query=B.poses(-10.)
        by_background=defaultdict(list)
        for split,split_rows in rows.items():
            folder=OUT/'data'/split;k=cfg['train_K'] if split.endswith('train') else cfg['eval_K']
            path=folder/'hist.npy'
            if path.exists():raise FileExistsError('Preserve partial histogram payload: '+str(path))
            histmaps[split]=np.lib.format.open_memmap(path,mode='w+',dtype=np.int32,shape=(len(split_rows),k,16,8,8,16))
            for row in split_rows:by_background[row['background_id']].append((split,row))
        ambient=None;parity=[];backends=[];draws=0
        for bgid,items in sorted(by_background.items()):
            stage.check()
            bg=items[0][1]['background_boxes']
            renderer=G.ExpectedRenderer(sensor,bg)
            engines.append(renderer)
            try:
                current_ambient=renderer.ambient.copy()
                if ambient is None:ambient=current_ambient
                else:np.testing.assert_array_equal(ambient,current_ambient)
                backends.append(dict(background_id=bgid,metadata=renderer.metadata))
                # This is the genuinely target-absent first-visible background.
                absent=renderer.background_expected(return_device=False,pose_batch=16)
                unique={r['physical_key']:r for _,r in items if r['presence']}
                keys=list(unique)
                expectations={}
                for begin,endpoints in renderer.iter_render([unique[key]['target_box'] for key in keys],candidate_batch=4,pose_batch=16,deadline_check=stage.check):
                    for j in range(len(endpoints)):
                        key=keys[begin+j];rho=unique[key]['rho']
                        expectations[key]=endpoints[j,0]+rho/G.ENDPOINT_RHO*(endpoints[j,1]-endpoints[j,0])
                # One foreground plus removed-world CPU parity per family instance.
                checkrows=[dict(boxes=bg), dict(boxes=unique[keys[0]]['boxes'])]
                checkmeans=[absent,expectations[keys[0]]]
                for row,mean in zip(checkrows,checkmeans):
                    stage.check();reference=R.expected(dict(poses=sensor[[0,3,13,15]],boxes=row['boxes']))
                    difference=float(np.max(np.abs(mean[[0,3,13,15]]-reference['expectation'])))
                    np.testing.assert_allclose(mean[[0,3,13,15]],reference['expectation'],atol=1e-8,rtol=1e-11)
                    np.testing.assert_array_equal(ambient[[0,3,13,15]],reference['ambient'])
                    parity.append(dict(background_id=bgid,present=len(row['boxes'])>len(bg),max_abs=difference))
                for split,row in items:
                    stage.check();mean=expectations[row['physical_key']] if row['presence'] else absent
                    for rep in range(histmaps[split].shape[1]):
                        histmaps[split][row['scene_id'],rep]=R.sample(mean,ambient,sampling_seed(row,rep))[0]
                        draws+=1
            finally:
                renderer.close();engines.remove(renderer)
            print('RENDERED_BACKGROUND',bgid,'world_rows',len(items),'draws',draws,flush=True)
        receipts={}
        for split,hist in histmaps.items():
            stage.check();hist.flush();folder=OUT/'data'/split
            with np.load(folder/'geometry.npz',allow_pickle=False) as z:
                cats=z['frame_category'];category=z['category']
            receipts[split]=materialize(hist,ambient,sensor,query,cats,folder,stage.check)
            np.savez_compressed(folder/'physics.npz',hist=hist,ambient=ambient,sensor=sensor,public_query=query,
                category=category,frame_category=cats,scene_ids=np.arange(len(hist)),
                scene_uids=np.asarray([r['scene_uid'] for r in rows[split]]))
            payloads={name:C.sha(folder/name) for name in ('physics.npz','histories.npy','transforms.npy','length.npy','labels.npy','mask.npy','weights.npy','rows.npz')}
            if split in ('cal','validation'):payloads['transforms_yaw3.npy']=C.sha(folder/'transforms_yaw3.npy')
            C.save(folder/'receipt.json',dict(status='COMPLETE',**receipts[split],payload_sha256=payloads,
                row_order='scene_id,replica,frame3..15; reshape(scene_count,K,13,2)',
                observations_only='histories/transforms/length; truth confined to labels/mask/weights and evaluator geometry'))
        for key in ('raw_weight_mass','normalized_weight_mass'):
            if receipts['ordinary_train'][key]!=receipts['cf_train'][key]:
                raise ValueError('Equal-size augmentation supervision mass differs')
        stage.check()
        result=stage.finish('COMPLETE',backend='FP64 CUDA expected renderer / CPU aggregate Skellam; inherited FP16 normalization',
            scene_rows_sha256=C.sha(OUT/'scene_rows.json'),draws=draws,renderer_parity=parity,background_backends=backends,splits=receipts)
        C.save(OUT/'data_render_receipt.json',result)
        print(json.dumps(dict(status='COMPLETE',seconds=result['seconds'],splits=receipts),ensure_ascii=False),flush=True)
    except Exception as exc:
        stage.finish('FAILED',error=repr(exc))
        raise
    finally:
        histmaps.clear()
        for engine in engines:engine.close()
        if cuda_dll_handle is not None:cuda_dll_handle.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','render'))
    args=parser.parse_args()
    (prepare if args.stage=='prepare' else render)()
