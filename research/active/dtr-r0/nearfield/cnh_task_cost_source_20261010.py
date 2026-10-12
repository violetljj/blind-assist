"""New task-cost train/cal/hold authoring. Prepare performs no rendering or fit.

All truth, support and cost classes are privileged authoring/evaluation metadata.
Labels become targets/sample weights only; none is a feature or model input.
"""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import time

import numpy as np
import cnh_counterfactual_common_dev as C
import cnh_counterfactual_data_dev as D
from cnh_cost_v2_source_20261010 import inventories, save_new, categories

SPLITS = ('train', 'cal', 'hold')
K = 2
PHOTON_PREFIX = 2026101091
DEFAULT_OUT = C.ROOT/'artifacts.local/work/cnh-task-cost-retrain-dev-20261010'
PASS_LAYERS = ('0-5cm', '5-10cm', '10-20cm', '20-35cm')
FAMILIES = {
    'train': ('task_cranked_side_pillars', 'task_twin_recessed_ledges',
              'task_side_corner_blocks', 'task_split_longitudinal_fins',
              'task_side_tier_posts', 'task_high_low_stagger',
              'task_nested_side_notches', 'task_detached_side_rails'),
    'cal': ('task_recessed_cross_blocks', 'task_double_side_shelves'),
    'hold': ('task_side_alternating_lobes', 'task_parallel_offset_panels'),
}
PARAMETERS = {
    'train': dict(width=(.353, .813), thickness=(.013, .083), rho=(.143, .683),
                  edge=(.027, .057), protrusion_width=(.183, .263),
                  protrusion_depth=(.063, .123), contact_inner=(.283, .233),
                  pass_gap=(.023, .073, .153, .283), clear_gap=(.423, .543)),
    'cal': dict(width=(.397, .857), thickness=(.017, .097), rho=(.183, .603),
                edge=(.033, .053), protrusion_width=(.203, .277),
                protrusion_depth=(.073, .113), contact_inner=(.277, .217),
                pass_gap=(.027, .077, .147, .297), clear_gap=(.457, .577)),
    'hold': dict(width=(.437, .897), thickness=(.023, .107), rho=(.223, .563),
                 edge=(.039, .061), protrusion_width=(.217, .307),
                 protrusion_depth=(.087, .127), contact_inner=(.267, .207),
                 pass_gap=(.037, .087, .187, .327), clear_gap=(.487, .607)),
}


def backgrounds(split):
    """Twelve distinct geometry families; all background surfaces strict clear.

    Different family numbers select different arrangements, not merely names.
    Side signatures lie outside +/-0.30m corridor and below 0.4m expanded band.
    """
    result = []
    allfamilies = [x for s in SPLITS for x in FAMILIES[s]]
    offset = {'train': 0, 'cal': 16, 'hold': 20}[split]
    for family in FAMILIES[split]:
        fi = allfamilies.index(family)
        for instance in range(2):
            floor = D.box([-8, 1.603+.002*fi, -8], [8, 1.803+.002*fi, 9], .327+.011*instance)
            wallz = 5.43+.067*fi+.173*instance
            wall = D.box([-8, -3, wallz], [8, 1.603+.002*fi, wallz+.193], .413+.007*fi)
            rho = .293+.023*instance+.005*fi
            near = []
            # Twelve fixed patterns use varied count, y tier, depth and side.
            for j in range(2+fi % 3):
                x = .783+.041*((fi+2*j) % 5)
                y = -.47+.293*((fi+j) % 6)
                z = 1.313+.257*j+.031*fi
                b = D.box([x, y, z], [x+.143+.017*((fi+j)%4),
                            y+.167+.029*((fi+2*j)%3), z+.233+.071*((fi+j)%3)], rho)
                near.extend(D.mirror_boxes([b], -1 if (fi+j)%2 else 1))
            near = D.mirror_boxes(near, 1 if instance == 0 else -1)
            result.append(dict(background_id=offset+len(result), background_family=family,
                split=split, instance=instance, boxes=[floor, wall, *near]))
    return result


def target(split, shape, height, side, rho_index, size, inner):
    p = PARAMETERS[split]
    y = .10 if height == 0 else .65
    thickness = p['thickness'][size]
    if shape == 'horizontal':
        width, depth = p['width'][size], thickness
        yl, yh = y-thickness/2, y+thickness/2
    elif shape == 'vertical':
        width = depth = thickness
        yl, yh = ((-.023, .323) if height == 0 else (.467, .823))
    elif shape == 'protrusion':
        width, depth = p['protrusion_width'][size], p['protrusion_depth'][size]
        yl, yh = y-(.077, .117)[size], y+(.077, .117)[size]
        thickness = depth
    elif shape == 'sign_edge':
        edge = p['edge'][size]
        width, depth = (p['width'][0], edge) if size == 0 else (edge, p['width'][0])
        yl, yh = y-.077, y+.077
        thickness = edge
    else:
        raise ValueError(shape)
    xl, xh = (inner, inner+width) if side == 1 else (-inner-width, -inner)
    return D.box([xl, yl, .65], [xh, yh, .65+depth], p['rho'][rho_index]), thickness


def scenes():
    result = {}
    for split in SPLITS:
        rows = []
        def append(bg, shape, height, side, ri, size, placement, layer, inner):
            t, thickness = target(split, shape, height, side, ri, size, inner)
            boxes = [t, *bg['boxes']]
            rows.append(dict(scene_id=len(rows), scene_uid=f'task-cost-20261010/{split}/{len(rows)}',
                split=split, shape_family=shape, group=height, side=side, rho=t['rho'],
                placement=placement, presence=True, size_variant=size, target_box=t,
                target_thickness_m=thickness, lateral_gap_m=inner-.30, pass_layer=layer,
                dark_thin=bool(ri == 0 and size == 0 and shape != 'protrusion'),
                background_id=bg['background_id'], background_family=bg['background_family'],
                background_boxes=bg['boxes'], boxes=boxes, physical_key=D.physical_key(boxes),
                occurrence_draw_id=0))
        for placement in ('contact', 'clear'):
            for bg, shape, h, side, ri, size in itertools.product(
                    backgrounds(split), D.SHAPES, range(2), (-1, 1), range(2), range(2)):
                inner = (PARAMETERS[split]['contact_inner'][size] if placement == 'contact'
                         else .30+PARAMETERS[split]['clear_gap'][size])
                append(bg, shape, h, side, ri, size, placement, '', inner)
        for li, layer in enumerate(PASS_LAYERS):
            for bg, shape, h, side in itertools.product(backgrounds(split), D.SHAPES, range(2), (-1, 1)):
                ri = (bg['background_id']+h+(side == 1)) % 2
                size = (D.SHAPES.index(shape)+bg['instance']+h+li) % 2
                append(bg, shape, h, side, ri, size, 'pass', layer, .30+PARAMETERS[split]['pass_gap'][li])
        expected = 3072 if split == 'train' else 768
        assert len(rows) == len({r['physical_key'] for r in rows}) == expected
        result[split] = rows
    for a, b in itertools.combinations(SPLITS, 2):
        assert not ({r['physical_key'] for r in result[a]} & {r['physical_key'] for r in result[b]})
    return result


def sampling_seed(row, replica):
    words = np.frombuffer(bytes.fromhex(row['physical_key']), dtype='<u4').tolist()
    return int(np.random.SeedSequence([PHOTON_PREFIX, *words, int(replica)]).generate_state(1)[0])


def ray_box_distances(sensor, rays, boxes):
    """CPU first surface intersection, [frames,rays,boxes], including inside rays."""
    directions = np.einsum('fij,rj->fri', sensor[:,:3,:3], rays)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origin = sensor[:,:3,3][:,None,None,:]
    d = directions[:,:,None,:]
    lo = np.asarray([b['lo'] for b in boxes])[None,None]
    hi = np.asarray([b['hi'] for b in boxes])[None,None]
    parallel = np.abs(d) < 1e-14
    safe = np.where(parallel, 1., d)
    a, b = (lo-origin)/safe, (hi-origin)/safe
    enter = np.where(parallel, -np.inf, np.minimum(a,b)).max(-1)
    leave = np.where(parallel, np.inf, np.maximum(a,b)).min(-1)
    distance = np.where(enter > 1e-10, enter, leave)
    outside = (parallel & ((origin < lo)|(origin > hi))).any(-1)
    valid = ~outside & (leave >= np.maximum(enter,0.)) & (distance > 1e-10) & np.isfinite(distance)
    return np.where(valid, distance, np.inf)


def target_support(rows, sensor):
    """Causal renderer quadrature support, evaluator-only. No photons sampled.

    Every target is wholly in front of background slabs/posts or above floor.
    Verify this first-hit separation before caching support across backgrounds.
    """
    S, _, G, _ = D.frozen_imports()
    rays, weights = S.angular_rays(16)
    rays = np.asarray(rays).reshape(-1,3)
    positive_ray_weight = np.asarray(weights).reshape(-1) > 0
    parameters, _ = S.nominal_parameters()
    now = np.zeros((len(rows),16,2),bool)
    cached = {}
    for i, row in enumerate(rows):
        target = row['target_box']
        assert all((b['lo'][2] > target['hi'][2]) or
                   (b['lo'][1] > target['hi'][1] and sensor[:,1,3].max() < b['lo'][1])
                   for b in row['background_boxes']), 'Cannot elide background intersection'
        key = json.dumps([target['lo'],target['hi']],separators=(',',':'))
        if key not in cached:
            t = ray_box_distances(sensor, rays, [target])[:,:,0]
            rawbin = np.floor((np.where(np.isfinite(t),t,0.)-parameters.range_zero_m)/G.SENSOR.RAW_BIN_M)
            visible = np.isfinite(t) & (rawbin >= 0) & (rawbin < 128) & positive_ray_weight[None]
            cached[key] = visible.any(-1)
        now[i,:,row['group']] = cached[key]
    cumulative = np.maximum.accumulate(now,axis=1)
    support = cumulative[:,D.FRAMES]
    cat = np.asarray([[r['placement'] if r['group']==h else 'clear' for h in range(2)] for r in rows])
    positive = support & (cat[:,None,:] == 'contact') & (D.FRAMES[None,:,None] <= 13)
    return support, now[:,D.FRAMES], positive


def prepare(out, recipe_file):
    began = time.monotonic()
    out = Path(out)
    if (out/'prepare_receipt.json').exists():
        raise FileExistsError('Prepared source is immutable')
    recipe = C.read(recipe_file)
    if not recipe or 'B' not in recipe:
        raise ValueError('Fixed B recipe must be supplied before PLAN authoring')
    rows = scenes()
    inventory, oldkeys, oldfamilies, oldbg = inventories(out)
    v2 = C.ROOT/'artifacts.local/work/cnh-cost-v2-holdout-dev-20261010/scene_rows.json'
    assert any(r['path'] == str(v2.relative_to(C.ROOT)) for r in inventory['files']), 'v2 inventory required'
    info = {}
    for split, values in rows.items():
        keys = {r['physical_key'] for r in values}
        families = {r['background_family'] for r in values}
        bgs = {D.physical_key(r['background_boxes']) for r in values}
        assert not keys & oldkeys and not families & oldfamilies and not bgs & oldbg
        info[split] = dict(scenes=len(values), unique_physical=len(keys), physical_overlap=0,
            background_geometry_overlap=0, background_family_overlap=0, families=sorted(families),
            physical_contact_per_height=[sum(r['placement']=='contact' and r['group']==h for r in values) for h in range(2)],
            contact_events_per_height=[K*sum(r['placement']=='contact' and r['group']==h for r in values) for h in range(2)],
            pass_layers=dict(Counter(r['pass_layer'] for r in values if r['placement']=='pass')),
            clear_physical=sum(r['placement']=='clear' for r in values))
    plan = dict(task='CNH_TASK_COST_RETRAIN_DEV_20261010', status='FROZEN_SOURCE_BEFORE_RENDER_OR_FORWARD',
        lane='EXPLORE', seeds=[2026100955,2026100956,2026100957], replicas=K,
        budgets_seconds=dict(data_render_gpu=1200,training_gpu_total=3600,B_gpu_total=2700,
                             B_single_seed=900,evaluation_audit_integration_command_wall=1500),
        source_parameters=PARAMETERS, background_specs={s:backgrounds(s) for s in SPLITS}, cohorts=info,
        training_recipe=recipe, recipe_sha256=C.sha(recipe_file),
        arms=['old5_fixed_strong','both_original','E_no_training_ordinary_mean',
              'S_HGB_three_seeds_and_mean','B_ordinary_finetune_three_seeds_and_mean'],
        near_pass_max_gap_m=.10,near_pass_light_weight=.25,near_pass_strong_weight=1.,
        far_pass_and_clear_any_notification_weight=1.,
        strong_archive='Every slot strong equals original old5; additions light only on nonstrong slots; strong-to-light=0',
        target_support='Cumulative first-visible renderer quadrature sub16 (64 zones times256 rays) f0..current, positive ray weight, target first surface before background and raw bin0..127; evaluator-only geometry, labels only',
        slot_label='contact query f3..13 and causal target_support is positive; pass/clear negative; contact before support and f14/15 excluded from fit',
        loss_weight='near-pass negative .25, far-pass/clear negative1; positive totalweight balanced to negative totalweight; perheight training support only',
        timely_frames=list(range(3,14)),late_frames=[14,15],notifier='unchanged gap1 joint max level',
        workpoints=[1.,1.25],calibration='New cal negatives only: enumerate all negative score ties without monotonic assumption, lowest feasible threshold under old5 weighted cap times1 or1.25; no contact benefit selection',
        split_access_order='Render train/cal and fit train; cal selection sealed; only then render/read hold',
        strong_signal='Arm main workpoint HEAD+BODY timely net>=10/512, far+clear notifications<=old5*1.10, hold weighted cost<=old5*1.05, seed directions consistent; E prioritized if eligible',
        failure='Preserve same source on failure, no source switch; profile before long training; fixed recipe no hold tuning; stop affected stage at cumulative cap',
        inference_boundary='Only public photon histories, ambient, public query/poses and frozen readout features enter models; author geometry/category/support used solely generation and target/sampleweight',
        clustering='Physical scene is cluster; K/frame/seed never independent samples',
        independence_scope=inventory.get('independence_scope','All key-bearing permitted named CNH metadata catalogs plus explicit non-AABB/protected entries; not a global historical-world exclusion proof'))
    resuming = (out/'PLAN.json').exists()
    if resuming:
        assert C.read(out/'PLAN.json') == json.loads(json.dumps(plan)), 'No frozen PLAN switching during repair'
        assert C.read(out/'scene_rows.json') == rows
        assert C.read(out/'background_rows.json') == {s:backgrounds(s) for s in SPLITS}
        assert C.read(out/'inventory.json') == inventory
    else:
        save_new(out/'PLAN.json',plan)
        save_new(out/'inventory.json',inventory)
        save_new(out/'scene_rows.json',rows)
        save_new(out/'background_rows.json',{s:backgrounds(s) for s in SPLITS})
    protocol = '# CNH task-cost retraining frozen protocol\n\n2026-10-10; EXPLORE; new training and new cohorts.\n\n'+json.dumps(plan,ensure_ascii=False,indent=2)+'\n'
    for name in ('PROTOCOL_FROZEN.md','protocol_snapshot.md'):
        p = out/name
        if not p.exists():
            p.write_bytes(protocol.encode('utf8'))
        assert p.read_text(encoding='utf8') == protocol
    _, _, _, B = D.frozen_imports()
    sensor, query = B.poses(-10.)
    for split, values in rows.items():
        category = categories(values,sensor)
        support, visible, positive = target_support(values,sensor)
        folder = out/'data'/split
        folder.mkdir(parents=True,exist_ok=True)
        geom = folder/'geometry.npz'
        if geom.exists():
            with np.load(geom) as existing:
                for key,val in [('category',category),('target_support',support),('target_visible_now',visible),('label_positive',positive)]:
                    np.testing.assert_array_equal(existing[key],val)
        else:
            np.savez_compressed(geom,category=category,sensor=sensor,public_query=query,
                scene_ids=np.arange(len(values)),scene_uids=np.array([r['scene_uid'] for r in values]),
                target_support=support,target_visible_now=visible,label_positive=positive)
        info[split]['geometry_sha256']=C.sha(geom)
        info[split]['positive_slots_per_height']=positive.sum(axis=(0,1)).tolist()
        info[split]['contact_scenes_without_support_per_height']=[sum(r['placement']=='contact' and r['group']==h and not support[i,:,h].any() for i,r in enumerate(values)) for h in range(2)]
    receipt=dict(status='COMPLETE',seconds=time.monotonic()-began,cohorts=info,replicas=K,
        source_sha256=C.sha(Path(__file__)),plan_sha256=C.sha(out/'PLAN.json'),
        protocol_sha256=C.sha(out/'protocol_snapshot.md'),inventory_sha256=C.sha(out/'inventory.json'),
        scene_rows_sha256=C.sha(out/'scene_rows.json'),new_raw=0,predictions=0,training=0,protected_access=0,
        inventory_file_count=inventory['file_count'],unique_prior_physical_keys=inventory['unique_physical_keys'])
    save_new(out/'prepare_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--recipe',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.out,args.recipe)
