"""Authoring-only fresh S-ensemble tradeoff source; never sample or score.

Geometry/support are privileged evaluation metadata. Frozen observations and
47-dimensional model features are implemented by the inherited byte-frozen
modules, not this source. No former observation/outcome payload is read here.
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
import cnh_task_cost_source_20261010 as T
from cnh_cost_v2_source_20261010 import inventories, save_new, categories

SPLITS = ('hold',)
K = 2
PHOTON_PREFIX = 20261010113
DEFAULT_OUT = C.ROOT/'artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010'
PASS_LAYERS = ('0-5cm', '5-10cm', '10-20cm', '20-35cm')
FAMILIES = {
    'hold': ('stradeoff_forward_side_risers', 'stradeoff_split_side_cantilevers',
             'stradeoff_nested_recess_frames', 'stradeoff_alternating_cross_piers'),
}
PARAMETERS = {
    'hold': dict(width=(.44937, .90937), thickness=(.02137, .10137), rho=(.20137, .62137),
                 edge=(.03537, .06537), protrusion_width=(.21137, .30137),
                 protrusion_depth=(.08137, .12137), protrusion_halfheight=(.08337, .12337),
                 vertical_height=(.35437, .36437), sign_halfheight=.08337,
                 contact_inner=(.26937, .20937), pass_gap=(.03137, .08137, .18137, .32137),
                 clear_gap=(.48137, .60137)),
}


def backgrounds(split):
    """Four fixed new arrangements, two mirrored physical instances each."""
    if split != 'hold':
        raise ValueError(split)
    result = []
    for fi, family in enumerate(FAMILIES[split]):
        for instance in range(2):
            floor_y = 1.61937+.00337*fi
            floor = D.box([-8, floor_y, -8], [8, floor_y+.18337, 9], .34937+.01937*instance)
            wall_z = 5.21937+.13937*fi+.19337*instance
            wall = D.box([-8, -3, wall_z], [8, floor_y, wall_z+.20937], .43937+.01337*fi)
            rho = .30937+.02937*instance+.00937*fi
            if fi == 0:
                near = [D.box([.76337, -.52337, 1.29937], [.98337, -.12337, 1.57337], rho),
                        D.box([.76337, -.12337, 1.57337], [1.20337, .25337, 1.87337], rho),
                        D.box([.99337, .25337, 1.87337], [1.31337, .89337, 2.15337], rho),
                        D.box([-.99337, .95337, 2.28337], [-.75337, 1.33337, 2.69337], rho)]
            elif fi == 1:
                near = [D.box([.74937, -.51337, 1.34937], [1.29937, -.31337, 2.11937], rho),
                        D.box([1.09937, -.31337, 1.94937], [1.29937, .25337, 2.11937], rho),
                        D.box([-.95937, .55337, 1.77937], [-.73937, 1.29337, 2.43937], rho),
                        D.box([-1.23937, .91337, 2.43937], [-.73937, 1.12337, 2.77937], rho)]
            elif fi == 2:
                near = [D.box([.75737, -.54337, 1.38937], [.94737, .81337, 1.61937], rho),
                        D.box([.94737, -.54337, 1.61937], [1.32737, -.31337, 2.48937], rho),
                        D.box([1.09737, -.31337, 2.25937], [1.32737, .81337, 2.48937], rho),
                        D.box([.94737, .58337, 1.61937], [1.09737, .81337, 2.25937], rho),
                        D.box([-1.17937, .99337, 2.11937], [-.75937, 1.22337, 2.74937], rho)]
            else:
                near = [D.box([.74737, -.53337, 1.33937], [.92737, .83337, 1.59937], rho),
                        D.box([.92737, .07337, 1.59937], [1.31737, .28337, 2.10937], rho),
                        D.box([-.94737, .17337, 1.88937], [-.73737, 1.33337, 2.14937], rho),
                        D.box([-1.29737, .95337, 2.14937], [-.94737, 1.17337, 2.65937], rho),
                        D.box([1.09737, -.42337, 2.71937], [1.33737, .33337, 3.17937], rho)]
            result.append(dict(background_id=len(result), background_family=family,
                split=split, instance=instance,
                boxes=[floor, wall, *D.mirror_boxes(near, 1 if instance == 0 else -1)]))
    return result



def target(split, shape, height, side, rho_index, size, inner):
    p = PARAMETERS[split]
    y = .10 if height == 0 else .65
    thickness = p['thickness'][size]
    if shape == 'horizontal':
        width, depth = p['width'][size], thickness
        halfheight = thickness/2
    elif shape == 'vertical':
        width = depth = thickness
        halfheight = p['vertical_height'][height]/2
    elif shape == 'protrusion':
        width, depth = p['protrusion_width'][size], p['protrusion_depth'][size]
        halfheight = p['protrusion_halfheight'][size]
        thickness = depth
    elif shape == 'sign_edge':
        edge = p['edge'][size]
        width, depth = (p['width'][0], edge) if size == 0 else (edge, p['width'][0])
        halfheight = p['sign_halfheight']
        thickness = edge
    else:
        raise ValueError(shape)
    xl, xh = (inner, inner+width) if side == 1 else (-inner-width, -inner)
    return D.box([xl, y-halfheight, .65], [xh, y+halfheight, .65+depth], p['rho'][rho_index]), thickness


def scenes():
    result = {}
    for split in SPLITS:
        rows = []
        def append(bg, shape, h, side, ri, size, placement, layer, inner):
            t, thickness = target(split, shape, h, side, ri, size, inner)
            boxes = [t, *bg['boxes']]
            rows.append(dict(scene_id=len(rows), scene_uid=f'stradeoff-20261010/{split}/{len(rows)}',
                split=split, shape_family=shape, group=h, side=side, rho=t['rho'],
                placement=placement, presence=True, size_variant=size, target_box=t,
                target_thickness_m=thickness, lateral_gap_m=inner-.30, pass_layer=layer,
                dark_thin=bool(ri == 0 and size == 0 and shape != 'protrusion'),
                background_id=bg['background_id'], background_family=bg['background_family'],
                background_boxes=bg['boxes'], boxes=boxes, physical_key=D.physical_key(boxes), occurrence_draw_id=0))
        for placement in ('contact', 'clear'):
            for bg, shape, h, side, ri, size in itertools.product(
                    backgrounds(split), D.SHAPES, range(2), (-1, 1), range(2), range(2)):
                inner = PARAMETERS[split]['contact_inner'][size] if placement == 'contact' else .30+PARAMETERS[split]['clear_gap'][size]
                append(bg, shape, h, side, ri, size, placement, '', inner)
        for li, layer in enumerate(PASS_LAYERS):
            for bg, shape, h, side in itertools.product(backgrounds(split), D.SHAPES, range(2), (-1, 1)):
                ri = (bg['background_id']+h+(side == 1)) % 2
                size = (D.SHAPES.index(shape)+bg['instance']+h+li) % 2
                append(bg, shape, h, side, ri, size, 'pass', layer, .30+PARAMETERS[split]['pass_gap'][li])
        expected = 1536
        assert len(rows) == len({r['physical_key'] for r in rows}) == expected
        result[split] = rows
    return result


def sampling_seed(row, replica):
    words = np.frombuffer(bytes.fromhex(row['physical_key']), dtype='<u4').tolist()
    return int(np.random.SeedSequence([PHOTON_PREFIX, *words, int(replica)]).generate_state(1)[0])


target_support = T.target_support
ray_box_distances = T.ray_box_distances


def _target_values(value, dimensions, rhos):
    if isinstance(value, dict):
        target = value.get('target_box')
        if isinstance(target, dict) and set(('lo','hi','rho')) <= set(target):
            dimensions.update(round(float(b)-float(a), 10) for a,b in zip(target['lo'],target['hi']))
            rhos.add(round(float(target['rho']), 10))
        if isinstance(value.get('target_thickness_m'), (int,float)):
            dimensions.add(round(float(value['target_thickness_m']), 10))
        for item in value.values():
            _target_values(item, dimensions, rhos)
    elif isinstance(value, list):
        for item in value:
            _target_values(item, dimensions, rhos)


def prepare(out, frozen_spec_file):
    began = time.monotonic()
    out = Path(out)
    if (out/'prepare_receipt.json').exists():
        raise FileExistsError('Prepared source is immutable')
    frozen_spec = C.read(frozen_spec_file)
    if not frozen_spec:
        raise ValueError('Frozen model/feature/protocol specification required before authoring')
    calibration_seal_file = out/'sealed_calibration.json'
    if not calibration_seal_file.exists():
        raise FileNotFoundError('Seal pooled negative calibration before authoring new hold')
    calibration_seal = C.read(calibration_seal_file)
    if not calibration_seal:
        raise ValueError('Pooled calibration seal is empty')
    rows = scenes()
    inventory, oldkeys, oldfamilies, oldbg = inventories(out)
    required_runs = ('cnh-cost-v2-holdout-dev-20261010', 'cnh-task-cost-retrain-dev-20261010', 'cnh-s-ensemble-confirm-dev-20261010')
    paths = {r['path'].replace('\\','/') for r in inventory['files']}
    for run in required_runs:
        for name in ('scene_rows.json','background_rows.json'):
            assert f'artifacts.local/work/{run}/{name}' in paths, 'Required prior metadata not inventoried'
    olddims, oldrhos = set(), set()
    for record in inventory['files']:
        _target_values(C.read(C.ROOT/record['path']), olddims, oldrhos)
    info, splitdims, splitrhos = {}, {}, {}
    for split, values in rows.items():
        keys = {r['physical_key'] for r in values}
        families = {r['background_family'] for r in values}
        bgs = {D.physical_key(r['background_boxes']) for r in values}
        assert not keys & oldkeys and not families & oldfamilies and not bgs & oldbg
        dims, rhos = set(), set()
        _target_values(values, dims, rhos)
        assert not dims & olddims and not rhos & oldrhos, 'Target dimensions/rho overlap named prior target metadata'
        splitdims[split], splitrhos[split] = dims, rhos
        info[split] = dict(scenes=len(values), unique_physical=len(keys), physical_overlap=0,
            background_geometry_overlap=0, background_family_overlap=0, families=sorted(families),
            target_dimensions_m=sorted(dims), target_rhos=sorted(rhos), target_parameter_overlap=0,
            physical_contact_per_height=[sum(r['placement']=='contact' and r['group']==h for r in values) for h in range(2)],
            contact_events_per_height=[K*sum(r['placement']=='contact' and r['group']==h for r in values) for h in range(2)],
            pass_layers=dict(Counter(r['pass_layer'] for r in values if r['placement']=='pass')),
            clear_physical=sum(r['placement']=='clear' for r in values))
    inventory['target_parameter_audit'] = dict(prior_dimensions=sorted(olddims),prior_rhos=sorted(oldrhos),
        method='Exact target_box axis extents and target_thickness_m, rounded 1e-10m; target rho rounded 1e-10. Metadata lacking target_box cannot establish a scalar-only exclusion proof.')
    plan = dict(task='CNH_S_ENSEMBLE_TRADEOFF_DEV_20261010', status='FROZEN_SOURCE_BEFORE_RENDER_OR_FORWARD',
        lane='EXPLORE', frozen_spec=frozen_spec, frozen_spec_sha256=C.sha(frozen_spec_file),
        pooled_calibration_seal_sha256=C.sha(calibration_seal_file),
        seeds=[2026100955,2026100956,2026100957], replicas=K, training=0, protected_access=0,
        budgets_seconds=dict(render_forward_gpu=1200,evaluation_audit_integration_command_wall=1200),
        source_parameters=PARAMETERS,background_specs={s:backgrounds(s) for s in SPLITS},cohorts=info,
        arms=['M3','old5','both955','Eensemble','S955','S956','S957','Sensemble'],
        near_pass_max_gap_m=.10,near_pass_light_weight=.25,near_pass_strong_weight=1.,far_pass_and_clear_any_notification_weight=1.,
        strong_archive='Every new arm strong slot equals original old5; additions light only on nonstrong slots; strong-to-light=0',
        timely_frames=list(range(3,14)),late_frames=[14,15],notifier='unchanged gap1 joint max level',
        calibration='Only pass/clear negative rows from cost-v2 cal/hold, retrain cal/hold, confirm cal/hold; all score ties enumerated without monotonic assumption; lowest threshold with pooled far+clear any-notification clip rate <= same-pool old5+0.030; contact benefit never used',
        split_access_order='Seal pooled negative calibration before authoring/rendering/reading fresh hold',
        decision='User promotes frozen S ensemble to simulation ToF default candidate; no binary cost gate; hold curves descriptive only; no App or bench integration',
        decision_check=dict(contact_events=1024,contact_per_height=512,far_clear_clips=1536,
            minimum_timely_rate_step=1/1024,minimum_far_clear_clip_rate_step=1/1536,
            observable_tradeoff='Paired timely rescue/loss versus far+clear any-notification clip rate, family variation and source-to-hold drift; no pass/fail gate'),
        curve='Hold threshold scans for frozen evidence are descriptive; selected deployment threshold exclusively from sealed pooled negative calibration',
        uncertainty='Physical-scene paired bootstrap and family cluster bootstrap; K/frame/seed not independent; four-family intervals descriptive',
        target_support='Inherited cumulative first-visible renderer quadrature sub16 f0..current, positive ray weight, first surface before background and raw bin0..127; evaluation only',
        failure='No source switching; preserve failures and protocol; stop affected work at applicable cap',
        inference_boundary='Only inherited public photon histories/ambient/query/poses/47D readout features enter models; author geometry/class/support evaluate only',
        clustering='Physical scene cluster; K/frame/seed are not independent samples',
        independence_scope='All key-bearing named permitted CNH metadata inventories including v2, task retrain and confirm; non-AABB/protected identity entries explicit, not global-world exclusion')
    if (out/'PLAN.json').exists():
        for name, expected in [('PLAN.json',plan),('scene_rows.json',rows),('background_rows.json',{s:backgrounds(s) for s in SPLITS}),('inventory.json',inventory)]:
            assert C.read(out/name) == json.loads(json.dumps(expected)), 'Frozen authoring metadata changed during same-source repair'
    else:
        for name, value in [('PLAN.json',plan),('scene_rows.json',rows),('background_rows.json',{s:backgrounds(s) for s in SPLITS}),('inventory.json',inventory)]:
            save_new(out/name,value)
    protocol = '# S ensemble frozen tradeoff source\n\n2026-10-10; EXPLORE; frozen models; no training.\n\n'+json.dumps(plan,ensure_ascii=False,indent=2)+'\n'
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
        arrays = dict(category=category,sensor=sensor,public_query=query,scene_ids=np.arange(len(values)),
            scene_uids=np.array([r['scene_uid'] for r in values]),target_support=support,target_visible_now=visible,label_positive=positive)
        if geom.exists():
            with np.load(geom) as existing:
                for key,val in arrays.items():
                    np.testing.assert_array_equal(existing[key],val)
        else:
            np.savez_compressed(geom,**arrays)
        info[split]['geometry_sha256']=C.sha(geom)
        info[split]['positive_slots_per_height']=positive.sum(axis=(0,1)).tolist()
        info[split]['contact_scenes_without_support_per_height']=[sum(r['placement']=='contact' and r['group']==h and not support[i,:,h].any() for i,r in enumerate(values)) for h in range(2)]
    receipt=dict(status='COMPLETE',seconds=time.monotonic()-began,cohorts=info,replicas=K,
        source_sha256=C.sha(Path(__file__)),pooled_calibration_seal_sha256=C.sha(calibration_seal_file),inherited_support_sha256=C.sha(Path(T.__file__)),plan_sha256=C.sha(out/'PLAN.json'),
        protocol_sha256=C.sha(out/'protocol_snapshot.md'),inventory_sha256=C.sha(out/'inventory.json'),scene_rows_sha256=C.sha(out/'scene_rows.json'),
        new_raw=0,predictions=0,training=0,protected_access=0,inventory_file_count=inventory['file_count'],unique_prior_physical_keys=inventory['unique_physical_keys'])
    save_new(out/'prepare_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--freeze-spec',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.out,args.freeze_spec)
