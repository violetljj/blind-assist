"""Authoring-only fresh S-ensemble confirmation source; never sample or score.

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

SPLITS = ('cal', 'hold')
K = 2
PHOTON_PREFIX = 20261010107
DEFAULT_OUT = C.ROOT/'artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010'
PASS_LAYERS = ('0-5cm', '5-10cm', '10-20cm', '20-35cm')
FAMILIES = {
    'cal': ('sconfirm_reverse_step_columns', 'sconfirm_detached_side_pockets'),
    'hold': ('sconfirm_side_zigzag_slabs', 'sconfirm_upper_lower_shingled_blocks',
             'sconfirm_offset_shelf_triplets', 'sconfirm_side_braced_notches'),
}
PARAMETERS = {
    'cal': dict(width=(.38917, .84917), thickness=(.01417, .09417), rho=(.16117, .64117),
                edge=(.03017, .05417), protrusion_width=(.19417, .27417),
                protrusion_depth=(.07417, .11417), protrusion_halfheight=(.07917, .11917),
                vertical_height=(.35017, .36017), sign_halfheight=.07917,
                contact_inner=(.28017, .22017), pass_gap=(.02417, .07417, .16417, .27417),
                clear_gap=(.44417, .56417)),
    'hold': dict(width=(.46923, .92923), thickness=(.02023, .10423), rho=(.24123, .58123),
                 edge=(.04023, .06423), protrusion_width=(.22423, .31423),
                 protrusion_depth=(.08423, .12423), protrusion_halfheight=(.08023, .12023),
                 vertical_height=(.35223, .36223), sign_halfheight=.08023,
                 contact_inner=(.27023, .21023), pass_gap=(.03423, .08423, .18423, .32423),
                 clear_gap=(.48423, .60423)),
}


def backgrounds(split):
    """Six new physical arrangements with two mirror instances per family."""
    result = []
    family_all = [f for s in SPLITS for f in FAMILIES[s]]
    for family in FAMILIES[split]:
        fi = family_all.index(family)
        for instance in range(2):
            floor_y = 1.61717+.00323*fi
            floor = D.box([-8, floor_y, -8], [8, floor_y+.18723, 9], .34317+.01923*instance)
            wall_z = 5.17123+.13717*fi+.19723*instance
            wall = D.box([-8, -3, wall_z], [8, floor_y, wall_z+.20717], .43723+.01317*fi)
            rho = .30717+.02723*instance+.00917*fi
            # Each family has a different arrangement/count, not only a label.
            if fi == 0:
                near = [D.box([.78117, -.51323, 1.37117], [.95723, -.11117, 1.64723], rho),
                        D.box([.99717, .31723, 1.74717], [1.27323, .91317, 2.09323], rho),
                        D.box([-.98723, .97317, 2.31723], [-.74717, 1.37323, 2.66317], rho)]
            elif fi == 1:
                near = [D.box([.75117, -.43323, 1.41717], [1.17723, -.24317, 2.53323], rho),
                        D.box([1.10717, -.24317, 2.06723], [1.30723, .82317, 2.37723], rho),
                        D.box([-.93723, .65317, 1.69323], [-.73717, 1.31323, 2.18317], rho)]
            elif fi == 2:
                near = [D.box([.77117, -.48323, 1.33317], [.95723, .29317, 1.74323], rho),
                        D.box([.95723, -.48323, 1.74323], [1.24317, .29317, 1.95323], rho),
                        D.box([-.97323, .45317, 2.14323], [-.76317, 1.17323, 2.63317], rho),
                        D.box([-1.24317, .45317, 2.63317], [-.76317, 1.17323, 2.83317], rho)]
            elif fi == 3:
                near = [D.box([.74717, -.55723, 1.34717], [1.13723, -.30717, 1.92323], rho),
                        D.box([.93717, .87323, 1.53717], [1.31723, 1.23317, 2.27323], rho),
                        D.box([-.94723, .17317, 2.11723], [-.72717, .75323, 2.63317], rho),
                        D.box([-1.25723, -.47717, 2.77323], [-.81717, -.20723, 3.14317], rho)]
            elif fi == 4:
                near = [D.box([.75717, -.47723, 1.32317], [1.27723, -.29317, 2.25323], rho),
                        D.box([.81717, .46323, 1.61317], [1.32723, .64717, 2.74323], rho),
                        D.box([-.99723, .99717, 1.93323], [-.73717, 1.18723, 3.17317], rho)]
            else:
                near = [D.box([.73717, -.48323, 1.35717], [.94723, .81317, 1.59323], rho),
                        D.box([.94723, .62317, 1.59323], [1.26717, .81317, 2.48723], rho),
                        D.box([-1.18723, .17317, 2.11723], [-.74717, .37323, 3.07317], rho),
                        D.box([-1.18723, -.43717, 2.11723], [-.98717, .17317, 2.31723], rho),
                        D.box([1.17717, -.42323, 2.91317], [1.37723, .25317, 3.17723], rho)]
            result.append(dict(background_id=(0 if split == 'cal' else 4)+len(result),
                background_family=family, split=split, instance=instance,
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
            rows.append(dict(scene_id=len(rows), scene_uid=f'sconfirm-20261010/{split}/{len(rows)}',
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
        expected = 768 if split == 'cal' else 1536
        assert len(rows) == len({r['physical_key'] for r in rows}) == expected
        result[split] = rows
    assert not ({r['physical_key'] for r in result['cal']} & {r['physical_key'] for r in result['hold']})
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
    rows = scenes()
    inventory, oldkeys, oldfamilies, oldbg = inventories(out)
    required_runs = ('cnh-cost-v2-holdout-dev-20261010', 'cnh-task-cost-retrain-dev-20261010')
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
    assert not splitdims['cal'] & splitdims['hold'] and not splitrhos['cal'] & splitrhos['hold']
    inventory['target_parameter_audit'] = dict(prior_dimensions=sorted(olddims),prior_rhos=sorted(oldrhos),
        method='Exact target_box axis extents and target_thickness_m, rounded 1e-10m; target rho rounded 1e-10. Metadata lacking target_box cannot establish a scalar-only exclusion proof.')
    plan = dict(task='CNH_S_ENSEMBLE_CONFIRM_DEV_20261010',status='FROZEN_SOURCE_BEFORE_RENDER_OR_FORWARD',
        lane='FORMAL_CONFIRMATORY_DEVELOPMENT', frozen_spec=frozen_spec, frozen_spec_sha256=C.sha(frozen_spec_file),
        seeds=[2026100955,2026100956,2026100957], replicas=K, training=0, protected_access=0,
        budgets_seconds=dict(render_forward_gpu=1200,evaluation_audit_integration_command_wall=1200),
        source_parameters=PARAMETERS,background_specs={s:backgrounds(s) for s in SPLITS},cohorts=info,
        arms=['M3','old5','both955','Eensemble_cost125','S955_cost125','S956_cost125','S957_cost125','Sensemble_cost125','Sensemble_tau_transfer'],
        near_pass_max_gap_m=.10,near_pass_light_weight=.25,near_pass_strong_weight=1.,far_pass_and_clear_any_notification_weight=1.,
        strong_archive='Every new arm strong slot equals original old5; additions light only on nonstrong slots; strong-to-light=0',
        timely_frames=list(range(3,14)),late_frames=[14,15],notifier='unchanged gap1 joint max level',
        workpoints=[1.25],calibration='All new-cal negative score ties enumerated without monotonic assumption; lowest feasible threshold under old5 weighted cost*1.25; no contact benefit selection',
        tau_transfer='Exact sealed retraining S ensemble secondary cut 2.4142577648162846, not rounded display 2.41426; drift description only',
        split_access_order='Render/read cal and seal cuts before render/read hold',
        strong_signal=dict(a='Total timely net>=40/512*actual contact events (hold >=80/1024), each height >=2% (>=11/512)',
            b='far+clear notifications<=old5 hold*1.10',c='weighted cost<=old5 hold*1.25*1.05',
            d='Each S seed timely total net>=half(a), hold >=40/1024'),
        target_support='Inherited cumulative first-visible renderer quadrature sub16 f0..current, positive ray weight, first surface before background and raw bin0..127; evaluation only',
        failure='No source switching; preserve failures and protocol; stop affected work at applicable cap',
        inference_boundary='Only inherited public photon histories/ambient/query/poses/47D readout features enter models; author geometry/class/support evaluate only',
        clustering='Physical scene cluster; K/frame/seed are not independent samples',
        independence_scope='All key-bearing named permitted CNH metadata inventories including v2 and task retrain; non-AABB/protected identity entries explicit, not global-world exclusion')
    if (out/'PLAN.json').exists():
        for name, expected in [('PLAN.json',plan),('scene_rows.json',rows),('background_rows.json',{s:backgrounds(s) for s in SPLITS}),('inventory.json',inventory)]:
            assert C.read(out/name) == json.loads(json.dumps(expected)), 'Frozen authoring metadata changed during same-source repair'
    else:
        for name, value in [('PLAN.json',plan),('scene_rows.json',rows),('background_rows.json',{s:backgrounds(s) for s in SPLITS}),('inventory.json',inventory)]:
            save_new(out/name,value)
    protocol = '# S ensemble frozen confirmation source\n\n2026-10-10; formal confirmatory Development; frozen models; no training.\n\n'+json.dumps(plan,ensure_ascii=False,indent=2)+'\n'
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
        source_sha256=C.sha(Path(__file__)),inherited_support_sha256=C.sha(Path(T.__file__)),plan_sha256=C.sha(out/'PLAN.json'),
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
