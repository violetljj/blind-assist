"""Fresh authoring-only cost-v2 fixtures; prepare never renders or scores.

Far-pass is an evaluator intent: its physical boxes are clear under the old
10cm-expanded query. That old analytic geometry is checked, not relabelled in
the models. Scene/category metadata never enters frozen inference.
"""
import argparse
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import cnh_counterfactual_common_dev as C
import cnh_counterfactual_data_dev as D

SPLITS = ('cal', 'hold')
K = 2
PHOTON_PREFIX = 2026101084
DEFAULT_OUT = C.ROOT/'artifacts.local/work/cnh-cost-v2-holdout-dev-20261010'
PASS_LAYERS = ('0-5cm', '5-10cm', '10-20cm', '20-35cm')
PARAMETERS = {
    'cal': dict(width=(.41, .87), thickness=(.015, .091), rho=(.17, .63),
                edge=(.031, .043), protrusion_width=(.197, .283),
                protrusion_depth=(.071, .109), contact_inner=(.279, .221),
                pass_gap=(.018, .067, .137, .267), clear_gap=(.40, .52)),
    'hold': dict(width=(.47, .93), thickness=(.019, .103), rho=(.21, .53),
                 edge=(.037, .049), protrusion_width=(.209, .297),
                 protrusion_depth=(.079, .117), contact_inner=(.273, .213),
                 pass_gap=(.032, .084, .173, .314), clear_gap=(.44, .56)),
}


def backgrounds(split):
    families = (('recessed_side_triplets', 'split_height_side_beams') if split == 'cal'
                else ('offset_side_lattices', 'asymmetric_side_buttresses'))
    result = []
    for family in families:
        for instance in range(2):
            rho = (.31, .55)[instance]
            floor = D.box([-8, 1.61, -8], [8, 1.79, 9], (.37, .46)[instance])
            wall = D.box([-8, -3, 4.97+.23*instance], [8, 1.61, 5.16+.23*instance], .42)
            if family == 'recessed_side_triplets':
                near = [D.box([.76, -.62, 1.29], [.94, .13, 1.54], rho),
                        D.box([.91, .42, 1.76], [1.23, 1.34, 2.01], .48),
                        D.box([-1.32, -.31, 2.31], [-.81, .82, 2.56], rho)]
            elif family == 'split_height_side_beams':
                near = [D.box([.73, -.57, 1.47], [1.34, -.36, 2.89], rho),
                        D.box([.73, 1.08, 1.66], [1.17, 1.25, 2.43], .48),
                        D.box([-1.19, .19, 2.09], [-.77, .39, 3.04], rho)]
            elif family == 'offset_side_lattices':
                near = [D.box([.79, -.49, 1.38], [.92, 1.32, 1.62], rho),
                        D.box([1.11, -.41, 2.08], [1.27, 1.27, 2.31], .48),
                        D.box([.79, .31, 1.62], [1.27, .47, 2.31], rho),
                        D.box([-1.38, -.61, 2.67], [-.83, -.36, 3.22], .48)]
            elif family == 'asymmetric_side_buttresses':
                near = [D.box([.75, -.33, 1.58], [1.09, .21, 1.91], rho),
                        D.box([.89, .62, 2.39], [1.31, 1.37, 2.73], .48),
                        D.box([-1.43, .07, 1.82], [-.74, .43, 2.08], rho),
                        D.box([-1.12, .84, 2.93], [-.74, 1.38, 3.27], .48)]
            else:
                raise ValueError(family)
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
        yl, yh = y-thickness/2, y+thickness/2
    elif shape == 'vertical':
        width = depth = thickness
        yl, yh = ((-.027, .317) if height == 0 else (.473, .817))
    elif shape == 'protrusion':
        width, depth = p['protrusion_width'][size], p['protrusion_depth'][size]
        yl, yh = y-(.081, .121)[size], y+(.081, .121)[size]
        thickness = depth
    elif shape == 'sign_edge':
        edge = p['edge'][size]
        width, depth = (p['width'][0], edge) if size == 0 else (edge, p['width'][0])
        yl, yh = y-.081, y+.081
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
            rows.append(dict(scene_id=len(rows), scene_uid=f'cost-v2-20261010/{split}/{len(rows)}',
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
                si = D.SHAPES.index(shape)
                ri = (bg['background_id']+h+(side == 1)) % 2
                size = (si+bg['instance']+h+li) % 2
                append(bg, shape, h, side, ri, size, 'pass', layer, .30+PARAMETERS[split]['pass_gap'][li])
        assert len(rows) == 768
        assert len({r['physical_key'] for r in rows}) == 768
        result[split] = rows
    assert not ({r['physical_key'] for r in result['cal']} & {r['physical_key'] for r in result['hold']})
    return result


def categories(rows, sensor):
    result = []
    for row in rows:
        intended = ['clear', 'clear']
        intended[row['group']] = row['placement']
        analytic = intended.copy()
        if row['placement'] == 'pass' and row['lateral_gap_m'] > .10:
            analytic[row['group']] = 'clear'
        for f in D.FRAMES:
            assert D.category_boxes(row['boxes'], sensor[f, :3, 3]) == analytic, row['scene_uid']
            assert D.category_boxes(row['background_boxes'], sensor[f, :3, 3]) == ['clear', 'clear']
        result.append(intended)
    return np.asarray(result)


def sampling_seed(row, replica):
    words = np.frombuffer(bytes.fromhex(row['physical_key']), dtype='<u4').tolist()
    return int(np.random.SeedSequence([PHOTON_PREFIX, *words, int(replica)]).generate_state(1)[0])


def _harvest(value, keys, families, backgrounds_found):
    if isinstance(value, dict):
        if isinstance(value.get('physical_key'), str):
            keys.add(value['physical_key'])
        family = value.get('background_family')
        if isinstance(family, str):
            families.add(family)
        boxes = value.get('boxes')
        if isinstance(boxes, list) and boxes and all(isinstance(b, dict) and set(('lo', 'hi', 'rho')) <= set(b) for b in boxes):
            keys.add(D.physical_key(boxes))
        bg = value.get('background_boxes')
        if isinstance(bg, list) and bg and all(isinstance(b, dict) and set(('lo', 'hi', 'rho')) <= set(b) for b in bg):
            backgrounds_found.add(D.physical_key(bg))
        if family and boxes and 'scene_id' not in value and 'scene_uid' not in value:
            backgrounds_found.add(D.physical_key(boxes))
        for v in value.values():
            _harvest(v, keys, families, backgrounds_found)
    elif isinstance(value, list):
        for v in value:
            _harvest(v, keys, families, backgrounds_found)


def inventories(out):
    """Read authoring/source metadata only, never observation or outcome payloads.

Exact named AABB catalogs and geometry manifests yield keys. Other inventories
are explicitly metadata-only or non-AABB; no claim of whole-history exclusion.
"""
    paths = set()
    work = C.ROOT/'artifacts.local/work'
    names = {'scene_rows.json', 'background_rows.json', 'scenes.json', 'scene_specs.json', 'backgrounds.json'}
    for run in sorted(work.glob('cnh-*')):
        if run.resolve() == out.resolve():
            continue
        for root, dirs, files in os.walk(run):
            # Protected identity docs may be referenced, but no protected payload is read.
            dirs[:] = [d for d in dirs if not any(x in d.lower() for x in ('test', '480', 'observations', 'scores', 'predictions', 'models', 'hist', 'raw', 'truth', 'cache', 'runtime', 'nfo', 'sanpo'))]
            if any(x in Path(root).name.lower() for x in ('real-head', 'protected')):
                dirs[:] = []
                continue
            for name in files:
                low = name.lower()
                if name in names or (low.endswith('.json') and ('inventory' in low or low.startswith('geometry_manifest'))):
                    paths.add(Path(root)/name)
    required = [work/'cnh-counterfactual-dev-20261009/scene_rows.json',
                work/'cnh-counterfactual-dev-20261009/background_rows.json',
                work/'cnh-frozen-e2e-20261010/scene_rows.json']
    if any(not p.exists() for p in required):
        raise FileNotFoundError('Required previous physical inventory metadata missing')
    paths.update(required)
    allkeys, allfamilies, allbg = set(), set(), set()
    receipts = []
    for path in sorted(paths, key=str):
        keys, families, bgs = set(), set(), set()
        _harvest(C.read(path), keys, families, bgs)
        allkeys.update(keys); allfamilies.update(families); allbg.update(bgs)
        receipts.append(dict(path=str(path.relative_to(C.ROOT)), sha256=C.sha(path),
            physical_keys=len(keys), background_keys=len(bgs), families=sorted(families),
            audit_type='AABB_PHYSICAL_KEYS' if keys or bgs else 'METADATA_ONLY_NO_AABB_KEYS'))
    doc = Path(__file__).parent/'CNH_RGB_FROZEN_E2E_INPUT_INVENTORY_20261010.md'
    explicit = [
        dict(source='Counterfactual Development', treatment='Key audit of all ordinary/cf train, cal, validation rows'),
        dict(source='real-head units400000–400479', treatment='Protected identity retained; payload not accessed. No permitted AABB inventory exposed by cited input document'),
        dict(source='3RScan/ARKit/TUM RGB-D', treatment='Non-AABB real RGB-D; no CNH physical-world keys'),
        dict(source='six UE alley Development sites', treatment='UE meshes, not this AABB authoring contract; metadata-only separate simulator source'),
        dict(source='164-layout test readiness', treatment='Protected/unauthored placeholder; no test payload access, no authored AABB world to duplicate'),
        dict(source='local paired recording20260921', treatment='Real hardware recording; no AABB scene key'),
        dict(source='H3 real replay20260927', treatment='Real hardware recording; no AABB scene key'),
        dict(source='SANPO-Synthetic', treatment='RGB-D renderer source; no complete AABB physical key; not used for current ToF fixtures'),
    ]
    return dict(files=receipts, source_document=str(doc.relative_to(C.ROOT)), source_document_sha256=C.sha(doc),
                explicitly_listed_sources=explicit, file_count=len(receipts), unique_physical_keys=len(allkeys),
                unique_background_keys=len(allbg), families=sorted(allfamilies)), allkeys, allfamilies, allbg


def save_new(path, value):
    if path.exists():
        raise FileExistsError(f'Preserve source evidence: {path}')
    C.save(path, value)


def prepare(out):
    began = time.monotonic()
    out = Path(out)
    if (out/'prepare_receipt.json').exists():
        raise FileExistsError('Prepared source is immutable')
    resuming = (out/'PLAN.json').exists()
    rows = scenes()
    inventory, oldkeys, oldfamilies, oldbg = inventories(out)
    info = {}
    for split, values in rows.items():
        keys = {r['physical_key'] for r in values}
        families = {r['background_family'] for r in values}
        bgs = {D.physical_key(r['background_boxes']) for r in values}
        assert not keys & oldkeys
        assert not families & oldfamilies
        assert not bgs & oldbg
        info[split] = dict(scenes=len(values), unique_physical=len(keys), physical_overlap=0,
            background_geometry_overlap=0, background_family_overlap=0, families=sorted(families),
            physical_contact_per_height=[sum(r['placement']=='contact' and r['group']==h for r in values) for h in range(2)],
            contact_events_per_height=[K*sum(r['placement']=='contact' and r['group']==h for r in values) for h in range(2)],
            pass_layers=dict(Counter(r['pass_layer'] for r in values if r['placement']=='pass')),
            clear_physical=sum(r['placement']=='clear' for r in values))
    plan = dict(task='CNH_COST_V2_HOLDOUT_DEV_20261010', status='FROZEN_SOURCE_BEFORE_RENDER_OR_FORWARD',
        lane='EXPLORE', main_seed=2026100955, secondary_seeds=[2026100956, 2026100957], replicas=K,
        scientific_command_wall_cap_seconds=1500, gpu_wall_cap_seconds=900,
        evaluation_audit_integration_command_wall_cap_seconds=900, training=0, protected_access=0,
        arms=['M3','old5','both_unchanged','both_costmatched_v2'],
        source_parameters=PARAMETERS, background_specs={s: backgrounds(s) for s in SPLITS}, cohorts=info,
        near_pass_max_gap_m=.10, near_pass_light_weight=.25, near_pass_strong_weight=1.,
        far_pass_and_clear_any_notification_weight=1., sensitivity_light_weights=[0., .5],
        cost='Sum weight times gap1 jointly counted notification; sensitivity weights display only',
        timely_frames=list(range(3,14)), late_frames=[14,15], notifier='unchanged gap1 joint max level',
        calibration='New cal negatives only: enumerate every negative margin tie without monotonic assumption; lowest feasible tau under old5 weighted cost; never use contact benefit to select',
        split_access_order='Render/score cal; seal calibration; only then render/score/read hold outcomes',
        gate='max(ordinary-single margin, joint-parent margin), preserve all old5 grades',
        strong_signal='Main seed both_costmatched_v2 or both_unchanged if weighted cost<=old5; HEAD+BODY timely net>=10/512; far-pass+clear full-weight notifications<=old5*1.10; same net direction in956/957',
        failure='Preserve source failures; no source switch; controlled repair same frozen scenes; stop at applicable cumulative budget cap',
        inference_boundary='Only normalized photon histories, ambient, sensor/public query/length enter frozen inference; author geometry/class/category never enter a model',
        clustering='Physical scene is cluster; K/frame/seed never independent samples',
        pass_label_note='Far-pass >10cm analytically clear in old expanded query, but evaluation retains pass intent using author geometry; all-frame old geometry verified',
        independence_scope='All key-bearing permitted named CNH metadata catalogs listed in inventory.json plus explicit non-AABB/protected entries; not a global historical-world exclusion proof')
    if resuming:
        assert C.read(out/'PLAN.json') == json.loads(json.dumps(plan)), 'No source/protocol switching allowed during repair'
        assert C.read(out/'scene_rows.json') == rows, 'Frozen scenes changed during repair'
        assert C.read(out/'background_rows.json') == {s: backgrounds(s) for s in SPLITS}
        assert C.read(out/'inventory.json') == inventory, 'Inventory changed during repair'
    else:
        save_new(out/'PLAN.json', plan)
        save_new(out/'inventory.json', inventory)
        save_new(out/'scene_rows.json', rows)
        save_new(out/'background_rows.json', {s: backgrounds(s) for s in SPLITS})
    protocol = '# CNH cost-v2 frozen protocol\n\n2026-10-10; EXPLORE; frozen models; no training.\n\n'+json.dumps(plan,ensure_ascii=False,indent=2)+'\n'
    if not resuming:
        (out/'PROTOCOL_FROZEN.md').write_text(protocol,encoding='utf8')
    assert (out/'PROTOCOL_FROZEN.md').read_text(encoding='utf8') == protocol
    if not (out/'protocol_snapshot.md').exists():
        (out/'protocol_snapshot.md').write_text(protocol,encoding='utf8')
    _, _, _, B = D.frozen_imports()
    sensor, query = B.poses(-10.)
    for split, values in rows.items():
        cat = categories(values,sensor)
        folder = out/'data'/split
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'geometry.npz',category=cat,sensor=sensor,public_query=query,
            scene_ids=np.arange(len(values)),scene_uids=np.array([r['scene_uid'] for r in values]))
        info[split]['geometry_sha256']=C.sha(folder/'geometry.npz')
    receipt = dict(status='COMPLETE',seconds=time.monotonic()-began,cohorts=info,replicas=K,
        source_sha256=C.sha(Path(__file__)),plan_sha256=C.sha(out/'PLAN.json'),
        protocol_sha256=C.sha(out/'PROTOCOL_FROZEN.md'),inventory_sha256=C.sha(out/'inventory.json'),
        scene_rows_sha256=C.sha(out/'scene_rows.json'),new_raw=0,predictions=0,training=0,protected_access=0,
        inventory_file_count=inventory['file_count'],unique_prior_physical_keys=inventory['unique_physical_keys'])
    save_new(out/'prepare_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    prepare(parser.parse_args().out)
