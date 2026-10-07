"""CPU-only physical geometry manifest for frozen-readout double-height tests.

No model scores, weights, rendering, fitting, or device initialization are read.
The three variants change only boxes[0]'s height; original trajectories remain.
"""
import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import cnh_sequence_observed_geometry as G

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work'
OUT = WORK/'cnh-double-height-dev-20261007'
ABLATION = WORK/'cnh-center-readout-ablation-dev-20261007'
AUG = WORK/'cnh-extrinsic-aug-20261006'
HEIGHTS = {'H': (-.10, .26), 'B': (.50, .84), 'HB': (-.10, .84)}
CONTACT_CATS = ('contact0-2cm', 'contact2-5cm', 'contact>5cm')
SEED = 2026100721


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths(unit):
    base = AUG if unit < 99000 else AUG/'continuation-r1'
    split = 'calibration' if unit < 99000 else 'evaluation'
    return {'truth': base/f'truth/{split}/unit{unit}.json',
            'observations': base/f'observations/{split}/unit{unit}.npz'}


def variant_boxes(scene):
    result = {}
    for name, (low, high) in HEIGHTS.items():
        boxes = copy.deepcopy(scene['boxes'])
        boxes[0]['lo'][1], boxes[0]['hi'][1] = low, high
        result[name] = boxes
    return result


def _finite_json(value):
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite_json(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    return value


def validate_candidate(scene, travel, role):
    """Return physical all-box/target-only categories and rejection reasons."""
    if role not in ('contact', 'clear'):
        raise ValueError('role must be contact or clear')
    travel = np.asarray(travel, float)
    if travel.shape != (16, 4, 4):
        raise ValueError('Original 16 travel poses required')
    poses = travel[G.FRAMES]
    variants, reasons = {}, []
    for name, boxes in variant_boxes(scene).items():
        # Check only changed target against backgrounds; preserve old backgrounds.
        t = boxes[0]
        collisions = [j for j, b in enumerate(boxes[1:], 1)
                      if np.all(np.minimum(t['hi'], b['hi'])-
                                np.maximum(t['lo'], b['lo']) > G.EPS)]
        if collisions:
            reasons.append(name+'/target_background_overlap')
        meshes = [G.S.box_mesh(b['lo'], b['hi']) for b in boxes]
        mesh = np.concatenate(meshes)
        ranges, reference = G.deadline_reference(G.corners(t), poses)
        if np.any(np.diff(ranges) > 1e-10):
            reasons.append(name+'/nonmonotonic_range')
        frames = [[G.surface_category(G.transform(mesh, pose), q) for q in (0, 1)]
                  for pose in poses]
        target_frames = [[G.surface_category(G.transform(meshes[0], pose), q) for q in (0, 1)]
                         for pose in poses]
        refcats = [G.surface_category(G.transform(mesh, reference['reference_pose']), q)
                   for q in (0, 1)] if reference['covered'] else ['censored']*2
        target_ref = [G.surface_category(G.transform(meshes[0], reference['reference_pose']), q)
                      for q in (0, 1)] if reference['covered'] else ['censored']*2
        cq = [cat in CONTACT_CATS for cat in refcats]
        tcq = [cat in CONTACT_CATS for cat in target_ref]
        clear_all = (np.asarray(frames) == 'clear').all(0).tolist()
        expected = {'H': [True, False], 'B': [False, True], 'HB': [True, True]}[name]
        if role == 'contact':
            if not reference['covered']:
                reasons.append(name+'/deadline_censored')
            if cq != expected:
                reasons.append(name+'/all_box_contact_pattern')
            if tcq != expected:
                reasons.append(name+'/target_contact_pattern')
        elif not all(clear_all):
            reasons.append(name+'/not_both_query_clear_all')
        idx = int(np.searchsorted(G.FRAMES, reference['reference_fraction']+1e-10, side='right')-1) if reference['covered'] else None
        variants[name] = _finite_json(dict(boxes=boxes, frame_category=frames,
            target_frame_category=target_frames, clear_all=clear_all,
            ref_category=refcats, target_ref_category=target_ref, contact_query=cq,
            physical_contact=any(cq), physical_control=all(clear_all),
            frame_ranges=ranges, deadline_index=idx,
            changed_target_background_overlaps=collisions,
            reusable_original_observation=(boxes == scene['boxes']), **reference))
    refs = [v['reference_fraction'] for v in variants.values()]
    if len({v['covered'] for v in variants.values()}) != 1 or (
            refs[0] is not None and max(refs)-min(refs) > 1e-10):
        reasons.append('height_changes_reference')
    if role == 'contact':
        depths = [max(CONTACT_CATS.index(c) for c in v['ref_category'] if c in CONTACT_CATS)
                  for v in variants.values() if v['physical_contact']]
        if len(set(depths)) > 1:
            reasons.append('height_changes_contact_depth')
    return {'valid': not reasons, 'reasons': sorted(set(reasons)), 'variants': variants}


def source_rows():
    """Read geometry arrays and fold unit IDs only; never open score ledgers."""
    plan = read(ABLATION/'PLAN.json')
    fold = {int(u): k for k, f in enumerate(plan['folds']) for u in f['evaluation']}
    rows = {}
    sources = [AUG/'geometry/calibration.npz', AUG/'continuation-r1/geometry/evaluation.npz']
    for path in sources:
        with np.load(path, allow_pickle=False) as z:
            fields = {k: z[k] for k in ('unit', 'config', 'query', 'ref_category', 'covered',
                                       'clear_all', 'reference_fraction')}
        for j, u in enumerate(fields['unit']):
            u, c, q = int(u), int(fields['config'][j]), int(fields['query'][j])
            row = rows.setdefault((u, c), dict(unit=u, config=c, fold=fold[u],
                category=[None, None], covered=[False, False], clear=[False, False], fraction=[None, None]))
            assert row['category'][q] is None
            row['category'][q] = str(fields['ref_category'][j])
            row['covered'][q] = bool(fields['covered'][j])
            row['clear'][q] = bool(fields['clear_all'][j])
            row['fraction'][q] = float(fields['reference_fraction'][j])
    for row in rows.values():
        assert None not in row['category']
        contact = any(cov and cat in CONTACT_CATS for cov, cat in zip(row['covered'], row['category']))
        row['role'] = 'contact' if contact else 'clear' if all(row['clear']) else 'excluded_source'
        row['depth'] = max((CONTACT_CATS.index(c) for c in row['category'] if c in CONTACT_CATS), default=-1) if contact else -1
    return list(rows.values()), sources


def add_driver_fields(manifest):
    """Schema aliases only; does not select, relabel, or recompute geometry."""
    path = WORK/'cnh-tristate-dev-r3-20261006/rows.json'
    metadata = {(r['unit'], r['config']): r for r in read(path)}
    counts = manifest['counts']
    counts.update(requested_contact_anchors=36, requested_clear_anchors=36,
                  requested_variants=216,
                  missing_contact_anchors=36-counts['selected_roles'].get('contact', 0),
                  missing_variants=216-counts['selected_variants'])
    for anchor in manifest['anchors']:
        anchor['anchor_id'] = f"u{anchor['unit']}_c{anchor['config']}_{anchor['role']}"
        anchor['turn'] = metadata[anchor['unit'], anchor['config']]['turn']
        anchor['depth_bin'] = CONTACT_CATS.index(anchor['depth']) if anchor['depth'] is not None else -1
        anchor['source_variant'] = anchor['original_variant']
        for tag, variant in anchor['variants'].items():
            variant['tag'] = tag
            variant['control'] = variant['physical_control']
            variant['contact'] = variant['physical_contact']
    manifest['hashes'][str(path.relative_to(ROOT))] = sha(path)
    return manifest


def build_manifest(budget_seconds=600):
    start = time.monotonic()
    def check():
        if time.monotonic()-start > budget_seconds:
            raise TimeoutError('CPU geometry budget reached; no score-dependent shrink')
    rows, sources = source_rows()
    cache, valid, screening = {}, [], []
    for number, r in enumerate(rows):
        if r['role'] == 'excluded_source':
            continue
        check(); u, c = r['unit'], r['config']
        if u not in cache:
            cache[u] = read(source_paths(u)['truth'])
        src = cache[u]; scenes = {s['config']: s for s in src['scenes']}; scene = scenes[c]
        result = validate_candidate(scene, src['travel'], r['role'])
        identity = dict(unit=u, config=c, fold=r['fold'], role=r['role'],
            depth=CONTACT_CATS[r['depth']] if r['depth'] >= 0 else None,
            source_group=int(scene['group']), family=scene['cond'], mode=int(src['mode']))
        screening.append(dict(**identity, valid=result['valid'], reasons=result['reasons']))
        if result['valid']:
            variants = result['variants']; original = 'H' if int(scene['group']) == 0 else 'B'
            assert variants[original]['reusable_original_observation']
            assert variants[original]['ref_category'] == r['category']
            assert variants[original]['clear_all'] == r['clear']
            if variants[original]['covered']:
                assert abs(variants[original]['reference_fraction']-r['fraction'][0]) < 1e-10
            valid.append(dict(**identity, source_paths={k: str(p.relative_to(ROOT)) for k, p in source_paths(u).items()},
                original_variant=original, variants=variants))
        if len(screening) % 100 == 0:
            print('geometry', len(screening), 'valid', len(valid), 'seconds', round(time.monotonic()-start, 1), flush=True)
    check(); rng = np.random.default_rng(SEED); selected, gaps, quota_rows = [], [], []
    used_contact, used_clear = set(), set()
    for role in ('contact', 'clear'):
        for fold in range(3):
            for family in ('none', 'corner'):
                for group in (0, 1):
                    depth_values = CONTACT_CATS if role == 'contact' else (None,)
                    for depth in depth_values:
                        wanted = 1 if role == 'contact' else 3
                        pool = [r for r in valid if r['role'] == role and r['fold'] == fold and
                                r['family'] == family and r['source_group'] == group and r['depth'] == depth]
                        order = rng.permutation(len(pool)).tolist(); candidates = [pool[i] for i in order]
                        if role == 'clear':
                            candidates.sort(key=lambda r: r['unit'] not in used_contact)
                        used = used_contact if role == 'contact' else used_clear
                        chosen, reused_units = [], []
                        while candidates and len(chosen) < wanted:
                            # Distinct units are preferred, not a hard quota gate.
                            pick = next((i for i, r in enumerate(candidates) if r['unit'] not in used), 0)
                            r = candidates.pop(pick)
                            if r['unit'] in used:
                                reused_units.append(r['unit'])
                            chosen.append(r); used.add(r['unit'])
                        selected.extend(chosen)
                        cell = dict(role=role, fold=fold, family=family, source_group=group,
                                    depth=depth, available=len(pool), wanted=wanted, selected=len(chosen),
                                    reused_units=reused_units)
                        quota_rows.append(cell)
                        if len(chosen) != wanted:
                            gaps.append(dict(**cell, reason='insufficient_geometrically_valid_candidates'))
    reason_counts = Counter(reason for r in screening for reason in r['reasons'])
    paths = [Path(__file__), ABLATION/'PLAN.json', ABLATION/'NEXT_DOUBLE_HEIGHT_PLAN.md', *sources,
             *[source_paths(u)['truth'] for u in sorted(cache)]]
    result = dict(status='COMPLETE_WITH_QUOTA_GAPS' if gaps else 'COMPLETE', selection_seed=SEED,
        no_model_scores_read=True, no_render_or_training=True, budget_seconds=budget_seconds,
        seconds=time.monotonic()-start, query_order=['HEAD', 'BODY'], variants={k:list(v) for k,v in HEIGHTS.items()},
        original_sequence_frames=list(range(16)), output_frames=G.FRAMES.tolist(),
        counts=dict(source_sequences=len(rows), source_roles=dict(Counter(r['role'] for r in rows)),
                    checked=len(screening), valid=len(valid), rejected=len(screening)-len(valid),
                    selected_roles=dict(Counter(r['role'] for r in selected)),
                    selected_units=len({r['unit'] for r in selected}), selected_variants=len(selected)*3),
        selection_contract='All contact/clear candidates geometrically checked before seeded stratified choice. Contact1 per fold/family/source-group/depth; clear3 per fold/family/source-group. Prefer distinct unit within role, allow valid unit reuse when needed; clear prefers chosen contact units. No score access.',
        exclusions_by_reason=dict(reason_counts), screening=screening, quota_cells=quota_rows,
        quota_gaps=gaps, anchors=selected, hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        limitations=['Consumed source units with new physical height interventions; not fresh independent scenes',
                     'Censoring and source noncontacts/noncontrols excluded explicitly',
                     'No model score, detector prediction, or object visibility used for selection'])
    return add_driver_fields(result)


def run(budget_seconds=600):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'geometry_manifest.json'
    if path.exists():
        raise FileExistsError('Preserve existing geometry manifest')
    result = build_manifest(budget_seconds)
    with path.open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','counts','quota_gaps','seconds')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['run'])
    parser.add_argument('--budget-seconds', type=float, default=600)
    args = parser.parse_args(); run(args.budget_seconds)
