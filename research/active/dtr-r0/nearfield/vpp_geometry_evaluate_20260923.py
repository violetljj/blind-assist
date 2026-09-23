"""Post-seal VPP versus frozen unguided geometry on consumed historical panels.

Four principal arms are derived from two frozen depth outputs: unguided/guided,
each alone and with the unchanged ToF union. Truth never enters a predictor.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import foundation_geometry_evaluate_20260923 as base

ROOT, WORK, m = base.ROOT, base.WORK, base.m
read, write, sha = base.read, base.write, base.sha
UNGUIDED = ROOT / 'artifacts.local/evidence/ba-foundation-geometry-20260923-frontend-v1'
UNGUIDED_EVAL = ROOT / 'artifacts.local/evidence/ba-foundation-geometry-20260923-evaluation-v1'
OLD_RGB = ROOT / 'artifacts.local/evidence/ba-foundation-geometry-20260923-rgb/rgb-inputs.json'
MODELS = ('sgbm', 'unguided', 'guided')
ARMS = ('tof', 'sgbm', 'sgbm_union', 'unguided', 'unguided_union', 'guided', 'guided_union')


def hint_pixel_distances(seeds, shape):
    """Chebyshev distance from rounded public seed centres, independent of truth."""
    yy, xx = np.indices(shape)
    distances = np.full(shape, np.inf)
    for seed in seeds:
        distances = np.minimum(distances, np.maximum(np.abs(xx-seed['u']), np.abs(yy-seed['v'])))
    return distances


def paired_geometry(unguided, guided, native, pose, seeds=None):
    """Fixed native-query denominator; a correction needs query membership + Z5cm."""
    a, b = base.eligible(unguided), base.eligible(guided)
    native_inside = base.membership(base.eligible(native), pose)
    ia, ib = base.membership(a, pose), base.membership(b, pose)
    ga = ia & (np.abs(a-native) <= .05)[:, :, None]
    gb = ib & (np.abs(b-native) <= .05)[:, :, None]
    distance = hint_pixel_distances(seeds, native.shape) if seeds is not None else None
    rows = []
    for j in range(2):
        n, x, y = native_inside[:, :, j], ga[:, :, j], gb[:, :, j]
        row = dict(native_query_pixels=int(n.sum()),
            both_correct=int((n & x & y).sum()), corrected=int((n & ~x & y).sum()),
            regressed=int((n & x & ~y).sum()), both_incorrect=int((n & ~x & ~y).sum()),
            unguided_correct=int((n & x).sum()), guided_correct=int((n & y).sum()))
        if distance is not None:
            groups = {'stamp_0to1px': distance <= 1,
                      'near_2to16px': (distance > 1) & (distance <= 16),
                      'beyond16px': (distance > 16) & np.isfinite(distance),
                      'zero_whole_frame': ~np.isfinite(distance)}
            for name, mask in groups.items():
                for key, selected in [('native_query_pixels',n),('corrected',n & ~x & y),('regressed',n & x & ~y)]:
                    row[name+'_'+key] = int((selected & mask).sum())
        rows.append(row)
    return rows


def transitions(a, b, gt):
    """a=unguided, b=guided; applies equally to depth-only and union arms."""
    return dict(retained_tp=int((a & b & gt).sum()), lost_tp=int((a & ~b & gt).sum()),
        new_tp=int((~a & b & gt).sum()), removed_fp=int((a & ~b & ~gt).sum()),
        added_fp=int((~a & b & ~gt).sum()), retained_fp=int((a & b & ~gt).sum()))


def aggregate(rows):
    """Reuse exact geometry aggregation, adding named public-hint query slices."""
    result = {}
    for model in MODELS:
        selected = [r for r in rows if r['model'] == model]
        mapped = [dict(r, model='candidate') for r in selected]
        summaries = base.aggregate_geometry(mapped)['candidate']
        for slice_name in ('no_direct_hint', 'nearby_hint', 'zero_whole_frame'):
            subset = [dict(r, model='candidate') for r in selected if r['hint_slices'][slice_name]]
            summaries[slice_name] = base.aggregate_geometry(subset)['candidate']['all']
        result[model] = summaries
    return result


def summarize_paired(rows):
    slices = {'all': rows}
    slices.update({f: [r for r in rows if r['family'] == f] for f in base.CRITICAL})
    slices.update({p: [r for r in rows if r['part'] == p] for p in m.PARTS})
    slices.update({name: [r for r in rows if r['hint_slices'][name]]
                   for name in ('no_direct_hint', 'nearby_hint', 'zero_whole_frame')})
    result = {}
    for name, selected in slices.items():
        total = Counter()
        for r in selected:
            total.update(r['paired_geometry'])
        result[name] = dict(queries=len(selected), **total)
    return result


def public_hint_counts(hints, pose):
    seeds = hints['seeds']
    assert hints['total_frame_hints'] == len(seeds)
    xyz = np.asarray([s['xyz_camera'] for s in seeds], dtype=float).reshape(-1,3)
    assert np.isfinite(xyz).all()
    body = xyz @ m.rotation(pose['yaw'],pose['pitch'],pose.get('roll',0.)).T
    body += np.asarray(pose['camera_in_body_m'])
    result = {}
    for part,(lo,hi) in zip(m.PARTS,m.BOXES):
        delta = np.maximum(np.maximum(lo-body,body-hi),0.)
        distance = np.linalg.norm(delta,axis=1)
        result[part] = dict(direct=int((distance == 0).sum()),
                            nearby=int(((distance > 0) & (distance <= .25)).sum()))
    return dict(total_frame_hints=len(seeds),query_hints=result)


def hint_slices(hints, part):
    """Public metadata contract; per-query absence never implies no frame hints."""
    total = hints['total_frame_hints']
    query = hints['query_hints'][part]
    assert isinstance(total, int) and total >= 0
    assert 0 <= query['direct'] <= total and 0 <= query['nearby'] <= total
    return dict(no_direct_hint=query['direct'] == 0,
                nearby_hint=query['nearby'] > 0, zero_whole_frame=total == 0)


def slice_point_scores(rows):
    """Query slices are pointwise; discontinuous hint masks do not define events."""
    result = {}
    for name in ('no_direct_hint', 'nearby_hint', 'zero_whole_frame'):
        chosen = [r for r in rows if r['hint_slices'][name]]
        result[name] = dict(queries=len(chosen), stages={})
        for stage in ('raw', 'final'):
            scores = {}
            for arm in ARMS:
                c = Counter(TP=0, FP=0, FN=0, TN=0)
                for r in chosen:
                    p, g = r['predictions'][stage][arm], r['truth']
                    c['TP' if p and g else 'FP' if p else 'FN' if g else 'TN'] += 1
                scores[arm] = dict(c)
            result[name]['stages'][stage] = scores
    return result


def evaluate(candidate, inputs, output, hints_path, unguided=UNGUIDED,
             unguided_eval=UNGUIDED_EVAL, old_rgb=OLD_RGB):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Fresh evaluation output required')
    assert output.resolve().is_relative_to((ROOT / 'artifacts.local').resolve())
    receipts = {name: base.verify(path) for name, path in
                [('guided', candidate), ('unguided', unguided),
                 ('unguided_eval', unguided_eval), ('reference', base.REFERENCE)]}
    manifest, old_manifest, hints = read(inputs), read(old_rgb), read(hints_path)
    assert manifest['rig'] == old_manifest['rig'] == m.RIG
    assert len(manifest['frames']) == len(old_manifest['frames']) == 576
    assert [(r['panel'], r['id']) for r in manifest['frames']] == [
        (r['panel'], r['id']) for r in old_manifest['frames']]
    assert receipts['guided']['frames'] == receipts['unguided']['frames'] == 576
    assert receipts['guided']['input_manifest_sha256'] == sha(inputs)
    assert receipts['unguided']['input_manifest_sha256'] == sha(old_rgb)
    assert hints['input_manifest_sha256'] == sha(inputs)
    assert hints_path.resolve().is_relative_to(candidate.resolve())
    hints_relative = hints_path.resolve().relative_to(candidate.resolve()).as_posix()
    assert sha(hints_path) == {k.replace('\\','/'):v for k,v in receipts['guided']['hashes'].items()}[hints_relative]
    hint_rows = {(r['panel'], r['id']): r for r in hints['frames']}
    assert len(hint_rows) == len(hints['frames']) == 576
    assert set(hint_rows) == {(r['panel'], r['id']) for r in manifest['frames']}
    guided_hashes = {k.replace('\\', '/'): v for k, v in receipts['guided']['hashes'].items()}
    unguided_hashes = {k.replace('\\', '/'): v for k, v in receipts['unguided']['hashes'].items()}
    sgbm_hashes = {k.replace('\\', '/'): v for k, v in read(base.SGBM_SEAL)['inputs'].items()}
    output.mkdir(parents=True, exist_ok=True)
    frozen = {}
    # Seal all derived arms on both panels before joining native or task truth.
    for panel, (folder, old_dir, depth_dir, _) in base.PANELS.items():
        obs = read(base.REFERENCE / panel / 'observations.json')
        entries = [r for r in manifest['frames'] if r['panel'] == panel]
        assert [r['id'] for r in entries] == [o['id'] for o in obs]
        capture = WORK / folder / 'capture-v1'
        captured = read(capture / 'receipt.json')
        previous = np.load(unguided_eval / panel / 'predictions.npz')
        reference = np.load(base.REFERENCE / panel / 'predictions.npz')
        support = {k: [] for k in ('tof', *MODELS)}
        paths = []
        for o, entry in zip(obs, entries):
            frame = capture / 'frame' / o['id']
            for view in ('left', 'right'):
                assert sha(Path(entry[view])) == entry[view + '_sha256'] == captured['hashes'][f"frame/{o['id']}/{view}.png"]
            for filename in ('tof-range.npy', 'tof-valid.npy'):
                assert sha(frame / filename) == captured['hashes'][f"frame/{o['id']}/{filename}"]
            depths = {'sgbm': WORK / folder / old_dir / depth_dir / (o['id'] + '.npy'),
                      'unguided': unguided / panel / 'depth' / (o['id'] + '.npy'),
                      'guided': candidate / panel / 'depth' / (o['id'] + '.npy')}
            relative = f"{panel}/depth/{o['id']}.npy"
            assert sha(depths['guided']) == guided_hashes[relative]
            assert sha(depths['unguided']) == unguided_hashes[relative]
            assert sha(depths['sgbm']) == sgbm_hashes[depths['sgbm'].relative_to(ROOT).as_posix()]
            support['tof'].append(m.readout(m.tof_points(np.load(frame / 'tof-range.npy'),
                np.load(frame / 'tof-valid.npy')), o['pose'], common_fov=True)[0])
            for model, path in depths.items():
                support[model].append(m.readout(m.depth_points(base.eligible(np.load(path))), o['pose'], common_fov=True)[0])
            paths.append(depths)
        support = {k: np.asarray(v) for k, v in support.items()}
        for model in MODELS:
            support[model + '_union'] = support[model] + support['tof']
        final = {k: m.hysteresis(v, [o['episode'] for o in obs]) for k, v in support.items()}
        for model in ('tof', 'sgbm', 'sgbm_union'):
            np.testing.assert_array_equal(support[model], reference[model + '_support'])
            np.testing.assert_array_equal(final[model], reference[model])
        for model, saved in [('unguided', 'candidate'), ('unguided_union', 'candidate_union')]:
            np.testing.assert_array_equal(support[model], previous[saved + '_support'])
            np.testing.assert_array_equal(final[model], previous[saved])
        dest = output / panel; dest.mkdir()
        np.savez_compressed(dest / 'predictions.npz', **final,
                            **{k + '_support': v for k, v in support.items()})
        write(dest / 'prediction-seal.json', dict(predictions_sha256=sha(dest / 'predictions.npz'),
            input_manifest_sha256=sha(inputs), hints_sha256=sha(hints_path),
            guided_receipt_sha256=sha(candidate / 'receipt.json'), baseline_parity=True,
            depth_hashes={str(p): sha(p) for frame in paths for p in frame.values()}))
        frozen[panel] = (obs, capture, captured, support, final, paths)
    panels, geometry, paired = {}, [], []
    for panel, (obs, capture, captured, support, final, paths) in frozen.items():
        spec_path = capture / 'spec.json'
        assert sha(spec_path) == captured['spec_sha256']
        spec = read(spec_path)
        assert base.observation_contract(spec) == obs
        gt = np.load(base.REFERENCE / panel / 'truth.npy')
        raw = {k: v > 0 for k, v in support.items()}
        predictions = {'raw': raw, 'final': final}
        scores = {stage: {k: base.metrics(v, gt, obs) for k, v in p.items()}
                  for stage, p in predictions.items()}
        panel_geometry, panel_paired = [], []
        for i, o in enumerate(obs):
            path = capture / 'frame' / o['id'] / 'native-left-depth.npy'
            assert sha(path) == captured['hashes'][f"frame/{o['id']}/native-left-depth.npy"]
            native = np.load(path)
            depths = {model: np.load(p) for model, p in paths[i].items()}
            hint = hint_rows[(panel, o['id'])]
            counts = public_hint_counts(hint,o['pose'])
            common = [dict(panel=panel, id=o['id'], part=part,
                family=spec['frames'][i]['family'], truth=bool(gt[i, j]),
                hint_slices=hint_slices(counts, part), total_frame_hints=counts['total_frame_hints'],
                query_hints=counts['query_hints'][part]) for j, part in enumerate(m.PARTS)]
            for model in MODELS:
                for j, row in enumerate(base.geometry_rows(depths[model], native, o['pose'])):
                    panel_geometry.append(dict(common[j], model=model, **row))
            changes = paired_geometry(depths['unguided'], depths['guided'], native, o['pose'], hint['seeds'])
            for j, row in enumerate(changes):
                panel_paired.append(dict(common[j], paired_geometry=row,
                    predictions={stage: {k: bool(p[k][i, j]) for k in ARMS} for stage, p in predictions.items()}))
        geometry += panel_geometry; paired += panel_paired
        detail = dict(scores=scores, geometry=aggregate(panel_geometry),
            paired_geometry=summarize_paired(panel_paired), hint_slices=slice_point_scores(panel_paired),
            transitions={stage: {suffix or 'depth_only': transitions(p['unguided'+suffix],p['guided'+suffix],gt)
                for suffix in ('','_union')} for stage,p in predictions.items()},
            event_comparison={suffix or 'depth_only': base.event_compare(scores['final']['guided'+suffix],
                scores['final']['unguided'+suffix]) for suffix in ('','_union')},
            family_slices={family: {stage: {k: base.metrics(v, gt, obs, [f['family']==family for f in spec['frames']])
                for k,v in p.items()} for stage,p in predictions.items()}
                for family in dict.fromkeys(f['family'] for f in spec['frames'])})
        write(output/panel/'summary.json',detail); panels[panel]=detail
    pooled={stage:{arm:{key:sum(p['scores'][stage][arm][key] for p in panels.values())
        for key in ('TP','FP','FN','events','missed_events','false_sessions','false_segments')}
        for arm in ARMS} for stage in ('raw','final')}
    historical=read(unguided_eval/'summary.json')['pooled_descriptive']
    for stage in ('raw','final'):
        for arm,saved in [('unguided','candidate'),('unguided_union','candidate_union'),('sgbm_union','sgbm_union')]:
            assert pooled[stage][arm]==historical[stage][saved]
    write(output/'geometry-queries.json',geometry)
    write(output/'paired-queries.json',paired)
    write(output/'summary.json',dict(evidence='CONSUMED_RENDERED_DEVELOPMENT',frames=576,
        primary_comparison='guided_union versus unguided_union; SGBM secondary',
        four_arms=['unguided','unguided_union','guided','guided_union'],
        pooled_descriptive=pooled,geometry=aggregate(geometry),paired_geometry=summarize_paired(paired),
        transitions={stage:{arm:{key:sum(p['transitions'][stage][arm][key] for p in panels.values())
            for key in ('retained_tp','lost_tp','new_tp','removed_fp','added_fp','retained_fp')}
            for arm in ('depth_only','_union')} for stage in ('raw','final')},
        event_comparison={panel:p['event_comparison'] for panel,p in panels.items()},
        hint_slices=slice_point_scores(paired),baseline_parity=True,
        hint_metadata_sha256=sha(hints_path),hint_definitions=hints.get('definitions',{}),
        definitions=dict(correct_geometry='Visible native query pixel, prediction within same query and axial Z error <=5cm',
            hint_slices='Overlapping public metadata query subsets; events only computed on complete panels/families',
            zero_whole_frame='No applied public hints anywhere in frame; distinct from no direct hint for HEAD/BODY',
            direct_hint='Public seed xyz transformed into query AABB; query-spatial, not same-object or rounded-pixel truth',
            nearby_hint='Public seed outside query AABB and Euclidean distance <=0.25m from AABB',
            stamp_distance='Chebyshev pixel distance from rounded seed centre: <=1, 2..16, >16, or no frame seeds',
            geometry_denominator='All visible native in-query pixels; missing eligible output counts as failure'),
        sources={k:sha(p/'receipt.json') for k,p in [('guided',candidate),('unguided',unguided),('unguided_eval',unguided_eval)]},
        evaluator_sha256=sha(Path(__file__)),base_evaluator_sha256=sha(Path(base.__file__))))
    write(output/'receipt.json',dict(status='PASS',hashes={p.relative_to(output).as_posix():sha(p)
        for p in output.rglob('*') if p.is_file()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('candidate','inputs','output','hints'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--unguided',type=Path,default=UNGUIDED)
    parser.add_argument('--unguided-evaluation',type=Path,default=UNGUIDED_EVAL)
    parser.add_argument('--old-rgb',type=Path,default=OLD_RGB)
    args=parser.parse_args()
    evaluate(args.candidate,args.inputs,args.output,args.hints,args.unguided,args.unguided_evaluation,args.old_rgb)
