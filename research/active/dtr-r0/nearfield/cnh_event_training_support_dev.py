"""Describe actual fixed-model positive training support; no causal attribution."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import time
import numpy as np
import cnh_model_stability_dev as S
import cnh_model_stability_geometry_dev as G

OUT = S.WORK/'cnh-event-training-support-dev-20261007/stored-truth-repair'


def truth_path(unit):
    role = 'calibration' if 98000 <= unit < 98048 else 'evaluation' if 99000 <= unit < 99096 else None
    if role is None:
        raise ValueError('Unsupported frozen training unit')
    batch = S.WORK/'cnh-extrinsic-aug-20261006'
    if role == 'evaluation':
        batch = batch/'continuation-r1'
    return batch/f'truth/{role}/unit{unit}.json'


def descriptor(box, family, mode, deadline):
    lo, hi = np.asarray(box['lo']), np.asarray(box['hi'])
    return dict(family=family, mode=int(mode), ylo=float(lo[1]), yhi=float(hi[1]),
                width=float(hi[0]-lo[0]), depth=float(hi[2]-lo[2]),
                rho=float(box['rho']), initial_z=float(lo[2]), deadline=int(deadline))


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), *G.source_files(), S.PARENT/'PLAN.json',
             S.PARENT/'ledger.npz', S.SUPPORT/'augmentation_rows.json',
             G.OUT/'geometry_manifest.json', S.WORK/'cnh-double-height-dev-20261007/geometry_manifest.json']
    paths += [S.SUPPORT/f'fold{f}_training_schedule.npz' for f in range(3)]
    units = {u for f in S.D.read(S.PARENT/'PLAN.json')['folds'] for u in f['train']}
    paths += [truth_path(u) for u in sorted(units)]
    G.save(OUT/'PLAN.json', dict(scope='Posthoc actual raw_hb_aug positive schedule support; physical metadata is diagnostic only',
        budget_cpu_seconds=50, no_gpu=True, no_fit=True,
        repair='Preserve120s full-scene timeout and two rejected regeneration attempts. Read authoritative stored training truth directly, including its mirror, and verify all72 augmentation targets. No geometry recomputation or changed training labels.',
        coverage='Counts by family/mode/target yspan; continuous min/max envelopes for width/depth/rho/initial_z/deadline. These are coarse metadata support, not observation sufficiency or a causal mechanism.',
        hashes={p.relative_to(S.ROOT).as_posix():S.D.sha(p) for p in paths}))


def run():
    start = time.monotonic()
    plan = S.D.read(OUT/'PLAN.json')
    if (OUT/'START.json').exists():
        raise FileExistsError('Preserve prior support analysis')
    G.save(OUT/'START.json', dict(unix=time.time()))
    for path, expected in plan['hashes'].items():
        assert S.D.sha(S.ROOT/path) == expected, path
    with np.load(S.PARENT/'ledger.npz') as z:
        unit, config, contact = z['unit'], z['config'], z['contact']
    aug = S.D.read(S.SUPPORT/'augmentation_rows.json')
    scene_cache = {}
    def scene(u, c):
        if u not in scene_cache:
            saved = S.D.read(truth_path(u))
            assert saved['unit'] == u and saved['mode'] == u % 3
            scene_cache[u] = [dict(boxes=r['boxes'], family=r['cond']) for r in saved['scenes']]
        return scene_cache[u][c]
    all_eval = S.D.read(G.OUT/'geometry_manifest.json')['rows']
    old_anchors = S.D.read(S.WORK/'cnh-double-height-dev-20261007/geometry_manifest.json')['anchors']
    for a in old_anchors:
        target = scene(a['unit'], a['config'])['boxes'][0]
        original_tag = a['source_variant']
        assert target == a['variants'][original_tag]['boxes'][0], a['anchor_id']
    eval_rows = [r for r in all_eval
                 if r['valid'] and r['evaluable'] and r['contact']]
    output, summaries, training = [], [], []
    for fold, roles in enumerate(S.D.read(S.PARENT/'PLAN.json')['folds']):
        with np.load(S.SUPPORT/f'fold{fold}_training_schedule.npz') as z:
            positive = np.flatnonzero(z['label'] == 1)
            source, anchor, ti = z['original_source_row_index'][positive], z['anchor_index'][positive], z['time_index'][positive]
        records = []
        for si, ai, t in zip(source, anchor, ti):
            u, c = int(unit[si]), int(config[si])
            assert contact[si] and u in roles['train']
            sc = scene(u, c)
            box = dict(sc['boxes'][0], lo=list(sc['boxes'][0]['lo']), hi=list(sc['boxes'][0]['hi']))
            if ai >= 0:
                assert aug[int(ai)]['source_row_index'] == int(si)
                box['lo'][1], box['hi'][1] = -.1, .84
            records.append(dict(fold=fold, unit=u, config=c, source_row_index=int(si),
                augmentation_anchor_index=int(ai), **descriptor(box, sc['family'], u % 3, t)))
        assert len({(r['source_row_index'], r['augmentation_anchor_index']) for r in records}) == len(records)
        assert sum(r['augmentation_anchor_index'] >= 0 for r in records) == [17, 19, 22][fold]
        training.extend(records)
        counts = Counter((r['family'], r['mode'], r['ylo'], r['yhi']) for r in records)
        summaries.append(dict(fold=fold, positive_samples=len(records), hb_positive_samples=[17, 19, 22][fold],
                              structure_counts=[dict(family=k[0], mode=k[1], ylo=k[2], yhi=k[3], count=v) for k,v in sorted(counts.items())]))
        for row in eval_rows:
            if time.monotonic()-start > plan['budget_cpu_seconds']:
                raise TimeoutError('50s remaining stored-truth support repair budget reached')
            d = descriptor(row['boxes'][0], row['family'], row['mode'], row['deadline_index'])
            key = d['family'], d['mode'], d['ylo'], d['yhi']
            matched = [r for r in records if (r['family'], r['mode'], r['ylo'], r['yhi']) == key]
            rec = dict(fold=fold, row_id=row['row_id'], anchor_id=row['anchor_id'], tag=row['tag'],
                       coarse_positive_support=len(matched), **d)
            for field in ('width', 'depth', 'rho', 'initial_z', 'deadline'):
                values = [r[field] for r in matched]
                rec[field+'_train_min'] = min(values) if values else None
                rec[field+'_train_max'] = max(values) if values else None
                rec[field+'_outside_envelope'] = bool(values and not min(values) <= d[field] <= max(values)) if values else None
            output.append(rec)
    for name, records in [('event_support.csv', output), ('training_positive_support.csv', training)]:
        with (OUT/name).open('x', encoding='utf8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    G.save(OUT/'result.json', dict(status='POSTHOC_COMPLETE', seconds=time.monotonic()-start,
        event_records=len(output), full_event_count=len(eval_rows), summaries=summaries,
        limits=plan['coverage'], output_hashes={n:S.D.sha(OUT/n) for n in ('event_support.csv','training_positive_support.csv')}))
    print('TRAINING SUPPORT', len(output), 'event-model records', round(time.monotonic()-start,3), 's')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['freeze','run'])
    globals()[parser.parse_args().stage]()
