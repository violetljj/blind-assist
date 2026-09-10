"""Saved-score and evaluator-geometry diagnosis; no learned inference or fitting."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from contact_retina_spec import BODY_BOXES
from mz5_ensemble_readout import read, sha, write
from mz8_attribution import angular_hypotheses

ORDER = ['BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR']


def bounds(q):
    lo, hi = (np.array(v, dtype=float) for v in BODY_BOXES[q // 2])
    lo[0] = hi[0] + 1.5 * (q % 2)
    hi[0] = lo[0] + 1.5
    return lo, hi


def member(points, q):
    lo, hi = bounds(q)
    yes = ((points >= lo) & (points <= hi)).all(-1)
    if q % 2 == 0:
        yes &= points[..., 0] < hi[0]
    return yes


def native_geometry(path, q):
    depth = np.load(path)
    yy, xx = np.mgrid[:360, :640]
    focal = 320 / np.tan(np.deg2rad(50))
    rays = np.stack([np.ones_like(xx), (xx - 319.5) / focal, -(yy - 179.5) / focal], -1)
    good = np.isfinite(depth) & (depth > 0) & (depth < 100) & (depth * np.linalg.norm(rays, axis=-1) <= 4)
    points = depth[..., None].astype(float) * rays + [0, 0, 1.7]
    counts = [int((good & member(points, j)).sum()) for j in range(4)]
    lo, hi = bounds(q)
    delta = np.maximum(np.maximum(lo - points, points - hi), 0)
    distance = np.linalg.norm(delta, axis=-1)
    distance[~good] = np.inf
    index = np.unravel_index(distance.argmin(), distance.shape)
    return dict(native_query_pixel_counts=counts, valid_native_pixels=int(good.sum()),
        nearest_valid_native_to_query_m=float(distance[index]), nearest_native_xyz=points[index].tolist(),
        nearest_native_pixel_yx=list(map(int, index)), nearest_axis_outside_m=delta[index].tolist())


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    work = root / 'artifacts.local/work'
    cache = work / 'mz15-shared-support-20260910/cache-v1'
    cr = read(cache / 'receipt.json')
    inputs = {}
    for name in ['selected.json', 'observations.npz', 'evaluator.npz']:
        inputs[str(cache / name)] = sha(cache / name)
        assert inputs[str(cache / name)] == cr['files'][name]
    selected = read(cache / 'selected.json')
    with np.load(cache / 'observations.npz') as a:
        ranges, valid = a['ranges'], a['valid']
    with np.load(cache / 'evaluator.npz') as a:
        truth, sources, queries, known = (a[k] for k in ['truth', 'source_presence', 'query_presence', 'known'])
    configs = [('MZ16', 'mz16-visual-detail-20260910', 'HIGH_DETAIL'),
               ('MZ18', 'mz18-body-objective-20260910', 'BODY_WITNESS'),
               ('MZ20', 'mz20-rank-objective-20260910', 'BODY_RANK')]
    saved, local, errors, cutoffs = {}, {}, {}, {}
    for tag, name, arm in configs:
        folder = work / name / 'run-v1'
        receipt = read(folder / 'receipt.json')
        for filename in ['predictions.npz', 'local_samples.npz', arm + '-cutoff.npy']:
            inputs[str(folder / filename)] = sha(folder / filename)
            assert inputs[str(folder / filename)] == receipt['outputs'][filename]
        cutoffs[tag] = np.load(folder / (arm + '-cutoff.npy'))
        saved[tag], local[tag], errors[tag] = {}, {}, []
        with np.load(folder / 'predictions.npz') as a, np.load(folder / 'local_samples.npz') as b:
            for cohort in ['relation10000', 'distance5000']:
                prefix = f'{arm}/{cohort}/'
                saved[tag][cohort] = {k: a[prefix+k] for k in ['truth','raw','margin','support','baseline','accepted','winning_source','winning_query']}
                local[tag][cohort] = [{k:b[prefix+f'{q}/{k}'] for k in ['score','label','frame']} for q in range(4)]
                aa = saved[tag][cohort]
                errors[tag] += [(cohort, int(i), int(q)) for i, q in np.argwhere(aa['accepted'] & ~aa['truth'])]
    rays = angular_hypotheses()[0].numpy()
    details = []
    for cohort, i, q in sorted(set(sum(errors.values(), []))):
        rows = [r for r in selected if r['dataset'] == cohort]
        row = rows[i]
        assert row['role'] == 'DEV_ONLY'
        ci = row['cache_index']
        r, v = ranges[ci], valid[ci]
        good = v & np.isfinite(r) & (r > 0) & (r <= 4)
        points = np.where(good, r, 0)[..., None, None] * rays[:, None] + np.array([0, 0, 1.7], np.float32)
        eligible = member(points, q) & good[..., None]
        mask = eligible & known[ci, :, None, :] & v[..., None]
        coordinates = np.argwhere(mask)
        qlabels = queries[ci, ..., q][mask]
        union_echo = eligible.any(-1)
        rowdetail = dict(cohort=cohort, frame=i, query=ORDER[q], query_index=q,
            record={k:row[k] for k in ['index','site','family','group','condition','cache_index','native','rgb']},
            truth=truth[ci].tolist(), valid_echoes=int(good.sum()),
            eligible_candidates=int(eligible.sum()), eligible_zone_echoes=int(union_echo.sum()),
            known_eligible_candidates=int(mask.sum()), actual_source_cells_any=int(sources[ci].sum()),
            actual_query_cells_any=queries[ci].sum(axis=(0,1,2)).tolist(),
            query_source_cells_within_eligible=int((queries[ci,...,q] & eligible).sum()),
            any_source_cells_within_eligible=int((sources[ci] & eligible).sum()),
            query_cells_in_geometrically_eligible_echoes=int(queries[ci,...,q][union_echo].sum()), models={})
        for tag in saved:
            aa = saved[tag][cohort]
            assert bool(aa['support'][i,q]) == bool(eligible.any())
            ll = local[tag][cohort][q]
            take = ll['frame'] == i
            scores, labels = ll['score'][take], ll['label'][take]
            np.testing.assert_array_equal(labels, qlabels)
            assert len(scores) == len(coordinates)
            order = np.argsort(-scores)[:3]
            nearest_scores = [dict(zone=int(coordinates[t,0]), echo=int(coordinates[t,1]), cell=int(coordinates[t,2]),
                score=float(scores[t]), actual_query=bool(labels[t]), actual_source=bool(sources[ci][tuple(coordinates[t])]),
                range_m=float(r[tuple(coordinates[t,:2])]), xyz=points[tuple(coordinates[t])].tolist()) for t in order]
            rowdetail['models'][tag] = dict(raw=float(aa['raw'][i,q]), cutoff=float(cutoffs[tag][q]),
                margin=float(aa['margin'][i,q]), added_fp=bool(aa['accepted'][i,q]),
                winning_query=bool(aa['winning_query'][i,q]), winning_source=bool(aa['winning_source'][i,q]),
                known_local_positive_count=int(labels.sum()), known_local_negative_count=int((~labels).sum()),
                top_known_local=nearest_scores,
                winner_present_in_known_local_scores=bool(len(scores) and np.isclose(scores.max(), aa['raw'][i,q], rtol=0, atol=1e-6)))
        assert sha(row['native']) == row['native_sha']
        inputs[row['native']] = row['native_sha']
        rowdetail['native_geometry'] = native_geometry(row['native'], q)
        np.testing.assert_array_equal(np.array(rowdetail['native_geometry']['native_query_pixel_counts']) >= 3, truth[ci])
        details.append(rowdetail)
    result = dict(errors={tag:[dict(cohort=c,frame=i,query=ORDER[q]) for c,i,q in rows] for tag,rows in errors.items()},
        counts={tag:len(rows) for tag,rows in errors.items()}, details=details,
        mz20_family_counts=dict(Counter(r['record']['family'] for r in details if r['models']['MZ20']['added_fp'])),
        scope='Consumed Development. No inference, training, threshold selection or EVAL. Exact winning location is not saved; local scores cover known-valid candidates only.',
        inputs=inputs, code_sha256=sha(Path(__file__)))
    write(output / 'result.json', result)
    write(output / 'receipt.json', dict(status='PASS', inputs=inputs, code_sha256=sha(Path(__file__)),
        outputs={'result.json':sha(output / 'result.json')}, backend='NumPy CPU saved scores and bounded evaluator geometry'))
    print('counts', result['counts'])
    for row in details:
        if row['models']['MZ20']['added_fp']:
            print({k:row[k] for k in ['cohort','frame','query','known_eligible_candidates','actual_query_cells_any','query_cells_in_geometrically_eligible_echoes']},
                  row['record']['site'],row['record']['family'],row['models']['MZ20'],row['native_geometry'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.root,args.output)
