"""Coordinate-gauge diagnostic on existing Development photons and models.

This changes world coordinates, not the physical scene, sensor, or query.
Saved public_query is current-sensor -> query. Therefore W=Q inv(S),
S'=G S and W'=W inv(G), while Q'=W' S'=Q. A left +3deg change of
Q is a different physical query and is reported separately, never as gauge.
No photons, training, threshold selection, or hardware work occur here.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
OUT = ROOT/'artifacts.local/work/cnh-pass-boundary-dev-20261009'
SEED = 2026100955
CHECKPOINT = OLD/f'models/ordinary_seed{SEED}.pt'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 23), b''):
            digest.update(block)
    return digest.hexdigest()


def rotation(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return matrix


def gauge():
    matrix = rotation(37.)
    matrix[:3, 3] = [.4, -.2, .7]
    return matrix


def chains(sensor, public_query, change):
    """Return full current/past chains; do not round/canonicalize matrices."""
    world_to_query = public_query@np.linalg.inv(sensor)
    changed_sensor = change@sensor
    changed_world_to_query = world_to_query@np.linalg.inv(change)
    changed_public = changed_world_to_query@changed_sensor
    original = world_to_query[:, None]@sensor[None]
    changed = changed_world_to_query[:, None]@changed_sensor[None]
    return original, changed, world_to_query, changed_world_to_query, changed_public


def corners(boxes):
    return np.asarray([[list(p)+[1.] for p in itertools.product(
        *zip(box['lo'], box['hi']))] for box in boxes], dtype=np.float64)


def prepare():
    """CPU-only all-pose chain/scene audit and <=32 history bindings."""
    import cnh_counterfactual_data_dev as D
    started = time.monotonic()
    rows = read(OLD/'scene_rows.json')
    out = OUT/'coordinate_equivalence'
    if (out/'prepare_receipt.json').exists():
        raise FileExistsError('Preserve completed coordinate preparation')
    out.mkdir(parents=True, exist_ok=True)
    changes = {'identity': np.eye(4), 'yaw37_translation': gauge()}
    selected, histories, lengths, originals, changeds = [], [], [], [], []
    full, bindings = [], {}
    for split in ('cal', 'validation'):
        folder = OLD/'data'/split
        with np.load(folder/'geometry.npz', allow_pickle=False) as archive:
            sensor, query, category = (archive[k] for k in ('sensor', 'public_query', 'category'))
        assert sensor.shape == query.shape == (16, 4, 4)
        for name, change in changes.items():
            original, changed, w, wg, qg = chains(sensor, query, change)
            np.testing.assert_allclose(original, changed, atol=1e-12, rtol=0)
            np.testing.assert_allclose(query, qg, atol=1e-12, rtol=0)
            point_error, label_checks = 0., 0
            for row in rows[split]:
                cp = corners(row['boxes'])
                shifted = np.einsum('ij,bpj->bpi', change, cp)
                a = np.einsum('fij,bpj->fbpi', w, cp)
                b = np.einsum('fij,bpj->fbpi', wg, shifted)
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
                point_error = max(point_error, float(np.max(np.abs(a-b))))
                for f in range(16):
                    # Transform oriented world corners into query coordinates
                    # first. Bounding boxes here are exact because the query-
                    # frame result is the original axis-aligned fixture.
                    qboxes = [dict(lo=pts[:, :3].min(0), hi=pts[:, :3].max(0)) for pts in b[f]]
                    before = D.category_boxes(row['boxes'], sensor[f, :3, 3])
                    after = D.category_boxes(qboxes, np.zeros(3))
                    assert before == after, (split, row['scene_id'], f, before, after)
                    if f >= 3:
                        assert before == category[row['scene_id']].tolist()
                    label_checks += 2
            full.append(dict(split=split, gauge=name, current_past_chains=16*16,
                scalar_chain_entries=16*16*16, scene_pose_query_label_checks=label_checks,
                chain_max_abs=float(np.max(np.abs(original-changed))),
                public_query_max_abs=float(np.max(np.abs(query-qg))), scene_corner_max_abs=point_error))
        sources = {k: np.load(folder/(k+'.npy'), mmap_mode='r', allow_pickle=False)
                   for k in ('histories', 'transforms', 'transforms_yaw3', 'length')}
        # Audit every saved valid transform, not just representative rows.
        for frame in range(3, 16):
            begin = max(0, frame-7)
            le, pad = frame-begin+1, 8-(frame-begin+1)
            ix = np.arange(frame-3, len(sources['length']), 13)
            assert np.all(sources['length'][ix] == le)
            base = query[frame]@np.linalg.inv(sensor[frame])@sensor[begin:frame+1]
            for name, expected in (('transforms', base), ('transforms_yaw3', rotation(3.)@base)):
                np.testing.assert_allclose(sources[name][ix, pad:],
                    np.broadcast_to(expected, (len(ix), le, 4, 4)), atol=1e-12, rtol=0)
        background_ids = sorted({r['background_id'] for r in rows[split]})
        for bg_position, background_id in enumerate(background_ids):
            for q in (0, 1):
                for ci, desired in enumerate(('contact', 'pass')):
                    row = next(r for r in rows[split] if r['background_id'] == background_id
                               and category[r['scene_id'], q] == desired)
                    frame = 3 if (bg_position+q+ci) % 2 == 0 else 13
                    index = row['scene_id']*4*13 + frame-3  # fixed noise replica0
                    le = int(sources['length'][index]); pad = 8-le; begin = frame-le+1
                    originals.append(np.stack([sources[k][index] for k in ('transforms', 'transforms_yaw3')]))
                    per_branch = []
                    for yaw in (0., 3.):
                        o, g, *_ = chains(sensor, rotation(yaw)@query, gauge())
                        valid = g[frame, begin:frame+1]
                        np.testing.assert_allclose(valid, o[frame, begin:frame+1], atol=1e-12, rtol=0)
                        matrix = np.repeat(np.eye(4)[None], 8, axis=0)
                        matrix[pad:] = valid
                        per_branch.append(matrix)
                    changeds.append(np.stack(per_branch))
                    histories.append(sources['histories'][index]); lengths.append(le)
                    selected.append(dict(split=split, scene_id=row['scene_id'], background_id=background_id,
                        background_family=row['background_family'], selected_query=q, category=desired,
                        frame=frame, replica=0, native_row=index))
        bindings[split] = {name: dict(size=(folder/name).stat().st_size,
            mtime_ns=(folder/name).stat().st_mtime_ns) for name in
            ('geometry.npz', 'histories.npy', 'transforms.npy', 'transforms_yaw3.npy', 'length.npy')}
        del sources
    assert len(selected) == 32 and sorted(set(lengths)) == [4, 8]
    np.savez_compressed(out/'representatives.npz', histories=np.stack(histories),
        lengths=np.asarray(lengths), original=np.stack(originals), gauge=np.stack(changeds))
    save(out/'selected_rows.json', selected)
    save(out/'prepare_receipt.json', dict(status='PASS', cpu_only=True, seconds=time.monotonic()-started,
        gauge=gauge().tolist(), all_pose_checks=full, selected_histories=len(selected),
        frames=[3, 13], branches=['ideal', 'actual_query_yaw_plus3'], replicas=[0],
        source_sha256=sha(__file__), checkpoint_sha256=sha(CHECKPOINT),
        checkpoint_metadata=read(CHECKPOINT.with_suffix('.json')),
        representative_sha256=sha(out/'representatives.npz'), source_array_bindings=bindings,
        limitations='Coordinate-gauge invariance only; not physical sensor turns, true query-angle accuracy, symbols or timing. No photon draws.'))
    print('COORDINATE_PREPARED', round(time.monotonic()-started, 3), flush=True)


def run():
    """Root-serialized CUDA feature/model/projection check after PLAN freeze."""
    started = time.monotonic()
    out = OUT/'coordinate_equivalence'
    assert (OUT/'PLAN.json').is_file(), 'Root must freeze PLAN before CUDA run'
    assert not (out/'result.json').exists(), 'Preserve completed result'
    prep = read(out/'prepare_receipt.json')
    assert prep['status'] == 'PASS'
    preparation_source = out/'prepare_source_snapshot.py'
    assert prep['source_sha256'] == sha(preparation_source)
    assert prep['checkpoint_sha256'] == sha(CHECKPOINT)
    assert prep['representative_sha256'] == sha(out/'representatives.npz')
    import torch
    import cnh_boundary_token_model as M
    import cnh_counterfactual_train_dev as T
    import cnh_aligned_boundary_dev as B
    import cnh_bar_local_readout_dev as L
    import cnh_pass_common_dev as C
    stage = C.Stage('coordinate_equivalence')
    stage.start = started  # Include imports and bindings in command-stage wall.
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine = None
    def check():
        stage.check()
        if time.monotonic()-started >= 120:
            raise TimeoutError('Coordinate diagnostic wall120s')
    try:
        with np.load(out/'representatives.npz', allow_pickle=False) as archive:
            native = {k: archive[k] for k in archive.files}
        model = M.BoundaryTokenReadout().to('cuda').eval()
        model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=False)['state_dict'])
        z = torch.as_tensor(native['histories'], device='cuda')
        lengths = torch.as_tensor(native['lengths'], device='cuda')
        scores, features, records = {}, {}, []
        with torch.no_grad():
            for branch in range(2):
                for coordinate in ('original', 'gauge'):
                    check()
                    transforms = torch.as_tensor(native[coordinate][:, branch], device='cuda')
                    geometry = M.feature_geometry(transforms, lengths, 'center')
                    compact = M.build_features(z, geometry, lengths)[:, :, :, list(T.KEPT_CHANNELS)].half()
                    features[branch, coordinate] = compact.cpu().numpy()
                    scores[branch, coordinate] = model(T.restore_features(compact), lengths).cpu().numpy()
                np.testing.assert_array_equal(features[branch, 'original'], features[branch, 'gauge'])
                np.testing.assert_array_equal(scores[branch, 'original'], scores[branch, 'gauge'])
                records.append(dict(branch=['ideal', 'yaw_plus3'][branch], compact_FP16_exact=True,
                    token_score_exact=True, compact_max_abs=0., score_max_abs=0.))
        A, *_ = B.imports()
        A.OUT = out/'m3_runtime'
        engine = A.Engine()
        selected = read(out/'selected_rows.json')
        projections, m3scores, projection_records = {}, {}, []
        for branch in range(2):
            all_features, all_scores = {}, {}
            for coordinate in ('original', 'gauge'):
                current_features, current_scores = {}, {}
                for frame in (3, 13):
                    check()
                    indices = [i for i, row in enumerate(selected) if row['frame'] == frame]
                    i = indices[0]
                    le = int(native['lengths'][i]); pad = 8-le
                    for j in indices:
                        np.testing.assert_array_equal(native[coordinate][j, branch], native[coordinate][i, branch])
                    projection = L.Projection(engine, native[coordinate][i, branch, pad:])
                    value = projection(native['histories'][indices, pad:]).cpu().numpy().astype(np.float16)
                    prediction = engine.predict(value)
                    for j, index in enumerate(indices):
                        current_features[index] = value[j]; current_scores[index] = prediction[j]
                    del projection
                all_features[coordinate] = np.stack([current_features[i] for i in range(len(selected))])
                all_scores[coordinate] = np.stack([current_scores[i] for i in range(len(selected))])
            np.testing.assert_array_equal(all_features['original'], all_features['gauge'])
            np.testing.assert_array_equal(all_scores['original'], all_scores['gauge'])
            projections[branch] = all_features['original']; m3scores[branch] = all_scores['original']
            projection_records.append(dict(branch=['ideal', 'yaw_plus3'][branch],
                projection_FP16_exact=True, m3_score_exact=True, projection_max_abs=0., score_max_abs=0.))
        # Check existing saved raw scores rather than recomputing thresholds.
        raw_check = []
        for branch in range(2):
            suffix = ['ideal', 'yaw_plus3'][branch]
            candidate_cache, m3_cache = {}, {}
            token_max_error, m3_max_error = 0., 0.
            for split in ('cal', 'validation'):
                candidate_suffix = '' if branch == 0 else '_yaw_plus3'
                with np.load(OLD/f'scores/ordinary_seed{SEED}_{split}{candidate_suffix}.npz') as archive:
                    candidate_cache[split] = archive['raw']
                with np.load(OLD/f'baselines/{split}_{suffix}.npz') as archive:
                    m3_cache[split] = archive['m3_raw']
            for i, row in enumerate(selected):
                ix = row['scene_id'], row['replica'], row['frame']-3
                token_reference, m3_reference = candidate_cache[row['split']][ix], m3_cache[row['split']][ix]
                # Existing inference had different batch sizes. FP32 CUDA
                # reduction algorithms can differ; this check binds numerical
                # replay, while original-vs-gauge above uses identical batches.
                np.testing.assert_allclose(scores[branch, 'original'][i], token_reference, atol=1e-5, rtol=0)
                np.testing.assert_allclose(m3scores[branch][i], m3_reference, atol=1e-5, rtol=0)
                token_max_error = max(token_max_error, float(np.max(np.abs(scores[branch, 'original'][i]-token_reference))))
                m3_max_error = max(m3_max_error, float(np.max(np.abs(m3scores[branch][i]-m3_reference))))
            raw_check.append(dict(branch=suffix, existing_raw_replay_tolerance=1e-5,
                token_max_abs=token_max_error, m3_max_abs=m3_max_error, histories=32))
        token_difference = scores[1, 'original']-scores[0, 'original']
        m3_difference = m3scores[1]-m3scores[0]
        assert np.any(token_difference != 0) and np.any(m3_difference != 0)
        np.savez_compressed(out/'scores.npz', token_ideal=scores[0, 'original'],
            token_yaw_plus3=scores[1, 'original'], m3_ideal=m3scores[0], m3_yaw_plus3=m3scores[1])
        save(out/'result.json', dict(status='PASS', seconds=time.monotonic()-started,
            plan_sha256=sha(OUT/'PLAN.json'), source_sha256=sha(__file__),
            preparation_source_sha256=sha(preparation_source),
            checkpoint_sha256=sha(CHECKPOINT), selected_histories=32,
            token_checks=records, projection_M3_checks=projection_records, saved_raw_checks=raw_check,
            actual_query_yaw_plus3=dict(token_changed_scores=int(np.count_nonzero(token_difference)),
                token_max_abs=float(np.max(np.abs(token_difference))),
                m3_changed_scores=int(np.count_nonzero(m3_difference)),
                m3_max_abs=float(np.max(np.abs(m3_difference)))),
            device=torch.cuda.get_device_name(), new_photons=0, training_steps=0,
            interpretation='World gauge preserves physical scene/query labels and all checked features/scores. Actual +3deg query differs; no physical-turn, symbol, time or real-hardware equivalence claim.'))
        print('COORDINATE_RUN_PASS', round(time.monotonic()-started, 3), flush=True)
        stage.finish('COMPLETE', source_sha256=sha(__file__), selected_histories=32)
    except BaseException as error:
        save(out/f'failure_{time.time_ns()}.json', dict(error=repr(error), seconds=time.monotonic()-started))
        stage.finish('FAILED', source_sha256=sha(__file__), error=repr(error))
        raise
    finally:
        if engine is not None:
            engine.nets = []; engine.projector = None
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else run)()
