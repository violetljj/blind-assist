"""Matched native-local enrichment; fixed limited packets and consumed Development."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from data_lightweight import CompactSource
from mz5_ensemble_readout import read, write, sha, load_npz
from mz9_contributors import reconstruct
from mz15_shared_support import LocalSupportReadout
from mz15_train import balanced_local, cutoff_zero_added
from mz20_rank_loss import body_rank_query_loss
from mz23_availability import restrict_candidates
from mz30_select import negative_branch_confidence
from mz37_restore import compose
from mz40_packets import constrain
from mz40_evaluate import metrics, paired
from mz45_object_transfer import Bindings, FrozenModels, CONDITIONS, setup

ARMS = ('REPLAY', 'ENRICH')
BRIEF = 'MZ47_LOCAL_ENRICHMENT_20260911.md'


def bound_outputs(bind, folder, names):
    receipt = read(bind(folder / 'receipt.json'))
    assert receipt['status'] == 'PASS'
    hashes = receipt.get('outputs', receipt.get('files', {}))
    return [bind(folder / name, hashes[name]) for name in names]


def packet(ranges, valid, condition):
    return dict(ranges=ranges, valid=valid) if condition == 'IDEAL' else constrain(ranges, valid, condition)[0]


def prepare(root, out, bind, models):
    """Reuse the old mmap; hold new dense features only in RAM."""
    work = root / 'artifacts.local/work'
    old = work / 'mz8-attribution-20260910/cache-v5'
    new = work / 'mz15-shared-support-20260910/cache-v1'
    old_obs, old_eval = map(load_npz, bound_outputs(bind, old, ['observations.npz', 'evaluator.npz']))
    new_obs, new_eval = map(load_npz, bound_outputs(bind, new, ['observations.npz', 'evaluator.npz']))
    label_folder = work / 'mz9-source-supervision-20260910/labels-v1'
    lr = read(bind(label_folder / 'receipt.json'))
    old_local = load_npz(bind(label_folder / 'evaluator.npz', lr['labels_sha256']))
    truth = np.concatenate([old_eval['truth'], new_eval['truth']])
    ranges, valid = (np.concatenate([old_obs[k], new_obs[k]]) for k in ('ranges', 'valid'))
    query = np.concatenate([old_local['query_counts'] > 0, new_eval['query_presence']])
    known_local = np.concatenate([old_local['cell_known_counts'] > 0, new_eval['known']])
    del old_local, new_eval
    cache = work / 'mz16-visual-detail-20260910/cache-v2'
    dense_path, ids_path = bound_outputs(bind, cache, ['dense_HIGH_DETAIL.npy', 'ids.npy'])
    maps = np.load(dense_path, mmap_mode='r'); map_ids = np.load(ids_path)
    lookup = np.full(len(truth), -1, dtype=np.int64); lookup[map_ids] = np.arange(len(map_ids))
    branches = load_npz(bound_outputs(bind, work / 'mz30-branch-responsibility-20260910/run-v1', ['branches.npz'])[0])
    np.testing.assert_array_equal(branches['global_ids'], map_ids)
    rh = branches['normal/features'][:, :128].copy()
    batches = np.load(bound_outputs(bind, work / 'mz20-rank-objective-20260910/run-v1', ['batches.npy'])[0])[:300].copy()
    train_ids = branches['train_ids']; assert len(train_ids) == 7562
    assert np.isin(batches, train_ids).all()
    cohorts = {name: branches['ids/' + name] for name in ('DEV', 'relation10000', 'distance5000')}
    for ids in cohorts.values(): assert not np.isin(ids, train_ids).any()

    frames, groups, truths, knowns, rr, vv, refs = [], [], [], [], [], [], []
    for eid, take in [('mz45-object-transfer-20260911', slice(None)), ('mz46-scaffold-context-20260911', slice(4, 8))]:
        folder = work / eid / 'prepared-v1'
        pp, ep, mp, gp = bound_outputs(bind, folder, ['packets.npz', 'evaluator.npz', 'predictor.json', 'groups.json'])
        p, e = load_npz(pp), load_npz(ep)
        manifest, metadata = read(mp)['frames'], read(gp)['records']
        np.testing.assert_array_equal(p['frame_ids'], e['frame_ids'])
        np.testing.assert_array_equal(p['frame_ids'], [r['frame_id'] for r in manifest])
        frames.extend(manifest[take]); groups.extend(metadata[take])
        truths.append(e['truth'][take]); knowns.append(e['known'][take]); rr.append(p['ranges'][take]); vv.append(p['valid'][take])
        refs.append((load_npz(bound_outputs(bind, work / eid / 'inference-v1', ['predictions.npz'])[0]), take))
    for row in groups[40:]: row.update(family='oblique_rod', block='unsupported')
    rich = dict(frame_ids=np.array([r['frame_id'] for r in frames]), truth=np.concatenate(truths), known=np.concatenate(knowns),
        ranges=np.concatenate(rr), valid=np.concatenate(vv), groups=groups)
    assert len(set(rich['frame_ids'])) == 44 and rich['known'].all()
    rich['reference'] = {c + '/' + k: np.concatenate([r[c + '/' + k][take] for r, take in refs])
        for c in CONDITIONS for k in ('rgb', 'tof', 'MZ5', 'MZ28', 'MZ35', 'MZ37', 'MZ43_ENSEMBLE', 'original_raw', 'restricted_raw')}
    native_by_rgb = {}
    for eid, name in [('mz42-rich-objects-20260910', 'result.json'), ('mz44-rich-far-objects-20260911', 'range-result.json'),
                      ('mz46-scaffold-context-20260911', 'result.json')]:
        folder = work / eid / 'returned-v1/dataset-v1'
        for row in read(bind(folder / name))['records']:
            native_by_rgb[(str(work / eid / 'returned-v1/capture.source.zip'), row['rgb'])] = row['native']
    dparts, hparts, qparts, kparts = [], [], [], []
    with ExitStack() as stack, torch.inference_mode():
        sources = {p: stack.enter_context(CompactSource(bind(p))) for p in sorted({r['archive'] for r in frames})}
        # Path spelling in source manifests may be canonical F: or the repository junction.
        native_lookup = {(str(Path(p).resolve()), rgb): native for (p, rgb), native in native_by_rgb.items()}
        for start in range(0, 44, 16):
            rows = frames[start:start + 16]
            images = [sources[r['archive']].load_image(r['rgb']) for r in rows]
            visual, detail = models.visual(images)
            dparts.append(detail.cpu().numpy()); hparts.append(models.original.rgb[:2](visual).cpu().numpy())
            depths = np.stack([sources[r['archive']].load_array(native_lookup[(str(Path(r['archive']).resolve()), r['rgb'])]) for r in rows])
            lab = reconstruct(torch.from_numpy(depths).cuda())
            np.testing.assert_array_equal(lab['valid'].cpu().numpy(), rich['valid'][start:start + len(rows)])
            np.testing.assert_allclose(torch.nan_to_num(lab['range_m']).float().cpu().numpy(), rich['ranges'][start:start + len(rows)], rtol=0, atol=1e-6)
            qparts.append((lab['query_counts'] > 0).cpu().numpy()); kparts.append((lab['cell_known_counts'] > 0).cpu().numpy())
    rich.update(dense=np.concatenate(dparts), rh=np.concatenate(hparts), query=np.concatenate(qparts), local_known=np.concatenate(kparts))
    fit_ids = np.array([i for i, r in enumerate(groups) if r['block'] == 'near' and r['family'] in ('pipe', 'ladder', 'pouch')])
    np.testing.assert_array_equal(fit_ids, np.arange(12))
    generator = np.random.default_rng(147)
    enrich_batches = np.stack([fit_ids[(i % 3) * 4:(i % 3 + 1) * 4] for i in range(300)])
    codes = (truth.astype(int) * (1 << np.arange(4))).sum(1)
    extras = []
    for ii in enrich_batches:
        row = []
        for y in rich['truth'][ii]:
            code = (y.astype(int) * (1 << np.arange(4))).sum()
            candidates = train_ids[codes[train_ids] == code]; assert len(candidates)
            row.append(generator.choice(candidates))
        extras.append(row)
    replay_batches = np.array(extras)
    np.testing.assert_array_equal(truth[replay_batches], rich['truth'][enrich_batches])
    np.savez_compressed(out / 'schedule.npz', old=batches, replay=replay_batches, enrich=enrich_batches, train_ids=train_ids)
    np.savez_compressed(out / 'rich-local-supervision.npz', frame_ids=rich['frame_ids'], truth=rich['truth'], known=rich['known'],
        query=rich['query'], local_known=rich['local_known'], ranges=rich['ranges'], valid=rich['valid'])
    write(out / 'groups.json', dict(records=groups))

    source36 = work / 'mz36-new-source-20260910'
    pred36 = load_npz(bound_outputs(bind, source36 / 'inference-v1', ['predictions.npz'])[0])
    manifest36 = read(bind(source36 / 'admission-v1/predictor-manifest.json'))['frames']
    np.testing.assert_array_equal(pred36['frame_ids'], [r['frame_id'] for r in manifest36])
    evaluator36 = load_npz(bind(source36 / 'admission-v1/evaluator.npz'))
    index36 = np.array([list(evaluator36['frame_ids']).index(i) for i in pred36['frame_ids']])
    assert len(index36) == 380 and (~evaluator36['known']).sum() == 80
    dd = []
    with torch.inference_mode():
        for start in range(0, 380, 16):
            images = []
            for row in manifest36[start:start + 16]:
                with Image.open(bind(row['rgb_path'])) as im: images.append(im.copy())
            _, detail = models.visual(images); dd.append(detail.cpu().numpy())
    mz36 = dict(dense=np.concatenate(dd), rh=pred36['features'][:, :128], ranges=pred36['ranges'], valid=pred36['valid'],
        frame_ids=pred36['frame_ids'], truth=evaluator36['truth'][index36], known=evaluator36['known'][index36],
        attempted=evaluator36, admitted_index=index36, reference=pred36)
    write(out / 'preparation.json', dict(status='PASS', old_mmap_bytes=dense_path.stat().st_size,
        new_rgb_feature_frames=424, dense_files_created=0, fitted_new_ids=rich['frame_ids'][fit_ids].tolist(),
        original_train_ids=7562, old_training_rows=300 * 16, extra_rows_per_arm=1200,
        exact_extra_truth_match=True, mz36_attempts=400, mz36_admitted=380, mz36_unknown_bits=80))
    return dict(maps=maps, lookup=lookup, rh=rh, ranges=ranges, valid=valid, truth=truth, query=query, local_known=known_local,
        cohorts=cohorts, batches=batches, replay_batches=replay_batches, enrich_batches=enrich_batches, rich=rich, mz36=mz36)


def get(data, models, ids, condition, rich=False):
    source = data['rich'] if rich else data
    dense = source['dense'][ids] if rich else np.array(source['maps'][source['lookup'][ids]])
    p = packet(source['ranges'][ids], source['valid'][ids], condition)
    return ((torch.from_numpy(dense).cuda() - models.detail_mean) / models.detail_std,
        torch.from_numpy(p['ranges']).cuda(), torch.from_numpy(p['valid']).cuda())


def common(models, rh, ranges, valid):
    th = models.original.tof[:2](torch.cat([ranges.flatten(1) / 4, valid.flatten(1).float()], 1))
    rgb, tof = models.original.rgb[2](rh), models.original.tof[2](th)
    selector = models.selector((torch.cat([rh, th, rgb, tof], 1) - models.selector_mean) / models.selector_std)
    # MZ43's RGB branch is identical; its ToF branch is a fixed strong comparator.
    mixed_tof = models.mixed.tof(torch.cat([ranges.flatten(1) / 4, valid.flatten(1).float()], 1))
    a = {k: v.cpu().numpy() for k, v in dict(rgb=rgb, tof=tof, MZ5=(rgb + tof) * .5,
        MZ43_ENSEMBLE=(rgb + mixed_tof) * .5, confidence=negative_branch_confidence(rgb, selector)).items()}
    return a, models.bank(ranges, valid)['availability']


def local_output(model, dense, ranges, valid, availability):
    original = model.inspect(dense, ranges, valid)
    restricted = restrict_candidates(original, availability)
    a = dict(raw=original['logits'].cpu().numpy(), support=original['support'].cpu().numpy(),
        restricted_raw=restricted['logits'].cpu().numpy(), restricted_support=restricted['support'].cpu().numpy())
    return a, original, restricted


def composed(a, local, cutoff, restore_cut, selector_cut):
    margin = np.where(local['restricted_support'], local['restricted_raw'].astype(float) - cutoff, -1e6)
    accepted = (a['MZ5'] < 0) & local['restricted_support'] & (margin >= 0)
    mz28 = np.where(accepted, margin, a['MZ5'])
    eligible = (a['MZ5'] >= 0) & ~local['support'] & ((a['rgb'] >= 0) != (a['tof'] >= 0))
    remove = eligible & (a['confidence'].astype(float) >= selector_cut)
    mz35 = np.where(remove, np.where(a['rgb'] < 0, a['rgb'], a['tof']), mz28)
    final = compose(dict(rgb=a['rgb'], tof=a['tof'], baseline=a['MZ5'], oldscore=mz35,
        support=local['support'], known=np.ones_like(mz35, bool), negative_confidence=a['confidence']), restore_cut)['candidate']
    return dict(local_before=np.where(local['support'], local['raw'].astype(float) - cutoff, -1e6),
        local_after=margin, MZ28=mz28, MZ35=mz35, MZ37=final)


def fit(data, models, out):
    models_out, cuts, stats = {}, {}, {}
    for arm in ARMS:
        torch.manual_seed(147)
        model = LocalSupportReadout(shared=False).cuda()
        model.load_state_dict(models.rank.state_dict())
        for key, value in model.state_dict().items(): torch.testing.assert_close(value, models.rank.state_dict()[key], rtol=0, atol=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        model.train(); history = []; start = time.perf_counter()
        no_candidate = np.zeros(4, int); no_witness = np.zeros(4, int)
        for step, ids in enumerate(data['batches']):
            extra_ids = data['replay_batches' if arm == 'REPLAY' else 'enrich_batches'][step]
            extra = data if arm == 'REPLAY' else data['rich']
            old_inputs = get(data, models, ids, 'DROP_CLOSE')
            extra_inputs = get(data, models, extra_ids, 'DROP_CLOSE', rich=arm == 'ENRICH')
            x, ranges, valid = (torch.cat([a, b]) for a, b in zip(old_inputs, extra_inputs))
            q = torch.from_numpy(np.concatenate([data['query'][ids], extra['query'][extra_ids]])).cuda()
            k = torch.from_numpy(np.concatenate([data['local_known'][ids], extra['local_known'][extra_ids]])).cuda()[:, :, None, :, None] & valid[:, :, :, None, None]
            q = q & valid[:, :, :, None, None]
            gt = torch.from_numpy(np.concatenate([data['truth'][ids], extra['truth'][extra_ids]])).cuda()
            output = model.inspect(x, ranges, valid)
            local = balanced_local(output['field'], q, k)
            query = body_rank_query_loss(output['field'], q, k, output['eligible'], gt, output['logits'], output['support'])
            loss = local + .25 * query; assert torch.isfinite(loss)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            no_candidate += (gt & ~output['support']).sum(0).cpu().numpy()
            no_witness += (gt & ~(q & k & output['eligible']).flatten(1, 3).any(1)).sum(0).cpu().numpy()
            if step % 100 == 0 or step == 299:
                row = dict(step=step + 1, loss=float(loss.detach()), local=float(local.detach()), query=float(query.detach()))
                history.append(row); print(arm, row, flush=True)
        torch.cuda.synchronize(); fit_seconds = time.perf_counter() - start
        model.eval().requires_grad_(False); torch.save(model.state_dict(), out / (arm + '.pt'))
        pieces = []; ids = data['cohorts']['DEV']
        with torch.inference_mode():
            for begin in range(0, len(ids), 16):
                ii = ids[begin:begin + 16]; x, r, v = get(data, models, ii, 'DROP_CLOSE')
                rh = torch.from_numpy(data['rh'][data['lookup'][ii]]).cuda()
                a, availability = common(models, rh, r, v)
                loc, _, _ = local_output(model, x, r, v, availability)
                pieces.append(dict(raw=loc['raw'], support=loc['support'], baseline=a['MZ5']))
        dev = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        cut = cutoff_zero_added(dev['raw'], dev['support'], dev['baseline'], data['truth'][ids])
        np.save(out / (arm + '-cutoff.npy'), cut)
        cuts[arm] = cut; models_out[arm] = model
        stats[arm] = dict(steps=300, parameters=sum(p.numel() for p in model.parameters()), fit_seconds=fit_seconds,
            history=history, cutoff=cut.tolist(), positive_presentations_without_candidate=no_candidate.tolist(),
            positive_presentations_without_eligible_native_witness=no_witness.tolist())
        write(out / 'fits.json', stats)
    return models_out, cuts


def evaluate(data, models, fitted, cuts, out):
    arrays = {}; parity = {}; start = time.perf_counter()
    for name in (*data['cohorts'], 'rich', 'mz36'):
        old = name in data['cohorts']; source = data if old else data[name]
        ids = data['cohorts'][name] if old else np.arange(len(source['truth']))
        arrays[name + '/truth'] = source['truth'][ids]
        arrays[name + '/known'] = np.ones((len(ids), 4), bool) if old else source['known'][ids]
        arrays[name + '/frame_ids'] = ids if old else source['frame_ids']
        if name == 'mz36':
            arrays[name + '/attempted_truth'] = source['attempted']['truth']
            arrays[name + '/attempted_known'] = source['attempted']['known']
            arrays[name + '/attempted_frame_ids'] = source['attempted']['frame_ids']
            arrays[name + '/admitted_index'] = source['admitted_index']
        for condition in CONDITIONS:
            pieces = []
            with torch.inference_mode():
                for begin in range(0, len(ids), 16):
                    ii = ids[begin:begin + 16]
                    if old:
                        x, r, v = get(data, models, ii, condition)
                        rh = torch.from_numpy(data['rh'][data['lookup'][ii]]).cuda()
                    else:
                        p = packet(source['ranges'][ii], source['valid'][ii], condition)
                        x = (torch.from_numpy(source['dense'][ii]).cuda() - models.detail_mean) / models.detail_std
                        r, v, rh = (torch.from_numpy(a).cuda() for a in (p['ranges'], p['valid'], source['rh'][ii]))
                    a, availability = common(models, rh, r, v)
                    for label, model in [('FROZEN', models.rank), *fitted.items()]:
                        local, original, restricted = local_output(model, x, r, v, availability)
                        local.update(composed(a, local, models.rank_cut if label == 'FROZEN' else cuts[label], models.restore_cut, models.selector_cut))
                        if name == 'rich' and condition != 'MERGE_CLOSE':
                            query = torch.from_numpy(source['query'][ii]).cuda() & v[:, :, :, None, None]
                            local['native_witness_before'] = (query & original['eligible']).flatten(1, 3).any(1).cpu().numpy()
                            local['native_witness_after'] = (query & restricted['eligible']).flatten(1, 3).any(1).cpu().numpy()
                        a.update({label + '/' + k: value for k, value in local.items()})
                    pieces.append(a)
            combined = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
            if name == 'rich':
                for method in ('rgb', 'tof', 'MZ5', 'MZ43_ENSEMBLE', 'MZ28', 'MZ35', 'MZ37', 'original_raw', 'restricted_raw'):
                    key = ('FROZEN/' + {'original_raw': 'raw'}.get(method, method)) if method.startswith('MZ2') or method in ('MZ35', 'MZ37', 'original_raw', 'restricted_raw') else method
                    expected = source['reference'][condition + '/' + method]
                    actual = combined[key]
                    np.testing.assert_array_equal(actual >= 0, expected >= 0, err_msg=condition + '/' + method)
                    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-6)
                    parity[condition + '/' + method] = dict(values=int(actual.size), max_abs=float(np.abs(actual - expected).max()))
            if name == 'mz36' and condition == 'IDEAL':
                for method in ('rgb', 'tof', 'MZ5', 'MZ28', 'MZ35'):
                    actual = combined['FROZEN/' + method if method in ('MZ28', 'MZ35') else method]
                    expected = source['reference'][method]
                    np.testing.assert_array_equal(actual >= 0, expected >= 0, err_msg='mz36/' + method)
                    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-6)
                    parity['mz36/' + method] = dict(values=int(actual.size), max_abs=float(np.abs(actual - expected).max()))
            for key, value in combined.items(): arrays[name + '/' + condition + '/' + key] = value
            print('EVALUATED', name, condition, len(ids), flush=True)
    np.savez_compressed(out / 'predictions.npz', **arrays)
    write(out / 'parity.json', dict(status='PASS', checks=parity))
    return time.perf_counter() - start


def run(root, task):
    out = task / 'run-v1'; out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); setup(); bind = Bindings()
    for name in (Path(__file__).name, BRIEF, 'mz15_shared_support.py', 'mz20_rank_loss.py', 'mz15_train.py',
                 'mz9_contributors.py', 'mz40_packets.py', 'mz45_object_transfer.py'): bind(Path(__file__).with_name(name))
    try:
        models = FrozenModels(root, bind)
        write(out / 'start.json', dict(status='STARTED', steps_per_arm=300, seed=147, backend='CUDA',
            device=torch.cuda.get_device_name(), cudnn_tf32=True, matmul_tf32=False,
            training_numerics='Unchanged MZ20 grid_sample backward; CUDA atomic reduction is not bitwise reproducible',
            protocol_sha256=sha(Path(__file__).with_name(BRIEF)), source_sha256=sha(__file__)))
        data = prepare(root, out, bind, models)
        write(out / 'input-binding.json', bind.inputs)
        fitted, cuts = fit(data, models, out)
        seconds = evaluate(data, models, fitted, cuts, out)
        bind.check()
        write(out / 'receipt.json', dict(status='PASS', inputs=bind.inputs, backend='CUDA', device=torch.cuda.get_device_name(),
            steps_per_arm=300, total_steps=600, evaluation_seconds=seconds, seconds=time.perf_counter() - started,
            dense_files_created=0, scope='Consumed Development; proxy limited packet, no hardware validity claim',
            outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
        print('RUN PASS', flush=True)
    except Exception:
        write(out / 'failure.json', dict(status='FAILED', error=traceback.format_exc())); raise
    finally:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve(); task = args.task.resolve()
    assert task.is_relative_to((root / 'artifacts.local').resolve())
    run(root, task)
