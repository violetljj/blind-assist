"""One compact spatial fit; identical OPEN/GATED weights and fixed calibration."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from data_lightweight import CompactSource
from mz5_ensemble_readout import read, write, sha, load_npz
from mz15_train import cutoff_zero_added
from mz45_object_transfer import Bindings, FrozenModels, CONDITIONS, setup
from mz47_local_enrichment import prepare as prepare_legacy, packet, bound_outputs
from mz50_echo_independent import SpatialQuery, loss_for
from mz50_source import load_source

READOUTS = ('OPEN', 'GATED')
BASELINES = ('rgb', 'tof', 'MZ5', 'MZ37', 'MZ43_ENSEMBLE')


def split_source(records):
    heldout = np.array([r['role'] == 'HELDOUT_SITE' for r in records])
    family = ~heldout & np.array([r['family'] == 'birch' for r in records])
    calibration = ~heldout & ~family & np.array([r['site_id'] == 'mz36_dense_candidate_06_site_002' for r in records])
    fit = ~heldout & ~family & ~calibration
    groups = dict(fit=np.flatnonzero(fit), calibration=np.flatnonzero(calibration),
                  heldout_site=np.flatnonzero(heldout), nonfit_family=np.flatnonzero(family))
    assert [len(groups[k]) for k in groups] == [1280, 256, 640, 384]
    assigned = {}
    for name, ids in groups.items():
        for i in ids:
            pair = records[i]['pair_id']
            assert assigned.setdefault(pair, name) == name
    assert len(assigned) == 1280
    return groups


def prepare_new(source, models, out):
    p = source['predictor']
    dense = np.empty((2560, 64, 28, 28), dtype=np.float32)
    arrays = {c + '/' + k: [] for c in CONDITIONS for k in BASELINES}
    started = time.perf_counter()
    with ExitStack() as stack, torch.inference_mode():
        archives = {path: stack.enter_context(CompactSource(path)) for path in sorted({row.archive for row in p['rgb_refs']})}
        for start in range(0, 2560, 16):
            ids = np.arange(start, min(start + 16, 2560))
            images = [archives[p['rgb_refs'][i].archive].load_image(p['rgb_refs'][i].member) for i in ids]
            visual, detail = models.visual(images)
            dense[ids] = detail.cpu().numpy()
            for condition in CONDITIONS:
                observed = packet(p['ranges'][ids], p['valid'][ids], condition)
                predictions = models.predict(visual, detail, observed['ranges'], observed['valid'])
                for key in BASELINES:
                    arrays[condition + '/' + key].append(predictions[key])
            if start % 512 == 0:
                write(out / 'progress.json', dict(stage='new-feature-extraction', frames=start + len(ids), total=2560))
                print('FEATURES', start + len(ids), flush=True)
    return dense, {k: np.concatenate(v) for k, v in arrays.items()}, time.perf_counter() - started


def infer(model, dense, ranges, valid, models):
    x = (torch.from_numpy(dense).cuda() - models.detail_mean) / models.detail_std
    out = model.inspect(x, torch.from_numpy(ranges).cuda(), torch.from_numpy(valid).cuda())
    values = {}
    for name in READOUTS:
        for key, value in out[name].items():
            values[name + '/' + key] = value.cpu().numpy()
        field = out['field'] if name == 'OPEN' else out['field'].masked_fill(~out['eligible'], -torch.inf)
        values[name + '/winner'] = field.flatten(1, 2).argmax(1).cpu().numpy().astype(np.int16)
    return values


def run(root, task):
    out = task / 'run-v1'
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bind = Bindings()
    for name in ('MZ50_ECHO_INDEPENDENT_LOCAL_20260911.md', 'mz50_echo_independent.py', 'mz50_source.py', 'mz50_train.py'):
        bind(Path(__file__).with_name(name))
    source_task = root / 'artifacts.local/work/mz48-rich-kilotier-20260911'
    index = bind(source_task / 'source-index.json')
    source = load_source(source_task, index)
    # The source loader verifies every referenced immutable artifact. Bind its
    # index and receipt rather than passing evaluator fields to the predictor.
    bind(source_task / 'source-total-receipt.json')
    records = source['records']
    groups = split_source(records)
    p = source['predictor']
    truth, known = source['truth'], source['known']
    cell_truth, cell_known = source['cell_event_presence'], source['cell_known']
    setup()
    torch.manual_seed(150)
    models = FrozenModels(root, bind)
    legacy_out = task / 'legacy-preparation-v1'; legacy_out.mkdir(exist_ok=False)
    legacy = prepare_legacy(root, legacy_out, bind, models)
    prior = root / 'artifacts.local/work/mz47-local-enrichment-20260911/run-v1'
    old = load_npz(bound_outputs(bind, prior, ['predictions.npz'])[0])
    dense, baseline, feature_seconds = prepare_new(source, models, out)
    write(out / 'groups.json', dict(records=records, groups={k: v.tolist() for k, v in groups.items()}))
    generator = np.random.default_rng(150)
    batches = generator.choice(groups['fit'], size=(600, 16), replace=True)
    np.save(out / 'batches.npy', batches)
    model = SpatialQuery(models.rank).cuda().train()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / 'initial.pt')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    write(out / 'start.json', dict(status='STARTED', seed=150, total_steps=600,
        batch_size=16, source_frames=2560, split={k: len(v) for k, v in groups.items()},
        inputs=bind.inputs, dense_cache_files_created=0, device=torch.cuda.get_device_name()))
    torch.cuda.synchronize(); fit_start = time.perf_counter()
    history = []
    for step, ids in enumerate(batches):
        condition = CONDITIONS[step % 3]
        obs = packet(p['ranges'][ids], p['valid'][ids], condition)
        x = (torch.from_numpy(dense[ids]).cuda() - models.detail_mean) / models.detail_std
        prediction = model.inspect(x, torch.from_numpy(obs['ranges']).cuda(), torch.from_numpy(obs['valid']).cuda())
        loss, parts = loss_for(prediction, torch.from_numpy(cell_truth[ids]).cuda(),
            torch.from_numpy(cell_known[ids]).cuda(), torch.from_numpy(truth[ids]).cuda(), torch.from_numpy(known[ids]).cuda())
        assert torch.isfinite(loss)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 100 == 0 or step == 599:
            row = dict(step=step + 1, condition=condition, loss=float(loss.detach()),
                       **{k: float(v.detach()) for k, v in parts.items()})
            history.append(row); write(out / 'progress.json', dict(stage='fit', **row)); print('FIT', row, flush=True)
    torch.cuda.synchronize(); fit_seconds = time.perf_counter() - fit_start
    model.eval()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / 'spatial.pt')
    predictions = {'mz48/truth': truth, 'mz48/known': known,
                   'mz48/frame_ids': np.array([r['frame_id'] for r in records])}
    with torch.inference_mode():
        for cohort in ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48'):
            if cohort == 'mz48':
                ids = np.arange(2560); rr, vv = p['ranges'], p['valid']
                get_dense = lambda ii: dense[ii]
            elif cohort in ('rich', 'mz36'):
                data = legacy[cohort]; ids = np.arange(len(data['truth']))
                rr, vv = data['ranges'], data['valid']; get_dense = lambda ii: data['dense'][ii]
            else:
                ids = legacy['cohorts'][cohort]; rr, vv = legacy['ranges'], legacy['valid']
                get_dense = lambda ii: np.array(legacy['maps'][legacy['lookup'][ii]])
            if cohort != 'mz48':
                for key in ('frame_ids', 'truth', 'known'):
                    predictions[cohort + '/' + key] = old[cohort + '/' + key]
                if cohort == 'mz36':
                    for key in ('attempted_frame_ids', 'attempted_truth', 'attempted_known', 'admitted_index'):
                        predictions[cohort + '/' + key] = old[cohort + '/' + key]
                np.testing.assert_array_equal(predictions[cohort + '/truth'],
                                              legacy[cohort]['truth'] if cohort in ('rich', 'mz36') else legacy['truth'][ids])
            for condition in CONDITIONS:
                parts = []
                for start in range(0, len(ids), 16):
                    ii = ids[start:start + 16]
                    observed = packet(rr[ii], vv[ii], condition)
                    parts.append(infer(model, get_dense(ii), observed['ranges'], observed['valid'], models))
                prefix = cohort + '/' + condition + '/'
                for key in parts[0]: predictions[prefix + key] = np.concatenate([row[key] for row in parts])
                for key in BASELINES:
                    original_key = 'FROZEN/MZ37' if key == 'MZ37' else key
                    predictions[prefix + key] = baseline[condition + '/' + key] if cohort == 'mz48' else old[prefix + original_key]
                if cohort == 'mz48':
                    flat = cell_truth.reshape(2560, 64 * 49, 4)
                    for name in READOUTS:
                        winner = predictions[prefix + name + '/winner']
                        predictions[prefix + name + '/winning_native'] = np.take_along_axis(flat, winner[:, None], 1)[:, 0] & predictions[prefix + name + '/support']
                print('INFER', cohort, condition, len(ids), flush=True)
    cuts = {}
    for name in READOUTS:
        def collect(key):
            return np.concatenate([predictions['DEV/DROP_CLOSE/' + key], predictions['mz48/DROP_CLOSE/' + key][groups['calibration']]])
        y = np.concatenate([old['DEV/truth'], truth[groups['calibration']]])
        cut = cutoff_zero_added(collect(name + '/raw'), collect(name + '/support'), collect('MZ37'), y)
        np.save(out / (name + '-cutoff.npy'), cut); cuts[name] = cut.tolist()
        for cohort in ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48'):
            for condition in CONDITIONS:
                prefix = cohort + '/' + condition + '/'
                margin = predictions[prefix + name + '/raw'].astype(float) - cut
                accepted = (predictions[prefix + 'MZ37'] < 0) & predictions[prefix + name + '/support'] & (margin >= 0)
                predictions[prefix + name + '/candidate'] = np.where(accepted, margin, predictions[prefix + 'MZ37'])
    np.savez_compressed(out / 'predictions.npz', **predictions)
    bind.check()
    receipt = dict(status='PASS', inputs=bind.inputs, source_index_sha256=sha(index),
        total_steps=600, fits=1, readouts_same_weights=True, seed=150, parameter_count=sum(p.numel() for p in model.parameters()),
        backend='CUDA; CPU compact I/O and metadata', device=torch.cuda.get_device_name(),
        feature_seconds=feature_seconds, fit_seconds=fit_seconds, seconds=time.perf_counter() - started,
        history=history, cutoffs=cuts, fit_unique_source_ids=int(len(np.unique(batches))),
        full_source_frames=2560, dense_cache_files_created=0,
        new_feature_bytes_in_ram=int(dense.nbytes), reused_old_mmap_bytes=int(legacy['maps'].nbytes),
        nondeterminism='Single fit; inherited CUDA grid_sample backward is not bitwise reproducible',
        predictor_fields=['RGB features', 'ranges', 'validity', 'fixed calibration'],
        local_unknown_masked=True, merge_echo_labels_not_used=True,
        outputs={f.name: sha(f) for f in out.iterdir() if f.is_file()})
    write(out / 'receipt.json', receipt)
    print('COMPLETE', {k: receipt[k] for k in ('status', 'total_steps', 'fit_seconds', 'seconds', 'parameter_count')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--task', type=Path, required=True)
    a = parser.parse_args()
    try:
        run(a.root.resolve(), a.task.resolve())
    except Exception:
        a.task.mkdir(parents=True, exist_ok=True)
        write(a.task / 'failure.json', dict(status='FAIL', error=traceback.format_exc()))
        raise
