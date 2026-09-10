"""Equal-budget query-negative replay from new or old TRAIN source only."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from data_lightweight import CompactSource
from mz5_ensemble_readout import read, write, sha, load_npz
from mz15_train import cutoff_zero_added
from mz16_detail_cache import CROP
from mz36_frozen_inference import fixed_batch_dense
from mz45_object_transfer import Bindings, FrozenModels, CONDITIONS, setup
from mz47_local_enrichment import prepare as prepare_legacy, packet, bound_outputs
from mz50_echo_independent import SpatialQuery, loss_for
from mz50_source import load_source
from mz50_train import split_source, infer, READOUTS

ARMS = ('NEW_NEG', 'OLD_NEG')
COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48')


def schedule(truth_new, fit_ids, truth_old, train_ids):
    generator = np.random.default_rng(151)
    shared = generator.choice(fit_ids, size=(600, 8), replace=True)
    query = np.tile(np.repeat(np.arange(4), 2), (600, 1))
    extras = {}
    for name, truth, ids in [('NEW_NEG', truth_new, fit_ids), ('OLD_NEG', truth_old, train_ids)]:
        candidates = [ids[~truth[ids, q]] for q in range(4)]
        assert all(len(c) for c in candidates)
        extras[name] = np.stack([np.concatenate([generator.choice(c, 2, replace=True) for c in candidates]) for _ in range(600)])
        assert not truth[extras[name], query].any()
    return dict(shared=shared, query=query, **extras)


def negative_loss(prediction, query):
    raw = prediction['OPEN']['raw'].gather(1, query[:, None]).squeeze(1)
    return F.softplus(raw).mean()


def detail_only(source, models, out):
    dense = np.empty((2560, 64, 28, 28), np.float32)
    refs = source['predictor']['rgb_refs']; started = time.perf_counter()
    with ExitStack() as stack, torch.inference_mode():
        archives = {p: stack.enter_context(CompactSource(p)) for p in sorted({r.archive for r in refs})}
        for begin in range(0, 2560, 16):
            rows = refs[begin:begin + 16]
            images = np.stack([np.array(archives[r.archive].load_image(r.member).convert('RGB').crop(CROP)) for r in rows])
            rgb = torch.from_numpy(images).permute(0, 3, 1, 2).cuda().float() / 255
            dense[begin:begin + len(rows)] = fixed_batch_dense(models.context.base, rgb).cpu().numpy()
            if begin % 512 == 0:
                write(out / 'progress.json', dict(stage='detail-only', frames=begin + len(rows)))
                print('DETAIL', begin + len(rows), flush=True)
    return dense, time.perf_counter() - started


def run(root, task):
    out = task / 'run-v1'; out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); bind = Bindings(); setup()
    for name in ('MZ51_TRAINING_COVERAGE_20260911.md', 'mz51_training_coverage.py',
                 'mz50_echo_independent.py', 'mz50_source.py', 'mz50_train.py'):
        bind(Path(__file__).with_name(name))
    work = root / 'artifacts.local/work'
    source_task = work / 'mz48-rich-kilotier-20260911'
    source = load_source(source_task, bind(source_task / 'source-index.json'))
    bind(source_task / 'source-total-receipt.json')
    records, p = source['records'], source['predictor']
    groups = split_source(records)
    prior = work / 'mz50-echo-independent-local-20260911/run-v1'
    old_path, state_path = bound_outputs(bind, prior, ['predictions.npz', 'spatial.pt'])
    predictions = load_npz(old_path)
    np.testing.assert_array_equal(predictions['mz48/frame_ids'], [r['frame_id'] for r in records])
    np.testing.assert_array_equal(predictions['mz48/truth'], source['truth'])
    models = FrozenModels(root, bind)
    legacy_out = task / 'legacy-preparation-v1'; legacy_out.mkdir(exist_ok=False)
    legacy = prepare_legacy(root, legacy_out, bind, models)
    # Old local labels are not converted into echo-independent supervision.
    del legacy['query'], legacy['local_known']
    branches = load_npz(bound_outputs(bind, work / 'mz30-branch-responsibility-20260910/run-v1', ['branches.npz'])[0])
    train_ids = branches['train_ids']; assert len(train_ids) == 7562
    assert (legacy['lookup'][train_ids] >= 0).all()
    assert all(not np.isin(train_ids, legacy['cohorts'][c]).any() for c in COHORTS[:3])
    batches = schedule(source['truth'], groups['fit'], legacy['truth'], train_ids)
    np.savez_compressed(out / 'schedule.npz', train_ids=train_ids, fit_ids=groups['fit'], **batches)
    write(out / 'groups.json', dict(records=records, groups={k: v.tolist() for k, v in groups.items()}))
    dense, detail_seconds = detail_only(source, models, out)
    initial = torch.load(state_path, map_location='cpu', weights_only=True)
    probe = SpatialQuery(models.rank).cuda(); probe.load_state_dict(initial); probe.eval()
    with torch.inference_mode():
        parity = infer(probe, dense[:16], p['ranges'][:16], p['valid'][:16], models)
    parity_checks = {}
    for name in READOUTS:
        actual, expected = parity[name + '/raw'], predictions['mz48/IDEAL/' + name + '/raw'][:16]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-6)
        np.testing.assert_array_equal(parity[name + '/support'], predictions['mz48/IDEAL/' + name + '/support'][:16])
        parity_checks[name] = dict(values=int(actual.size), max_abs=float(np.abs(actual - expected).max()))
    del probe
    write(out / 'parity.json', dict(status='PASS', detail_only_replay=parity_checks, new_baseline_inferences=0))
    def forward(model, ids, condition, old=False):
        rr, vv = (legacy['ranges'], legacy['valid']) if old else (p['ranges'], p['valid'])
        detail = np.array(legacy['maps'][legacy['lookup'][ids]]) if old else dense[ids]
        observed = packet(rr[ids], vv[ids], condition)
        x = (torch.from_numpy(detail).cuda() - models.detail_mean) / models.detail_std
        return model.inspect(x, torch.from_numpy(observed['ranges']).cuda(), torch.from_numpy(observed['valid']).cuda())
    fits = {}
    write(out / 'start.json', dict(status='STARTED', inputs=bind.inputs, arms=ARMS, steps_per_arm=600,
        initial_checkpoint_sha256=sha(state_path), schedule_sha256=sha(out / 'schedule.npz'),
        backend='CUDA', device=torch.cuda.get_device_name(), old_train_ids=len(train_ids), new_fit_ids=len(groups['fit'])))
    for arm in ARMS:
        torch.manual_seed(151)
        model = SpatialQuery(models.rank).cuda(); model.load_state_dict(initial); model.train()
        assert all(torch.equal(value.cpu(), initial[key]) for key, value in model.state_dict().items())
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        torch.cuda.synchronize(); fit_start = time.perf_counter(); history = []
        for step, ids in enumerate(batches['shared']):
            condition = CONDITIONS[step % 3]
            native = forward(model, ids, condition)
            local, parts = loss_for(native, torch.from_numpy(source['cell_event_presence'][ids]).cuda(),
                torch.from_numpy(source['cell_known'][ids]).cuda(), torch.from_numpy(source['truth'][ids]).cuda(),
                torch.from_numpy(source['known'][ids]).cuda())
            extra = forward(model, batches[arm][step], condition, old=arm == 'OLD_NEG')
            negative = negative_loss(extra, torch.from_numpy(batches['query'][step]).cuda())
            loss = local + .25 * negative; assert torch.isfinite(loss)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if step % 100 == 0 or step == 599:
                row = dict(step=step + 1, condition=condition, loss=float(loss.detach()), native=float(local.detach()), negative=float(negative.detach()))
                history.append(row); print(arm, row, flush=True); write(out / 'progress.json', dict(stage='fit', arm=arm, **row))
        torch.cuda.synchronize(); model.eval()
        fits[arm] = dict(seconds=time.perf_counter() - fit_start, steps=600, history=history,
                         unique_extra_frames=int(len(np.unique(batches[arm]))), negative_presentations_per_query=[1200] * 4)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / (arm + '.pt'))
        with torch.inference_mode():
            for cohort in COHORTS:
                if cohort == 'mz48':
                    ids = np.arange(2560); rr, vv = p['ranges'], p['valid']; get_dense = lambda ii: dense[ii]
                elif cohort in ('rich', 'mz36'):
                    data = legacy[cohort]; ids = np.arange(len(data['truth']))
                    rr, vv = data['ranges'], data['valid']; get_dense = lambda ii: data['dense'][ii]
                else:
                    ids = legacy['cohorts'][cohort]; rr, vv = legacy['ranges'], legacy['valid']
                    get_dense = lambda ii: np.array(legacy['maps'][legacy['lookup'][ii]])
                np.testing.assert_array_equal(predictions[cohort + '/truth'], source['truth'] if cohort == 'mz48' else
                    (legacy[cohort]['truth'] if cohort in ('rich', 'mz36') else legacy['truth'][ids]))
                for condition in CONDITIONS:
                    parts = []
                    for begin in range(0, len(ids), 16):
                        ii = ids[begin:begin + 16]; observed = packet(rr[ii], vv[ii], condition)
                        parts.append(infer(model, get_dense(ii), observed['ranges'], observed['valid'], models))
                    prefix = cohort + '/' + condition + '/' + arm + '/'
                    for key in parts[0]: predictions[prefix + key] = np.concatenate([row[key] for row in parts])
                    if cohort == 'mz48':
                        flat = source['cell_event_presence'].reshape(2560, 64 * 49, 4)
                        for name in READOUTS:
                            predictions[prefix + name + '/winning_native'] = np.take_along_axis(flat, predictions[prefix + name + '/winner'][:, None], 1)[:, 0] & predictions[prefix + name + '/support']
                    print('INFER', arm, cohort, condition, flush=True)
        for name in READOUTS:
            def collect(key):
                return np.concatenate([predictions['DEV/DROP_CLOSE/' + key], predictions['mz48/DROP_CLOSE/' + key][groups['calibration']]])
            y = np.concatenate([predictions['DEV/truth'], source['truth'][groups['calibration']]])
            cut = cutoff_zero_added(collect(arm + '/' + name + '/raw'), collect(arm + '/' + name + '/support'), collect('MZ37'), y)
            np.save(out / (arm + '-' + name + '-cutoff.npy'), cut)
            fits[arm][name + '_cutoff'] = cut.tolist()
            for cohort in COHORTS:
                for condition in CONDITIONS:
                    prefix = cohort + '/' + condition + '/'; key = prefix + arm + '/' + name + '/'
                    margin = predictions[key + 'raw'].astype(float) - cut
                    accepted = (predictions[prefix + 'MZ37'] < 0) & predictions[key + 'support'] & (margin >= 0)
                    predictions[key + 'candidate'] = np.where(accepted, margin, predictions[prefix + 'MZ37'])
        del model, optimizer
    np.savez_compressed(out / 'predictions.npz', **predictions)
    bind.check()
    receipt = dict(status='PASS', inputs=bind.inputs, steps_per_arm=600, total_steps=1200, fits=fits,
        seconds=time.perf_counter() - started, detail_seconds=detail_seconds, backend='CUDA; CPU compact I/O',
        device=torch.cuda.get_device_name(), new_baseline_inferences=0, exact_initial_weights=True,
        dense_cache_files_created=0, dense_ram_bytes=int(dense.nbytes), old_mmap_bytes=int(legacy['maps'].nbytes),
        unknown_is_not_negative_local_supervision=True, inherited_single_fit_nondeterminism=True,
        outputs={f.name: sha(f) for f in out.iterdir() if f.is_file()})
    write(out / 'receipt.json', receipt)
    print('PASS', dict(seconds=receipt['seconds'], fits={k: v['seconds'] for k, v in fits.items()}), flush=True)


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
