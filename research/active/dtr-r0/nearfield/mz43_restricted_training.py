"""Matched ideal/mixed restricted-input fits on existing frozen feature bytes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from mz1_tiny_fusion import TinyFusion, state_sha, schedule, STEPS, BATCH_SIZE
from mz5_ensemble_readout import read, write, sha, load_npz
from mz40_packets import constrain
from mz40_evaluate import metrics, paired

CONDITIONS = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
HEADS = ('TOF_ONLY', 'FUSION')


def packet_features(ranges, valid):
    assert ranges.shape == valid.shape and ranges.shape[1:] == (64, 2)
    assert ranges.dtype == np.float32 and valid.dtype == np.bool_
    assert np.isfinite(ranges).all() and (ranges[~valid] == 0).all()
    return np.concatenate((ranges.reshape(-1, 128) / 4,
                           valid.reshape(-1, 128).astype(np.float32)), 1)


def conditions(tof):
    ranges = (tof[:, :128] * 4).reshape(-1, 64, 2)
    valid = tof[:, 128:].reshape(-1, 64, 2).astype(bool)
    np.testing.assert_array_equal(packet_features(ranges, valid), tof)
    out = [tof]
    for condition in CONDITIONS[1:]:
        changed, _ = constrain(ranges, valid, condition)
        out.append(packet_features(changed['ranges'], changed['valid']))
    return np.stack(out)


def infer(model, visual, tof):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(visual[i:i+256], tof[i:i+256]).cpu().numpy()
                               for i in range(0, len(visual), 256)])


def run(root, output):
    root = root.resolve(); work = root / 'artifacts.local/work'
    output = output.resolve()
    assert output.is_relative_to((root / 'artifacts.local').resolve()) and not output.exists()
    output.mkdir(parents=True)
    started = time.perf_counter(); inputs = {}; models = {}; fit_receipts = {}
    def bind(path, expected=None):
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        inputs[str(path)] = digest
        return path
    old = work / 'mz1-tiny-fusion-20260910'
    receipt = read(bind(old / 'run-v1/receipt.json'))
    feature_receipt = read(bind(old / 'cache-v1/features-receipt.json'))
    assert receipt['status'] == feature_receipt['status'] == 'PASS'
    data = load_npz(bind(old / 'cache-v1/features.npz', feature_receipt['feature_sha256']))
    assert receipt['feature_sha256'] == feature_receipt['feature_sha256']
    initial = torch.load(bind(old / 'run-v1/initial.pt', receipt['initial_file_sha256']),
                         map_location='cpu', weights_only=True)
    assert state_sha(initial) == receipt['initial_state_sha256']
    indices = np.load(bind(old / 'run-v1/schedule.npy', receipt['schedule_sha256']))
    np.testing.assert_array_equal(indices, schedule(data['role']))
    assert indices.shape == (300, 128) and (data['role'][indices] == 'TRAIN_ONLY').all()
    train = np.flatnonzero(data['role'] == 'TRAIN_ONLY')
    assert len(train) == 2500 and len(data['role']) == 5000
    for name in (*HEADS, 'RGB_ONLY'):
        bind(old / f'run-v1/{name}.pt', receipt['arms'][name]['checkpoint_sha256'])
    bind(Path(__file__).with_name('mz1_tiny_fusion.py'), receipt['source_sha256'])
    code = {name: sha(bind(Path(__file__).with_name(name))) for name in
            [Path(__file__).name, 'mz40_packets.py', 'mz40_evaluate.py',
             'MZ43_RESTRICTED_TRAINING_20260911.md']}
    all_tof = conditions(data['tof'])
    exposure = (np.arange(STEPS * BATCH_SIZE).reshape(STEPS, BATCH_SIZE) % 3).astype(np.int64)
    assert np.bincount(exposure.ravel()).tolist() == [12800] * 3
    np.save(output / 'schedule.npy', indices)
    np.save(output / 'condition-schedule.npy', exposure)
    assert torch.cuda.is_available()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    visual = torch.from_numpy(data['visual']).cuda()
    packet = torch.from_numpy(all_tof).cuda()
    target = torch.from_numpy(data['truth'][train].astype(np.float32)).cuda()
    lookup = np.full(len(data['role']), -1, np.int64); lookup[train] = np.arange(len(train))
    targets = torch.from_numpy(lookup[indices]).cuda()
    batches = torch.from_numpy(indices).cuda(); modes = torch.from_numpy(exposure).cuda()
    assert (lookup[indices] >= 0).all()
    sys.path.insert(0, str(root / 'tools'))
    from research_backend import torch_observation
    observed = asdict(torch_observation(output=(visual, packet, target)))
    assert observed['device_type'] == 'cuda'
    write(output / 'start.json', dict(status='STARTED', inputs=inputs.copy(), code_sha256=code,
        backend=observed, torch=str(torch.__version__), cuda=torch.version.cuda,
        initial_state_sha256=state_sha(initial), train_rows=2500, steps_per_fit=300,
        batch_size=128, fit_count=4, training_conditions=CONDITIONS,
        exposure_counts=[12800] * 3, normalization='Unchanged MZ1 ranges/4 and validity',
        predictor_fields=['frozen visual 772', 'normalized ranges 128', 'validity 128'],
        cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'], seed_search=False,
        eval_truth_used_for_training=False, backbone_inference_frames=0))
    try:
        # Finish both ideal reproductions before either intervention fit.
        for training in ('IDEAL', 'MIXED'):
            for head in HEADS:
                name = training + '/' + head
                model = TinyFusion(head); model.load_state_dict(initial)
                assert state_sha(model.state_dict()) == receipt['initial_state_sha256']
                model = model.cuda().train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
                losses = []; torch.cuda.synchronize(); tick = time.perf_counter()
                for step in range(STEPS):
                    batch = batches[step]
                    tof = packet[0, batch] if training == 'IDEAL' else packet[modes[step], batch]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(visual[batch], tof)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target[targets[step]])
                    assert torch.isfinite(loss)
                    loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
                torch.cuda.synchronize(); seconds = time.perf_counter() - tick
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                if training == 'IDEAL':
                    expected = torch.load(old / f'run-v1/{head}.pt', map_location='cpu', weights_only=True)
                    if any(not torch.equal(state[k], expected[k]) for k in state):
                        raise RuntimeError('INVALID_FOR_REQUESTED_COMPARISON: ideal final tensors differ: ' + head)
                    np.testing.assert_array_equal(losses, receipt['arms'][head]['train_loss_by_step'])
                filename = name.replace('/', '-')
                torch.save(state, output / (filename + '.pt'))
                np.save(output / (filename + '-loss.npy'), np.array(losses))
                fit_receipts[name] = dict(seconds=seconds, steps=300, final_loss=losses[-1],
                    state_sha256=state_sha(state), checkpoint_sha256=sha(output / (filename + '.pt')),
                    exact_historical_reproduction=training == 'IDEAL')
                models[name] = model.eval(); del optimizer
                print('FIT', name, fit_receipts[name], flush=True)
        # Evaluation starts only after all four fixed fits are complete.
        write(output / 'fit-complete.json', dict(status='PASS', fits=fit_receipts,
            input_sha256=inputs.copy(), evaluation_started=False))
        rgb_model = TinyFusion('RGB_ONLY').cuda()
        rgb_model.load_state_dict(torch.load(old / 'run-v1/RGB_ONLY.pt', map_location='cuda', weights_only=True))
        original_predictions = load_npz(bind(old / 'run-v1/predictions.npz', receipt['predictions_sha256']))
        rgb = infer(rgb_model, visual, packet[0])
        np.testing.assert_array_equal(rgb, original_predictions['RGB_ONLY_logits'])
        saved = dict(old_role=data['role'], old_truth=data['truth'],
                     old_known=np.ones_like(data['truth']), original_alerts=data['original_alerts'])
        results = dict(old={}, MZ36={}, fit=fit_receipts, parity={})
        def evaluate_family(prefix, vv, tt, rgb_logits, truth, known):
            scores = {'RGB': rgb_logits}
            for name, model in models.items(): scores[name] = infer(model, vv, tt)
            for training in ('IDEAL', 'MIXED'):
                scores[training + '/ENSEMBLE'] = .5 * (rgb_logits + scores[training + '/TOF_ONLY'])
            for name, score in scores.items(): saved[prefix + '/' + name] = score
            return scores
        for i, condition in enumerate(CONDITIONS):
            scores = evaluate_family('old/' + condition, visual, packet[i], rgb,
                                     data['truth'], np.ones_like(data['truth']))
            if condition == 'IDEAL':
                for head in HEADS:
                    np.testing.assert_array_equal(scores['IDEAL/' + head], original_predictions[head + '_logits'])
            results['old'][condition] = {role: {name: metrics(score[data['role'] == role],
                data['truth'][data['role'] == role], np.ones_like(data['truth'][data['role'] == role]))
                for name, score in scores.items()} for role in ('TRAIN_ONLY', 'DEV_ONLY', 'EVAL_ONLY')}
        source = work / 'mz40-l8cx-constrained-20260910'
        score_receipt = read(bind(source / 'score-v1/receipt.json'))
        scored = load_npz(bind(source / 'score-v1/scored.npz', score_receipt['outputs']['scored.npz']))
        ids, truth, known = (scored[k] for k in ('frame_ids', 'truth', 'known'))
        take = np.flatnonzero(known.all(1))
        assert len(ids) == 400 and len(take) == 380 and (~known).sum() == 80
        saved.update(mz36_frame_ids=ids, mz36_truth=truth, mz36_known=known)
        for condition in CONDITIONS:
            folder = work / 'mz36-new-source-20260910/inference-v1' if condition == 'IDEAL' else source / condition
            sr = read(bind(folder / 'receipt.json')); assert sr['status'] == 'PASS'
            raw = load_npz(bind(folder / 'predictions.npz', sr['outputs']['predictions.npz']))
            np.testing.assert_array_equal(raw['frame_ids'], ids[take])
            if condition == 'IDEAL': first_visual = raw['visual'].copy()
            else: np.testing.assert_array_equal(raw['visual'], first_visual)
            tv = torch.from_numpy(raw['visual']).cuda()
            tf = torch.from_numpy(packet_features(raw['ranges'], raw['valid'])).cuda()
            rr = infer(rgb_model, tv, tf)
            small = evaluate_family('admitted/' + condition, tv, tf, rr, truth[take], known[take])
            # Dense historical head and compact MZ40 head may round differently;
            # require exactly equal decisions and record raw logit error.
            parity = {}
            for name, prior in [('RGB', 'rgb'), ('IDEAL/TOF_ONLY', 'tof'), ('IDEAL/ENSEMBLE', 'MZ5')]:
                np.testing.assert_array_equal(small[name] >= 0, raw[prior] >= 0)
                parity[name] = float(np.max(np.abs(small[name] - raw[prior])))
            results['parity'][condition] = parity
            full = {name: np.zeros((400, 4), np.float32) for name in small}
            for name in full: full[name][take] = small[name]
            for name in ('MZ28', 'MZ37'): full[name] = scored[condition + '/' + name]
            for name, score in full.items(): saved['MZ36/' + condition + '/' + name] = score
            results['MZ36'][condition] = dict(metrics={name: metrics(score, truth, known) for name, score in full.items()},
                paired={head: paired(full['MIXED/' + head], full['IDEAL/' + head], truth, known)
                        for head in ('TOF_ONLY', 'FUSION', 'ENSEMBLE')})
        results['criteria'] = {}
        for head in ('ENSEMBLE', 'FUSION'):
            def totals(condition, training, key):
                return sum(results['MZ36'][condition]['metrics'][training + '/' + head][key])
            change = {key: sum(totals(c, 'MIXED', key) - totals(c, 'IDEAL', key) for c in CONDITIONS[1:]) for key in ('fp', 'fn')}
            ideal_mz36 = all(totals('IDEAL', 'MIXED', key) <= totals('IDEAL', 'IDEAL', key) for key in ('fp', 'fn'))
            ev = results['old']['IDEAL']['EVAL_ONLY']
            old_eval = all(sum(ev['MIXED/' + head][key]) <= sum(ev['IDEAL/' + head][key]) for key in ('fp', 'fn'))
            results['criteria'][head] = dict(restricted_pooled_change=change, ideal_mz36_no_regression=ideal_mz36,
                ideal_old_eval_no_regression=old_eval, useful_candidate=change['fn'] < 0 and change['fp'] <= 0 and ideal_mz36 and old_eval)
        np.testing.assert_array_equal(saved['original_alerts'], data['original_alerts'])
        np.savez_compressed(output / 'predictions.npz', **saved)
        results.update(status='PASS', scope='Consumed controlled Development; restricted-input proxies, no calibrated hardware result',
            original_alert_parity=5000, mz36_attempts=400, mz36_known_frames=380, mz36_unknown_bits=80)
        write(output / 'result.json', results)
        for path, digest in inputs.items(): assert sha(path) == digest, path
        write(output / 'receipt.json', dict(status='PASS', backend=observed, inputs=inputs, code_sha256=code,
            fits=fit_receipts, backbone_inference_frames=0, total_training_steps=1200,
            seconds=time.perf_counter()-started, outputs={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
        print('RESULT', results['criteria'], flush=True)
    except Exception:
        write(output / 'failure.json', dict(status='FAILED', error=traceback.format_exc(), completed_fits=fit_receipts))
        raise
    finally:
        models.clear()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.root, args.output)
