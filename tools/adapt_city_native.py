"""One fixed TRAIN-only G13 native City fit; lock DEV thresholds before regressions."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import evaluate_city_native_route as route
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from city_data import pixel_support_bce
from research_backend import torch_observation

CONFIG = dict(seed=17, steps=300, batch_size=16, optimizer='AdamW',
    head_lr=1e-4, backbone_lr=1e-6, weight_decay=1e-4,
    support_weight=.25, batchnorm='All running buffers frozen; affine trainable',
    selection='FINAL_STEP_300_ONLY', sampling='Uniform TRAIN replacement seed17',
    dev_rule='Per head FPR<=.10; maximum recall, then lower FP, then higher threshold',
    minimum_positive=8, minimum_negative=8, support_threshold=.5,
    scope='Single-seed native City Development; consumed route and Willow regression only')


def masked_near_bce(logits, target):
    known = target >= 0
    loss = F.binary_cross_entropy_with_logits(logits, target.clamp_min(0).float(), reduction='none')
    return (loss * known).sum() / known.sum().clamp_min(1)


def counts(target):
    return [dict(positive=int((target[:, h] == 1).sum()), negative=int((target[:, h] == 0).sum()),
                 unknown=int((target[:, h] == -1).sum())) for h in range(2)]


def select_thresholds(prob, target):
    prob = np.asarray(prob, dtype=np.float64)
    target = np.asarray(target)
    if prob.shape != target.shape or prob.ndim != 2 or prob.shape[1] != 2:
        raise ValueError('Expected paired Nx2 scores/targets')
    if not np.isfinite(prob).all() or not ((prob >= 0) & (prob <= 1)).all():
        raise ValueError('Invalid probabilities')
    if not np.isin(target, [-1, 0, 1]).all():
        raise ValueError('Invalid targets')
    choices = []
    for h, count in enumerate(counts(target)):
        row = dict(head=('BODY', 'HEAD')[h], counts=count)
        if min(count['positive'], count['negative']) < 8:
            choices.append(dict(**row, status='NOT_EVALUABLE', value=None))
            continue
        candidates = np.unique(np.r_[0., prob[:, h], np.nextafter(1., np.inf)])
        eligible = []
        for threshold in candidates:
            decision = prob[:, h] >= threshold
            tp = int((decision & (target[:, h] == 1)).sum())
            fp = int((decision & (target[:, h] == 0)).sum())
            if fp * 10 <= count['negative']:
                eligible.append((tp, -fp, float(threshold)))
        tp, minus_fp, value = max(eligible)
        choices.append(dict(**row, status='EVALUABLE', value=value,
            TP=tp, FP=-minus_fp, recall=tp/count['positive'], FPR=-minus_fp/count['negative']))
    return choices


def rgb(paths):
    images = []
    for path in paths:
        with Image.open(path) as im:
            if im.size != (640, 360):
                raise ValueError('Expected native 640x360 image')
            images.append(np.array(im.convert('RGB').resize((256, 144), Image.Resampling.BOX)))
    return torch.from_numpy(np.stack(images)).cuda().permute(0, 3, 1, 2).float().div_(255.)


def capture(path):
    path = path.resolve(strict=True)
    receipt = route.read(path / 'receipt.json')
    if receipt.get('status') != 'PASS' or receipt.get('source_unchanged') is not True:
        raise ValueError('Completed source-preserving native capture required')
    spec_path = path / 'source/spec.json'
    if route.sha(spec_path) != receipt['spec_sha256']:
        raise ValueError('Capture spec hash mismatch')
    frames = route.read(path / 'model/dataset.json')['frames']
    ids = [int(f['sample_index']) for f in frames]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Invalid frame identities')
    paths = [route.within(path / 'model', f['rgb_path']) for f in frames]
    evidence = dict(path=str(path), ids=ids, spec_sha256=route.sha(spec_path),
        receipt_sha256=route.sha(path / 'receipt.json'),
        dataset_sha256=route.sha(path / 'model/dataset.json'),
        rgb_sha256=[route.sha(p) for p in paths])
    return ids, paths, evidence


def label_file(path):
    return path / 'native-route-labels.json' if path.is_dir() else path


def native_labels(path, ids, evidence):
    path = label_file(path)
    near, support, info = route.labels(path, ids)
    provenance = info.get('provenance', {})
    if provenance.get('spec_sha256') != evidence['spec_sha256'] or provenance.get('receipt_sha256') != evidence['receipt_sha256']:
        raise ValueError('Native labels do not belong to supplied capture')
    return near, support, info


@torch.inference_mode()
def predict(model, values):
    model.eval()
    near, support = [], []
    for batch in values.split(16):
        n, s = model(batch)
        near.append(n.sigmoid().cpu().numpy())
        support.append(s.sigmoid().cpu().numpy())
    return np.concatenate(near), np.concatenate(support)


def metrics(prob, maps, near, support, thresholds):
    # Retain localization and UNKNOWN counts even if a head's DEV point is unavailable.
    values = [r['value'] if r['value'] is not None else .5 for r in thresholds]
    result = route.evaluate((prob >= values).astype(float), maps, near, support)
    result['thresholds']['near'] = [r['value'] for r in thresholds]
    for h, name in enumerate(('BODY', 'HEAD')):
        near_metrics = result['heads'][name]['near']
        denominator = near_metrics['FP'] + near_metrics['TN']
        near_metrics['FPR'] = near_metrics['FP'] / denominator if denominator else None
        if thresholds[h]['value'] is None:
            result['heads'][name]['near'] = dict(status='NOT_EVALUABLE', reason='Insufficient DEV classes',
                counts=counts(near)[h])
            result['near_balanced_error'] = None
    return result


def bn_buffers(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers()
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


def retention(result, willow_losses):
    checks = {}
    for policy in ('historical', 'dev_selected'):
        initial, adapted = result['initial'], result['adapted']
        dev_heads = adapted['dev']['dev_selected']['heads']
        dev_ok = all(h['near'].get('recall') is not None and h['near']['recall'] >= .5
            and h['near']['FP'] / max(1, h['near']['FP'] + h['near']['TN']) <= .1
            and (h['support']['positive_frame_mean_iou'] or 0) > 0 for h in dev_heads.values())
        before, after = initial['consumed_route40'][policy], adapted['consumed_route40'][policy]
        fp_ok = all(after['heads'][h]['near'].get('FP') is not None
            and before['heads'][h]['near'].get('FP') is not None
            and after['heads'][h]['near']['FP'] <= before['heads'][h]['near']['FP'] + 2
            for h in ('BODY', 'HEAD'))
        target_ok = None
        if isinstance(before.get('targets'), list) and isinstance(after.get('targets'), list):
            def misses(rows):
                return {(r['target_id'], h): m['alert_FN'] for r in rows
                    for h, m in r.get('methods', {}).get('g13', {}).items()
                    if m['reliable_positive_frames'] > 0}
            b, a = misses(before['targets']), misses(after['targets'])
            target_ok = bool(b) and b.keys() == a.keys() and all(a[k] <= b[k] for k in b)
        lost = willow_losses[policy]
        checks[policy] = dict(dev_recall_fpr_and_localization=dev_ok, route_fp_within_plus2=fp_ok,
            route_target_misses_not_increased=target_ok, willow_lost_correct_head_decisions=lost,
            willow_within2=lost is not None and lost <= 2,
            retain_development_challenger=dev_ok and fp_ok and target_ok is True and lost is not None and lost <= 2)
    return dict(primary_policy='dev_selected', policies=checks,
        disposition='DEVELOPMENT_CHALLENGER' if checks['dev_selected']['retain_development_challenger'] else 'RETAIN_ORIGINAL')


def willow_data(path, original_receipt):
    data = route.read(path / 'model/dataset.json')
    samples = [s for s in data['samples'] if s['split'] == 'val'][:16]
    if len(samples) != 16:
        raise ValueError('Expected first 16 original Willow VAL')
    frames = {f['sample_index']: f for f in data['frames']}
    paths = [route.within(path / 'model', frames[s['frame_indices'][0]]['rgb_path']) for s in samples]
    receipt = original_receipt['caches']['training']
    input_paths = ('model/dataset.json', 'training/labels.json', 'training/support.npz')
    for key in input_paths:
        if route.sha(path / key) != receipt['input_sha256'][key]:
            raise ValueError('Original Willow regression input changed: ' + key)
    for s, p in zip(samples, paths):
        if route.sha(p) != receipt['rgb_sha256'][str(s['frame_indices'][0])]:
            raise ValueError('Original Willow RGB changed')
    labels = route.read(path / 'training/labels.json')['targets']
    near = np.array([labels[s['sample_id']][:2] for s in samples])
    with np.load(path / 'training/support.npz', allow_pickle=False) as archive:
        support = np.stack([archive[s['sample_id']] for s in samples])
    return paths, near, support, dict(sample_ids=[s['sample_id'] for s in samples],
        input_sha256={key: route.sha(path / key) for key in input_paths})


def run(args):
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fit fallback')
    out = args.output.resolve()
    if out.exists() or out == route.ARTIFACTS.resolve() or not out.is_relative_to(route.ARTIFACTS.resolve()):
        raise ValueError('Fresh canonical artifact output required')
    sources = [args.train_capture.resolve(), args.dev_capture.resolve(), args.route_capture.resolve()]
    if len(set(sources)) != 3:
        raise ValueError('TRAIN, DEV, consumed-route sources must be distinct')
    learned = route.NF / 'decoupled-20260908/main/learned'
    original_receipt = route.read(learned / 'receipt.json')
    checkpoint = learned / 'decoupled_seed17.pt'
    expected = next(r['checkpoint_sha256'] for r in original_receipt['records'] if r['arm'] == 'decoupled' and r['seed'] == 17)
    if route.sha(checkpoint) != expected:
        raise ValueError('Original G13 seed17 checkpoint mismatch')
    historical_path = route.NF / 'decoupled-20260908/main/evaluation/result.json'
    historical = route.read(historical_path)['primary']['evaluations']['decoupled/ensemble/normal']['all_TEST']['thresholds'][:2]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    admission = route.read(args.source_admission)
    if admission.get('status') != 'PASS':
        raise ValueError('Source admission must PASS before fitting')
    out.mkdir(parents=True)
    steps = 0
    try:
        ids, paths, train_info = capture(args.train_capture)
        dids, dpaths, dev_info = capture(args.dev_capture)
        if len(ids) + len(dids) > 96:
            raise ValueError('Frozen maximum 96 TRAIN+DEV views exceeded')
        if set(train_info['rgb_sha256']) & set(dev_info['rgb_sha256']):
            raise ValueError('TRAIN/DEV duplicate RGB')
        y, support, label_info = native_labels(args.train_labels, ids, train_info)
        train_counts = counts(y)
        contract = dict(config=CONFIG, initial_checkpoint_sha256=expected, train=train_info, dev_rgb=dev_info,
            source_admission_sha256=route.sha(args.source_admission),
            frozen_protocol_sha256=route.sha(route.SOURCE / 'CITY_NATIVE_ADAPT_PROTOCOL_20260908.md'),
            train_labels=label_info, train_counts=train_counts,
            historical_thresholds=historical, historical_threshold_source_sha256=route.sha(historical_path),
            source_sha256={p.name: route.sha(p) for p in [Path(__file__), Path(route.__file__),
                route.SOURCE / 'city_data.py', route.SOURCE / 'city_pilot_metrics.py',
                route.SOURCE / 'decoupled_model.py', route.SOURCE / 'representation_model.py']})
        route.write(out / 'protocol.json', contract)
        if any(min(c['positive'], c['negative']) < 8 for c in train_counts):
            route.write(out / 'receipt.json', dict(status='NOT_EVALUABLE', reason='TRAIN requires >=8 positive and >=8 negative per head', optimizer_steps=0, train_counts=train_counts))
            return
        x, ty, ts = rgb(paths), torch.from_numpy(y).cuda(), torch.from_numpy(support).cuda()
        model = route.DecoupledModel(args.pretrained).cuda()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        before = bn_buffers(model)
        groups = [dict(params=[p for n, p in model.named_parameters() if n.startswith('backbone.')], lr=1e-6),
                  dict(params=[p for n, p in model.named_parameters() if not n.startswith('backbone.')], lr=1e-4)]
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        schedule = np.random.default_rng(17).integers(0, len(ids), size=(300, 16))
        np.save(out / 'training_indices.npy', schedule, allow_pickle=False)
        model.train()
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layer.eval()
        tick = time.perf_counter()
        history = []
        for indices in schedule:
            index = torch.from_numpy(indices).cuda()
            n, s = model(x[index])
            loss = masked_near_bce(n, ty[index]) + .25 * pixel_support_bce(s, ts[index])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; no rescue')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                raise RuntimeError('Nonfinite gradient; no rescue')
            optimizer.step()
            steps += 1
            if steps % 25 == 0:
                history.append(dict(step=steps, loss=float(loss.detach()), elapsed_s=time.perf_counter()-tick))
                route.write(out / 'progress.json', history[-1])
                print(history[-1], flush=True)
        torch.cuda.synchronize()
        after = bn_buffers(model)
        unchanged = before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
        if not unchanged:
            raise RuntimeError('BN running buffers changed')
        final = out / 'native_seed17_step300.pt'
        torch.save(model.state_dict(), final)
        fit = dict(steps=steps, seconds=time.perf_counter()-tick, history=history,
            bn_buffers_unchanged=unchanged, bn_buffer_count=len(before), checkpoint_sha256=route.sha(final),
            schedule_sha256=route.sha(out / 'training_indices.npy'))
        route.write(out / 'fit-complete.json', fit)
        del optimizer, x, ty, ts, n, s, loss
        # DEV is first opened after the final checkpoint. No further updates.
        dy, ds, dlabels = native_labels(args.dev_labels, dids, dev_info)
        dev_tick = time.perf_counter()
        dx = rgb(dpaths)
        choices, result = {}, {}
        for arm, weight in (('initial', checkpoint), ('adapted', final)):
            model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)
            prob, maps = predict(model, dx)
            np.savez_compressed(out / f'{arm}-dev.npz', near=prob, support=maps)
            choices[arm] = select_thresholds(prob, dy)
            result[arm] = dict(dev=dict(historical=metrics(prob, maps, dy, ds, historical),
                dev_selected=metrics(prob, maps, dy, ds, choices[arm])))
        lock = out / 'locked-dev-choice.json'
        route.write(lock, dict(rule=CONFIG['dev_rule'], choices=choices, dev=dev_info, labels=dlabels,
            checkpoint_sha256=fit['checkpoint_sha256'], optimizer_steps=steps,
            regression_labels_opened=False))
        locked_hash = route.sha(lock)
        dev_seconds = time.perf_counter() - dev_tick
        del dx
        # Only now open consumed route labels and original Willow VAL labels.
        rids, rpaths, route_info = capture(args.route_capture)
        if len(rids) != 40:
            raise ValueError('Expected consumed 40-frame route')
        if (set(route_info['rgb_sha256']) & (set(train_info['rgb_sha256']) | set(dev_info['rgb_sha256']))):
            raise ValueError('Regression RGB overlaps TRAIN/DEV')
        ry, rs, rlabels = native_labels(args.route_labels, rids, route_info)
        wpaths, wy, ws, winfo = willow_data(args.willow_capture, original_receipt)
        times, willow_losses = {'dev': dev_seconds}, {}
        for domain, paths, near, truth in (('consumed_route40', rpaths, ry, rs), ('willow_val16', wpaths, wy, ws)):
            tick = time.perf_counter()
            values = rgb(paths)
            paired = {}
            for arm, weight in (('initial', checkpoint), ('adapted', final)):
                model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)
                prob, maps = predict(model, values)
                paired[arm] = prob
                np.savez_compressed(out / f'{arm}-{domain}.npz', near=prob, support=maps)
                result[arm][domain] = dict(historical=metrics(prob, maps, near, truth, historical),
                    dev_selected=metrics(prob, maps, near, truth, choices[arm]))
                if domain == 'consumed_route40':
                    # Existing adapter iterates its method registry; constrain it locally.
                    old = route.METHODS
                    try:
                        route.METHODS = {'g13': old['g13']}
                        for key, thresholds in (('historical', historical), ('dev_selected', choices[arm])):
                            if all(t['value'] is not None for t in thresholds):
                                result[arm][domain][key]['targets'] = route.target_metrics(label_file(args.route_labels), rids,
                                    dict(g13_near=prob.astype(np.float64), g13_support=maps), {'g13': dict(near_thresholds=thresholds)})
                            else:
                                result[arm][domain][key]['targets'] = dict(status='NOT_EVALUABLE', reason='Missing DEV operating point')
                    finally:
                        route.METHODS = old
            if domain == 'willow_val16':
                for policy in ('historical', 'dev_selected'):
                    selected = {arm: historical if policy == 'historical' else choices[arm] for arm in paired}
                    if any(t['value'] is None for rows in selected.values() for t in rows):
                        willow_losses[policy] = None
                    else:
                        correct = {arm: (prob >= [t['value'] for t in selected[arm]]) == near for arm, prob in paired.items()}
                        willow_losses[policy] = int((correct['initial'] & ~correct['adapted'] & (near >= 0)).sum())
            torch.cuda.synchronize()
            times[domain] = time.perf_counter()-tick
            del values
        if route.sha(lock) != locked_hash or route.sha(final) != fit['checkpoint_sha256'] or route.sha(checkpoint) != expected:
            raise RuntimeError('Frozen artifact changed during evaluation')
        route.write(out / 'result.json', dict(results=result, route=route_info, route_labels=rlabels,
            willow=winfo, scope=CONFIG['scope'], dev_choice_sha256=locked_hash,
            retention=retention(result, willow_losses)))
        route.write(out / 'receipt.json', dict(status='PASS', actual_backend=asdict(torch_observation(model=model)),
            device=torch.cuda.get_device_name(), fit=fit, evaluation_seconds=times,
            total_seconds=time.perf_counter()-started, result_sha256=route.sha(out / 'result.json')))
    except BaseException as error:
        route.write(out / 'receipt.json', dict(status='FAIL', error=repr(error), optimizer_steps=steps,
            total_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('train-capture', 'train-labels', 'dev-capture', 'dev-labels', 'route-capture', 'route-labels', 'source-admission', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, default=route.NF / 'representation-20260908/pretrained')
    parser.add_argument('--willow-capture', type=Path, default=route.NF / 'diversity-20260907/main/capture')
    run(parser.parse_args())
