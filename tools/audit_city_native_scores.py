"""Inference-only TRAIN fit versus consumed native-route transfer score audit."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import adapt_city_native as adapt
import numpy as np
import torch

route = adapt.route
NATIVE = route.NF / 'city-native-adapt-20260908'
REPLAY = route.NF / 'city-native-replay-20260908'


def loaded(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def negative_rank(score, scores, labels):
    negative = np.asarray(scores)[np.asarray(labels) == 0]
    count = int((negative >= score).sum())
    return dict(known_negatives=len(negative), negatives_at_or_above=count,
        negative_fraction_at_or_above=count / len(negative) if len(negative) else None,
        max_negative=float(negative.max()) if len(negative) else None,
        unknown_labels=int((np.asarray(labels) == -1).sum()))


def target_masks(label_path, target_id, ids):
    manifest = route.read(label_path)
    target = next(t for t in manifest['targets'] if t['target_id'] == target_id)
    if target.get('status') != 'EVALUABLE' or target.get('frame_indices') != ids or not target.get('authority'):
        raise ValueError('Reliable declared target and exact ordering required')
    path = route.within(label_path.parent, target['masks'])
    return route.pooled(np.load(path, allow_pickle=False)), path


def validate_fit_cache(fit, train_label, dev_label, route_label):
    """Verify historical sources/label/lock identities and cached metric parity."""
    protocol = route.read(fit / 'protocol.json')
    result = route.read(fit / 'result.json')
    lock = route.read(fit / 'locked-dev-choice.json')
    receipt = route.read(fit / 'receipt.json')
    if route.sha(fit / 'result.json') != receipt['result_sha256']:
        raise ValueError('Completed result hash changed')
    for name in ('decoupled_model.py', 'representation_model.py'):
        if route.sha(route.SOURCE / name) != protocol['source_sha256'][name]:
            raise ValueError('Historical model source changed: ' + name)
    if route.sha(fit / 'locked-dev-choice.json') != result['dev_choice_sha256']:
        raise ValueError('Locked DEV threshold identity changed')
    for path, info in ((train_label, protocol['train_labels']), (dev_label, lock['labels']),
                       (route_label, result['route_labels'])):
        if route.sha(path) != info['manifest_sha256']:
            raise ValueError('Historical label manifest changed')
        meta = route.read(path)
        for key in ('near', 'support'):
            if route.sha(route.within(path.parent, meta[key])) != info['array_sha256'][key]:
                raise ValueError('Historical label array changed')
    checked = []
    for domain, suffix, label_path in (('dev', 'dev', dev_label), ('consumed_route40', 'consumed_route40', route_label)):
        meta = route.read(label_path)
        near, support, _ = route.labels(label_path, meta['sample_indices'])
        for arm in ('initial', 'adapted'):
            prediction = loaded(fit / f'{arm}-{suffix}.npz')
            for policy, thresholds in (('historical', protocol['historical_thresholds']), ('dev_selected', lock['choices'][arm])):
                actual = adapt.metrics(prediction['near'], prediction['support'], near, support, thresholds)
                expected = {k: v for k, v in result['results'][arm][domain][policy].items() if k != 'targets'}
                if actual != expected:
                    raise ValueError(f'Cached metric mismatch: {fit.name}/{arm}/{domain}/{policy}')
                checked.append(f'{arm}/{domain}/{policy}')
    return dict(status='PASS', fit=str(fit), historical_model_sources_match=True,
        label_manifest_and_array_hashes_match=True, locked_dev_hash_matches=True,
        exact_cached_metric_checks=checked,
        limitation='Original fit receipts did not store NPZ byte hashes; exact metric parity supplements the audit-time byte-hash snapshot')


def rows_for_target(ids, selected, predictions, masks, thresholds, ranks):
    rows = []
    for sid in selected:
        i = ids.index(sid)
        score = float(predictions['near'][i, 0])
        support = predictions['support'][i, 0]
        mask = masks[i, 0]
        positive = mask == 1
        visible = bool(positive.any())
        overlap = bool(((support >= .5) & positive).any()) if visible else None
        peak_label = int(mask.ravel()[int(support.argmax())])
        policies = {name: dict(threshold=float(value), margin=score-float(value), alert=score >= value,
            joint_alert_and_overlap=visible and score >= value and overlap) for name, value in thresholds.items()}
        rows.append(dict(sample_index=sid, head='BODY', near_probability=score,
            reliable_target_positive=visible, target_positive_cells=int(positive.sum()),
            support_overlap=overlap, support_max=float(support.max()), support_peak_label=peak_label,
            support_peak_hit=peak_label == 1, policies=policies,
            negative_ranks={name: negative_rank(score, probabilities[:, 0], labels[:, 0])
                for name, (probabilities, labels) in ranks.items()}))
    return rows


def run(args):
    started = time.perf_counter()
    out = args.output.resolve()
    if out.exists() or out == route.ARTIFACTS.resolve() or not out.is_relative_to(route.ARTIFACTS.resolve()):
        raise ValueError('Fresh canonical artifact output required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no inference fallback')
    native_fit, replay_fit = args.native_fit, args.replay_fit
    original = route.NF / 'decoupled-20260908/main/learned/decoupled_seed17.pt'
    fits = {'original': native_fit, 'native_only': native_fit, 'replay_thin': replay_fit}
    arms = {'original': 'initial', 'native_only': 'adapted', 'replay_thin': 'adapted'}
    checkpoints = {'original': original, 'native_only': native_fit / 'native_seed17_step300.pt',
        'replay_thin': replay_fit / 'replay_thin_seed17_step300.pt'}
    contract = dict(schema='city-native-score-audit-v1', optimizer_steps=0,
        intent='Distinguish failure to fit exposed TRAIN bollard positives from consumed route transfer failure',
        selection=dict(train_body_target=[6, 8, 11, 13], route_body_target=[6, 7]),
        no_selection='No fitting, threshold changes, capture, model choice or promotion',
        scope='Consumed Development diagnostic; TRAIN is in-sample, route is previously inspected regression',
        source_sha256={str(p): route.sha(p) for p in [Path(__file__), Path(adapt.__file__), Path(route.__file__),
            route.SOURCE / 'decoupled_model.py', route.SOURCE / 'representation_model.py']},
        input_sha256={})
    def record(path):
        contract['input_sha256'][str(path)] = route.sha(path)
        return route.read(path)
    train_ids, train_paths, train_info = adapt.capture(args.train_capture)
    train_label = adapt.label_file(args.train_labels)
    ty, ts, train_label_info = adapt.native_labels(train_label, train_ids, train_info)
    contract.update(train_capture=train_info, train_labels=train_label_info)
    label_paths = {'train': train_label, 'dev': adapt.label_file(args.dev_labels), 'route': adapt.label_file(args.route_labels)}
    labels, masks, ids = {'train': ty}, {'train': ts}, {'train': train_ids}
    for domain, path in label_paths.items():
        meta = record(path)
        ids[domain] = meta['sample_indices']
        for key in ('near', 'support'):
            payload = route.within(path.parent, meta[key])
            contract['input_sha256'][str(payload)] = route.sha(payload)
        if domain != 'train':
            labels[domain], masks[domain], _ = route.labels(path, ids[domain])
    if len(train_ids) != 45 or len(ids['dev']) != 45 or len(ids['route']) != 40:
        raise ValueError('Frozen TRAIN45/DEV45/route40 required')
    targets = {}
    for domain, target in (('train', 'train_bollard'), ('route', 'bollard')):
        targets[domain], path = target_masks(label_paths[domain], target, ids[domain])
        contract['input_sha256'][str(path)] = route.sha(path)
    cached, choices, historical = {}, {}, None
    for name, fit in fits.items():
        receipt = record(fit / 'receipt.json')
        protocol = record(fit / 'protocol.json')
        lock = record(fit / 'locked-dev-choice.json')
        if receipt['status'] != 'PASS':
            raise ValueError('Completed source fit required')
        if route.sha(fit / 'result.json') != receipt['result_sha256']:
            raise ValueError('Source result hash mismatch')
        contract['input_sha256'][str(fit / 'result.json')] = route.sha(fit / 'result.json')
        expected = protocol['initial_checkpoint_sha256'] if name == 'original' else receipt['fit']['checkpoint_sha256']
        if route.sha(checkpoints[name]) != expected:
            raise ValueError('Original/final checkpoint identity mismatch')
        contract['input_sha256'][str(checkpoints[name])] = expected
        historical = protocol['historical_thresholds'] if historical is None else historical
        if protocol['historical_thresholds'] != historical:
            raise ValueError('Historical thresholds disagree')
        choices[name] = lock['choices'][arms[name]]
        cached[name] = {}
        for domain, suffix in (('dev', 'dev'), ('route', 'consumed_route40')):
            path = fit / f'{arms[name]}-{suffix}.npz'
            contract['input_sha256'][str(path)] = route.sha(path)
            cached[name][domain] = loaded(path)
    # Cross-fit original cache agreement guards the shared comparator identity.
    for suffix, domain in (('dev', 'dev'), ('consumed_route40', 'route')):
        path = replay_fit / f'initial-{suffix}.npz'
        contract['input_sha256'][str(path)] = route.sha(path)
        comparison = loaded(path)
        if any(not np.array_equal(comparison[k], cached['original'][domain][k]) for k in ('near', 'support')):
            raise ValueError('Original prediction cache mismatch across fits')
    contract.update(checkpoints={name: str(p) for name, p in checkpoints.items()},
        thresholds=dict(historical=historical, dev_locked=choices), baseline_cache_parity=True)
    contract['historical_cache_validation'] = [validate_fit_cache(fit, train_label, label_paths['dev'], label_paths['route'])
        for fit in (native_fit, replay_fit)]
    # Freeze this small intent/input manifest BEFORE the first new TRAIN inference.
    out.mkdir(parents=True)
    route.write(out / 'input-manifest.json', contract)
    manifest_hash = route.sha(out / 'input-manifest.json')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    times, results = {}, {}
    try:
        x = adapt.rgb(train_paths)
        for name, checkpoint in checkpoints.items():
            tick = time.perf_counter()
            model = route.DecoupledModel(route.NF / 'representation-20260908/pretrained').cuda()
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            near, support = adapt.predict(model, x)
            torch.cuda.synchronize()
            times[name] = time.perf_counter() - tick
            cached[name]['train'] = dict(near=near, support=support)
            np.savez_compressed(out / f'{name}-train.npz', near=near, support=support)
            thresholds = dict(historical=historical[0]['value'], dev_selected=choices[name][0]['value'])
            ranks = {domain: (cached[name][domain]['near'], labels[domain]) for domain in labels}
            results[name] = dict(train_all_heads={policy: adapt.metrics(near, support, ty, ts, thresholds)
                for policy, thresholds in (('historical', historical), ('dev_selected', choices[name]))})
            results[name]['train_bollard'] = rows_for_target(train_ids, [6, 8, 11, 13], cached[name]['train'], targets['train'], thresholds, ranks)
            results[name]['route_bollard'] = rows_for_target(ids['route'], [6, 7], cached[name]['route'], targets['route'], thresholds, ranks)
            for domain in ('train', 'route'):
                rows = results[name][domain + '_bollard']
                results[name][domain + '_summary'] = dict(positive_opportunities=sum(r['reliable_target_positive'] for r in rows),
                    support_overlap_hits=sum(r['support_overlap'] is True for r in rows),
                    support_peak_hits=sum(r['support_peak_hit'] for r in rows),
                    policies={p:dict(alert_hits=sum(r['policies'][p]['alert'] for r in rows),
                        joint_hits=sum(r['policies'][p]['joint_alert_and_overlap'] for r in rows)) for p in thresholds})
            observation = asdict(adapt.torch_observation(model=model))
            del model
        if route.sha(out / 'input-manifest.json') != manifest_hash:
            raise ValueError('Frozen input manifest changed')
        route.write(out / 'result.json', dict(schema='city-native-score-audit-result-v1', methods=results,
            scope=contract['scope'], thresholds='Existing historical and DEV choices only; no tuning'))
        route.write(out / 'receipt.json', dict(status='PASS', optimizer_steps=0, new_inference_frames=45,
            checkpoints_inferred=3, cached_dev_frames=45, cached_route_frames=40, actual_backend=observation,
            train_inference_load_seconds=times, total_seconds=time.perf_counter()-started,
            input_manifest_sha256=manifest_hash, result_sha256=route.sha(out / 'result.json'),
            prediction_sha256={n: route.sha(out / f'{n}-train.npz') for n in checkpoints}))
    except BaseException as error:
        route.write(out / 'receipt.json', dict(status='FAIL', error=repr(error), optimizer_steps=0))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name, default in (('native-fit', NATIVE / 'fit-v1'), ('replay-fit', REPLAY / 'fit-v1'),
        ('train-capture', NATIVE / 'train-v1'), ('train-labels', NATIVE / 'train-labels-v2'),
        ('dev-labels', NATIVE / 'dev-labels-v2'), ('route-labels', NATIVE / 'route-labels-v2'),
        ('output', route.NF / 'city-native-score-audit-20260908/scores-v1')):
        p.add_argument('--' + name, type=Path, default=default)
    run(p.parse_args())
