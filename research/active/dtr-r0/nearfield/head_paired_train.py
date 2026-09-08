"""Conditional HEAD-P1 matched ERM/consistency; two final-only 2000-step fits.

Use --source for the ORIGINAL worker runtime, never the current checkout.
No optimizer dispatch is allowed without a completed positive HEAD-P1 receipt.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np

HEADS = ('BODY', 'HEAD')
RELATIONS = ('CLEAR', 'HEAD_ONLY')
APPEARANCES = ('A', 'L', 'M')
STEPS, SEED = 2000, 17
BASELINE_SHA = 'd60c871fd0a292bfd5b30c625e5a4189632a7c42576fd6d951c71e40e6d8b7bc'
RUNTIME_FILES = ('city_data.py', 'decoupled_model.py', 'representation_model.py',
    'whisker_model.py', 'city_full_fit.py', 'city_coverage64_fit.py',
    'city_dev_baseline.py', 'city_dev_selection.py', 'city_score_separation.py',
    'city_finetune_pilot.py', 'city_pilot_metrics.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def referenced(cache, record, allowed):
    cache = Path(cache).resolve(strict=True)
    path = (cache/record['path']).resolve(strict=True)
    boundary = cache/allowed if allowed else cache
    require(path.is_relative_to(boundary), 'Cache reference crosses '+str(allowed or 'root')+' boundary')
    require(sha(path) == record['sha256'], 'Cache reference hash mismatch: '+str(path))
    return path


def train_units(records, allowed_groups):
    """Return eight complete TRAIN units in source order, without EVAL access."""
    require(len(records) == 96, 'Exactly 96 diagnostic records required')
    require(len({r['sample_index'] for r in records}) == 96, 'Duplicate diagnostic IDs')
    require(set(r['source_partition'] for r in records) == {'train', 'eval'}, 'Invalid source partition')
    train = [r for r in records if r['source_partition'] == 'train']
    require(len(train) == 48, 'Exactly 48 TRAIN frames required')
    order = list(dict.fromkeys(r['group_id'] for r in train))
    require(order == list(allowed_groups)[:8] and len(order) == 8, 'TRAIN units differ from first eight source groups')
    require(not set(order) & {r['group_id'] for r in records if r['source_partition'] == 'eval'}, 'TRAIN/EVAL group overlap')
    bykey = {}
    for r in train:
        key = (r['group_id'], r['relation'], r['appearance'])
        require(key not in bykey, 'Duplicate TRAIN relation/appearance')
        require(r.get('source_role') == 'TRAIN_ONLY', 'TRAIN role mismatch')
        bykey[key] = r['sample_index']
    expected = {(g, r, a) for g in order for r in RELATIONS for a in APPEARANCES}
    require(set(bykey) == expected, 'Incomplete or unexpected TRAIN unit')
    return np.array([[bykey[g, r, a] for r in RELATIONS for a in APPEARANCES] for g in order], dtype=np.int64)


def validate_pairs(records, units, near, support, native=None):
    """Inputs are TRAIN-only arrays in units.flatten order, never all96 labels."""
    require(near.shape == (48, 2) and support.shape == (48, 2, 18, 32), 'TRAIN target shape mismatch')
    require(np.isin(near, [-1, 0, 1]).all() and np.isin(support, [-1, 0, 1]).all(), 'Invalid TRAIN target')
    byid = {r['sample_index']: r for r in records}
    if native is not None:
        require(native.shape == (48, 2, 360, 640), 'Native TRAIN mask shape mismatch')
    for u in range(8):
        for relation in range(2):
            start = 6*u+3*relation
            indices = [start, start+1, start+2]
            rows = [byid[int(units.reshape(-1)[i])] for i in indices]
            hashes = [r['native_mask_sha256'] for r in rows]
            require(all(len(h) == 64 for h in hashes) and len(set(hashes)) == 1, 'Native mask hashes unequal')
            require(len({json.dumps(r['camera'], sort_keys=True) for r in rows}) == 1, 'Camera changed within appearance pair')
            require(len({r['original_case_name'] for r in rows}) == 1, 'Appearance parent mismatch')
            for j in indices:
                require(np.array_equal(near[j], [0, relation]), 'Unexpected CLEAR/HEAD_ONLY labels')
                require(np.array_equal(support[j], support[start]), 'Pooled known/UNKNOWN mask changed')
                if native is not None:
                    buffer = io.BytesIO()
                    np.save(buffer, native[j], allow_pickle=False)
                    require(hashlib.sha256(buffer.getvalue()).hexdigest() == hashes[j-start], 'Native array does not match original mask hash')
                    require(np.array_equal(native[j], native[start]), 'Native values changed')
            require(rows[0]['native_rgb_sha256'] != rows[1]['native_rgb_sha256'] and
                rows[0]['native_rgb_sha256'] != rows[2]['native_rgb_sha256'], 'Appearance RGB unchanged')


def make_schedule(units, steps=STEPS):
    """Uniform1198 replacement plus two distinct uniformly selected TRAIN units."""
    require(units.shape == (8, 6) and len(np.unique(units)) == 48, 'Eight complete units required')
    rng = np.random.default_rng(SEED)
    original = rng.integers(0, 1198, size=(steps, 20), dtype=np.int64)
    selected = np.stack([rng.choice(8, 2, replace=False) for _ in range(steps)])
    paired_local = (selected[:, :, None]*6+np.arange(6)[None, None, :]).reshape(steps, 12)
    return dict(original_indices=original, unit_indices=selected, paired_local_indices=paired_local,
        paired_sample_indices=units.reshape(-1)[paired_local])


def consistency_losses(near_logits, support_logits, near_targets, support_targets):
    """Probability MSE: only A-L/M, same relation, common known pixels.

    Batch positions20:26 and26:32 are complete CLEAR A/L/M, HEAD A/L/M units.
    Pixel classes are averaged per pair/head, absent classes omitted; active
    pair/heads are then averaged. UNKNOWN has exactly zero gradient.
    """
    import torch
    require(near_logits.shape == near_targets.shape == (32, 2), 'Expected32x2 near tensors')
    require(support_logits.shape == support_targets.shape and support_logits.shape[:2] == (32, 2), 'Support batch mismatch')
    a = [20, 20, 23, 23, 26, 26, 29, 29]
    b = [21, 22, 24, 25, 27, 28, 30, 31]
    pn, ps = near_logits.sigmoid(), support_logits.sigmoid()
    ya, yb = near_targets[a], near_targets[b]
    ta, tb = support_targets[a], support_targets[b]
    nk, sk = (ya >= 0) & (yb >= 0), (ta >= 0) & (tb >= 0)
    # Single batch validation/synchronization, not one synchronization per pixel class.
    require(not bool(((ya != yb) & nk).any() | ((ta != tb) & sk).any()),
        'Consistency crosses near or support label relation')
    near = ((pn[a]-pn[b]).square()*nk).sum()/nk.sum().clamp_min(1)
    error = (ps[a]-ps[b]).square()
    masks = torch.stack((sk & (ta == 0), sk & (ta == 1)), dim=2)
    counts = masks.sum(dim=(-2, -1))
    sums = (error[:, :, None]*masks).sum(dim=(-2, -1))
    present = counts > 0
    per_pair_head = (sums/counts.clamp_min(1)).sum(-1)/present.sum(-1).clamp_min(1)
    active = present.any(-1)
    support = (per_pair_head*active).sum()/active.sum().clamp_min(1)
    return near, support


def supervised_loss(near_logits, support_logits, near, support, pixel_bce, all_near_known=False):
    import torch.nn.functional as F
    if all_near_known:
        nl = F.binary_cross_entropy_with_logits(near_logits, near)
    else:
        known = near >= 0
        per = F.binary_cross_entropy_with_logits(near_logits, near.clamp_min(0), reduction='none')
        nl = (per*known).sum()/known.sum().clamp_min(1)
    sl = pixel_bce(support_logits, support)
    return nl+.25*sl, nl, sl


def load_runtime(source, baseline_protocol, pilot):
    source = source.resolve(strict=True)
    hashes = {}
    for name in RUNTIME_FILES:
        digest = sha(source/name)
        require(digest == baseline_protocol['source_sha256'][name], 'Historical runtime mismatch: '+name)
        if name in ('city_data.py', 'decoupled_model.py', 'representation_model.py', 'city_pilot_metrics.py'):
            require(digest == pilot['source_sha256'][name], 'Original pilot runtime mismatch: '+name)
        require(name[:-3] not in sys.modules, 'Historical runtime already imported: '+name)
        hashes[name] = digest
    sys.path.insert(0, str(source))
    modules = {name[:-3]: importlib.import_module(name[:-3]) for name in RUNTIME_FILES}
    for name, module in modules.items():
        require(Path(module.__file__).resolve() == source/(name+'.py'), 'Imported wrong runtime: '+name)
    return modules, hashes


def predict_raw(model, dataset):
    import torch
    model.eval()
    ns, ms, pn, pm = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(dataset), 32):
            x = torch.stack([dataset[i]['rgb'] for i in range(start, min(start+32, len(dataset)))]).cuda()
            n, m = model(x)
            ns.append(n.cpu()); ms.append(m.cpu())
            pn.append(n.sigmoid().cpu().numpy()); pm.append(m.sigmoid().cpu().numpy())
    n, m = torch.cat(ns), torch.cat(ms)
    return dict(near=np.concatenate(pn), support=np.concatenate(pm),
        near_logits=n.numpy(), support_logits=m.numpy(), sample_indices=np.array(dataset.ids))


def save_checkpoint(torch, model, optimizer, output, step, identity):
    torch.save(model.state_dict(), output/'weights.pt')
    torch.save(dict(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
        python_rng=random.getstate(), completed_steps=step, identity=identity), output/'optimizer-rng.pt')


def run(a):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    started = time.perf_counter()
    base = read(a.baseline_run/'protocol.json')
    pilot = read(a.pilot/'protocol.json')
    br = read(a.baseline_run/'receipt.json')
    require(br['status'] == 'PASS' and sha(a.baseline_run/'result.json') == br['result_sha256'], 'Baseline receipt mismatch')
    require(sha(a.baseline_run/'B-step2000.pt') == BASELINE_SHA, 'Wrong Coverage1198 baseline')
    require(sha(a.baseline_run/'selection.json') == br['selection_sha256'], 'Baseline selection mismatch')
    require(sha(a.initial) == pilot['checkpoint_sha256'], 'Original G13 initialization mismatch')
    require(sha(a.initial) in base['input_sha256'].values(), 'Initialization differs from original Coverage1198 fit')
    require(sha(a.old_cache/'manifest.json') == pilot['cache_manifest_sha256'], 'Original750 source mismatch')
    caches = (a.old_cache, a.relational_cache, a.train64, a.eval64, a.dev_cache)
    for cache in caches:
        require(sha(cache/'manifest.json') in base['input_sha256'].values(), 'Historical cache identity mismatch: '+str(cache))
    diagnostic = read(a.diagnostic_run/'result.json')
    dr = read(a.diagnostic_run/'receipt.json')
    require(dr['status'] == diagnostic['status'] == 'PASS' and sha(a.diagnostic_run/'result.json') == dr['result_sha256'], 'Diagnostic receipt mismatch')
    require(diagnostic['conditional_training_signal'] is True and
        diagnostic['models']['Coverage1198']['appearance_failure_signal'] is True, 'No positive diagnostic; training branch closed')
    require(diagnostic['identities']['Coverage1198']['checkpoint_sha256'] == BASELINE_SHA, 'Wrong diagnostic primary model')
    require(diagnostic['identities']['Coverage1198']['original_DEV32_exact_parity'] is True,
        'Original DEV32 forward parity was not verified')
    require(diagnostic['cache_manifest_sha256'] == sha(a.paired_cache/'manifest.json'), 'Diagnostic cache mismatch')
    require(diagnostic['groups_sha256'] == sha(a.paired_cache/'evaluator/groups.json') and
        diagnostic['labels_sha256'] == sha(a.paired_cache/'evaluator/diagnostic.json'), 'Diagnostic record mismatch')
    modules, source_hashes = load_runtime(a.source, base, pilot)
    for name, digest in diagnostic['source_sha256'].items():
        require(source_hashes.get(name) == digest, 'Diagnostic runtime differs from training: '+name)
    import torch
    require(torch.cuda.is_available() and torch.__version__ == br['torch_version'] == '2.9.1+cu128', 'Original Torch2.9.1+cu128 CUDA runtime required')
    data_api = modules['city_data']
    score_api = modules['city_coverage64_fit']
    admission = score_api.coverage_admission(a.train64, a.eval64)
    records = read(a.paired_cache/'evaluator/groups.json')
    trainrec = read(a.train64/'supervision/train.json')
    units = train_units(records, list(dict.fromkeys(trainrec['group_ids'])))
    paired_meta = read(a.paired_cache/'manifest.json')
    require(paired_meta['schema'] == 'city-training-cache-v1' and paired_meta['status'] == 'PASS' and
        paired_meta['role'] == 'DIAGNOSTIC_ONLY' and paired_meta['admission']['status'] == 'PASS' and
        paired_meta['admission']['labels_match_intended'], 'Strict native appearance admission required')
    specpath = a.paired_cache/'source/spec.json'
    sourcespec = read(specpath)
    capture_spec = Path(paired_meta['source_capture'])/'source/spec.json'
    require(sha(capture_spec) == paired_meta['source_spec_sha256'] and read(capture_spec) == sourcespec,
        'Paired source spec mismatch')
    source64 = read(referenced(a.train64, read(a.train64/'manifest.json')['source_spec'], None))
    original_cases = {c['name']: c for c in source64['cases'] if c.get('source_role', source64.get('source_role')) == 'TRAIN_ONLY'}
    for r in records:
        if r['source_partition'] != 'train':
            continue
        parent = original_cases[r['original_case_name']]
        require(parent['group_id'] == r['group_id'] and parent['variant_id'] == r['relation'] and
            parent['camera'] == r['camera'], 'Paired TRAIN not bound to original64 geometry')
    require([c['name'] for c in sourcespec['cases']] == [r['sample_id'] for r in records], 'Paired source record order mismatch')
    paired_rgb = data_api.CityRGBDataset(a.paired_cache, 'diagnostic')
    require(paired_rgb.ids == [r['sample_index'] for r in records], 'Paired RGB order mismatch')
    labels = read(a.paired_cache/'evaluator/diagnostic.json')
    require(labels['sample_indices'] == paired_rgb.ids, 'Paired target order mismatch')
    positions = {sid: i for i, sid in enumerate(paired_rgb.ids)}
    train_positions = np.array([positions[int(sid)] for sid in units.reshape(-1)])
    # Hash entire immutable files, but only TRAIN slices become fit tensors.
    target_paths = {k: referenced(a.paired_cache, labels[k], 'evaluator') for k in ('near', 'support', 'native_support')}
    paired_y, paired_s, native = [np.array(np.load(target_paths[k], mmap_mode='r', allow_pickle=False)[train_positions], copy=True)
        for k in ('near', 'support', 'native_support')]
    validate_pairs(records, units, paired_y, paired_s, native)
    del native
    old, rel, added = [data_api.CitySupervisedDataset(c) for c in (a.old_cache, a.relational_cache, a.train64)]
    require((len(old), len(rel), len(added)) == (750, 384, 64), 'Expected1198 original TRAIN images')
    original = torch.utils.data.ConcatDataset([old, rel, added])
    paired = []
    for j, position in enumerate(train_positions):
        item = paired_rgb[int(position)]
        item.update(near=torch.from_numpy(paired_y[j]).float(), support=torch.from_numpy(paired_s[j]))
        paired.append(item)
    out, artifacts = a.output.resolve(), a.artifact_root.resolve(strict=True)
    require(not out.exists() and out.is_relative_to(artifacts) and out != artifacts, 'Fresh artifact output required')
    out.mkdir(parents=True)
    schedule = make_schedule(units)
    np.savez(out/'schedule.npz', **schedule)
    immutable_paths = [a.initial, a.pilot/'protocol.json', a.baseline_run/'protocol.json', a.baseline_run/'receipt.json',
        a.baseline_run/'selection.json', a.baseline_run/'B-step2000.pt', a.diagnostic_run/'result.json',
        a.diagnostic_run/'receipt.json', a.paired_cache/'manifest.json', a.paired_cache/'evaluator/groups.json',
        a.paired_cache/'evaluator/diagnostic.json', specpath, capture_spec, a.protocol, Path(__file__),
        Path(__file__).with_name('head_paired_evaluate.py')] + list(target_paths.values()) + [c/'manifest.json' for c in caches]
    immutable = {str(p.resolve()): sha(p) for p in immutable_paths}
    identity = dict(input_sha256=immutable, source_sha256=source_hashes, schedule_sha256=sha(out/'schedule.npz'))
    exposure = dict(original_draws=np.bincount(schedule['original_indices'].reshape(-1), minlength=1198).tolist(),
        paired_draws=np.bincount(schedule['paired_local_indices'].reshape(-1), minlength=48).tolist(),
        paired_TRAIN_ids=units.reshape(-1).tolist(), train_groups=[records[positions[int(u[0])]]['group_id'] for u in units],
        duplication='All16 A parent geometries were exposed in original1198; paired A rerenders add duplicate geometry exposure, not new independent sources.',
        sampling='20 uniform1198 replacement draws plus two distinct uniformly drawn TRAIN units, complete6frames each; same saved sequence both arms',
        diagnostic_EVAL_fit_draws=0)
    write(out/'protocol.json', dict(**identity, steps=STEPS, seed=SEED, batch=32, learning_rate=1e-5,
        weight_decay=1e-4, optimizer='AdamW', all_parameters_trainable=True, BN_running_buffers_frozen=True,
        supervised='near known BCE + .25 original global class-balanced known support BCE',
        consistency='weak arm only: .1*(near probability MSE + per-pair/head class-balanced support probability MSE); common known labels; A-L/M within relation only',
        arms=['ERM', 'CONSISTENCY'], exposure=exposure, admission=admission,
        selection='Final2000 only; original DEV128 per-head FPR<=.1; no other selection or rescue fitting',
        interruption='Save partial weights/optimizer/RNG on failure; no automatic optimizer resume or redispatch'))
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    model = None; optimizer = None; arm = None; step = 0
    fits = {}
    try:
        initial_digest = None
        for arm in ('ERM', 'CONSISTENCY'):
            step = 0
            armout = out/arm; armout.mkdir()
            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            model = modules['decoupled_model'].DecoupledModel(a.pretrained).cuda()
            model.load_state_dict(torch.load(a.initial, map_location='cpu', weights_only=True), strict=True)
            digest = modules['city_dev_baseline'].tensor_digest(model)
            require(initial_digest is None or initial_digest == digest, 'Arm initialization mismatch')
            initial_digest = digest
            for p in model.parameters(): p.requires_grad_(True)
            buffers = modules['city_dev_baseline'].tensor_digest(model, buffers_only=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
            model.train()
            for layer in model.modules():
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm): layer.eval()
            history = []; fit_start = time.perf_counter()
            for row in range(STEPS):
                rows = [original[int(i)] for i in schedule['original_indices'][row]] + [paired[int(i)] for i in schedule['paired_local_indices'][row]]
                x, y, s = [torch.stack([item[k] for item in rows]).cuda() for k in ('rgb', 'near', 'support')]
                n, m = model(x)
                supervised, nl, sl = supervised_loss(n, m, y, s, data_api.pixel_support_bce, all_near_known=True)
                cn, cs = consistency_losses(n, m, y, s) if arm == 'CONSISTENCY' else (n.sum()*0, m.sum()*0)
                loss = supervised + (.1*(cn+cs) if arm == 'CONSISTENCY' else 0)
                require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                require(bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()), 'Nonfinite gradient')
                optimizer.step(); step = row+1
                if step % 100 == 0:
                    history.append(dict(step=step, near_BCE=float(nl.detach()), support_BCE=float(sl.detach()),
                        near_consistency=float(cn.detach()), support_consistency=float(cs.detach()), loss=float(loss.detach())))
                    write(out/'progress.json', dict(arm=arm, total_steps=4000, completed_steps=(0 if arm == 'ERM' else 2000)+step, **history[-1]))
            torch.cuda.synchronize()
            require(modules['city_dev_baseline'].tensor_digest(model, buffers_only=True) == buffers, 'BN buffers changed')
            save_checkpoint(torch, model, optimizer, armout, step, identity)
            fits[arm] = dict(steps=step, seconds=time.perf_counter()-fit_start, initialization_digest=digest,
                checkpoint_sha256=sha(armout/'weights.pt'), optimizer_rng_sha256=sha(armout/'optimizer-rng.pt'), BN_unchanged=True, history=history)
            write(armout/'fit-complete.json', fits[arm])
            del optimizer, model, x, y, s, n, m, nl, sl, cn, cs, loss, supervised
            optimizer = model = None
        # Both fits finish before any final DEV selection or EVAL labels are read.
        result = dict(status='PASS', fits=fits, exposure=exposure, arms={})
        for arm in ('ERM', 'CONSISTENCY'):
            model = modules['decoupled_model'].DecoupledModel(a.pretrained).cuda()
            model.load_state_dict(torch.load(out/arm/'weights.pt', map_location='cpu', weights_only=True), strict=True)
            dev, dy, ds, dg = modules['city_full_fit'].evaluator(a.dev_cache, 'dev')
            require(len(dev) == 128, 'Expected original DEV128')
            pred = predict_raw(model, dev)
            heads = {h: modules['city_dev_selection'].select_threshold(pred['near'][:, j], dy[:, j], min_count=48) for j, h in enumerate(HEADS)}
            thresholds = [heads[h]['threshold'] for h in HEADS]
            selection = dict(heads=heads, thresholds=thresholds, checkpoint_sha256=fits[arm]['checkpoint_sha256'], dev_manifest_sha256=sha(a.dev_cache/'manifest.json'))
            write(out/arm/'selection.json', selection)
            np.savez_compressed(out/arm/'DEV-predictions.npz', **pred)
            metrics = dict(DEV=score_api.score(pred['near'], pred['support'], dy, ds, dg, thresholds), TRAIN={}, EVAL={})
            for tag, cache, split in [('old750', a.old_cache, 'train'), ('relational384', a.relational_cache, 'train'),
                    ('added64', a.train64, 'train'), ('originaleval64', a.eval64, 'eval')]:
                data, y, s, g = modules['city_full_fit'].evaluator(cache, split)
                pred = predict_raw(model, data)
                np.savez_compressed(out/arm/(tag+'-predictions.npz'), **pred)
                metrics['TRAIN' if split == 'train' else 'EVAL'][tag] = score_api.score(pred['near'], pred['support'], y, s, g, thresholds)
            pred = predict_raw(model, paired_rgb)
            np.savez_compressed(out/arm/'diagnostic-predictions.npz', **pred)
            # Diagnostic EVAL values are first materialized only here, after fit and DEV selection.
            dy = np.load(target_paths['near'], allow_pickle=False)
            ds = np.load(target_paths['support'], allow_pickle=False)
            from head_paired_evaluate import assess
            metrics['diagnostic'] = assess(pred['near'], pred['near_logits'], pred['support'], dy, ds, records, thresholds)
            result['arms'][arm] = dict(selection=selection, metrics=metrics)
            del model; model = None
        for p, digest in immutable.items(): require(sha(p) == digest, 'Input mutated: '+p)
        require(sha(out/'schedule.npz') == identity['schedule_sha256'], 'Schedule mutated')
        for name, digest in source_hashes.items(): require(sha(a.source/name) == digest, 'Runtime mutated: '+name)
        write(out/'result.json', result)
        write(out/'receipt.json', dict(status='PASS', new_fits=2, optimizer_steps=4000, batch=32,
            backend='CUDA', device=torch.cuda.get_device_name(), torch_version=torch.__version__,
            seconds=time.perf_counter()-started, result_sha256=sha(out/'result.json'), protocol_sha256=sha(out/'protocol.json'),
            scope='Consumed same-world Development; matched exposure; no additional fit or promotion'))
    except BaseException as exc:
        if model is not None and optimizer is not None:
            save_checkpoint(torch, model, optimizer, out/arm, step, identity)
        write(out/'failure.json', dict(status='FAIL', arm=arm, completed_arm_steps=step,
            completed_fits=list(fits), error=repr(exc), automatic_resume=False))
        raise
    finally:
        model = optimizer = None
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'pilot', 'baseline-run', 'diagnostic-run', 'initial', 'pretrained',
            'old-cache', 'relational-cache', 'train64', 'eval64', 'dev-cache', 'paired-cache',
            'artifact-root', 'output', 'protocol'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
