"""Frozen HEAD-P1 inference and signed, native-equal paired diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HEADS = ('BODY', 'HEAD')
CHECKPOINTS = {
    'Coverage1198': ('city-coverage64-20260908/model-run-v1/B-step2000.pt',
        'd60c871fd0a292bfd5b30c625e5a4189632a7c42576fd6d951c71e40e6d8b7bc'),
    'F1134': ('city-full-fit-20260908/model-run-v1/F-step2000.pt',
        'd911a3a717efb5a93fadb3268ba962db94455565805effc7077f75215670cf46')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def ranking(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return dict(auc=None, ap=None)
    auc = float(((pos[:, None] > neg).sum() + .5*(pos[:, None] == neg).sum())/(len(pos)*len(neg)))
    ap, previous = 0., 0
    for threshold in sorted(set(p.tolist()), reverse=True):
        chosen = p >= threshold
        tp = int(y[chosen].sum())
        ap += (tp-previous)/len(pos)*tp/int(chosen.sum())
        previous = tp
    return dict(auc=auc, ap=ap)


def map_rows(m, s):
    positive, negative = s == 1, s == 0
    pred = m >= .5
    tp = (pred & positive).sum(axis=(-1, -2))
    fp = (pred & negative).sum(axis=(-1, -2))
    fn = (~pred & positive).sum(axis=(-1, -2))
    count = positive.sum(axis=(-1, -2))
    peak = m.reshape(len(m), 2, -1).argmax(-1)
    target_at_peak = np.take_along_axis(s.reshape(len(m), 2, -1), peak[..., None], axis=-1)[..., 0]
    return dict(tp=tp, fp=fp, fn=fn, count=count, negative=negative.sum(axis=(-1, -2)),
        recall=np.divide(tp, count, out=np.full(tp.shape, np.nan), where=count > 0),
        iou=np.divide(tp, tp+fp+fn, out=np.full(tp.shape, np.nan), where=count > 0),
        peak_target=target_at_peak)


def assess(near, logits, maps, y, s, records, thresholds):
    n = len(records)
    assert near.shape == logits.shape == y.shape == (n, 2)
    assert maps.shape == s.shape == (n, 2, 18, 32)
    assert np.isfinite(near).all() and np.isfinite(logits).all() and np.isfinite(maps).all()
    assert np.isin(y, [0, 1]).all() and np.isin(s, [-1, 0, 1]).all()
    assert len({r['sample_index'] for r in records}) == n
    metric = map_rows(maps, s)
    pred = near >= np.array(thresholds)
    groups = {}
    for i, r in enumerate(records):
        key = (r['group_id'], r['relation'], r['appearance'])
        assert key not in groups
        groups[key] = i
    summaries = {}
    for role in ('train', 'eval'):
        summaries[role] = {}
        for appearance in ('A', 'L', 'M'):
            ids = np.array([i for i, r in enumerate(records) if r['source_partition'] == role and r['appearance'] == appearance])
            assert len(ids) > 0
            summaries[role][appearance] = {}
            for j, head in enumerate(HEADS):
                p, t = pred[ids, j], y[ids, j].astype(bool)
                positives = ids[metric['count'][ids, j] > 0]
                negmaps = ids[metric['count'][ids, j] == 0]
                pc = int(metric['count'][ids, j].sum())
                nc = int(metric['negative'][negmaps, j].sum())
                summaries[role][appearance][head] = dict(frames=len(ids), positives=int(t.sum()), negatives=int((~t).sum()),
                    TP=int((p & t).sum()), FP=int((p & ~t).sum()), FN=int((~p & t).sum()), TN=int((~p & ~t).sum()),
                    **ranking(near[ids, j], y[ids, j]),
                    positive_map_iou=float(metric['iou'][positives, j].mean()) if len(positives) else None,
                    positive_pixel_recall=float(metric['tp'][ids, j].sum()/pc) if pc else None,
                    positive_pixels=pc, TP_pixels=int(metric['tp'][ids, j].sum()), FP_pixels=int(metric['fp'][ids, j].sum()),
                    empty_map_activation=float(metric['fp'][negmaps, j].sum()/nc) if nc else None,
                    peak_hits=int((metric['peak_target'][positives, j] == 1).sum()),
                    peak_unknown=int((metric['peak_target'][positives, j] == -1).sum()))
    pairs = []
    for gid in dict.fromkeys(r['group_id'] for r in records):
        for appearance in ('L', 'M'):
            ia = [groups[(gid, relation, 'A')] for relation in ('CLEAR', 'HEAD_ONLY')]
            ib = [groups[(gid, relation, appearance)] for relation in ('CLEAR', 'HEAD_ONLY')]
            row0 = records[ia[0]]
            native_equal = all(records[a]['native_mask_sha256'] == records[b]['native_mask_sha256'] for a, b in zip(ia, ib))
            pooled_equal = all(np.array_equal(s[a], s[b]) for a, b in zip(ia, ib))
            truth_equal = all(np.array_equal(y[a], y[b]) for a, b in zip(ia, ib))
            desired = np.array([[0, 0], [0, 1]])
            labels_match = np.array_equal(y[ia], desired) and np.array_equal(y[ib], desired)
            changed = all(records[a]['native_rgb_sha256'] != records[b]['native_rgb_sha256'] for a, b in zip(ia, ib))
            valid = native_equal and pooled_equal and truth_equal and labels_match and changed
            da = float(logits[ia[1], 1]-logits[ia[0], 1])
            db = float(logits[ib[1], 1]-logits[ib[0], 1])
            ca, cb = pred[ia, 1] == y[ia, 1], pred[ib, 1] == y[ib, 1]
            support_delta = float(metric['recall'][ib[1], 1]-metric['recall'][ia[1], 1]) if labels_match else None
            damage = dict(correct_to_wrong=int((ca & ~cb).sum()), ordering_lost=bool(da > 0 and db <= 0),
                support_recall_drop20=bool(support_delta is not None and support_delta <= -.20))
            pairs.append(dict(group_id=gid, source_partition=row0['source_partition'], family=row0['family'], factor=appearance,
                sample_indices_A=[records[i]['sample_index'] for i in ia], sample_indices_B=[records[i]['sample_index'] for i in ib],
                strict_evaluable=bool(valid), native_equal=native_equal, pooled_equal=pooled_equal, desired_labels=bool(labels_match), rgb_changed=changed,
                signed_HEAD_logit_relation_A=da, signed_HEAD_logit_relation_B=db,
                signed_HEAD_probability_relation_A=float(near[ia[1], 1]-near[ia[0], 1]),
                signed_HEAD_probability_relation_B=float(near[ib[1], 1]-near[ib[0], 1]),
                appearance_HEAD_logit_delta=(logits[ib, 1]-logits[ia, 1]).astype(float).tolist(),
                appearance_HEAD_probability_delta=(near[ib, 1]-near[ia, 1]).astype(float).tolist(),
                four_HEAD_correct=bool(ca.all() and cb.all()), A_correct=int(ca.sum()), B_correct=int(cb.sum()),
                HEAD_alert_flips=int((pred[ia, 1] != pred[ib, 1]).sum()), HEAD_positive_support_recall_delta=support_delta,
                damage=damage, damaged=bool(valid and any(damage.values()))))
    factors = {}
    for factor in ('L', 'M'):
        rows = [p for p in pairs if p['factor'] == factor]
        good = [p for p in rows if p['strict_evaluable']]
        damaged = [p for p in good if p['damaged']]
        factors[factor] = dict(total_units=len(rows), evaluable_units=len(good), damaged_units=len(damaged),
            damaged_groups=[p['group_id'] for p in damaged], damaged_roles=sorted({p['source_partition'] for p in damaged}),
            damaged_families=sorted({p['family'] for p in damaged}), continuation_signal=len(damaged) >= 2,
            four_HEAD_correct=sum(p['four_HEAD_correct'] for p in good),
            positive_relation_A=sum(p['signed_HEAD_logit_relation_A'] > 0 for p in good),
            positive_relation_B=sum(p['signed_HEAD_logit_relation_B'] > 0 for p in good))
    return dict(partitions=summaries, pairs=pairs, factors=factors,
        appearance_failure_signal=any(f['continuation_signal'] for f in factors.values()))


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(args.source))
    import torch
    from city_data import CityRGBDataset
    from decoupled_model import DecoupledModel
    assert torch.cuda.is_available() and torch.__version__ == '2.9.1+cu128'
    work = args.artifact_root/'work'
    ref = read(work/'city-coverage64-20260908/model-run-v1/protocol.json')
    source_hashes = {}
    for name in ('city_data.py', 'decoupled_model.py', 'representation_model.py', 'whisker_model.py'):
        source_hashes[name] = sha(args.source/name)
        assert source_hashes[name] == ref['source_sha256'][name], name
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    data = CityRGBDataset(args.cache, 'diagnostic')
    assert len(data) == 96
    devcache = work/'city-dev-baseline-20260908/cache-dev-v1'
    dev = CityRGBDataset(devcache, 'dev')
    model = DecoupledModel(work/'city-finetune-pilot-20260908/prepared-v1/payload/pretrained').cuda().eval()
    outputs, identities = {}, {}
    with torch.inference_mode():
        for arm, (relative, digest) in CHECKPOINTS.items():
            checkpoint = work/relative
            assert sha(checkpoint) == digest
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            reference = checkpoint.parent/('B-dev-predictions.npz' if arm == 'Coverage1198' else 'dev-predictions.npz')
            cached = np.load(reference, allow_pickle=False)
            original = torch.stack([dev[i]['rgb'] for i in range(32)]).cuda()
            dn, dm = model(original)
            assert np.array_equal(dn.sigmoid().cpu().numpy(), cached['near'][:32]), arm+' near parity'
            assert np.array_equal(dm.sigmoid().cpu().numpy(), cached['support'][:32]), arm+' support parity'
            ns, ss, nl, sl = [], [], [], []
            for begin in range(0, len(data), 32):
                rgb = torch.stack([data[i]['rgb'] for i in range(begin, min(begin+32, len(data)))]).cuda()
                n, m = model(rgb)
                ns.append(n.sigmoid().cpu().numpy()); ss.append(m.sigmoid().cpu().numpy())
                nl.append(n.cpu().numpy()); sl.append(m.cpu().numpy())
            path = args.output/f'{arm}-predictions.npz'
            np.savez_compressed(path, near=np.concatenate(ns), support=np.concatenate(ss), near_logits=np.concatenate(nl),
                support_logits=np.concatenate(sl), sample_indices=np.array(data.ids))
            assert sha(checkpoint) == digest
            outputs[arm] = path
            selection = read(checkpoint.parent/'selection.json')
            identities[arm] = dict(checkpoint_sha256=digest, predictions_sha256=sha(path),
                DEV_thresholds=selection['thresholds'], DEV_selection_sha256=sha(checkpoint.parent/'selection.json'),
                original_DEV32_exact_parity=True)
    # Prediction files for BOTH checkpoints exist before diagnostic evaluator access.
    labels = read(args.cache/'evaluator/diagnostic.json')
    assert labels['sample_indices'] == data.ids
    def array(key):
        p = (args.cache/labels[key]['path']).resolve()
        assert p.is_relative_to(args.cache.resolve()) and sha(p) == labels[key]['sha256']
        return np.load(p, allow_pickle=False)
    y, s = array('near'), array('support')
    records = read(args.cache/'evaluator/groups.json')
    assert [r['sample_index'] for r in records] == data.ids
    result = dict(status='PASS', training_steps=0, models={}, identities=identities,
        cache_manifest_sha256=sha(args.cache/'manifest.json'), groups_sha256=sha(args.cache/'evaluator/groups.json'),
        labels_sha256=sha(args.cache/'evaluator/diagnostic.json'), source_sha256=source_hashes,
        protocol_sha256=sha(args.protocol), runner_sha256=sha(__file__),
        scope='Consumed same-world paired diagnostic; source roles retained; no fitted threshold or baseline promotion')
    for arm, path in outputs.items():
        z = np.load(path, allow_pickle=False)
        result['models'][arm] = assess(z['near'], z['near_logits'], z['support'], y, s, records, identities[arm]['DEV_thresholds'])
    result['conditional_training_signal'] = result['models']['Coverage1198']['appearance_failure_signal']
    result['seconds'] = time.perf_counter()-start
    write(args.output/'result.json', result)
    write(args.output/'receipt.json', dict(status='PASS', optimizer_steps=0, backend='CUDA', device=torch.cuda.get_device_name(),
        torch_version=torch.__version__, frames_per_model=len(data), models=2, batch=32, seconds=result['seconds'],
        result_sha256=sha(args.output/'result.json'), protocol_sha256=result['protocol_sha256'], source_sha256=source_hashes))
    print(json.dumps(dict(status='PASS', conditional_training_signal=result['conditional_training_signal'], seconds=result['seconds'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'source', 'artifact-root', 'output', 'protocol'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
