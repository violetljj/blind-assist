"""Zero-training support-to-decision score audit on fixed cached predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from city_score_separation import curve


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scores(near, support):
    flat = support.astype(np.float64).reshape(len(support), 2, -1)
    assert flat.shape[-1] == 576 and np.isfinite(flat).all()
    assert ((flat >= 0) & (flat <= 1)).all()
    # Recover logit LSE from cached sigmoid probabilities. Clipping is disclosed.
    clipped = np.clip(flat, np.finfo(np.float32).tiny,
                      1 - np.finfo(np.float32).eps/2)
    odds_sum = (clipped/(1-clipped)).sum(axis=-1)
    # Monotone squash permits the existing probability-range ranking evaluator.
    lse_rank = odds_sum/(1+odds_sum)
    return dict(near=near.astype(float), support_max=flat.max(axis=-1),
                support_top6=np.partition(flat, -6, axis=-1)[..., -6:].mean(axis=-1),
                support_logit_lse=lse_rank), dict(
                    zero_pixels=int((flat == 0).sum()), one_pixels=int((flat == 1).sum()),
                    clipped_pixels=int((flat != clipped).sum()))


def main(a):
    start = time.perf_counter()
    repo = a.repo.resolve(); work = repo/'artifacts.local/work'
    oldcache = repo/'artifacts.local/nearfield/city-gate-intervention-20260908/inputs/collection500-training-cache-v1'
    rel = work/'city-relational-train-20260908/evidence/cache'
    dev = work/'city-dev-baseline-20260908/evidence/cache'
    cov = work/'city-coverage64-20260908'
    f = work/'city-full-fit-20260908/model-run-v1'
    c = cov/'model-run-v1'
    identities = []

    def load(path, cache, split, fresh=False):
        if not fresh:
            manifest = {r['path']:r for r in read(path.parent/'transfer-manifest.json')}
            assert sha(path) == manifest[path.name]['sha256']
        else:
            receipt = read(path.parent/'receipt.json')
            assert receipt['status'] == 'PASS' and receipt['training_steps'] == 0
            assert sha(path) == receipt['prediction_sha256']
            assert receipt['checkpoint_sha256'] == 'd60c871fd0a292bfd5b30c625e5a4189632a7c42576fd6d951c71e40e6d8b7bc'
            assert sha(cache/'manifest.json') == receipt['cache_manifest_sha256']
        prefix = 'supervision' if split == 'train' else 'evaluator'
        label = read(cache/prefix/f'{split}.json')
        target_path = cache/label['near']['path']
        assert sha(target_path) == label['near']['sha256']
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['sample_indices'], label['sample_indices'])
            near, support = z['near'].copy(), z['support'].copy()
        truth = np.load(target_path, allow_pickle=False)
        assert near.shape == truth.shape and np.isin(truth, [0, 1]).all()
        identities.append(dict(prediction=str(path), prediction_sha256=sha(path),
                               labels=str(target_path), labels_sha256=sha(target_path)))
        return near, support, truth

    def combined(parts):
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))

    models = {
        'F1134': {
            'TRAIN': combined([load(f/'old750-predictions.npz', oldcache, 'train'),
                               load(f/'new384-predictions.npz', rel, 'train')]),
            'DEV': load(f/'dev-predictions.npz', dev, 'dev'),
            'plaza': load(f/'plaza-predictions.npz', oldcache, 'test'),
            'coverage_EVAL': load(c/'A-eval-predictions.npz', cov/'evidence/cache-eval-v1', 'eval')},
        'Coverage1198': {
            'TRAIN': combined([load(c/'B-old750-predictions.npz', oldcache, 'train'),
                               load(c/'B-relational384-predictions.npz', rel, 'train'),
                               load(c/'B-added64-predictions.npz', cov/'evidence/cache-train-v1', 'train')]),
            'DEV': load(c/'B-dev-predictions.npz', dev, 'dev'),
            'plaza': load(a.plaza, oldcache, 'test', fresh=True),
            'coverage_EVAL': load(c/'B-eval-predictions.npz', cov/'evidence/cache-eval-v1', 'eval')}}
    result = dict(status='PASS', training_steps=0, k=6, inputs=identities, results={},
                  scope='Consumed descriptive ranking; no threshold or architecture selection',
                  lse='sigmoid(logsumexp(logit(cached support))); full576 pixels, no truth mask')
    for model, partitions in models.items():
        result['results'][model] = {}
        for part, (near, support, truth) in partitions.items():
            aggregates, clipping = scores(near, support)
            entry = dict(clipping=clipping, scores={})
            for name, values in aggregates.items():
                entry['scores'][name] = {}
                for j, head in enumerate(('BODY', 'HEAD')):
                    v = curve(values[:, j].tolist(), truth[:, j].astype(int).tolist())
                    # Independent pairwise tie-aware AUC verifies rank aggregation.
                    pos, neg = values[truth[:, j] == 1, j], values[truth[:, j] == 0, j]
                    auc = ((pos[:, None] > neg).mean()+.5*(pos[:, None] == neg).mean())
                    assert np.isclose(auc, v['roc_auc'])
                    entry['scores'][name][head] = {k:x for k,x in v.items() if k != 'curve'}
            result['results'][model][part] = entry
    result['seconds'] = time.perf_counter()-start
    result['source_sha256'] = sha(Path(__file__))
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['| Model / partition / head | near AUC/AP/R@10% | max | top6 | logit-LSE |',
             '| --- | --- | --- | --- | --- |']
    for model, parts in result['results'].items():
        for part, entry in parts.items():
            for head in ('BODY', 'HEAD'):
                cells=[]
                for name in ('near', 'support_max', 'support_top6', 'support_logit_lse'):
                    v=entry['scores'][name][head]
                    cells.append(f'{v["roc_auc"]:.3f}/{v["average_precision"]:.3f}/{v["descriptive_recall_envelope"]["0.1"]:.3f}')
                lines.append('| '+f'{model} / {part} / {head} | '+' | '.join(cells)+' |')
    (a.output/'table.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--plaza', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
