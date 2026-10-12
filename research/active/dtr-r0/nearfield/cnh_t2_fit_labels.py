"""CPU-only, evaluator-owned hard supervision and metadata for the T2 fit assay.

Never writes into a student input folder. Histories/voxels/geometry remain
observations; all labels below are diagnostic outcomes, not model features.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT/'artifacts.local/work/cnh-temporal-readout-20261004'
PILOT2 = ROOT/'artifacts.local/work/cnh-readout-pilot2-20261005'
OUT = ROOT/'artifacts.local/work/cnh-t2-fit-diagnostic-20261004'
CONTACTS = ('contact0-2cm', 'contact2-5cm', 'contact>5cm')
VALID_CATEGORIES = CONTACTS + ('pass0-10cm', 'clear')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def hard_supervision(category, front, domain, training, far_weight=2.):
    """Frozen contact/pass/clear rule, without target-only shortcuts.

    Both query categories are all-physical-surface truth. Positive is contact;
    pass includes exact outside10cm and is masked. Normalize training domains
    to N/2 each; single fresh evaluation domain to N, as in the original pilot.
    """
    category = np.asarray(category)
    front, domain = np.asarray(front), np.asarray(domain)
    n = len(category)
    if category.shape != (n, 2) or not np.isin(category, VALID_CATEGORIES).all():
        raise ValueError('Expected two all-object contact/pass/clear categories')
    labels = np.isin(category, CONTACTS).astype(np.float32)
    mask = (category != 'pass0-10cm').astype(np.float32)
    far = np.where((front >= 1.6) & (front < 2.6), far_weight, 1.).astype(np.float32)
    weights = mask*far[:, None]
    raw_sums, factors = {}, {}
    domains = np.unique(domain)
    if training and set(domains.tolist()) != {0, 1}:
        raise ValueError('Training must retain both natural and displacement domains')
    if not training and set(domains.tolist()) != {1}:
        raise ValueError('Fresh holdout contains only displacement domain1')
    for d in domains:
        selected = domain == d
        total = float(weights[selected].sum(dtype=np.float64))
        if not total > 0:
            raise ValueError('Empty effective supervision domain')
        factor = np.float32((n/2 if training else n)/total)
        weights[selected] *= factor
        raw_sums[str(int(d))], factors[str(int(d))] = total, float(factor)
    return labels, mask, weights, dict(raw_mass=raw_sums, domain_factors=factors)


def support_metadata(folder, n):
    """Retained SUB3 query-overlap mass over zone x bin, not voxel counts.

    Sum each [query,8,8,16] table cell, then valid history exposures. Padding
    index0 has zero mass. Stratification cuts are frozen by build() on train.
    """
    folder = Path(folder)
    index = np.load(folder/'index.npy', mmap_mode='r')
    table = np.load(folder/'table.npy', mmap_mode='r')
    if index.shape != (n, 8) or table.shape[1:] != (2, 8, 8, 16):
        raise ValueError('Frozen geometry cache axes differ from T2 query weights')
    masses = np.empty((len(table), 2), np.float64)
    for start in range(0, len(table), 256):
        chunk = table[start:start+256]
        masses[start:start+len(chunk)] = chunk.sum(axis=(2, 3, 4), dtype=np.float64)
    if not np.allclose(masses[0], 0):
        raise ValueError('Padding support must be zero')
    history = masses[index]
    total = history.sum(axis=1)
    length = (index != 0).sum(axis=1)
    if (length == 0).any():
        raise ValueError('Every retained row needs an observed exposure')
    mean = total/length[:, None]
    return dict(query_support_sum=total, query_support_mean=mean,
                query_support_current=history[:, -1])


def build(split, out=OUT, old=OLD, pilot2=PILOT2):
    """Require root-frozen diagnostic PLAN before scientific cache execution."""
    out, old, pilot2 = map(Path, (out, old, pilot2))
    plan_path = out/'PLAN.json'
    if not plan_path.exists():
        raise FileNotFoundError('Freeze the diagnostic PLAN before label/metadata materialization')
    read(plan_path)
    source_plan_path = pilot2/'PLAN.json'
    source_plan = read(source_plan_path)
    if source_plan['far_weight'] != 2 or source_plan['far_range_m'] != [1.6, 2.6]:
        raise ValueError('Original far weighting law changed')
    training = split == 'train'
    if split not in ('train', 'fresh_evaluation'):
        raise ValueError('Only frozen39936 train and new17472 holdout are supported')
    source = (old if training else pilot2)/'inputs'/split
    started = time.monotonic()
    with np.load(source/'rows.npz', allow_pickle=False) as stored:
        rows = {key: stored[key].copy() for key in stored.files}
    n = len(rows['unit'])
    if n != (39936 if training else 17472):
        raise ValueError('Frozen diagnostic row count changed')
    hashes = {str(path): sha(path) for path in (plan_path, source_plan_path, source/'rows.npz')}
    category = rows['category'].copy()
    context = np.full(n, 'NOT_AVAILABLE', dtype='<U16')
    visible = np.full(n, -1, np.int8)
    scene_fraction = np.full(n, np.nan, np.float64)
    scene_fov_in = np.full(n, -1, np.int8)
    delta_cm = np.full(n, np.nan, np.float64)
    fresh_units = np.unique(rows['unit'][rows['domain'] == 1])
    expected_units = source_plan['train_units'] if training else source_plan['eval_units']
    if fresh_units.tolist() != expected_units:
        raise ValueError('Displacement scene identities differ from frozen source PLAN')
    truth_root = old if training else pilot2
    for unit in fresh_units:
        selected = rows['unit'] == unit
        truth_path = truth_root/'truth'/('train' if training else 'evaluation')/f'unit{int(unit)}.json'
        truth = read(truth_path)
        frames, variants = rows['frame'][selected], rows['variant'][selected]
        expected_category = np.asarray(truth['categories'])[variants, frames]
        if not np.array_equal(expected_category, category[selected]):
            raise ValueError('Row categories differ from independently saved all-object truth')
        if not np.array_equal(rows['group'][selected], np.full(selected.sum(), truth['group'])):
            raise ValueError('Target height group identity changed')
        if not np.allclose(rows['front'][selected], np.asarray(truth['front_range_m'])[frames], atol=0, rtol=0):
            raise ValueError('Distance routing differs from evaluator truth')
        context[selected] = truth['context']
        delta_cm[selected] = np.asarray(truth['intrusion_cm'])[variants]
        hashes[str(truth_path)] = sha(truth_path)
        # Original training did not retain object_id templates. Its FOV is NA;
        # do not render new truth or relabel actualdelta as inside1/2 visibility.
        if not training:
            template_path = truth_root/'templates/evaluation'/f'unit{int(unit)}.npz'
            with np.load(template_path, allow_pickle=False) as templates:
                object_id = templates['object_id']
                flag = np.any(object_id.reshape(len(truth['boxes']), 16, -1) == 0, axis=-1)
            visible[selected] = flag[variants, frames].astype(np.int8)
            fraction = float(flag[:2, 3:16].mean())
            scene_fraction[selected], scene_fov_in[selected] = fraction, int(fraction >= .5)
            hashes[str(template_path)] = sha(template_path)
    labels, mask, weights, normalization = hard_supervision(category, rows['front'], rows['domain'], training)
    if training:
        for name, reconstructed in (('labels', labels), ('mask', mask), ('weights', weights)):
            path = source/f'{name}.npy'
            frozen = np.load(path, mmap_mode='r')
            if not np.array_equal(frozen, reconstructed):
                raise ValueError(f'Diagnostic reconstruction differs from frozen training {name}')
            hashes[str(path)] = sha(path)
    geometry = pilot2/'geometry'/split
    if not (geometry/'receipt.json').exists():
        geometry = out/'geometry'/split
    if not (geometry/'receipt.json').exists():
        raise FileNotFoundError('Query support needs the exact retained SUB3 geometry cache; no approximate fallback')
    support = support_metadata(geometry, n)
    if training:
        cuts = [float(np.median(support['query_support_sum'][:, q][support['query_support_sum'][:, q] > 1e-8])) for q in range(2)]
        if not np.isfinite(cuts).all():
            raise ValueError('Each query needs positive training geometric support')
    else:
        train_receipt = out/'supervision/train/receipt.json'
        prior = read(train_receipt)
        if prior['plan_sha256'] != hashes[str(plan_path)]:
            raise ValueError('Training geometric support cuts belong to another diagnostic PLAN')
        cuts = prior['support_positive_train_median']
        hashes[str(train_receipt)] = sha(train_receipt)
    support_bins = np.zeros((n, 2), np.int8)
    for q in range(2):
        positive = support['query_support_sum'][:, q] > 1e-8
        support_bins[positive, q] = 1 + (support['query_support_sum'][positive, q] > cuts[q]).astype(np.int8)
    support['query_support_bin'] = support_bins
    metadata = dict(rows)
    metadata.update(context=context, target_visible=visible, scene_visibility_fraction=scene_fraction,
                    scene_fov_in=scene_fov_in, intrusion_cm=delta_cm,
                    target_query=rows['group'].astype(np.int8),
                    distance_bin=np.digitize(rows['front'], [.6, .9, 1.2, 1.6, 2.1, 2.6]).astype(np.int8),
                    **support)
    for name in ('index.npy', 'table.npy', 'receipt.json'):
        path = geometry/name
        hashes[str(path)] = sha(path)
    destination = out/'supervision'/split
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'receipt.json').exists():
        raise FileExistsError('Diagnostic supervision is already sealed')
    for name, array in (('labels', labels), ('mask', mask), ('weights', weights)):
        np.save(destination/f'{name}.npy', array)
    np.savez_compressed(destination/'metadata.npz', **metadata)
    outputs = {str(path): sha(path) for path in destination.iterdir() if path.is_file()}
    receipt = dict(status='COMPLETE', split=split, rows=n,
        plan_sha256=hashes[str(plan_path)], source_sha256=sha(__file__), input_sha256=hashes,
        output_sha256=outputs, elapsed_s=time.monotonic()-started, **normalization,
        domains={str(int(d)): int((rows['domain'] == d).sum()) for d in np.unique(rows['domain'])},
        valid_queries=int(mask.sum()), positive_valid_queries=int((labels*mask).sum()),
        diagnostic_only='No supervision is written into student input folders or passed into forward()',
        loss='sum(BCEWithLogits(raw,hard)*mask*weights)/N; original training domain mass retained',
        comparison='Train overall mixes two equal loss-mass domains; compare train displacement-only vs fresh separately using sum(weighted BCE)/sum(weights)',
        support_positive_train_median=cuts,
        support='Sum validframe,zone,bin retained originalSUB3 query_weights over1024 zone-by-bin cells; bin0<=1e-8,bin1 positive<=trainingquerymedian,bin2>trainingquerymedian. Cuts read only geometry, frozen on39936 train and reused on fresh',
        fov='Current visibility=any object_id==0, evaluator-only/no photon or likelihood threshold; fresh sceneFOV>=.5 over inside1/2 and13 anchors; all train sceneFOV NOT_AVAILABLE because its actualdelta differs',
        training_visibility='NOT_AVAILABLE=-1 and NaN for all training; no retained exact object_id template cache',
        strata='domain; front edges .6/.9/1.2/1.6/2.1/2.6; target/other query; fresh HEAD/BODY x none/panel; support; current target visibility',
        limits='Fresh holdout has no natural domain; query AUC and pooled AUC describe different label mixtures. Reused synthetic Development, not deployment accuracy')
    (destination/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'fresh_evaluation', 'all'], default='all')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    for split in (('train', 'fresh_evaluation') if args.split == 'all' else (args.split,)):
        result = build(split, args.out)
        print('T2_DIAGNOSTIC_SUPERVISION', split, result['rows'], result['elapsed_s'], flush=True)
