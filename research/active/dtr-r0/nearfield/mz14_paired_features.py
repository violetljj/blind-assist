"""Frozen paired local-feature descriptors; no learned probe or new inference."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from mz5_ensemble_readout import read, write, sha, load_npz
from mz8_attribution import AngularReturnReadout


def auc(positive, negative):
    """Empirical pairwise AUC with half credit for ties, not independent trials."""
    assert len(positive) and len(negative)
    return float(((positive[:, None] > negative[None, :]).sum()
                  + .5*(positive[:, None] == negative[None, :]).sum())
                 / (len(positive)*len(negative)))


def angular_image(values):
    return values.reshape(25, 8, 8, 7, 7).transpose(0, 1, 3, 2, 4).reshape(25, 56, 56)


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    work = root/'artifacts.local/work'
    cache = work/'mz8-attribution-20260910/cache-v5'
    labels = work/'mz9-source-supervision-20260910/labels-v1'
    run9 = work/'mz9-source-supervision-20260910/run-v1'
    specpath = work/'mz6-short-sequence-20260910/capture-v3/evaluator/spec.json'
    bindings = [(cache/'dense.npy', read(cache/'receipt.json')['files']['dense.npy']),
                (labels/'evaluator.npz', read(labels/'receipt.json')['labels_sha256']),
                (run9/'normalization.npz', read(run9/'receipt.json')['outputs']['normalization.npz'])]
    inputs = {}
    for path, expected in bindings:
        inputs[str(path)] = sha(path)
        assert inputs[str(path)] == expected
    for path in [specpath, cache/'receipt.json', labels/'receipt.json', run9/'receipt.json']:
        inputs[str(path)] = sha(path)
    cases = read(specpath)['cases']
    camera_parity = [cases[100+i]['camera'] == cases[125+i]['camera'] for i in range(25)]
    assert all(camera_parity)
    maps = np.load(cache/'dense.npy', mmap_mode='r')
    normalization = load_npz(run9/'normalization.npz')
    labels_npz = load_npz(labels/'evaluator.npz')
    torch.set_num_threads(1)
    with torch.no_grad():
        dense = (np.array(maps[3600:3650])-normalization['mean'])/normalization['std']
        tokens = AngularReturnReadout().sample_visual(torch.from_numpy(dense)).numpy()
    # The module's learned layers are never called; only its fixed sampling grid
    # samples existing feature maps. Native labels define diagnostic groups only.
    q = labels_npz['query_counts'][3600:3650]
    known = labels_npz['cell_known_counts'][3600:3650] > 0
    foreground = (q[:25, ..., 1]+q[:25, ..., 3]).sum(2) > 0
    empty_foreground = (q[25:, ..., 1]+q[25:, ..., 3]).sum(2) > 0
    foreground &= ~empty_foreground
    zones = foreground.any(2)
    background = known[:25] & known[25:] & zones[:, :, None] & ~foreground & ~empty_foreground
    response = np.linalg.norm(tokens[:25]-tokens[25:], axis=-1)
    full_foreground, full_background = angular_image(foreground), angular_image(background)
    full_response = angular_image(response)
    adjacent_background = np.zeros_like(full_foreground)
    adjacent_background[:, :, 1:] |= full_foreground[:, :, :-1]
    adjacent_background[:, :, :-1] |= full_foreground[:, :, 1:]
    adjacent_background &= full_background
    rows = []
    for i in range(25):
        fg, bg = tokens[i][foreground[i]], tokens[i][background[i]]
        fp, bp = response[i][foreground[i]], response[i][background[i]]
        separation = np.linalg.norm(fg.mean(0)-bg.mean(0))
        spread = np.sqrt(.5*((fg-fg.mean(0))**2).sum(1).mean()
                         + .5*((bg-bg.mean(0))**2).sum(1).mean())
        empty_separation = np.linalg.norm(tokens[i+25][foreground[i]].mean(0)
                                         - tokens[i+25][background[i]].mean(0))
        neighbors = full_response[i][adjacent_background[i]]
        rows.append(dict(i=i, fg=int(foreground[i].sum()), bg=int(background[i].sum()),
            zones=int(zones[i].sum()), fg_delta=float(np.median(fp)), bg_delta=float(np.median(bp)),
            delta_auc=auc(fp, bp), centroid_distance=float(separation), pooled_spread=float(spread),
            separation_ratio=float(separation/spread), empty_centroid_distance=float(empty_separation),
            adjacent_bg=int(adjacent_background[i].sum()), adjacent_bg_delta=float(np.median(neighbors)),
            adjacent_delta_auc=auc(full_response[i][full_foreground[i]], neighbors)))
    summary = dict(medians={k: float(np.median([r[k] for r in rows])) for k in rows[0] if k != 'i'},
        delta_auc_range=[min(r['delta_auc'] for r in rows), max(r['delta_auc'] for r in rows)],
        adjacent_delta_auc_range=[min(r['adjacent_delta_auc'] for r in rows), max(r['adjacent_delta_auc'] for r in rows)],
        delta_auc_above_half=sum(r['delta_auc'] > .5 for r in rows),
        adjacent_delta_auc_above_half=sum(r['adjacent_delta_auc'] > .5 for r in rows),
        foreground_cell_opportunities=int(foreground.sum()), background_cell_opportunities=int(background.sum()),
        whole_pair_delta_rms=np.sqrt(((tokens[:25]-tokens[25:])**2).mean((1, 2, 3))).tolist())
    # Reproduction check against the already reported no-fit diagnostic.
    assert summary['foreground_cell_opportunities'] == 1316
    assert summary['background_cell_opportunities'] == 5642
    assert summary['delta_auc_above_half'] == 22 and summary['adjacent_delta_auc_above_half'] == 14
    np.testing.assert_allclose(summary['medians']['delta_auc'], .7534869617950273, atol=1e-12)
    np.testing.assert_allclose(summary['medians']['adjacent_delta_auc'], .6266339869281046, atol=1e-12)
    write(output/'result.json', dict(status='PASS', per_frame=rows, summary=summary,
        formulas=dict(response='L2 norm of standardized64-channel paired token difference',
            foreground='Actual BODY_FAR or HEAD_FAR contributors in target cell, summed over returns; exclude same-cell far contribution in empty pair',
            background='Known in both frames, in a foreground-containing zone, no foreground/empty-far contribution',
            adjacent='Background cells immediately left or right of a foreground cell on56x56 angular grid, no wrap',
            separation='L2 foreground/background centroid distance divided by pooled RMS within-group Euclidean spread'),
        caveats=['One consumed25-pair configuration; cells and adjacent frames are not independent samples.',
                 'Pair differences and native grouping are diagnostic, not inference inputs.',
                 'Contributor cell labels are not full-object segmentation.',
                 'Interpolation does not create independent spatial measurements.',
                 'Measurable response does not prove sufficient semantic separability; weak localization does not prove representational impossibility.']))
    write(output/'receipt.json', dict(status='PASS', backend='CPU', reason='TASK_NOT_GPU_SUITABLE',
        task='Small descriptor-only statistics on cached50 feature maps; fixed CPU bilinear sampling, no backbone inference',
        training_steps=0, new_capture_frames=0, pairs=25, camera_parity=camera_parity,
        dense_indices=[3600, 3650], selected_present_indices=[100, 125], selected_empty_indices=[125, 150],
        inputs=inputs, source_sha256=sha(Path(__file__)),
        fixed_sampling_source_sha256=sha(Path(__file__).with_name('mz8_attribution.py')),
        elapsed_seconds=time.perf_counter()-start, outputs={'result.json': sha(output/'result.json')},
        validation='Exact reproduction of previously computed cell counts, AUC counts and median AUC'))
    print('PASS', summary, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    main(args.root, args.output)
