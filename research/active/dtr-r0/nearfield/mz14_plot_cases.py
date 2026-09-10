"""Scientific case figures from frozen traces; overlays are evaluator-only."""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

from mz5_ensemble_readout import read, write, sha, load_npz
from mz8_attribution import AngularReturnReadout
from mz9_contributors import reconstruct


def edges():
    f = 320/math.tan(math.radians(50))
    return 319.5+f*np.tan(np.deg2rad(np.linspace(-22.5, 22.5, 57))), 179.5-f*np.tan(np.deg2rad(np.linspace(22.5, -22.5, 57)))


def full_grid(a):
    return a.reshape(8, 8, 7, 7).transpose(0, 2, 1, 3).reshape(56, 56)


def source_overlay(ax, mask, x, y, alpha=.25):
    color = np.zeros((*mask.shape, 4))
    color[mask] = [.1, .95, .85, alpha]
    ax.pcolormesh(x, y, color, shading='flat', rasterized=True)


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    work = root/'artifacts.local/work'
    trace = work/'mz14-evidence-trace-20260910/run-v2'
    receipt = read(trace/'receipt.json')
    cases = read(trace/'cases.json')
    assert sha(trace/'cases.json') == receipt['outputs']['cases.json']
    sign = next(c for c in cases if c.get('family') == 'hanging_sign' and not c['truth'] and c['previous'] >= 0 and c['baseline'] < 0)
    pole = next(c for c in cases if c['cohort'] == 'clean' and c['row'] == 112 and c['q'] == 3)
    empty = work/'mz6-short-sequence-20260910/capture-v3/model/sample/0137.png'
    inputs = {str(trace/'cases.json'): sha(trace/'cases.json')}
    images = []
    for c in [sign, pole]:
        for key in ['rgb', 'native']:
            assert sha(c[key]) == c[key+'_sha']
            inputs[c[key]] = c[key+'_sha']
        images.append(np.array(Image.open(c['rgb']).convert('RGB')))
    inputs[str(empty)] = sha(empty)
    empty_image = np.array(Image.open(empty).convert('RGB'))
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    packet = reconstruct(torch.from_numpy(np.load(sign['native'], allow_pickle=False)[None]).cuda())
    count = packet['source_counts'][0, sign['zone'], sign['echo']].cpu().numpy().reshape(7, 7)
    assert count.flat[sign['cell']] == sign['winning_source_pixels'] == 0
    distance = float(packet['range_m'][0, sign['zone'], sign['echo']])
    x, y = edges()
    zr, zc = divmod(sign['zone'], 8)
    sx, sy = x[zc*7:zc*7+8], y[zr*7:zr*7+8]
    cr, cc = divmod(sign['cell'], 7)
    rays, grid = AngularReturnReadout().rays, AngularReturnReadout().grid
    point = grid[sign['zone'], sign['cell']].numpy()
    wx, wy = (point[0]+1)*320-.5, (point[1]+1)*180-.5
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.7), gridspec_kw={'width_ratios':[1.45, 1]})
    axes[0].imshow(images[0]); axes[0].add_patch(Rectangle((sx[0], sy[0]), sx[-1]-sx[0], sy[-1]-sy[0], fill=False, edgecolor='#ef476f', lw=2))
    axes[0].scatter(wx, wy, marker='x', c='#ef476f', s=65, linewidths=2)
    axes[0].set_title('A  First sign false addition: HEAD_ONLY scene', loc='left')
    axes[1].imshow(images[0])
    source_overlay(axes[1], count > 0, sx, sy, .38)
    for v in sx: axes[1].axvline(v, color='white', lw=.6, alpha=.65)
    for v in sy: axes[1].axhline(v, color='white', lw=.6, alpha=.65)
    axes[1].scatter(wx, wy, marker='x', c='#ef476f', s=160, linewidths=3)
    axes[1].set_xlim(sx[0], sx[-1]); axes[1].set_ylim(sy[-1], sy[0])
    axes[1].set_title(f'B  Winning zone {sign["zone"]}, return {sign["echo"]}', loc='left')
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('A positive local maximum without a source at its location', x=.06, ha='left', fontsize=15, weight='bold')
    fig.text(.06, .135, f'{sign["query"]}: SOURCE margin {sign["source"]:.2f}; MZ5 margin {sign["baseline"]:.2f}.  Return range {distance:.2f} m.', fontsize=10)
    fig.text(.06, .09, 'Cyan cells: actual pixels contributing to this return.  Pink x: winning hypothesis (0 source pixels).', fontsize=9, color='#444444')
    fig.text(.06, .045, 'Evaluator-only overlays; this is a selected diagnostic case, not an independent trial.', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.04, right=.98, top=.85, bottom=.22, wspace=.12)
    fig.savefig(output/'sign_source_trace.png', dpi=180); plt.close(fig)
    cache = work/'mz8-attribution-20260910/cache-v5'
    run9 = work/'mz9-source-supervision-20260910/run-v1'
    labels = work/'mz9-source-supervision-20260910/labels-v1'
    n = load_npz(run9/'normalization.npz')
    maps = np.load(cache/'dense.npy', mmap_mode='r')
    with torch.no_grad():
        tokens = AngularReturnReadout().sample_visual(torch.from_numpy((np.array(maps[[3612, 3637]])-n['mean'])/n['std'])).numpy()
    delta = full_grid(np.linalg.norm(tokens[0]-tokens[1], axis=-1))
    q = load_npz(labels/'evaluator.npz')['query_counts'][3612]
    foreground = full_grid((q[...,1]+q[...,3]).sum(1) > 0)
    for path in [cache/'dense.npy', run9/'normalization.npz', labels/'evaluator.npz']:
        inputs[str(path)] = sha(path)
    paired_path = work/'mz14-evidence-trace-20260910/paired-v1/result.json'
    pair_stats = next(r for r in read(paired_path)['per_frame'] if r['i'] == 12)
    inputs[str(paired_path)] = sha(paired_path)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.7))
    for ax, im, title in zip(axes[:2], [images[1], empty_image], ['A  Pole present: frame 112', 'B  Paired pole absent: frame 137']):
        ax.imshow(im); ax.set_title(title, loc='left')
    source_overlay(axes[0], foreground, x, y, .25)
    heat = axes[2].pcolormesh(x, y, delta, cmap='magma', vmin=0, vmax=float(delta.max()), shading='flat', rasterized=True)
    axes[2].contour((x[:-1]+x[1:])/2, (y[:-1]+y[1:])/2, foreground.astype(float), levels=[.5], colors=['#32e0c4'], linewidths=1.3)
    axes[2].set_title('C  Paired feature-change magnitude', loc='left')
    for ax in axes:
        ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[-1], y[0]); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(heat, ax=axes[2], shrink=.78, pad=.03, label='L2 of standardized 64-D token difference')
    fig.suptitle('The pole changes features; the response also spreads into nearby cells', x=.055, ha='left', fontsize=14, weight='bold')
    fig.text(.055, .135, 'All panels show the same 45-degree ToF field. Cyan: actual far-query contributor cells (evaluator-only).', fontsize=10)
    fig.text(.055, .09, 'Fixed 18 x 32 feature map, interpolated to 7 x 7 cells per zone. Paired absent RGB is used for diagnosis only.', fontsize=9, color='#444444')
    fig.text(.055, .045, f'Frame 112 is the preselected middle pair; feature-change localization AUC: same-zone {pair_stats["delta_auc"]:.3f}, adjacent {pair_stats["adjacent_delta_auc"]:.3f}.', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.04, right=.98, top=.83, bottom=.22, wspace=.13)
    fig.savefig(output/'pole_pair_features.png', dpi=180); plt.close(fig)
    write(output/'metadata.json', dict(status='PASS', selection='First hanging-sign addedFP in cases order; fixed middle pole112/empty137, HEAD_FAR trace',
        sign=sign, pole=pole, paired_frame_stats=pair_stats, empty_rgb=str(empty), sign_range_m=distance,
        sign_source_cell_count=int((count > 0).sum()), sign_source_pixel_count=int(count.sum()),
        inputs=inputs, source_sha256=sha(Path(__file__)), outputs={p.name: sha(p) for p in output.glob('*.png')},
        interpretation='Actual native contributors are evaluator-only overlays. Images do not provide a model input or novel performance evidence.',
        backend='Matplotlib CPU visualization; CUDA native contributor reconstruction for one frozen sign frame'))
    print('PASS', sign['group'], sign['zone'], sign['cell'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.output)
