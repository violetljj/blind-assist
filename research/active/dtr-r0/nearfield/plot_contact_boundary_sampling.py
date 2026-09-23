"""Render the complete saved aggregate comparison; no fit or prediction calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[3] / 'artifacts.local/evidence/ba-contact-sampling-20260923'
OLD_COLOR = '#94A1B2'
NEW_COLOR = '#007F83'
INK = '#172B3A'
MUTED = '#566877'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def verify(root, previous):
    source = read(root/'source-seal.json')
    expected = next(d for p, d in source['inputs'].items()
                    if Path(p).resolve() == (previous/'result.json').resolve())
    assert sha(previous/'result.json') == expected, 'Old result differs from sealed control'
    for directory in (previous, root):
        seal = read(directory/'prediction-seal.json')
        for filename, expected in seal['files'].items():
            assert sha(directory/filename) == expected, filename
        for arm, expected in seal['selection_hashes'].items():
            assert sha(directory/(arm+'-selection.json')) == expected


def before(result, arm, split):
    item = result['arms'][arm]
    return item['training_layouts'] if split == 'train' else item


def after(result, arm, split):
    return result['arms'][arm][split]['metrics']


def clean_axis(ax):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#CBD4DA')
    ax.tick_params(axis='both', length=0, colors=MUTED, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E4E9ED', linewidth=.8)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel('Percent', fontsize=10, color=MUTED)


def bar_pairs(ax, old, new, labels):
    x = np.arange(len(labels))
    width = .32
    for values, offset, color in ((old, -width/2, OLD_COLOR), (new, width/2, NEW_COLOR)):
        bars = ax.bar(x+offset, values, width, color=color, linewidth=0, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+1.8, f'{value:.1f}',
                    ha='center', va='bottom', fontsize=10, color=INK, fontweight='normal')
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_xlim(-.65, len(labels)-.35)
    clean_axis(ax)


def panel_title(ax, title, subtitle):
    ax.set_title(title, loc='left', color=INK, fontsize=14, fontweight='bold', pad=31)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=10, color=MUTED)


def render(root):
    previous = root.with_name('ba-contact-boundary-20260923')
    verify(root, previous)
    old, new = read(previous/'result.json'), read(root/'result.json')
    assert old['status'] == new['status'] == 'PASS'
    receipts = read(root/'sampling-receipts.json')
    assert len(receipts) == 864 and all(r['queries'] == 72 and r['global_queries'] == 36 for r in receipts)
    for arm in ('direct', 'geometry'):
        assert old['selection'][arm] == read(previous/(arm+'-selection.json'))
        assert new['selection'][arm] == read(root/(arm+'-selection.json'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'axes.titleweight': 'bold',
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.7))
    fig.subplots_adjust(left=.065, right=.97, bottom=.145, top=.79, wspace=.23, hspace=.64)
    fig.text(.065, .957, 'Contact-boundary sampling: fixed-budget comparison',
             fontsize=22, fontweight='bold', color=INK)
    fig.text(.065, .92, 'Consumed simulation Development  |  24 train / 8 dev / 16 held-out layouts',
             fontsize=12, color=MUTED)
    fig.legend(handles=[Patch(color=OLD_COLOR, label='Old: fixed query grid'),
                        Patch(color=NEW_COLOR, label='New: continuous + boundary-aware sampling')],
               loc='upper left', bbox_to_anchor=(.06, .9), frameon=False, ncol=2, fontsize=11)
    rows = [(arm, kind) for arm in ('direct', 'geometry') for kind in ('width', 'horizon')]
    for ax, split, title in ((axes[0, 0], 'train', 'A  Training-layout boundary accuracy'),
                             (axes[0, 1], 'evaluation', 'B  Held-out-layout boundary accuracy')):
        old_rates, new_rates, labels = [], [], []
        for arm, kind in rows:
            b = before(old, arm, split)['boundaries'][kind]
            a = after(new, arm, split)['boundaries'][kind]
            assert b['interior_true_boundaries'] == a['interior_true_boundaries']
            old_rates.append(100*b['joint_within_5cm'])
            new_rates.append(100*a['joint_within_5cm'])
            labels.append(f'{arm.title()}\n{kind} · n={a["interior_true_boundaries"]}')
        bar_pairs(ax, old_rates, new_rates, labels)
        panel_title(ax, title, 'Within ±5 cm / all finite interior true boundaries; higher is better')
    b, a = before(old, 'geometry', 'evaluation'), after(new, 'geometry', 'evaluation')
    old_values = [100*b['both']['recall'], 100*b['both']['FPR']]
    new_values = [100*a['both']['recall'], 100*a['both']['FPR']]
    for kind in ('width', 'horizon'):
        old_values.append(100*b['boundaries'][kind]['wrong_crossings_on_right_censored'] /
                          b['boundaries'][kind]['right_censored_truth'])
        new_values.append(100*a['boundaries'][kind]['wrong_crossings_on_right_censored'] /
                          a['boundaries'][kind]['right_censored_truth'])
    bar_pairs(axes[1, 0], old_values, new_values,
              ['Recall ↑\nn=4,482 positives', 'FPR ↓\nn=23,166 negatives',
               'False width\ncrossing ↓ · n=624', 'False horizon\ncrossing ↓ · n=800'])
    panel_title(axes[1, 0], 'C  Geometry model: held-out benefits and costs',
                'Fixed off-grid queries (first 2); right-censored cases (last 2)')
    ax = axes[1, 1]
    panel_title(ax, 'D  Realized supervision: 72 unique queries / image',
                'One sampled set per training image, shared by both models for 100 epochs')
    total = sum(r['queries'] for r in receipts)
    global_count = sum(r['global_queries'] for r in receipts)
    boundary_count = sum(r['boundary_queries'] for r in receipts)
    fallback_count = sum(r['fallback_queries'] for r in receipts)
    assert global_count+boundary_count+fallback_count == total
    counts = [global_count, boundary_count, fallback_count]
    colors = ['#536D85', NEW_COLOR, '#DEB879']
    names = ['Uniform allocation', 'Valid boundary queries', 'Uniform fallback']
    left = 0.
    for count, color in zip(counts, colors):
        value = count/len(receipts)
        ax.barh(.81, value, left=left, height=.20, color=color)
        ax.text(left+value/2, .81, f'{value:.1f}', va='center', ha='center', fontsize=12,
                fontweight='bold', color='white' if color != colors[-1] else INK)
        left += value
    ax.set_xlim(0, 72)
    ax.set_ylim(0, 1)
    ax.axis('off')
    for row, (name, count, color) in enumerate(zip(names, counts, colors)):
        y = .56 - row*.15
        ax.scatter([1], [y], marker='s', s=78, color=color)
        ax.text(4, y, name, va='center', color=INK, fontsize=11)
        ax.text(70, y, f'{count:,} / {total:,}  ({100*count/total:.1f}%)',
                va='center', ha='right', color=MUTED, fontsize=11)
    fallback_fraction = fallback_count/(boundary_count+fallback_count)
    ax.text(0, .035, f'{100*fallback_fraction:.1f}% of intended boundary slots fell back to uniform sampling.',
            fontsize=10.5, color='#8A5D24')
    fig.text(.065, .079,
        'Boundary denominators include missing predictions as failures. Right-censored = no true boundary inside the sweep range; false crossings are counted separately.',
        color=MUTED, fontsize=10)
    fig.text(.065, .053,
        'Each fit uses its own cutoff selected only on the original dev query grid. Sampling changes both continuous coverage and boundary allocation; neither effect is isolated.',
        color=MUTED, fontsize=10)
    fig.text(.065, .027,
        'Same source, frozen features, initialization, class weight and 72-query budget. Aggregate results only; no natural-scene, hardware or safety claim.',
        color=MUTED, fontsize=10)
    description = json.dumps(dict(old_result_sha256=sha(previous/'result.json'),
        new_result_sha256=sha(root/'result.json'), sampling_receipts_sha256=sha(root/'sampling-receipts.json'),
        backend='TASK_NOT_GPU_SUITABLE', producer=Path(__file__).name))
    png, svg = root/'comparison.png', root/'comparison.svg'
    fig.savefig(png, dpi=160, metadata={'Description': description})
    fig.savefig(svg, metadata={'Description': description})
    plt.close(fig)
    return dict(png=str(png), svg=str(svg), queries=total, boundary_queries=boundary_count,
                fallback_queries=fallback_count, fallback_fraction_of_requested_boundary=fallback_fraction,
                geometry_evaluation_old=old_values, geometry_evaluation_new=new_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(render(args.root), indent=2))
