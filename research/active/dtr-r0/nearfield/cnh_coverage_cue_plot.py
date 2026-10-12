"""CPU-only visualization of frozen coverage-cue results; no metric replacement."""
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-coverage-cue-geometry-20261005'
COLORS = dict(A='#d48124', B='#2b74b8', v1='#7a459f')
MARKERS = {.2:'o', .4:'s'}


def main():
    started = time.perf_counter()
    source = OUT/'result.json'
    result = json.loads(source.read_text(encoding='utf8'))
    cells = result['cells']
    selected = [c for c in cells if c['method'] in COLORS and not c['diagnostic']]
    if not selected:
        raise ValueError('No complete parameter cell available for plotting')
    degenerate = result['primary_unnecessary_metric_degenerate']
    floors = [c['total_k3_per_min'] for c in cells if c['method']=='zero']
    floor_text = f'{min(floors):.0f}' if np.ptp(floors)<1e-8 else f'{min(floors):.0f}-{max(floors):.0f}'
    y = np.array([c['total_k3_per_min'] for c in cells])
    dy = max(np.ptp(y), 100.)
    ylimits = (max(0, np.min(y)-.07*dy), np.max(y)+.07*dy)
    xs = [c['restorable_unnecessary_total_per_min'] for c in selected]
    xmax = max(max(xs)*1.05, 1.)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','axes.labelcolor':'#26303c','text.color':'#26303c'})
    fig = plt.figure(figsize=(20, 20), facecolor='white')
    gs = fig.add_gridspec(8, 6, height_ratios=[.32, .82, 1, 1, 1, 1, 1, 1],
        left=.075, right=.975, bottom=.083, top=.90, hspace=.51, wspace=.27)
    primary = fig.add_subplot(gs[1, :])
    for method, color in COLORS.items():
        values = [c for c in selected if c['method']==method]
        for turn, marker in MARKERS.items():
            group = [c for c in values if c['T']==turn]
            primary.scatter([c['unnecessary_total_per_min'] for c in group],
                [c['total_k3_per_min'] for c in group], s=13, alpha=.36,
                c=color, marker=marker, linewidths=.3, edgecolors='white')
    primary.set_ylim(*ylimits)
    all_primary_x = [c['unnecessary_total_per_min'] for c in selected]
    if degenerate:
        primary.set_xlim(-.5,.5)
        primary.set_xticks([0])
    else:
        primary.set_xlim(-.05*max(max(all_primary_x),1),1.05*max(max(all_primary_x),1))
    primary.grid(alpha=.13)
    primary.set_xlabel('Frozen PRIMARY: unnecessary tagged-side cues / min (all parameter cells; no jitter)', labelpad=4)
    primary.set_ylabel('Insufficient\ncell arrivals / min', fontsize=9)
    primary.set_title('PRIMARY axis is degenerate: all x = 0; it cannot establish tradeoff superiority'
        if degenerate else 'Frozen PRIMARY any-cell unnecessary-cue metric (all parameter cells)',
        loc='left', fontsize=12, fontweight='bold', pad=7)
    primary.axhline(min(floors), color='#737d87', ls=':', lw=1)
    primary.annotate(f'Yaw-zero geometric floor: {floor_text} / min',
        xy=(.98,.055), xycoords='axes fraction', ha='right', fontsize=9,
        bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=2))
    facets = []
    missing = []
    for si, sigma in enumerate((5,10,15)):
        for ti, tau in enumerate((.5,1.,2.,4.)):
            base = {c['method']:c for c in cells if c['sigma']==sigma and c['tau']==tau and c['method'] in ('none','dual','zero')}
            for li, latency in enumerate((.3,.6,1.)):
                row, col = si*2+ti//2, (ti%2)*3+li
                ax = fig.add_subplot(gs[row+2,col]);facets.append(ax)
                ax.set_title(rf'$\sigma={sigma}^\circ,\ \tau={tau:g}$ s  |  $L={latency:g}$ s', fontsize=10, loc='left', pad=7)
                ax.set_xlim(-.035*xmax, xmax);ax.set_ylim(*ylimits)
                ax.grid(color='#d7dde4', lw=.6, alpha=.6)
                if not base:
                    ax.text(.5,.5,'NOT RUN', transform=ax.transAxes,ha='center')
                    missing.append(dict(sigma=sigma,tau=tau,L=latency));continue
                ax.axhline(base['none']['total_k3_per_min'], color='#40515f', ls='-', lw=1.1, alpha=.8)
                ax.axhline(base['dual']['total_k3_per_min'], color='#168177', ls='-.', lw=1.1, alpha=.9)
                ax.axhline(base['zero']['total_k3_per_min'], color='#929da7', ls=':', lw=1.2)
                for method, color in COLORS.items():
                    for turn, marker in MARKERS.items():
                        points = sorted([c for c in selected if c['sigma']==sigma and c['tau']==tau and c['L']==latency and c['T']==turn and c['method']==method], key=lambda c:c['p'])
                        if not points:continue
                        xx = [c['restorable_unnecessary_total_per_min'] for c in points]
                        yy = [c['total_k3_per_min'] for c in points]
                        ax.plot(xx, yy, color=color, ls='--', lw=1.25, marker=marker,
                            markersize=4, markeredgecolor='white', markeredgewidth=.4, alpha=.85)
                if col==0:
                    ax.set_ylabel('Insufficient cell arrivals / min', fontsize=9)
                else:
                    ax.tick_params(labelleft=False)
                if row==5:
                    ax.set_xlabel('DIAGNOSTIC unnecessary\ntagged-side cues / min',fontsize=9)
                ax.tick_params(labelsize=8)
    handles = [Line2D([0],[0],color=color,lw=2,label=method) for method,color in COLORS.items()]
    handles += [Line2D([0],[0],color='#66727e',ls='--',marker=m,lw=1.2,label=f'Turn {t:g} s') for t,m in MARKERS.items()]
    handles += [Line2D([0],[0],color='#40515f',lw=1.1,label='No cue'),
        Line2D([0],[0],color='#168177',ls='-.',lw=1.1,label='Dual +/-15 deg, no cue'),
        Line2D([0],[0],color='#929da7',ls=':',lw=1.2,label='Single yaw-zero')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.535,.918), ncol=8, frameon=False, fontsize=10)
    fig.suptitle('Coverage-cue tradeoffs under synthetic head motion',x=.075,ha='left',y=.987,fontsize=24,fontweight='bold')
    fig.text(.075,.958,'Frozen primary above; dashed facets use an evaluator-only yaw-restorable-cell diagnostic.\n'
        f'All y values retain the full finite-cell residual, including the {floor_text}/min yaw-zero geometric floor.',
        ha='left',va='top',fontsize=12,linespacing=1.5)
    fig.text(.075,.038,'Facets: 3 OU amplitudes x 4 correlation times x 3 reaction delays; curves connect p = 0, 1, 2, 3, 5 in order (A has one point).\n'
        'The diagnostic asks whether yaw-restorable cells would fail without this cue; it does not replace the frozen any-cell primary metric.\n'
        f'4 synthetic replicas per motion setting; 120 s streams. {result["completed_streams"]}/{result["expected_streams"]} streams complete. '
        'No human-motion frequency, product threshold, or primary-policy superiority claim.',
        fontsize=10,ha='left',va='top',linespacing=1.5)
    png, svg = OUT/'coverage_cue_tradeoff.png', OUT/'coverage_cue_tradeoff.svg'
    fig.savefig(png,dpi=160,facecolor='white')
    fig.savefig(svg,facecolor='white')
    plt.close(fig)
    receipt = dict(status='COMPLETE',backend='matplotlib Agg + numpy CPU; no torch',
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        figure_paths=[str(png),str(svg)],facets=36,missing_facets=missing,
        primary_axis_degenerate=degenerate,primary_x_max=float(max(all_primary_x)),
        full_residual_y=True,diagnostic_not_primary=True,yaw_zero_floor_per_min=[min(floors),max(floors)],
        seconds=time.perf_counter()-started)
    (OUT/'plot_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
