"""Fixed scores: original-only versus two-group scalar calibration points."""
import json
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from cnh_model_stability_plot_dev import curve, plt

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-group-calibration-dev-20261007'


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf8'))
    with np.load(OUT/'ledger.npz') as saved:
        z = dict(saved)
    colors = ['#2676b8', '#c76d27', '#38856c']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    for ax, tag in zip(axes, ('original', 'HB')):
        mask = (z['tag'] == tag) & z['evaluable']
        event, control = mask & z['contact'], mask & z['control']
        ax.axvspan(0, 5, color='#edf2f7', zorder=0)
        methods = [('original_center', 0, '#202c39', 'M3 max')]
        methods += [('raw_hb_aug', f, colors[f], f'Raw + HB, model {f}') for f in range(3)]
        xmax = 5
        for method, fold, color, label in methods:
            x, y = curve(z[f'score/fold{fold}/{method}'], event, control, z['deadline'])
            ax.step(x, y, where='post', color=color, lw=1.65, label=label)
            points = []
            for policy, cap, marker, filled in (
                ('shared_calibrated', '0.025', 'o', False),
                ('group_calibrated', '0.025', 'o', True),
                ('group_calibrated', '0.050', 'D', True),
            ):
                m = result['metrics'][f'fold{fold}/{policy}/{cap}/{method}'][tag]['all']
                px = 100*m['false_alarm_intervals']/m['control_intervals']
                py = 100*m['timely_excluding_warmup']/m['events']
                xmax = max(xmax, px)
                ax.plot(px, py, marker=marker, ms=6, color=color,
                        mfc=color if filled else 'white', mew=1.3, zorder=4)
                if cap == '0.025':
                    points.append((px, py))
            if points[0] != points[1]:
                ax.annotate('', xy=points[1], xytext=points[0],
                            arrowprops=dict(arrowstyle='->', color=color, lw=1), zorder=3)
        ax.set(title=f'{tag}: {int(event.sum())} events / {int(control.sum())} controls',
               xlabel='Observed false alarm intervals (%)', xlim=(0, max(6, np.ceil(xmax+0.5))), ylim=(0, 100))
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Timely events excluding warmup (%)')
    handles, _ = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color='#555', marker='o', mfc='white', ls='', label='Original-only target 2.5%'),
                Line2D([], [], color='#555', marker='o', ls='', label='Each-group target 2.5%'),
                Line2D([], [], color='#555', marker='D', ls='', label='Each-group target 5%')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .065), ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Same scores: one threshold constraining both calibration groups', fontsize=14, y=.99)
    fig.text(.5, .018, 'Arrows show calibration changes. Evaluation is consumed; actual false alarm rates need not equal calibration targets.',
             ha='center', fontsize=8, color='#555')
    fig.tight_layout(rect=(0, .20, 1, .94))
    for suffix in ('png', 'pdf'):
        path = OUT/f'group_calibration_operating_points.{suffix}'
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Saved group calibration figure; CPU plotting only')


if __name__ == '__main__':
    main()
