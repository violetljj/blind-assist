"""Two separate interval scopes, drawn from immutable bootstrap results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-readout-uncertainty-dev-20261007'


def main():
    r = json.loads((OUT/'result.json').read_text(encoding='utf8'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    labels = [f'{tag}, fit {f}' for tag in ('original', 'HB') for f in range(3)]
    y = np.arange(6)
    for i, (tag, fold) in enumerate((t,f) for t in ('original','HB') for f in range(3)):
        a = r['A_fixed_score_ranking']['results'][f'{tag}/fit{fold}']
        b = r['B_recalibrated_policy']['results'][f'0.025/{tag}/fit{fold}']
        color = '#2676b8' if tag == 'original' else '#c76d27'
        for ax, point, ci in ((axes[0], a['point_difference'], a['ci95']),
                              (axes[1], b['point_timely_gain_pp'], b['timely_gain_pp']['ci95'])):
            ax.plot(ci, [i,i], color=color, lw=2)
            ax.plot(point, i, 'o', color=color, ms=5)
    for ax in axes:
        ax.axvline(0, color='#666', lw=1, ls='--')
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set(title='A: fixed score ranking', xlabel='Normalized whole-tie area gain, FPR 0–5%')
    axes[1].set(title='B: recalibrated policy, target 2.5%', xlabel='Timely event gain (percentage points)')
    fig.text(.27, .075, 'Evaluation units resampled; fixed scores', ha='center', fontsize=9)
    fig.text(.77, .075, 'Calibration and evaluation units resampled', ha='center', fontsize=9)
    fig.text(.5, .02, '1000 paired unit draws, percentile 95% intervals. Fixed models; training uncertainty excluded. Consumed Development.',
             ha='center', fontsize=8, color='#555')
    fig.tight_layout(rect=(0,.13,1,1))
    for ext in ('png','pdf'):
        p = OUT/f'readout_uncertainty_scopes.{ext}'
        if p.exists():
            raise FileExistsError(p)
        fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Saved two-scope interval figure')


if __name__ == '__main__':
    main()
