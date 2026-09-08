"""Descriptive score curves on consumed predictions; no chosen operating threshold."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def curve(scores, targets):
    if len(scores) != len(targets) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('Finite aligned probabilities required')
    if any(y not in (0, 1) for y in targets):
        raise ValueError('Known binary labels required')
    positives, negatives = sum(targets), len(targets)-sum(targets)
    if not positives or not negatives:
        return dict(roc_auc=None, average_precision=None, reason='Missing positive or negative class')
    groups = {}
    for s, y in zip(scores, targets):
        counts = groups.setdefault(s, [0, 0])
        counts[y] += 1
    tp = fp = 0
    points = [dict(fpr=0., recall=0., precision=1.)]
    auc = ap = 0.
    for score in sorted(groups, reverse=True):
        n, p = groups[score]
        tp += p; fp += n
        fpr, recall, precision = fp/negatives, tp/positives, tp/(tp+fp)
        last = points[-1]
        auc += (fpr-last['fpr'])*(recall+last['recall'])/2
        ap += (recall-last['recall'])*precision
        points.append(dict(fpr=fpr, recall=recall, precision=precision))
    return dict(n=len(targets), positives=positives, negatives=negatives,
        prevalence=positives/len(targets), roc_auc=auc, average_precision=ap,
        descriptive_recall_envelope={str(cap): max(p['recall'] for p in points if p['fpr'] <= cap)
            for cap in (.01, .05, .1)}, curve=points,
        warning='Label-derived consumed-set curve envelope, NOT a calibrated operating point; no thresholds exported')


def main(a):
    # Tie-aware checks cover the aggregation used for all score curves.
    assert curve([.1,.9], [0,1])['roc_auc'] == 1.
    assert curve([.1,.9], [1,0])['roc_auc'] == 0.
    tied = curve([.5,.5], [0,1])
    assert tied['roc_auc'] == .5 and tied['average_precision'] == .5
    assert curve([.5], [0])['roc_auc'] is None
    rows = json.loads(a.frames.read_text())
    outputs = {}
    for domain in ('city','willow_regression'):
        for head in ('BODY','HEAD'):
            for arm in ('baseline','city_finetuned'):
                selected = [r for r in rows if (r['domain'],r['head'],r['arm']) == (domain,head,arm)]
                if len(selected) != (750 if domain == 'city' else 96):
                    raise ValueError('Unexpected prediction coverage')
                outputs[f'{domain}/{head}/{arm}'] = curve([r['probability'] for r in selected], [r['target'] for r in selected])
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='PASS', optimizer_steps=0, backend='CPU: TASK_NOT_GPU_SUITABLE_SMALL_SCORE_ARRAYS',
        input_sha256=hashlib.sha256(a.frames.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), focused_checks='PASS', results=outputs)
    (a.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(9,7))
    for h, head in enumerate(('BODY','HEAD')):
        for arm in ('baseline','city_finetuned'):
            r=outputs[f'city/{head}/{arm}']; points=r['curve']
            axes[h,0].plot([p['fpr'] for p in points],[p['recall'] for p in points],label=f'{arm} AUC={r["roc_auc"]:.3f}')
            axes[h,1].step([p['recall'] for p in points],[p['precision'] for p in points],where='pre',label=f'{arm} AP={r["average_precision"]:.3f}')
        axes[h,0].set(xlabel='False-positive rate',ylabel='Recall',title=f'Plaza {head} ROC')
        axes[h,1].set(xlabel='Recall',ylabel='Precision',title=f'Plaza {head} PR')
        for ax in axes[h]:
            ax.set_xlim(0,1); ax.set_ylim(0,1.02); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Consumed Development score curves; no threshold selected')
    fig.tight_layout(); fig.savefig(a.output/'score-curves.png',dpi=150); plt.close(fig)
    print(json.dumps({k:{f:r[f] for f in ('roc_auc','average_precision','descriptive_recall_envelope')} for k,r in outputs.items()},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
