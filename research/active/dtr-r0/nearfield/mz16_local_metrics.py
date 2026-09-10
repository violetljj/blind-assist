"""Exact tied-score AP and fixed-FPR local diagnostic thresholds."""
import numpy as np


def average_precision(scores, labels):
    if not labels.any():return None
    order=np.argsort(-scores,kind='stable');s=scores[order];y=labels[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.cumsum(y,dtype=np.int64)[ends]
    return float(np.sum(np.diff(np.r_[0,tp])/(int(y.sum())) * tp/(ends+1)))


def threshold_at_fpr(scores, labels, rate=.01):
    negatives=np.sort(scores[~labels])
    if not len(negatives):return -np.inf
    budget=int(np.floor(rate*len(negatives)))
    return float(np.nextafter(np.float64(negatives[-budget-1]),np.inf))


def summarize_local(scores,labels,thresholds):
    rows=[]
    for q,(s,y) in enumerate(zip(scores,labels)):
        p=s.astype(np.float64)>=thresholds[q];tp=int((p&y).sum());fp=int((p&~y).sum());fn=int((~p&y).sum())
        rows.append(dict(ap=average_precision(s,y),tp=tp,fp=fp,fn=fn,positive=int(y.sum()),negative=int((~y).sum()),
            precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
            fpr=fp/int((~y).sum()) if (~y).any() else None))
    valid=[v['ap'] for v in rows if v['ap'] is not None]
    return dict(per_query=rows,macro_ap=float(np.mean(valid)) if valid else None)
