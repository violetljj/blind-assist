"""Constrained post-alert rejection; never alters the frozen A state."""
import numpy as np
import run_mz96_decision_head as common


def apply(base, false_score, threshold):
    return base.copy() if threshold is None else base & (false_score < threshold)


def compare(ids, labels, base, candidate):
    metrics, events=common.score(ids,labels,{'reference':base,'candidate':candidate})
    truth=labels['truth'];future=labels['future_only_truth'];a=metrics['reference'];b=metrics['candidate']
    delays=[];lost_events=0
    for e in events:
        first=e['arms']['reference']['first'];now=e['arms']['candidate']['first']
        if first is not None:
            if now is None:lost_events+=1
            else:delays.append((now-first)*.1)
    retained=int((truth&base&candidate).sum());future_base=int((future&base).sum())
    future_retained=int((future&base&candidate).sum())
    gates=dict(tp_retention_ge_98=retained>=.98*a['TP'],future_tp_retention_ge_98=future_retained>=.98*future_base,
        no_lost_reference_event=lost_events==0,delay_le_0p2=max(delays,default=0)<=.2+1e-9,
        false_segments_no_increase=b['false_segments']<=a['false_segments'],fragments_no_increase=b['fragments']<=a['fragments'])
    return dict(metrics=metrics,gates=gates,retained_TP=retained,reference_TP=a['TP'],
        TP_retention=retained/a['TP'] if a['TP'] else None,
        future_retained_TP=future_retained,future_reference_TP=future_base,
        future_TP_retention=future_retained/future_base if future_base else None,
        removed_FP=int((base&~candidate&~truth).sum()),added_FP=int((~base&candidate&~truth).sum()),
        lost_TP=int((base&~candidate&truth).sum()),added_TP=int((~base&candidate&truth).sum()),
        lost_reference_events=lost_events,max_added_delay_s=max(delays,default=0))


def select(ids, labels, base, false_score):
    rows=[];eligible=[]
    for threshold in np.arange(101)/100:
        outcome=compare(ids,labels,base,apply(base,false_score,threshold))
        row=dict(threshold=float(threshold),**outcome);rows.append(row)
        if all(outcome['gates'].values()) and outcome['removed_FP']>0:
            m=outcome['metrics']['candidate']
            eligible.append((-m['FP'],m['TP'],float(threshold)))
    threshold=max(eligible)[2] if eligible else None
    return dict(threshold=threshold,disabled=threshold is None,candidates=rows,
        selected=compare(ids,labels,base,apply(base,false_score,threshold)))
