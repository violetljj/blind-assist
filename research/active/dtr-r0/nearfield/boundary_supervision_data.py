"""Geometry-independent dense slice schedule and frozen decision rules."""
from __future__ import annotations
import numpy as np
from contact_boundary_data import queries

SEED=202609278


def schedule():
    source=queries()
    key=lambda row:tuple(np.round(np.asarray(row,float),6))
    anchors={key(row) for row in source['seen'] if row[2]==0}
    banks={}
    for name,field in [('A','seen'),('W','width_curve'),('H','horizon_curve')]:
        banks[name]=[(field,i) for i,row in enumerate(source[field]) if row[2]==0
                     and (name=='A' or key(row) not in anchors)]
    assert [len(banks[k]) for k in ('A','W','H')]==[36,47,46]
    rng=np.random.default_rng(SEED)
    banks={k:[v[int(i)] for i in rng.permutation(len(v))] for k,v in banks.items()}
    qlist=[];mapping=[]
    for epoch in range(100):
        selected=[]
        for layer in range(2):
            for bank in ('A','W','H'):
                entries=banks[bank]
                for slot in range(12):
                    field,index=entries[(12*epoch+slot)%len(entries)]
                    selected.append((field,index+layer*(len(source[field])//2)))
        q=np.stack([source[field][index] for field,index in selected])
        assert len({key(row) for row in q})==72
        qlist.append(q);mapping.append(selected)
    return np.stack(qlist),mapping


def training_labels(targets,mapping):
    return np.stack([np.stack([targets[field][:,index] for field,index in epoch],axis=1)
                     for epoch in mapping])


def decision(old,new):
    gain={}
    for split in ('train','evaluation'):
        gain[split]={kind:new[split]['boundaries'][kind]['joint_within_5cm']-
                    old[split]['boundaries'][kind]['joint_within_5cm'] for kind in ('width','horizon')}
    train=all(v>=.1-1e-12 for v in gain['train'].values())
    transfer=all(v>=.1-1e-12 for v in gain['evaluation'].values())
    a,b=old['evaluation'],new['evaluation']
    cost=dict(recall=b['both']['recall']>=a['both']['recall']-.03-1e-12,
              FPR=b['both']['FPR']<=a['both']['FPR']+.01+1e-12,
              false_crossings=all(b['boundaries'][k]['wrong_crossings_on_right_censored']<=
                  a['boundaries'][k]['wrong_crossings_on_right_censored']+5 for k in ('width','horizon')))
    supported=train and transfer and all(cost.values())
    status=('MECHANISM_SUPPORTED' if supported else 'ACCURACY_COST_TRADEOFF' if train and transfer
            else 'FITTING_ONLY' if train else 'DENSE_SUPERVISION_INSUFFICIENT')
    return dict(decision=status,train_gain_pass=train,transfer_gain_pass=transfer,costs=cost,
                boundary_accuracy_gain=gain,mechanism_supported=supported,
                full_component_pass=bool(b['component_gate']))
