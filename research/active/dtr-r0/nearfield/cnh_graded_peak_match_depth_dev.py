"""Fixed observable inner/ring peak-depth contrasts on matched alert events.

Descriptive consumed Development diagnosis only. No alert rule, score/cutoff
change, fit or raw readout. Farther positive peak is not visible background/free.
"""
import csv
import argparse
import json
from pathlib import Path
import time
import numpy as np
import cnh_graded_corridor_eval_dev as C

ROOT=C.ROOT
TASK=ROOT/'artifacts.local/work/cnh-graded-peak-match-dev-20261010'
OUT=TASK/'depth'
BIN=.3002784
FIELDS=('inner_current_top1_forward_depth_m','ring_current_top1_forward_depth_m',
    'inner_current_top1_forward_depth_low_m','inner_current_top1_forward_depth_high_m',
    'ring_current_top1_forward_depth_low_m','ring_current_top1_forward_depth_high_m',
    'inner_current_top1_extent_membership','ring_current_top1_extent_membership',
    'inner_current_top1_native_zscore_max','ring_current_top1_native_zscore_max',
    'inner_current_top1_positive_peak_log','ring_current_top1_positive_peak_log',
    'inner_current_samebin_peak_share')


def rows(path):
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def summary(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    return dict(n=len(a),q25=float(np.quantile(a,.25)) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,q75=float(np.quantile(a,.75)) if len(a) else None)


def run(output=OUT,repair_of=None):
    global OUT
    OUT=output
    began=time.monotonic()
    if (OUT/'PLAN.json').exists():raise FileExistsError('Preserve depth-contrast diagnosis')
    C.save(OUT/'PLAN.json',dict(task='CNH_GRADED_PEAK_MATCH_DEV_20261010',phase='fixed observable depth contrast',
        total_task_budget=dict(unit='CPU command wall seconds',cap=600,matching_cap=180,audit_cap=90,depth_cap=90,integration_cap=180,GPU_seconds=0),
        decision_check='BODY rescue4/6/8 each384. Strict matched clear0/1/0 lacks coverage. All matched and unmatched reported, earlier events secondary not substitutes. Need actionable calibration-versus-observation distinction, not strict separability.',
        fields=FIELDS,gap_bins=dict(inner_closer='gap < -.3002784m',indistinct='-.3002784 <= gap <= .3002784m',inner_farther='gap > .3002784m'),
        gap='inner current strongest positive peak forward z minus ring current strongest positive peak forward z; overlap uses each supported depth interval',
        scope='All575 frozen selected alert-event rows and133 fixed pairs; cal and validation separate; perseed/panel no independent-count pooling',
        limitation='Descriptive only; no background attribution, coverage/free, quieting, effectiveness or causal separability claim. Peak distance from finite angular bins is not target depth.',
        stop='Complete fixed fields/partitions and report or caps; no caliper relaxation/cutoff change/model fit/rule/old-run reopening/backbone/raw/480/device',
        repair_of_plan_sha256=C.sha(repair_of/'PLAN.json') if repair_of else None,
        inputs_sha256={str(p.relative_to(ROOT)):C.sha(p) for p in (TASK/'events.csv',TASK/'matched_pairs.csv',
            C.OUT/'features/cal_features.npz',C.OUT/'features/validation_features.npz')},source_sha256=C.sha(Path(__file__))))
    event=rows(TASK/'events.csv');pairs=rows(TASK/'matched_pairs.csv');arrays={}
    for split in ('cal','validation'):
        with np.load(C.OUT/'features'/f'{split}_features.npz',allow_pickle=False) as a:
            arrays[split]=(a['features'],a['valid'],a['names'].tolist(),{int(s):i for i,s in enumerate(a['scene_ids'])})
    annotated=[];byid={}
    for e in event:
        f,v,n,ids=arrays[e['split']];index=(ids[int(e['scene'])],int(e['replica']),int(e['frame'])-3,1)
        r=dict(event_id=e['event_id'],split=e['split'],seed=e['seed'],kind=e['kind'],scene=e['scene'],replica=e['replica'],frame=e['frame'])
        for name in FIELDS:
            j=n.index(name);value=float(f[index][j]);r[name]=value if v[index][j] and np.isfinite(value) else None
        inner=r[FIELDS[0]];ring=r[FIELDS[1]];gap=inner-ring if inner is not None and ring is not None else None
        r['depth_gap_m']=gap
        r['depth_class']='MISSING' if gap is None else 'inner_farther' if gap>BIN else 'inner_closer' if gap<-BIN else 'indistinct'
        il,ih,rl,rh=(r[name] for name in FIELDS[2:6])
        r['depth_intervals_overlap']=None if any(x is None for x in (il,ih,rl,rh)) else bool(max(il,rl)<=min(ih,rh))
        annotated.append(r);byid[r['event_id']]=r
    from cnh_graded_evidence_dev import write_csv
    write_csv(OUT/'events.csv',annotated)
    groups={}
    for e in annotated:groups.setdefault(f"{e['split']}/{e['seed']}/{e['kind']}",[]).append(e)
    stats={}
    for key,ee in groups.items():
        stats[key]=dict(events=len(ee),depth_classes={k:sum(e['depth_class']==k for e in ee) for k in ('inner_farther','indistinct','inner_closer','MISSING')},
            overlap=sum(e['depth_intervals_overlap'] is True for e in ee),gap=summary([e['depth_gap_m'] for e in ee if e['depth_gap_m'] is not None]),
            fields={name:summary([e[name] for e in ee if e[name] is not None]) for name in FIELDS})
    output=[]
    for pair in pairs:
        a,b=byid[pair['anchor_event_id']],byid[pair['control_event_id']]
        r=dict(pair,anchor_depth_class=a['depth_class'],control_depth_class=b['depth_class'],anchor_gap=a['depth_gap_m'],control_gap=b['depth_gap_m'])
        for name in FIELDS+('depth_gap_m',):r['delta_'+name]=None if a[name] is None or b[name] is None else a[name]-b[name]
        output.append(r)
    write_csv(OUT/'pairs.csv',output)
    paired={}
    for r in output:
        key=r['panel']
        paired.setdefault(key,[]).append(r)
    pair_stats={k:dict(n=len(v),depth_combinations={f'{a}/{b}':sum(r['anchor_depth_class']==a and r['control_depth_class']==b for r in v)
        for a in ('inner_farther','indistinct','inner_closer','MISSING') for b in ('inner_farther','indistinct','inner_closer','MISSING')},
        deltas={name:summary([r['delta_'+name] for r in v if r['delta_'+name] is not None]) for name in FIELDS+('depth_gap_m',)}) for k,v in paired.items()}
    C.save(OUT/'stats.json',dict(events=stats,pairs=pair_stats))
    C.save(OUT/'result.json',dict(status='COMPLETE',seconds=time.monotonic()-began,events=len(annotated),pairs=len(output),GPU_seconds=0))
    print(json.dumps(dict(status='COMPLETE',seconds=time.monotonic()-began,events=len(annotated),pairs=len(output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--repair-of',type=Path)
    args=parser.parse_args()
    run(args.output,args.repair_of)
