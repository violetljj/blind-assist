"""Independent fixed event/pair observed-depth contrast audit; no raw/labels."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
TASK=ROOT/'artifacts.local/work/cnh-graded-peak-match-dev-20261010'
OUT=TASK/'depth/repair-1'
CORRIDOR=ROOT/'artifacts.local/work/cnh-graded-corridor-dev-20261010/features'
FIELDS=('inner_current_top1_forward_depth_m','ring_current_top1_forward_depth_m',
    'inner_current_top1_forward_depth_low_m','inner_current_top1_forward_depth_high_m',
    'ring_current_top1_forward_depth_low_m','ring_current_top1_forward_depth_high_m',
    'inner_current_top1_extent_membership','ring_current_top1_extent_membership',
    'inner_current_top1_native_zscore_max','ring_current_top1_native_zscore_max',
    'inner_current_top1_positive_peak_log','ring_current_top1_positive_peak_log','inner_current_samebin_peak_share')
CLASSES=('inner_farther','indistinct','inner_closer','MISSING')


def read(path):
    with path.open(encoding='utf8',newline='') as stream: return list(csv.DictReader(stream))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<22),b''): h.update(block)
    return h.hexdigest()


def distribution(values):
    good=sorted(float(x) for x in values if x is not None and np.isfinite(x))
    if not good: return dict(n=0,q25=None,median=None,q75=None)
    # Explicit linear order-statistic interpolation, independent of np.quantile.
    def q(frac):
        location=(len(good)-1)*frac; lower=int(location); upper=min(lower+1,len(good)-1)
        return good[lower]+(location-lower)*(good[upper]-good[lower])
    return dict(n=len(good),q25=q(.25),median=q(.5),q75=q(.75))


def same(actual,expected):
    if isinstance(expected,dict):
        assert set(actual)==set(expected)
        for key,value in expected.items(): same(actual[key],value)
    elif isinstance(expected,float):
        assert actual is not None
        np.testing.assert_allclose(float(actual),expected,atol=2e-12,rtol=2e-12)
    else: assert actual==expected,(actual,expected)


def audit():
    began=time.monotonic(); destination=OUT/'audit'
    if (destination/'result.json').exists(): raise FileExistsError('Preserve existing audit')
    destination.mkdir(parents=True,exist_ok=True)
    originals=read(TASK/'events.csv'); pairs=read(TASK/'matched_pairs.csv')
    saved_events=read(OUT/'events.csv'); saved_pairs=read(OUT/'pairs.csv')
    assert len(originals)==len(saved_events)==575 and len(pairs)==len(saved_pairs)==133
    arrays={}
    for split in ('cal','validation'):
        with np.load(CORRIDOR/f'{split}_features.npz') as a:
            arrays[split]={key:a[key] for key in ('features','valid','names','scene_ids','frames','queries')}
        assert arrays[split]['queries'][1]=='BODY'
        np.testing.assert_array_equal(arrays[split]['frames'],np.arange(3,16))
    expected=[]; masks=0; scalar_checks=0
    for old,saved in zip(originals,saved_events):
        r={key:old[key] for key in ('event_id','split','seed','kind','scene','replica','frame')}
        a=arrays[old['split']]; scene=np.flatnonzero(a['scene_ids']==int(old['scene']))
        assert len(scene)==1; replica=int(old['replica']); frame=int(old['frame'])
        assert 0<=replica<4 and 3<=frame<=15
        index=(int(scene[0]),replica,frame-3,1)
        for name in FIELDS:
            ix=int(np.flatnonzero(a['names']==name)[0]); value=float(a['features'][(*index,ix)])
            supported=bool(a['valid'][(*index,ix)]) and np.isfinite(value)
            r[name]=value if supported else None; masks+=not supported
            raw=saved[name]
            assert (raw!='')==supported
            if supported: assert float(raw)==value
        gap=None if r[FIELDS[0]] is None or r[FIELDS[1]] is None else r[FIELDS[0]]-r[FIELDS[1]]
        r['depth_gap_m']=gap
        r['depth_class']='MISSING' if gap is None else CLASSES[0] if gap>.3002784 else CLASSES[2] if gap<-.3002784 else CLASSES[1]
        endpoints=[r[name] for name in FIELDS[2:6]]
        r['depth_intervals_overlap']=None if any(x is None for x in endpoints) else max(endpoints[0],endpoints[2])<=min(endpoints[1],endpoints[3])
        for key,value in r.items():
            if value is None: assert saved[key]==''
            elif isinstance(value,float): assert float(saved[key])==value
            else: assert saved[key]==str(value)
        expected.append(r); scalar_checks+=len(r)
    byid={r['event_id']:r for r in expected}; assert len(byid)==575
    groups={}
    for r in expected: groups.setdefault(f"{r['split']}/{r['seed']}/{r['kind']}",[]).append(r)
    stats={}
    for key,rr in groups.items():
        stats[key]=dict(events=len(rr),depth_classes={name:sum(r['depth_class']==name for r in rr) for name in CLASSES},
            overlap=sum(r['depth_intervals_overlap'] is True for r in rr),gap=distribution([r['depth_gap_m'] for r in rr]),
            fields={name:distribution([r[name] for r in rr]) for name in FIELDS})
    expected_pairs=[]
    for original,saved in zip(pairs,saved_pairs):
        for key,value in original.items(): assert saved[key]==value
        a=byid[original['anchor_event_id']]; b=byid[original['control_event_id']]
        r=dict(original,anchor_depth_class=a['depth_class'],control_depth_class=b['depth_class'],anchor_gap=a['depth_gap_m'],control_gap=b['depth_gap_m'])
        for name in (*FIELDS,'depth_gap_m'):
            r['delta_'+name]=None if a[name] is None or b[name] is None else a[name]-b[name]
        for key,value in r.items():
            if value is None: assert saved[key]==''
            elif isinstance(value,float): assert float(saved[key])==value
            else: assert saved[key]==str(value)
        expected_pairs.append(r); scalar_checks+=len(r)
    panels={}
    for r in expected_pairs: panels.setdefault(r['panel'],[]).append(r)
    pair_stats={}
    for panel,rr in panels.items():
        pair_stats[panel]=dict(n=len(rr),depth_combinations={f'{a}/{b}':sum(r['anchor_depth_class']==a and r['control_depth_class']==b for r in rr) for a in CLASSES for b in CLASSES},
            deltas={name:distribution([r['delta_'+name] for r in rr]) for name in (*FIELDS,'depth_gap_m')})
    saved_stats=json.loads((OUT/'stats.json').read_text(encoding='utf8'))
    same(saved_stats,dict(events=stats,pairs=pair_stats))
    plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf8'))
    oldplan=json.loads((TASK/'depth/PLAN.json').read_text(encoding='utf8'))
    for key in ('fields','gap_bins','gap','scope','stop','inputs_sha256'): same(plan[key],oldplan[key])
    assert plan['repair_of_plan_sha256']==sha(TASK/'depth/PLAN.json')
    for rel,digest in plan['inputs_sha256'].items(): assert sha(ROOT/Path(rel))==digest
    oldevents=read(TASK/'depth/events.csv')
    assert oldevents==saved_events[:len(oldevents)]
    assert (TASK/'depth/failure.json').exists() and (TASK/'depth/initial_source.py').exists()
    assert time.monotonic()-began<30
    result=dict(status='PASS',events=575,pairs=133,event_groups=len(groups),pair_panels=len(panels),scalar_checks=scalar_checks,
        missing_field_values=masks,full_stats_independent=True,repair_preserved_same_criteria=True,
        CPU_seconds=time.monotonic()-began,CPU_seconds_cap=30,GPU_seconds=0,source_sha256=sha(Path(__file__)),
        diagnosis_source_sha256=sha(Path(__file__).with_name('cnh_graded_peak_match_depth_dev.py')),command=[sys.executable,*sys.argv],
        scope='Full BODY current13 feature indices/values/validity/gaps/bins/interval-overlap, fixed133 pair values/deltas and all linear-interpolated group summaries.',
        limitation='Observed peak depth and matched distributions only; no target depth, visible background, coverage, free state or silence benefit established.')
    (destination/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    inputs=[TASK/'events.csv',TASK/'matched_pairs.csv',OUT/'PLAN.json',OUT/'events.csv',OUT/'pairs.csv',OUT/'stats.json',
        TASK/'depth/PLAN.json',TASK/'depth/events.csv',TASK/'depth/failure.json',TASK/'depth/initial_source.py',
        *(CORRIDOR/f'{split}_features.npz' for split in ('cal','validation'))]
    receipt=dict(CPU_seconds=time.monotonic()-began,GPU_seconds=0,input_sha256={str(p.relative_to(ROOT/'artifacts.local')):sha(p) for p in inputs},
        output_sha256={'result.json':sha(destination/'result.json')},source_sha256=result['source_sha256'],diagnosis_source_sha256=result['diagnosis_source_sha256'],command=result['command'])
    (destination/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__': audit()
