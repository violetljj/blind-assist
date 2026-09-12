"""Evaluate fixed interval readout on authenticated visual inputs, then truth."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
import mz108_competitive_association as visual
import mz109_interval_extent as model
from run_mz107_four_sensor import ROOT,sha,write,readrows,truth,metrics
from run_mz108_competitive_association import association_audit
from research_backend import BackendCandidate,DeviceObservation,select_backend


def run(new_capture,out):
    out=out.resolve();assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists();out.mkdir(parents=True)
    for p in (Path(__file__),Path(model.__file__),Path(visual.__file__),Path(__file__).with_name('MZ109_PROTOCOL_20260913.md')):shutil.copyfile(p,out/p.name)
    work=ROOT/'artifacts.local/work';old_study=work/'mz108-competitive-association-20260913/analysis-v2'
    old_seal=json.loads((old_study/'prediction-seal.json').read_text())
    assert old_seal['code_sha256'][Path(visual.__file__).name]==sha(Path(visual.__file__))
    sources=dict(consumed_mz107=work/'mz107-rgb-tof-radar-imu-20260913/capture-v1',
                 consumed_mz108=work/'mz108-competitive-association-20260913/capture-v2',new_mz109=new_capture)
    packets={};values={};seals={};costs={}
    for panel,capture in sources.items():
        capture=capture.resolve();receipt=json.loads((capture/'receipt.json').read_text());assert receipt['status']=='PASS' and receipt['frames']==96
        for name,digest in receipt['hashes'].items():assert sha(capture/name)==digest,name
        manifest=json.loads((capture/'manifest.json').read_text());assert manifest['rgb_camera_count']==1 and not manifest['depth_images_produced']
        rows=readrows(capture/'raw.jsonl');assert len(rows)==96 and len({r['id'] for r in rows})==96
        if panel!='new_mz109':
            key='consumed_mz107' if panel=='consumed_mz107' else 'new_mz108';path=old_study/key/'predictions.json'
            assert old_seal['panels'][key]['predictions_sha256']==sha(path)
            assert old_seal['panels'][key]['raw_sha256']==sha(capture/'raw.jsonl')
            nominal=json.loads(path.read_text())['combined'];assert [r['id'] for r in rows]==[p['id'] for p in nominal]
            visual_seconds=None
        else:
            images={r['id']:cv2.imread(str(capture/r['rgb_path'])) for r in rows};assert all(im is not None for im in images.values())
            t=time.perf_counter();nominal=visual.predict(rows,lambda r:images[r['id']]);visual_seconds=(time.perf_counter()-t)/96
            disabled=visual.predict(rows,lambda _:None)
            assert all(p['baseline']==d['candidate'] for p,d in zip(nominal,disabled))
        if not packets:
            select_backend('scalar-scoring',cpu=BackendCandidate('python-interval-cpu','cpu',lambda:model.refine(rows[0],nominal[0]),
                lambda _:DeviceObservation('cpu','host CPU','Python analytic interval arithmetic')),
                record_path=out/'backend.json',capabilities={'interval_workload':'small scalar trigonometric intervals',
                 'visual_backend':'MZ108 OpenCV CPU; reused authenticated cache for old panels; no CUDA MSER backend'})
        started=time.perf_counter();interval=[model.refine(r,p) for r,p in zip(rows,nominal)];extra=(time.perf_counter()-started)/96
        assert all(p['baseline']==q['baseline'] and (not p['tof_support'] or q['candidate']) for p,q in zip(nominal,interval))
        target=out/panel;target.mkdir()
        write(target/'predictions.json',dict(nominal=[dict(p,id=r['id']) for r,p in zip(rows,nominal)],
                                            interval=[dict(p,id=r['id']) for r,p in zip(rows,interval)]))
        seals[panel]=dict(raw_sha256=sha(capture/'raw.jsonl'),capture_receipt_sha256=sha(capture/'receipt.json'),prediction_sha256=sha(target/'predictions.json'))
        packets[panel]=(capture,rows);values[panel]=(nominal,interval);costs[panel]=dict(visual_seconds_per_frame=visual_seconds,extra_interval_seconds_per_frame=extra)
    write(out/'prediction-seal.json',dict(status='ALL_THREE_PANELS_SEALED_BEFORE_EVALUATOR_READ',panels=seals,
        code_sha256={p.name:sha(p) for p in out.glob('*.py')},protocol_sha256=sha(out/'MZ109_PROTOCOL_20260913.md')))
    summaries={}
    for panel,(capture,rows) in packets.items():
        es=readrows(capture/'evaluator.jsonl');vs=readrows(capture/'provenance.jsonl')
        assert [r['id'] for r in rows]==[e['id'] for e in es]==[v['id'] for v in vs]
        nominal,interval=values[panel];t=np.array([truth(e) for e in es],bool)
        b=np.array([p['baseline'] for p in nominal],bool);n=np.array([p['candidate'] for p in nominal],bool);c=np.array([p['candidate'] for p in interval],bool)
        groups={};states=Counter();changes=[];audit=[]
        for r,e,v,p,q in zip(rows,es,vs,nominal,interval):
            for a in q['associations']:states[a['geometry']['state']]+=1
            audit.extend(association_audit(r,q,e,v))
            if q['candidate']!=p['baseline'] or q['candidate']!=p['candidate']:
                changes.append(dict(id=r['id'],truth=truth(e),baseline=p['baseline'],nominal=p['candidate'],interval=q['candidate'],
                                    tof_support=q['tof_support'],associations=q['associations']))
        for family in sorted({e['family'] for e in es}):
            mask=np.array([e['family']==family for e in es]);groups[family]=dict(baseline=metrics(t[mask],b[mask]),
                nominal=metrics(t[mask],n[mask]),interval=metrics(t[mask],c[mask]),lost_baseline_TP=int((mask&t&b&~c).sum()),lost_nominal_TP=int((mask&t&n&~c).sum()))
        changed_ids={r['id'] for r in changes if r['baseline']!=r['interval']}
        bad=sum(not a['identity_supported'] and a['id'] in changed_ids and a['baseline_support']!=a['refined_support'] for a in audit)
        summary=dict(baseline=metrics(t,b),nominal=metrics(t,n),interval=metrics(t,c),
            lost_baseline_TP=int((t&b&~c).sum()),lost_nominal_TP=int((t&n&~c).sum()),new_TP=int((t&~b&c).sum()),
            removed_baseline_FP=int((~t&b&~c).sum()),added_baseline_FP=int((~t&~b&c).sum()),
            geometry_states=dict(states),changed_decisions_with_wrong_association=bad,groups=groups,timing=costs[panel])
        summaries[panel]=summary;write(out/panel/'changes.json',changes);write(out/panel/'association-audit.json',audit)
        print(panel,json.dumps({k:v for k,v in summary.items() if k not in ('groups','timing')}))
    gain=any(s['interval']['FP']<s['baseline']['FP'] for s in summaries.values()) and all(
        s['interval']['FP']<=s['baseline']['FP'] and s['lost_baseline_TP']==0 and s['changed_decisions_with_wrong_association']==0 for s in summaries.values())
    write(out/'summary.json',dict(status='CONTROLLED_TASK_GAIN' if gain else 'NO_QUALIFIED_TASK_GAIN',panels=summaries,
        scope='CONDITIONAL_INTERVAL_GEOMETRY_CONTROLLED_DEVELOPMENT',independent_tof_preserved=True,
        limits=['not calibrated probability','filled-box radial-shell proxy','fixed empirical tolerances','no CLEAR state']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--new-capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.new_capture,a.output)
