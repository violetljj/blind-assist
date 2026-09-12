"""One fixed four-arm study; seal both panels before evaluator access."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
import mz107_rgb_association as legacy
import mz108_competitive_association as model
from run_mz107_four_sensor import ROOT, sha, write, readrows, truth, metrics
from research_backend import BackendCandidate, DeviceObservation, select_backend

ARMS = dict(original=(False,False),regions_only=(True,False),competition_only=(False,True),combined=(True,True))


def association_audit(row, pred, e, prov):
    objects = {row['episode_id']+'/'+o['name']:o for o in e['native_bounds']}
    cam=e['camera'];origin=np.array([cam['x'],cam['y'],cam['z']]);results=[]
    for a in pred['associations']:
        if a['state']!='ASSOCIATED': continue
        source=prov['radar_slots'][a['slot']];owners=[]
        for k in a['tof_zones']:
            point=origin+e['tof_native_ranges_m'][k]*model.ray(row['tof64_theta_deg'][k],row['tof64_phi_deg'][k],cam['pitch'],cam['yaw'])
            owners.append([key for key,o in objects.items() if np.all(point>=np.array(o['center_m'])-o['extent_m']-.002)
                           and np.all(point<=np.array(o['center_m'])+o['extent_m']+.002)])
        supported=source.get('actor_id') is not None and all(o==[source['actor_id']] for o in owners)
        results.append(dict(id=row['id'],**a,source=source['kind'],identity_supported=supported))
    return results


def projected_objects(row,e):
    from mz107_sensors import basis  # Evaluator-only camera geometry.
    cam=e['camera'];pos=np.array([cam['x'],cam['y'],cam['z']]);rotation=np.array(basis(cam));intr=row['rgb_intrinsics']
    result=[]
    for obj in e['native_bounds']:
        points=[rotation@(np.array(obj['center_m'])+np.array(sign)*obj['extent_m']-pos)
                for sign in itertools.product((-1,1),repeat=3)]
        if any(p[0]<=0 for p in points): continue
        us=[intr['cx']+intr['fx']*p[1]/p[0] for p in points];vs=[intr['cy']-intr['fy']*p[2]/p[0] for p in points]
        box=[max(0,min(us)),max(0,min(vs)),min(intr['width'],max(us)),min(intr['height'],max(vs))]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        result.append(box)
    return result


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0


def run(old_capture,new_capture,out):
    out=out.resolve();assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists();out.mkdir(parents=True)
    for f in (Path(__file__),Path(model.__file__),Path(legacy.__file__),Path(__file__).with_name('MZ108_PROTOCOL_20260913.md')):shutil.copyfile(f,out/f.name)
    packets={};predictions={};timings={};seals={}
    for panel,path in dict(consumed_mz107=old_capture,new_mz108=new_capture).items():
        path=path.resolve();receipt=json.loads((path/'receipt.json').read_text())
        assert receipt['status']=='PASS' and receipt['frames']==96
        for name,digest in receipt['hashes'].items():assert sha(path/name)==digest
        manifest=json.loads((path/'manifest.json').read_text());assert manifest['rgb_camera_count']==1 and not manifest['depth_images_produced']
        rows=readrows(path/'raw.jsonl');assert len(rows)==96 and len({r['id'] for r in rows})==96
        images={}
        for r in rows:
            im=cv2.imread(str(path/r['rgb_path']));assert im is not None
            images[r['id']]=im
        if not packets:
            select_backend('batch-tensor',cpu=BackendCandidate('opencv-cpu','cpu',lambda:model.proposals(images[rows[0]['id']]),
                lambda _:DeviceObservation('cpu','host CPU','OpenCV '+cv2.__version__)),cpu_reason='GPU_BACKEND_UNAVAILABLE',
                record_path=out/'backend.json',capabilities={'opencv':cv2.__version__,'cuda_devices':cv2.cuda.getCudaEnabledDeviceCount(),
                'reason':'No installed CUDA MSER/connected-component backend'})
        loader=lambda r:images[r['id']];values={};cost={}
        for arm,(regions,competition) in ARMS.items():
            t=time.perf_counter();values[arm]=model.predict(rows,loader,regions,competition);cost[arm]=(time.perf_counter()-t)/len(rows)
        old=legacy.predict(rows,loader)
        assert all(p['candidate']==q['candidate'] and p['baseline']==q['baseline'] and p['proposals']==q['proposals'] for p,q in zip(values['original'],old))
        disabled=model.predict(rows,lambda _:None)
        for vals in values.values():
            assert all(p['baseline']==d['candidate'] and (not p['tof_support'] or p['candidate']) for p,d in zip(vals,disabled))
        if panel=='consumed_mz107':
            cached=json.loads((path.parent/'analysis-v1/predictions.json').read_text())
            assert all(p['baseline']==q['baseline'] and p['candidate']==q['candidate'] for p,q in zip(old,cached))
        target=out/panel;target.mkdir();write(target/'predictions.json',{arm:[dict(id=r['id'],**p) for r,p in zip(rows,v)] for arm,v in values.items()})
        seals[panel]=dict(source_receipt_sha256=sha(path/'receipt.json'),raw_sha256=sha(path/'raw.jsonl'),predictions_sha256=sha(target/'predictions.json'))
        packets[panel]=(path,rows);predictions[panel]=values;timings[panel]=cost
    write(out/'prediction-seal.json',dict(status='ALL_PANELS_ALL_ARMS_SEALED_BEFORE_EVALUATOR_READ',panels=seals,
        code_sha256={p.name:sha(p) for p in out.glob('*.py')},protocol_sha256=sha(out/'MZ108_PROTOCOL_20260913.md')))
    summaries={}
    for panel,(path,rows) in packets.items():
        es=readrows(path/'evaluator.jsonl');vs=readrows(path/'provenance.jsonl')
        assert [r['id'] for r in rows]==[e['id'] for e in es]==[v['id'] for v in vs]
        t=np.array([truth(e) for e in es],bool);b=np.array([p['baseline'] for p in predictions[panel]['original']],bool)
        result=dict(baseline=metrics(t,b),arms={},baseline_fp_origins=Counter())
        for r,e,v,p in zip(rows,es,vs,predictions[panel]['original']):
            if p['baseline'] and not truth(e):
                origins={v['radar_slots'][a['slot']]['kind'] for a in p['associations'] if a['baseline_support']}
                if p['tof_support']:origins.add('ToF')
                result['baseline_fp_origins']['+'.join(sorted(origins))]+=1
        for arm,vals in predictions[panel].items():
            c=np.array([p['candidate'] for p in vals],bool);audit=[];groups={}
            for r,e,v,p in zip(rows,es,vs,vals):
                audit.extend(association_audit(r,p,e,v))
                group=groups.setdefault(e['family'],dict(frames=0,proposal_frames=0,projected_objects=0,matched_objects_iou50=0,old_TP_lost=0))
                boxes=projected_objects(r,e);group['frames']+=1;group['proposal_frames']+=bool(p['proposals']);group['projected_objects']+=len(boxes)
                group['matched_objects_iou50']+=int(sum(max([iou(box,pb) for pb in p['proposals']],default=0)>=.5 for box in boxes))
            for family,g in groups.items():
                mask=np.array([e['family']==family for e in es]);g.update(metrics=metrics(t[mask],c[mask]),old_TP_lost=int((t&b&~c&mask).sum()))
            correct=sum(a['identity_supported'] for a in audit)
            changes=[dict(id=r['id'],truth=bool(gt),**p) for r,p,gt in zip(rows,vals,t) if p['baseline']!=p['candidate']]
            changed_ids={r['id'] for r in changes}
            bad_changes=sum(not a['identity_supported'] and a['id'] in changed_ids and a['baseline_support']!=a['refined_support'] for a in audit)
            result['arms'][arm]=dict(metrics=metrics(t,c),old_TP_lost=int((t&b&~c).sum()),new_TP_gained=int((t&~b&c).sum()),
                removed_FP=int((~t&b&~c).sum()),added_FP=int((~t&~b&c).sum()),associations=len(audit),correct_associations=correct,
                wrong_associations=len(audit)-correct,wrong_associations_in_changed_decisions=bad_changes,
                family=groups,seconds_per_frame=timings[panel][arm])
            write(out/panel/(arm+'-association-audit.json'),audit);write(out/panel/(arm+'-changed-frames.json'),changes)
        summaries[panel]=result
    write(out/'summary.json',dict(status='FIXED_STUDY_COMPLETE',panels=summaries,scope='CONTROLLED_DEVELOPMENT_NOT_CONFIRMATION',
        rgb_disabled_parity=True,legacy_parity=True,independent_tof_preserved=True))
    for name,p in summaries.items():
        print(name,'baseline',p['baseline'],'FP origins',p['baseline_fp_origins'])
        for a,v in p['arms'].items():print(a,{k:v[k] for k in ('metrics','correct_associations','wrong_associations','old_TP_lost','removed_FP','seconds_per_frame')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old-capture',type=Path,required=True);p.add_argument('--new-capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.old_capture,a.new_capture,a.output)
