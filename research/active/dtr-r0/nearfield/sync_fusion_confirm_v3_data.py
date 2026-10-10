"""v3 fresh-visit data, inherited v2 windows/gates and strict resource ceilings.

This stage observes no predictor output or query method metric. References are
used only by the registered admission gate; cal/eval rosters stay separate.
"""
from pathlib import Path
import argparse
import json
import time
import csv
import zipfile
import shutil
import numpy as np
from rgb_near_readout_scale_acquire import Allocation
from threadpoolctl import threadpool_limits

import sync_fusion_confirm_v2_data as V2
from rgb_body_query_input_diagnostic import write, sha

CPU_LIMIT_S = 1600.0
DOWNLOAD_LIMIT_BYTES = 4_000_000_000


def protocol_check(root, plan_hash):
    receipt = json.loads((root / 'protocol_commit.json').read_text())
    assert receipt, 'Protocol must be committed before any new data preparation'
    assert sha(root / 'PLAN.json') == plan_hash


def prepare(root):
    """Use the full preexisting registry, including failed admission visits."""
    V2.prepare(root)
    registry = json.loads((root / 'consumed_inventory.json').read_text())
    old = root.parent / 'sync-fusion-confirm-v2-dev-20261011'
    manifest = json.loads((old / 'dataset_manifest.json').read_text())
    prior = root.parent / 'sync-rgb-tof-dataset-v1-dev-20261011'
    prior_manifest = json.loads((prior / 'public_roster.json').read_text())
    expected = {str(e['visit_id']) for e in manifest['entries']}
    expected.update(str(e['visit_id']) for e in prior_manifest['rows'])
    assert len(expected) == 30, 'Frozen consumed training visit count must be 30'
    assert expected <= set(registry['visit_ids'])
    gates = json.loads((old / 'candidate_gate_records.json').read_text())
    failed = {str(e['visit_id']) for e in gates if e['status'] == 'SKIP_REFERENCE_GATE'}
    assert failed <= set(registry['visit_ids'])
    write(root / 'v3_inventory_check.json', dict(
        training_visits=sorted(expected), training_visit_count=len(expected),
        v2_failed_gate_visits=sorted(failed),
        consumed_inventory_sha256=sha(root / 'consumed_inventory.json'),
        protected_payloads_read=False, new_observations_read=False))


def acquire(root, wall, plan_hash, resume=False):
    V2.acquire(root, wall, plan_hash, resume)
    # Cohort names are provenance only; membership/order/windows remain exact.
    for name in ('dataset_manifest.json', 'public_roster.json',
                 'reference_roster.json', 'train_cal_reference_roster.json'):
        obj = json.loads((root / name).read_text())
        for row in obj.get('rows', []):
            row['cohort'] = 'syncconfirmv3_arkit_' + row['capture']
        for entry in obj.get('entries', []):
            for row in entry['frames']:
                row['cohort'] = 'syncconfirmv3_arkit_' + row['capture']
        write(root / name, obj)
    receipt = json.loads((root / 'admission_seal.json').read_text())
    receipt['public_roster_sha256'] = sha(root / 'public_roster.json')
    write(root / 'admission_seal.json', receipt)


def acquire_faro_prefix(root,wall,plan_hash):
    """Post-admission full target-capture released FARO; never selection input."""
    start=time.perf_counter();assert sha(root/'PLAN.json')==plan_hash
    old=root.parent/'sync-rgb-tof-dataset-v1-1-dev-20261011';license_records=[]
    for name in ('scenefun_DATA_LICENSE.html','scenefun_LICENSE'):
        shutil.copy2(old/name,root/'official'/name);license_records.append(dict(path=str((root/'official'/name).resolve()),sha256=sha(root/'official'/name),source=str(old/name)))
    write(root/'license_receipt.json',dict(use='local noncommercial research only; raw data not redistributed',arkitscenes_license_path=str((root/'official/LICENSE').resolve()),arkitscenes_license_sha256=sha(root/'official/LICENSE'),supplementary=license_records))
    prior=json.loads((root/'network_progress.json').read_text());alloc=Allocation(root,wall-3,prior['received_bytes'],4_000_000_000);alloc.records=prior['requests'];alloc.session.trust_env=False
    manifest=json.loads((root/'dataset_manifest.json').read_text());sources=[];records=[];error=None;transforms=[]
    try:
        for e in manifest['entries']:
            alloc.remaining();dest=root/'source'/e['capture'];details=json.loads((dest/'asset_HEAD.json').read_text());frames=[];archives={}
            rec=dict(capture=e['capture'],visit_id=e['visit_id'],role=e['role'],whole_visit=False,scope='target capture only; sibling capture registration not fetched under fixed resource priority')
            try:
                needed=sum(details[a]['full_asset_bytes'] for a in ('highres_depth','wide_intrinsics'))
                if alloc.received+needed>alloc.limit:
                    rec['status']='NOT_DOWNLOADED_BUDGET';records.append(rec)
                    raise TimeoutError('First incomplete optional target exceeds download cap; stop fixed prefix')
                for a in ('highres_depth','wide_intrinsics'):
                    d=details[a];path=dest/(a+'_full.zip');data=alloc.request(d['url'],(0,d['full_asset_bytes']-1,d['full_asset_bytes']));path.write_bytes(data);archives[a]=zipfile.ZipFile(path)
                    rec.setdefault('archives',[]).append(dict(asset=a,path=str(path.resolve()),bytes=len(data),sha256=sha(path)))
                depths={Path(x.filename).stem:x.filename for x in archives['highres_depth'].infolist() if x.filename.endswith('.png')}
                cams={Path(x.filename).stem:x.filename for x in archives['wide_intrinsics'].infolist() if x.filename.endswith('.pincam')}
                output=dest/'faro_all';output.mkdir(exist_ok=True)
                for sid in sorted(set(depths)&set(cams),key=lambda x:float(x.rsplit('_',1)[1])):
                    alloc.remaining();path=output/(sid+'.png');path.write_bytes(archives['highres_depth'].read(depths[sid]));kb=archives['wide_intrinsics'].read(cams[sid]);w,h,fx,fy,cx,cy=np.fromstring(kb.decode(),sep=' ')
                    frames.append(dict(source_id=sid,timestamp_s=float(sid.rsplit('_',1)[1]),depth_path=str(path.resolve()),depth_sha256=sha(path),K=[[fx,0,cx],[0,fy,cy],[0,0,1]],shape=[int(h),int(w)],trajectory_path=e['trajectory_path']))
                source=dict(capture=e['capture'],visit_id=e['visit_id'],role=e['role'],frames=frames);sources.append(source);rec.update(status='COMPLETE_TARGET_CAPTURE',frames=len(frames))
            except TimeoutError:raise
            except Exception as exc:rec.update(status='SOURCE_FAILURE',error=repr(exc))
            finally:
                for a in archives.values():a.close()
                if rec not in records:records.append(rec)
                write(root/'faro_source_manifest.json',dict(entries=sources,records=records,plan_sha256=plan_hash));print(json.dumps(rec),flush=True)
        # Only after all target captures, append same-visit registered siblings in
        # accepted visit order and then original metadata order. No ICP/native.
        meta=list(csv.DictReader((root/'official/metadata.csv').open()));bycap={r['video_id']:r for r in meta};bytarget={s['capture']:s for s in sources}
        for e in manifest['entries']:
            if e['capture'] not in bytarget:continue
            alloc.remaining();r=bycap[e['capture']];T,receipt=V2.small_transform(alloc,root,e['capture'],e['visit_id'],r['fold']);transforms.append(receipt);write(root/'transform_receipts.json',dict(records=transforms))
            if T is None:continue
            siblings=[row for row in meta if row['visit_id']==e['visit_id'] and row['video_id']!=e['capture'] and row['is_in_upsampling']=='True']
            for sibling in siblings:
                alloc.remaining();cap=sibling['video_id'];U,receipt=V2.small_transform(alloc,root,cap,e['visit_id'],sibling['fold']);transforms.append(receipt);write(root/'transform_receipts.json',dict(records=transforms))
                if U is None:continue
                dest=root/'source'/cap;dest.mkdir(exist_ok=True);base=f"https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/{sibling['fold']}/{cap}"
                details=V2.asset_heads(alloc,dest,[(a,f'{base}/{a}'+('' if a.endswith('.traj') else '.zip')) for a in ('highres_depth','wide_intrinsics','lowres_wide.traj')]);write(dest/'asset_HEAD.json',details)
                needed=sum(d['full_asset_bytes'] for d in details.values());rec=dict(capture=cap,visit_id=e['visit_id'],role=e['role'],target_capture=e['capture'],scope='registered sibling')
                if alloc.received+needed>alloc.limit:
                    rec['status']='BYTE_BUDGET_STOP';records.append(rec);raise TimeoutError('First incomplete optional component exceeds download cap; stop accepted-order prefix')
                arcs={}
                try:
                    td=details['lowres_wide.traj'];traj=dest/'lowres_wide.traj';traj.write_bytes(alloc.request(td['url'],(0,td['full_asset_bytes']-1,td['full_asset_bytes'])))
                    for a in ('highres_depth','wide_intrinsics'):
                        d=details[a];path=dest/(a+'_full.zip');path.write_bytes(alloc.request(d['url'],(0,d['full_asset_bytes']-1,d['full_asset_bytes'])));arcs[a]=zipfile.ZipFile(path)
                    depths={Path(x.filename).stem:x.filename for x in arcs['highres_depth'].infolist() if x.filename.endswith('.png')};cams={Path(x.filename).stem:x.filename for x in arcs['wide_intrinsics'].infolist() if x.filename.endswith('.pincam')};out=dest/'faro_all';out.mkdir(exist_ok=True);added=[];target_T=T@np.linalg.inv(U)
                    for sid in sorted(set(depths)&set(cams),key=lambda x:float(x.rsplit('_',1)[1])):
                        alloc.remaining();path=out/(sid+'.png');path.write_bytes(arcs['highres_depth'].read(depths[sid]));w,h,fx,fy,cx,cy=np.fromstring(arcs['wide_intrinsics'].read(cams[sid]).decode(),sep=' ')
                        added.append(dict(source_id=sid,timestamp_s=float(sid.rsplit('_',1)[1]),depth_path=str(path.resolve()),depth_sha256=sha(path),K=[[fx,0,cx],[0,fy,cy],[0,0,1]],shape=[int(h),int(w)],trajectory_path=str(traj.resolve()),target_from_source_world=target_T.tolist()))
                    bytarget[e['capture']]['frames'].extend(added);rec.update(status='COMPLETE_REGISTERED_SIBLING',frames=len(added),target_from_source_world=target_T.tolist())
                finally:
                    for a in arcs.values():a.close()
                    records.append(rec);write(root/'faro_source_manifest.json',dict(entries=sources,records=records,plan_sha256=plan_hash));print(json.dumps(rec),flush=True)
    except Exception as exc:error=repr(exc)
    finally:
        alloc.session.close()
        for source in sources:
            source['registered_sibling_captures']=sorted({r['capture'] for r in records if r.get('target_capture')==source['capture'] and r.get('status')=='COMPLETE_REGISTERED_SIBLING'})
            source['whole_visit_geometry_guaranteed']=False
        write(root/'faro_source_manifest.json',dict(entries=sources,records=records,plan_sha256=plan_hash,scope='All acquired target FARO plus official registered sibling FARO; unregistered or budget-limited siblings excluded, never native-assisted registration'))
        write(root/'faro_source_terminal.json',dict(status='COMPLETE' if error is None else 'PARTIAL',error=error,command_wall_s=time.perf_counter()-start,download_bytes=alloc.received,sources=len(sources),records=records,resources_released=True))

def faro_sources(root, wall, plan_hash):
    # v2 already enforces the cumulative 4 GB cap for every received retry byte,
    # fixed accepted-target order followed by registered-sibling metadata order.
    acquire_faro_prefix(root, wall, plan_hash)
    source = json.loads((root / 'faro_source_manifest.json').read_text())
    terminal = json.loads((root / 'faro_source_terminal.json').read_text())
    manifest = json.loads((root / 'dataset_manifest.json').read_text())
    present = {e['capture'] for e in source['entries']}
    missing = [e['capture'] for e in manifest['entries'] if e['capture'] not in present]
    budget_stops = [r for r in source['records'] if r.get('status') in
                    ('NOT_DOWNLOADED_BUDGET', 'BYTE_BUDGET_STOP')]
    if missing or budget_stops or terminal['error']:
        terminal['status'] = 'PARTIAL'
    terminal.update(download_limit_bytes=DOWNLOAD_LIMIT_BYTES,
                    missing_target_captures=missing,
                    budget_stopped_components=budget_stops,
                    resource_extension_authorized=False)
    assert terminal['download_bytes'] <= DOWNLOAD_LIMIT_BYTES
    write(root / 'faro_source_terminal.json', terminal)
    write(root / 'faro_source_prefix_receipt.json', dict(
        accepted_target_order=[e['capture'] for e in manifest['entries']],
        missing_target_captures=missing, status=terminal['status'],
        budget_stopped_components=budget_stops,
        download_limit_bytes=DOWNLOAD_LIMIT_BYTES,
        order='All targets first; registered siblings in accepted-visit and metadata order',
        incomplete_geometry_is_missing=True, native_assisted_registration=False))


def continue_faro_prefix(root, wall, plan_hash):
    """Continue only unfinished official siblings; no completed source transfer."""
    start=time.perf_counter()
    for name in ('faro_source_manifest.json','faro_source_terminal.json',
                 'faro_source_completion_scope.json'):
        path=root/name
        if path.exists(): shutil.copy2(path,root/(path.stem+'_pre_continuation.json'))
    prior=json.loads((root/'network_progress.json').read_text())
    alloc=Allocation(root,wall-3,prior['received_bytes'],DOWNLOAD_LIMIT_BYTES)
    alloc.records=prior['requests'];alloc.session.trust_env=False
    original=json.loads((root/'faro_source_manifest.json').read_text())
    sources=original['entries'];records=original['records']
    manifest=json.loads((root/'dataset_manifest.json').read_text())
    assert {e['capture'] for e in manifest['entries']}=={s['capture'] for s in sources}
    meta=list(csv.DictReader((root/'official/metadata.csv').open()))
    bycap={r['video_id']:r for r in meta};bytarget={s['capture']:s for s in sources}
    transforms=json.loads((root/'transform_receipts.json').read_text())['records']
    done={r['capture'] for r in records if r.get('status')=='COMPLETE_REGISTERED_SIBLING'}
    prior_transform={r['capture']:r for r in transforms};newrecords=[];error=None
    def transform(cap,visit,fold):
        if cap in prior_transform:
            receipt=prior_transform[cap]
            return np.load(receipt['path']) if receipt['status']=='VALID' else None
        matrix,receipt=V2.small_transform(alloc,root,cap,visit,fold)
        transforms.append(receipt);prior_transform[cap]=receipt
        write(root/'transform_receipts.json',dict(records=transforms))
        return matrix
    try:
        for e in manifest['entries']:
            alloc.remaining();r=bycap[e['capture']]
            T=transform(e['capture'],e['visit_id'],r['fold'])
            if T is None:continue
            for sibling in meta:
                if sibling['visit_id']!=e['visit_id'] or sibling['video_id']==e['capture'] or sibling['is_in_upsampling']!='True':continue
                cap=sibling['video_id']
                if cap in done:continue
                alloc.remaining();U=transform(cap,e['visit_id'],sibling['fold'])
                if U is None:continue
                dest=root/'source'/cap;dest.mkdir(exist_ok=True)
                base=f"https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/{sibling['fold']}/{cap}"
                hp=dest/'asset_HEAD.json'
                details=json.loads(hp.read_text()) if hp.exists() else V2.asset_heads(alloc,dest,[(a,f'{base}/{a}'+('' if a.endswith('.traj') else '.zip')) for a in ('highres_depth','wide_intrinsics','lowres_wide.traj')])
                write(hp,details)
                rec=dict(capture=cap,visit_id=e['visit_id'],role=e['role'],target_capture=e['capture'],scope='registered sibling',continuation=True)
                needed=sum(d['full_asset_bytes'] for d in details.values())
                if alloc.received+needed>alloc.limit:
                    rec['status']='BYTE_BUDGET_STOP';records.append(rec);newrecords.append(rec)
                    raise TimeoutError('First incomplete optional component exceeds download cap; stop fixed prefix')
                arcs={}
                try:
                    d=details['lowres_wide.traj'];traj=dest/'lowres_wide.traj'
                    traj.write_bytes(alloc.request(d['url'],(0,d['full_asset_bytes']-1,d['full_asset_bytes'])))
                    for a in ('highres_depth','wide_intrinsics'):
                        d=details[a];path=dest/(a+'_full.zip')
                        path.write_bytes(alloc.request(d['url'],(0,d['full_asset_bytes']-1,d['full_asset_bytes'])))
                        arcs[a]=zipfile.ZipFile(path)
                    depths={Path(x.filename).stem:x.filename for x in arcs['highres_depth'].infolist() if x.filename.endswith('.png')}
                    cams={Path(x.filename).stem:x.filename for x in arcs['wide_intrinsics'].infolist() if x.filename.endswith('.pincam')}
                    out=dest/'faro_all';out.mkdir(exist_ok=True);added=[];target_T=T@np.linalg.inv(U)
                    for sid in sorted(set(depths)&set(cams),key=lambda x:float(x.rsplit('_',1)[1])):
                        alloc.remaining();path=out/(sid+'.png');path.write_bytes(arcs['highres_depth'].read(depths[sid]))
                        w,h,fx,fy,cx,cy=np.fromstring(arcs['wide_intrinsics'].read(cams[sid]).decode(),sep=' ')
                        added.append(dict(source_id=sid,timestamp_s=float(sid.rsplit('_',1)[1]),depth_path=str(path.resolve()),depth_sha256=sha(path),K=[[fx,0,cx],[0,fy,cy],[0,0,1]],shape=[int(h),int(w)],trajectory_path=str(traj.resolve()),target_from_source_world=target_T.tolist()))
                    assert not {f['source_id'] for f in bytarget[e['capture']]['frames']}&{f['source_id'] for f in added}
                    bytarget[e['capture']]['frames'].extend(added)
                    rec.update(status='COMPLETE_REGISTERED_SIBLING',frames=len(added),target_from_source_world=target_T.tolist());done.add(cap)
                finally:
                    for arc in arcs.values():arc.close()
                    records.append(rec);newrecords.append(rec)
                    write(root/'faro_source_manifest.json',dict(entries=sources,records=records,plan_sha256=plan_hash))
                    print(json.dumps(rec),flush=True)
    except Exception as exc:error=repr(exc)
    finally:
        alloc.session.close()
        for e in sources:e['registered_sibling_captures']=sorted({r['capture'] for r in records if r.get('target_capture')==e['capture'] and r.get('status')=='COMPLETE_REGISTERED_SIBLING'})
        write(root/'faro_source_manifest.json',dict(entries=sources,records=records,plan_sha256=plan_hash,scope=original.get('scope'),fixed_prefix_continuation=True))
        terminal=dict(status='COMPLETE' if error is None else 'PARTIAL',error=error,command_wall_s=time.perf_counter()-start,download_bytes=alloc.received,sources=len(sources),records=records,resources_released=True,continuation=True)
        write(root/'faro_source_terminal.json',terminal)
        write(root/'faro_source_continuation_terminal.json',dict(**terminal,new_records=newrecords,completed_source_payloads_redownloaded=False))


def seal_inputs(root):
    source_completion_scope(root)
    V2.seal_inputs(root)
    obj = json.loads((root / 'dataset_manifest_v2.json').read_text())
    obj['task'] = 'SYNC_FUSION_CONFIRM_V3_DEV_20261011'
    obj['new_visit_count'] = len(obj['entries'])
    obj['frames_per_visit'] = 32
    obj['download_limit_bytes'] = DOWNLOAD_LIMIT_BYTES
    obj['source_completion_scope_sha256'] = sha(root/'faro_source_completion_scope.json')
    obj['resource_reallocation_receipt_sha256'] = sha(root/'resource_reallocation_receipt.json')
    write(root / 'dataset_manifest_v3.json', obj)
    receipt = json.loads((root / 'data_input_seal.json').read_text())
    receipt.update(manifest_path=str((root / 'dataset_manifest_v3.json').resolve()),
                   manifest_sha256=sha(root / 'dataset_manifest_v3.json'),
                   data_code_sha256=sha(__file__),
                   inherited_v2_code_sha256=sha(V2.__file__))
    write(root / 'data_input_seal.json', receipt)


def source_completion_scope(root):
    terminal=json.loads((root/'faro_source_terminal.json').read_text())
    sources=json.loads((root/'faro_source_manifest.json').read_text())
    manifest=json.loads((root/'dataset_manifest.json').read_text())
    meta=list(csv.DictReader((root/'official/metadata.csv').open()))
    transforms={r['capture']:r for r in json.loads((root/'transform_receipts.json').read_text())['records']}
    progress=json.loads((root/'network_progress.json').read_text())
    done={r['capture'] for r in sources['records'] if r.get('status')=='COMPLETE_REGISTERED_SIBLING'}
    candidates=[]
    for e in manifest['entries']:
        for index,r in enumerate(meta):
            cap=r['video_id']
            if r['visit_id']!=e['visit_id'] or cap==e['capture'] or r['is_in_upsampling']!='True':continue
            status='COMPLETE_REGISTERED_SIBLING' if cap in done else transforms.get(cap,{}).get('status','UNINSPECTED_AFTER_PREFIX')
            if cap not in done and transforms.get(e['capture'],{}).get('status')=='NOT_AVAILABLE':status='EXCLUDED_TARGET_REGISTRATION_NOT_AVAILABLE'
            candidates.append(dict(capture=cap,target_capture=e['capture'],visit_id=e['visit_id'],role=e['role'],metadata_index=index,status=status,transform_receipt=transforms.get(cap)))
    used={f['source_id'] for e in sources['entries'] for f in e['frames']}
    unused=[dict(path=str(p.resolve()),bytes=p.stat().st_size) for p in (root/'source').glob('*/faro_all/*.png') if p.stem not in used]
    unadopted_archives=[]
    adopted={e['capture'] for e in sources['entries']}|done
    for p in (root/'source').glob('*/*_full.zip'):
        if p.parent.name not in adopted:unadopted_archives.append(dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)))
    failed=[r for r in progress['requests'] if r.get('status')=='FAILED' and r.get('received_bytes',0)>0]
    stops=[r for r in sources['records'] if r.get('status') in ('NOT_DOWNLOADED_BUDGET','BYTE_BUDGET_STOP')]
    write(root/'faro_source_completion_scope.json',dict(status=terminal['status'],error=terminal['error'],reason='Fixed optional-source prefix preserved; stage wall stop' if not stops and terminal['error'] else 'First incomplete full component exceeds remaining4GB allocation' if stops else 'Complete',target_captures=len(sources['entries']),completed_registered_siblings=sorted(done),ordered_sibling_candidates=candidates,first_uncompleted_candidate=next((c for c in candidates if c['status'] not in ('COMPLETE_REGISTERED_SIBLING','NOT_AVAILABLE','EXCLUDED_TARGET_REGISTRATION_NOT_AVAILABLE')),None),unadopted_extracted_payloads=unused,unadopted_archives=unadopted_archives,failed_received_payloads=failed,download_bytes=progress['received_bytes'],download_limit_bytes=DOWNLOAD_LIMIT_BYTES,budget_stopped_components=stops,source_manifest_sha256=sha(root/'faro_source_manifest.json'),no_registration_from_native=True))


def run(root, stage, wall, plan_hash, resume=False):
    root = root.resolve()
    protocol_check(root, plan_hash)
    ledger_path = root / 'data_resource_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(
        CPU_limit_command_wall_s=CPU_LIMIT_S, GPU_wall_s=0,
        download_limit_bytes=DOWNLOAD_LIMIT_BYTES, stages=[])
    ledger['CPU_limit_command_wall_s'] = CPU_LIMIT_S
    spent = sum(r['command_wall_s'] for r in ledger['stages'])
    allocation = min(float(wall), CPU_LIMIT_S - spent - 2)
    if allocation <= 0:
        raise TimeoutError('v3 data CPU command-wall allocation exhausted')
    start = time.perf_counter()
    record = dict(stage=stage, allocated_command_wall_s=allocation,
                  plan_sha256=plan_hash, status='STARTING')
    try:
        if stage == 'prepare': prepare(root)
        elif stage == 'acquire': acquire(root, allocation, plan_hash, resume)
        elif stage == 'native':
            with threadpool_limits(1): V2.synthesize(root, allocation, plan_hash)
        elif stage == 'faro-source': faro_sources(root, allocation, plan_hash)
        elif stage == 'faro-source-continuation': continue_faro_prefix(root, allocation, plan_hash)
        elif stage == 'faro-synthesis':
            with threadpool_limits(1): V2.synthesize(root, allocation, plan_hash, True, resume)
        elif stage == 'seal-inputs': seal_inputs(root)
        record['status'] = 'RETURNED'
    except Exception as exc:
        record.update(status='FAILED', error=repr(exc))
        raise
    finally:
        record['command_wall_s'] = time.perf_counter() - start
        progress = root / 'network_progress.json'
        if progress.exists():
            record['download_bytes'] = json.loads(progress.read_text())['received_bytes']
            assert record['download_bytes'] <= DOWNLOAD_LIMIT_BYTES
        ledger['stages'].append(record)
        ledger['CPU_command_wall_s'] = sum(r['command_wall_s'] for r in ledger['stages'])
        write(ledger_path, ledger)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['prepare', 'acquire', 'native', 'faro-source',
                                      'faro-source-continuation', 'faro-synthesis', 'seal-inputs'], required=True)
    p.add_argument('--wall-s', type=float, default=300)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--resume-engineering', action='store_true')
    a = p.parse_args()
    run(a.root, a.stage, a.wall_s, a.plan_sha256, a.resume_engineering)
