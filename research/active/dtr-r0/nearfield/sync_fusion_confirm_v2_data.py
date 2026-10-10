"""Confirm v2 fresh admission plus frozen native/FARO CNH; no method metrics.
New labels are used only for prespecified admission; eval reference is sealed.
"""
from pathlib import Path
import argparse, csv, io, json, time, zipfile, traceback
import numpy as np
from PIL import Image
from rgb_near_readout_scale_acquire import Allocation, RangeZip
from rgb_body_query_fixed_grid import queries
from rgb_body_query_3rscan import sensor_labels, optical_z
from rgb_body_query_input_diagnostic import write, sha
import shutil,re
from concurrent.futures import ThreadPoolExecutor
import sync_rgb_tof_dataset_v1 as V
import sync_rgb_tof_dataset_v1_1 as G
from sync_rgb_tof_pilot_dev import reproject_depth,write_csv

ASSETS=('lowres_wide','lowres_depth','lowres_wide_intrinsics','highres_depth','wide_intrinsics')
def asset_heads(alloc,dest,urls):
    """Independent body-free HEADs; merge receipts into the sole byte ledger."""
    def one(item):
        asset,url=item;folder=dest/'head_receipts'/asset;folder.mkdir(parents=True,exist_ok=True)
        local=Allocation(folder,max(.01,alloc.remaining()),0,alloc.limit);local.session.trust_env=False
        try:return asset,dict(url=url,full_asset_bytes=local.request(url)),local.records,None
        except Exception as exc:return asset,None,local.records,exc
        finally:local.session.close()
    details={};errors=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for asset,detail,records,error in pool.map(one,urls):
            alloc.records.extend(records)
            if detail:details[asset]=detail
            if error:errors.append(error)
    write(alloc.root/'network_progress.json',dict(received_bytes=alloc.received,outstanding_reserved_bytes=alloc.reserved,requests=alloc.records))
    if errors:raise errors[0]
    return details
def windows(stamps):
    ts=np.array(stamps); origin=ts[0]; grid=origin+np.arange(int((ts[-1]-origin)/.2)+1)*.2
    j=np.searchsorted(ts,grid); j=np.clip(j,0,len(ts)-1); left=np.maximum(j-1,0)
    j=np.where(abs(ts[left]-grid)<=abs(ts[j]-grid),left,j)
    good=abs(ts[j]-grid)<=.10000001; out=[]; start=0; used=set()
    while start+16<=len(grid) and len(out)<2:
        ids=j[start:start+16]
        if good[start:start+16].all() and len(set(ids))==16 and not (set(ids)&used):
            out.append([(float(g),int(i)) for g,i in zip(grid[start:start+16],ids)]);used.update(ids);start+=16
        else:start+=1
    return out

def acquire(root,wall,plan_hash,resume=False):
    start=time.perf_counter();root=root.resolve();plan=root/'PLAN.json'
    assert sha(plan)==plan_hash
    assert (root/'protocol_commit.json').exists(),'Protocol must be committed before acquisition'
    recover=False
    if (root/'acquisition_terminal.json').exists():
        if not resume:raise FileExistsError('No implicit rerun')
        previous=json.loads((root/'acquisition_terminal.json').read_text());assert previous['status']!='COMPLETE18'
        n=len(list(root.glob('acquisition_stage_*_terminal.json')));shutil.copy2(root/'acquisition_terminal.json',root/f'acquisition_stage_{n:02d}_terminal.json')
    inv=json.loads((root/'consumed_inventory.json').read_text());metadata=list(csv.DictReader((root/'official/metadata.csv').open()))
    initial=json.loads((root/'network_progress.json').read_text())['received_bytes'] if resume else 0
    alloc=Allocation(root,wall-3,initial,4_000_000_000);alloc.session.trust_env=False
    if resume:alloc.records=json.loads((root/'network_progress.json').read_text())['requests']
    fixed=queries();done=set(inv['visit_ids']);records=[];entries=[];public=[];refs=[];resume_after=-1
    if resume:
        records=json.loads((root/'candidate_gate_records.json').read_text());completed=[x for x in records if x['status']!='STARTING']
        if previous['status']=='BUDGET_STOP' and records and records[-1]['status']=='SKIP_SOURCE_FAILURE':completed=[x for x in completed if x['metadata_index']!=records[-1]['metadata_index']]
        resume_after=max((x['metadata_index'] for x in completed),default=-1)
        if recover:resume_after=-1
        entries=json.loads((root/'dataset_manifest.json').read_text())['entries'];public=[f for e in entries for f in e['frames']]
        for rec in records:
            if rec['status'] in ('ACCEPTED','SKIP_REFERENCE_GATE') or ((root/'source'/rec['capture']/'selection.json').exists() and rec['metadata_index']<=resume_after):done.add(rec['visit_id'])
        for e in entries:
            states={x['source_id']:x['queries'] for x in json.loads((root/'source'/e['capture']/'gate_states.json').read_text())}
            for f in e['frames']:
                ref=Path(f['rgb_path']).with_name(f['source_id']+'_reference.npz');refs.append(dict(**f,reference_path=str(ref),reference_sha256=sha(ref),queries=states[f['source_id']]))
    base='https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw'
    status='PARTIAL';error=None
    try:
        for index,r in enumerate(metadata):
            alloc.remaining();cap=r['video_id'];visit=r['visit_id']
            if index<=resume_after or visit in done or visit=='NA' or r['is_in_upsampling']!='True':continue
            rec=dict(capture=cap,visit_id=visit,metadata_index=index,official_split=r['fold'],status='STARTING')
            prior=[x for x in records if x['metadata_index']==index]
            if prior:rec['previous_attempts']=prior;records[:]=[x for x in records if x['metadata_index']!=index]
            records.append(rec);records.sort(key=lambda x:x['metadata_index'])
            dest=root/'source'/cap;dest.mkdir(parents=True,exist_ok=True);arcs={};streams={};indices={};details={}
            try:
                # HEAD all assets before any payload transfer for this candidate.
                urls=[(asset,f"{base}/{r['fold']}/{cap}/{asset}"+('' if asset.endswith('.traj') else '.zip')) for asset in (*ASSETS,'lowres_wide.traj')]
                details=asset_heads(alloc,dest,urls)
                write(dest/'asset_HEAD.json',details)
                for asset in ASSETS:
                    d=details[asset];stream=RangeZip(d['url'],d['full_asset_bytes'],alloc);arc=zipfile.ZipFile(stream);streams[asset]=stream;arcs[asset]=arc
                    suffix='.pincam' if 'intrinsics' in asset else '.png';indices[asset]={Path(x.filename).stem:x.filename for x in arc.infolist() if x.filename.endswith(suffix)}
                    write(dest/(asset+'_directory.json'),[dict(name=x.filename,crc=x.CRC,compressed_size=x.compress_size,header_offset=x.header_offset) for x in arc.infolist()])
                common=sorted(set(indices[ASSETS[0]])&set(indices[ASSETS[1]])&set(indices[ASSETS[2]]),key=lambda s:float(s.rsplit('_',1)[1]))
                times=[float(x.rsplit('_',1)[1]) for x in common];ww=windows(times) if times else []
                if len(ww)<2:
                    rec.update(status='SKIP_TIME_INCOMPLETE',common_native_frames=len(common));continue
                done.add(visit) # First time-complete capture consumes visit even if label gate fails.
                selected=[dict(window_id=w,window_frame=f,source_id=common[i],grid_timestamp_s=g,rgb_timestamp_s=times[i]) for w,win in enumerate(ww) for f,(g,i) in enumerate(win)]
                write(dest/'selection.json',dict(rows=selected,native_common_timestamps_s=times,selected_before_labels=True,FARO_not_selection_input=True,plan_sha256=sha(plan)))
                payload={}
                def member(asset,name):
                    key=(asset,name)
                    if key not in payload:
                        info=arcs[asset].getinfo(indices[asset][name]); details[asset].setdefault('planned_members',[]).append(dict(name=info.filename,compressed_bytes=info.compress_size,planned_exact_member_bytes=30+len(info.filename.encode())+len(info.extra)+info.compress_size))
                        write(dest/'planned_ranges.json',details);payload[key]=arcs[asset].read(info)
                    return payload[key]
                temporary=[];near=free=0
                for n,s in enumerate(selected):
                    alloc.remaining();name=s['source_id'];db=member('lowres_depth',name);kb=member('lowres_wide_intrinsics',name);raw=np.array(Image.open(io.BytesIO(db)))
                    w,h,fx,fy,cx,cy=np.fromstring(kb.decode(),sep=' ');assert raw.shape==(192,256) and (w,h)==(256,192)
                    k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]],float);depth=optical_z(raw,1000.);observed=np.ones(depth.shape,bool)
                    labels,states=zip(*(sensor_labels(depth,k,q,observed) for q in fixed))
                    near+=sum(st['state']=='POSITIVE' for q,st in zip(fixed,states) if q['low'][2]==.3)
                    free+=sum(st['state']=='FREE_ON_SAMPLED_RAYS' for q,st in zip(fixed,states) if q['low'][2]>=.8)
                    temporary.append((s,db,k,depth,np.stack(labels),states))
                rec.update(near_POS=near,midfar_strict_FREE=free,frames=32,selection_sha256=sha(dest/'selection.json'))
                np.savez_compressed(dest/'gate_inputs.npz',depth=np.stack([x[3] for x in temporary]),K=np.stack([x[2] for x in temporary]))
                write(dest/'gate_states.json',[dict(source_id=x[0]['source_id'],queries=list(x[5])) for x in temporary])
                if near<16 or free<32:
                    rec.update(status='SKIP_REFERENCE_GATE',reason='nearPOS<16 or midfarstrictFREE<32; no alternative window/capture');continue
                ordinal=len(entries);role='cal' if ordinal<6 else 'eval';sensor=root/'sensor'/role/cap;sensor.mkdir(parents=True,exist_ok=True)
                td=details['lowres_wide.traj'];traj=dest/'lowres_wide.traj';traj.write_bytes(alloc.request(td['url'],(0,td['full_asset_bytes']-1,td['full_asset_bytes'])))
                hi_names=sorted(set(indices['highres_depth'])&set(indices['wide_intrinsics']),key=lambda x:float(x.rsplit('_',1)[1]));hi_ts=np.array([float(x.rsplit('_',1)[1]) for x in hi_names]);frames=[]
                for n,(s,db,k,depth,labels,states) in enumerate(temporary):
                    name=s['source_id'];rgb=sensor/(name+'.png');rgb.write_bytes(member('lowres_wide',name));native=sensor/(name+'_native.png');native.write_bytes(db)
                    reference=sensor/(name+'_reference.npz');my,mx=np.indices(depth.shape,dtype=np.float32);np.savez_compressed(reference,depth=depth,labels=labels,depth_K=k,color_K=k,map_x=mx,map_y=my,observed=np.ones(depth.shape,bool))
                    hi_path=hi_k=hi_time=None
                    if len(hi_ts):
                        hi=int(np.argmin(abs(hi_ts-s['grid_timestamp_s'])))
                        if abs(hi_ts[hi]-s['grid_timestamp_s'])<=.10000001:
                            hn=hi_names[hi];hp=sensor/(hn+'_faro.png');hp.write_bytes(member('highres_depth',hn));hw,hh,hfx,hfy,hcx,hcy=np.fromstring(member('wide_intrinsics',hn).decode(),sep=' ');hi_path=str(hp);hi_k=[[hfx,0,hcx],[0,hfy,hcy],[0,0,1]];hi_time=float(hi_ts[hi])
                    fr=dict(**s,frame=n,cohort='syncconfirmv2_arkit_'+cap,capture=cap,visit_id=visit,role=role,split=role,environment='arkitscenes_'+cap,scan='arkitscenes_'+cap,timestamp_s=s['grid_timestamp_s'],rgb_path=str(rgb),rgb_sha256=sha(rgb),native_depth_path=str(native),native_depth_sha256=sha(native),color_K=k.tolist(),depth_K=k.tolist(),native_K=k.tolist(),color_shape=[192,256],depth_shape=[192,256],native_shape=[192,256],highres_depth_path=hi_path,highres_K=hi_k,highres_timestamp_s=hi_time,trajectory_path=str(traj),grid_K=k.tolist(),grid_shape=[192,256])
                    frames.append(fr);public.append(fr);refs.append(dict(**fr,reference_path=str(reference),reference_sha256=sha(reference),queries=list(states)))
                entry=dict(capture=cap,visit_id=visit,role=role,accepted_ordinal=ordinal,frames=frames,near_POS=near,midfar_strict_FREE=free,trajectory_path=str(traj));entries.append(entry);rec.update(status='ACCEPTED',role=role)
                write(root/'dataset_manifest.json',dict(entries=entries,rows=public,queries=fixed,plan_sha256=sha(plan)))
            except TimeoutError:raise
            except Exception as exc:rec.update(status='SKIP_SOURCE_FAILURE',error=repr(exc),traceback=traceback.format_exc())
            finally:
                for a in arcs.values():a.close()
                for s in streams.values():s.close()
                write(dest/'candidate_terminal.json',rec);write(root/'candidate_gate_records.json',records);print(json.dumps(rec),flush=True)
                if len(entries)==18:status='COMPLETE18';break
    except Exception as exc:error=repr(exc);status='BUDGET_STOP' if isinstance(exc,TimeoutError) else 'FAILED_PARTIAL'
    finally:
        entries.sort(key=lambda e:next(r['metadata_index'] for r in records if r['capture']==e['capture']))
        for ordinal,e in enumerate(entries):
            role='cal' if ordinal<6 else 'eval';e['role']=role;e['accepted_ordinal']=ordinal
            for f in e['frames']:f['role']=role;f['split']=role
            for rec in records:
                if rec['capture']==e['capture']:rec['role']=role
            for f in refs:
                if f['capture']==e['capture']:f['role']=role;f['split']=role
        public=[f for e in entries for f in e['frames']]
        allowed={'environment','scan','split','frame','source_id','capture','rgb_path','rgb_sha256','color_K','color_shape','depth_K','depth_shape','role','cohort','visit_id','window_id','window_frame','timestamp_s','grid_timestamp_s','rgb_timestamp_s','frame_id'}
        alloc.session.close();write(root/'public_roster.json',dict(rows=[{k:v for k,v in row.items() if k in allowed} for row in public],queries=fixed));write(root/'reference_roster.json',dict(rows=refs,queries=fixed))
        write(root/'dataset_manifest.json',dict(status=status,entries=entries,rows=public,queries=fixed,plan_sha256=sha(plan)))
        write(root/'train_cal_reference_roster.json',dict(rows=[f for f in refs if f['role']!='eval'],queries=fixed))
        write(root/'candidate_gate_records.json',records)
        write(root/'admission_seal.json',dict(public_roster_sha256=sha(root/'public_roster.json'),accepted_visit_ids=[e['visit_id'] for e in entries],plan_sha256=sha(plan)))
        with (root/'visit_split.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['capture','visit_id','role','accepted_ordinal']);writer.writeheader();writer.writerows({k:e[k] for k in writer.fieldnames} for e in entries)
        write(root/'acquisition_terminal.json',dict(status=status,error=error,command_wall_s=time.perf_counter()-start,download_bytes=alloc.received,accepted_visits=len(entries),records=records,resources_released=True))

def prepare(root):
    """History metadata only: no new visit observation or evaluator labels."""
    start=time.perf_counter();root=root.resolve();root.mkdir(parents=True,exist_ok=True)
    old=root.parent/'sync-rgb-tof-dataset-v1-dev-20261011'
    old11=root.parent/'sync-rgb-tof-dataset-v1-1-dev-20261011'
    inv=json.loads((old/'consumed_inventory.json').read_text());caps=set(inv['capture_ids']);visits=set(inv['visit_ids']);evidence=[]
    metadata_path=old/'official/metadata.csv';metadata=list(csv.DictReader(metadata_path.open()))
    known={r['video_id']:r for r in metadata};pattern=re.compile(r'(?<!\d)(\d{8})(?!\d)')
    repo=Path(__file__).resolve().parents[4]
    paths=list((repo/'research').rglob('*.md'))+list((repo/'docs').rglob('*.md'))
    # Existing development registries, without inspecting sensor or evaluator payloads.
    for folder in root.parent.iterdir():
        if not folder.is_dir() or folder==root:continue
        for name in ('consumed_inventory.json','acquisition_terminal.json','candidate_gate_records.json','source_manifest.json'):
            p=folder/name
            if p.is_file() and p.stat().st_size<30_000_000 and ('rgb' in folder.name.lower() or 'arkit' in folder.name.lower() or 'sync' in folder.name.lower()):paths.append(p)
    for p in paths:
        try:text=p.read_text(encoding='utf-8-sig')
        except (UnicodeError,OSError):continue
        if 'arkit' not in text.lower() and p.suffix=='.md':continue
        found=set(pattern.findall(text))&known.keys()
        if found:caps.update(found);evidence.append(dict(path=str(p),sha256=sha(p),capture_ids=sorted(found)))
    for cap in caps:
        if cap in known and known[cap]['visit_id']!='NA':visits.add(known[cap]['visit_id'])
    root.joinpath('official').mkdir(exist_ok=True)
    shutil.copy2(metadata_path,root/'official/metadata.csv')
    if (old/'official/LICENSE').exists():shutil.copy2(old/'official/LICENSE',root/'official/LICENSE')
    write(root/'consumed_inventory.json',dict(status='CONSERVATIVE_PRE_DATA_GLOBAL_REGISTRY_EXCLUSION',capture_ids=sorted(caps),visit_ids=sorted(visits),evidence=evidence,base_inventory_path=str(old/'consumed_inventory.json'),base_inventory_sha256=sha(old/'consumed_inventory.json'),metadata_sha256=sha(metadata_path),protected_payloads_read=False,method='Existing ARKit markdown and RGB/ARKit/sync work registries; mentions conservatively consume entire official visit; includes gate-failed v1 visits',command_wall_s=time.perf_counter()-start))
    shutil.copy2(old11/'frozen_readout_seal.json',root/'frozen_readout_seal.json')
    shutil.copy2(old11/'residual_pool.npz',root/'residual_pool.npz')
    write(root/'frozen_residual_receipt.json',dict(path=str((root/'residual_pool.npz').resolve()),sha256=sha(root/'residual_pool.npz'),original_path=str(old11/'residual_pool.npz'),original_fit_path=str(old11/'residual_fit.json'),original_fit_sha256=sha(old11/'residual_fit.json'),new_data_used=False))
    print(json.dumps(dict(captures=len(caps),visits=len(visits),inventory_sha256=sha(root/'consumed_inventory.json'),command_wall_s=time.perf_counter()-start)))

def small_transform(alloc,root,cap,visit,fold):
    """Official small coordinate asset, fixed URL; never estimate from native."""
    url=f'https://cvg-data.inf.ethz.ch/scenefun3d/v1/{"train" if fold=="Training" else "test"}/{visit}/{cap}/{cap}_refined_transform.npy'
    rec=dict(capture=cap,visit_id=visit,url=url,method='HEAD_THEN_GET',received_bytes=0,status='STARTING')
    try:
        size=alloc.request(url)
        if size!=256:raise ValueError('Expected official 256B transform')
        if alloc.received+4096>alloc.limit:raise TimeoutError('Download cap before bounded transform')
        response=alloc.session.get(url,headers={'Accept-Encoding':'identity'},timeout=(min(8,alloc.remaining()),min(20,alloc.remaining())),stream=True)
        try:
            data=bytearray()
            for chunk in response.iter_content(chunk_size=4096):
                alloc.remaining();alloc.received+=len(chunk);data.extend(chunk);rec['received_bytes']+=len(chunk)
                if len(data)>size:raise ValueError('Transform response exceeded exact HEAD length')
            data=bytes(data);rec.update(http_status=response.status_code,received_bytes=len(data))
            if response.status_code!=200 or len(data)!=size:raise ValueError('Transform HTTP200 exact length required')
        finally:response.close()
        matrix=np.load(io.BytesIO(data),allow_pickle=False);assert matrix.shape==(4,4) and np.isfinite(matrix).all()
        np.testing.assert_allclose(matrix[3],[0,0,0,1],atol=1e-6);np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-4);assert abs(np.linalg.det(matrix[:3,:3])-1)<1e-4
        target=root/'transforms'/f'{cap}_refined_transform.npy';target.parent.mkdir(exist_ok=True);target.write_bytes(data);rec.update(status='VALID',path=str(target.resolve()),sha256=sha(target));return matrix,rec
    except TimeoutError:raise
    except Exception as exc:rec.update(status='NOT_AVAILABLE',error=repr(exc));return None,rec
    finally:
        alloc.records.append(rec);write(root/'network_progress.json',dict(received_bytes=alloc.received,outstanding_reserved_bytes=alloc.reserved,requests=alloc.records))

def faro_sources(root,wall,plan_hash):
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
                if alloc.received+needed>alloc.limit:rec['status']='NOT_DOWNLOADED_BUDGET';records.append(rec);break
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
            alloc.remaining();r=bycap[e['capture']];T,receipt=small_transform(alloc,root,e['capture'],e['visit_id'],r['fold']);transforms.append(receipt);write(root/'transform_receipts.json',dict(records=transforms))
            if T is None:continue
            siblings=[row for row in meta if row['visit_id']==e['visit_id'] and row['video_id']!=e['capture'] and row['is_in_upsampling']=='True']
            for sibling in siblings:
                alloc.remaining();cap=sibling['video_id'];U,receipt=small_transform(alloc,root,cap,e['visit_id'],sibling['fold']);transforms.append(receipt);write(root/'transform_receipts.json',dict(records=transforms))
                if U is None:continue
                dest=root/'source'/cap;dest.mkdir(exist_ok=True);base=f"https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/{sibling['fold']}/{cap}"
                details=asset_heads(alloc,dest,[(a,f'{base}/{a}'+('' if a.endswith('.traj') else '.zip')) for a in ('highres_depth','wide_intrinsics','lowres_wide.traj')]);write(dest/'asset_HEAD.json',details)
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

def supplement_sources(root,wall,plan_hash):
    """Resource-only extension, retry the recorded oversized prefix components."""
    start=time.perf_counter();assert sha(root/'PLAN.json')==plan_hash;amend=root/'resource_amendment_receipt.json';assert amend.exists()
    original=json.loads((root/'faro_source_manifest.json').read_text());terminal=json.loads((root/'faro_source_terminal.json').read_text());prior=json.loads((root/'network_progress.json').read_text());records=[];error=None
    for name in ('faro_source_manifest.json','faro_source_terminal.json','faro_source_manifest_final.json'):
        src=root/name;out=root/(src.stem+'_pre_extension.json')
        if src.exists() and not out.exists():shutil.copy2(src,out)
    allocation=Allocation(root,wall-3,prior['received_bytes'],2**60);allocation.records=prior['requests'];allocation.session.trust_env=False
    meta=list(csv.DictReader((root/'official/metadata.csv').open()));lookup={r['video_id']:r for r in meta};targets={e['capture']:e for e in original['entries']}
    candidates=[r for r in terminal['records'] if r.get('status')=='NOT_DOWNLOADED_BUDGET']
    before={e['capture']:[f['source_id'] for f in e['frames']] for e in original['entries']}
    try:
        for candidate in candidates:
            allocation.remaining();cap=candidate['capture'];target_cap=candidate['target_capture'];row=lookup[cap];target=targets[target_cap];T=np.load(root/'transforms'/f'{target_cap}_refined_transform.npy');U=np.load(root/'transforms'/f'{cap}_refined_transform.npy');transform=T@np.linalg.inv(U)
            dest=root/'source'/cap;urls=[(a,f"https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/{row['fold']}/{cap}/{a}"+('' if a.endswith('.traj') else '.zip')) for a in ('highres_depth','wide_intrinsics','lowres_wide.traj')];details=asset_heads(allocation,dest,urls);write(dest/'asset_HEAD_extension.json',details);arcs={};rec=dict(capture=cap,target_capture=target_cap,visit_id=row['visit_id'],role=target['role'],resource_amendment_sha256=sha(amend),status='STARTING')
            try:
                td=details['lowres_wide.traj'];traj=dest/'lowres_wide.traj';traj.write_bytes(allocation.request(td['url'],(0,td['full_asset_bytes']-1,td['full_asset_bytes'])))
                for asset in ('highres_depth','wide_intrinsics'):
                    d=details[asset];path=dest/(asset+'_full.zip');path.write_bytes(allocation.request(d['url'],(0,d['full_asset_bytes']-1,d['full_asset_bytes'])));arcs[asset]=zipfile.ZipFile(path);rec.setdefault('archives',[]).append(dict(asset=asset,path=str(path.resolve()),sha256=sha(path),bytes=d['full_asset_bytes']))
                depths={Path(x.filename).stem:x.filename for x in arcs['highres_depth'].infolist() if x.filename.endswith('.png')};cams={Path(x.filename).stem:x.filename for x in arcs['wide_intrinsics'].infolist() if x.filename.endswith('.pincam')};out=dest/'faro_all';out.mkdir(exist_ok=True);frames=[]
                for sid in sorted(set(depths)&set(cams),key=lambda x:float(x.rsplit('_',1)[1])):
                    allocation.remaining();path=out/(sid+'.png');path.write_bytes(arcs['highres_depth'].read(depths[sid]));w,h,fx,fy,cx,cy=np.fromstring(arcs['wide_intrinsics'].read(cams[sid]).decode(),sep=' ');frames.append(dict(source_id=sid,timestamp_s=float(sid.rsplit('_',1)[1]),depth_path=str(path.resolve()),depth_sha256=sha(path),K=[[fx,0,cx],[0,fy,cy],[0,0,1]],shape=[int(h),int(w)],trajectory_path=str(traj.resolve()),trajectory_sha256=sha(traj),target_from_source_world=transform.tolist()))
                old_ids={f['source_id'] for f in target['frames']};assert not old_ids&{f['source_id'] for f in frames};target['frames'].extend(frames);rec.update(status='COMPLETE_REGISTERED_SIBLING_EXTENSION',frames=len(frames),target_from_source_world=transform.tolist())
            finally:
                for arc in arcs.values():arc.close()
                records.append(rec);write(root/'faro_source_extension_progress.json',dict(records=records));print(json.dumps(rec),flush=True)
        for e in original['entries']:
            if e['capture']!='42446467':assert [f['source_id'] for f in e['frames']]==before[e['capture']]
        original['resource_extension_records']=records;original['resource_amendment_sha256']=sha(amend);write(root/'faro_source_manifest.json',original)
    except Exception as exc:error=repr(exc)
    finally:
        allocation.session.close();write(root/'faro_source_extension_terminal.json',dict(status='COMPLETE' if error is None else 'PARTIAL',error=error,command_wall_s=time.perf_counter()-start,download_bytes=allocation.received,original_download_cap_bytes=4_000_000_000,resource_amendment_sha256=sha(amend),records=records,prior_17_source_members_unchanged=True,resources_released=True))

def synthesize(root,wall,plan_hash,with_faro=False,resume=False):
    start=time.perf_counter();deadline=start+wall;assert sha(root/'PLAN.json')==plan_hash
    manifest=json.loads((root/'dataset_manifest.json').read_text());pool=dict(np.load(root/'residual_pool.npz'));outputs=[];checks=[];surfaces=[]
    prior={f['frame_id']:f for f in json.loads((root/'synthesis_manifest.json').read_text())['frames']} if with_faro else {}
    sources={e['capture']:e for e in json.loads((root/'faro_source_manifest.json').read_text())['entries']} if with_faro else {}
    terminal='COMPLETE';error=None
    try:
        for e in manifest['entries']:
            points=np.empty((0,3));counts=np.empty(0,int)
            if with_faro and e['capture'] in sources:
                path=root/'surfaces'/(e['capture']+'.npz');expected_ids={f['source_id'] for f in sources[e['capture']]['frames']};cached=json.loads(path.with_suffix('.json').read_text()) if path.exists() and path.with_suffix('.json').exists() else None
                if resume and cached and {f['source_id'] for f in cached['frames']}==expected_ids:assert sha(path)==cached['sha256'];sr=cached
                else:
                    if resume and path.exists():
                        archive=root/'geometry_lineage'/'pre_extension';archive.mkdir(parents=True,exist_ok=True);shutil.copy2(path,archive/path.name);shutil.copy2(path.with_suffix('.json'),archive/path.with_suffix('.json').name)
                    sr=G.fuse(sources[e['capture']],path,deadline)
                surfaces.append(sr)
                with np.load(path) as data:points=data['points'];counts=data['source_frame_count']
                write(root/'surface_manifest.json',dict(entries=surfaces))
            traj=np.loadtxt(e['trajectory_path'])
            for ordinal,fr in enumerate(e['frames']):
                if time.perf_counter()>deadline:raise TimeoutError('Synthesis allocation reached')
                K=np.asarray(fr['depth_K']);shape=tuple(fr['depth_shape']);fid=f"{e['capture']}_{fr['window_id']}_{ordinal:03d}"
                cached_frame=prior.get(fid)
                if resume and with_faro and cached and {f['source_id'] for f in cached['frames']}==expected_ids and cached_frame and len([a for a in cached_frame['arms'] if a['arm']=='faro_rho030_ambient1'])==2:
                    for arm in cached_frame['arms']:assert sha(arm['path'])==arm['sha256']
                    outputs.append(cached_frame);continue
                try:gp,gb=V.pose_at(traj,fr['grid_timestamp_s']);rp,rb=V.pose_at(traj,fr['rgb_timestamp_s']);valid_pose=True
                except ValueError:gp=rp=np.full((4,4),np.nan);gb=rb=dict(unavailable=True);valid_pose=False
                target=root/'cnh'/e['role']/fid;target.mkdir(parents=True,exist_ok=True)
                frame=dict(capture=e['capture'],visit_id=e['visit_id'],role=e['role'],window_id=fr['window_id'],frame_id=fid,source_id=fr['source_id'],rgb_path=fr['rgb_path'],grid_timestamp_s=fr['grid_timestamp_s'],rgb_timestamp_s=fr['rgb_timestamp_s'],grid_pose_bracket=gb,rgb_pose_bracket=rb,source_available=False,common_available=False,arms=[])
                if with_faro:
                    frame=dict(prior[fid]);frame['arms']=[a for a in prior[fid]['arms'] if a['arm']=='native_perturbed'];geometry,support=G.render(points,counts,gp,K,shape) if valid_pose and len(points) else (np.full(shape,np.inf),np.zeros(shape,int))
                    if resume and prior[fid].get('geometry_path'):
                        archive=root/'geometry_lineage'/'pre_extension'/fid;archive.mkdir(parents=True,exist_ok=True)
                        for oldpath in [prior[fid]['geometry_path']]+[a['path'] for a in prior[fid]['arms'] if a['arm']=='faro_rho030_ambient1']:shutil.copy2(oldpath,archive/Path(oldpath).name)
                    p=target/'input_geometry.npz';np.savez_compressed(p,depth=geometry,source_frame_count=support,grid_pose=gp,rgb_pose=rp,K=K);frame.update(geometry_path=str(p.resolve()),source_available=bool(np.isfinite(geometry).any()))
                    arm='faro_rho030_ambient1';specs=[(arm,k,geometry) for k in range(2)]
                else:
                    native=np.asarray(Image.open(fr['native_depth_path']),float)/1000;specs=[]
                    for k in range(2):
                        parts=[20261011,e['accepted_ordinal'],fr['window_id'],fr['window_frame'],k];pseed=int(np.random.SeedSequence([20261011,1101,*parts[1:]]).generate_state(1)[0]);nz=G.perturb(native,pool,pseed);geometry=reproject_depth(nz,K,rp,gp,K,shape) if valid_pose else np.full(shape,np.inf);specs.append(('native_perturbed',k,geometry))
                for arm,k,geometry in specs:
                    ex,amb,cov=V.expectation(geometry,K,.3,1);seed=int(np.random.SeedSequence([20261011,e['accepted_ordinal'],fr['window_id'],fr['window_frame'],k]).generate_state(1)[0]);hist,n,bg=V.sample(ex,amb,seed)
                    path=target/f'{arm}_k{k}.npz';np.savez_compressed(path,hist=hist[0],ambient=amb[0],counts=n[0],background=bg[0],expectation=ex[0],coverage=cov,valid=bool(np.any(cov>0)),grid_K=K,grid_pose=gp,rgb_pose=rp,rgb_K=K,seed=seed)
                    frame['arms'].append(dict(arm=arm,repeat=k,path=str(path.resolve()),sha256=sha(path),coverage_mean=float(cov.mean()),valid=bool(np.any(cov>0))))
                    peak=hist[0].argmax(-1);height=np.take_along_axis(hist[0],peak[...,None],-1)[...,0];back=np.take_along_axis(bg[0],peak[...,None],-1)[...,0];snr=height/np.sqrt(np.maximum(height,0)+2*back+1)
                    checks.append(dict(frame_id=fid,role=e['role'],arm=arm,K=k,joint_pass_zones=int(((cov>=.75)&(snr>=3)).sum()),geometry_pass_zones=int((cov>=.75).sum()),valid_grid_frame=valid_pose))
                frame['faro_source_available']=frame['source_available'];outputs.append(frame)
                write(root/'synthesis_partial_manifest.json',dict(status='PARTIAL',frames=outputs,eval_query_metrics_computed=False))
            print('synthesis',e['capture'],len(outputs),time.perf_counter()-start,flush=True)
    except Exception as exc:terminal='BUDGET_STOP' if isinstance(exc,TimeoutError) else 'FAILED';error=repr(exc)
    finally:
        if with_faro:
            # Retain all primary grid frames even if optional FARO hits its cap.
            updated={f['frame_id']:f for f in outputs};outputs=[updated.get(f['frame_id'],f) for f in prior.values()]
            oldchecks=list(csv.DictReader((root/'input_coverage_gate.csv').open()));newkeys={(r['frame_id'],r['arm'],str(r['K'])) for r in checks};checks=[r for r in oldchecks if (r['frame_id'],r['arm'],str(r['K'])) not in newkeys]+checks
        write(root/'synthesis_manifest.json',dict(status=terminal,frames=outputs,manifest_sha256=sha(root/'dataset_manifest.json'),plan_sha256=plan_hash,eval_query_metrics_computed=False))
        write_csv(root/'input_coverage_gate.csv',checks);write(root/('faro_synthesis_continuation_terminal.json' if resume else 'faro_synthesis_terminal.json' if with_faro else 'native_synthesis_terminal.json'),dict(status=terminal,error=error,frames=len(outputs),command_wall_s=time.perf_counter()-start,GPU_s=0))

def seal_inputs(root):
    """Public observation provenance only; references are linked by roster hash."""
    start=time.perf_counter();manifest=json.loads((root/'dataset_manifest.json').read_text());synth=json.loads((root/'synthesis_manifest.json').read_text());sframes={f['source_id']:f for f in synth['frames']};surfaces={}
    if (root/'surface_manifest.json').exists():surfaces={e['capture']:e for e in json.loads((root/'surface_manifest.json').read_text())['entries']}
    for e in manifest['entries']:
        surface=surfaces.get(e['capture']);e['fused_surface']=surface and {k:surface[k] for k in ('path','sha256','voxels','frames_total','frames_used')}
        for fr in e['frames']:
            reference=Path(fr['rgb_path']).with_name(fr['source_id']+'_reference.npz');fr['reference']=dict(path=str(reference.resolve()),sha256=sha(reference))
            fr['trajectory_sha256']=sha(fr['trajectory_path']);sf=sframes.get(fr['source_id']);fr['CNH']=sf['arms'] if sf else [];fr['pose_brackets']=dict(grid=sf['grid_pose_bracket'],rgb=sf['rgb_pose_bracket']) if sf else None
            if sf and sf.get('geometry_path'):fr['rendered_depth']=dict(path=sf['geometry_path'],sha256=sha(sf['geometry_path']))
    manifest.update(public_roster_sha256=sha(root/'public_roster.json'),reference_roster_sha256=sha(root/'reference_roster.json'),cal_reference_roster_sha256=sha(root/'train_cal_reference_roster.json'),synthesis_manifest_sha256=sha(root/'synthesis_manifest.json'),input_gate_sha256=sha(root/'input_coverage_gate.csv'),consumed_inventory_sha256=sha(root/'consumed_inventory.json'),frozen_residual_sha256=sha(root/'residual_pool.npz'),frozen_readout_seal_sha256=sha(root/'frozen_readout_seal.json'),query_method_metrics_computed=False)
    if (root/'faro_source_manifest.json').exists():manifest['faro_source_manifest_sha256']=sha(root/'faro_source_manifest.json')
    if (root/'surface_manifest.json').exists():manifest['surface_manifest_sha256']=sha(root/'surface_manifest.json')
    if (root/'license_receipt.json').exists():manifest['license_receipt_sha256']=sha(root/'license_receipt.json')
    if (root/'faro_source_terminal.json').exists():manifest['faro_source_terminal_sha256']=sha(root/'faro_source_terminal.json')
    if (root/'faro_source_prefix_receipt.json').exists():manifest['faro_source_prefix_receipt_sha256']=sha(root/'faro_source_prefix_receipt.json')
    for name in ('faro_source_manifest_final.json','faro_source_extension_receipt.json','source_render_binding_receipt.json','resource_amendment_receipt.json'):
        if (root/name).exists():manifest[name.removesuffix('.json')+'_sha256']=sha(root/name)
    write(root/'dataset_manifest_v2.json',manifest);write(root/'data_input_seal.json',dict(manifest_sha256=sha(root/'dataset_manifest_v2.json'),public_roster_sha256=sha(root/'public_roster.json'),synthesis_manifest_sha256=sha(root/'synthesis_manifest.json'),input_gate_sha256=sha(root/'input_coverage_gate.csv'),command_wall_s=time.perf_counter()-start,eval_method_metrics_computed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['prepare','acquire','native','faro-source','faro-supplement','faro-synthesis','seal-inputs'],required=True);p.add_argument('--wall-s',type=float,default=300);p.add_argument('--plan-sha256');p.add_argument('--resume-engineering',action='store_true');a=p.parse_args();a.root=a.root.resolve()
    if a.stage=='prepare':prepare(a.root)
    elif a.stage=='seal-inputs':seal_inputs(a.root)
    elif a.stage=='acquire':acquire(a.root,a.wall_s,a.plan_sha256,a.resume_engineering)
    elif a.stage=='faro-source':faro_sources(a.root,a.wall_s,a.plan_sha256)
    elif a.stage=='faro-supplement':supplement_sources(a.root,a.wall_s,a.plan_sha256)
    else:synthesize(a.root,a.wall_s,a.plan_sha256,a.stage=='faro-synthesis',a.resume_engineering)
