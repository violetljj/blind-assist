"""Sealed chronological two-window ARKit source construction, no method scores."""
from pathlib import Path
import argparse, csv, io, json, time, zipfile, traceback
import numpy as np
from PIL import Image
from rgb_near_readout_scale_acquire import Allocation, RangeZip
from rgb_body_query_fixed_grid import queries
from rgb_body_query_3rscan import sensor_labels, optical_z
from rgb_body_query_input_diagnostic import write, sha

ASSETS=('lowres_wide','lowres_depth','lowres_wide_intrinsics','highres_depth','wide_intrinsics')
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

def run(root,wall,resume=False,recover=False):
    start=time.perf_counter();root=root.resolve();plan=root/'PLAN.json'
    assert sha(plan)=='4f87e9875d1c990dc1a6ad1dd2e7606ca3c6476431b1f5f69dcea46f2b4e619e'
    if (root/'acquisition_terminal.json').exists():raise FileExistsError('No implicit rerun')
    inv=json.loads((root/'consumed_inventory.json').read_text());metadata=list(csv.DictReader((root/'official/metadata.csv').open()))
    initial=json.loads((root/'network_progress.json').read_text())['received_bytes'] if resume else 0
    alloc=Allocation(root,wall-3,initial,3_000_000_000);alloc.session.trust_env=False
    fixed=queries();done=set(inv['visit_ids']);records=[];entries=[];public=[];refs=[];resume_after=-1
    if resume:
        records=json.loads((root/'candidate_gate_records.json').read_text());resume_after=max(x['metadata_index'] for x in records)
        if recover:resume_after=-1
        entries=json.loads((root/'dataset_manifest.json').read_text())['entries'];public=[f for e in entries for f in e['frames']]
        for rec in records:
            if rec['status'] in ('ACCEPTED','SKIP_REFERENCE_GATE'):done.add(rec['visit_id'])
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
                for asset in (*ASSETS,'lowres_wide.traj'):
                    url=f"{base}/{r['fold']}/{cap}/{asset}"+('' if asset.endswith('.traj') else '.zip')
                    details[asset]=dict(url=url,full_asset_bytes=alloc.request(url))
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
                ordinal=len(entries);role='train' if ordinal<6 else 'cal' if ordinal<8 else 'eval';sensor=root/'sensor'/role/cap;sensor.mkdir(parents=True,exist_ok=True)
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
                    fr=dict(**s,frame=n,cohort='syncv1_arkit_'+cap,capture=cap,visit_id=visit,role=role,split=role,environment='arkitscenes_'+cap,scan='arkitscenes_'+cap,timestamp_s=s['grid_timestamp_s'],rgb_path=str(rgb),rgb_sha256=sha(rgb),native_depth_path=str(native),native_depth_sha256=sha(native),color_K=k.tolist(),depth_K=k.tolist(),native_K=k.tolist(),color_shape=[192,256],depth_shape=[192,256],native_shape=[192,256],highres_depth_path=hi_path,highres_K=hi_k,highres_timestamp_s=hi_time,trajectory_path=str(traj),grid_K=k.tolist(),grid_shape=[192,256])
                    frames.append(fr);public.append(fr);refs.append(dict(**fr,reference_path=str(reference),reference_sha256=sha(reference),queries=list(states)))
                entry=dict(capture=cap,visit_id=visit,role=role,accepted_ordinal=ordinal,frames=frames,near_POS=near,midfar_strict_FREE=free,trajectory_path=str(traj));entries.append(entry);rec.update(status='ACCEPTED',role=role)
                write(root/'dataset_manifest.json',dict(entries=entries,rows=public,queries=fixed,plan_sha256=sha(plan)))
            except TimeoutError:raise
            except Exception as exc:rec.update(status='SKIP_SOURCE_FAILURE',error=repr(exc),traceback=traceback.format_exc())
            finally:
                for a in arcs.values():a.close()
                for s in streams.values():s.close()
                write(dest/'candidate_terminal.json',rec);write(root/'candidate_gate_records.json',records);print(json.dumps(rec),flush=True)
            if len(entries)==12:status='COMPLETE12';break
    except Exception as exc:error=repr(exc);status='BUDGET_STOP' if isinstance(exc,TimeoutError) else 'FAILED_PARTIAL'
    finally:
        entries.sort(key=lambda e:next(r['metadata_index'] for r in records if r['capture']==e['capture']))
        for ordinal,e in enumerate(entries):
            role='train' if ordinal<6 else 'cal' if ordinal<8 else 'eval';e['role']=role;e['accepted_ordinal']=ordinal
            for f in e['frames']:f['role']=role;f['split']=role
            for rec in records:
                if rec['capture']==e['capture']:rec['role']=role
            for f in refs:
                if f['capture']==e['capture']:f['role']=role;f['split']=role
        public=[f for e in entries for f in e['frames']]
        allowed={'environment','scan','split','frame','rgb_path','rgb_sha256','color_K','color_shape','depth_K','depth_shape','role','cohort','visit_id','window_id','timestamp_s','frame_id'}
        alloc.session.close();write(root/'public_roster.json',dict(rows=[{k:v for k,v in row.items() if k in allowed} for row in public],queries=fixed));write(root/'reference_roster.json',dict(rows=refs,queries=fixed))
        write(root/'dataset_manifest.json',dict(status=status,entries=entries,rows=public,queries=fixed,plan_sha256=sha(plan)))
        write(root/'train_cal_reference_roster.json',dict(rows=[f for f in refs if f['role']!='eval'],queries=fixed))
        write(root/'candidate_gate_records.json',records)
        write(root/'admission_seal.json',dict(public_roster_sha256=sha(root/'public_roster.json'),accepted_visit_ids=[e['visit_id'] for e in entries],plan_sha256=sha(plan)))
        with (root/'visit_split.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['capture','visit_id','role','accepted_ordinal']);writer.writeheader();writer.writerows({k:e[k] for k in writer.fieldnames} for e in entries)
        write(root/'acquisition_terminal.json',dict(status=status,error=error,command_wall_s=time.perf_counter()-start,download_bytes=alloc.received,accepted_visits=len(entries),records=records,resources_released=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wall-s',type=float,default=900);p.add_argument('--resume-engineering',action='store_true');p.add_argument('--recover-network-gaps',action='store_true');a=p.parse_args();run(a.root,a.wall_s,a.resume_engineering,a.recover_network_gaps)
