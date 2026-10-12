"""Acquire complete FARO-derived depth sequences with official frame provenance."""
from pathlib import Path
import argparse, json, time, io, zipfile, hashlib, shutil
import numpy as np
from rgb_near_readout_scale_acquire import Allocation
from rgb_body_query_input_diagnostic import write, sha

def run(root,wall):
    t=time.perf_counter();root=root.resolve()
    assert sha(root/'PLAN.json')=='ccf3a6e12040aa32ce4adae7967ce03fefb23b1bff8f358f74841b38ca87bc4a'
    if (root/'source_terminal.json').exists():raise FileExistsError('Preserve existing terminal')
    inv=json.loads((root/'visit_capture_inventory.json').read_text());heads=json.loads((root/'asset_HEAD.json').read_text())['records']
    transforms={r['capture']:r for r in json.loads((root/'transform_receipts.json').read_text())['records'] if r.get('status')==200}
    initial=sum(r.get('bytes',0) for r in transforms.values())+json.loads((root/'coordinate_documentation_receipt.json').read_text())['bytes']+sum(r['bytes'] for r in json.loads((root/'transform_discovery_receipts.json').read_text()))
    alloc=Allocation(root,wall-3,initial,3_000_000_000);alloc.session.trust_env=False;entries=[];records=[];status='PARTIAL';error=None
    # First expose every original target component, then registered siblings only.
    todo=[(v,next(c for c in v['captures'] if c['video_id']==v['v1_capture'])) for v in inv]
    todo += [(v,c) for v in inv for c in v['captures'] if c['video_id']!=v['v1_capture'] and c['video_id'] in transforms and v['v1_capture'] in transforms and c['is_in_upsampling']=='True']
    try:
        for v,c in todo:
            alloc.remaining();cap=c['video_id'];dest=root/'source'/cap;dest.mkdir(parents=True,exist_ok=True);rec=dict(capture=cap,visit_id=v['visit_id'],role=v['role'],status='STARTING');records.append(rec)
            hh=[h for h in heads if h['capture']==cap and h.get('status')==200];required=sum(h['bytes'] for h in hh)
            if alloc.received+required>alloc.limit:rec['status']='BYTE_BUDGET_STOP';break
            try:
                for h in hh:
                    alloc.remaining();asset=h['asset'];size=h['bytes'];payload=alloc.request(h['url'],(0,size-1,size))
                    target=dest/asset;target.write_bytes(payload)
                    rec.setdefault('assets',[]).append(dict(asset=asset,url=h['url'],full_archive_bytes=size,planned_range_bytes=size,received_bytes=len(payload),sha256=sha(target)))
                    if asset.endswith('.zip'):
                        with zipfile.ZipFile(io.BytesIO(payload)) as z:
                            for info in z.infolist():
                                alloc.remaining()
                                if info.is_dir():continue
                                output=dest/info.filename
                                if not output.resolve().is_relative_to(dest.resolve()):raise ValueError('Unsafe archive path')
                                output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(z.read(info))
                traj=dest/'lowres_wide.traj';poses=np.loadtxt(traj);lo=float(poses[:,0].min());hi=float(poses[:,0].max());frames=[]
                for p in sorted((dest/'highres_depth').glob('*.png')):
                    alloc.remaining();stamp=float(p.stem.rsplit('_',1)[1]);kp=dest/'wide_intrinsics'/(p.stem+'.pincam');reason=None
                    if not kp.exists():K=None;reason='MISSING_INTRINSICS'
                    else:
                        w,h,fx,fy,cx,cy=np.fromstring(kp.read_text(),sep=' ');K=[[fx,0,cx],[0,fy,cy],[0,0,1]]
                    if stamp<lo or stamp>hi:reason='OUTSIDE_TRAJECTORY_NO_EXTRAPOLATION'
                    frames.append(dict(source_id=p.stem,depth_path=str(p),depth_sha256=sha(p),K=K,timestamp_s=stamp,trajectory_path=str(traj),usable_pose=lo<=stamp<=hi,exclusion_reason=reason))
                tr=transforms.get(cap);transform=None
                if tr:
                    pp=Path(tr['path']).resolve();a=np.load(pp);assert a.shape==(4,4) and np.isfinite(a).all();assert np.allclose(a[3],[0,0,0,1],atol=1e-6)
                    assert np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-3) and abs(np.linalg.det(a[:3,:3])-1)<1e-3;transform=str(pp)
                entry=dict(visit_id=v['visit_id'],capture=cap,role=v['role'],v1_capture=v['v1_capture'],frames=frames,trajectory_path=str(traj),trajectory_sha256=sha(traj),transform_path=transform,transform_sha256=sha(Path(transform)) if transform else None,cross_capture_alignment='OFFICIAL_FARO_TO_ARKIT_TRANSFORM' if transform else 'UNVERIFIED_SEPARATE_WORLD',pose_time_range_s=[lo,hi],source_frame_time_range_s=[min(f['timestamp_s'] for f in frames),max(f['timestamp_s'] for f in frames)] if frames else None)
                entries.append(entry);rec.update(status='COMPLETE',frames=len(frames),pose_missing_frames=sum(not f['usable_pose'] for f in frames));write(dest/'capture_manifest.json',entry)
            except TimeoutError:raise
            except Exception as e:rec.update(status='FAILED',error=repr(e))
            finally:
                write(dest/'terminal.json',rec);write(root/'source_manifest.json',dict(status='RUNNING',entries=entries,records=records,plan_sha256=sha(root/'PLAN.json')));print(json.dumps({k:v for k,v in rec.items() if k!='assets'}),flush=True)
        else:status='COMPLETE_ALL_TARGET_AND_REGISTERED_COMPONENTS'
    except Exception as e:status='BUDGET_STOP' if isinstance(e,TimeoutError) else 'FAILED_PARTIAL';error=repr(e)
    finally:
        alloc.session.close();manifest=dict(status=status,entries=entries,records=records,plan_sha256=sha(root/'PLAN.json'));write(root/'source_manifest.json',manifest);write(root/'source_terminal.json',dict(status=status,error=error,command_wall_s=time.perf_counter()-t,download_bytes=alloc.received,captures=len(entries),frames=sum(len(e['frames']) for e in entries),resources_released=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wall-s',type=float,default=590);a=p.parse_args();run(a.root,a.wall_s)
