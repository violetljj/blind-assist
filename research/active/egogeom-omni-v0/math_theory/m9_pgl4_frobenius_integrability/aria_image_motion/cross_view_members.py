"""Test the *same model-supported voxels* with the other wearer's simultaneous depth."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from build import HERE, OUT, OLD_OUT, dump, sha
from threadpoolctl import threadpool_limits

M9=HERE.parent
SOURCE=OUT.parent/"aria_surface_witness"
PEER=M9/"aria_cross_device"/"audit.py"
spec=importlib.util.spec_from_file_location("paired_aria_cam_audit",PEER)
cross=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=cross
spec.loader.exec_module(cross)
from reference import keys3


def main():
    cameras=[cross.Recording("walk_"+str(i)) for i in (0,1)]
    assert cameras[0].graph==cameras[1].graph
    archive=json.loads((OLD_OUT/"v5_aria_alert_support.json").read_text())["results"]
    peer_ledger=OUT.parent/"aria_cross_device/audit.json"
    all_peer=json.loads(peer_ledger.read_text())["frames"]
    ranked=json.loads((OUT/"results.json").read_text())["versions"]
    clips=sorted((SOURCE/"clips").glob("*.json"))
    assert len(clips)==19
    results={}
    for ver in ("v4","v41"):
        rows={}
        for path in clips:
            name=path.stem
            w=int(name.split("_")[2])
            block=int(name.rsplit("_c",1)[1])*900
            own,peer=cameras[w],cameras[1-w]
            ledger=json.loads(path.read_text())
            assert sha(path.with_suffix(".npz"))==ledger["pixel_receipt_sha256"]
            with np.load(path.with_suffix(".npz")) as pix:
                offsets=pix[ver+"_offsets"]
                coords=pix[ver+"_native_pixel_512"]
                depths=pix[ver+"_stereo_axial_mm"]
                flags=pix[ver+"_flags"]
            frames={}
            for row in ledger["rows"][ver]:
                if not row.get("hits",{}).get("full_witness"):
                    continue
                i=row["frame"]
                rep=row["representative"]
                assert rep["full"]
                j=block+6*rep["source_clip_frame"]
                f0,f1=offsets[i:i+2]
                mask=flags[f0:f1]==127
                loc=coords[f0:f1][mask]
                z=depths[f0:f1][mask].astype(float)*.001
                fx,fy,cx,cy=own.k[j]
                rays=np.column_stack(((loc%512-cx)/fx,(loc//512-cy)/fy,np.ones(len(loc))))
                xyz=(rays*z[:,None])@own.rot[j].T+own.pos[j]
                keys=np.unique(keys3(xyz))
                assert len(keys)==rep["unique_target_voxels"]>=8
                jj,delta=peer.match(own.utc[block+6*i])
                res=cross.target_test(keys,peer,jj,peer.frame(jj)) if jj is not None else {}
                frames[i]=dict(model_supported_voxels=len(keys),peer=res,peer_delta_ns=int(delta))
            rows[name]=frames
        windows=[]
        for e in archive[ver]["events"]:
            hz=e["timely_hazard_frames"]
            if not hz:
                continue
            witnessed=[rows[e["clip"]][i] for i in hz if i in rows[e["clip"]]]
            support=any(r["peer"].get("015",{}).get("supports",False) for r in witnessed)
            rejects=bool(witnessed) and all(r["peer"].get("015",{}).get("rejects_cluster",False) for r in witnessed)
            windows.append(dict(clip=e["clip"],walker=e["walker"],start=e["event_start"],
                                own=bool(witnessed),supported=support,rejected_all_representatives=rejects))
        totals={k:sum(e[k] for e in windows) for k in ("own","supported","rejected_all_representatives")}
        assert len(windows)==129 and totals["own"]=={"v4":6,"v41":12}[ver]
        evaluated=ranked[ver]
        related=[]
        for g,a,event in zip(evaluated['geometry']['events'],evaluated['aligned']['events'],archive[ver]['events'],strict=True):
            n=a['clip']
            assert (n,a['walker'],a['start'])==(g['clip'],g['walker'],g['start'])==(event['clip'],event['walker'],event['event_start'])
            if g['distance_hit'] or not a['distance_hit']:
                continue
            with np.load(OUT/ver/'aligned'/(n+'.npz')) as saved:
                distance=saved['distance']
            matched=[i for i in event['timely_hazard_frames'] if abs(distance[i]-all_peer[n][i]['reference_m'])<=.55]
            assert matched
            related.append(dict(clip=n,walker=a['walker'],start=a['start'],matching_frames=matched,
                any_own_witness_same_frame=any(i in rows[n] for i in matched),
                any_peer_full_target_same_frame=any((all_peer[n][i]['cluster'] or {}).get('015',{}).get('supports',False) for i in matched)))
        results[ver]=dict(windows=windows,frames=rows,totals=totals,aligned_distance_rescues=related)
    dump(OUT/"cross_view_members.json",dict(versions=results,own=sha(SOURCE/"surface_witness_report.json"),
         reference_events=sha(OLD_OUT/"v5_aria_alert_support.json"),ranking=sha(OUT/"results.json"),
         peer_ledger=sha(peer_ledger),geometry=sha(PEER),code=sha(Path(__file__)),
         inputs={cam.walk:dict(clock=cam.receipt,depths=cam.hashes) for cam in cameras}))
    print({v:x["totals"] for v,x in results.items()})

if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
