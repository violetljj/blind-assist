"""Aggregate held characteristics, one physical frame per record and scorer-exact opportunity windows."""
from collections import Counter
import json
from pathlib import Path

from audit import ART, HERE, OLD, OUT, NATIVE, FIELDS, SIBLING, sha, dump
from reference import cv2,NATIVE_U,NATIVE_V
import numpy as np


def read(path):
    return json.loads(path.read_text())


def stage(rows):
    return {"count":len(rows),**{f:sum(r["hits"][f] for r in rows) for f in FIELDS}}


def check_pixels(path,clip,version,rows):
    meta={}
    failures={k:Counter() for k in ("hazard","clear")}
    seen=set()
    walk=clip.removeprefix("aria_").rsplit("_c",1)[0]
    with np.load(path) as p:
        offsets=p[version+"_offsets"]
        src,nid,zref,flag=[p[version+"_"+k] for k in
                           ("source_pixel_s4","native_pixel_512","stereo_axial_mm","flags")]
    for row in rows:
        i=row["frame"]
        r=row.get("representative")
        if not r:
            assert offsets[i]==offsets[i+1]
            continue
        take=slice(offsets[i],offsets[i+1])
        indices=src[take]
        np.testing.assert_array_equal(nid[take],NATIVE_V[indices]*512+NATIVE_U[indices])
        flags=flag[take]
        assert len(flags)==r["points"] and int(((flags&1)>0).sum())==r["valid_same_pixels"]
        assert int(((flags&2)>0).sum())==r["axial_matches"]
        typ=row["reference"]
        counts=failures[typ]
        remain=np.ones(len(flags),bool)
        counts["all"]+=len(flags)
        # Disjoint accounting ordered by observable definition, not causal error attribution.
        for bit,name in ((1,"invalid_stereo"),(2,"axial_mismatch"),(4,"outside_causal_corridor"),
                         (8,"height_unavailable_or_outside"),(16,"different_forward_slab"),(32,"outside_label_heading")):
            passed=(flags&bit)>0
            counts[name]+=int((remain&~passed).sum())
            remain &= passed
        counts["retained"]+=int(remain.sum())
        if abs(int(remain.sum())-r["strict_same_slab"])>1:
            raise AssertionError((clip,version,i,r["strict_same_slab"],int(remain.sum())))
        if typ not in seen:
            native=r["native_frame"]
            file=NATIVE/walk/"x/depth/depth"/f"depth_{native:08d}.png"
            mm=cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(mm.ravel()[nid[take]],zref[take])
            seen.add(typ)
            meta[typ]={"query":i,"source_clip_frame":r["source_clip_frame"],"native_frame":native,
                       "matching_depth_pixels":len(flags)}
    return failures,meta


def main():
    files=sorted((OUT/"clips").glob("aria_walk_*.json"))
    assert len(files)==19
    oldpath=OLD/"v5_aria_alert_support.json"
    prev=read(oldpath)["results"]
    report={"code_sha256":{p.name:sha(p) for p in (HERE/"reference.py",HERE/"audit.py",Path(__file__),
         HERE/"PROTOCOL.md",SIBLING/"support.py",HERE.parents[3]/"egoclear-v0"/"pack_aria_g2_local.py")},
         "source_receipt_sha256":sha(NATIVE/"RECEIPT.json"),"event_windows_sha256":sha(oldpath),
         "clip_report_sha256":{p.name:sha(p) for p in files},"versions":{}}
    records={p.stem:read(p) for p in files}
    report["reference_bridge"]={"scored_frames":sum(c["checks"]["scored"] for c in records.values()),
           **{k:max(c["checks"][k] for c in records.values())
              for k in ("max_distance_error_m","max_coverage_error","max_pose_error")},
           "native_frames":sum(len(c["stereo_png_sha256"]) for c in records.values())}
    for ver in ("v4","v41"):
        rows={(name,r["frame"]):r for name,c in records.items() for r in c["rows"][ver]}
        cases={}
        for source in ("all","aria_walk_0","aria_walk_1"):
            chosen=[r for (name,i),r in rows.items() if source=="all" or name.startswith(source+"_")]
            cases[source]={s:stage([r for r in chosen if r["reference"]==s]) for s in ("hazard","clear")}
            cases[source]["unknown"]=sum(r["reference"]=="unknown" for r in chosen)
        events=[]
        for e in prev[ver]["events"]:
            hz=e["timely_hazard_frames"]
            flags={f:any(rows[e["clip"],i]["hits"][f] for i in hz) for f in FIELDS}
            events.append({"clip":e["clip"],"event_start":e["event_start"],"walker":e["walker"],
                           "has_hazard_time":bool(hz),"hits":flags})
        true_events=[e for e in events if e["has_hazard_time"]]
        assert len(true_events)==129 and sum(e["hits"]["predicted"] for e in true_events)==(115 if ver=="v4" else 96)
        by_walk={w:stage([e for e in true_events if e["clip"].startswith(w)]) for w in ("aria_walk_0","aria_walk_1")}
        names=sorted({(e["clip"],e["event_start"]) for e in true_events})
        unique=[{"hits":{f:any(e["hits"][f] for e in true_events if (e["clip"],e["event_start"])==key)
                         for f in FIELDS}} for key in names]
        traced={"hazard":Counter(),"clear":Counter()}
        verified={}
        for name,c in records.items():
            pixel=OUT/"clips"/(name+".npz")
            assert sha(pixel)==c["pixel_receipt_sha256"]
            stats,ref=check_pixels(pixel,name,ver,c["rows"][ver])
            for typ in traced:
                traced[typ].update(stats[typ])
            verified[name]=ref
        report["versions"][ver]={"frames":cases,"timely_hazard_events":stage(true_events),
                   "event_counts_by_recording":by_walk,"distinct_segments_with_hazard_time":stage(unique),
                   "representative_pixel_funnel":traced,"sampled_native_replay":verified,"events":events}
    dump(OUT/"surface_witness_report.json",report)
    for v,r in report["versions"].items():
        print(v,"events",r["timely_hazard_events"],"frames",r["frames"]["all"],flush=True)


if __name__=="__main__":
    main()
