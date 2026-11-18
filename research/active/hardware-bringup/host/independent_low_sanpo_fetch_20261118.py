"""Acquire and screen new official SANPO train recordings without evaluating model alerts.

Use the project's GPU-venv Python (CPU acquisition) with OPENBLAS_NUM_THREADS=1,
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1. Call this file with the following subcommands.
pool; for each [start,end) = [0,48), [48,144), [144,282): sparse, screen, windows with
--start and --end; controls; dense; review; review_dense; nominated; nominated_dense;
freeze. For the separate supplement run sweep with --start/--end [0,144), [144,282);
review_approach; supplement --indices 2 7 8 24 25 26 57; inventory; coverage.
The nomination indices preserve independent visual selections, never model outcomes.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import sanpo_wall_attribution_20261118 as prior
from sanpo_head_seg_fetch_20261117 import MEDIA, ROOT

OUT = prior.ART / "work/independent-low-20261118/sanpo-new"
DATA = OUT / "data"
COVER = prior.ART / "work/sanpo-coverage-20260927"
BASE = "camera_chest/left/"
LIMIT = 15 * 2**30


def read(p):
    return json.loads(p.read_bytes())


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def hash_id(s):
    return hashlib.sha256(("independent-low-20261118:" + s).encode()).hexdigest()


def fetch(path):
    dst, receipt = DATA / path, OUT / "receipts" / (path + ".json")
    if receipt.exists():
        meta = read(receipt)
        if dst.is_file() and dst.stat().st_size == meta["size"]:
            return True
        if meta.get("http_status") == 404:
            return False
    url = MEDIA + ROOT + path
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
                encoded = ",".join(response.headers.get_all("x-goog-hash") or [])
                md5 = base64.b64encode(hashlib.md5(content).digest()).decode()
                assert len(content) == int(response.headers["Content-Length"])
                assert md5 in encoded or response.headers.get("etag", "").strip('"') == hashlib.md5(content).hexdigest()
                info = dict(path=path, url=url, size=len(content), md5=md5, generation=response.headers.get("x-goog-generation"),
                            sha256=hashlib.sha256(content).hexdigest(), retrieved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(content)
            os.replace(tmp, dst)
            save(receipt, info)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                save(receipt, dict(path=path, url=url, size=0, http_status=404))
                return False
            if attempt == 4:
                raise
        except (OSError, TimeoutError):
            if attempt == 4:
                raise
        time.sleep(attempt+1)


def bulk(paths):
    paths = list(dict.fromkeys(paths))
    with ThreadPoolExecutor(max_workers=16) as pool:
        work = [pool.submit(fetch, p) for p in paths]
        total = sum(f.result() for f in as_completed(work))
    print("fetched or verified", total, "/", len(paths), flush=True)


def filename(s, i, rgb=False):
    return f"{s}/{BASE}" + (f"video_frames/{i:06d}.png" if rgb else f"depth_maps/{i:06d}.float16.gz")


def pool_setup():
    sources = {}
    for path in (prior.ART / "downloads").glob("sanpo-real-*"):
        sources[path.name] = sorted(read(path / "manifest.json"))
    sources["coverage"] = sorted({i["path"].split("\\")[0] for i in read(COVER/"acquisition_receipt.json")["files"]})
    sources["suitability"] = ["d3CKbaNbOPz-ZjLPUd6qD6wSNC3OKQxL"]
    old = set().union(*sources.values())
    all_train = (COVER/"train_session_ids.txt").read_text().split()
    available = sorted(set(all_train)-old, key=hash_id)
    assert len(old) == 351 and len(available) == 282
    save(OUT / "pool.json", dict(train_split_url=MEDIA+ROOT+"splits/train_session_ids.txt", old=sources,
         excluded_in_train=len(set(all_train)&old), train=len(all_train), available=available,
         hash_order="sha256('independent-low-20261118:'+raw id)"))
    bulk([s+"/description.json" for s in available])
    # First 48 in shuffled new pool, with a separately fixed negative-control window in first 12.
    chosen = available[:48]
    save(OUT/"initial.json", dict(sessions=chosen, anchors=list(range(30,390,45)),
                                  controls=[dict(seq=s, frames=list(range(60,121,7))) for s in chosen[:12]]))


def sparse(start,end):
    chosen=read(OUT/"pool.json")["available"][start:end]
    bulk([filename(s,i) for s in chosen for i in range(30,390,45)])


def cams(s):
    spec=read(DATA/s/"description.json")
    cp=spec["session_camera_details"][spec["session_camera_location"].index("camera_chest")]["left_camera_params"]
    k=1280/cp["image_width"]
    return np.array([[k*cp["fx"],0,k*cp["cx"]],[0,k*cp["fy"],k*cp["cy"]],[0,0,1.]])


def geometry(s,i):
    path=DATA/filename(s,i)
    if not path.exists():
        return None
    a=np.frombuffer(gzip.decompress(path.read_bytes()),"<f2")
    z=a[2:].reshape(int(a[0]),int(a[1])).astype(np.float32)
    # Coarse search only: exclude upper half when fitting a visible ground surface.
    # The final independent scorer uses the full-resolution frozen sky/geometry protocol.
    sample=3
    v,u=np.mgrid[:z.shape[0]:sample,:z.shape[1]:sample]
    d=z[::sample,::sample]
    K=cams(s)
    ray=np.stack(((u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones_like(v)),axis=-1)
    valid=np.isfinite(d)&(d>.30)&(d<10.)
    points=ray*np.where(valid,d,0)[:,:,None]
    seed=int(hashlib.sha256(f"{s}:{i}".encode()).hexdigest()[:8],16)
    gp=prior.core.ground_plane(points[valid&(v>=360)],np.array([0.,1.,0.]),np.random.default_rng(seed),max_deg=35.)
    result=dict(seq=s,frame=i,t=i/15,rgb=str(DATA/filename(s,i,True)),depth=str(path),K=K.tolist())
    if gp is None:
        return dict(**result,status="no_ground",truth=None,components=[])
    n,h=gp
    down=-n
    fwd=np.array([0.,0.,1.])-down*down[2]
    fwd/=np.linalg.norm(fwd)
    right=np.cross(down,fwd)
    along=points@fwd; high=h-points@down
    corridor=valid&(np.abs(points@right)<=.35)&(along>=.30)&(along<=2.50)
    area=d.astype(np.float64)**2*sample**2/(K[0,0]*K[1,1]*np.linalg.norm(ray,axis=-1))
    truth,components={},[]
    for name,(lo,hi) in {"LOW":(.15,.50),"BODY":(.50,1.30),"HEAD":(1.30,2.),"SHALLOW":(.05,.15)}.items():
        raw=corridor&(high>=lo)&(high<hi)
        opened=cv2.morphologyEx(raw.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        truth[name]=1 if area[opened>0].sum()>=.005 else (-1 if raw.any() else 0)
        if name not in ("SHALLOW","LOW") or truth[name]!=1:
            continue
        cnts,_=cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x,y,w,k=cv2.boundingRect(c)
            if w*k<4:
                continue
            mask=np.zeros_like(opened)
            cv2.drawContours(mask,[c],-1,1,-1)
            m=(mask>0)&(opened>0)
            a0=float(area[m].sum())
            if a0<.005:
                continue
            components.append(dict(band=name,area_m2=a0,forward_min=float(along[m].min()),
                        forward_median=float(np.median(along[m])),height_median=float(np.median(high[m])),
                        xmin=float((points[m]@right).min()),xmax=float((points[m]@right).max()),
                        bbox=[sample*x,sample*y,sample*w,sample*k],
                        polygon=(cv2.approxPolyDP(c,1.,True).reshape(-1,2)*sample).tolist()))
    return dict(**result,status="provisional",truth=truth,components=components,ground=[n.tolist(),h])


def screen(start,end,tag=None,sessions=None,frames=None):
    sessions=sessions or read(OUT/"pool.json")["available"][start:end]
    tag=tag or f"{start}_{end}"
    jobs=[(s,i) for s in sessions for i in (frames or range(30,390,45))]
    with ThreadPoolExecutor(4) as ex:
        records=[r for r in ex.map(lambda a:geometry(*a),jobs) if r]
    save(OUT/f"screen_{tag}.json",records)
    picked={}
    for s in sessions:
        matches=[]
        for r in records:
            if r["seq"]!=s:
                continue
            for c in r["components"]:
                if c["forward_min"]<=(2.5 if tag=="stairs_junction" else 2.0):
                    matches.append((c["band"]!="LOW", c["forward_min"]>1.5,
                                    abs(c["height_median"]-.24),r["frame"], c))
        if matches:
            best=min(matches,key=lambda x:x[:4])
            picked[s]=dict(frame=best[3],component=best[4])
    save(OUT/f"picked_{tag}.json",picked)
    bulk([filename(s,r["frame"],True) for s,r in picked.items()])
    print("candidates", len(picked), flush=True)


def window(start,end,controls=False,suffix=None):
    suffix=suffix or f"{start}_{end}"
    tag="controls" if controls else f"events_{suffix}"
    if controls:
        windows={r["seq"]:r["frames"] for r in read(OUT/"initial.json")["controls"]}
    else:
        picked=read(OUT/f"picked_{suffix}.json")
        windows={s:[r["frame"]+d for d in range(-30,19,6) if r["frame"]+d>=0]
                 for s,r in picked.items()}
    paths=[filename(s,i,rgb) for s,frames in windows.items() for i in frames for rgb in (False,True)]
    # Entire 282-session survey stays inside the request's 15 GiB cap.
    # Max original PNG observed in SANPO (2208x1242x3) is bounded by raw bytes + container overhead.
    # A 1280x720 gzip-float16 is bounded by 1.9 MB. Check this stage, including overlapping jobs.
    bytes_done=sum(read(p)["size"] for p in (OUT/"receipts").glob("*/camera_chest/left/*/*.json"))
    remaining=[p for p in paths if not (DATA/p).exists()]
    upper=sum(8_400_000 if p.endswith(".png") else 1_900_000 for p in remaining)
    if bytes_done + upper > LIMIT:
        raise RuntimeError("Download bound would exceed 15 GiB; use a smaller stage")
    bulk(paths)
    complete={s:[i for i in ix if (DATA/filename(s,i)).exists() and (DATA/filename(s,i,True)).exists()]
              for s,ix in windows.items()}
    save(OUT/(tag+".json"),dict(dataset="sanpo_chest_independent_20261118",data_root="data",
         fps=15,camera="camera_chest",selection=tag,sessions={s:v for s,v in complete.items() if v}))


def dense_strata():
    ordered=read(OUT/"pool.json")["available"]
    metadata={s:read(DATA/s/"description.json")["session_video_metadata"] for s in ordered}
    stairs=[s for s in ordered if "ELEVATION_CHANGE_STAIRS" in metadata[s]["elevation_changes"]]
    junction=[s for s in ordered if "ENVIRONMENT_TYPE_ROAD_JUNCTION" in metadata[s]["environment_types"] and s not in stairs][:20]
    sessions=stairs+junction
    frames=list(range(12,444,12))
    save(OUT/"stairs_junction_strata.json",dict(stairs=stairs,junction=junction,frames=frames))
    bulk([filename(s,i) for s in sessions for i in frames])
    screen(0,0,"stairs_junction",sessions,frames)
    window(0,0,suffix="stairs_junction")


def nominated(dense=False):
    rows=read(OUT/("blind_review/dense.json" if dense else "blind_review/sparse.json"))
    if dense:
        nomination=[("VWw2W_ob",24),("DGTZ7ga3",12),("DGTZ7ga3",96),("vQPpHa0p",12),
                    ("vQPpHa0p",24),("OauQwDO",312),("OauQwDO",336),("OauQwDO",348),
                    ("OauQwDO",396),("G5cL56",12),("oOsH0KW",108)]
    else:
        nomination=[("M-DsjqPJ",210),("h9uXYC5d",30),("EmUvfKbc",120),("OauQwDO",345),
                    ("kddYWMP",210),("9RLDrka",120)]
    windows={}
    for prefix,i in nomination:
        seq=next(r["seq"] for r in rows if r["seq"].lower().replace("0","o").startswith(prefix.lower().replace("0","o")) and r["frame"]==i)
        windows[seq]=sorted(set(windows.get(seq,[]))|set(range(max(0,i-48),i+49,6)))
    bulk([filename(s,i,rgb) for s,frames in windows.items() for i in frames for rgb in (False, True)])
    available={s:[i for i in ix if (DATA/filename(s,i)).exists() and (DATA/filename(s,i,True)).exists()]
               for s,ix in windows.items()}
    name="events_dense_nominations.json" if dense else "events_blind_nominations.json"
    save(OUT/name,dict(dataset="sanpo_chest_independent_20261118", data_root="data",
         fps=15,camera="camera_chest",selection="independent RGB visual nomination, no predictions",sessions=available))


def sweep(start,end):
    ordered=read(OUT/"pool.json")["available"][start:end]
    strata=read(OUT/"stairs_junction_strata.json")
    seen=set(strata["stairs"]+strata["junction"])
    sessions=[s for s in ordered if s not in seen]
    frames=[i for i in range(15,451,15) if i not in range(30,390,45)]
    tag=f"approach_{start}_{end}"
    bulk([filename(s,i) for s in sessions for i in frames])
    screen(0,0,tag,sessions,frames)


def review(dense=False,approach=False):
    tags=(["approach_0_144","approach_144_282"] if approach else
          ["stairs_junction"] if dense else ["0_48","48_144","144_282"])
    records=[r for tag in tags for r in read(OUT/f"screen_{tag}.json") if r["components"]]
    if approach:
        # A new recording beyond the frozen first release can count as independent support.
        # Keep all object types and distance bands within this independent subset.
        previous=set(read(OUT/"release.json")["sessions"])
        records=[r for r in records if r["seq"] not in previous and not r["seq"].startswith("oTc35OY")]
    records.sort(key=lambda r:(hash_id(r["seq"]),r["frame"]))
    label="approach" if approach else "dense" if dense else "sparse"
    bulk([filename(r["seq"],r["frame"],True) for r in records])
    tiles=[]
    directory=OUT/"blind_review"
    directory.mkdir(parents=True,exist_ok=True)
    for j,r in enumerate(records):
        img=cv2.resize(cv2.imread(r["rgb"]),(1280,720),interpolation=cv2.INTER_AREA)
        for c in r["components"]:
            pts=np.array(c["polygon"],np.int32).reshape(-1,1,2)
            cv2.polylines(img,[pts],True,(20,245,245) if c["band"]=="LOW" else (240,130,20),3)
        tile=cv2.resize(img,(384,216),interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile,(0,0),(265,20),(0,0,0),-1)
        cv2.putText(tile,f"{j:03d} {r['seq'][:8]} fr{r['frame']}",(4,15),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
        tiles.append(tile)
        if len(tiles)==12 or j==len(records)-1:
            tiles+=[np.zeros_like(tile)]*((-len(tiles))%3)
            image=np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)])
            assert cv2.imwrite(str(directory/f"{label}_{j//12:02d}.jpg"),image)
            tiles=[]
    save(directory/f"{label}.json",records)


def freeze(label,names):
    union, sources={},{}
    for name in names:
        for seq,ix in read(OUT/name)["sessions"].items():
            union.setdefault(seq,set()).update(ix)
            sources.setdefault(seq,{})[name]=ix
    with urllib.request.urlopen(MEDIA+ROOT+"splits/train_session_ids.txt",timeout=15) as response:
        official=response.read()
        generation=response.headers["x-goog-generation"]
    assert official==(COVER/"train_session_ids.txt").read_bytes()
    pool=read(OUT/"pool.json")
    assert set(union)<=set(pool["available"])
    release=dict(dataset="sanpo_chest_independent_20261118",data_root="data",fps=15,camera="camera_chest",
                 sessions={s:sorted(ix) for s,ix in sorted(union.items())},source_masks=sources)
    dest=OUT/f"{label}.json"
    if dest.exists() and read(dest)!=release:
        raise ValueError("Changing frozen release membership is prohibited")
    if not dest.exists():
        save(dest,release)
    receipts=[]
    for s,ix in release["sessions"].items():
        paths=[s+"/description.json"]+[filename(s,i,rgb) for i in ix for rgb in (False,True)]
        for path in paths:
            rec=read(OUT/"receipts"/(path+".json"))
            assert (DATA/path).stat().st_size==rec["size"]
            receipts.append(rec)
    save(OUT/f"{label}_receipt.json",dict(release_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        public_train_sha256=hashlib.sha256(official).hexdigest(), public_train_generation=generation,
        source=MEDIA+ROOT, acquisition_protocol="CC BY 4.0, official train; past recording IDs cross-camera excluded",
        id_sha256={s:hashlib.sha256(s.encode()).hexdigest() for s in union}, files=receipts,
        geolocation="Neither city/route ID nor recording date occurs in the official description schema",
        possible_development_site_match="oTc35OY at 165: distinctive concordant concentric stone bench, not an independent object",
        frames=sum(len(i) for i in union.values()),session_count=len(union),bytes=sum(r["size"] for r in receipts),
        input_manifests={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in names}))
    print(label,len(union),sum(len(i) for i in union.values()),flush=True)


def supplement(indices):
    review_rows=read(OUT/"blind_review/approach.json")
    windows={}
    for j in indices:
        r=review_rows[j]
        i,s=r["frame"],r["seq"]
        windows[s]=sorted(set(windows.get(s,[]))|set(range(max(0,i-48),i+49,6)))
    bulk([filename(s,i,rgb) for s,frames in windows.items() for i in frames for rgb in (False,True)])
    usable={s:[i for i in ix if (DATA/filename(s,i)).exists() and (DATA/filename(s,i,True)).exists()] for s,ix in windows.items()}
    save(OUT/"events_approach_nominations.json",dict(dataset="sanpo_chest_independent_20261118",
         data_root="data",fps=15,camera="camera_chest",selection="additional reference/RGB nominations; disjoint from first release",
         nominal_review_indices=indices,sessions=usable))
    freeze("release_supplement",["events_approach_nominations.json"])


def inventory():
    from collections import Counter
    available=read(OUT/"pool.json")["available"]
    assert len(available)==282 and len(set(available))==282
    entries=[read(p) for p in (OUT/"receipts").glob("*/description.json.json")]
    entries.extend(read(p) for p in (OUT/"receipts").glob("*/camera_chest/left/*/*.json"))
    verified=[x for x in entries if x.get("size",0)]
    missing=[x for x in entries if x.get("http_status")==404]
    unique={x["path"] for x in verified}
    assert len(unique)==len(verified)
    nbytes=sum(x["size"] for x in verified)
    assert nbytes<=LIMIT
    paired=sum(1 for name in unique if "/video_frames/" in name and
       name.replace("/video_frames/","/depth_maps/").replace(".png",".float16.gz") in unique)
    screens=[p for p in OUT.glob("screen*.json")]
    groups={}
    sample=set()
    for p in screens:
        rows=read(p)
        groups[p.stem]=dict(frames=len(rows),sessions=len({r["seq"] for r in rows}),
           with_components=sum(bool(r["components"]) for r in rows))
        sample.update((r["seq"],r["frame"]) for r in rows)
    releases={p.stem:read(p) for p in (OUT/"release.json",OUT/"release_supplement.json") if p.exists()}
    membership=[set(v["sessions"]) for v in releases.values()]
    if len(membership)>1:
        assert membership[0].isdisjoint(membership[1])
    save(OUT/"acquisition_summary.json",dict(source=MEDIA+ROOT,official_train_ids=560,previous_ids_across_splits=351,
      new_train_sessions=len(available),raw_payload_bytes=nbytes,cap_bytes=LIMIT,paired_rgb_depth=paired,
      frames_submitted_to_reference_screen=len(sample),source_groups=groups,verified_objects=len(verified),missing_objects=len(missing),
      object_types=Counter("meta" if "/camera_chest/" not in x["path"] else x["path"].split("/")[-2] for x in verified),
      releases={k:dict(session_count=len(v["sessions"]),frames=sum(len(ix) for ix in v["sessions"].values()),
                       sha256=hashlib.sha256((OUT/(k+".json")).read_bytes()).hexdigest()) for k,v in releases.items()}))
    print("verified bytes",nbytes,"pairs",paired,"screened frames",len(sample),flush=True)


def coverage():
    path=OUT/"acquisition_summary.json"
    report=read(path)
    all_rows={}
    groups={}
    missing=set(read(OUT/"pool.json")["available"])
    stages={"sparse":["0_48","48_144","144_282"],"stairs_junction":["stairs_junction"],
            "other_one_second_supplement":["approach_0_144","approach_144_282"]}
    for label,tags in stages.items():
        rows=[r for t in tags for r in read(OUT/f"screen_{t}.json")]
        unique={(r["seq"],r["frame"]):r for r in rows}
        groups[label]=dict(frames=len(unique), ground_fit_available=sum(r["truth"] is not None for r in unique.values()),
                           sessions=len({s for s,i in unique}), additional_unique_frames=len(unique.keys()-all_rows.keys()))
        all_rows.update(unique)
    missing-={s for s,i in all_rows}
    controls=read(OUT/"controls.json")["sessions"]
    report.update(reference_screen=groups, reference_ground_fit_unique=sum(r["truth"] is not None for r in all_rows.values()),
         screened_sessions=len({s for s,i in all_rows}), unobserved_ids=sorted(missing),
         fixed_control_windows=len(controls), fixed_control_frames=sum(map(len,controls.values())),
         control_role="randomly fixed candidate control windows; negative status determined independently",
         raw_bytes_metric="sum of verified unique official object sizes; excludes derived files, HTTP overhead and retry traffic")
    save(path,report)
    print(json.dumps({k:report[k] for k in ("reference_screen","reference_ground_fit_unique","screened_sessions",
          "unobserved_ids","fixed_control_windows","fixed_control_frames")},indent=1))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("op")
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--end",type=int,default=48)
    p.add_argument("--indices",type=int,nargs="+")
    args=p.parse_args()
    if args.op=="pool":
        pool_setup()
    elif args.op=="sparse":
        sparse(args.start,args.end)
    elif args.op=="screen":
        screen(args.start,args.end)
    elif args.op in ("windows","controls"):
        window(args.start,args.end,args.op=="controls")
    elif args.op=="dense":
        dense_strata()
    elif args.op in ("review","review_dense","review_approach"):
        review(args.op=="review_dense",args.op=="review_approach")
    elif args.op in ("nominated","nominated_dense"):
        nominated(args.op=="nominated_dense")
    elif args.op=="sweep":
        sweep(args.start,args.end)
    elif args.op=="freeze":
        freeze("release",["controls.json","events_0_48.json","events_48_144.json","events_144_282.json",
              "events_stairs_junction.json","events_blind_nominations.json","events_dense_nominations.json"])
    elif args.op=="supplement":
        supplement(args.indices)
    elif args.op=="inventory":
        inventory()
    elif args.op=="coverage":
        coverage()
