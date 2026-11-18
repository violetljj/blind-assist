"""Frozen outdoor obstacle replay, with reference preparation separated from model comparison.

python -B independent_low_replay_20261118.py reference <source-manifest.json>
python -B independent_low_replay_20261118.py predict <source-manifest.json>

For SANPO a manifest specifies dataset, data_root (relative to manifest or absolute), camera,
fps and sessions (sequence -> list of original frame numbers). Other rectified streams can use
frames, a list of dataset-local relative RGB and depth paths, seq, i, t, K, optional down,
eye_height, ref_scale (m/unit, optical z) and mount. Observations never read target labels.
Each output run is bound to its input manifest SHA-256. Release only aggregate results.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

HOST = Path(__file__).resolve().parent
sys.path.insert(0, str(HOST.parent / "evalkit"))
import core
from methods import Methods

ROOT = HOST.parents[3]
OUT = ROOT / "artifacts.local/work/independent-low-20261118/replay"
ARMS = ("baseline", "sem80", "geom")
BANDS = tuple(core.BANDS)


def read(p):
    return json.loads(Path(p).read_bytes())


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=1),encoding="utf-8")
    temp.replace(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def unikey(seq, index):
    return hashlib.sha256(f"{seq}/{index}".encode()).hexdigest()[:22]


def frame_rng(dataset, row, operation):
    # Per-observation seed keeps preselection reference truth identical after windows are pooled.
    key = f"{dataset}/{row['seq']}/{row['i']}/{operation}"
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:12],16))


def cache_sky(dataset,row):
    return OUT/"reference_sky"/(unikey(dataset+"/"+row["seq"],row["i"])+".png")


def inputs(manifest):
    spec = read(manifest)
    root = manifest.parent / spec.get("data_root","data")
    records = []
    if "frames" in spec:
        for f in spec["frames"]:
            shape = f.get("size")
            records.append({**f, "rgb":root/f["rgb"], "depth":root/f["depth"],
                            "sky":root/f["sky"] if f.get("sky") else None, "size":shape,
                            "confidence":root/f["confidence"] if f.get("confidence") else None,
                            "mount":f.get("mount",spec.get("mount","head")), "sanpo":False})
    else:
        cam = spec.get("camera","camera_chest")
        for seq, frames in sorted(spec["sessions"].items()):
            desc = read(root/seq/"description.json")
            calib = desc["session_camera_details"][desc["session_camera_location"].index(cam)]["left_camera_params"]
            scale = 1280/calib["image_width"]
            k = [[scale*calib["fx"],0,scale*calib["cx"]],[0,scale*calib["fy"],scale*calib["cy"]],[0,0,1]]
            base = root/seq/cam/"left"
            for i in sorted(frames):
                name = f"{i:06d}"
                records.append(dict(seq=seq,i=i,t=i/spec.get("fps",15),K=k,rgb=base/"video_frames"/(name+".png"),
                                    depth=base/"depth_maps"/(name+".float16.gz"),sky=None,size=(1280,720),
                                    mount="chest" if cam=="camera_chest" else "head",sanpo=True))
    records.sort(key=lambda r:(r["seq"],r["t"]))
    assert len({(r["seq"],r["i"]) for r in records}) == len(records) and records
    return spec["dataset"], records


def image(row, target=None):
    x = cv2.imread(str(row["rgb"]))
    if x is None:
        raise FileNotFoundError(row["rgb"])
    size = target or row["size"]
    return cv2.resize(x,tuple(size),interpolation=cv2.INTER_AREA) if size else x


def load_frame(dataset,row,directory,with_image=False,model_grid=False):
    path = row["depth"]
    if row["sanpo"]:
        a = np.frombuffer(gzip.decompress(path.read_bytes()),"<f2")
        z = a[2:].reshape(int(a[0]),int(a[1])).astype(np.float32)
    else:
        z = np.load(path,allow_pickle=False) if path.suffix==".npy" else cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if z is None:
            raise FileNotFoundError(path)
        z = z.astype(np.float32)*row.get("ref_scale",1.)
        if row.get("confidence"):
            conf=cv2.imread(str(row["confidence"]),cv2.IMREAD_UNCHANGED)
            if conf is None or conf.shape!=z.shape:
                raise ValueError(f"Unregistered confidence at {path}")
            z[conf<row.get("min_confidence",1)]=np.nan
    if row["size"] and z.shape != (row["size"][1],row["size"][0]):
        raise ValueError(f"Depth and calibrated RGB do not share a pixel grid: {path}")
    z[~np.isfinite(z) | (z<=0)] = np.nan
    sky = cache_sky(dataset,row) if row["sanpo"] else row["sky"]
    if sky is not None:
        sky_data = cv2.imread(str(sky),cv2.IMREAD_GRAYSCALE)
        if sky_data is None or sky_data.shape != z.shape:
            raise ValueError(f"Missing or unregistered sky mask: {sky}")
        z[sky_data>0] = np.nan
    k=np.asarray(row["K"],float).copy()
    scale=row.get("native_scale",1) if model_grid else 1
    if scale!=1:
        if not isinstance(scale,int) or not 1<scale<=10:
            raise ValueError("High-resolution native LiDAR analysis needs an integer image scale")
        hh,ww=z.shape
        z=cv2.resize(z,(ww*scale,hh*scale),interpolation=cv2.INTER_NEAREST)
        k[:2]*=scale
        rgb_size=row.get("rgb_native_size") or row.get("rgb_size")
        if row.get("rgb_K") and rgb_size:
            # Use the actual RGB projection when supplied. The sources must describe the same
            # rectified rays to within one native pixel, otherwise nearest registration is invalid.
            rgb_k=np.asarray(row["rgb_K"],float).copy()
            rgb_k[0]*=z.shape[1]/rgb_size[0]
            rgb_k[1]*=z.shape[0]/rgb_size[1]
            if np.max(np.abs(rgb_k[:2]-k[:2]))>scale:
                raise ValueError("RGB and native LiDAR intrinsics require explicit registration")
            k=rgb_k
    im = image(row,(z.shape[1],z.shape[0])) if with_image else None
    if im is not None and im.shape[:2] != z.shape:
        raise ValueError(f"RGB and reference pixels do not coincide: {path}")
    down = np.array(row["down"],float) if row.get("down") is not None else None
    # If the independent reference provider locked a native floor, keep its normal and offset
    # exactly. No camera height/ground annotation is passed to the geometry protection itself.
    extra={"ground":(np.asarray(row["ground"][0],float),float(row["ground"][1]))} if row.get("ground") else {}
    return core.Frame(dataset,row["seq"],row["i"],row["t"],im,k,z,
                      row.get("ref_grade","pseudo_ground_truth"),down,row.get("eye_height"),row["mount"],
                      block=row["seq"],extra=extra)


def prepare_sky(dataset, records):
    # The SANPO sky rule runs only in this reference stage. This is the legacy independent
    # class-2 argmax of upsampled logits. It never reads the ground probability or obstacle arm.
    sky_rows = [r for r in records if r["sanpo"] and not cache_sky(dataset,r).is_file()]
    if sky_rows:
        import torch
        from sanpo_auto_sky_20261117 import load, sky_masks
        proc,net = load()
        (OUT/"reference_sky").mkdir(parents=True,exist_ok=True)
        for j in range(0,len(sky_rows),2):
            batch = sky_rows[j:j+2]
            masks = sky_masks(proc,net,[image(r) for r in batch])
            for r,m in zip(batch,masks):
                p = cache_sky(dataset,r)
                if not cv2.imwrite(str(p), m.astype(np.uint8)*255):
                    raise OSError(str(p))
        del proc,net
        torch.cuda.empty_cache()


def reference(manifest,dataset,records,directory):
    from sanpo_ground_alert_replay_20261118 import evaluator
    prepare_sky(dataset,records)
    historical={}
    if all(r["sanpo"] for r in records):
        # All of these windows refer to the identical immutable SANPO source tree. The random
        # reference draws are keyed by dataset + scene + index, independent of window order.
        for prior in sorted(OUT.glob("*/reference.json")):
            prior_ref=read(prior)
            if prior_ref["dataset"]==dataset:
                for row in prior_ref["rows"]:
                    key=(row["seq"],row["i"])
                    if key not in historical or "eye_sample" in row:
                        historical[key]=row
    heights = defaultdict(list)
    per_frame_height={}
    for r in records:
        key=(r["seq"],r["i"])
        if "eye_sample" in historical.get(key,{}):
            h=historical[key]["eye_sample"]
            per_frame_height[key]=h
            if h is not None:
                heights[r["seq"]].append(h)
            continue
        f=load_frame(dataset,r,directory)
        if f.eye_height is not None:
            heights[f.seq].append(float(f.eye_height))
            per_frame_height[key]=float(f.eye_height)
            continue
        # Reuse the same floor estimator and sequence-median eye-height contract as run_a2.
        z = f.ref_depth.ravel()
        ok = np.isfinite(z) & (z>.3) & (z<10)
        points = core.rays(f.K,*f.ref_depth.shape)[ok] * z[ok,None]
        rng_height=frame_rng(dataset,r,"mount")
        gp = (core.ground_plane(points, f.down, rng_height) if f.down is not None else
              core.ground_plane(points,np.array([0,1.,0]),rng_height,max_deg=35.))
        per_frame_height[key]=float(gp[1]) if gp is not None else None
        if gp is not None:
            heights[f.seq].append(float(gp[1]))
    eye = {s:float(np.median(v)) for s,v in heights.items()}
    rng = np.random.default_rng(20261117)
    sample = list(range(0,len(records),max(1,len(records)//20)))[:20]
    if {r["mount"] for r in records}.issubset({"head","chest"}):
        check = core.self_check([load_frame(dataset,records[i],directory) for i in sample],rng)
    else:
        if not all(r.get("ground") is not None and r.get("ref_grade")=="sparse_geometry" and
                   r.get("confidence") and r.get("down") and r.get("native_scale") for r in records):
            raise ValueError("Handheld replay needs provider-verified native floor, scale, gravity and LiDAR confidence")
        check = dict(pass_for_handheld_native=True,mounts=sorted({r["mount"] for r in records}),
                     rule="Provider supplied native sensor calibration, valid-depth confidence and per-frame gravity/floor; standing head/chest height acceptance is inapplicable.")
    planes = []
    for j,r in enumerate(records):
        key=(r["seq"],r["i"])
        if key in historical:
            cached=historical[key]
            assert cached["K"]==r["K"] and cached["t"]==r["t"]
            planes.append({**cached,"eye_sample":per_frame_height[key]})
            continue
        f=load_frame(dataset,r,directory)
        reg=core.regions(f,frame_rng(dataset,r,"reference"))
        gp=reg["ground"]
        truth = {b:reg.get("truth_"+b) for b in BANDS}
        target = dict(seq=f.seq,i=f.index,t=f.t,shape=list(f.ref_depth.shape),K=f.K.tolist(),
                      truth=truth, truth_dist={b:reg.get("truth_dist_"+b) for b in BANDS},
                      ground=([gp[0].tolist(),gp[1]] if gp is not None else None),
                      strict_empty=all(v==0 for v in truth.values()),evaluable=gp is not None,
                      eye_sample=per_frame_height[key])
        if gp is not None:
            recovered = evaluator(target["ground"])
            assert np.allclose(np.asarray(recovered["axes"]),np.asarray(reg["axes"]))
        planes.append(target)
        if (j+1)%100==0:
            print("reference geometry",j+1,len(records),flush=True)
    save(directory/"reference.json",dict(dataset=dataset,manifest_sha256=sha(manifest),eye_height=eye,
                                        self_check=check,rows=planes))


def supported(depth, f, reg, remove=None, reference=False):
    h,w = depth.shape
    z = depth.ravel()
    valid = np.isfinite(z) & ((z>.3)&(z<10) if reference else z>.05)
    z = np.where(valid,z,0)
    rays = core.rays(f.K,h,w)
    height,_,planar = core._corridor(rays*z[:,None],valid,reg["axes"],reg["ground"][1])
    if remove is not None:
        planar &= ~remove.ravel()
    area = z**2/(f.K[0,0]*f.K[1,1]*np.linalg.norm(rays,axis=1))
    masks=[]
    for lo,hi in core.BANDS.values():
        masks.append(core._open(planar&(height>=lo)&(height<hi),h,w).reshape(h,w))
    return np.asarray(masks),area.reshape(h,w)


def blind_roi(manifest,seq,index,polygon,polygon_size=None,return_masks=False):
    """Reference-only polygon audit. Coordinates are x,y in polygon_size=[width,height],
    defaulting to the registered reference grid. Does not open predictions or any alarm files.
    Areas are projected frontal m². Distances use horizontal forward, in metres.
    """
    from sanpo_ground_alert_replay_20261118 import evaluator
    manifest=Path(manifest).resolve()
    dataset,rows=inputs(manifest)
    row=next(r for r in rows if r["seq"]==seq and r["i"]==index)
    directory=OUT/sha(manifest)[:16]
    f=load_frame(dataset,row,directory)
    cached=directory/"reference.json"
    if cached.exists():
        ref=next(r for r in read(cached)["rows"] if r["seq"]==seq and r["i"]==index)
        if ref["ground"] is None:
            return {"evaluable":False}
        reg=evaluator(ref["ground"])
    else:
        reg=core.regions(f,frame_rng(dataset,row,"reference"))
        if reg["ground"] is None:
            return {"evaluable":False}
    h,w=f.ref_depth.shape
    xy=np.asarray(polygon,float)
    if polygon_size is not None:
        xy=xy*np.array([w,h])/np.asarray(polygon_size)
    roi=np.zeros((h,w),np.uint8)
    cv2.fillPoly(roi,[np.rint(xy).astype(np.int32)],1)
    z=f.ref_depth.ravel()
    valid=np.isfinite(z)&(z>.3)&(z<10)
    z=np.where(valid,z,0)
    rays=core.rays(f.K,h,w)
    above,along,scope=core._corridor(rays*z[:,None],valid,reg["axes"],reg["ground"][1])
    area=core.pixel_area(f.K,h,w,z)
    result={"evaluable":True,"shape":[h,w],"roi_px":int(roi.sum()),"roi_valid_px":int((valid&roi.ravel().astype(bool)).sum())}
    for name,(lo,hi) in dict(shallow=(.05,.15),**core.BANDS).items():
        raw=scope&roi.ravel().astype(bool)&(above>=lo)&(above<hi)
        def measure(mask):
            m=core._open(mask,h,w)
            result=dict(px=int(m.sum()),area_m2=float(area[m].sum()),
                        forward_min_m=float(along[m].min()) if m.any() else None,
                        forward_p10_m=float(np.percentile(along[m],10)) if m.any() else None,
                        forward_median_m=float(np.median(along[m])) if m.any() else None)
            if return_masks:
                result["mask"]=m.reshape(h,w)
            return result
        result[name]={"near":measure(raw & (along<=1.5)), "full":measure(raw)}
    return result


def packet(path, depth, f, reg, sem, geom, native_ref=None):
    pred, area = supported(depth,f,reg)
    sm = supported(depth,f,reg,sem)[0]
    gm = supported(depth,f,reg,geom)[0]
    ref, refarea = supported(f.ref_depth,f,reg,reference=True)
    masks = np.stack([pred,sm,gm])
    assert not np.any(sm&~gm) and not np.any(gm&~pred)
    alerts = {arm:{b:bool(area[masks[ai,bi]].sum()>=core.MIN_AREA) for bi,b in enumerate(BANDS)}
              for ai,arm in enumerate(ARMS)}
    iu = np.flatnonzero(np.any(pred,axis=0)).astype(np.int32)
    gu = np.flatnonzero(np.any(ref,axis=0)).astype(np.int32)
    native = {}
    if native_ref is not None:
        ns, na = supported(native_ref.ref_depth,native_ref,reg,reference=True)
        ni = np.flatnonzero(ns.any(axis=0)).astype(np.int32)
        native=dict(native_shape=native_ref.ref_depth.shape,
                    native_masks=np.packbits(ns,axis=-1,bitorder="little"),
                    native_indices=ni,native_area_m2=na.ravel()[ni])
    np.savez_compressed(path,shape=depth.shape,
                        pred_masks=np.packbits(masks,axis=-1,bitorder="little"),
                        ref_masks=np.packbits(ref,axis=-1,bitorder="little"),
                        pred_indices=iu,pred_area_m2=area.ravel()[iu],
                        ref_indices=gu,ref_area_m2=refarea.ravel()[gu],**native)
    return alerts


def roi_areas(path, roi):
    """Compute overlap on one visible object. For sparse LiDAR, accept its native or registered RGB grid."""
    with np.load(path,allow_pickle=False) as d:
        h,w=map(int,d["shape"])
        native_roi=None
        if roi.dtype==np.bool_ and "native_shape" in d.files and roi.shape==tuple(d["native_shape"]):
            native_roi=roi
            roi=cv2.resize(roi.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        if roi.dtype!=np.bool_ or roi.shape!=(h,w):
            raise ValueError("Supply a boolean ROI in the reference/prediction pixel grid.")
        p=np.unpackbits(d["pred_masks"],axis=-1,count=w,bitorder="little").astype(bool)
        g=np.unpackbits(d["ref_masks"],axis=-1,count=w,bitorder="little").astype(bool)
        pi,gi=d["pred_indices"],d["ref_indices"]
        pa,ga=d["pred_area_m2"],d["ref_area_m2"]
        def measure(mask,indices,area):
            selected=mask.ravel()[indices]&roi.ravel()[indices]
            return dict(px=int(selected.sum()),area_m2=float(area[selected].sum()))
        result={a:{b:measure(p[ai,bi],pi,pa) for bi,b in enumerate(BANDS)} for ai,a in enumerate(ARMS)}
        result["reference"]={b:measure(g[bi],gi,ga) for bi,b in enumerate(BANDS)}
        if "native_shape" in d.files:
            nh,nw=map(int,d["native_shape"])
            # Select ROI in the native grid by centre sampling; never count upsampled LiDAR hits as new observations.
            roi=native_roi if native_roi is not None else cv2.resize(
                roi.astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
            nm=np.unpackbits(d["native_masks"],axis=-1,count=nw,bitorder="little").astype(bool)
            result["native_reference"]={b:measure(nm[bi],d["native_indices"],d["native_area_m2"]) for bi,b in enumerate(BANDS)}
        return result


def predict(manifest,dataset,records,directory):
    from ground_geometry_filter_20261118 import GroundEvidence,protection
    from sanpo_ground_alert_replay_20261118 import evaluator
    reference_data=read(directory/"reference.json")
    assert sha(manifest)==reference_data["manifest_sha256"] and len(records)==len(reference_data["rows"])
    if not reference_data["self_check"].get("pass",reference_data["self_check"].get("pass_for_handheld_native",False)):
        raise ValueError("Frozen evaluator intake failed. Inspect reference.json before claiming any method result.")
    model,evidence=Methods(),GroundEvidence(device="cuda")
    model.set_unidepth_pixels(315560)
    predictions=[]
    (directory/"supports").mkdir(exist_ok=True)
    checks=set()
    for j,(r,truth) in enumerate(zip(records,reference_data["rows"])):
        f=load_frame(dataset,r,directory,True,model_grid=True)
        assert (f.seq,f.index,f.t)==(truth["seq"],truth["i"],truth["t"])
        f.eye_height=reference_data["eye_height"].get(f.seq)
        # Always advance the scale state in timestamp order, including non-evaluable reference frames.
        z=model.hybrid_out(f)
        source=model.source["hybrid_out"]
        empty={b:None for b in BANDS}
        row=dict(seq=f.seq,i=f.index,t=f.t,scale_source=source,support_path=None,
                 evidence={a:empty.copy() for a in ARMS})
        if truth["evaluable"] and z is not None:
            reg=evaluator(truth["ground"])
            prob=evidence.probability(cv2.cvtColor(f.image,cv2.COLOR_BGR2RGB))
            sem=prob>=.80
            geom=sem&~protection(z,f.K,prob)
            path=Path("supports")/(unikey(f.seq,f.index)+".npz")
            native=load_frame(dataset,r,directory) if r.get("native_scale",1)!=1 else None
            row["evidence"]=packet(directory/path,z,f,reg,sem,geom,native)
            row["support_path"]=path.as_posix()
            if f.seq not in checks:
                masks=[np.zeros_like(sem),sem,geom]
                for a,m in zip(ARMS,masks):
                    assert core.band_alert(np.where(m,np.nan,z),f,reg)==row["evidence"][a]
                checks.add(f.seq)
        predictions.append(row)
        if (j+1)%25==0:
            print("paired model rows",j+1,len(records),flush=True)
    save(directory/"predictions.json",dict(dataset=dataset,manifest_sha256=sha(manifest),arms=ARMS,bands=BANDS,
        coordinate_units="original rectified RGB grid; pixel area in square metres; masks use little-endian np.packbits on width",
        reference_path="reference.json",source_counts=dict(Counter(r["scale_source"] for r in predictions)),
        original_readout_crosschecked_sequences=len(checks),rows=predictions))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["sky","reference","predict"])
    parser.add_argument("manifest",type=Path)
    args=parser.parse_args()
    import torch
    import psutil
    if psutil.virtual_memory().available < 2*1024**3:
        raise RuntimeError("Less than 2 GiB free host RAM")
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    manifest=args.manifest.resolve()
    dataset,records=inputs(manifest)
    directory=OUT/sha(manifest)[:16]
    directory.mkdir(parents=True,exist_ok=True)
    if args.stage=="sky":
        prepare_sky(dataset,records)
    else:
        (reference if args.stage=="reference" else predict)(manifest,dataset,records,directory)
    print(args.stage,len(records),directory,flush=True)
