"""Trace the remaining strict-empty SANPO chest alerts of the consumed hybrid_out run.

Only read the original 138 sessions. Prepare camera-ground frames on CPU in the original
evaluation RNG order. Score only strict false alarms and the preceding scale updates they
actually need. All removals are evaluator-side attribution, never a proposed online method.
"""
import argparse
import gzip
import json
import os
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import cv2
import numpy as np

HOST = Path(__file__).resolve().parent
sys.path.insert(0, str(HOST.parent / "evalkit"))
import core
from methods import Methods

ROOT = HOST.parents[3]
ART = ROOT / "artifacts.local"
RUN = ART / "work/direction-a-confirm-20261117/run"
DATA = ART / "downloads/sanpo-real-chest-seg-20261117"
OUT = ART / "work/sidewall-attribution-20261118/sanpo"
ARM = "hybrid_out"
WALK = [1, 2, 3, 5, 6, 17, 30]


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=1), encoding="utf-8")


def read(name):
    return json.loads(name.read_bytes())


@lru_cache(maxsize=150)
def camera(seq):
    spec = read(DATA / seq / "description.json")
    p = spec["session_camera_details"][spec["session_camera_location"].index("camera_chest")]["left_camera_params"]
    sc = 1280 / p["image_width"]
    return np.array([[p["fx"]*sc, 0, p["cx"]*sc], [0, p["fy"]*sc, p["cy"]*sc], [0, 0, 1.]])


def frame(row, rgb=False):
    seq = row["seq"]
    index = int(round(row["t"] * 15))
    base = DATA / seq / "camera_chest/left"
    name = f"{index:06d}"
    a = np.frombuffer(gzip.decompress((base / "depth_maps" / (name+".float16.gz")).read_bytes()), "<f2")
    ref = a[2:].reshape(int(a[0]), int(a[1])).astype(np.float32)
    labels = cv2.resize(cv2.imread(str(base / "segmentation_masks" / (name+".png")), -1)[:, :, 2],
                        (1280, 720), interpolation=cv2.INTER_NEAREST)
    assert ref.shape == labels.shape
    ref[~np.isfinite(ref) | (labels == 27)] = np.nan
    image = cv2.resize(cv2.imread(str(base / "video_frames" / (name+".png"))), (1280, 720),
                       interpolation=cv2.INTER_AREA) if rgb else None
    f = core.Frame("sanpo_chest_seg", seq, index, index/15, image, camera(seq), ref,
                   "pseudo_ground_truth", None, None, "chest", block=seq)
    return f, labels


def plan(rows):
    target = []
    anchors = set()
    last = {}
    for i, row in enumerate(rows):
        source = row["source"][ARM]
        if source in ("tof", "floor"):
            last[row["seq"]] = i
        if set(row["truth"].values()) == {0} and any(row["alert"][ARM].values()):
            target.append(i)
            if source == "last":
                anchors.add(last[row["seq"]])
    return target, sorted(anchors - set(target))


def prepare(rows):
    target, anchors = plan(rows)
    need = set(target)
    rng = np.random.default_rng(20261117)
    k = max(1, len(rows)//20)
    samples = [frame(rows[i])[0] for i in list(range(0, len(rows), k))[:20]]
    check = core.self_check(samples, rng)
    original = read(RUN / "report.json")["sanpo_chest_seg"]["self_check"]
    assert np.isclose(check["sensor_height_m"], original["sensor_height_m"])
    del samples
    geometry = {}
    for i, row in enumerate(rows):
        f, _ = frame(row)
        if i in need:
            reg = core.regions(f, rng)
            assert {band: reg["truth_"+band] for band in core.BANDS} == row["truth"], i
            n,h = reg["ground"]
            geometry[i] = [n.tolist(), float(h)]
        else:
            # Identical draws to core.ground_plane. In these chest frames down=[0,1,0].
            z = f.ref_depth
            ys = (np.arange(z.shape[0], dtype=float)[:, None]-f.K[1, 2])/f.K[1, 1]
            below = np.count_nonzero((z > .3) & (z < 10) & (ys*z > .5))
            if below >= 2000:
                n = min(30000, below)
                rng.choice(below, n, replace=False)
                for _ in range(300):
                    rng.choice(n, 3, replace=False)
        if (i+1)%400==0: print("reference frames",i+1,flush=True)
    save("ground.json", {"target":target, "anchors":anchors, "planes":geometry})


def directions(K, shape, plane):
    n, h0 = np.asarray(plane[0]), plane[1]
    down = -n
    forward = np.array([0.,0.,1.]) - down * down[2]
    forward /= np.linalg.norm(forward)
    right = np.cross(down,forward)
    rays = core.rays(K, *shape)
    return rays @ right, rays @ forward, rays @ down, np.linalg.norm(rays,axis=1), h0


def attribution(f, labels, pred, plane):
    shape=f.ref_depth.shape
    dx,dz,dy,norm,h0 = directions(f.K,shape,plane)
    p=pred.ravel(); z=f.ref_depth.ravel(); s=labels.ravel()
    h=h0-p*dy; forward=p*dz
    footprint=np.isfinite(p)&(p>.05)&(np.abs(p*dx)<=core.X_HALF)&(forward>=core.Z_NEAR)&(forward<=core.Z_FAR)
    area=p**2 / (f.K[0,0]*f.K[1,1]*norm)
    ref_valid=(z>.3)&(z<10)&np.isfinite(z)
    side=ref_valid & (np.abs(z*dx)>core.X_HALF)
    wall=s==8; building=s==7; maybe=(s==0)|wall|building
    fwd_ok=(z*dz>=core.Z_NEAR)&(z*dz<=core.Z_FAR)
    ref_h=h0-z*dy
    groups={"wall_fence_lateral":wall&side,"building_lateral":building&side,
            "wall_or_building_lateral":(wall|building)&side,
            "wall_fence_anywhere":wall, "any_lateral":side,
            "permissive_wall": maybe & (side|~ref_valid),
            "walkable_semantic":np.isin(s,WALK),
            "reference_below_low":ref_valid & (ref_h<core.BANDS["LOW"][0]),
            "reference_invalid":~ref_valid,"reference_beyond_forward":ref_valid&(z*dz>core.Z_FAR)}
    base={}; suppressed={k:{} for k in [*groups,"wall_fence_lateral_only"]}
    stats={}; visual=np.zeros(shape,bool)
    for b,(lo,hi) in core.BANDS.items():
        raw=footprint&(h>=lo)&(h<hi)
        opened=core._open(raw,*shape)
        base[b]=bool(area[opened].sum()>=core.MIN_AREA)
        if not base[b]:
            continue
        visual|=opened.reshape(shape)
        denom=float(area[opened].sum())
        tally=np.bincount(s[opened],weights=area[opened],minlength=31)
        stats[b]=dict(area=denom,area_by_class={int(k):round(float(tally[k]),7) for k in np.flatnonzero(tally)},
                      reference_lateral=float(area[opened&side].sum()),
                      reference_lateral_same_band=float(area[opened&side&fwd_ok&(ref_h>=lo)&(ref_h<hi)].sum()),
                      reference_below_low=float(area[opened&groups["reference_below_low"]].sum()),
                      reference_invalid=float(area[opened&~ref_valid].sum()))
        same=groups["wall_fence_lateral"]&fwd_ok&(ref_h>=lo)&(ref_h<hi)
        for key,remove in [*groups.items(),("wall_fence_lateral_only",same)]:
            # Remove evidence before the same 3x3 opening used by the original reader.
            remaining=core._open(raw&~remove,*shape)
            suppressed[key][b]=bool(area[remaining].sum()<core.MIN_AREA)
    return base,stats,{k:bool(v) and all(v.values()) for k,v in suppressed.items()}, visual


def score(rows):
    prep=read(OUT/"ground.json")
    target=set(prep["target"]); anchors=set(prep["anchors"])
    report=read(RUN/"report.json")["sanpo_chest_seg"]
    model=Methods(); model.set_unidepth_pixels(315560)
    result=[]; tiles=[]
    for ct,i in enumerate(sorted(target|anchors),1):
        row=rows[i]; f,labels=frame(row,True); f.eye_height=report["eye_heights"][f.seq]
        pred=model.hybrid_out(f)
        assert model.source[ARM]==row["source"][ARM], (i,model.source[ARM],row["source"][ARM])
        if i in anchors: continue
        alerts,stats,removed,visual=attribution(f,labels,pred,prep["planes"][str(i)])
        z=f.ref_depth.ravel()
        near=core.depth_metrics(pred,f.ref_depth,np.isfinite(z)&(z>=.5)&(z<=3.))
        stored=row["depth"][ARM]["near"]
        error_difference=(abs(near["absrel"]-stored["absrel"]) if near and stored and
                          near["absrel"] is not None and stored["absrel"] is not None else None)
        result.append(dict(row=i,seq=f.seq,frame=f.index,source=model.source[ARM],original=row["alert"][ARM],
                           recovered=alerts,match=alerts==row["alert"][ARM],removed=removed,bands=stats,
                           near_absrel_diff=error_difference))
        # Leave original RGB readable; add the outlines of predicted alert support only.
        draw=f.image.copy()
        contour,_=cv2.findContours(visual.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(draw,contour,-1,(0,240,250),3)
        tile=cv2.resize(draw,(248,140)); cv2.rectangle(tile,(0,0),(84,19),(0,0,0),-1)
        cv2.putText(tile,str(i),(3,14),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
        tiles.append(tile)
        if len(result)%20==0: print("target reprojections",len(result),flush=True)
    for n in range(0,len(tiles),20):
        part=tiles[n:n+20];part+= [np.zeros_like(part[0])]*((-len(part))%4)
        cv2.imwrite(str(OUT/f"support_{n:03d}.jpg"),np.vstack([np.hstack(part[k:k+4]) for k in range(0,len(part),4)]))
    save("attribution.json",result)
    summarize(rows,result)


def summarize(rows,result):
    target,anchors=plan(rows)
    valid=[r for r in result if r["match"]]
    denominator=Counter(r["source"][ARM] for r in rows if set(r["truth"].values())=={0})
    errors=[r["near_absrel_diff"] for r in result if r["near_absrel_diff"] is not None]
    wall_any=0
    wall_majority=0
    for r in valid:
        total=sum(b["area"] for b in r["bands"].values())
        wall=sum(b["area_by_class"].get("8",b["area_by_class"].get(8,0)) for b in r["bands"].values())
        wall_any+=wall>0
        wall_majority+=wall>total/2
    summary=dict(source_run=str(RUN.relative_to(ROOT)),strict_negative=sum(denominator.values()),by_source=denominator,
                 strict_false=len(target),false_sessions=len({r["seq"] for r in result}),matched=len(valid),
                 support_only_forwards=len(anchors),false_by_source=Counter(r["source"] for r in result),
                 band_positives_among_false=Counter(b for r in result for b,v in r["original"].items() if v),
                 pixel_removal_suppresses_all_bands={k:sum(r["removed"][k] for r in valid) for k in result[0]["removed"]},
                 any_opened_wall_fence_area=wall_any,majority_opened_wall_fence_area=wall_majority,
                 both_ground_height_and_walkable_semantic=sum(r["removed"]["reference_below_low"] and
                     r["removed"]["walkable_semantic"] for r in valid),
                 ground_height_and_no_current_scale=sum(r["removed"]["reference_below_low"] and
                     r["source"] in ("none","last") for r in valid),
                 near_absrel_rows=len(errors),max_absrel_difference=max(errors,default=None),
                 failed_rows=[r["row"] for r in result if not r["match"]])
    save("summary.json",summary);print(json.dumps(summary,indent=1))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["prepare","score","summary"]);args=p.parse_args()
    rows=read(RUN/"rows_sanpo_chest_seg.json")
    if args.mode=="summary":
        summarize(rows,read(OUT/"attribution.json"))
    else:
        (prepare if args.mode=="prepare" else score)(rows)
