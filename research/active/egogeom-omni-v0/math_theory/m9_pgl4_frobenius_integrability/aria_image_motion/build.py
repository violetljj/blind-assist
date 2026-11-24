"""Archive label-free, image-verified candidate features for two frozen Aria streams."""
import os
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key]="1"

from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/"aria_alert_support"
sys.path.insert(0,str(OLD))
from predict import ART, dump, sha
from support import axes, world_rays
from motion import gray_frame, image_evidence, motion_channels

OUT = ART/"math_theory"/HERE.parent.name/HERE.name
PACKED = ART/"zeroshot_eval/aria_g2_20clips"
OLD_OUT = OUT.parent/OLD.name
GEOM = ("range_m","views","min_age","log_hits","mean_abs_lateral","height","height2",
        "face","delta_height","uncertainty","geometric_parallax")
VIS = ("gray","texture","valid_flow","reliable_flow","motion_match","vs_infinity","flow_magnitude")


def new_view(obs, i, image, delta):
    depth, K, R, t = (obs[k][i] for k in ("depth_s4","K","poses_R","poses_t"))
    fw, side = axes(R,obs["strict_fwd"][i])
    points = world_rays(depth.shape,K,R)*depth.ravel()[:,None]
    along,lat = points[:,:2]@fw, points[:,:2]@side
    valid = np.isfinite(depth.ravel()) & (depth.ravel()>.25) & (depth.ravel()<10.)
    ground_pixels = valid & (np.abs(lat)<=1.2) & (along>=.8) & (along<=4.5)
    ground = np.percentile(points[ground_pixels,2],8) if ground_pixels.sum()>=40 else -1.50
    h = points[:,2]-ground
    keep = valid & (h>=.15) & (h<=1.90)
    geom_motion, appearance = motion_channels(image,depth,K,R,delta)
    physical = np.column_stack((h,h*h,1.-obs["nrm_s4"][i,1].ravel()**2,
                                obs["dh_s4"][i].ravel(),obs["ls_s4"][i].ravel(),geom_motion.ravel()))
    features = np.column_stack((physical,appearance.reshape(len(h),-1),
                                appearance[:,::-1].reshape(len(h),-1)))
    return points[keep,:2]+t[:2], features[keep]


def candidates(history,t,fw,side):
    size = 81
    ncols = len(GEOM)+2*len(VIS)
    totals = np.zeros((size,ncols),np.float64)
    views = np.zeros(size,int)
    first_age = np.full(size,6)
    for age,(coords,features) in enumerate(reversed(history)):
        delta = coords-t[:2]
        along, lat = delta@fw, delta@side
        idx = np.flatnonzero((np.abs(lat)<=.30)&(along>=.30)&(along<=4.))
        idx = idx[np.argsort(along[idx],kind="stable")]
        z = along[idx]
        end = np.searchsorted(z,z+.30,side="right")
        starts = np.flatnonzero(end-np.arange(len(z))>=18)
        if not len(starts):
            continue
        cells, ix = np.unique(np.floor(z[starts]/.05+.5).astype(int),return_index=True)
        start = starts[ix]
        stop = end[start]
        count = stop-start
        columns = np.column_stack((np.abs(lat[idx]),features[idx]))
        prefix = np.vstack((np.zeros(columns.shape[1]),columns.cumsum(0)))
        totals[cells,3] += np.log1p(count)
        totals[cells,4:] += (prefix[stop]-prefix[start])/count[:,None]
        views[cells] += 1
        first_age[cells] = np.minimum(first_age[cells],age)
    cells = np.flatnonzero(views)
    out = totals[cells]/views[cells,None]
    out[:,:3] = np.column_stack((cells*.05,views[cells],first_age[cells]))
    assert np.isfinite(out).all()
    return out.astype(np.float32)


def build(path):
    with np.load(path) as f:
        hw,intr,blob,offs = [f[k] for k in ("orig_hw","K_orig","rgb_blob","rgb_offsets")]
    streams, source_hash = {}, {}
    for ver in ("v4","v41"):
        source = ART/"math_theory"/(ver+"_stream")/"aria_gen2"/path.name
        with np.load(source) as f:
            streams[ver] = {k:f[k] for k in ("K","strict_fwd","poses_R","poses_t","gray_s4","depth_s4",
                                             "dh_s4","ls_s4","nrm_s4")}
        source_hash[ver] = sha(source)
    v = streams["v4"]
    for k in ("K","poses_R","poses_t","strict_fwd","gray_s4"):
        np.testing.assert_array_equal(v[k],streams["v41"][k])
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    past = {ver:deque(maxlen=6) for ver in streams}
    rows = {ver:[] for ver in streams}
    gray = None
    for i in range(len(offs)-1):
        K,R,t = (v[k][i] for k in ("K","poses_R","poses_t"))
        old = max(0,i-1)
        current = gray_frame(blob[offs[i]:offs[i+1]],hw,intr,K,v["gray_s4"][i])
        image = image_evidence(current,gray,K,v["K"][old],R,v["poses_R"][old],flow)
        fw,side = axes(R,v["strict_fwd"][i])
        for ver,obs in streams.items():
            past[ver].append(new_view(obs,i,image,t-v["poses_t"][old]))
            rows[ver].append(candidates(past[ver],t,fw,side))
        gray = current
    entry = {"name":path.stem,"packed":sha(path),"streams":source_hash,"features":{}}
    for ver,frames in rows.items():
        baseline = OLD_OUT/ver/"selector_predictions"/path.name
        with np.load(baseline) as f:
            nearest = f["nearest"]
        ours = np.array([x[0,0] if len(x) else np.inf for x in frames])
        np.testing.assert_allclose(ours,nearest,rtol=0,atol=2.e-7)
        dest = OUT/ver/path.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,features=np.concatenate(frames),offsets=np.cumsum([0]+[len(x) for x in frames]))
        entry["features"][ver] = dict(sha256=sha(dest),baseline=sha(baseline),candidates=sum(map(len,frames)))
    return entry


def main():
    cv2.setNumThreads(1)
    files = sorted(PACKED.glob("aria_walk_*.npz"))
    assert len(files)==19
    ledger = []
    for p in files:
        ledger.append(build(p))
        print(p.stem,flush=True)
    dump(OUT/"features.json",dict(files=ledger,geom=GEOM,visual=VIS,
         source={p.name:sha(p) for p in (HERE/"PROTOCOL.md",HERE/"motion.py",HERE/"build.py",OLD/"support.py")},
         bridge="both backbones, archived nearest distances and finite masks"))


if __name__=="__main__":
    main()
