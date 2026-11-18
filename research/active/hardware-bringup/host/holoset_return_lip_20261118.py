"""Compare the pale band's returns and its visible end with the measured right wall in HoloSet seq4."""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from holoset_tube_range_20261118 import project
from real_wall_holoset_score_20261118 import Recording, records


WORK = Path("artifacts.local/work")
ROOT = WORK / "holoset-return-20261118"
SEQ = "research-center-seq4"
# Leftmost dark mark on the rounded end, in native 760 x 428 images.
TIPS = {
    133070611251923116: (600, 11),
    133070611255255089: (533, 7),
    133070611257920667: (527, 9),
    133070611259919851: (533, 22),
    133070611261919034: (521, 27),
    133070611263918218: (508, 29),
    133070611267916585: (510, 17),
}


def returns(cap, row):
    dt, rt = row["depth_ts"], row["rgb_ts"]
    depth = cv2.imread(str(cap.directory / "depth" / f"{dt}.pgm"), cv2.IMREAD_UNCHANGED)
    ok = cap.valid & (depth > 0)
    sensor = cap.rig[dt] @ cap.extr
    world = (cap.native_rays[ok] * (depth[ok, None]*.001)) @ sensor[:3, :3].T + sensor[:3, 3]
    u, v, _, ok = project(cap, rt, world)
    return world[ok], u[ok], v[ok]


def wall_fit(patches, mask):
    xyz = np.concatenate([p for p, keep in zip(patches, mask) if keep])
    origin = xyz.mean(0)
    xy = (xyz-origin)[:, [0, 2]]
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    n = vt[-1]
    fit = least_squares(lambda par: xy @ np.array([np.cos(par[0]), np.sin(par[0])])-par[1],
                        [np.arctan2(n[1], n[0]), 0.], loss="soft_l1", f_scale=.005)
    n = np.array([np.cos(fit.x[0]), 0., np.sin(fit.x[0])])
    return n, float(origin@n + fit.x[1])


def reprojection(par, cap, keys):
    delta = []
    for t in keys:
        fx, fy, pose = cap.pv[t]
        a = (par-pose[:3, 3]) @ pose[:3, :3]
        uv = np.array([cap.cx-fx*a[0]/a[2], cap.cy+fy*a[1]/a[2]])
        delta.extend(uv-TIPS[t])
    return np.array(delta)


def quant(values):
    return np.round(np.percentile(values,[10,50,90]), 4).tolist() if len(values) else []


def compare_shell(fits, world):
    offsets = []
    for fit in fits:
        x, y, z = ((world-fit["world_origin"]) @ fit["world_axes"]).T
        h, f, sh, sf, bend = fit["par"]
        radial = np.hypot(y-(h+sh*x+bend*x*x), z-(f+sf*x))
        span = fit["observed_x"]
        offsets.append(dict(radius=fit["radius"], form=fit["form"], parity=fit["fit_parity"],
                            median_radial_minus_radius_m=float(np.median(radial)-fit["radius"]),
                            n_within_observed_span=int(np.sum((x>=span[0]) & (x<=span[1])))))
    return offsets


def main():
    cap = Recording(WORK / "real-wall-search-20261118/head" / SEQ)
    rows = json.loads((WORK / "holoset-tube-20261118/masks/histories.json").read_bytes())[SEQ]
    measurements = [returns(cap,r) for r in rows]
    patch = []
    for i in [7,8,9]:
        world, u, v = measurements[i]
        patch.append(world[(u>=558)&(u<=645)&(v>=90)&(v<=180)])
    n, d = wall_fit(patch, [True,False,True])
    if cap.pv[rows[7]["rgb_ts"]][2][:3,3]@n < d:
        n, d = -n, -d
    wall = dict(normal_xyz=n.tolist(), intercept_m=d, counts=[len(x) for x in patch],
                signed_p10_median_p90_m=[quant(x@n-d) for x in patch])
    keys=list(TIPS)
    # Initialize on the independent wall; compare an unconstrained common landmark
    # and a landmark constrained to that wall under the same camera model.
    fx, fy, pose=cap.pv[keys[0]]
    a, b = TIPS[keys[0]]
    ray = pose[:3,:3] @ np.array([(a-cap.cx)/fx,(cap.cy-b)/fy,-1])
    x0 = pose[:3,3] + ray*(d-pose[:3,3]@n)/(ray@n)
    cross = np.cross(n, [0,1,0])
    def fit_tip(train):
        out=least_squares(lambda xyz: reprojection(xyz, cap, train),x0)
        point=out.x
        planar = least_squares(lambda q: reprojection(x0+q[0]*cross+np.array([0,q[1],0]), cap, train), [0,0])
        onwall=x0+planar.x[0]*cross+np.array([0,planar.x[1],0])
        return dict(signed_wall_offset_m=float(point@n-d), point=point.tolist(),
                    errors_px=np.round(np.linalg.norm(reprojection(point,cap,keys).reshape(-1,2),axis=1),2).tolist(),
                    wall_errors_px=np.round(np.linalg.norm(reprojection(onwall,cap,keys).reshape(-1,2),axis=1),2).tolist())
    tip=dict(clicks={str(k):v for k,v in TIPS.items()}, baseline_m=float(np.linalg.norm(cap.pv[keys[-1]][2][:3,3]-cap.pv[keys[0]][2][:3,3])),
             all_views=fit_tip(keys), leave_one_out=[fit_tip([k for k in keys if k!=held]) for held in keys])
    floors = {r["depth_ts"]: r.get("floor_y") for r in records(cap.directory/"geometry.jsonl")}
    fits = json.loads((WORK/"holoset-tube-20261118/geometry/lower-rim.json").read_bytes())[SEQ+"-extended"]
    result=dict(wall=wall, tip=tip, rows=[], modal_surface=[])
    native_labels = [cv2.imread(str(ROOT/"scene/labels"/f"{r['rgb_ts']}.png"),cv2.IMREAD_GRAYSCALE) for r in rows]
    if any(lab is None or lab.shape!=(cap.height, cap.width) for lab in native_labels):
        raise ValueError("Incomplete frozen scene labels")
    cluster = []
    for i, row in enumerate(rows):
        rt=row["rgb_ts"]; world,u,v=measurements[i]
        labels=native_labels[i]
        for erode in [0,3,7]:
            lip=(labels==15).astype(np.uint8)
            if erode:
                lip=cv2.erode(lip,np.ones((erode*2+1,erode*2+1),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
            pts=world[lip[v,u]>0]
            delta=pts@n-d
            result["rows"].append(dict(index=i,rgb_ts=rt,erosion_px=erode,area=int(lip.sum()),returns=len(pts),
                                       within_3cm=int((abs(delta)<=.03).sum()),
                                       corridor_side_over_10cm=int((delta>.10).sum()),signed_p10_median_p90_m=quant(delta)))
        if i not in (7,8,9):
            continue
        # Describe the observed ~0.60 m foreground mode and keep the original
        # all-return summaries above, including mixed or missing early returns.
        offset=world@n-d
        mode=(labels[v,u]==15) & (offset>.50) & (offset<.70)
        pts=world[mode]
        cluster.append(pts)
        gap=cv2.distanceTransform((labels!=1).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        reproject=[]
        for j in range(4,10):
            if i==j:
                continue
            uj,vj,_,ok=project(cap,rows[j]["rgb_ts"],pts)
            tags=native_labels[j][vj[ok],uj[ok]]
            reproject.append(dict(index=j,visible=int(ok.sum()),same_lip=int((tags==15).sum()),
                                  upper_surface=int((tags==17).sum()),tube=int((tags==1).sum())))
        target=[]
        for j in (8,9):
            pose=cap.pv[rows[j]["rgb_ts"]][2]
            fwd=-pose[:3,2].copy()
            fwd[1]=0.
            fwd/=np.linalg.norm(fwd)
            right=np.cross(fwd,[0.,1.,0.])
            rel=pts-pose[:3,3]
            target.append(dict(index=j,right_of_gaze_m=quant(rel@right),forward_m=quant(rel@fwd),
                               height_m=quant(pts[:,1]-floors[rows[j]["depth_ts"]])))
        result["modal_surface"].append(dict(index=i,returns=int(mode.sum()),u_px=quant(u[mode]),
                                            gap_to_tube_px=quant(gap[v[mode],u[mode]]),cross_view=reproject,
                                            target_pose=target,original_shell=compare_shell(fits,pts)))
    axes=np.column_stack((n,np.array([0.,1.,0.]),np.cross([0,1,0],n)))
    result["mode_span_wall_normal_vertical_tangent_m"]=np.ptp(np.concatenate(cluster)@axes,axis=0).tolist()
    out=ROOT/"lip/geometry.json"; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=1,allow_nan=False),encoding="utf-8")
    print(wall,tip["baseline_m"],tip["all_views"],[(r["index"],r["returns"],r["signed_p10_median_p90_m"]) for r in result["rows"] if r["erosion_px"]==0])


if __name__=="__main__":
    main()
