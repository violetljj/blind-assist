"""Estimate tube geometry from two passes, in their separate recording coordinates.

The silhouette-only check retains out-of-view and unbounded candidate depths.
A separate conditional fit projects straight and curved circular tubes into
alternating exposures, checks the withheld rims and intersects the wearer path.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from real_wall_holoset_score_20261118 import Recording, records


WORK = Path("artifacts.local/work")
HEAD = WORK / "real-wall-search-20261118/head"
ROOT = WORK / "holoset-tube-20261118"
UP = np.array([0., 1., 0.])
BINS = np.unique(np.r_[np.arange(0, 6, .02), np.geomspace(6, 500, 180), np.inf])
BOX_LO = np.array([-.25, .30, .60])
BOX_HI = np.array([.25, 2., 1.90])
# Native PV points scoping the isolated lower rim on additional exposures.
# Only endpoint-marked interior spans enter the independent long-baseline fits.
EXTRA = {
    "research-center-seq3": [
        (133070576845030618, 390, 13, 424, 21),
        (133070576868687631, 599, 13, 658, 31),
        (133070576891345052, 364, 9, 436, 19),
        (133070576903673354, 325, 15, 401, 32),
        (133070576912669682, 313, 18, 391, 35),
        (133070576925331182, 323, 30, 409, 46),
        (133070576939325470, 284, 37, 382, 57),
        (133070576953319760, 308, 24, 425, 56),
        (133070576963315680, 331, 18, 479, 51),
        (133070577005298546, 407, 8, 723, 121)],
    "research-center-seq4": [
        (133070611267916585, 55, 113, 466, 39),
        (133070611269915768, 75, 87, 417, 19),
        (133070611271581754, 38, 73, 344, 12),
        (133070611274247332, 32, 47, 202, 8),
        (133070611275580122, 38, 51, 207, 11),
        (133070611277912502, 16, 40, 106, 8)],
}

# The whole cross-corridor segment and both ends are visible through the open
# doorway in these five earlier exposures. Rectangles deliberately include the
# red fixtures and nearby background around the segmented material.
FAR_BOXES = [
    (133070576749402976, 342, 0, 407, 41),
    (133070576763397265, 364, 0, 438, 41),
    (133070576781723120, 356, 16, 438, 64),
    (133070576787054278, 341, 8, 422, 61),
    (133070576801048567, 345, 0, 434, 56),
]


def additional_edges(cap, seq):
    rows, tiles = [], []
    for ts,x1,y1,x2,y2 in EXTRA[seq]:
        image = cv2.imread(str(cap.directory / "rgb" / f"{ts}.png"))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
        gradient = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        sample = []
        for x in np.linspace(x1, x2, 16).round().astype(int):
            y = int(np.rint(y1+(x-x1)*(y2-y1)/(x2-x1)))
            ys = np.arange(max(2, y-7), y+8)
            scores = np.median(gradient[ys, x-2:x+3], axis=1)
            sample.append([int(x), int(ys[np.argmax(scores)])])
        rows.append(dict(rgb_ts=ts, edge=sample))
        displayed = image[:155].copy()
        for x,y in sample:
            cv2.circle(displayed, (x,y+4), 1, (255, 30, 255), -1)
        cv2.putText(displayed, str(ts), (3,145), 0, .45, (255,0,255), 1)
        tiles.append(displayed)
    cv2.imwrite(str(ROOT / f"geometry/edges-{seq}.jpg"), np.vstack(tiles))
    return rows


def mask(seq, kind, ts):
    image = cv2.imread(str(ROOT / "masks" / kind / seq / f"{ts}.png"), 0)
    if image is None or image.shape != (428, 760):
        raise ValueError(f"Invalid {kind} at {ts}")
    return image != 0


def witnesses(core):
    pts = []
    for x in range(4, core.shape[1], 10):
        ys = np.flatnonzero(core[:, x])
        if len(ys):
            for y in np.unique(np.quantile(ys, [0, .33, .67, 1.]).round().astype(int)):
                if core[y, x] and y >= 3:
                    pts.append((x, y))
    return np.array(pts, dtype=float).reshape(-1, 2)


def exclusion_slack(cap, source, rays, other, distance):
    """Certified outside-distance lower bound in pixels for a continuous z bin.

    A positive-depth 3-D segment projects to a 2-D segment. Where either end
    is offscreen, or the segment crosses the camera plane, keep it as unknown.
    Distance to the foreground is 1-Lipschitz. A whole segment lies outside
    the dilated mask if distance at an endpoint exceeds its length + slack.
    Infinity is handled via the exact vanishing point.
    """
    fx, fy, q = other
    a = (source - q[:3, 3]) @ q[:3, :3]
    b = rays @ q[:3, :3]
    loc = a + b[:, None] * np.r_[BINS[:-1], 0.][None, :, None]
    loc[:, -1] = b
    z = -loc[:, :, 2]
    valid = z > 1e-6
    u = cap.cx + fx * loc[:, :, 0] / np.maximum(z, 1e-6)
    v = cap.cy - fy * loc[:, :, 1] / np.maximum(z, 1e-6)
    valid &= (u >= .5) & (u < cap.width - 1.5) & (v >= .5) & (v < cap.height - 1.5)
    duv = np.hypot(np.diff(u, axis=1), np.diff(v, axis=1))
    d = distance[np.rint(v.clip(0, cap.height-1)).astype(int),
                 np.rint(u.clip(0, cap.width-1)).astype(int)]
    bound = np.maximum(d[:, :-1], d[:, 1:]) - duv - 1.
    return np.where(valid[:, :-1] & valid[:, 1:], bound, -np.inf)


def compute(cap, seq, context, far_boxes=()):
    stamps = [r["rgb_ts"] for r in context]
    outer = [mask(seq, "outer", t) for t in stamps]
    reference_stamps = stamps.copy()
    for stamp, left, top, right, bottom in far_boxes:
        box = np.zeros((cap.height, cap.width), bool)
        box[top:bottom+1, left:right+1] = True
        outer.append(box)
        stamps.append(stamp)
    distances = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                 for m in outer]
    tracks = []
    for stamp in reference_stamps:
        fx, fy, pose = cap.pv[stamp]
        origin = pose[:3, 3]
        uv = witnesses(mask(seq, "core", stamp))
        if not len(uv):
            continue
        local = np.column_stack([(uv[:, 0]-cap.cx)/fx, -(uv[:, 1]-cap.cy)/fy, -np.ones(len(uv))])
        rays = local @ pose[:3, :3].T
        bounds = np.full((2, len(uv), len(BINS)-1), -np.inf)
        for k, other in enumerate(stamps):
            qp = cap.pv[other]
            if not outer[k].any() or np.linalg.norm(qp[2][:3, 3]-origin) < .25:
                continue
            bounds[k % 2] = np.maximum(bounds[k % 2], exclusion_slack(cap, origin, rays, qp, distances[k]))
        tracks.append((stamp, origin, uv, rays, bounds))
    return tracks


def interval_summary(track, mode, padding):
    ts, origin, uv, rays, bounds = track
    bound = np.max(bounds, axis=0) if mode == "all" else bounds[mode]
    ok = bound <= padding
    finite = ok.any(axis=1) & ~ok[:, -1]
    idx = np.flatnonzero(finite)
    low = BINS[ok[idx].argmax(axis=1)]
    high = BINS[len(BINS)-1-ok[idx, ::-1].argmax(axis=1)]
    return idx, low, high, (~ok.any(axis=1)).sum(), ok[:, -1].sum()


def analyze(cap, cases, floors, tracks):
    results = []
    for row in cases:
        pose = cap.pv[row["rgb_ts"]][2]
        fwd = -pose[:3, 2].copy()
        fwd[1] = 0
        fwd /= np.linalg.norm(fwd)
        basis = np.column_stack([np.cross(fwd, UP), fwd, UP])
        box_origin = pose[:3, 3].copy()
        box_origin[1] = floors[row["depth_ts"]]
        for mode in ["all", 0, 1]:
            for padding in [3, 8, 15]:
                best = None
                counts = dict(tested=0, empty=0, infinite=0, finite=0, inside=0)
                for track in tracks:
                    ts, origin, uv, rays, _ = track
                    i, low, high, empty, infinite = interval_summary(track, mode, padding)
                    counts["tested"] += len(uv)
                    counts["empty"] += int(empty)
                    counts["infinite"] += int(infinite)
                    counts["finite"] += len(i)
                    if not len(i):
                        continue
                    xyz_lo = origin + low[:, None]*rays[i]
                    xyz_hi = origin + high[:, None]*rays[i]
                    a, b = (xyz_lo-box_origin) @ basis, (xyz_hi-box_origin) @ basis
                    lo, hi = np.minimum(a, b), np.maximum(a, b)
                    margins = np.minimum(lo-BOX_LO, BOX_HI-hi).min(axis=1)
                    counts["inside"] += int((margins>0).sum())
                    k = np.argmax(margins)
                    if best is None or margins[k] > best["margin_m"]:
                        best = dict(rgb_ts=ts, uv=uv[i[k]].tolist(), margin_m=float(margins[k]),
                                    optical_z_m=[low[k], high[k]], low_lateral_forward_height=lo[k].tolist(),
                                    high_lateral_forward_height=hi[k].tolist(),
                                    endpoints_world_m=[xyz_lo[k].tolist(), xyz_hi[k].tolist()])
                results.append(dict(case=row["case"], mode=mode, padding_px=padding, counts=counts, best=best))
    return results


def cylinder(cap, cases, floors, outlines):
    """Independent image-only conditional fit to the long visible lower rim.

    Parameterize a vertical/forward circular section swept along the lateral
    coordinate. Allow the centre to sag quadratically, and compare to a straight
    axis. The radius is profiled because the upper rim is truncated in the PV.
    """
    row = cases[0]
    reference = cap.pv[row["rgb_ts"]][2]
    fwd = -reference[:3, 2].copy()
    fwd[1] = 0.
    fwd /= np.linalg.norm(fwd)
    axes = np.column_stack([np.cross(fwd, UP), UP, fwd])
    origin = reference[:3, 3].copy()
    origin[1] = floors[row["depth_ts"]]
    phis = np.linspace(0, np.pi*2, 49)[:-1]
    points, observations, frames, cameras = [], [], [], []
    for j, record in enumerate(outlines):
        if "edge" in record:
            uv = record["edge"]
        else:
            contour = np.asarray(record["outline"], dtype=np.int32)
            if not len(contour):
                continue
            binary = np.zeros((428, 760), np.uint8)
            cv2.fillPoly(binary, [contour], 255)
            left, right = contour[:, 0].min(), contour[:, 0].max()
            if right - left < 100:
                continue
            uv = []
            for u in np.rint(left+(right-left)*np.linspace(.10, .92, 12)).astype(int):
                vs = np.flatnonzero(binary[:, u])
                if len(vs) and vs[-1] >= 4:
                    uv.append((u, vs[-1]))
        fx, fy, pose = cap.pv[record["rgb_ts"]]
        # In PV coordinates: camera X, camera Y, positive optical Z.
        transform = np.column_stack([pose[:3, 0], pose[:3, 1], -pose[:3, 2]])
        camera = np.vstack([axes.T@transform, (origin-pose[:3, 3])@transform])
        for u,v in uv:
            cameras.append(camera)
            points.append(camera[:, 0]*fx - (u-cap.cx)*camera[:, 2])
            observations.append((v, fy))
            frames.append(j)
    points, cameras, observations = map(np.array, [points, cameras, observations])
    obs_v, fy = observations.T
    frames = np.array(frames)

    def project(par, radius, details=False):
        height, forward, sh, sf, bend = par
        cy = height + radius*np.cos(phis)
        cz = forward + radius*np.sin(phis)
        n = points
        constant = n[:, 1, None]*cy + n[:, 2, None]*cz + n[:, 3, None]
        linear = n[:, 0] + n[:, 1]*sh + n[:, 2]*sf
        quadratic = n[:, 1]*bend
        # Continuous root nearest the corresponding linear intersection.
        disc = np.maximum(linear[:, None]**2-4*quadratic[:, None]*constant, 1e-10)
        denom = linear[:, None] + np.sign(linear[:, None])*np.sqrt(disc)
        x = -2*constant/(denom+1e-12)
        y = cy + sh*x + bend*x*x
        z = cz + sf*x
        qy = cameras[:, 0, 1, None]*x + cameras[:, 1, 1, None]*y + cameras[:, 2, 1, None]*z + cameras[:, 3, 1, None]
        qz = cameras[:, 0, 2, None]*x + cameras[:, 1, 2, None]*y + cameras[:, 2, 2, None]*z + cameras[:, 3, 2, None]
        v = cap.cy-fy[:, None]*qy/np.maximum(.01, qz)
        bottom = v.argmax(axis=1)
        ids = np.arange(len(v))
        return (x[ids, bottom], y[ids, bottom], z[ids, bottom]) if details else v[ids, bottom]

    fits = []
    for form in ["straight", "sag"]:
        for radius in [0., .10, .20, .30, .40]:
            for parity in [0, 1]:
                train = frames % 2 == parity
                def residual(par):
                    return (project(par, radius)-obs_v)[train]
                bendmax = .0000001 if form=="straight" else .5
                fit = least_squares(residual, [1.6+radius, 2., .1, 0., 0.],
                                    bounds=([.7,.3,-1.,-2.,-bendmax], [3.2,5.,1.,2.,bendmax]),
                                    loss="soft_l1", f_scale=2.5)
                par = fit.x
                e = abs(project(par, radius)-obs_v)
                observed_x = project(par, radius, True)[0]
                # A dense material-surface mesh on the fitted span checks a
                # possible intersection away from the centre gaze ray.
                xx, angle = np.meshgrid(np.linspace(observed_x.min(), observed_x.max(), 401),
                                        np.linspace(0, 2*np.pi, 121))
                xx, angle = xx.ravel(), angle.ravel()
                yy = par[0]+par[2]*xx+par[4]*xx**2+radius*np.cos(angle)
                zz = par[1]+par[3]*xx+radius*np.sin(angle)
                surface = origin+np.column_stack([xx,yy,zz])@axes.T
                crossing = {}
                for case in cases:
                    pose = cap.pv[case["rgb_ts"]][2]
                    f = -pose[:3, 2].copy()
                    f[1] = 0.
                    f /= np.linalg.norm(f)
                    lateral = np.cross(f, UP)
                    # Intersect gravity-vertical plane on gaze centre with lowermost generator.
                    h0, z0, sy, sz, bend = par
                    dx = -((origin + axes[:, 2]*z0-pose[:3, 3])@lateral)/((axes[:, 0]+axes[:, 2]*sz)@lateral)
                    xyz = origin+axes@np.array([dx,h0+sy*dx+bend*dx*dx-radius,z0+sz*dx])
                    relative = surface-pose[:3,3]
                    height = surface[:,1]-floors[case["depth_ts"]]
                    side = relative@lateral
                    depths = (relative@f)[(abs(side)<=.25)&(height>=.60)&(height<=1.90)]
                    crossing[case["case"]] = dict(lateral_parameter=dx, forward_m=(xyz-pose[:3,3])@f,
                                                   floor_height_m=xyz[1]-floors[case["depth_ts"]],
                                                   within_observed_span=bool(observed_x.min()<=dx<=observed_x.max()),
                                                   shell_forward_range_m=[depths.min(),depths.max()] if len(depths) else [],
                                                   shell_enters_box=bool(np.any((depths>=.30)&(depths<=2.))))
                # Compute the time and height when the recorded camera actually
                # passes the fitted tube's axis in top-down view.
                rt = row["rgb_ts"]
                times = np.array([t for t in sorted(cap.pv) if rt-1e7 <= t <= rt+6e7], dtype=np.int64)
                trajectory = (np.array([cap.pv[t][2][:3, 3] for t in times])-origin) @ axes
                station = trajectory[:, 2]-(par[1]+par[3]*trajectory[:, 0])
                kinks = np.flatnonzero((station[:-1]<=0) & (station[1:]>0))
                walk = {}
                if len(kinks):
                    k = kinks[0]
                    alpha = station[k]/(station[k]-station[k+1])
                    x,head_height,z = trajectory[k] + alpha*(trajectory[k+1]-trajectory[k])
                    rim = par[0]+par[2]*x+par[4]*x*x-radius
                    span_views = sum(bool(observed_x[frames==j].min()<=x<=observed_x[frames==j].max())
                                     for j in np.unique(frames))
                    walk = dict(relative_sec=((times[k]-rt)+alpha*(times[k+1]-times[k]))/1e7,
                                lateral=x,head_camera_height=head_height, model_rim_height=rim,
                                rim_minus_camera_m=rim-head_height,
                                n_spanning_views=span_views,
                                within_observed_span=bool(observed_x.min()<=x<=observed_x.max()))
                by_view = {str(outlines[j]["rgb_ts"]):dict(median=float(np.median(e[frames==j])),
                           maximum=float(np.max(e[frames==j])), held=bool(j%2!=parity)) for j in np.unique(frames)}
                fits.append(dict(form=form, radius=radius, fit_parity=parity, walk=walk, by_view=by_view,
                                 held_med_px=np.median(e[~train]), held_max_px=np.max(e[~train]),
                                 held_p90_px=np.percentile(e[~train],90),fit_med_px=np.median(e[train]),
                                 par=par.tolist(), observed_x=[observed_x.min(), observed_x.max()],
                                 crossing=crossing, world_axes=axes.tolist(), world_origin=origin.tolist()))
    return fits


def main():
    (ROOT / "geometry").mkdir(parents=True, exist_ok=True)
    cases = json.loads((HEAD / "alert-review/cases.json").read_bytes())
    contexts = json.loads((ROOT / "masks/histories.json").read_bytes())
    results = []
    fits = {}
    outlines = json.loads((ROOT / "masks/manifest.json").read_bytes())
    for seq, context in contexts.items():
        cap = Recording(HEAD / seq)
        floors = {r["depth_ts"]: r.get("floor_y") for r in records(cap.directory / "geometry.jsonl")}
        tracks = compute(cap, seq, context)
        selected = [r for r in cases if r["seq"] == seq]
        results.extend(analyze(cap, selected, floors, tracks))
        if seq == "research-center-seq3":
            wide = compute(cap, seq, context, FAR_BOXES)
            wide_result = analyze(cap, selected, floors, wide)
            (ROOT / "geometry/ray-bounds-far-boxes.json").write_text(
                json.dumps(dict(boxes=FAR_BOXES, results=wide_result), indent=1, allow_nan=False), encoding="utf-8")
        fits[seq] = cylinder(cap, selected, floors, [r for r in outlines if r["seq"] == seq])
        extended = [r for r in outlines if r["seq"] == seq] + additional_edges(cap, seq)
        if len({r["rgb_ts"] for r in extended}) != len(extended):
            raise ValueError("Training and held-out views must be distinct exposures")
        fits[seq+"-extended"] = cylinder(cap, selected, floors, sorted(extended,key=lambda r:r["rgb_ts"]))
    (ROOT / "geometry/lower-rim.json").write_text(json.dumps(fits, indent=1, allow_nan=False), encoding="utf-8")
    out = ROOT / "geometry/ray-bounds.json"
    out.write_text(json.dumps(results, indent=1, allow_nan=False), encoding="utf-8")
    for r in results:
        if r["mode"] == "all" and r["padding_px"] == 8:
            print(r["case"], r["counts"], r["best"])


if __name__ == "__main__":
    main()
