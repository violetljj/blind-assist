"""Replay the same RGB contradiction filter on HoloSet furniture/wall and the prior ETH3D table.

Run from the checkout root. Default uses archived HoloSet predictions. For the ETH3D
control, --eth-predict creates its cache with the frozen backbone on a GPU host, and
--eth-table consumes that cache for scoring.
"""
import argparse
from pathlib import Path
import json

import cv2
import numpy as np
import torch

import holoset_return_time_20261118 as d8
from holoset_return_objects_20261118 import coordinates

OUT = d8.WORK / "holoset-contradiction-20261119/static"
CONTROLS = (
    ("campus-center-seq2", "cafeteria table and green chair", 133080857769764183, 133080857779764391),
    ("research-center-seq1", "projecting wall corner", 133058565898194325, 133058565900194368),
)


def check_window(label, rows, exposures, floors, archived, net, verifier):
    results = []
    start = rows[0]["depth_ts"]
    for row in rows:
        ts, rt = row["depth_ts"], row["rgb_ts"]
        selected = [q for q in exposures if ts-d8.original.WINDOW <= q["depth_ts"] <= ts]
        if not selected or selected[-1]["depth_ts"] != ts or ts not in floors:
            raise ValueError("Missing target input or floor")
        floor = floors[ts]
        history = [q["frame"] for q in selected]
        reference = d8.original.features(history, floor, d8.V)
        baseline = d8.score(net, reference)
        if ts in archived and abs(baseline - archived[ts]) > 1.e-4:
            raise ValueError("Frozen score differs from archival HoloSet result")
        rec = dict(dt=ts, rgb_ts=rt, time_s=(ts-start)/1.e7, baseline=baseline,
                   historical_exposures=len(history)-1, archived=archived.get(ts))
        if verifier is not None:
            changed, occupancy = [], []
            counts = dict(marked=0, marked_pred_grid=0, marked_true_grid=0, marked_true_body=0,
                          pred_grid=0, true_grid=0, paired_grid=0, marked_paired_grid=0)
            for source in selected[:-1]:
                f = source["frame"]
                cutoff = min(ts, rt)
                if source["rgb_ts"] >= cutoff:
                    mask = np.zeros(f["depth"].shape, bool)
                else:
                    mask, _ = verifier.classify(source["rgb_ts"], cutoff, f["maps"], source["mono"])
                valid = (f["depth"] > 0) & (f["depth"] < 8)
                drops = mask[valid]
                pred, _body, *_ = coordinates(f["points"], history[-1]["pose"], floor)
                real, real_body, *_ = coordinates(f["measured"], history[-1]["pose"], floor)
                counts["marked"] += int(drops.sum())
                counts["marked_pred_grid"] += int((drops & pred).sum())
                counts["marked_true_grid"] += int((drops & real).sum())
                counts["marked_true_body"] += int((drops & real_body).sum())
                counts["pred_grid"] += int(pred.sum())
                counts["true_grid"] += int(real.sum())
                counts["paired_grid"] += int((pred & real).sum())
                counts["marked_paired_grid"] += int((drops & pred & real).sum())
                z = np.where(mask, 0., f["depth"])
                hits = f["points"][~drops]
                changed.append({**f, "points":hits, "depth":z})
                occupancy.append({**f, "points":hits})
            rec.update(counts=counts,
                       filtered=d8.score(net, d8.original.features([*changed, history[-1]], floor, d8.V)),
                       occupancy_only=d8.score(net, d8.original.features([*occupancy, history[-1]], floor, d8.V)))
        results.append(rec)
    return dict(label=label, rows=results)


def verifier_for(views, exclude_near):
    if exclude_near:
        from holoset_rgb_exclude_near_20261119 import RGBNearExcluder
        return RGBNearExcluder(views)
    from holoset_rgb_contradiction_20261119 import RGBVerifier
    return RGBVerifier(views)


def holoset(control, net, verify, exclude_near=False):
    seq, name, start, end = control
    cap = d8.original.Recording(d8.HEAD / seq)
    geom = d8.original.records(cap.directory / "geometry.jsonl")
    rows = [r for r in geom if r["tag"] not in ("no_depth_pose","no_rgb_match")
            and d8.original.rigid(cap.rig[r["depth_ts"]]) and d8.original.rigid(cap.pv[r["rgb_ts"]][2])]
    cache = d8.SCORES / seq / "predicted"
    targets = []
    skipped = []
    for r in rows:
        ts = r["depth_ts"]
        if start-20_000_000 <= ts <= end+6_000_000:
            support = [q for q in rows if ts-d8.original.WINDOW <= q["depth_ts"] <= ts]
            if "floor_y" in r and all((cache / f"{q['depth_ts']}.npy").exists() for q in support):
                targets.append(r)
            else:
                skipped.append(dict(ts=ts,reason="floor" if "floor_y" not in r else "cache"))
    exposures = []
    for r in rows:
        t = r["depth_ts"]
        if any(0 <= q["depth_ts"]-t <= d8.original.WINDOW for q in targets):
            path = cache / f"{t}.npy"
            frame = d8.replay.cached_frame(cap, r, path, d8.A.MAX_MM*.001)
            exposures.append(dict(**r,frame=frame,mono=np.load(path)))
    verifier = None
    if verify:
        from holoset_rgb_contradiction_20261119 import holoset_views
        verifier = verifier_for(holoset_views(cap, OUT / "extra" / seq), exclude_near)
    archived = json.loads((d8.SCORES/seq/"scores.json").read_bytes())["scores"]
    scores = {r["depth_ts"]: r["score"] for r in archived}
    result = check_window(name, targets, exposures, {r["depth_ts"]:r["floor_y"] for r in targets}, scores, net, verifier)
    if seq == "campus-center-seq2" and verify:
        result["furniture_source"] = furniture_source(exposures, cap, rows, net)
    return dict(seq=seq, encounter_start=start, encounter_end=end, skipped=skipped, **result)


def furniture_source(exposures, cap, rows, net):
    """Test the dining furniture in the only positive source image; diners remain in the scene."""
    ts = 133080857771764225
    support = [r for r in exposures if ts-d8.original.WINDOW <= r["depth_ts"] <= ts]
    source = support[-1]
    if source["depth_ts"] != ts:
        raise ValueError("Missing reviewed café alert")
    target = source["frame"]
    floor = next(r["floor_y"] for r in rows if r["depth_ts"]==ts)
    # Conservative tabletop and empty chair polygons, excluding the diner.
    furniture = np.zeros((cap.height,cap.width),np.uint8)
    polygons = [[(98,282),(150,255),(218,236),(235,221),(349,255),(350,261),(201,341)],
                [(335,193),(348,185),(399,194),(408,206),(402,276),(353,257),(350,248),(332,244)],
                [(279,304),(350,267),(399,278),(399,287),(337,325),(279,308)]]
    for poly in polygons:
        cv2.fillPoly(furniture,[np.int32(poly)],1)
    roi = cv2.remap(furniture,*target["maps"],cv2.INTER_NEAREST) != 0
    hp,wp = roi.shape
    keep, erase, counts = [], [], []
    for item in support:
        f = item["frame"]
        ok = (f["depth"]>0)&(f["depth"]<8.)
        pts = (f["measured"]-target["pose"][:3,3])@target["pose"][:3,:3]
        zz = pts[:,2]
        uv = (d8.original.FOCAL*pts[:,:2]/np.maximum(.05,zz)[:,None]+target["center"])/d8.original.STEP
        u,v = np.rint(uv).astype(int).T
        inside = (zz>.05)&(u>=0)&(u<wp)&(v>=0)&(v<hp)
        uc,vc = u.clip(0,wp-1),v.clip(0,hp-1)
        matched = inside & roi[vc,uc] & (abs(zz-target["ref"][vc,uc])<=.12)
        mask = np.zeros_like(ok)
        mask[ok] = matched
        for stack,sel in ((keep,mask),(erase,~mask)):
            stack.append({**f,"points":f["points"][sel[ok]],"depth":np.where(sel,f["depth"],0.)})
        counts.append(dict(ts=item["depth_ts"],rays=int(matched.sum())))
    return dict(only=d8.score(net,d8.original.features(keep,floor,d8.V)),
                erased=d8.score(net,d8.original.features(erase,floor,d8.V)),verified_rays=counts,
                polygons=polygons,range_tolerance_m=.12)


def eth_table(net, output_dir, exclude_near=False):
    """The existing 1.806s ETH3D tabletop control, with the original table masks and Vicon poses."""
    from holoset_rgb_contradiction_20261119 import View
    folder = d8.WORK / "real-multiframe-static-occlusion-20261117-eth3d/eth3d_39.5_41.5"
    receipt = json.loads((OUT / "receipt.json").read_bytes())
    original_run = json.loads((d8.SCORES / "run.json").read_bytes())
    if (receipt["model_sha"] != original_run["dav2_sha256"]
            or receipt["input_sha"] != d8.original.file_digest(folder / "frames.npz")
            or receipt["output_sha"] != d8.original.file_digest(OUT / "mono.npz")):
        raise ValueError("Static table cache differs from the checked input/model")
    raw = np.load(folder / "frames.npz")
    target_masks = np.load(folder / "references.npz")["static_reference"]
    predictions = np.load(OUT / "mono.npz")
    focal, step = d8.original.FOCAL, d8.original.STEP
    fx, fy, cx, cy = raw["intrinsics"]
    K = np.array([[fx,0.,cx],[0.,fy,cy],[0.,0.,1.]])
    bgr, native_range = raw["bgr"], raw["depth_png"]
    height, width = native_range.shape[1:]
    sx, sy = focal/fx, focal/fy
    v, u = np.mgrid[:round(height*sy):step, :round(width*sx):step].astype(np.float32)
    maps = ((u/sx).astype(np.float32), (v/sy).astype(np.float32))
    center = (cx*sx,cy*sy)
    rays = np.stack(((u-center[0])/focal, (v-center[1])/focal, np.ones_like(u)),axis=-1)
    imgdir = OUT / "eth_rgb"
    imgdir.mkdir(parents=True, exist_ok=True)
    exposures, views = [], []
    for i, source_id in enumerate(raw["source_indices"]):
        ts = round(raw["rgb_timestamp"][i]*1e7)
        dt = round(raw["depth_timestamp"][i]*1e7)
        pose = np.eye(4)
        pose[:3,:3] = raw["rotations"][i]
        pose[:3,3] = raw["locations"][i]
        image_path = imgdir/f"{ts}.png"
        if not cv2.imwrite(str(image_path),bgr[i]):
            raise OSError(image_path)
        views.append(View(ts,image_path,K,pose))
        mono = cv2.remap(predictions[str(source_id)],*maps,cv2.INTER_LINEAR)
        ref = cv2.remap(native_range[i],*maps,cv2.INTER_NEAREST).astype(np.float32)/float(raw["depth_scale"])
        common = (ref>0)&(ref<8)&(mono>0)&np.isfinite(mono)
        scale = float(np.median(ref[common]/mono[common]))
        depth = np.where(ref>0, scale*mono, 0.)
        valid = (depth>0)&(depth<8)
        def world(z):
            return ((rays[valid]*z[valid,None])@pose[:3,:3].T+pose[:3,3]).astype(np.float32)
        points = world(depth)
        frame = dict(depth=depth, ref=ref, points=points, measured=world(ref), pose=pose, center=center, maps=maps)
        table = cv2.remap(target_masks[i].astype(np.uint8),*maps,cv2.INTER_NEAREST)>0
        exposures.append(dict(depth_ts=dt, rgb_ts=ts, frame=frame, mono=mono, table=table))
    # The surveyed table is 0.72--0.77 m above the original Vicon floor (z=0).
    result = check_window("ETH3D motion_1 natural round table",exposures,exposures,
                          {r["depth_ts"]:0. for r in exposures},{},net,verifier_for(views, exclude_near))
    # Keep or erase only rays originally assigned to the independently surveyed tabletop.
    only, erased = [], []
    for i,(r,obs) in enumerate(zip(result["rows"],exposures)):
        f, m = obs["frame"], obs["table"]
        valid = (f["depth"]>0)&(f["depth"]<8)
        for history,keep in [(only,m),(erased,~m)]:
            history.append({**f,"points":f["points"][keep[valid]],"depth":np.where(keep,f["depth"],0.)})
        r.update(native_index=int(raw["source_indices"][i]),
                 only_table=d8.score(net,d8.original.features(only,0.,d8.V)),
                 remove_table=d8.score(net,d8.original.features(erased,0.,d8.V)))
    result["start_condition"]="empty history at the first retained exposure (native crop)"
    (output_dir / "eth-table.json").write_text(json.dumps(result,indent=1),encoding="utf-8")


def predict_eth():
    from spires_insert_dev_20261117 import _dav2s
    folder = d8.WORK / "real-multiframe-static-occlusion-20261117-eth3d/eth3d_39.5_41.5"
    raw = np.load(folder / "frames.npz")
    model = _dav2s()
    output = {}
    for idx, image in zip(raw["source_indices"],raw["bgr"]):
        with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):
            estimate = model(model.image2tensor(image,518)[0])
        dense = torch.nn.functional.interpolate(estimate[:,None].float(),image.shape[:2],
                                                mode="bilinear",align_corners=True)
        output[str(int(idx))] = dense[0,0].cpu().numpy()
    OUT.mkdir(parents=True,exist_ok=True)
    path = OUT / "mono.npz"
    np.savez_compressed(path,**output)
    weights = d8.WORK.parent / "models/depth-anything--Depth-Anything-V2-Metric-Hypersim-Small/depth_anything_v2_metric_hypersim_vits.pth"
    record = dict(model_sha=d8.original.file_digest(weights), input_sha=d8.original.file_digest(folder/"frames.npz"),
                  output_sha=d8.original.file_digest(path), device=torch.cuda.get_device_name(), torch=torch.__version__)
    (OUT / "receipt.json").write_text(json.dumps(record),encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--eth-table",action="store_true",help="Score the existing ETH3D control using its recovered frozen DAV2 cache")
    ap.add_argument("--eth-predict",action="store_true",help="Generate fresh DAV2 predictions for the ten existing ETH3D views")
    ap.add_argument("--exclude-near",action="store_true",help="Use RGB near-depth exclusion; write separate scores under near-exclusion")
    args = ap.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    if args.eth_predict:
        predict_eth()
        return
    net = d8.I.D.make_head(8)
    weights = d8.WORK/"ase-cane-reach-pilot-20261015/decision_head8_all.pt"
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.eval()
    output_dir = OUT / "near-exclusion" if args.exclude_near else OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.eth_table:
        eth_table(net, output_dir, args.exclude_near)
        return
    results = [holoset(c, net, not args.baseline_only, args.exclude_near) for c in CONTROLS]
    fname = "baseline.json" if args.baseline_only else "result.json"
    (output_dir/fname).write_text(json.dumps(dict(weights_sha256=d8.original.file_digest(weights),
                                          threshold=.5367,controls=results), indent=1),encoding="utf-8")


if __name__=="__main__":
    main()
