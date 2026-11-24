"""Transfer frozen M14 boundary fills to the registered Oxford shallow-control alarm."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
M14 = HERE.parent
sys.path.insert(0, str(M14))
from atlas import level_basis, pin_rays, project_kb, pull
from eval_empirical import CONTROL, MEAN, MODEL, OX, OUT, ROOT, WORK, digest, infer, session

HOST = ROOT / "research/active/hardware-bringup/host"
sys.path[:0] = [str(HOST), str(HOST.parent / "evalkit")]
import core
import shallow_event_comparison_20261118 as score
from shallow_surface_evidence_20261118 import shallow_masks
from shallow_valid_view_20261118 import _oxford_view
from spires_insert_dev_20261117 import K0, D0, R_CV

DEST = OUT / HERE.name
OLD = WORK / "shallow-alerts-20261118"
AREA_RECORD = OLD / "oxford-areas-camera-valid.json"
OLD_SCORES = OLD / "comparison-camera-valid.json"
LABELS = WORK / "paired-low-sites-20261118/scoring/labels/oxford.json"
MANIFEST = OX / "locked/manifest.json"
FILLS = ("black", "mean", "nearest_valid")


def read(path):
    return json.loads(path.read_bytes())


def bytes_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def key(row):
    return row["seq"], row["i"]


def panels(manifest, labels, controls):
    rows = {key(row): row for row in manifest["frames"]}
    selected = {key(r): OX / "original" / r["seq"] / r["source"] for r in controls["rows"]}
    if len(selected) != 22:
        raise ValueError("Expected 22 frozen control timestamps")
    for case in labels["cases"]:
        for obs in case["observations"]:
            if obs.get("near", True):
                r = rows[key(obs)]
                p = OX / "original" / r["seq"] / (Path(r["rgb"]).name.rsplit("_rgb", 1)[0] + ".jpg")
                if key(obs) in selected:
                    raise ValueError("Approach footprint overlaps a clear timestamp")
                selected[key(obs)] = p
    if len(selected) != 32:
        raise ValueError("Expected 22 controls and 10 disjoint registered near timestamps")
    return [(rows[k],selected[k]) for k in sorted(selected)]


def predict(sess, row, raw_path, packet, native_view, calibration, cam, model_xy, model_valid, nearest):
    # Only RGB, current orientation, intrinsics and cached RGB semantic probabilities enter the readout.
    with np.load(packet, allow_pickle=False) as p:
        prob = p["probability"].copy()
        k = p["K"]
    h,w = prob.shape
    assert (h,w) == native_view.shape and np.allclose(k,calibration)
    rgb = cv2.imread(str(raw_path))
    if rgb is None or rgb.shape != (1080,1440,3):
        raise ValueError(f"Missing raw calibrated frame {raw_path}")
    image, valid = pull(rgb,*model_xy)
    assert np.array_equal(valid, model_valid)
    frames = {"black":image, "mean":image.copy(), "nearest_valid":image[tuple(nearest)]}
    frames["mean"][~valid] = np.rint(255*MEAN[::-1]).astype(np.uint8)
    assert np.all(frames["black"][~valid] == 0)
    assert all(np.array_equal(image[valid],f[valid]) for f in frames.values())
    rays = cam @ level_basis(row["down"]).T
    # Native k describes rays through integer centres. Model pin_rays uses half-integer centres.
    v,u = np.indices((h,w),dtype=np.float32)
    uv = np.stack((224*(u-k[0,2])/k[0,0]+192-.5,126*(v-k[1,2])/k[1,1]+144-.5),axis=-1)
    mask_on_native,inside = pull(valid.astype(np.float32),uv,native_view)
    shared = native_view & inside & (mask_on_native >= 1.)
    prob[~shared]=0.
    diag = {"raw_sha256":digest(raw_path), "cache_sha256":digest(packet),
            "effective_native_valid":int(shared.sum()), "model_invalid":int((~valid).sum()),
            "effective_native_mask_sha256":bytes_sha(shared), "fills":{}}
    supports, depths = {}, {}
    for name, frame in frames.items():
        depth,ms = infer(sess,frame,rays,valid)
        if not np.isfinite(depth).all():
            raise ValueError("Missing model predictions")
        depths[name] = depth
        z,_ = pull(depth,uv,shared)
        z[~shared] = np.nan
        _,mask,status = shallow_masks(z,k,prob)
        assert not np.any(mask & ~shared)
        area = core.pixel_area(k,h,w,z.ravel()).reshape(h,w)
        idx = np.flatnonzero(mask)
        masses = area.ravel()[idx]
        supports[name] = idx,masses
        diag["fills"][name] = {
            "area_m2":float(masses.sum()), "down_available":status["down_available"],
            "corridor_px":status["corridor_px"], "anchored_corridor_px":status["local_anchored_corridor_px"],
            "prediction_sha256":bytes_sha(depth), "valid_rgb_sha256":bytes_sha(frame[valid]),
            "inference_cpu_ms":ms}
    return supports,depths,diag


def adjudicate(areas,labels,predictions,times,controls):
    control_index = {(r["win"],r["i"]):key(r) for r in controls["rows"]}
    assert {(w["id"],r["i"]) for w in areas["windows"] for r in w["rows"]} == set(control_index)
    for w in areas["windows"]:
        for r in w["rows"]:
            src = control_index[w["id"],r["i"]]
            assert times[src] == r["t"]
            for fill in FILLS:
                r["total"][fill] = predictions[src][fill][1].sum().item()
    footprints = {(c["encounter_id"],c["object_id"],r["i"]):r for c in labels["cases"]
                  for r in c["observations"] if r.get("near",True)}
    for e in areas["events"]:
        for r in e["observations"]:
            obs = footprints[e["id"],r["object_id"],r["i"]]
            p = LABELS.parent / obs["reference_footprint"]
            assert digest(p)==obs["reference_mask_sha256"] and times[key(obs)]==r["t"]
            roi = cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
            if roi is None or roi.shape != (640,480):
                raise ValueError("Incorrect Oxford footprint canvas")
            for fill,(idx,masses) in predictions[key(obs)].items():
                r["total"][fill] = float(masses.sum())
                r["target"][fill] = float(masses[roi.ravel()[idx]>0].sum())
    sites = {w["site"] for w in areas["windows"]}
    cuts = {"registered_default":dict.fromkeys(sites,score.MIN_AREA),
            "historical_other_site_cutoffs":{s:read(OLD_SCORES)["leave_site_out"]["local"]["area_threshold_by_site_m2"][s]
                                           for s in sites}}
    result = {}
    for strategy, thresholds in cuts.items():
        res = {}
        for fill in FILLS:
            windows = [score.score_window(w,fill,thresholds[w["site"]]) for w in areas["windows"]]
            events = [score.score_event(e,fill,thresholds[e["site"]]) for e in areas["events"]]
            res[fill] = dict(windows=windows, controls=score.window_counts(windows), events=events,
                             entrance_object_hits=sum(e["hit"] for e in events),
                             entrance_branch_only_hits=sum(e["first_strict_new_support"] is not None for e in events))
        def ids(fill):
            return {(w["id"],i) for w in res[fill]["windows"] for i in w["alarm_i"]}
        changes = {f:{"black_only":sorted(ids("black")-ids(f)), "fill_only":sorted(ids(f)-ids("black")),
                      "both":sorted(ids("black")&ids(f))} for f in FILLS[1:]}
        result[strategy] = dict(area_cutoff_m2=thresholds,fills=res,paired_timestamps=changes)
    return result


def main():
    cv2.setNumThreads(1)
    DEST.mkdir(parents=True,exist_ok=True)
    cache = OLD / "cache" / digest(MANIFEST)[:16]
    index = read(cache / "index.json")
    assert index["manifest_sha256"] == digest(MANIFEST)
    packets = {key(r):cache/r["file"] for r in index["rows"]}
    areas, labels, controls, manifest = map(read,(AREA_RECORD,LABELS,CONTROL,MANIFEST))
    assert digest(LABELS)==areas["sources"][0]["labels_sha256"]
    native = _oxford_view()
    assert bytes_sha(native)==controls["mask_sha256"]
    camera = pin_rays(288,384,(224.,126.,192.,144.))
    xy = project_kb(camera @ R_CV.T,K0,D0,np.deg2rad(80))
    _,valid = pull(np.zeros((1080,1440),np.uint8),*xy)
    nearest = distance_transform_edt(~valid,return_distances=False,return_indices=True)
    sess=session()
    old_panel = read(OUT/"padding_control.json")
    assert old_panel["model_sha256"] == digest(MODEL)
    old = {(r["source"],int(r["key"])):r for r in old_panel["padding"]["rows"]}
    pred,diags,times,bridge = {},{}, {}, {}
    for n,(row,raw) in enumerate(panels(manifest,labels,controls),1):
        src=key(row)
        predictions,depths,diag = predict(sess,row,raw,packets[src],native,np.array(row["K"]),
                                          camera,xy,valid,nearest)
        diag["seq"],diag["i"] = src
        pred[src], diags[src],times[src] = predictions,diag,row["t"]
        if src in old:
            # Evaluator-only depth bridge after prediction and mask construction.
            gt = cv2.imread(str(OX/row["depth"]),cv2.IMREAD_UNCHANGED).astype(np.float32)*row["ref_scale"]
            k=tuple(np.array(row["K"])[[0,1,0,1],[0,1,2,2]])
            uv=camera[...,:2]/camera[...,2:] * k[:2]+k[2:]
            gt,g = pull(gt,uv,camera[...,2]>0,nearest=True)
            ok = g & valid & (gt>=.3) & (gt<=12)
            sums = {f:float(np.abs(dep[ok]-gt[ok]).sum(dtype=np.float64)) for f,dep in depths.items()}
            diffs = {f:abs(sums[f]-old[src]["error_m_sum"][f]) for f in FILLS}
            assert ok.sum()==old[src]["reference_pixels"] and diag["model_invalid"]==old[src]["invalid"]
            if max(diffs.values())/ok.sum() > 1.e-5:
                raise ValueError(f"M14 original panel output mismatch: {src}, {diffs}")
            bridge[src] = max(diffs.values())/int(ok.sum())
        if n%8==0:
            print(f"Completed {n}/32",flush=True)
    assert len(bridge)==22
    result=adjudicate(areas,labels,pred,times,controls)
    inputs = [MANIFEST,CONTROL,AREA_RECORD,LABELS,OLD_SCORES,cache/"index.json",OUT/"padding_control.json",MODEL]
    code = [Path(__file__),M14/"eval_empirical.py",M14/"atlas.py",M14/"mask_control.py",
            HOST/"shallow_surface_evidence_20261118.py",HOST/"ground_geometry_filter_20261118.py",
            HOST/"shallow_valid_view_20261118.py",HOST/"spires_insert_dev_20261117.py",Path(score.__file__),Path(core.__file__)]
    doc=dict(selection="All three Oxford control windows and all registered near observations of the three LOW approaches",
             comparison=result, reference_bridge_max_per_frame_mean_abs_drift_m=max(bridge.values()),
             valid_camera_sha256=bytes_sha(native),model_valid_sha256=bytes_sha(valid),
             inputs_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
             code_sha256={str(p.relative_to(ROOT)):digest(p) for p in code},
             controls_fixed_n=22,windows_fixed_n=3,positive_near_timestamps_n=10,diagnostics=list(diags.values()),
             kms=None,km_reason="The registered shallow event protocol supplies windows and timestamps, not walked distance.")
    out=DEST/"alert_transfer.json"
    tmp=out.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc,indent=1,ensure_ascii=False,allow_nan=False),encoding="utf8")
    tmp.replace(out)
    print({s:{f:(v["controls"],v["entrance_object_hits"]) for f,v in a["fills"].items()} for s,a in result.items()})


if __name__=="__main__":
    main()
