from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from contract import BUNDLE_SCHEMA, REPORT_SCHEMA, quantile, read_json, safe_file, sha256, validate_pose, write_json


def _depth(path: Path, encoding: str, scale: float) -> np.ndarray:
    if encoding == "uint16_png_z_meters":
        return np.asarray(Image.open(path), dtype=np.float32) / scale
    if encoding == "float32_npy_z_meters":
        value = np.load(path).astype(np.float32, copy=False)
        return value / scale
    raise ValueError(f"unsupported depth encoding: {encoding}")


def _reproject(first: dict[str, Any], second: dict[str, Any], root: Path, stride: int) -> tuple[float, float, float]:
    d1 = _depth(safe_file(root, first["depth_path"]), first["depth_encoding"], float(first["depth_scale"]))
    d2 = _depth(safe_file(root, second["depth_path"]), second["depth_encoding"], float(second["depth_scale"]))
    if d1.shape != d2.shape or d1.ndim != 2:
        raise ValueError("depth raster shape changed")
    fx, fy, cx, cy = [float(v) for v in first["intrinsics_fx_fy_cx_cy"]]
    height, width = d1.shape
    vv, uu = np.mgrid[0:height:stride, 0:width:stride]
    z = d1[vv, uu]
    points = np.stack(((uu-cx)*z/fx, (vv-cy)*z/fy, z, np.ones_like(z)), axis=-1).reshape(-1,4)
    relative = np.linalg.inv(np.asarray(second["camera_to_world"])) @ np.asarray(first["camera_to_world"])
    transformed = points @ relative.T
    target_z = transformed[:,2]
    u2 = np.rint(fx*transformed[:,0]/np.maximum(target_z,1e-6)+cx).astype(int)
    v2 = np.rint(fy*transformed[:,1]/np.maximum(target_z,1e-6)+cy).astype(int)
    valid = (points[:,2] > 0) & (target_z > 0) & (u2 >= 0) & (u2 < width) & (v2 >= 0) & (v2 < height)
    observed = np.zeros_like(target_z); observed[valid] = d2[v2[valid],u2[valid]]
    valid &= np.isfinite(observed) & (observed > 0)
    residual = np.abs(observed[valid] - target_z[valid])
    if not residual.size:
        raise ValueError("no valid temporal reprojection samples")
    return float(valid.mean()), float(np.median(residual)), float(np.quantile(residual,.95))


def _source(bundle_dir: Path, prereg: dict[str, Any]) -> dict[str, Any]:
    bundle = read_json(bundle_dir / "bundle.json")
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("bundle schema mismatch")
    ledger = bundle_dir / "frames.jsonl"
    if sha256(ledger) != bundle["frames_sha256"]:
        raise ValueError("frames ledger hash mismatch")
    frames = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    root = Path(bundle["source_root"]).resolve()
    sync = []; valid_depth = []; clearance = []
    for row in frames:
        validate_pose(row["camera_to_world"])
        rgb = safe_file(root,row["rgb_path"]); depth_path = safe_file(root,row["depth_path"])
        if sha256(rgb) != row["rgb_sha256"] or sha256(depth_path) != row["depth_sha256"]:
            raise ValueError(f"modality hash mismatch: {row['frame_id']}")
        aligned = abs(float(row["rgb_timestamp_s"])-float(row["depth_timestamp_s"]))*1000 <= prereg["synchronization"]["maximum_rgb_depth_delta_ms"] and abs(float(row["rgb_timestamp_s"])-float(row["pose_timestamp_s"]))*1000 <= prereg["synchronization"]["maximum_rgb_pose_delta_ms"] and row["pose_stability"] == "INTER_FRAME_STABLE"
        sync.append(aligned)
        depth = _depth(depth_path,row["depth_encoding"],float(row["depth_scale"])); mask=np.isfinite(depth)&(depth>0)
        valid_depth.append(float(mask.mean()))
        h,w=depth.shape; corridor=depth[h//2:, int(w*.4):int(w*.6)]; values=corridor[np.isfinite(corridor)&(corridor>0)]
        if values.size: clearance.append(float(np.quantile(values,.05)))
    count=min(int(prereg["geometry"]["reprojection_pair_count"]),len(frames)-1)
    indexes=np.unique(np.linspace(0,len(frames)-2,num=count,dtype=int))
    reprojections=[_reproject(frames[int(i)],frames[int(i)+1],root,int(prereg["geometry"]["reprojection_pixel_stride"])) for i in indexes]
    p95s=[value[2] for value in reprojections]
    aligned_fraction=float(np.mean(sync)); valid_fraction=float(np.mean(valid_depth)); registration_p95=quantile(p95s,.5)
    geometry_passed=aligned_fraction >= prereg["synchronization"]["minimum_source_aligned_fraction"] and valid_fraction >= prereg["geometry"]["minimum_valid_depth_fraction"] and registration_p95 is not None and registration_p95 <= prereg["geometry"]["maximum_temporal_reprojection_p95_m"]
    return {
      "source_id":bundle["source"]["source_id"], "kind":bundle["source"]["kind"], "license":bundle["source"]["license"], "frame_count":len(frames),
      "synchronization":{"source_aligned_fraction":aligned_fraction,"maximum_rgb_depth_delta_ms":max(abs(float(r["rgb_timestamp_s"])-float(r["depth_timestamp_s"]))*1000 for r in frames),"maximum_rgb_pose_delta_ms":max(abs(float(r["rgb_timestamp_s"])-float(r["pose_timestamp_s"]))*1000 for r in frames)},
      "depth":{"mean_valid_fraction":valid_fraction,"temporal_registration_pair_count":len(reprojections),"median_valid_projection_fraction":quantile([v[0] for v in reprojections],.5),"median_pair_median_residual_m":quantile([v[1] for v in reprojections],.5),"median_pair_p95_residual_m":registration_p95},
      "clearance":{"geometry_proxy_p05_median_m":quantile(clearance,.5),"meaning":"source-depth lower-center p05; not alert lifecycle clearance"},
      "pose_drift":{"status":"not_evaluable","reason":"no independent estimated trajectory was supplied"},
      "route_projection_error":{"status":"not_evaluable","reason":"no hash-bound independent route truth/review receipt was supplied"},
      "events":{"status":"not_evaluable","event_recall":None,"false_alerts_per_minute":None,"alert_clearance":None,"reason":"no isolated model-consensus event truth and no candidate alert trace"},
      "geometry_transport_passed":geometry_passed,
    }


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine=np.clip((np.trace(first.T@second)-1.0)/2.0,-1.0,1.0)
    return float(np.degrees(np.arccos(cosine)))


def _apply_evaluation(source_dir: Path, result: dict[str, Any], value: dict[str, Any]) -> None:
    frames=[json.loads(line) for line in (source_dir/"frames.jsonl").read_text(encoding="utf-8").splitlines() if line]
    truth_pose={row["frame_id"]:np.asarray(row["camera_to_world"],dtype=np.float64) for row in frames}
    estimates=value.get("pose_estimates",[])
    if estimates:
        translation=[]; rotation=[]
        for row in estimates:
            estimate=np.asarray(row["camera_to_world"],dtype=np.float64); validate_pose(estimate.tolist()); truth=truth_pose[row["frame_id"]]
            translation.append(float(np.linalg.norm(estimate[:3,3]-truth[:3,3]))); rotation.append(_rotation_error_deg(estimate[:3,:3],truth[:3,:3]))
        result["pose_drift"]={"status":"evaluable","translation_rmse_m":float(np.sqrt(np.mean(np.square(translation)))),"translation_endpoint_m":translation[-1],"rotation_rmse_deg":float(np.sqrt(np.mean(np.square(rotation)))),"evaluated_frames":len(estimates)}
    truth_route={row["frame_id"]:np.asarray(row["uv"],dtype=np.float64) for row in value.get("route_truth",[])}
    predicted_route={row["frame_id"]:np.asarray(row["uv"],dtype=np.float64) for row in value.get("route_predictions",[])}
    common=sorted(set(truth_route)&set(predicted_route))
    if common:
        errors=[float(np.linalg.norm(predicted_route[key]-truth_route[key])) for key in common]
        result["route_projection_error"]={"status":"evaluable","median_px":quantile(errors,.5),"p95_px":quantile(errors,.95),"evaluated_frames":len(common),"unknown_rate":1-len(common)/max(1,len(truth_route))}
    events=value.get("events",[]); alerts=value.get("alerts",[]); duration=float(value.get("duration_s",0))
    if events and duration>0:
        hit=[]
        for event in events:
            hit.append(any(int(alert["end_frame"])>=int(event["start_frame"]) and int(alert["start_frame"])<=int(event["end_frame"]) for alert in alerts))
        false_alerts=[alert for alert in alerts if not any(int(alert["end_frame"])>=int(event["start_frame"]) and int(alert["start_frame"])<=int(event["end_frame"]) for event in events)]
        clear=[]; latencies=[]
        for event in events:
            overlapping=[alert for alert in alerts if int(alert["end_frame"])>=int(event["start_frame"]) and int(alert["start_frame"])<=int(event.get("clear_deadline_frame",event["end_frame"]))]
            if not overlapping: continue
            latency=max(0,max(int(alert["end_frame"]) for alert in overlapping)-int(event["end_frame"])); latencies.append(latency); clear.append(latency<=int(event.get("maximum_clearance_frames",0)))
        result["events"]={"status":"evaluable","event_recall":float(np.mean(hit)),"false_alerts_per_minute":len(false_alerts)/(duration/60.0),"alert_clearance":{"rate":float(np.mean(clear)) if clear else None,"p95_frames":quantile(latencies,.95)},"event_count":len(events),"alert_count":len(alerts)}


def run(bundle_root: Path, prereg_path: Path, report_path: Path, evaluation_path: Path | None = None, review_consensus_path: Path | None = None) -> dict[str, Any]:
    prereg=read_json(prereg_path)
    sources=[_source(path,prereg) for path in sorted(bundle_root.iterdir()) if path.is_dir() and (path/"bundle.json").is_file()]
    evaluation=read_json(evaluation_path) if evaluation_path else {"sources":[]}
    evaluation_by_id={row["source_id"]:row for row in evaluation.get("sources",[])}
    for row in sources:
        if row["source_id"] in evaluation_by_id: _apply_evaluation(bundle_root/row["source_id"],row,evaluation_by_id[row["source_id"]])
    worst=max(sources,key=lambda row: row["depth"]["median_pair_p95_residual_m"] or float("inf")) if sources else None
    geometry_sources=sum(bool(row["geometry_transport_passed"]) for row in sources)
    missing=sorted({metric for metric in prereg["required_closure_metrics"] for row in sources if (metric=="pose_drift" and row["pose_drift"]["status"]!="evaluable") or (metric=="route_projection_error" and row["route_projection_error"]["status"]!="evaluable") or (metric in {"event_recall","false_alerts_per_minute","alert_clearance"} and row["events"]["status"]!="evaluable")})
    review=read_json(review_consensus_path) if review_consensus_path else None
    report={"schema":REPORT_SCHEMA,"source_count":len(sources),"sources":sources,"geometry_transport_source_pass_count":geometry_sources,"worst_source":worst["source_id"] if worst else None,"missing_closure_metrics":missing,"route_event_review_consensus":review,"algorithm_closed_loop_proven":len(sources)>=prereg["minimum_admitted_sources"] and geometry_sources==len(sources) and not missing,"hardware_selection_authorized":False,"u0_authorized":False,"model_proxy_120_authorized":False,"production_authority":False}
    report["verdict"]="MULTISOURCE_ALGORITHM_CLOSED_LOOP" if report["algorithm_closed_loop_proven"] else "DO_NOT_SELECT_HARDWARE"
    write_json(report_path,report); return report


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--bundle-root",type=Path,required=True); parser.add_argument("--prereg",type=Path,required=True); parser.add_argument("--report",type=Path,required=True); parser.add_argument("--evaluation",type=Path); parser.add_argument("--review-consensus",type=Path); args=parser.parse_args()
    try:
        if args.report.exists(): raise ValueError(f"refusing to overwrite report: {args.report}")
        report=run(args.bundle_root.resolve(),args.prereg.resolve(),args.report.resolve(),args.evaluation.resolve() if args.evaluation else None,args.review_consensus.resolve() if args.review_consensus else None); print(json.dumps({"verdict":report["verdict"],"sources":report["source_count"],"worst_source":report["worst_source"],"missing":report["missing_closure_metrics"]})); return 0 if report["algorithm_closed_loop_proven"] else 3
    except (OSError,ValueError,KeyError) as error: print(json.dumps({"ok":False,"error":str(error)})); return 2


if __name__=="__main__": raise SystemExit(main())
