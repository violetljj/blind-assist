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
    estimates=[row for row in value.get("pose_estimates",[]) if row.get("status") != "unknown"]
    if estimates:
        if len({row["frame_id"] for row in estimates}) != len(estimates): raise ValueError("duplicate pose estimate frame id")
        first=estimates[0]
        alignment=truth_pose[first["frame_id"]]@np.linalg.inv(np.asarray(first["camera_to_world"],dtype=np.float64))
        translation=[]; rotation=[]
        for row in estimates:
            estimate=alignment@np.asarray(row["camera_to_world"],dtype=np.float64); validate_pose(estimate.tolist()); truth=truth_pose[row["frame_id"]]
            translation.append(float(np.linalg.norm(estimate[:3,3]-truth[:3,3]))); rotation.append(_rotation_error_deg(estimate[:3,:3],truth[:3,:3]))
        result["pose_drift"]={"status":"evaluable","alignment":"first_common_frame_se3_only","translation_rmse_m":float(np.sqrt(np.mean(np.square(translation)))),"translation_endpoint_m":translation[-1],"rotation_rmse_deg":float(np.sqrt(np.mean(np.square(rotation)))),"evaluated_frames":len(estimates),"estimated_fraction":len(estimates)/len(frames)}
    truth_rows=value.get("route_truth",[]); prediction_rows=value.get("route_predictions",[])
    if any(float(row.get("predicted_at_s",float("inf"))) > float(row.get("timestamp_s",float("-inf")))+1e-9 for row in prediction_rows): raise ValueError("non-causal route prediction timestamp")
    truth_route={row["frame_id"]:np.asarray(row["uv"],dtype=np.float64) for row in truth_rows if row.get("status")=="known"}
    predicted_route={row["frame_id"]:np.asarray(row["uv"],dtype=np.float64) for row in prediction_rows if row.get("status")=="known"}
    common=sorted(set(truth_route)&set(predicted_route))
    if common:
        errors=[float(np.linalg.norm(predicted_route[key]-truth_route[key])) for key in common]
        result["route_projection_error"]={"status":"evaluable","median_px":quantile(errors,.5),"p95_px":quantile(errors,.95),"evaluated_frames":len(common),"truth_known_frames":len(truth_route),"prediction_known_frames":len(predicted_route),"unknown_rate":1-len(common)/max(1,len(frames))}
    events=value.get("events",[]); alerts=value.get("alerts",[]); duration=float(value.get("duration_s",0))
    if events and duration>0:
        timestamps=[float(row["rgb_timestamp_s"]) for row in frames]
        hit=[]; critical_hit=[]
        for event in events:
            start=int(event["alertable_frame"]); end=int(event["passed_or_cleared_frame"])
            matched=any(int(alert["end_frame"])>=start and int(alert["start_frame"])<=end for alert in alerts)
            hit.append(matched)
            if bool(event.get("critical")): critical_hit.append(matched)
        false_alerts=[alert for alert in alerts if not any(int(alert["end_frame"])>=int(event["alertable_frame"]) and int(alert["start_frame"])<=int(event["passed_or_cleared_frame"]) for event in events)]
        clear=[]; latencies_ms=[]
        for event,matched in zip(events,hit):
            if not matched:
                clear.append(False); continue
            overlapping=[alert for alert in alerts if int(alert["end_frame"])>=int(event["alertable_frame"]) and int(alert["start_frame"])<=int(event["passed_or_cleared_frame"])]
            last_end=max(int(alert["end_frame"]) for alert in overlapping)
            clear_frame=int(event["passed_or_cleared_frame"])
            latency_ms=max(0.0,(timestamps[min(last_end,len(timestamps)-1)]-timestamps[clear_frame])*1000.0)
            latencies_ms.append(latency_ms); clear.append(latency_ms<=500.0)
        critical_miss=None if not critical_hit else 1.0-float(np.mean(critical_hit))
        result["events"]={"status":"evaluable" if critical_miss is not None and latencies_ms else "not_evaluable","event_recall":float(np.mean(hit)),"critical_miss_rate":critical_miss,"false_alerts_per_minute":len(false_alerts)/(duration/60.0),"clearance_rate":float(np.mean(clear)),"clearance_p95_ms":quantile(latencies_ms,.95),"event_count":len(events),"critical_event_count":len(critical_hit),"alert_count":len(alerts),"reason":None if critical_miss is not None and latencies_ms else "critical event or observable alert clearance missing"}


def run(bundle_root: Path, prereg_path: Path, report_path: Path, evaluation_path: Path | None = None, review_consensus_path: Path | None = None) -> dict[str, Any]:
    prereg=read_json(prereg_path)
    is_r3=prereg.get("schema")=="blindassist_ustrf_sensor_replay_r3_prereg_v1"
    sources=[_source(path,prereg) for path in sorted(bundle_root.iterdir()) if path.is_dir() and (path/"bundle.json").is_file()]
    evaluation=read_json(evaluation_path) if evaluation_path else {"sources":[]}
    if evaluation_path and is_r3:
        if evaluation.get("schema")!="blindassist_ustrf_sensor_replay_r3_candidate_evaluation_v1": raise ValueError("R3 evaluation schema mismatch")
        if evaluation.get("prereg_sha256")!=sha256(prereg_path): raise ValueError("R3 evaluation prereg binding mismatch")
        if evaluation.get("candidate_alerts_frozen_before_review") is not True: raise ValueError("candidate alert trace was not frozen before review")
    review=read_json(review_consensus_path) if review_consensus_path else None
    evaluation_by_id={row["source_id"]:row for row in evaluation.get("sources",[])}
    review_by_id={row["source_id"]:row for row in (review or {}).get("sources",[])}
    for row in sources:
        source_id=row["source_id"]
        if source_id in evaluation_by_id:
            value=dict(evaluation_by_id[source_id])
            if source_id in review_by_id and review_by_id[source_id].get("route_event_admitted") is True: value["events"]=review_by_id[source_id]["events"]
            _apply_evaluation(bundle_root/source_id,row,value)
    geometry_sources=sum(bool(row["geometry_transport_passed"]) for row in sources)
    if not is_r3:
        worst=max(sources,key=lambda row: row["depth"]["median_pair_p95_residual_m"] or float("inf")) if sources else None
        missing=sorted({metric for metric in prereg["required_closure_metrics"] for row in sources if (metric=="pose_drift" and row["pose_drift"]["status"]!="evaluable") or (metric=="route_projection_error" and row["route_projection_error"]["status"]!="evaluable") or (metric in {"event_recall","false_alerts_per_minute","alert_clearance"} and row["events"]["status"]!="evaluable")})
        report={"schema":REPORT_SCHEMA,"source_count":len(sources),"sources":sources,"geometry_transport_source_pass_count":geometry_sources,"worst_source":worst["source_id"] if worst else None,"missing_closure_metrics":missing,"route_event_review_consensus":review,"algorithm_closed_loop_proven":len(sources)>=prereg["minimum_admitted_sources"] and geometry_sources==len(sources) and not missing,"hardware_selection_authorized":False,"u0_authorized":False,"model_proxy_120_authorized":False,"production_authority":False}
        report["verdict"]="MULTISOURCE_ALGORITHM_CLOSED_LOOP" if report["algorithm_closed_loop_proven"] else "DO_NOT_SELECT_HARDWARE"
        write_json(report_path,report); return report
    thresholds=prereg.get("event_thresholds",{})
    minimum_pose_fraction=float(prereg.get("pose_estimator",{}).get("minimum_estimated_fraction",0.0))
    maximum_unknown=float(prereg.get("route",{}).get("maximum_unknown_rate",1.0))
    for row in sources:
        events=row["events"]
        event_gates={
            "event_recall": events.get("event_recall") is not None and events["event_recall"]>=float(thresholds.get("minimum_event_recall",0.90)),
            "critical_miss_rate": events.get("critical_miss_rate") is not None and events["critical_miss_rate"]<=float(thresholds.get("maximum_critical_miss_rate",0.05)),
            "false_alerts_per_minute": events.get("false_alerts_per_minute") is not None and events["false_alerts_per_minute"]<=float(thresholds.get("maximum_false_alerts_per_minute",0.50)),
            "clearance_rate": events.get("clearance_rate") is not None and events["clearance_rate"]>=float(thresholds.get("minimum_clearance_rate",0.90)),
            "clearance_p95_ms": events.get("clearance_p95_ms") is not None and events["clearance_p95_ms"]<=float(thresholds.get("maximum_clearance_p95_ms",500.0)),
        }
        row["r3_gate"]={"event_metrics":event_gates,"pose_evaluable":row["pose_drift"]["status"]=="evaluable" and row["pose_drift"].get("estimated_fraction",0)>=minimum_pose_fraction,"route_evaluable":row["route_projection_error"]["status"]=="evaluable" and row["route_projection_error"].get("unknown_rate",1)>-1 and row["route_projection_error"].get("unknown_rate",1)<=maximum_unknown,"review_admitted":review_by_id.get(row["source_id"],{}).get("route_event_admitted") is True,"geometry_transport_passed":row["geometry_transport_passed"]}
        row["r3_gate"]["passed"]=all(event_gates.values()) and all(row["r3_gate"][key] for key in ("pose_evaluable","route_evaluable","review_admitted","geometry_transport_passed"))
    missing=sorted({metric for metric in prereg["required_closure_metrics"] for row in sources if not row["r3_gate"]["event_metrics"].get(metric,False)})
    worst_by_metric={}
    metric_direction={"event_recall":"min","critical_miss_rate":"max","false_alerts_per_minute":"max","clearance_rate":"min","clearance_p95_ms":"max"}
    for metric,direction in metric_direction.items():
        evaluable=[row for row in sources if row["events"].get(metric) is not None]
        if evaluable:
            selected=(min if direction=="min" else max)(evaluable,key=lambda row:row["events"][metric])
            worst_by_metric[metric]={"source_id":selected["source_id"],"value":selected["events"][metric]}
        else: worst_by_metric[metric]={"source_id":None,"value":None}
    passed=len(sources)>=prereg["minimum_admitted_sources"] and all(row["r3_gate"]["passed"] for row in sources)
    report={"schema":"blindassist_ustrf_sensor_replay_r3_report_v1","source_count":len(sources),"sources":sources,"geometry_transport_source_pass_count":geometry_sources,"worst_sources_by_metric":worst_by_metric,"missing_or_failed_closure_metrics":missing,"route_event_review_consensus":review,"algorithm_closed_loop_proven":passed,"hardware_selection_authorized":False,"u0_authorized":False,"model_proxy_120_authorized":False,"repeat_arcore_authorized":False,"production_authority":False}
    report["verdict"]="MULTISOURCE_ALGORITHM_CLOSED_LOOP" if report["algorithm_closed_loop_proven"] else "DO_NOT_SELECT_HARDWARE"
    write_json(report_path,report); return report


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--bundle-root",type=Path,required=True); parser.add_argument("--prereg",type=Path,required=True); parser.add_argument("--report",type=Path,required=True); parser.add_argument("--evaluation",type=Path); parser.add_argument("--review-consensus",type=Path); args=parser.parse_args()
    try:
        if args.report.exists(): raise ValueError(f"refusing to overwrite report: {args.report}")
        report=run(args.bundle_root.resolve(),args.prereg.resolve(),args.report.resolve(),args.evaluation.resolve() if args.evaluation else None,args.review_consensus.resolve() if args.review_consensus else None)
        summary={"verdict":report["verdict"],"sources":report["source_count"]}
        if "worst_sources_by_metric" in report: summary.update({"worst_sources":report["worst_sources_by_metric"],"missing_or_failed":report["missing_or_failed_closure_metrics"]})
        else: summary.update({"worst_source":report.get("worst_source"),"missing":report.get("missing_closure_metrics")})
        print(json.dumps(summary)); return 0 if report["algorithm_closed_loop_proven"] else 3
    except (OSError,ValueError,KeyError) as error: print(json.dumps({"ok":False,"error":str(error)})); return 2


if __name__=="__main__": raise SystemExit(main())
