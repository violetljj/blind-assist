from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


SOURCES = {
    "lilocbench_dynamics_0_front": {
        "name": "dynamics_0",
        "bundle": Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1/dynamics_0-normalized-v1/lilocbench_dynamics_0_front"),
        "route": Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1/dynamics_0-candidate-evaluation-frozen-v1.json"),
        "events": Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1/dynamics_0-review-consensus-v2.json"),
    },
    "lilocbench_lt_changes_dynamics_0_front": {
        "name": "lt_changes_dynamics_0",
        "bundle": Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1/lt_changes_dynamics_0-normalized-v1/lilocbench_lt_changes_dynamics_0_front"),
        "route": Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1/lt_changes_dynamics_0-candidate-v1.json"),
        "events": Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1/lt_changes_dynamics_0-review-v1/review-consensus-v2.json"),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-encoder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--proposal-conf", type=float, default=0.01)
    args = parser.parse_args()
    final_output = args.output_dir / "target_truth_review_bundle.json"
    if final_output.exists():
        raise SystemExit(f"refusing to overwrite frozen review bundle: {final_output}")
    checkpoint_dir = args.output_dir / "window-checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    windows_payload = read_json(args.windows)
    model_path = args.model.resolve()
    prompt_encoder = args.prompt_encoder.resolve()
    if prompt_encoder.name != "mobileclip_blt.ts" or not prompt_encoder.is_file():
        raise ValueError("prompt encoder must be an existing mobileclip_blt.ts")
    model = YOLO(str(model_path))
    previous_cwd = Path.cwd()
    try:
        os.chdir(prompt_encoder.parent)
        model.set_classes(["person"])
    finally:
        os.chdir(previous_cwd)
    source_outputs: list[dict] = []
    for source_id, source in SOURCES.items():
        bundle = read_json(source["bundle"] / "bundle.json")
        frame_rows = {
            row["frame_id"]: row
            for row in map(json.loads, (source["bundle"] / "frames.jsonl").read_text(encoding="utf-8").splitlines())
        }
        route_payload = read_json(source["route"])
        route_source = route_payload["sources"][0]
        route_by_frame = {row["frame_id"]: row for row in route_source["route_truth"]}
        events = read_json(source["events"])["sources"][0]["events"]
        event_by_id = {row["event_id"]: row for row in events}
        source_windows = [row for row in windows_payload["windows"] if row["source_id"] == source_id]
        output_windows: list[dict] = []
        for window in source_windows:
            checkpoint = checkpoint_dir / f"{window['window_id']}.json"
            if checkpoint.exists():
                saved = read_json(checkpoint)
                if saved.get("window_id") != window["window_id"] or saved.get("source_id") != source_id:
                    raise ValueError(f"checkpoint identity mismatch: {checkpoint}")
                output_windows.append(saved)
                print(f"truth_proposal_resume {source['name']} {window['window_id']} frames={len(saved['frames'])}", flush=True)
                continue
            frame_ids = [f"{value:06d}" for value in range(int(window["start_frame"]), int(window["end_frame"]) + 1)]
            paths = [Path(bundle["source_root"]) / frame_rows[frame_id]["rgb_path"] for frame_id in frame_ids]
            proposal_rows: list[dict] = []
            for batch_start in range(0, len(paths), 8):
                batch_paths = paths[batch_start:batch_start + 8]
                batch_ids = frame_ids[batch_start:batch_start + 8]
                results = model.predict(
                    [str(path) for path in batch_paths], imgsz=args.imgsz, conf=args.proposal_conf,
                    iou=0.5, device=0, half=True, batch=len(batch_paths), verbose=False,
                )
                for frame_id, path, result in zip(batch_ids, batch_paths, results, strict=True):
                    boxes = []
                    if result.boxes is not None:
                        for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), strict=True):
                            boxes.append({"bbox_xyxy": [round(float(v), 3) for v in box], "proposal_confidence": round(float(confidence), 6)})
                    route = route_by_frame.get(frame_id, {"status": "missing", "uv": None})
                    proposal_rows.append({
                        "frame_id": frame_id,
                        "image_path": str(path),
                        "image_sha256": sha256(path),
                        "route_status": route.get("status"),
                        "route_uv": route.get("uv"),
                        "person_proposals": boxes,
                        "review": {"status": "pending", "all_person_boxes_xyxy": None, "confirmed_absent": None, "target_box_index": None},
                    })
                del results
            positive = "positive-" in window["window_id"]
            event_id = window["window_id"].split("positive-", 1)[1] if positive else None
            event = event_by_id.get(event_id) if event_id else None
            window_output = {
                **window,
                "truth_role": "target_event" if positive else "negative_all_person",
                "event_lifecycle": event,
                "target_identity": {"event_scoped_id": f"{source_id}/{event_id}/target_person" if event_id else None, "review_status": "pending"},
                "frames": proposal_rows,
            }
            checkpoint.write_text(json.dumps(window_output, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
            output_windows.append(window_output)
            print(f"truth_proposal {source['name']} {window['window_id']} frames={len(frame_ids)}", flush=True)
        source_outputs.append({
            "source_id": source_id,
            "source_name": source["name"],
            "route_input_sha256": sha256(source["route"]),
            "event_consensus_sha256": sha256(source["events"]),
            "windows": output_windows,
        })
    payload = {
        "schema": "blindassist_ustrf_detector_target_truth_review_bundle_v1",
        "authority": "annotation_proposal_only_not_truth_until_review_frozen",
        "baseline_detector_outputs_accessed": False,
        "annotation_model_role": "proposal_only_never_candidate_or_promotion_credit",
        "annotation_model_path": str(model_path),
        "annotation_model_sha256": sha256(model_path),
        "prompt_encoder_path": str(prompt_encoder),
        "prompt_encoder_sha256": sha256(prompt_encoder),
        "windows_sha256": sha256(args.windows),
        "proposal_imgsz": args.imgsz,
        "proposal_confidence": args.proposal_conf,
        "sources": source_outputs,
    }
    final_output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"sources": len(source_outputs), "windows": sum(len(row["windows"]) for row in source_outputs), "output_sha256": sha256(final_output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
