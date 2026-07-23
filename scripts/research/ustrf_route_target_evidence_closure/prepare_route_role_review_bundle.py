from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from contract import load_json, sha256_file, validate_prereg


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bound_path(repo: Path, binding: dict[str, str]) -> Path:
    path = repo / binding["path"]
    if sha256_file(path) != binding["sha256"]:
        raise ValueError(f"bound input hash mismatch: {path}")
    return path


def frame_range(window: dict[str, Any]) -> list[str]:
    return [f"{index:06d}" for index in range(int(window["start_frame"]), int(window["end_frame"]) + 1)]


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def blind_projection(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    review_sources = []
    scorer_sources = []
    for source in payload["sources"]:
        review_windows = []
        scorer_windows = []
        for window in source["windows"]:
            blind_id = "blind_" + hashlib.sha256(
                f"{source['source_id']}|{window['window_id']}".encode("utf-8")
            ).hexdigest()[:20]
            review_frames = []
            for frame in window["frames"]:
                review_frame = {key: value for key, value in frame.items() if key != "legacy_target_lifecycle_anchor"}
                review_frame["person_seed_boxes"] = []
                for seed in frame["person_seed_boxes"]:
                    copied = dict(seed)
                    hint = copied.get("person_identity_hint")
                    if hint is not None:
                        copied["person_identity_hint"] = "person_hint_" + hashlib.sha256(hint.encode("utf-8")).hexdigest()[:12]
                    review_frame["person_seed_boxes"].append(copied)
                review_frames.append(review_frame)
            frame_inventory = [{"frame_id": frame["frame_id"], "image_sha256": frame["image_sha256"]} for frame in review_frames]
            review_windows.append({"blind_window_id": blind_id, "frames": review_frames})
            scorer_windows.append({
                "blind_window_id": blind_id,
                "window_id": window["window_id"],
                "window_type": window["window_type"],
                "event_id": window["event_id"],
                "critical": window["critical"],
                "truth_anchors": window["truth_anchors"],
                "frame_inventory_sha256": canonical_sha256(frame_inventory),
                "legacy_target_lifecycle_anchors": [
                    {"frame_id": frame["frame_id"], "anchor": frame["legacy_target_lifecycle_anchor"]}
                    for frame in window["frames"] if frame["legacy_target_lifecycle_anchor"] is not None
                ],
            })
        review_sources.append({"source_id": source["source_id"], "windows": review_windows})
        scorer_sources.append({"source_id": source["source_id"], "windows": scorer_windows})
    review = {key: value for key, value in payload.items() if key != "sources"}
    review.update({
        "schema": "blindassist_ustrf_route_role_blind_review_bundle_r2",
        "window_labels_exposed": False,
        "future_truth_anchors_exposed": False,
        "sources": review_sources,
    })
    scorer = {
        "schema": "blindassist_ustrf_route_role_isolated_scorer_binding_r1",
        "authority": "isolated_scorer_only_never_reviewer_input",
        "review_bundle_sha256": None,
        "source_count": len(scorer_sources),
        "window_count": sum(len(source["windows"]) for source in scorer_sources),
        "sources": scorer_sources,
    }
    return review, scorer


def materialize(config: dict[str, Any], *, repo: Path) -> dict[str, Any]:
    validate_prereg(config, repo=repo)
    truth_path = bound_path(repo, config["parent_bindings"]["frozen_person_truth"])
    truth = load_json(truth_path)
    windows = load_json(bound_path(repo, config["seen_inputs"]["windows"]))["windows"]
    truth_sources = {source["source_id"]: source for source in truth["sources"]}
    windows_by_source: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        windows_by_source.setdefault(window["source_id"], []).append(window)

    output_sources = []
    total_frames = image_hashes_verified = seed_box_count = 0
    for source_id, bindings in config["seen_inputs"]["sources"].items():
        frames_path = bound_path(repo, bindings["frames"])
        source_bundle = load_json(bound_path(repo, bindings["bundle"]))
        if source_bundle.get("frames_sha256") != bindings["frames"]["sha256"]:
            raise ValueError(f"source bundle frames binding mismatch: {source_id}")
        source_root = Path(source_bundle["source_root"]).resolve()
        if not source_root.is_relative_to((repo / "artifacts.local").resolve()):
            raise ValueError(f"source root escapes artifacts.local: {source_root}")
        route_path = bound_path(repo, bindings["route"])
        frame_rows = {row["frame_id"]: row for row in read_jsonl(frames_path)}
        route_payload = load_json(route_path)
        route_source = next(source for source in route_payload["sources"] if source["source_id"] == source_id)
        route_rows = {row["frame_id"]: row for row in route_source["route_predictions"]}
        truth_source = truth_sources[source_id]
        target_events = {event["event_id"]: event for event in truth_source["target_events"]}
        negatives = {window["window_id"]: window for window in truth_source["negative_windows"]}
        source_windows = []
        for window in windows_by_source[source_id]:
            target_frames = {}
            negative_frames = {}
            if window["window_type"] == "positive":
                event = target_events[window["event_id"]]
                target_frames = {frame["frame_id"]: frame for frame in event["frames"]}
            else:
                event = None
                negative_frames = {frame["frame_id"]: frame for frame in negatives[window["window_id"]]["frames"]}
            review_frames = []
            for frame_id in frame_range(window):
                source_frame = frame_rows[frame_id]
                image_path = source_root / source_frame["rgb_path"]
                if not image_path.is_file() or sha256_file(image_path) != source_frame["rgb_sha256"]:
                    raise ValueError(f"RGB evidence mismatch: {image_path}")
                image_hashes_verified += 1
                route = route_rows[frame_id]
                route_age_ms = max(0.0, (float(route["timestamp_s"]) - float(route["predicted_at_s"])) * 1000.0)
                seeds = []
                legacy_target_lifecycle_anchor = None
                if frame_id in target_frames:
                    frame = target_frames[frame_id]
                    if frame["target_bbox_xyxy"] is not None:
                        seeds.append({
                            "seed_type": "reviewed_target_truth",
                            "person_identity_hint": event["target_person_identity"],
                            "visibility": frame["visible_state"],
                            "bbox_xyxy": frame["target_bbox_xyxy"],
                            "bbox_origin": frame["bbox_origin"],
                        })
                    else:
                        legacy_target_lifecycle_anchor = {
                            "person_identity_hint": event["target_person_identity"],
                            "visible_state": frame["visible_state"],
                            "bbox_origin": frame["bbox_origin"],
                            "route_role_authority": "none_must_remain_unknown_without_independent_exit_evidence",
                        }
                if frame_id in negative_frames:
                    frame = negative_frames[frame_id]
                    for index, person_box in enumerate(frame["all_person_boxes"]):
                        seeds.append({
                            "seed_type": "reviewed_unlinked_person_box",
                            "person_identity_hint": None,
                            "frame_local_index": index,
                            "visibility": "visible",
                            "bbox_xyxy": person_box["bbox_xyxy"],
                            "bbox_origin": person_box["origin"],
                        })
                seed_box_count += len(seeds)
                review_frames.append({
                    "frame_id": frame_id,
                    "source_capture_timestamp_ns": round(float(source_frame["rgb_timestamp_s"]) * 1_000_000_000),
                    "image_path": str(image_path.relative_to(repo)).replace("\\", "/"),
                    "image_sha256": source_frame["rgb_sha256"],
                    "route_receipt_id": f"{source_id}:{bindings['route']['sha256']}",
                    "route_status": route["status"],
                    "route_uv": route.get("uv"),
                    "route_evidence_age_ms": route_age_ms,
                    "person_seed_boxes": seeds,
                    "legacy_target_lifecycle_anchor": legacy_target_lifecycle_anchor,
                    "full_frame_person_discovery_required": True,
                })
                total_frames += 1
            source_windows.append({
                "window_id": window["window_id"],
                "window_type": window["window_type"],
                "event_id": window["event_id"],
                "critical": bool(window["critical"]),
                "truth_anchors": window["truth_anchors"],
                "frames": review_frames,
            })
        output_sources.append({"source_id": source_id, "windows": source_windows})
    return {
        "schema": "blindassist_ustrf_route_role_review_bundle_r1",
        "authority": "seen_diagnostic_candidate_hidden_truth_proposal_only",
        "config_sha256": sha256_file(repo / "configs/ustrf_route_target_evidence_closure_r1.json"),
        "candidate_alerts_exposed": False,
        "detector_outputs_exposed": False,
        "all_person_discovery_required": True,
        "route_roles": list(config["route_role_truth"]["roles"]),
        "unknown_is_abstention_not_role": True,
        "source_count": len(output_sources),
        "window_count": sum(len(source["windows"]) for source in output_sources),
        "frame_count": total_frames,
        "image_hashes_verified": image_hashes_verified,
        "seed_box_count": seed_box_count,
        "sources": output_sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scorer-binding-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite review bundle: {args.output}")
    if args.scorer_binding_output.exists():
        raise SystemExit(f"refusing to overwrite scorer binding: {args.scorer_binding_output}")
    repo = args.repo.resolve()
    internal = materialize(load_json(args.config), repo=repo)
    payload, scorer = blind_projection(internal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scorer["review_bundle_sha256"] = sha256_file(args.output)
    args.scorer_binding_output.parent.mkdir(parents=True, exist_ok=True)
    args.scorer_binding_output.write_text(json.dumps(scorer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "schema", "source_count", "window_count", "frame_count", "image_hashes_verified", "seed_box_count"
    )} | {"review_bundle_sha256": sha256_file(args.output), "scorer_binding_sha256": sha256_file(args.scorer_binding_output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
