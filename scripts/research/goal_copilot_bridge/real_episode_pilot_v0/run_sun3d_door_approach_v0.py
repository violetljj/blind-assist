"""Freeze, materialize, and run V0 on one native-truth SUN3D door approach."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0 import provider_adapter
from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as dino
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0 import run_cmp_facade_native_door_89 as cmp


SCHEMA_VERSION = "sun3d_native_door_approach_v0"
OFFICIAL_LIST_URL = "http://sun3d.cs.princeton.edu/listNow.html"
DATA_ROOT = "http://sun3d.cs.princeton.edu/data"
GOAL_TEXT = "the door"
OBSERVATION_COUNT = 15
ARRIVAL_THRESHOLD_M = 2.0
IOU_THRESHOLD = 0.5
CLAIM_CEILING = (
    "PRERECORDED_REAL_RGBD_DOOR_APPROACH_CURRENT_FRAME_GROUNDING_ONLY_"
    "NO_CLOSED_LOOP_CONTROL_USER_SAFETY_OR_PRODUCT_CLAIM"
)


class RunError(RuntimeError):
    pass


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fetch(url: str) -> bytes:
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response.content


def _fully_annotated_sequences(html: str) -> list[str]:
    match = re.search(
        r"<h2>Fully Annotated Sequences with Pose Correction</h2>(.*?)(?:<h2>|$)",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise RunError("official SUN3D page lacks the fully annotated section")
    result = []
    for name in re.findall(r"player/\?name=([^\"&]+)", match.group(1)):
        if name not in result:
            result.append(name)
    if not result:
        raise RunError("official SUN3D fully annotated roster is empty")
    return result


def _exact_door(name: str) -> bool:
    return name.strip().lower() == "door"


def _camera_centroid(points: Sequence[Sequence[float]]) -> tuple[float, float, float] | None:
    valid = [point for point in points if math.sqrt(sum(float(value) ** 2 for value in point)) < 20.0]
    if len(valid) < 3:
        return None
    return tuple(statistics.median(float(point[axis]) for point in valid) for axis in range(3))


def _world_point(matrix: Sequence[float], point: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3],
        matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7],
        matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11],
    )


def _camera_point(matrix: Sequence[float], world: Sequence[float]) -> tuple[float, float, float]:
    delta = [world[axis] - matrix[(axis + 1) * 4 - 1] for axis in range(3)]
    return tuple(sum(matrix[row * 4 + axis] * delta[row] for row in range(3)) for axis in range(3))


def _uniform_indices(start: int, end: int, count: int) -> list[int]:
    if count < 2 or end <= start or end - start + 1 < count:
        raise RunError("episode window cannot support the frozen observation count")
    indices = [round(start + offset * (end - start) / (count - 1)) for offset in range(count)]
    if len(set(indices)) != count or indices[0] != start or indices[-1] != end:
        raise RunError("uniform observation selection is not unique or endpoint preserving")
    return indices


def _episode_from_metadata(sequence: str, annotation: Mapping[str, Any], extrinsic_text: str) -> dict[str, Any] | None:
    door_ids = [
        index for index, item in enumerate(annotation["objects"])
        if item is not None and _exact_door(item["name"])
    ]
    if not door_ids:
        return None
    target_id = min(door_ids)
    values = [float(value) for value in extrinsic_text.split()]
    matrices = [values[index:index + 12] for index in range(0, len(values), 12)]
    if not matrices or len(matrices[-1]) != 12:
        raise RunError("SUN3D extrinsics are malformed")

    visible_by_frame: dict[int, dict[str, Any]] = {}
    world_centroids = []
    for frame_id, frame in enumerate(annotation["frames"]):
        polygons = [polygon for polygon in frame.get("polygon", []) if polygon["object"] == target_id]
        if not polygons:
            continue
        polygon = max(
            polygons,
            key=lambda item: (max(item["x"]) - min(item["x"])) * (max(item["y"]) - min(item["y"])),
        )
        center = _camera_centroid(polygon["XYZ"])
        if center is None:
            continue
        source_frame_id = int(annotation["fileList"][frame_id].split("-", 1)[0]) - 1
        if not 0 <= source_frame_id < len(matrices):
            raise RunError("SUN3D frame-to-extrinsics join is out of bounds")
        world_centroids.append(_world_point(matrices[source_frame_id], center))
        visible_by_frame[frame_id] = {
            "polygon_xy": [[float(x), float(y)] for x, y in zip(polygon["x"], polygon["y"])],
            "native_camera_range_m": math.sqrt(sum(value * value for value in center)),
        }
    if len(world_centroids) < 5:
        return None
    target_world = [statistics.median(point[axis] for point in world_centroids) for axis in range(3)]

    frame_rows = []
    for frame_id, filename in enumerate(annotation["fileList"]):
        source_frame_id = int(filename.split("-", 1)[0]) - 1
        matrix = matrices[source_frame_id]
        target_camera = _camera_point(matrix, target_world)
        distance = math.sqrt(sum(value * value for value in target_camera))
        frame_rows.append({
            "frame_id": frame_id,
            "source_frame_id": source_frame_id,
            "source_filename": filename,
            "map_trajectory_range_m": distance,
            "map_trajectory_bearing_deg": math.degrees(math.atan2(target_camera[0], target_camera[2])),
            "visibility": "VISIBLE" if frame_id in visible_by_frame else "NOT_VISIBLE",
            **visible_by_frame.get(frame_id, {}),
        })

    starts = [row for row in frame_rows if row["visibility"] == "VISIBLE" and 4.0 <= row["map_trajectory_range_m"] <= 10.0]
    if not starts:
        return None
    start = starts[0]
    terminals = [
        row for row in frame_rows
        if row["frame_id"] > start["frame_id"]
        and row["visibility"] == "VISIBLE"
        and row["map_trajectory_range_m"] < ARRIVAL_THRESHOLD_M
    ]
    if not terminals:
        return None
    terminal = terminals[0]
    selected = [frame_rows[index] for index in _uniform_indices(start["frame_id"], terminal["frame_id"], OBSERVATION_COUNT)]
    for index, row in enumerate(selected, start=1):
        row["observation_id"] = f"sun3d-door-{index:03d}"
        row["image_url"] = f"{DATA_ROOT}/{sequence}/image/{row['source_filename']}"
        row["arrival_truth"] = row["map_trajectory_range_m"] < ARRIVAL_THRESHOLD_M
        if row["visibility"] == "VISIBLE":
            xs = [point[0] for point in row["polygon_xy"]]
            ys = [point[1] for point in row["polygon_xy"]]
            row["native_bbox_xyxy"] = [min(xs), min(ys), max(xs), max(ys)]
    return {
        "sequence": sequence,
        "target_object_id": target_id,
        "target_object_name": annotation["objects"][target_id]["name"],
        "target_world_xyz_map_derived": target_world,
        "start_frame_id": start["frame_id"],
        "terminal_frame_id": terminal["frame_id"],
        "observations": selected,
    }


def freeze(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    roster_path = output_dir / "frozen-roster-private-truth.json"
    if roster_path.exists():
        raise RunError("frozen SUN3D roster already exists; refusing reselection")
    list_bytes = _fetch(OFFICIAL_LIST_URL)
    sequences = _fully_annotated_sequences(list_bytes.decode("utf-8"))
    selected = None
    selected_sources = None
    for sequence in sequences:
        base = f"{DATA_ROOT}/{sequence}"
        annotation_bytes = _fetch(f"{base}/annotation/index.json")
        annotation = json.loads(annotation_bytes)
        extrinsics_name = annotation["extrinsics"]
        extrinsics_bytes = _fetch(f"{base}/extrinsics/{extrinsics_name}")
        candidate = _episode_from_metadata(sequence, annotation, extrinsics_bytes.decode("utf-8"))
        if candidate is not None:
            selected = candidate
            selected_sources = {
                "official_list_url": OFFICIAL_LIST_URL,
                "official_list_sha256": _sha256_bytes(list_bytes),
                "annotation_url": f"{base}/annotation/index.json",
                "annotation_sha256": _sha256_bytes(annotation_bytes),
                "extrinsics_url": f"{base}/extrinsics/{extrinsics_name}",
                "extrinsics_sha256": _sha256_bytes(extrinsics_bytes),
            }
            break
    if selected is None or selected_sources is None:
        raise RunError("no fully annotated SUN3D sequence satisfies the frozen door-approach rule")
    roster = {
        "schema_version": SCHEMA_VERSION,
        "data_role": "FRESH_CONFIRMATION",
        "selection_rule": (
            "OFFICIAL_FULLY_ANNOTATED_ORDER_FIRST_EXACT_DOOR_WITH_5_VALID_POLYGONS_"
            "FIRST_VISIBLE_4_TO_10M_THEN_FIRST_VISIBLE_LT_2M_UNIFORM_15"
        ),
        "goal_text": GOAL_TEXT,
        "truth_authority": "NATIVE_POLYGON_PLUS_MAP_TRAJECTORY_DERIVED",
        "provider_calls": 0,
        "teacher_calls": 0,
        "pixels_downloaded_before_freeze": 0,
        "source": selected_sources,
        **selected,
    }
    roster["content_sha256"] = materializer.content_sha256(roster)
    _atomic_json(roster_path, roster)
    print(json.dumps({"roster": str(roster_path), "sha256": _sha256_file(roster_path), "sequence": roster["sequence"]}))
    return roster


def download(roster_path: Path, roster_sha256: str, output_dir: Path) -> dict[str, Any]:
    roster_path = roster_path.resolve()
    if _sha256_file(roster_path) != roster_sha256:
        raise RunError("frozen roster hash mismatch before pixel download")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "pixels-manifest.json"
    if manifest_path.exists():
        raise RunError("pixels manifest already exists; refusing rematerialization")
    pixels_dir = output_dir / "pixels"
    pixels_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for observation in roster["observations"]:
        target = pixels_dir / f"{observation['observation_id']}.jpg"
        payload = _fetch(observation["image_url"])
        if target.exists() and _sha256_file(target) != _sha256_bytes(payload):
            raise RunError(f"owned pixel path conflicts: {target}")
        if not target.exists():
            target.write_bytes(payload)
        with Image.open(target) as image:
            width, height = image.size
            image.verify()
        rows.append({
            "observation_id": observation["observation_id"],
            "relative_path": target.relative_to(output_dir).as_posix(),
            "sha256": _sha256_file(target),
            "width": width,
            "height": height,
        })
    manifest = {"schema_version": f"{SCHEMA_VERSION}_pixels", "roster_sha256": roster_sha256, "observations": rows}
    manifest["content_sha256"] = materializer.content_sha256(manifest)
    _atomic_json(manifest_path, manifest)
    print(json.dumps({"pixels_manifest": str(manifest_path), "sha256": _sha256_file(manifest_path), "count": len(rows)}))
    return manifest


def _build_episode(item: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = []
    for rank, proposal in enumerate(proposals, start=1):
        x0, y0, x1, y1 = proposal["bbox_xyxy"]
        candidates.append({
            "candidate_id": f"gdino-{item['observation_id']}-{rank:03d}",
            "region": {
                "frame_id": item["observation_id"], "coordinate_space": "NORMALIZED_XYXY",
                "x_min": x0 / item["width"], "y_min": y0 / item["height"],
                "x_max": x1 / item["width"], "y_max": y1 / item["height"],
            },
            "category_label": proposal["label"], "proposal_score": proposal["score"], "provider_rank": rank,
        })
    return {
        "episode_id": item["observation_id"], "goal_text": GOAL_TEXT,
        "image_path": item["absolute_path"], "candidates": candidates,
        "evaluator_episode": {
            "goal_spec": {"target_name": GOAL_TEXT},
            "observation_window": {"frame_ids": [item["observation_id"]], "start_timestamp_ms": 0, "end_timestamp_ms": 0},
        },
    }


def _evaluate(
    truth_rows: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    decision_by_id = {item["episode_id"]: item for item in decisions}
    seen = False
    rows = []
    for truth, episode in zip(truth_rows, episodes):
        visible = truth["visibility"] == "VISIBLE"
        if visible:
            truth_state = "VISIBLE"
            seen = True
        else:
            truth_state = "LOST" if seen else "NOT_VISIBLE"
        decision = decision_by_id[episode["episode_id"]]
        selected = decision["selected_candidate_ids"][0] if decision["selected_candidate_ids"] else None
        correct = []
        if visible:
            native = [
                truth["native_bbox_xyxy"][0] / truth["width"], truth["native_bbox_xyxy"][1] / truth["height"],
                truth["native_bbox_xyxy"][2] / truth["width"], truth["native_bbox_xyxy"][3] / truth["height"],
            ]
            correct = [
                candidate["candidate_id"] for candidate in episode["candidates"]
                if cmp.iou([candidate["region"][key] for key in ("x_min", "y_min", "x_max", "y_max")], native) >= IOU_THRESHOLD
            ]
            if not correct:
                outcome = "PROPOSAL_MISS"
            elif decision["action"] != "SELECT":
                outcome = "REFERENT_SELECTION_ABSTAINED_WITH_USABLE_PROPOSAL"
            elif selected not in correct:
                outcome = "WRONG_CONFIDENT_GUIDANCE"
            else:
                outcome = "CORRECT_GROUNDING"
        else:
            outcome = "TARGET_NOT_VISIBLE" if decision["action"] != "SELECT" else "WRONG_CONFIDENT_GUIDANCE_ON_NOT_VISIBLE"
        rows.append({
            "observation_id": truth["observation_id"], "visibility_truth": truth["visibility"],
            "episode_truth_state": truth_state, "arrival_truth": truth["arrival_truth"],
            "map_trajectory_range_m": truth["map_trajectory_range_m"],
            "map_trajectory_bearing_deg": truth["map_trajectory_bearing_deg"],
            "proposal_count": len(episode["candidates"]), "correct_candidate_ids": correct,
            "brain_action": decision["action"], "selected_candidate_id": selected, "outcome": outcome,
        })
    visible_rows = [row for row in rows if row["visibility_truth"] == "VISIBLE"]
    usable = [row for row in visible_rows if row["correct_candidate_ids"]]
    correct = [row for row in visible_rows if row["outcome"] == "CORRECT_GROUNDING"]
    wrong = [row for row in rows if row["outcome"].startswith("WRONG_CONFIDENT_GUIDANCE")]
    return {
        "truth_coverage": {"numerator": len(rows), "denominator": len(rows), "authority": "NATIVE_PLUS_MAP_TRAJECTORY_DERIVED"},
        "observation_count": len(rows), "visible_count": len(visible_rows),
        "not_visible_count": len(rows) - len(visible_rows),
        "lost_after_visible_count": sum(row["episode_truth_state"] == "LOST" for row in rows),
        "arrival_count": sum(row["arrival_truth"] for row in rows),
        "proposal_availability_visible": {"numerator": len(usable), "denominator": len(visible_rows)},
        "selection_accuracy_given_usable_proposal": {"numerator": len(correct), "denominator": len(usable)},
        "wrong_confident_guidance_all_observations": {"numerator": len(wrong), "denominator": len(rows)},
        "outcome_counts": dict(sorted(Counter(row["outcome"] for row in rows).items())),
        "observations": rows,
    }


def run(
    roster_path: Path, roster_sha256: str, pixels_manifest_path: Path, pixels_manifest_sha256: str,
    run_dir: Path, codex_exe: Path, model_dir: Path, batch_size: int,
) -> dict[str, Any]:
    roster_path, pixels_manifest_path = roster_path.resolve(), pixels_manifest_path.resolve()
    if _sha256_file(roster_path) != roster_sha256 or _sha256_file(pixels_manifest_path) != pixels_manifest_sha256:
        raise RunError("frozen SUN3D input hash mismatch")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    pixels = json.loads(pixels_manifest_path.read_text(encoding="utf-8"))
    if pixels["roster_sha256"] != roster_sha256 or len(roster["observations"]) != OBSERVATION_COUNT:
        raise RunError("SUN3D roster/pixels contract mismatch")
    if roster["provider_calls"] != 0 or roster["teacher_calls"] != 0:
        raise RunError("SUN3D roster was not frozen before model calls")
    provider_lock = provider_adapter.preflight_provider(codex_exe=codex_exe, model_dir=model_dir)
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise RunError("formal SUN3D run directory already exists; refusing replay")
    run_dir.mkdir(parents=True)
    _atomic_json(run_dir / "provider-lock.json", provider_lock)

    pixels_by_id = {item["observation_id"]: item for item in pixels["observations"]}
    public_rows, metadata, truth_rows = [], [], []
    for truth in roster["observations"]:
        pixel = pixels_by_id[truth["observation_id"]]
        path = (pixels_manifest_path.parent / pixel["relative_path"]).resolve()
        if _sha256_file(path) != pixel["sha256"]:
            raise RunError(f"SUN3D pixel hash mismatch: {truth['observation_id']}")
        public = {
            "observation_id": truth["observation_id"], "absolute_path": str(path),
            "width": pixel["width"], "height": pixel["height"],
        }
        public_rows.append(public)
        truth_rows.append(dict(truth, width=pixel["width"], height=pixel["height"]))
        metadata.append({"id": truth["observation_id"], "path": str(path), "image_sha256": pixel["sha256"]})

    proposals, proposal_runtime = dino.run_inference(model_dir, metadata)
    _atomic_json(run_dir / "proposal-provider-output.json", {"runtime": proposal_runtime, "outputs": proposals})
    episodes = [_build_episode(item, result["proposals"]) for item, result in zip(public_rows, proposals)]
    _atomic_json(run_dir / "public-provider-input.json", {"episodes": episodes})
    decisions, receipts = cmp.run_brain(
        episodes=episodes, run_dir=run_dir / "brain", executable=codex_exe,
        model=provider_adapter.CODEX_MODEL, reasoning_effort=provider_adapter.CODEX_REASONING_EFFORT,
        batch_size=batch_size,
    )
    evaluation = _evaluate(truth_rows, episodes, decisions)
    report = {
        "schema_version": f"{SCHEMA_VERSION}_result", "roster_sha256": roster_sha256,
        "pixels_manifest_sha256": pixels_manifest_sha256, "provider_lock": provider_lock,
        "provider_calls": len(receipts), "provider_attempts": len(receipts), "provider_in_doubt": 0,
        "teacher_calls": 0, "baseline_reruns": 0, "brain_batch_receipts": receipts,
        "raw_brain_decisions": decisions, "evaluation": evaluation,
        "arrival_semantics": "TRUTH_ONLY_BASELINE_HAS_NO_COMPLETION_OUTPUT",
        "claim_ceiling": CLAIM_CEILING,
    }
    report["content_sha256"] = materializer.content_sha256(report)
    _atomic_json(run_dir / "final-report.json", report)
    print(json.dumps({"report": str(run_dir / 'final-report.json'), "content_sha256": report["content_sha256"], "evaluation": evaluation["outcome_counts"]}))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--output-dir", type=Path, required=True)
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--roster", type=Path, required=True)
    download_parser.add_argument("--roster-sha256", required=True)
    download_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--roster", type=Path, required=True)
    run_parser.add_argument("--roster-sha256", required=True)
    run_parser.add_argument("--pixels-manifest", type=Path, required=True)
    run_parser.add_argument("--pixels-manifest-sha256", required=True)
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--codex-exe", type=Path, default=Path(r"E:\codex-tools\bin\codex.exe"))
    run_parser.add_argument("--model-dir", type=Path, required=True)
    run_parser.add_argument("--batch-size", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args(argv)
    if args.phase == "freeze":
        freeze(args.output_dir)
    elif args.phase == "download":
        download(args.roster, args.roster_sha256, args.output_dir)
    else:
        run(
            args.roster, args.roster_sha256, args.pixels_manifest, args.pixels_manifest_sha256,
            args.run_dir, args.codex_exe.resolve(), args.model_dir.resolve(), args.batch_size,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
