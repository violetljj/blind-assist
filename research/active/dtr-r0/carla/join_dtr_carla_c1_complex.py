"""Join four fresh-server C1 sensor shards into sealed model/evaluator roots."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from html import escape
from pathlib import Path
from typing import Any

from dtr_carla_c1_complex import (
    EXPERIMENT_ID,
    build_plan_receipt,
    forbidden_model_paths,
    sha256_file,
    trajectory_position,
    validate_protocol,
    write_json_atomic,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rounded(value: Any, digits: int = 5) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _rounded(child, digits) for key, child in sorted(value.items())}
    if isinstance(value, list):
        return [_rounded(child, digits) for child in value]
    return value


def replay_projection(row: dict[str, Any]) -> dict[str, Any]:
    actors: dict[str, Any] = {}
    for key, actor in sorted(row["actors"].items()):
        actors[key] = {
            "track_id": actor["track_id"],
            "asset_key": actor["asset_key"],
            "scenario_role": actor["scenario_role"],
            "kind": actor["kind"],
            "actual_blueprint": actor["actual_blueprint"],
            "transform": _rounded(actor["transform"]),
            "local_position": _rounded(actor["local_position"]),
            "command_velocity": _rounded(actor["command_velocity"]),
            "bounding_box": _rounded(actor["bounding_box"]),
        }
    truth = row["truth"]
    return {
        "episode_id": row["episode_id"],
        "sample_index": int(row["sample_index"]),
        "time_s": round(float(row["time_s"]), 8),
        "plan_receipt_sha256": row["plan_receipt_sha256"],
        "plan_authority": row["plan_authority"],
        "actors": actors,
        "truth": {
            "scenario_role": truth["scenario_role"],
            "twin_role": truth["twin_role"],
            "expected_contact": bool(truth["expected_contact"]),
            "current_contact": bool(truth["current_contact"]),
            "minimum_distance_m": round(float(truth["minimum_distance_m"]), 5),
            "responsible_asset": sorted(truth["responsible_asset"]),
            "collision_polygons_xy": _rounded(truth["collision_polygons_xy"]),
            "future_contact_within_horizon": bool(
                truth["future_contact_within_horizon"]
            ),
            "realized_time_to_contact_seconds": (
                round(float(truth["realized_time_to_contact_seconds"]), 5)
                if truth["realized_time_to_contact_seconds"] is not None
                else None
            ),
        },
    }


def compare_replays(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    candidate_sensor: str,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(reference) != len(candidate):
        return [
            {
                "sensor": candidate_sensor,
                "reason": "frame_count",
                "reference": len(reference),
                "candidate": len(candidate),
            }
        ]
    for first, second in zip(reference, candidate, strict=True):
        if replay_projection(first) != replay_projection(second):
            failures.append(
                {
                    "sensor": candidate_sensor,
                    "episode_id": first["episode_id"],
                    "sample_index": first["sample_index"],
                    "reason": "actual_replay_state_mismatch",
                }
            )
            if len(failures) >= 50:
                break
    return failures


def _orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _on_segment(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return (
        min(first[0], second[0]) - 1e-7 <= point[0] <= max(first[0], second[0]) + 1e-7
        and min(first[1], second[1]) - 1e-7
        <= point[1]
        <= max(first[1], second[1]) + 1e-7
        and abs(_orientation(first, second, point)) <= 1e-7
    )


def segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    values = (_orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b))
    if values[0] * values[1] < 0.0 and values[2] * values[3] < 0.0:
        return True
    return (
        (abs(values[0]) <= 1e-7 and _on_segment(c, a, b))
        or (abs(values[1]) <= 1e-7 and _on_segment(d, a, b))
        or (abs(values[2]) <= 1e-7 and _on_segment(a, c, d))
        or (abs(values[3]) <= 1e-7 and _on_segment(b, c, d))
    )


def line_intersects_polygon(
    start: tuple[float, float],
    end: tuple[float, float],
    polygon: list[list[float]],
) -> bool:
    points = [(float(value[0]), float(value[1])) for value in polygon]
    return any(
        segments_intersect(start, end, points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def angle_delta_degrees(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def physical_occlusion_indices(
    rows: list[dict[str, Any]], fov_degrees: float
) -> list[int]:
    values: list[int] = []
    for row in rows:
        camera = row["camera_transform"]
        target = row["actors"]["target_primary"]["transform"]
        start = (float(camera["x"]), float(camera["y"]))
        end = (float(target["x"]), float(target["y"]))
        bearing = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
        in_fov = angle_delta_degrees(bearing, float(camera["yaw"])) <= fov_degrees / 2.0 - 1.0
        van_polygon = row["truth"]["collision_polygons_xy"]["occluder_van"]
        hidden = not bool(row["instance_visibility"]["target_primary"]["visible"])
        if in_fov and hidden and line_intersects_polygon(start, end, van_polygon):
            values.append(int(row["sample_index"]))
    return values


def copy_selected(
    source_root: Path,
    row: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    source = source_root / row["sensor_path"]
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def render_layout_svg(protocol: dict[str, Any], path: Path) -> None:
    width, height = 1000, 700
    scale = 45.0
    origin_x, origin_y = 430.0, 350.0

    def xy(forward_m: float, right_m: float) -> tuple[float, float]:
        return origin_x + forward_m * scale, origin_y - right_m * scale

    colors = {
        "wearer": "#2E86DE",
        "primary_crossing": "#E74C3C",
        "cyclist_distractor": "#8E44AD",
        "parallel_walker": "#16A085",
        "child_distractor": "#F39C12",
        "physical_occluder": "#34495E",
        "barrier": "#D35400",
        "traffic_cone": "#E67E22",
        "warning_sign": "#F1C40F",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<text x="30" y="42" font-family="sans-serif" font-size="26" font-weight="700">DTR-CARLA-C1 procedural scene layout</text>',
        '<text x="30" y="70" font-family="sans-serif" font-size="15" fill="#475569">Evaluator-only local coordinates; witness and instance views never enter the model root.</text>',
    ]
    route_y = xy(0.0, 0.0)[1]
    lines.append(
        f'<line x1="80" y1="{route_y:.1f}" x2="820" y2="{route_y:.1f}" stroke="#2E86DE" stroke-width="10" opacity="0.18"/>'
    )
    trajectory_name = {
        "wearer": "wearer_straight",
        "target_primary": "primary_cross",
        "cyclist": "cyclist_safe_cross",
        "parallel": "parallel_walker",
        "child": "child_hold",
    }
    for asset in protocol["asset_cluster"]:
        if "fixed_pose" in asset:
            forward_m = float(asset["fixed_pose"]["forward_m"])
            right_m = float(asset["fixed_pose"]["right_m"])
        else:
            forward_m, right_m = trajectory_position(
                protocol["trajectory_library"][trajectory_name[asset["asset_key"]]], 0.0
            )
        px, py = xy(forward_m, right_m)
        role = str(asset["role"])
        radius = 13 if role in {"physical_occluder", "barrier"} else 8
        lines.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{colors[role]}" stroke="#0F172A"/>'
        )
        lines.append(
            f'<text x="{px + radius + 4:.1f}" y="{py + 5:.1f}" font-family="sans-serif" font-size="12">{escape(str(asset["asset_key"]))}</text>'
        )
    lines.extend(
        [
            '<text x="30" y="650" font-family="sans-serif" font-size="14" fill="#334155">15 task-owned actors + four source-disjoint fresh-server camera shards</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_timeline_svg(summaries: list[dict[str, Any]], path: Path) -> None:
    width = 1150
    height = 130 + len(summaries) * 62
    x0, x1 = 320.0, 1080.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="28" y="38" font-family="sans-serif" font-size="25" font-weight="700">C1 route authority and realized contact</text>',
        '<text x="28" y="66" font-family="sans-serif" font-size="14" fill="#64748B">Post-join evaluator view; contact is absent from the model root.</text>',
    ]
    for tick in range(8):
        x = x0 + (x1 - x0) * tick / 7.0
        lines.append(
            f'<line x1="{x:.1f}" y1="90" x2="{x:.1f}" y2="{height - 25}" stroke="#E2E8F0"/>'
        )
        lines.append(
            f'<text x="{x - 5:.1f}" y="88" font-family="sans-serif" font-size="12" fill="#64748B">{tick}s</text>'
        )
    for index, summary in enumerate(summaries):
        y = 120 + index * 62
        observed = bool(summary["observed_contact"])
        color = "#DC2626" if observed else "#16A34A"
        label = (
            f"{summary['episode_id']} {summary['plan_authority_at_start']} "
            f"expected={'CONTACT' if summary['expected_contact'] else 'SAFE'}"
        )
        lines.append(
            f'<text x="28" y="{y + 5}" font-family="sans-serif" font-size="14">{escape(label)}</text>'
        )
        lines.append(
            f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#94A3B8" stroke-width="6" stroke-linecap="round"/>'
        )
        if summary["first_contact_time_s"] is not None:
            contact_x = x0 + (x1 - x0) * float(summary["first_contact_time_s"]) / 7.0
            lines.append(f'<circle cx="{contact_x:.1f}" cy="{y}" r="10" fill="{color}"/>')
        lines.append(
            f'<text x="{x1 + 12:.1f}" y="{y + 5}" font-family="sans-serif" font-size="13" font-weight="700" fill="{color}">{"CONTACT" if observed else "SAFE"}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def seal_tree(root: Path, relative_roots: list[str], output: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for relative_root in relative_roots:
        source = root / relative_root
        for path in sorted(value for value in source.rglob("*") if value.is_file()):
            entries.append(
                {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "schema_version": "dtr-c1-sealed-evidence-manifest-v1",
        "file_count": len(entries),
        "files": entries,
    }
    write_json_atomic(output, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    root = args.root.resolve(strict=True)
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    sensor_order = [str(value) for value in protocol["capture"]["sensor_order"]]
    if sensor_order != ["instance", "wearable", "depth", "witness"]:
        raise ValueError(f"unexpected C1 sensor order: {sensor_order}")
    model_root = root / "model"
    evaluator_root = root / "evaluator"
    if model_root.exists() or evaluator_root.exists() or (root / "result.json").exists():
        raise FileExistsError(f"refusing partial join or overwrite: {root}")
    model_root.mkdir()
    evaluator_root.mkdir()

    shard_results: dict[str, dict[str, Any]] = {}
    payload_integrity_failures: list[dict[str, Any]] = []
    for sensor in sensor_order:
        shard_root = root / "shards" / sensor
        result_path = shard_root / "result.json"
        result = read_json(result_path)
        shard_results[sensor] = result
        if result["status"] != "DTR_CARLA_C1_RAW_SHARD_CAPTURE_COMPLETE":
            payload_integrity_failures.append(
                {"sensor": sensor, "reason": "shard_not_complete", "status": result["status"]}
            )
        inventory_path = shard_root / "payload_inventory.json"
        if sha256_file(inventory_path) != result["payload_inventory_sha256"]:
            payload_integrity_failures.append(
                {"sensor": sensor, "reason": "payload_inventory_hash"}
            )
        for item in read_json(inventory_path):
            path = Path(item["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(item["bytes"])
                or sha256_file(path) != item["sha256"]
            ):
                payload_integrity_failures.append(
                    {
                        "sensor": sensor,
                        "episode_id": item["episode_id"],
                        "sample_index": item["sample_index"],
                        "reason": "payload_integrity",
                    }
                )
                if len(payload_integrity_failures) >= 50:
                    break

    keyframe_indices = {
        int(round(float(value) / float(protocol["environment"]["sample_seconds"])))
        for value in protocol["capture"]["keyframe_times_s"]
    }
    receipts = {
        str(value["episode_id"]): build_plan_receipt(value.get("issued_plan"))
        for value in protocol["scenarios"]
    }
    plans_root = model_root / "plans"
    plans_root.mkdir()
    for episode_id, receipt in receipts.items():
        if receipt is not None:
            write_json_atomic(plans_root / f"{episode_id}.json", receipt)

    replay_failures: list[dict[str, Any]] = []
    truth_failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    visual_payloads: list[dict[str, Any]] = []
    occlusion_by_episode: dict[str, list[int]] = {}
    model_episode_manifests: list[dict[str, Any]] = []
    evaluator_episode_manifests: list[dict[str, Any]] = []
    for scenario in protocol["scenarios"]:
        episode_id = str(scenario["episode_id"])
        rows_by_sensor = {
            sensor: read_jsonl(
                root / "shards" / sensor / "episodes" / episode_id / "frames.jsonl"
            )
            for sensor in sensor_order
        }
        reference = rows_by_sensor["instance"]
        for sensor in sensor_order[1:]:
            replay_failures.extend(
                compare_replays(reference, rows_by_sensor[sensor], candidate_sensor=sensor)
            )
        occlusion_by_episode[episode_id] = physical_occlusion_indices(
            reference, float(protocol["capture"]["fov_degrees"])
        )
        model_episode = model_root / "episodes" / episode_id
        evaluator_episode = evaluator_root / "episodes" / episode_id
        model_episode.mkdir(parents=True)
        evaluator_episode.mkdir(parents=True)
        model_records: list[dict[str, Any]] = []
        evaluator_records: list[dict[str, Any]] = []
        for index, reference_row in enumerate(reference):
            wearable_row = rows_by_sensor["wearable"][index]
            depth_row = rows_by_sensor["depth"][index]
            witness_row = rows_by_sensor["witness"][index]
            sample_index = int(reference_row["sample_index"])
            wearable_path = None
            depth_path = None
            witness_path = None
            instance_path = None
            if sample_index in keyframe_indices:
                destinations = {
                    "wearable": model_episode / "wearable" / f"{sample_index:06d}.png",
                    "depth": model_episode / "depth" / f"{sample_index:06d}.png",
                    "witness": evaluator_episode / "witness" / f"{sample_index:06d}.png",
                    "instance": evaluator_episode / "instance" / f"{sample_index:06d}.png",
                }
                for sensor, row in (
                    ("wearable", wearable_row),
                    ("depth", depth_row),
                    ("witness", witness_row),
                    ("instance", reference_row),
                ):
                    value = copy_selected(root / "shards" / sensor, row, destinations[sensor])
                    value.update(
                        {
                            "sensor": sensor,
                            "episode_id": episode_id,
                            "sample_index": sample_index,
                        }
                    )
                    visual_payloads.append(value)
                wearable_path = str(destinations["wearable"].relative_to(model_root)).replace("\\", "/")
                depth_path = str(destinations["depth"].relative_to(model_root)).replace("\\", "/")
                witness_path = str(destinations["witness"].relative_to(evaluator_root)).replace("\\", "/")
                instance_path = str(destinations["instance"].relative_to(evaluator_root)).replace("\\", "/")

            wearer = wearable_row["actors"]["wearer"]
            current_actors = []
            for key, actor in sorted(wearable_row["actors"].items()):
                if key == "wearer":
                    continue
                current_actors.append(
                    {
                        "track_id": actor["track_id"],
                        "kind": actor["kind"],
                        "blueprint_id": actor["actual_blueprint"],
                        "transform": actor["transform"],
                        "command_velocity": actor["command_velocity"],
                        "bounding_box_extent": actor["bounding_box"]["extent"],
                    }
                )
            receipt = receipts[episode_id]
            model_record = {
                "schema_version": "dtr-c1-model-observation-v1",
                "episode_id": episode_id,
                "sample_index": sample_index,
                "time_s": float(wearable_row["time_s"]),
                "wearable_rgb_path": wearable_path,
                "wearable_depth_path": depth_path,
                "wearer": {
                    "track_id": wearer["track_id"],
                    "transform": wearer["transform"],
                    "command_velocity": wearer["command_velocity"],
                    "bounding_box_extent": wearer["bounding_box"]["extent"],
                },
                "current_actors": current_actors,
                "issued_plan": {
                    "path": f"plans/{episode_id}.json" if receipt else None,
                    "receipt_sha256": receipt["receipt_sha256"] if receipt else None,
                    "authority": wearable_row["plan_authority"],
                },
            }
            failures = forbidden_model_paths(model_record)
            if failures:
                truth_failures.append(
                    {
                        "episode_id": episode_id,
                        "sample_index": sample_index,
                        "paths": failures,
                    }
                )
            model_records.append(model_record)
            evaluator_records.append(
                {
                    "schema_version": "dtr-c1-evaluator-frame-v1",
                    "episode_id": episode_id,
                    "sample_index": sample_index,
                    "time_s": float(reference_row["time_s"]),
                    "witness_path": witness_path,
                    "instance_path": instance_path,
                    "actors": reference_row["actors"],
                    "camera_transform": reference_row["camera_transform"],
                    "instance_visibility": reference_row["instance_visibility"],
                    "actual_future_trajectory": {
                        "wearer_trajectory_name": scenario["wearer_trajectory"],
                        "current_local_position": reference_row["actors"]["wearer"][
                            "local_position"
                        ],
                    },
                    "truth": reference_row["truth"],
                }
            )

        model_frames = model_episode / "observations.jsonl"
        evaluator_frames = evaluator_episode / "truth.jsonl"
        write_jsonl(model_frames, model_records)
        write_jsonl(evaluator_frames, evaluator_records)
        shard_summary = read_json(
            root / "shards" / "instance" / "episodes" / episode_id / "summary.json"
        )
        summary = {
            **shard_summary,
            "physical_occlusion_frames": len(occlusion_by_episode[episode_id]),
            "physical_occlusion_sample_indices": occlusion_by_episode[episode_id],
        }
        write_json_atomic(evaluator_episode / "summary.json", summary)
        summaries.append(summary)
        model_manifest_path = model_episode / "manifest.json"
        evaluator_manifest_path = evaluator_episode / "manifest.json"
        write_json_atomic(
            model_manifest_path,
            {
                "episode_id": episode_id,
                "observation_count": len(model_records),
                "observations_sha256": sha256_file(model_frames),
                "plan_receipt_sha256": receipt["receipt_sha256"] if receipt else None,
            },
        )
        write_json_atomic(
            evaluator_manifest_path,
            {
                "episode_id": episode_id,
                "truth_count": len(evaluator_records),
                "truth_sha256": sha256_file(evaluator_frames),
                "summary_sha256": sha256_file(evaluator_episode / "summary.json"),
            },
        )
        model_episode_manifests.append(
            {
                "episode_id": episode_id,
                "path": str(model_manifest_path.relative_to(model_root)).replace("\\", "/"),
                "sha256": sha256_file(model_manifest_path),
            }
        )
        evaluator_episode_manifests.append(
            {
                "episode_id": episode_id,
                "path": str(evaluator_manifest_path.relative_to(evaluator_root)).replace("\\", "/"),
                "sha256": sha256_file(evaluator_manifest_path),
            }
        )

    truth_report = {
        "schema_version": "dtr-c1-truth-boundary-report-v1",
        "model_contains_truth": bool(truth_failures),
        "forbidden_paths": truth_failures,
    }
    twin_report = {
        "schema_version": "dtr-c1-twin-admission-report-v1",
        "cross_sensor_replay_failures": replay_failures,
        "protocol_twin_contracts": protocol["twin_contracts"],
    }
    write_json_atomic(evaluator_root / "truth_boundary_report.json", truth_report)
    write_json_atomic(evaluator_root / "twin_admission_report.json", twin_report)
    write_json_atomic(evaluator_root / "payload_inventory.json", visual_payloads)
    write_json_atomic(
        model_root / "manifest.json",
        {
            "schema_version": "dtr-c1-model-root-manifest-v1",
            "experiment_id": EXPERIMENT_ID,
            "contains_evaluator_truth": False,
            "episodes": model_episode_manifests,
        },
    )
    write_json_atomic(
        evaluator_root / "manifest.json",
        {
            "schema_version": "dtr-c1-evaluator-root-manifest-v1",
            "experiment_id": EXPERIMENT_ID,
            "episodes": evaluator_episode_manifests,
        },
    )
    render_layout_svg(protocol, evaluator_root / "scene_layout.svg")
    render_timeline_svg(summaries, evaluator_root / "timeline.svg")
    sheet_index = int(
        round(
            float(protocol["capture"]["contact_sheet_time_s"])
            / float(protocol["environment"]["sample_seconds"])
        )
    )
    witness_frames = [
        evaluator_root
        / "episodes"
        / str(value["episode_id"])
        / "witness"
        / f"{sheet_index:06d}.png"
        for value in protocol["scenarios"]
    ]
    magick = shutil.which("magick")
    if magick is None:
        raise RuntimeError("ImageMagick is required to seal the C1 contact sheet")
    contact_sheet = evaluator_root / "contact_sheet.png"
    subprocess.run(
        [
            magick,
            "montage",
            *[str(value) for value in witness_frames],
            "-tile",
            "4x2",
            "-geometry",
            "320x180+8+8",
            "-background",
            "white",
            str(contact_sheet),
        ],
        check=True,
    )
    visual_manifest = {
        "schema_version": "dtr-c1-visual-manifest-v1",
        "contact_sheet_time_s": float(protocol["capture"]["contact_sheet_time_s"]),
        "artifacts": [
            {
                "path": str(path.relative_to(evaluator_root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                contact_sheet,
                evaluator_root / "scene_layout.svg",
                evaluator_root / "timeline.svg",
            )
        ],
    }
    write_json_atomic(evaluator_root / "visual_manifest.json", visual_manifest)
    model_seal = seal_tree(root, ["model"], model_root / "sealed_manifest.json")
    sealed_manifest_path = root / "sealed_evidence_manifest.json"
    sealed_manifest = seal_tree(
        root, ["shards", "model", "evaluator"], sealed_manifest_path
    )

    checks = {
        "all_four_fresh_server_shards_complete": all(
            value["status"] == "DTR_CARLA_C1_RAW_SHARD_CAPTURE_COMPLETE"
            for value in shard_results.values()
        ),
        "raw_payload_inventories_verified": not payload_integrity_failures,
        "cross_sensor_actual_replay_identical": not replay_failures,
        "all_expected_contact_relations_match": all(
            value["expected_contact"] == value["observed_contact"] for value in summaries
        ),
        "physical_van_occlusion_proven_in_every_episode": all(
            bool(value) for value in occlusion_by_episode.values()
        ),
        "model_root_zero_truth_keys": not truth_failures,
        "all_keyframe_views_materialized": len(visual_payloads)
        == len(protocol["scenarios"]) * len(keyframe_indices) * 4,
        "depth_shard_present": shard_results["depth"]["payload_count"] > 0,
        "sealed_model_root_nonempty": model_seal["file_count"] > 0,
        "sealed_evidence_root_nonempty": sealed_manifest["file_count"] > 0,
        "visual_evidence_materialized": all(
            (evaluator_root / value["path"]).is_file()
            for value in visual_manifest["artifacts"]
        ),
    }
    result = {
        "schema_version": "dtr-carla-c1-complex-scene-result-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "DTR_CARLA_C1_COMPLEX_SCENE_ASSET_CANARY_COMPLETE"
            if all(checks.values())
            else "DTR_CARLA_C1_COMPLEX_SCENE_ASSET_CANARY_NOT_EVALUABLE"
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "join_script_sha256": sha256_file(Path(__file__).resolve()),
        "checks": checks,
        "payload_integrity_failures": payload_integrity_failures,
        "replay_failures": replay_failures,
        "episodes": summaries,
        "shards": {
            sensor: {
                "payload_count": value["payload_count"],
                "asset_count": value["asset_count"],
                "result_sha256": sha256_file(root / "shards" / sensor / "result.json"),
            }
            for sensor, value in shard_results.items()
        },
        "sealed_evidence_manifest_path": str(sealed_manifest_path),
        "sealed_evidence_manifest_sha256": sha256_file(sealed_manifest_path),
        "sealed_file_count": sealed_manifest["file_count"],
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(root / "result.json", result)
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
