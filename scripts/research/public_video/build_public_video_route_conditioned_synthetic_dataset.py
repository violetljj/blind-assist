#!/usr/bin/env python3
"""Build exact-mask train-only route-conditioned obstruction counterfactuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2

from build_public_silver_synthetic_static_counterfactuals import compose_obstacle


SCHEMA = "blindassist_route_conditioned_synthetic_dataset_v1"
PROMPT = "Deterministically place the licensed obstacle cutout at the frozen lateral position over an exact-copy public RGB parent frame; preserve exact alpha mask and bbox."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_independent_direction(path: Path) -> None:
    if "secondary-corridor-causal" in str(path.resolve()).lower().replace("_", "-"):
        raise ValueError("independent-direction paths are forbidden")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def verify_sha(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")


def resolve_contract(contract_path: Path) -> tuple[dict[str, Any], str]:
    raw = load_json(contract_path)
    schema = raw.get("schema")
    if schema == "blindassist_route_conditioned_synthetic_dataset_contract_erratum_v1":
        binding = raw["bound_original_contract"]
        original_path = (Path.cwd() / binding["path"]).resolve()
        reject_independent_direction(original_path)
        verify_sha(original_path, binding["sha256"])
        contract, _old_class = resolve_contract(original_path)
        contract["contract_id"] = raw["contract_id"]
        contract["frozen_at_utc"] = raw["frozen_at_utc"]
        annotation_class = str(raw["correction"]["annotation_class"])
        if not annotation_class:
            raise ValueError("erratum annotation class must not be empty")
        return contract, annotation_class
    if schema == "blindassist_route_conditioned_synthetic_factorial_followup_v1":
        binding = raw["bound_base_contract"]
        base_path = (Path.cwd() / binding["path"]).resolve()
        reject_independent_direction(base_path)
        verify_sha(base_path, binding["sha256"])
        contract, annotation_class = resolve_contract(base_path)
        contract["contract_id"] = raw["contract_id"]
        contract["frozen_at_utc"] = raw["frozen_at_utc"]
        assets = [str(value) for value in raw["factorial_asset_names"]]
        contract["parents"] = [{"slug": row["slug"], "asset_names": assets} for row in contract["parents"]]
        return contract, annotation_class
    if schema == "blindassist_route_conditioned_synthetic_factorial_reexecution_v1":
        binding = raw["bound_invalid_timestamp_contract"]
        base_path = (Path.cwd() / binding["path"]).resolve()
        reject_independent_direction(base_path)
        verify_sha(base_path, binding["sha256"])
        contract, annotation_class = resolve_contract(base_path)
        contract["contract_id"] = raw["contract_id"]
        contract["frozen_at_utc"] = raw["frozen_at_utc"]
        return contract, annotation_class
    if schema == "blindassist_route_conditioned_synthetic_asset_extension_v1":
        binding = raw["bound_base_contract"]
        base_path = (Path.cwd() / binding["path"]).resolve()
        reject_independent_direction(base_path)
        verify_sha(base_path, binding["sha256"])
        contract, annotation_class = resolve_contract(base_path)
        contract["contract_id"] = raw["contract_id"]
        contract["frozen_at_utc"] = raw["frozen_at_utc"]
        assets = [str(value) for value in raw["factorial_asset_names"]]
        contract["parents"] = [{"slug": row["slug"], "asset_names": assets} for row in contract["parents"]]
        for name, asset_binding in raw.get("asset_sources", {}).items():
            review_path = (Path.cwd() / asset_binding["manual_review_path"]).resolve()
            reject_independent_direction(review_path)
            verify_sha(review_path, asset_binding["manual_review_sha256"])
            review = load_json(review_path)
            if review.get("disposition") != asset_binding["manual_review_disposition"]:
                raise ValueError(f"asset review disposition mismatch: {name}")
        contract.setdefault("asset_sources", {}).update(raw.get("asset_sources", {}))
        contract["composition"]["width_fractions_by_asset"].update(raw["width_fractions_by_asset"])
        return contract, annotation_class
    return raw, str(raw.get("annotation_class") or "static_obstacle")


def route_waypoints(template: dict[str, Any], choice: str) -> list[list[float]]:
    fixed = template["fixed_templates"]
    xs = fixed[f"{choice}_x_norm"]
    ys = fixed["y_norm"]
    if len(xs) != len(ys) or len(xs) != 3:
        raise ValueError("route template must contain three aligned waypoints")
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def point_bbox_overlap_fraction(waypoints: Sequence[Sequence[float]], bbox: Sequence[int], width: int, height: int) -> float:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    inside = 0
    for x_norm, y_norm in waypoints:
        x, y = float(x_norm) * width, float(y_norm) * height
        inside += int(x1 <= x < x2 and y1 <= y < y2)
    return inside / len(waypoints)


def find_parent_rows(records: Sequence[dict[str, Any]], slug: str) -> tuple[list[dict[str, Any]], str]:
    clear = sorted(
        [row for row in records if row["id"].startswith(f"{slug}_clear_")],
        key=lambda row: row["id"],
    )
    alert = [row for row in records if row["id"].startswith(f"{slug}_alert_")]
    if len(clear) != 3 or len(alert) != 3:
        raise ValueError(f"parent slug must have three clear and alert frames: {slug}")
    sources = {str(row["attributes"]["parent_source_id"]) for row in clear + alert}
    if len(sources) != 1:
        raise ValueError(f"parent slug spans multiple sources: {slug}")
    return clear, next(iter(sources))


def lifecycle_open(states: Sequence[bool], consecutive: int) -> bool:
    run = 0
    for state in states:
        run = run + 1 if state else 0
        if run >= consecutive:
            return True
    return False


def build(contract_path: Path, output_root: Path) -> dict[str, Any]:
    contract_path, output_root = contract_path.resolve(), output_root.resolve()
    for path in (contract_path, output_root):
        reject_independent_direction(path)
    if output_root.exists():
        raise ValueError(f"refusing to overwrite output root: {output_root}")
    contract, annotation_class = resolve_contract(contract_path)
    parent_root = (Path.cwd() / contract["bound_parent_dataset"]["root"]).resolve()
    route_path = (Path.cwd() / contract["bound_route_template"]["path"]).resolve()
    for path in (parent_root, route_path):
        reject_independent_direction(path)
    parent_binding = contract["bound_parent_dataset"]
    verify_sha(parent_root / "build_receipt.json", parent_binding["build_receipt_sha256"])
    verify_sha(parent_root / "generation_records.jsonl", parent_binding["generation_records_sha256"])
    verify_sha(parent_root / "qa" / "manual_review.json", parent_binding["manual_review_sha256"])
    verify_sha(route_path, contract["bound_route_template"]["sha256"])
    route_template = load_json(route_path)
    parent_records = load_jsonl(parent_root / "generation_records.jsonl")

    images_root = output_root / "images" / "train"
    masks_root = output_root / "masks" / "train"
    assets_root = output_root / "assets"
    for directory in (images_root, masks_root, assets_root):
        directory.mkdir(parents=True, exist_ok=True)
    asset_paths = {
        "barricade": parent_root / "assets" / "barricade_rgba.png",
        "sand_pile": parent_root / "assets" / "sand_pile_rgba.png",
    }
    for name, binding in contract.get("asset_sources", {}).items():
        source = (Path.cwd() / binding["path"]).resolve()
        reject_independent_direction(source)
        verify_sha(source, binding["sha256"])
        asset_paths[str(name)] = source
    copied_assets = {}
    for name, source in asset_paths.items():
        target = assets_root / source.name
        shutil.copy2(source, target)
        copied_assets[name] = target

    generation: list[dict[str, Any]] = []
    route_examples: list[dict[str, Any]] = []
    compositions: list[dict[str, Any]] = []
    sequence_states: dict[tuple[str, str, str, str], list[bool]] = {}
    source_ids: set[str] = set()
    choices = list(contract["route_relation"]["route_choices"])
    centers = contract["composition"]["obstacle_center_x_norm"]
    threshold = float(contract["route_relation"]["frame_intersection_fraction_threshold"])

    for parent in contract["parents"]:
        slug = str(parent["slug"])
        asset_names = [str(value) for value in parent.get("asset_names", [parent.get("asset_name")])]
        if not asset_names or any(name not in copied_assets for name in asset_names):
            raise ValueError(f"unsupported asset list for parent: {slug}")
        clear_rows, parent_source_id = find_parent_rows(parent_records, slug)
        source_ids.add(parent_source_id)
        for asset_name in asset_names:
            variant_slug = slug if len(asset_names) == 1 else f"{slug}_{asset_name}"
            asset = cv2.imread(str(copied_assets[asset_name]), cv2.IMREAD_UNCHANGED)
            if asset is None:
                raise FileNotFoundError(copied_assets[asset_name])
            widths = contract["composition"]["width_fractions_by_asset"][asset_name]
            for frame_index, (clear_row, width_fraction) in enumerate(zip(clear_rows, widths)):
                source_path = Path(clear_row["source"]).resolve()
                background = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                if background is None:
                    raise FileNotFoundError(source_path)
                height, width = background.shape[:2]
                clear_name = f"{variant_slug}_clear_{frame_index:02d}.png"
                clear_path = images_root / clear_name
                shutil.copy2(source_path, clear_path)
                clear_id = clear_path.stem
                generation.append({
                    "id": clear_id, "image_path": clear_path.relative_to(output_root).as_posix(), "split": "train",
                    "width": width, "height": height, "labels": [], "prompt": PROMPT, "objects": [],
                    "attributes": {"variant": "clear_exact_copy", "parent_source_id": parent_source_id,
                                   "parent_slug": variant_slug, "asset_name": asset_name,
                                   "distance_index": frame_index, "train_only": True},
                    "status": "accepted", "source": str(source_path),
                })
                for choice in choices:
                    route_examples.append({
                        "example_id": f"{clear_id}__route_{choice.lower()}", "image_id": clear_id,
                        "image_path": clear_path.relative_to(output_root).as_posix(), "parent_source_id": parent_source_id,
                        "asset_name": asset_name, "obstacle_direction": "NONE", "route_choice": choice,
                        "route_waypoints_xy_norm": route_waypoints(route_template, choice),
                        "intersection_fraction": 0.0, "route_blocked": False, "train_only": True,
                    })
                for obstacle_direction in choices:
                    composite, mask, bbox = compose_obstacle(
                        background, asset, width_fraction=float(width_fraction), center_x=float(centers[obstacle_direction]),
                        bottom_y=float(contract["composition"]["bottom_y_norm"]),
                    )
                    image_id = f"{variant_slug}_{obstacle_direction.lower()}_{frame_index:02d}"
                    image_path, mask_path = images_root / f"{image_id}.png", masks_root / f"{image_id}.png"
                    if not cv2.imwrite(str(image_path), composite) or not cv2.imwrite(str(mask_path), mask):
                        raise ValueError(f"cannot write composite: {image_id}")
                    generation.append({
                        "id": image_id, "image_path": image_path.relative_to(output_root).as_posix(), "split": "train",
                        "width": width, "height": height, "labels": [annotation_class], "prompt": PROMPT,
                        "objects": [{"class": annotation_class, "bbox_xyxy": bbox,
                                     "bbox_source": "deterministic_alpha_composition",
                                     "mask_path": mask_path.relative_to(output_root).as_posix()}],
                        "attributes": {"variant": "static_obstacle_composite", "parent_source_id": parent_source_id,
                                       "parent_slug": variant_slug, "asset_name": asset_name,
                                       "obstacle_direction": obstacle_direction, "distance_index": frame_index,
                                       "width_fraction": float(width_fraction), "train_only": True},
                        "status": "accepted", "source": str(source_path),
                    })
                    compositions.append({
                        "image_id": image_id, "parent_image_sha256": sha256_file(source_path),
                        "image_sha256": sha256_file(image_path), "mask_sha256": sha256_file(mask_path),
                        "asset_sha256": sha256_file(copied_assets[asset_name]), "bbox_xyxy": bbox,
                        "parent_source_id": parent_source_id, "asset_name": asset_name,
                        "obstacle_direction": obstacle_direction, "distance_index": frame_index,
                    })
                    for choice in choices:
                        waypoints = route_waypoints(route_template, choice)
                        overlap = point_bbox_overlap_fraction(waypoints, bbox, width, height)
                        blocked = overlap >= threshold
                        route_examples.append({
                            "example_id": f"{image_id}__route_{choice.lower()}", "image_id": image_id,
                            "image_path": image_path.relative_to(output_root).as_posix(),
                            "mask_path": mask_path.relative_to(output_root).as_posix(), "bbox_xyxy": bbox,
                            "parent_source_id": parent_source_id, "asset_name": asset_name,
                            "obstacle_direction": obstacle_direction, "route_choice": choice,
                            "route_waypoints_xy_norm": waypoints, "intersection_fraction": overlap,
                            "route_blocked": blocked, "train_only": True,
                        })
                        sequence_states.setdefault((parent_source_id, asset_name, obstacle_direction, choice), []).append(blocked)

    open_consecutive = int(contract["route_relation"]["open_consecutive_frames"])
    lifecycles = [{
        "parent_source_id": key[0], "asset_name": key[1], "obstacle_direction": key[2], "route_choice": key[3],
        "frame_blocked": states, "intervention_open": lifecycle_open(states, open_consecutive),
    } for key, states in sorted(sequence_states.items())]
    left_sources = {row["parent_source_id"] for row in lifecycles if row["route_choice"] == "LEFT" and row["intervention_open"]}
    right_sources = {row["parent_source_id"] for row in lifecycles if row["route_choice"] == "RIGHT" and row["intervention_open"]}
    blocked_count = sum(bool(row["route_blocked"]) for row in route_examples)
    clear_count = len(route_examples) - blocked_count
    gate_spec = contract["quality_gate"]
    geometry_gate = {
        "minimum_parent_source_count": len(source_ids) >= int(gate_spec["minimum_parent_source_count"]),
        "minimum_left_open_source_count": len(left_sources) >= int(gate_spec["minimum_open_parent_sources_per_left_or_right_choice"]),
        "minimum_right_open_source_count": len(right_sources) >= int(gate_spec["minimum_open_parent_sources_per_left_or_right_choice"]),
        "both_blocked_and_clear_examples": blocked_count > 0 and clear_count > 0,
    }

    dataset_spec = {
        "name": "blindassist_route_conditioned_synthetic_r812", "task": "segmentation+detection+route_conditioned_classification",
        "classes": [{"id": 0, "name": annotation_class}],
        "scenes": ["licensed_public_first_person_navigation_rgb"],
        "attributes": {"obstacle_direction": choices, "route_choice": choices,
                       "distance_index": [0, 1, 2], "geometry_source": ["deterministic_alpha_composition"]},
        "negative_cases": ["exact_parent_clear", "obstacle_not_intersecting_selected_route"],
        "counts": {"parent_source_count": len(source_ids), "image_count": len(generation),
                   "route_example_count": len(route_examples), "blocked_route_examples": blocked_count,
                   "clear_route_examples": clear_count},
        "splits": {"train": 1.0, "val": 0.0, "test": 0.0},
        "image_style": "source-native public RGB plus deterministic reviewed obstacle alpha composition",
        "output_resolution": "source-native", "annotation_target": "exact PNG masks + bbox + YOLO + COCO + route_examples.jsonl",
        "intended_use": "train-only route-conditioned risk-profile and interaction diagnosis",
        "exclusions": ["not real event truth", "not provider evaluation", "not calibration", "not blind", "not production evidence"],
    }
    write_json(output_root / "dataset_spec.json", dataset_spec)
    write_jsonl(output_root / "generation_records.jsonl", generation)
    write_jsonl(output_root / "route_examples.jsonl", route_examples)
    write_json(output_root / "composition_records.json", {"schema": SCHEMA, "records": compositions})
    write_json(output_root / "lifecycle_records.json", {"schema": SCHEMA, "records": lifecycles})
    receipt = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract_path), "contract_sha256": sha256_file(contract_path),
        "annotation_class": annotation_class,
        "output_root": str(output_root), "parent_source_ids": sorted(source_ids),
        "image_count": len(generation), "route_example_count": len(route_examples),
        "blocked_route_example_count": blocked_count, "clear_route_example_count": clear_count,
        "left_open_parent_source_count": len(left_sources), "right_open_parent_source_count": len(right_sources),
        "geometry_gate": geometry_gate, "geometry_gate_passed": all(geometry_gate.values()),
        "visual_review_pending": True, "train_only": True, "real_event_truth": False,
        "provider_evaluation_credit": False, "calibration_authorized": False,
        "blind_authorized": False, "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    receipt_path = output_root / "build_receipt.json"
    write_json(receipt_path, receipt)
    Path(str(receipt_path) + ".sha256").write_text(sha256_file(receipt_path) + "\n", encoding="ascii")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build(args.contract, args.output)
    print(json.dumps({key: receipt[key] for key in ("image_count", "route_example_count", "geometry_gate_passed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
