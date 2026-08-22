"""Audit ABotN-POIBench as an independent public truth substrate.

This entrypoint deliberately does not run a baseline.  It materializes only
the small public task annotations, verifies their pinned provenance, and
separates metric named-POI arrival truth from unavailable frame-region truth.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

import requests


SCHEMA = "blindassist_abotn_poibench_truth_source_audit_v0"
DATASET_ID = "acvlab/ABotN-POIBench"
DATASET_REVISION = "fbb62cc3382d8ff84f7fe3b6a3e7d48e4c21e974"
REPOSITORY_ID = "amap-cvlab/ABot-Navigation"
REPOSITORY_REVISION = "2a0aefb56f1e2d315bba924239e9e8ad9dca9d92"
REGION_KEYS = {
    "bbox",
    "bounding_box",
    "doorway_frame",
    "entrance_frame",
    "entrance_line",
    "entrance_points",
    "goal_bbox",
    "mask",
    "polygon",
    "segmentation",
}
PRIVATE_AGENT_FIELDS = ("target_position", "distance_to_goal")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _all_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


def _xy(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be a two-element list")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must contain finite numbers")
    return result


def inspect_task(payload: Mapping[str, Any], *, endpoint_tolerance_m: float = 0.25) -> dict[str, Any]:
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("trajectory must be a non-empty list")
    first = trajectory[0]
    last = trajectory[-1]
    if not isinstance(first, Mapping) or not isinstance(last, Mapping):
        raise ValueError("trajectory endpoints must be objects")
    trajectory_start = (float(first["x"]), float(first["y"]))
    trajectory_end = (float(last["x"]), float(last["y"]))
    if not all(math.isfinite(item) for item in trajectory_start + trajectory_end):
        raise ValueError("trajectory endpoints must be finite")

    label = payload.get("label")
    if not isinstance(label, Mapping):
        raise ValueError("label must be an object")
    extend = label.get("extend")
    if not isinstance(extend, Mapping):
        raise ValueError("label.extend must be an object")
    goal_label = extend.get("goal_label")
    if not isinstance(goal_label, str) or not goal_label.strip():
        raise ValueError("label.extend.goal_label must be non-empty")
    start = _xy(extend.get("start_point"), "start_point")
    end = _xy(extend.get("end_point"), "end_point")

    start_delta = math.dist(start, trajectory_start)
    end_delta = math.dist(end, trajectory_end)
    keys = _all_keys(payload)
    explicit_region_keys = sorted(keys.intersection(REGION_KEYS))
    return {
        "goal_label": goal_label.strip(),
        "trajectory_points": len(trajectory),
        "reported_start_xy": list(start),
        "reported_end_xy": list(end),
        "trajectory_start_xy": list(trajectory_start),
        "trajectory_end_xy": list(trajectory_end),
        "start_endpoint_delta_m": start_delta,
        "end_endpoint_delta_m": end_delta,
        "endpoint_consistent": start_delta <= endpoint_tolerance_m and end_delta <= endpoint_tolerance_m,
        "explicit_region_keys": explicit_region_keys,
    }


def summarize_tasks(rows: Iterable[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    materialized = []
    for path, payload in rows:
        result = inspect_task(payload)
        result["path"] = path
        result["scene_id"] = path.split("/")[1]
        materialized.append(result)
    if not materialized:
        raise ValueError("no task annotations")

    scenes: dict[str, int] = {}
    for row in materialized:
        scenes[row["scene_id"]] = scenes.get(row["scene_id"], 0) + 1
    return {
        "task_count": len(materialized),
        "scene_count": len(scenes),
        "tasks_per_scene": dict(sorted(scenes.items())),
        "named_goal_count": sum(bool(row["goal_label"]) for row in materialized),
        "metric_endpoint_count": sum(bool(row["reported_end_xy"]) for row in materialized),
        "trajectory_count": sum(row["trajectory_points"] > 0 for row in materialized),
        "endpoint_consistent_within_0_25m_count": sum(row["endpoint_consistent"] for row in materialized),
        "explicit_frame_or_pixel_region_count": sum(bool(row["explicit_region_keys"]) for row in materialized),
        "explicit_region_keys_seen": sorted({key for row in materialized for key in row["explicit_region_keys"]}),
        "goal_labels_unique_count": len({row["goal_label"] for row in materialized}),
    }


def classify_source(
    summary: Mapping[str, Any],
    *,
    dataset_license: str | None,
    dataset_root_files: Iterable[str],
    repository_readme: str,
    repository_license_present: bool,
    evaluator_source: str,
) -> dict[str, Any]:
    task_count = int(summary["task_count"])
    arrival_complete = all(
        int(summary[field]) == task_count
        for field in ("named_goal_count", "metric_endpoint_count", "trajectory_count")
    )
    region_complete = int(summary["explicit_frame_or_pixel_region_count"]) == task_count
    exposed = [
        field
        for field in PRIVATE_AGENT_FIELDS
        if f"{field}=base_obs.{field}" in evaluator_source
    ]
    root_names = {Path(path).name.lower() for path in dataset_root_files}
    dataset_license_attached = bool(dataset_license) or any(name.startswith("license") for name in root_names)
    repository_declares_apache = "Apache-2.0" in repository_readme

    if dataset_license_attached:
        license_status = "DATASET_LICENSE_ATTACHED"
    elif repository_declares_apache and not repository_license_present:
        license_status = "REPOSITORY_README_DECLARES_APACHE_2_0_BUT_DATASET_CARD_AND_PINNED_REPOSITORY_LACK_LICENSE_FILE"
    else:
        license_status = "LICENSE_NOT_ESTABLISHED"

    return {
        "independence": "INDEPENDENT_OF_SEALED_8X89_OUTCOMES_AND_TEACHERS",
        "named_metric_arrival_truth": (
            "SUFFICIENT_FOR_INTERNAL_REVERSIBLE_ARRIVAL_CANARY"
            if arrival_complete
            else "NOT_EVALUABLE_INCOMPLETE_ANNOTATIONS"
        ),
        "functional_frame_region_truth": (
            "AVAILABLE" if region_complete else "NOT_EVALUABLE_NO_EXPLICIT_ENTRANCE_FRAME_OR_PIXEL_REGION"
        ),
        "official_agent_envelope": (
            "PRIVATE_GOAL_GEOMETRY_EXPOSED_MUST_FILTER"
            if exposed
            else "NO_EXPECTED_PRIVATE_FIELDS_DETECTED"
        ),
        "private_fields_exposed_to_official_agent": exposed,
        "license_status": license_status,
        "redistribution_or_commercial_permission": "NOT_ESTABLISHED",
        "overall": (
            "ARRIVAL_TRUTH_ONLY_INTERNAL_RESEARCH_CANDIDATE"
            if arrival_complete and not region_complete
            else "NOT_EVALUABLE_AS_PROPOSED"
        ),
    }


class SourceClient:
    def __init__(self, timeout_s: float = 60.0) -> None:
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "BlindAssist-ABotN-source-audit/0"

    def get_json(self, url: str) -> Any:
        response = self.session.get(url, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def get_bytes(self, url: str, *, allow_missing: bool = False) -> bytes | None:
        response = self.session.get(url, timeout=self.timeout_s)
        if allow_missing and response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content


def run_audit(cache_dir: Path, output: Path, client: SourceClient | None = None) -> dict[str, Any]:
    client = client or SourceClient()
    api_url = f"https://huggingface.co/api/datasets/{DATASET_ID}?blobs=true"
    metadata = client.get_json(api_url)
    if metadata.get("id") != DATASET_ID:
        raise ValueError("unexpected dataset identity")
    if metadata.get("private") or metadata.get("gated") or metadata.get("disabled"):
        raise ValueError("dataset must remain public, ungated, and enabled")
    if metadata.get("sha") != DATASET_REVISION:
        raise ValueError(f"dataset revision drift: {metadata.get('sha')!r}")

    siblings = metadata.get("siblings") or []
    task_files = sorted(
        (
            row
            for row in siblings
            if str(row.get("rfilename", "")).startswith("annotations/")
            and str(row.get("rfilename", "")).endswith(".json")
        ),
        key=lambda row: row["rfilename"],
    )
    rows: list[tuple[str, Mapping[str, Any]]] = []
    file_receipts = []
    for entry in task_files:
        relative = str(entry["rfilename"])
        url = (
            f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
            f"{DATASET_REVISION}/{quote(relative, safe='/')}?download=true"
        )
        payload = client.get_bytes(url)
        if payload is None:
            raise ValueError(f"missing annotation: {relative}")
        target = cache_dir / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != payload:
            target.write_bytes(payload)
        parsed = json.loads(payload.decode("utf-8"))
        rows.append((relative, parsed))
        file_receipts.append({"path": relative, "bytes": len(payload), "sha256": _sha256_bytes(payload)})

    summary = summarize_tasks(rows)
    repo_raw = f"https://raw.githubusercontent.com/{REPOSITORY_ID}/{REPOSITORY_REVISION}"
    readme_bytes = client.get_bytes(f"{repo_raw}/README.md")
    evaluator_bytes = client.get_bytes(f"{repo_raw}/abotn_evaluator/poi_goal/evaluator.py")
    license_bytes = client.get_bytes(f"{repo_raw}/LICENSE", allow_missing=True)
    if readme_bytes is None or evaluator_bytes is None:
        raise ValueError("pinned repository evidence is missing")
    root_files = [str(row.get("rfilename")) for row in siblings if "/" not in str(row.get("rfilename", ""))]
    card_data = metadata.get("cardData") if isinstance(metadata.get("cardData"), Mapping) else {}
    classification = classify_source(
        summary,
        dataset_license=card_data.get("license"),
        dataset_root_files=root_files,
        repository_readme=readme_bytes.decode("utf-8"),
        repository_license_present=license_bytes is not None,
        evaluator_source=evaluator_bytes.decode("utf-8"),
    )
    manifest_digest = hashlib.sha256()
    for receipt in file_receipts:
        manifest_digest.update(
            f"{receipt['path']}\0{receipt['bytes']}\0{receipt['sha256']}\n".encode("utf-8")
        )

    result = {
        "schema_version": SCHEMA,
        "created_at_utc": _utc_now(),
        "mode": "REVERSIBLE_EXPLORATION_CANARY_LITE",
        "question": "Can ABotN-POIBench supply independent functional truth after the sealed public-real 8x89 run?",
        "sources": {
            "dataset": {
                "id": DATASET_ID,
                "revision": DATASET_REVISION,
                "api_url": api_url,
                "public": True,
                "gated": False,
                "reported_storage_bytes": metadata.get("usedStorage"),
                "root_files": root_files,
                "card_license": card_data.get("license"),
            },
            "repository": {
                "id": REPOSITORY_ID,
                "revision": REPOSITORY_REVISION,
                "readme_sha256": _sha256_bytes(readme_bytes),
                "poi_evaluator_sha256": _sha256_bytes(evaluator_bytes),
                "root_license_present": license_bytes is not None,
            },
        },
        "annotation_summary": summary,
        "annotation_manifest_sha256": manifest_digest.hexdigest(),
        "annotation_files": file_receipts,
        "classification": classification,
        "claim_ceiling": "PUBLIC_DATASET_SOURCE_AVAILABILITY_AND_ANNOTATION_INTERFACE_ONLY",
        "forbidden_inferences": [
            "Do not convert the metric endpoint into a fabricated entrance bounding box.",
            "Do not expose target_position or distance_to_goal to the visual provider.",
            "Do not use this audit to reopen, supplement, or relabel the sealed 8x89 run.",
            "Do not claim redistribution, commercial, user, product, or safety permission.",
        ],
        "next_action": (
            "ONE_SCENE_RGB_RENDER_AND_PROVIDER_FIREWALL_CANARY"
            if classification["overall"] == "ARRIVAL_TRUTH_ONLY_INTERNAL_RESEARCH_CANDIDATE"
            else "STOP_SOURCE_NOT_EVALUABLE"
        ),
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.cache_dir.resolve(), args.output.resolve())
    print(json.dumps({
        "output": str(args.output.resolve()),
        "tasks": result["annotation_summary"]["task_count"],
        "scenes": result["annotation_summary"]["scene_count"],
        "overall": result["classification"]["overall"],
        "next_action": result["next_action"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
