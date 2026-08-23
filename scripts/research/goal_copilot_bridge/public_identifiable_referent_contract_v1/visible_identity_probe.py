"""Run one visible-only passive identity probe over the consumed C2 roster.

The provider sees only the public reference evidence and one later image. Native
SUN3D instance annotations stay in the evaluator-private side of the run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests
from PIL import Image, ImageDraw

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import c2_small_roster as c2
from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import contract


SCHEMA_VERSION = "blindassist_visible_only_passive_identity_probe_v0"
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "high"
EXPECTED_EPISODES = 7
EXPECTED_OBSERVATIONS = 21
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CLAIM_CEILING = (
    "CONSUMED_PUBLIC_REAL_VISIBLE_ONLY_DISCOVERY_FAILURE_ANATOMY_NO_NOT_VISIBLE_"
    "CALIBRATION_ALGORITHM_NAVIGATION_CONTROL_SAFETY_OR_PRODUCT_CLAIM"
)
FORBIDDEN_PROVIDER_TOKENS = (
    "physical_instance_id",
    "native_object_id",
    "private_identity",
    "private-truth",
    "target_regions",
    "world_anchor",
)
PROMPT_TEMPLATE = """You are a passive visual referent localizer. Exactly three images are attached in this order:
1. the clean reference scene;
2. the same reference scene with a red rectangle marking the only target instance;
3. a later view from the same real environment.

The red rectangle is only an input selector; red is not part of the object's appearance. Find that same physical object instance in image 3. Other objects of the same category are distractors. Use instance-specific appearance and stable scene context across the viewpoint change, not category alone.

If the same physical instance cannot be identified from the supplied visual evidence, return ABSTAIN. If it can, return a tight bounding box in image 3 using integer coordinates from 0 to 1000, where x increases left-to-right and y increases top-to-bottom. Do not inspect files, call tools, browse, or use any information beyond the three attached images. Return only the JSON object required by the output schema.

The public target rectangle in image 1, normalized to 0..1, is: {target_bbox}.
"""

OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["FOUND", "ABSTAIN"]},
        "region_xyxy_norm_1000": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "minItems": 4,
                    "maxItems": 4,
                },
                {"type": "null"},
            ]
        },
    },
    "required": ["decision", "region_xyxy_norm_1000"],
    "additionalProperties": False,
}


class ProbeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _body_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("body_sha256", None)
    return contract.content_sha256(payload)


def preflight_codex(codex_exe: Path, model: str) -> dict[str, Any]:
    _require(codex_exe.is_file(), f"Codex executable is missing: {codex_exe}")
    version_run = subprocess.run(
        [str(codex_exe), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        shell=False,
    )
    version = version_run.stdout.strip()
    _require(version_run.returncode == 0, "Codex --version failed")
    _require(version.startswith("codex-cli ") and "unknown" not in version.lower(), "Codex version is invalid")
    login_run = subprocess.run(
        [str(codex_exe), "login", "status"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        shell=False,
    )
    login_status = (login_run.stdout + login_run.stderr).strip()
    _require(login_run.returncode == 0 and "Logged in using ChatGPT" in login_status, "ChatGPT login is unavailable")
    catalog_run = subprocess.run(
        [str(codex_exe), "debug", "models"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        shell=False,
    )
    _require(catalog_run.returncode == 0, "Codex model catalog lookup failed")
    catalog = json.loads(catalog_run.stdout)
    models = catalog if isinstance(catalog, list) else catalog.get("models", [])
    selected = next((item for item in models if item.get("slug") == model), None)
    _require(selected is not None and "image" in selected.get("input_modalities", []), f"{model} image input unavailable")
    return {
        "resolved_executable_path": str(codex_exe.resolve()),
        "version": version,
        "executable_sha256": _sha256_file(codex_exe),
        "authentication": "CHATGPT",
        "model": model,
        "reasoning_effort": REASONING_EFFORT,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_body_hash(value: Mapping[str, Any], name: str) -> None:
    _require(value.get("body_sha256") == _body_hash(value), f"{name} body SHA mismatch")


def _download_native_annotation(episode: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    response = requests.get(episode["source"]["annotation_url"], timeout=90)
    response.raise_for_status()
    payload = response.content
    expected = episode["source"]["annotation_sha256"]
    _require(_sha256_bytes(payload) == expected, f"{episode['case_id']} native annotation SHA mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    value = json.loads(payload)
    _require(isinstance(value, Mapping), f"{episode['case_id']} annotation root is not an object")
    return value


def native_instances_for_frame(
    annotation: Mapping[str, Any], sequence: str, frame_id: int
) -> list[dict[str, Any]]:
    frames = annotation.get("frames", [])
    _require(0 <= frame_id < len(frames), "native frame index is out of range")
    best_by_object: dict[int, tuple[float, list[float]]] = {}
    for polygon in frames[frame_id].get("polygon", []):
        object_id = int(polygon["object"])
        if object_id < 0 or object_id >= len(annotation.get("objects", [])):
            continue
        item = annotation["objects"][object_id]
        clipped = c2._clipped_bbox(polygon)
        if item is None or clipped is None:
            continue
        bbox, area = clipped
        previous = best_by_object.get(object_id)
        if previous is None or area > previous[0]:
            best_by_object[object_id] = (area, bbox)
    instances = []
    for object_id, (area, bbox) in sorted(best_by_object.items()):
        raw_label = str(annotation["objects"][object_id]["name"])
        instances.append(
            {
                "native_object_id": object_id,
                "physical_instance_id": f"sun3d:{sequence}:object:{object_id}",
                "raw_label": raw_label,
                "normalized_label": c2._normalize_label(raw_label),
                "bbox_xyxy_normalized": bbox,
                "bbox_area_fraction": area,
            }
        )
    return instances


def _bbox_close(left: Sequence[float], right: Sequence[float], tolerance: float = 1e-9) -> bool:
    return len(left) == len(right) == 4 and all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))


def _draw_target_overlay(reference_path: Path, target_bbox: Sequence[float], output_path: Path) -> None:
    with Image.open(reference_path) as source:
        image = source.convert("RGB")
    _require(image.size == (IMAGE_WIDTH, IMAGE_HEIGHT), "reference image dimensions drifted")
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = (
        int(round(float(target_bbox[0]) * IMAGE_WIDTH)),
        int(round(float(target_bbox[1]) * IMAGE_HEIGHT)),
        int(round(float(target_bbox[2]) * IMAGE_WIDTH)),
        int(round(float(target_bbox[3]) * IMAGE_HEIGHT)),
    )
    for offset in range(5):
        draw.rectangle((x0 - offset, y0 - offset, x1 + offset, y1 + offset), outline=(255, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def _assert_provider_firewall(workspace: Path) -> None:
    for path in workspace.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_PROVIDER_TOKENS:
            _require(token not in text, f"provider firewall rejected token {token} in {path.name}")


def prepare_run(c2_root: Path, run_dir: Path, provider: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(not run_dir.exists(), f"run directory already exists: {run_dir}")
    public_manifest_path = c2_root / "public-manifest.json"
    private_manifest_path = c2_root / "private-evidence-manifest.json"
    roster_path = c2_root.parent / "frozen-roster-private.json"
    public_manifest = _load_json(public_manifest_path)
    private_manifest = _load_json(private_manifest_path)
    roster = _load_json(roster_path)
    _verify_body_hash(public_manifest, "C2 public manifest")
    _verify_body_hash(private_manifest, "C2 private manifest")
    _verify_body_hash(roster, "C2 frozen roster")
    _require(public_manifest.get("episode_count") == EXPECTED_EPISODES, "C2 episode count drifted")
    _require(sum(len(item["later_observations"]) for item in public_manifest["episodes"]) == EXPECTED_OBSERVATIONS, "C2 observation count drifted")
    run_dir.mkdir(parents=True, exist_ok=False)
    provider_root = run_dir / "provider-public"
    private_root = run_dir / "evaluator-private"
    _atomic_json(provider_root / "output-schema.json", OUTPUT_SCHEMA)
    roster_by_id = {item["case_id"]: item for item in roster["episodes"]}
    private_by_id = {item["case_id"]: item for item in private_manifest["episodes"]}
    observations: list[dict[str, Any]] = []
    for public_episode in public_manifest["episodes"]:
        case_id = public_episode["case_id"]
        roster_episode = roster_by_id[case_id]
        private_episode = private_by_id[case_id]
        contract_path = c2_root / public_episode["public_contract_relative_path"]
        public_contract = _load_json(contract_path)
        _verify_body_hash(public_contract, f"{case_id} public contract")
        reference_source = c2_root / public_episode["reference_image_relative_path"]
        _require(_sha256_file(reference_source) == public_episode["reference_image_sha256"], f"{case_id} reference SHA mismatch")
        annotation_path = private_root / "native-annotations" / f"{case_id}.json"
        annotation = _download_native_annotation(roster_episode, annotation_path)
        truth = _load_json(c2_root / private_episode["private_truth_relative_path"])
        truth_by_id = {item["observation_id"]: item for item in truth["observations"]}
        later_by_id = {item["observation_id"]: item for item in roster_episode["later_observations"]}
        target_bbox = public_contract["reference_image"]["public_target_region_xyxy"]
        for public_observation in public_episode["later_observations"]:
            observation_id = public_observation["observation_id"]
            truth_row = truth_by_id[observation_id]
            later_row = later_by_id[observation_id]
            _require(truth_row["visibility"] == "VISIBLE", f"{observation_id} is not VISIBLE")
            _require(len(truth_row["target_regions"]) == 1, f"{observation_id} target truth is not singular")
            later_source = c2_root / public_observation["image_relative_path"]
            _require(_sha256_file(later_source) == public_observation["image_sha256"], f"{observation_id} later SHA mismatch")
            instances = native_instances_for_frame(annotation, roster_episode["sequence"], int(later_row["frame_id"]))
            target_instance = next(
                (item for item in instances if item["native_object_id"] == roster_episode["native_object_id"]), None
            )
            _require(target_instance is not None, f"{observation_id} target is absent from native instances")
            truth_target = truth_row["target_regions"][0]
            _require(_bbox_close(target_instance["bbox_xyxy_normalized"], truth_target["bbox_xyxy_normalized"]), f"{observation_id} target bbox disagrees with C2 truth")
            workspace = provider_root / "cases" / observation_id
            workspace.mkdir(parents=True, exist_ok=False)
            reference_copy = workspace / "01-reference.jpg"
            overlay_path = workspace / "02-reference-target.png"
            later_copy = workspace / "03-later.jpg"
            shutil.copy2(reference_source, reference_copy)
            shutil.copy2(later_source, later_copy)
            _draw_target_overlay(reference_copy, target_bbox, overlay_path)
            prompt = PROMPT_TEMPLATE.format(target_bbox=json.dumps(target_bbox, separators=(",", ":")))
            _atomic_text(workspace / "prompt.txt", prompt)
            public_input = {
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "observation_id": observation_id,
                "roles": ["CLEAN_REFERENCE", "PUBLIC_TARGET_OVERLAY", "LATER_IMAGE"],
                "public_target_region_xyxy_normalized": target_bbox,
                "reference_image_sha256": public_episode["reference_image_sha256"],
                "later_image_sha256": public_observation["image_sha256"],
            }
            _atomic_json(workspace / "public-input.json", public_input)
            private_input = {
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "observation_id": observation_id,
                "target_physical_instance_id": truth_target["physical_instance_id"],
                "target_normalized_label": roster_episode["private_normalized_label"],
                "native_instances": instances,
            }
            private_path = private_root / "observations" / f"{observation_id}.json"
            _atomic_json(private_path, private_input)
            _assert_provider_firewall(workspace)
            observations.append(
                {
                    "case_id": case_id,
                    "observation_id": observation_id,
                    "workspace_relative_path": workspace.relative_to(run_dir).as_posix(),
                    "private_input_relative_path": private_path.relative_to(run_dir).as_posix(),
                }
            )
    _require(len(observations) == EXPECTED_OBSERVATIONS, "prepared observation count mismatch")
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "mode": "REVERSIBLE_EXPLORATION",
        "data_role": "CONSUMED_C2_PUBLIC_REAL_VISIBLE_ONLY",
        "provider": dict(provider),
        "prompt_sha256": _sha256_bytes(PROMPT_TEMPLATE.encode("utf-8")),
        "output_schema_sha256": _sha256_file(provider_root / "output-schema.json"),
        "c2_public_manifest_body_sha256": public_manifest["body_sha256"],
        "c2_private_manifest_body_sha256": private_manifest["body_sha256"],
        "c2_roster_body_sha256": roster["body_sha256"],
        "observation_count": len(observations),
        "assignment_rule": "PREDICTED_BOX_CENTER_CONTAINMENT_THEN_HIGHEST_IOU_NO_THRESHOLD",
        "retry_policy": "NO_AUTOMATIC_RETRY_DISPATCHED_OR_COMPLETED_CALLS",
        "claim_ceiling": CLAIM_CEILING,
        "observations": observations,
    }
    run_config["body_sha256"] = _body_hash(run_config)
    _atomic_json(run_dir / "run-config.json", run_config)
    return observations


def _provider_command(codex_exe: Path, workspace: Path, output_schema: Path, prompt: str) -> list[str]:
    return [
        str(codex_exe),
        "exec",
        "--model",
        MODEL,
        "--config",
        f'model_reasoning_effort="{REASONING_EFFORT}"',
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--json",
        "--cd",
        str(workspace),
        "--output-schema",
        str(output_schema),
        "--output-last-message",
        str(workspace / "last-message.json"),
        prompt,
        "--image",
        str(workspace / "01-reference.jpg"),
        "--image",
        str(workspace / "02-reference-target.png"),
        "--image",
        str(workspace / "03-later.jpg"),
    ]


def _run_provider_call(codex_exe: Path, run_dir: Path, observation: Mapping[str, Any], timeout_s: int) -> dict[str, Any]:
    workspace = run_dir / observation["workspace_relative_path"]
    dispatch_path = workspace / "dispatch.json"
    _require(not dispatch_path.exists(), f"{observation['observation_id']} was already dispatched")
    dispatch = {
        "schema_version": SCHEMA_VERSION,
        "observation_id": observation["observation_id"],
        "state": "DISPATCHED",
        "dispatched_at_utc": _utc_now(),
    }
    _atomic_json(dispatch_path, dispatch)
    prompt = (workspace / "prompt.txt").read_text(encoding="utf-8")
    command = _provider_command(codex_exe, workspace, run_dir / "provider-public" / "output-schema.json", prompt)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
            shell=False,
        )
        duration = time.monotonic() - started
        _atomic_text(workspace / "stdout.jsonl", completed.stdout)
        _atomic_text(workspace / "stderr.txt", completed.stderr)
        state = "COMPLETED" if completed.returncode == 0 else "PROVIDER_ERROR"
        result = {
            **dispatch,
            "state": state,
            "completed_at_utc": _utc_now(),
            "duration_seconds": round(duration, 3),
            "returncode": completed.returncode,
        }
    except subprocess.TimeoutExpired as error:
        duration = time.monotonic() - started
        _atomic_text(workspace / "stdout.jsonl", error.stdout or "")
        _atomic_text(workspace / "stderr.txt", error.stderr or "")
        result = {
            **dispatch,
            "state": "IN_DOUBT_TIMEOUT_NO_RETRY",
            "completed_at_utc": _utc_now(),
            "duration_seconds": round(duration, 3),
            "returncode": None,
        }
    _atomic_json(workspace / "provider-result.json", result)
    return result


def execute_provider(
    codex_exe: Path,
    run_dir: Path,
    observations: Sequence[Mapping[str, Any]],
    workers: int,
    timeout_s: int,
) -> list[dict[str, Any]]:
    _require(bool(observations), "no observations were prepared")
    first = _run_provider_call(codex_exe, run_dir, observations[0], timeout_s)
    print(f"[1/{len(observations)}] {observations[0]['observation_id']} {first['state']} {first['duration_seconds']:.1f}s", flush=True)
    _require(first["state"] == "COMPLETED", "first provider call failed transport qualification")
    first_workspace = run_dir / observations[0]["workspace_relative_path"]
    parse_prediction(_load_json(first_workspace / "last-message.json"))
    results = [first]
    completed_count = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_provider_call, codex_exe, run_dir, observation, timeout_s): observation
            for observation in observations[1:]
        }
        for future in concurrent.futures.as_completed(futures):
            observation = futures[future]
            result = future.result()
            results.append(result)
            completed_count += 1
            print(
                f"[{completed_count}/{len(observations)}] {observation['observation_id']} {result['state']} {result['duration_seconds']:.1f}s",
                flush=True,
            )
    return results


def parse_prediction(value: Mapping[str, Any]) -> dict[str, Any]:
    decision = value.get("decision")
    region = value.get("region_xyxy_norm_1000")
    _require(decision in {"FOUND", "ABSTAIN"}, "prediction decision is invalid")
    if decision == "ABSTAIN":
        _require(region is None, "ABSTAIN must have a null region")
        return {"decision": decision, "region_xyxy_normalized": None}
    _require(isinstance(region, list) and len(region) == 4, "FOUND must have a four-value region")
    _require(all(type(item) is int and 0 <= item <= 1000 for item in region), "FOUND region values are invalid")
    normalized = [item / 1000.0 for item in region]
    _require(normalized[0] < normalized[2] and normalized[1] < normalized[3], "FOUND region is degenerate")
    return {"decision": decision, "region_xyxy_normalized": normalized}


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix0 = max(float(left[0]), float(right[0]))
    iy0 = max(float(left[1]), float(right[1]))
    ix1 = min(float(left[2]), float(right[2]))
    iy1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def assign_region(region: Sequence[float], private_input: Mapping[str, Any]) -> dict[str, Any]:
    center_x = (float(region[0]) + float(region[2])) / 2.0
    center_y = (float(region[1]) + float(region[3])) / 2.0
    containing = []
    for instance in private_input["native_instances"]:
        bbox = instance["bbox_xyxy_normalized"]
        if float(bbox[0]) <= center_x <= float(bbox[2]) and float(bbox[1]) <= center_y <= float(bbox[3]):
            containing.append((bbox_iou(region, bbox), instance))
    target = next(
        item for item in private_input["native_instances"] if item["physical_instance_id"] == private_input["target_physical_instance_id"]
    )
    target_iou = bbox_iou(region, target["bbox_xyxy_normalized"])
    if not containing:
        return {"identity_outcome": "BACKGROUND", "assigned_instance": None, "assigned_iou": 0.0, "target_iou": target_iou}
    assigned_iou, assigned = sorted(
        containing,
        key=lambda item: (-item[0], float(item[1]["bbox_area_fraction"]), int(item[1]["native_object_id"])),
    )[0]
    if assigned["physical_instance_id"] == private_input["target_physical_instance_id"]:
        outcome = "SAME_INSTANCE"
    elif assigned["normalized_label"] == private_input["target_normalized_label"]:
        outcome = "SAME_CLASS_DISTRACTOR"
    else:
        outcome = "UNRELATED_OBJECT"
    return {
        "identity_outcome": outcome,
        "assigned_instance": assigned,
        "assigned_iou": assigned_iou,
        "target_iou": target_iou,
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {
        "observation_count": len(rows),
        "found_count": sum(item["identity_outcome"] in {"SAME_INSTANCE", "SAME_CLASS_DISTRACTOR", "UNRELATED_OBJECT", "BACKGROUND"} for item in rows),
        "same_instance_correct_count": sum(item["identity_outcome"] == "SAME_INSTANCE" for item in rows),
        "same_class_distractor_count": sum(item["identity_outcome"] == "SAME_CLASS_DISTRACTOR" for item in rows),
        "unrelated_object_count": sum(item["identity_outcome"] == "UNRELATED_OBJECT" for item in rows),
        "background_count": sum(item["identity_outcome"] == "BACKGROUND" for item in rows),
        "abstain_count": sum(item["identity_outcome"] == "ABSTAIN" for item in rows),
        "provider_failure_count": sum(item["identity_outcome"] == "PROVIDER_FAILURE" for item in rows),
        "invalid_output_count": sum(item["identity_outcome"] == "INVALID_OUTPUT" for item in rows),
    }
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    episodes = []
    for case_id, case_rows in sorted(by_case.items()):
        ordered = sorted(case_rows, key=lambda item: str(item["observation_id"]))
        outcomes = [str(item["identity_outcome"]) for item in ordered]
        episodes.append(
            {
                "case_id": case_id,
                "outcomes": outcomes,
                "found_count": sum(item in {"SAME_INSTANCE", "SAME_CLASS_DISTRACTOR", "UNRELATED_OBJECT", "BACKGROUND"} for item in outcomes),
                "same_instance_correct_count": outcomes.count("SAME_INSTANCE"),
                "three_view_stable_same_instance": outcomes == ["SAME_INSTANCE"] * 3,
            }
        )
    counts["three_view_stable_same_instance_episode_count"] = sum(item["three_view_stable_same_instance"] for item in episodes)
    counts["episodes"] = episodes
    return counts


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    config = _load_json(run_dir / "run-config.json")
    _verify_body_hash(config, "run config")
    rows = []
    for observation in config["observations"]:
        workspace = run_dir / observation["workspace_relative_path"]
        private_input = _load_json(run_dir / observation["private_input_relative_path"])
        provider_result = _load_json(workspace / "provider-result.json")
        row: dict[str, Any] = {
            "case_id": observation["case_id"],
            "observation_id": observation["observation_id"],
            "provider_state": provider_result["state"],
            "duration_seconds": provider_result["duration_seconds"],
        }
        if provider_result["state"] != "COMPLETED":
            row["identity_outcome"] = "PROVIDER_FAILURE"
        else:
            try:
                raw_prediction = _load_json(workspace / "last-message.json")
                prediction = parse_prediction(raw_prediction)
                row["prediction"] = prediction
                if prediction["decision"] == "ABSTAIN":
                    row["identity_outcome"] = "ABSTAIN"
                else:
                    row.update(assign_region(prediction["region_xyxy_normalized"], private_input))
            except (OSError, json.JSONDecodeError, ProbeError, StopIteration) as error:
                row["identity_outcome"] = "INVALID_OUTPUT"
                row["invalid_output_error"] = str(error)
        rows.append(row)
    summary = summarize_rows(rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_at_utc": _utc_now(),
        "run_config_body_sha256": config["body_sha256"],
        "model": config["provider"]["model"],
        "reasoning_effort": config["provider"]["reasoning_effort"],
        "data_role": config["data_role"],
        "visibility_scope": "VISIBLE_ONLY_21_OF_21",
        "metrics": summary,
        "rows": rows,
        "success_gate": None,
        "claim_ceiling": CLAIM_CEILING,
        "terminal": "VISIBLE_ONLY_PASSIVE_IDENTITY_FAILURE_ANATOMY_OBSERVED",
    }
    report["body_sha256"] = _body_hash(report)
    _atomic_json(run_dir / "final-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--codex-exe", type=Path, default=Path(r"E:\codex-tools\bin\codex.exe"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=420)
    args = parser.parse_args(argv)
    _require(1 <= args.workers <= 3, "workers must be between 1 and 3")
    _require(args.timeout_seconds >= 60, "timeout must be at least 60 seconds")
    provider = preflight_codex(args.codex_exe.resolve(), MODEL)
    observations = prepare_run(args.c2_root.resolve(), args.run_dir.resolve(), provider)
    provider_results = execute_provider(
        args.codex_exe.resolve(), args.run_dir.resolve(), observations, args.workers, args.timeout_seconds
    )
    report = evaluate_run(args.run_dir.resolve())
    print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if all(item["state"] == "COMPLETED" for item in provider_results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
