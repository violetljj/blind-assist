#!/usr/bin/env python3
"""Validate a hash-bound, model-generated and independently model-reviewed USTRF proxy pilot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from validate_ai_review_receipt import ContractError as AiReviewContractError
from validate_ai_review_receipt import validate_model_pass


EXPECTED_CONTRACT_ID = "ustrf_sc_model_proxy_route_event_pilot_v1"
EXPECTED_SESSION_ID = "model_proxy_session_01"
EXPECTED_SCENES = (
    "parallel_boundary",
    "step_curb",
    "route_obstacle",
    "lateral_pedestrian_or_ebike",
    "unknown_low_obstacle",
)
EXPECTED_ROLES = ("positive", "matched_negative")
EXPECTED_EPISODE_COUNT = 10
EXPECTED_PAIR_COUNT = 5
EXPECTED_REVIEW_COUNT = 2
EXPECTED_RELATIONS = {"outside", "boundary", "inside", "clear", "uncertain"}
EXPECTED_STAGES = ("first_visible", "approach", "alertable", "cleared")
EXPECTED_REVIEW = {
    "reviewer_type": "ai_model",
    "review_schema": "blindassist_ustrf_sc_independent_model_review_v2",
    "adjudication_schema": "blindassist_ustrf_sc_model_adjudication_v1",
    "independent_review_count": 2,
    "required_reviewer_roles": ["gpt_multimodal_reviewer", "codex_evidence_reviewer"],
    "allowed_adjudicator_roles": ["gpt_adjudicator", "codex_adjudicator"],
    "workflow_id": "ustrf_sc_model_proxy_event_review_v1",
    "candidate_output_hidden_from_reviewers": True,
    "minimum_confidence": 0.65,
    "other_review_visible_before_submission": False,
    "adjudication_model_required_on_disagreement": True,
    "all_episodes_must_be_accepted": True,
}


class ContractError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_text(value: Any, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ContractError(f"{where} must be a lowercase SHA-256")
    return value


def local(root: Path, relative: Any, expected: Any, *, where: str) -> Path:
    expected_sha = _sha_text(expected, where=f"{where}.sha256")
    if not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute():
        raise ContractError(f"{where}.path must be a non-empty relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ContractError(f"{where} escapes evidence root") from error
    if not path.is_file() or sha256(path) != expected_sha:
        raise ContractError(f"{where} is missing or SHA-256 mismatched")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def load_json_list(path: Path, *, where: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ContractError(f"{where} must contain a JSON array of objects")
    return value


def decoded_rows(ffmpeg: Path, video: Path) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-map", "0:v:0", "-f", "framehash", "-hash", "sha256", "-",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if line and not line.startswith("#"):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 6:
                raise ContractError("unexpected ffmpeg framehash output")
            rows.append({
                "frame_index": len(rows),
                "stream_index": int(parts[0]),
                "dts": int(parts[1]),
                "pts": int(parts[2]),
                "duration": int(parts[3]),
                "decoded_size_bytes": int(parts[4]),
                "decoded_frame_sha256": parts[5].lower(),
            })
    return rows


def probe_media(ffprobe: Path, video: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(ffprobe), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,time_base:format=duration",
            "-of", "json", str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = json.loads(completed.stdout)
    streams = value.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise ContractError("ffprobe did not return exactly one video stream")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream["avg_frame_rate"],
        "time_base": stream["time_base"],
        "duration_ms": round(float(value["format"]["duration"]) * 1000),
    }


def probe_image(ffprobe: Path, image: Path) -> dict[str, int]:
    completed = subprocess.run(
        [
            str(ffprobe), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(image),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = json.loads(completed.stdout)
    streams = value.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise ContractError("ffprobe did not return exactly one image stream")
    return {"width": int(streams[0]["width"]), "height": int(streams[0]["height"])}


def tool_version(tool: Path) -> str:
    completed = subprocess.run(
        [str(tool), "-version"], check=True, capture_output=True, text=True, encoding="utf-8",
    )
    lines = completed.stdout.splitlines()
    if not lines:
        raise ContractError(f"tool returned no version line: {tool}")
    return lines[0].strip()


def toolchain_receipt(ffmpeg: Path, ffprobe: Path) -> dict[str, str]:
    return {
        "ffmpeg_sha256": sha256(ffmpeg),
        "ffmpeg_version": tool_version(ffmpeg),
        "ffprobe_sha256": sha256(ffprobe),
        "ffprobe_version": tool_version(ffprobe),
    }


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "blindassist_ustrf_sc_model_proxy_route_event_config_v1":
        raise ContractError("unexpected config schema")
    if config.get("contract_id") != EXPECTED_CONTRACT_ID:
        raise ContractError("unexpected frozen contract_id")
    if config.get("status") != "active_model_proxy_only" or config.get("benchmark_only") is not True:
        raise ContractError("config status/benchmark boundary differs from the frozen proxy pilot")
    if config.get("authority") != {
        "human_truth": False,
        "proxy_full_matrix_expansion": True,
        "proxy_u0_evaluation": False,
        "training": False,
        "android_runtime": False,
        "production": False,
    }:
        raise ContractError("config authority differs from the frozen proxy pilot")
    if config.get("matrix") != {
        "session_count": 1,
        "scene_ids": list(EXPECTED_SCENES),
        "roles": list(EXPECTED_ROLES),
        "expected_episode_count": EXPECTED_EPISODE_COUNT,
        "expected_pair_count": EXPECTED_PAIR_COUNT,
    }:
        raise ContractError("config matrix differs from the frozen proxy pilot")
    if config.get("media") != {
        "contact_sheet_width": 1536,
        "contact_sheet_height": 1024,
        "panel_count": 4,
        "panel_width": 1280,
        "panel_height": 720,
        "panel_duration_ms": 2500,
        "episode_duration_ms": 10000,
        "video_fps": 10,
        "decoded_frame_hash": "sha256",
    }:
        raise ContractError("config media contract differs from the frozen proxy pilot")
    if config.get("route_relation") != {
        "allowed": ["outside", "boundary", "inside", "clear", "uncertain"],
        "positive_required": {"panel_3": "inside", "panel_4": "clear"},
        "matched_negative_allowed": ["outside", "clear"],
    }:
        raise ContractError("config route relation contract differs from the frozen proxy pilot")
    if config.get("review") != EXPECTED_REVIEW:
        raise ContractError("config review contract differs from the frozen proxy pilot")
    if config.get("evidence_boundary") != {
        "synthetic_or_model_proxy_only": True,
        "model_generated_geometry_is_not_human_truth": True,
        "model_review_is_not_human_review": True,
        "no_real_world_effectiveness_claim": True,
        "no_default_model_replacement": True,
    }:
        raise ContractError("config evidence boundary differs from the frozen proxy pilot")
    return EXPECTED_REVIEW


def _validate_review_input(value: dict[str, Any], episode: dict[str, Any], *, where: str) -> None:
    if value.get("schema") != "blindassist_ustrf_sc_model_review_input_v1":
        raise ContractError(f"{where} schema mismatch")
    for key in ("episode_id", "contact_sheet_sha256", "video_sha256", "frame_ledger_sha256"):
        if value.get(key) != episode.get(key):
            raise ContractError(f"{where}.{key} binding mismatch")
    if value.get("candidate_output_included") is not False:
        raise ContractError(f"{where} must exclude candidate output")
    if value.get("panel_sha256s") != [panel.get("sha256") for panel in episode.get("panels", [])]:
        raise ContractError(f"{where}.panel_sha256s binding mismatch")


def validate_review(
    review: dict[str, Any], *, episode: dict[str, Any], review_input_sha: str,
    policy: dict[str, Any], where: str,
) -> tuple[str, str]:
    if review.get("schema") != policy["review_schema"] or review.get("episode_id") != episode.get("episode_id"):
        raise ContractError(f"{where} schema/episode binding mismatch")
    try:
        reviewer_id, role, _ = validate_model_pass(
            review,
            policy=policy,
            where=where,
            allowed_roles=set(policy["required_reviewer_roles"]),
        )
    except AiReviewContractError as error:
        raise ContractError(str(error)) from error
    if review.get("workflow_id") != policy["workflow_id"] or review.get("input_sha256") != review_input_sha:
        raise ContractError(f"{where} workflow/input binding mismatch")
    if review.get("contact_sheet_sha256") != episode.get("contact_sheet_sha256"):
        raise ContractError(f"{where} contact sheet binding mismatch")
    if review.get("video_sha256") != episode.get("video_sha256"):
        raise ContractError(f"{where} video binding mismatch")
    if review.get("frame_ledger_sha256") != episode.get("frame_ledger_sha256"):
        raise ContractError(f"{where} frame ledger binding mismatch")
    if review.get("verdict") not in {"accept", "reject"} or review.get("accept") is not (review.get("verdict") == "accept"):
        raise ContractError(f"{where} verdict/accept mismatch")
    if review.get("observed_role") != episode.get("pair_role"):
        raise ContractError(f"{where} observed_role mismatch")
    relations = review.get("panel_relations")
    if not isinstance(relations, list) or len(relations) != 4 or any(value not in EXPECTED_RELATIONS for value in relations):
        raise ContractError(f"{where} panel relations are invalid")
    if not isinstance(review.get("raw_panel_relations"), list) or len(review["raw_panel_relations"]) != 4:
        raise ContractError(f"{where} raw_panel_relations are required")
    if not isinstance(review.get("temporal_coherence"), bool):
        raise ContractError(f"{where} temporal_coherence must be boolean")
    if review.get("accept") is True and review.get("temporal_coherence") is not True:
        raise ContractError(f"{where} accepted a temporally incoherent episode")
    if not isinstance(review.get("issue_tags"), list) or not all(isinstance(value, str) for value in review["issue_tags"]):
        raise ContractError(f"{where} issue_tags must be a string list")
    if not isinstance(review.get("rationale"), str) or not review["rationale"].strip():
        raise ContractError(f"{where} rationale is required")
    if episode.get("pair_role") == "positive" and review.get("accept") is True:
        if relations[2] != "inside" or relations[3] != "clear":
            raise ContractError(f"{where} accepted positive lacks intrusion then clearance")
    if episode.get("pair_role") == "matched_negative" and review.get("accept") is True:
        if any(value not in {"outside", "clear"} for value in relations):
            raise ContractError(f"{where} accepted negative enters the route")
    return reviewer_id, role


def _validate_adjudication(
    value: dict[str, Any], *, episode: dict[str, Any], review_input_sha: str,
    review_hashes: list[str], reviewer_ids: list[str], policy: dict[str, Any], root: Path, where: str,
) -> None:
    if value.get("schema") != policy["adjudication_schema"] or value.get("episode_id") != episode.get("episode_id"):
        raise ContractError(f"{where} schema/episode binding mismatch")
    if value.get("method") != "independent_ai_adjudicator" or value.get("reviewer_type") != "ai_model":
        raise ContractError(f"{where} method/reviewer_type mismatch")
    for key in ("reviewer_id", "reviewer_role", "provider", "model", "model_version", "review_run_id", "workflow_id"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ContractError(f"{where}.{key} must be a non-empty string")
    if value["reviewer_role"] not in policy["allowed_adjudicator_roles"] or value["reviewer_id"] in reviewer_ids:
        raise ContractError(f"{where} must use a fresh allowed adjudicator")
    if value["workflow_id"] != policy["workflow_id"] or value.get("input_sha256") != review_input_sha:
        raise ContractError(f"{where} workflow/input binding mismatch")
    if value.get("input_review_sha256s") != review_hashes:
        raise ContractError(f"{where} must bind both ordered review hashes")
    if value.get("isolated_context") is not True or value.get("other_review_visible_before_submission") is not True:
        raise ContractError(f"{where} must be isolated while explicitly seeing both reviews")
    if value.get("candidate_output_visible") is not False:
        raise ContractError(f"{where} must remain blind to candidate output")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not policy["minimum_confidence"] <= float(confidence) <= 1:
        raise ContractError(f"{where}.confidence is below policy or outside [0,1]")
    if value.get("abstained") is not False or value.get("abstain_reasons") not in ([], None):
        raise ContractError(f"{where} abstained and cannot grant authority")
    prompt = local(root, value.get("prompt_path"), value.get("prompt_sha256"), where=f"{where}.prompt")
    if sha256(prompt) != value.get("prompt_sha256"):
        raise ContractError(f"{where} prompt binding mismatch")
    if value.get("verdict") not in {"accept", "reject"} or value.get("accept") is not (value.get("verdict") == "accept"):
        raise ContractError(f"{where} verdict/accept mismatch")


def validate(
    config: dict[str, Any], manifest: dict[str, Any], *, root: Path,
    config_sha256: str, manifest_sha256: str, actual_toolchain: dict[str, str],
    frame_provider: Callable[[Path], list[dict[str, Any]]] | None = None,
    media_provider: Callable[[Path], dict[str, Any]] | None = None,
    image_provider: Callable[[Path], dict[str, int]] | None = None,
) -> dict[str, Any]:
    review_policy = validate_config(config)
    _sha_text(config_sha256, where="config_sha256")
    _sha_text(manifest_sha256, where="manifest_sha256")
    if frame_provider is None or media_provider is None or image_provider is None:
        raise ContractError("raw media recomputation providers are required")
    if manifest.get("schema") != "blindassist_ustrf_sc_model_proxy_route_event_manifest_v1":
        raise ContractError("unexpected manifest schema")
    if manifest.get("contract_id") != config.get("contract_id"):
        raise ContractError("contract_id mismatch")
    frozen_config = local(
        root, manifest.get("frozen_config_path"), manifest.get("frozen_config_sha256"), where="manifest.frozen_config",
    )
    if manifest.get("frozen_config_sha256") != config_sha256:
        raise ContractError("manifest frozen config differs from the validated config")
    if manifest.get("toolchain") != actual_toolchain:
        raise ContractError("manifest toolchain receipt differs from the executing tools")
    provenance = manifest.get("model_generation_provenance")
    if not isinstance(provenance, dict) or set(provenance) != {"dataset_spec", "generation_records", "qa_preview"}:
        raise ContractError("manifest model-generation provenance inventory mismatch")
    provenance_paths: dict[str, Path] = {}
    for key, expected_path in {
        "dataset_spec": "dataset_spec.json",
        "generation_records": "generation_records.jsonl",
        "qa_preview": "qa/preview.html",
    }.items():
        binding = provenance.get(key)
        if not isinstance(binding, dict) or binding.get("path") != expected_path:
            raise ContractError(f"manifest provenance {key} path mismatch")
        provenance_paths[key] = local(root, binding.get("path"), binding.get("sha256"), where=f"manifest.provenance.{key}")
    generation_rows = []
    for line_number, line in enumerate(provenance_paths["generation_records"].read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ContractError(f"generation_records line {line_number} must be an object")
        generation_rows.append(value)
    accepted_generation = {
        row.get("file"): row.get("sha256") for row in generation_rows if row.get("accepted") is True
    }
    review_sources = manifest.get("independent_review_run_sources")
    if not isinstance(review_sources, list) or len(review_sources) != EXPECTED_REVIEW_COUNT:
        raise ContractError("manifest must bind exactly two raw independent review runs")
    review_source_by_id: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(review_sources):
        if not isinstance(source, dict) or not isinstance(source.get("reviewer_id"), str):
            raise ContractError(f"manifest review source {index} is invalid")
        local(root, source.get("raw_output_path"), source.get("raw_output_sha256"), where=f"manifest.review_sources[{index}].raw")
        local(root, source.get("prompt_path"), source.get("prompt_sha256"), where=f"manifest.review_sources[{index}].prompt")
        review_source_by_id[source["reviewer_id"]] = source
    if len(review_source_by_id) != EXPECTED_REVIEW_COUNT:
        raise ContractError("manifest review source reviewer IDs must be unique")
    for key, expected in {
        "benchmark_only": True,
        "human_truth": False,
        "training_eligible": False,
        "android_runtime_authorized": False,
        "production_authorized": False,
        "proxy_full_matrix_expansion_eligible": True,
        "proxy_u0_evaluation_eligible": False,
    }.items():
        if manifest.get(key) is not expected:
            raise ContractError(f"manifest must declare {key}={str(expected).lower()}")
    if manifest.get("status") != "review_complete":
        raise ContractError("manifest status must be review_complete")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != EXPECTED_EPISODE_COUNT:
        raise ContractError("model-proxy episode count mismatch")
    expected_ids = {
        f"{EXPECTED_SESSION_ID}__{scene}__pair_01__{role}"
        for scene in EXPECTED_SCENES for role in EXPECTED_ROLES
    }
    actual_ids = [episode.get("episode_id") for episode in episodes if isinstance(episode, dict)]
    if len(actual_ids) != EXPECTED_EPISODE_COUNT or len(set(actual_ids)) != EXPECTED_EPISODE_COUNT or set(actual_ids) != expected_ids:
        raise ContractError("episode IDs are not the frozen unique matrix")
    expected_matrix = Counter((scene, role) for scene in EXPECTED_SCENES for role in EXPECTED_ROLES)
    actual_matrix = Counter((row.get("scene_id"), row.get("pair_role")) for row in episodes if isinstance(row, dict))
    if actual_matrix != expected_matrix:
        raise ContractError("model-proxy matrix mismatch")

    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    contact_hashes: set[str] = set()
    video_hashes: set[str] = set()
    accepted = 0
    review_receipt_hashes: list[str] = []
    for index, episode in enumerate(episodes):
        where = f"episodes[{index}]"
        if not isinstance(episode, dict) or episode.get("authority") != "synthetic_or_model_proxy_only":
            raise ContractError(f"{where} authority mismatch")
        scene = episode.get("scene_id")
        role = episode.get("pair_role")
        pair_id = f"{EXPECTED_SESSION_ID}__{scene}__pair_01"
        if episode.get("session_id") != EXPECTED_SESSION_ID or episode.get("matched_pair_id") != pair_id:
            raise ContractError(f"{where} session/pair binding mismatch")
        if episode.get("episode_id") != f"{pair_id}__{role}":
            raise ContractError(f"{where} episode identity binding mismatch")
        if episode.get("expected_should_alert") is not (role == "positive"):
            raise ContractError(f"{where} expected_should_alert mismatch")
        sheet = local(root, episode.get("contact_sheet_path"), episode.get("contact_sheet_sha256"), where=f"{where}.contact_sheet")
        if image_provider(sheet) != {"width": 1536, "height": 1024}:
            raise ContractError(f"{where} contact sheet dimensions mismatch")
        if episode["contact_sheet_sha256"] in contact_hashes:
            raise ContractError(f"{where} reuses a contact sheet from another episode")
        contact_hashes.add(episode["contact_sheet_sha256"])
        if accepted_generation.get(Path(episode["contact_sheet_path"]).name) != episode["contact_sheet_sha256"]:
            raise ContractError(f"{where} is not bound to an accepted model-generation record")
        panels = episode.get("panels")
        if not isinstance(panels, list) or len(panels) != 4:
            raise ContractError(f"{where} must contain four panels")
        for panel_index, panel in enumerate(panels):
            if (
                not isinstance(panel, dict)
                or panel.get("panel_index") != panel_index + 1
                or panel.get("lifecycle_stage") != EXPECTED_STAGES[panel_index]
            ):
                raise ContractError(f"{where}.panels index/stage mismatch")
            panel_path = local(root, panel.get("path"), panel.get("sha256"), where=f"{where}.panels[{panel_index}]")
            if image_provider(panel_path) != {"width": 1280, "height": 720}:
                raise ContractError(f"{where}.panels[{panel_index}] dimensions mismatch")
        video = local(root, episode.get("video_path"), episode.get("video_sha256"), where=f"{where}.video")
        if episode["video_sha256"] in video_hashes:
            raise ContractError(f"{where} reuses a video from another episode")
        video_hashes.add(episode["video_sha256"])
        recomputed_media = media_provider(video)
        if episode.get("media") != recomputed_media:
            raise ContractError(f"{where} media probe mismatch")
        if (
            recomputed_media.get("width") != 1280
            or recomputed_media.get("height") != 720
            or recomputed_media.get("avg_frame_rate") != "10/1"
            or recomputed_media.get("duration_ms") != 10000
            or not isinstance(recomputed_media.get("time_base"), str)
            or not recomputed_media["time_base"]
        ):
            raise ContractError(f"{where} media does not match the frozen 10-second contract")
        ledger_path = local(root, episode.get("frame_ledger_path"), episode.get("frame_ledger_sha256"), where=f"{where}.ledger")
        ledger = load_json(ledger_path)
        if (
            ledger.get("schema") != "blindassist_ustrf_sc_model_proxy_frame_ledger_v1"
            or ledger.get("episode_id") != episode.get("episode_id")
            or ledger.get("video_sha256") != episode.get("video_sha256")
            or ledger.get("time_base") != recomputed_media["time_base"]
        ):
            raise ContractError(f"{where} ledger binding mismatch")
        frames = ledger.get("frames")
        if not isinstance(frames, list) or len(frames) != 100:
            raise ContractError(f"{where} expected exactly 100 decoded frames")
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise ContractError(f"{where}.frames[{frame_index}] must be an object")
            if frame.get("frame_index") != frame_index:
                raise ContractError(f"{where} frame ledger is not contiguous")
            digest = frame.get("decoded_frame_sha256")
            _sha_text(digest, where=f"{where}.frames[{frame_index}].decoded_frame_sha256")
            for key in ("stream_index", "dts", "pts", "duration", "decoded_size_bytes"):
                if not isinstance(frame.get(key), int) or isinstance(frame.get(key), bool):
                    raise ContractError(f"{where}.frames[{frame_index}].{key} must be an integer")
        if any(current["pts"] <= previous["pts"] for previous, current in zip(frames, frames[1:])):
            raise ContractError(f"{where} frame PTS must be strictly increasing")
        if frame_provider(video) != frames:
            raise ContractError(f"{where} decoded frame ledger mismatch")

        review_input_path = local(
            root, episode.get("review_input_path"), episode.get("review_input_sha256"), where=f"{where}.review_input",
        )
        review_input = load_json(review_input_path)
        _validate_review_input(review_input, episode, where=f"{where}.review_input")
        review_input_sha = episode["review_input_sha256"]
        reviews = episode.get("independent_model_reviews")
        if not isinstance(reviews, list) or len(reviews) != EXPECTED_REVIEW_COUNT:
            raise ContractError(f"{where} independent model review count mismatch")
        review_values: list[dict[str, Any]] = []
        reviewer_ids: list[str] = []
        reviewer_roles: list[str] = []
        review_hashes: list[str] = []
        review_run_ids: list[str] = []
        for review_index, binding in enumerate(reviews):
            if not isinstance(binding, dict):
                raise ContractError(f"{where}.reviews[{review_index}] binding must be an object")
            review_path = local(root, binding.get("path"), binding.get("sha256"), where=f"{where}.reviews[{review_index}]")
            review = load_json(review_path)
            prompt_path = local(root, review.get("prompt_path"), review.get("prompt_sha256"), where=f"{where}.reviews[{review_index}].prompt")
            if sha256(prompt_path) != review.get("prompt_sha256"):
                raise ContractError(f"{where}.reviews[{review_index}] prompt binding mismatch")
            reviewer_id, reviewer_role = validate_review(
                review,
                episode=episode,
                review_input_sha=review_input_sha,
                policy=review_policy,
                where=f"{where}.reviews[{review_index}]",
            )
            source = review_source_by_id.get(reviewer_id)
            if source is None:
                raise ContractError(f"{where}.reviews[{review_index}] lacks a raw review source binding")
            if (
                review.get("raw_output_path") != source.get("raw_output_path")
                or review.get("raw_output_sha256") != source.get("raw_output_sha256")
                or review.get("prompt_path") != source.get("prompt_path")
                or review.get("prompt_sha256") != source.get("prompt_sha256")
            ):
                raise ContractError(f"{where}.reviews[{review_index}] source/prompt binding mismatch")
            raw_rows = load_json_list(root / source["raw_output_path"], where=f"{where}.reviews[{review_index}].raw_output")
            raw_row = next((row for row in raw_rows if row.get("file") == Path(episode["contact_sheet_path"]).name), None)
            if not isinstance(raw_row, dict):
                raise ContractError(f"{where}.reviews[{review_index}] raw output lacks this episode")
            for key in ("accept", "observed_role", "temporal_coherence", "issue_tags", "rationale"):
                if review.get(key) != raw_row.get(key):
                    raise ContractError(f"{where}.reviews[{review_index}] differs from raw output field {key}")
            if review.get("raw_panel_relations") != raw_row.get("panel_relations"):
                raise ContractError(f"{where}.reviews[{review_index}] raw relation binding mismatch")
            review_values.append(review)
            reviewer_ids.append(reviewer_id)
            reviewer_roles.append(reviewer_role)
            review_hashes.append(binding["sha256"])
            review_run_ids.append(review["review_run_id"])
            review_receipt_hashes.append(binding["sha256"])
        if len(set(reviewer_ids)) != EXPECTED_REVIEW_COUNT or len(set(review_run_ids)) != EXPECTED_REVIEW_COUNT:
            raise ContractError(f"{where} model reviewer identities/runs must be distinct")
        if set(reviewer_roles) != set(review_policy["required_reviewer_roles"]):
            raise ContractError(f"{where} does not satisfy both frozen reviewer roles")
        agrees = len({(review["accept"], tuple(review["panel_relations"])) for review in review_values}) == 1
        adjudication = episode.get("model_adjudication")
        if not agrees:
            if not isinstance(adjudication, dict):
                raise ContractError(f"{where} disagreement requires model adjudication")
            adjudication_path = local(root, adjudication.get("path"), adjudication.get("sha256"), where=f"{where}.adjudication")
            final_review = load_json(adjudication_path)
            _validate_adjudication(
                final_review,
                episode=episode,
                review_input_sha=review_input_sha,
                review_hashes=review_hashes,
                reviewer_ids=reviewer_ids,
                policy=review_policy,
                root=root,
                where=f"{where}.adjudication",
            )
            review_receipt_hashes.append(adjudication["sha256"])
        else:
            if adjudication is not None:
                raise ContractError(f"{where} consensus episode must not carry hidden adjudication")
            final_review = review_values[0]
        if final_review.get("accept") is True:
            accepted += 1
        pairs[str(episode.get("matched_pair_id"))].append(episode)

    if len(pairs) != EXPECTED_PAIR_COUNT:
        raise ContractError("model-proxy matched-pair count mismatch")
    for pair_id, members in pairs.items():
        if len(members) != 2 or {row.get("pair_role") for row in members} != set(EXPECTED_ROLES):
            raise ContractError(f"{pair_id} needs exactly one positive and one matched negative")
        if len({row.get("scene_id") for row in members}) != 1 or len({row.get("session_id") for row in members}) != 1:
            raise ContractError(f"{pair_id} crosses scene/session boundaries")
        if len({row.get("contact_sheet_sha256") for row in members}) != 2 or len({row.get("video_sha256") for row in members}) != 2:
            raise ContractError(f"{pair_id} positive/negative media must differ")
    all_accepted = accepted == EXPECTED_EPISODE_COUNT
    if review_policy["all_episodes_must_be_accepted"] and not all_accepted:
        raise ContractError("one or more model-proxy episodes were rejected")
    return {
        "schema": "blindassist_ustrf_sc_model_proxy_route_event_audit_v1",
        "contract_id": config["contract_id"],
        "model_proxy_pipeline_audit_passed": True,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "toolchain": actual_toolchain,
        "review_receipt_sha256s": review_receipt_hashes,
        "episode_count": len(episodes),
        "matched_pair_count": len(pairs),
        "accepted_episode_count": accepted,
        "proxy_full_matrix_expansion_eligible": all_accepted,
        "proxy_u0_evaluation_eligible": False,
        "human_truth": False,
        "training_eligible": False,
        "android_runtime_authorized": False,
        "production_authorized": False,
    }
