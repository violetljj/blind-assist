"""Single-frame adapter for the unchanged frozen P0 providers.

This module deliberately reuses the exact Grounding DINO inference function and
the exact single-Brain baseline render/prompt/schema/output functions.  It adds
only operational packaging for a live frame; it does not select a model,
checkpoint, threshold, prompt policy, or competing arm.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as dino
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_silver_b_brain_baseline as brain


CODEX_VERSION = "codex-cli 0.149.0"
CODEX_SHA256 = "14b7e6b2356e82d1d9275579eaa588757b4e0a501b65dcc19fccdf77bd83dc00"
CODEX_MODEL = "gpt-5.6-terra"
CODEX_REASONING_EFFORT = "medium"
DECISION_POLICY_ID = brain.POLICY_ID
PROVIDER_LOCK_SCHEMA = "blindassist_last_10m_p0_provider_lock_v1"


class ProviderAdapterError(RuntimeError):
    """Raised when the frozen provider cannot be reused exactly."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_text(command: list[str], *, timeout: int = 30) -> str:
    result = subprocess.run(
        command,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        raise ProviderAdapterError(f"provider preflight failed: {' '.join(command[1:])}: {result.stderr.strip()}")
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    return stdout if stdout else stderr


def preflight_provider(*, codex_exe: Path, model_dir: Path) -> dict[str, Any]:
    """Validate the already-selected P0 providers before any model call."""

    executable = codex_exe.resolve()
    model = model_dir.resolve()
    if not executable.is_file():
        raise ProviderAdapterError("frozen Codex executable is missing")
    version = _run_text([str(executable), "--version"])
    if version != CODEX_VERSION or "unknown" in version.lower():
        raise ProviderAdapterError(f"Codex version drift: {version!r}")
    login = _run_text([str(executable), "login", "status"])
    if "Logged in using ChatGPT" not in login:
        raise ProviderAdapterError("Codex ChatGPT authentication is unavailable")
    executable_sha256 = _sha256_file(executable)
    if executable_sha256 != CODEX_SHA256:
        raise ProviderAdapterError("Codex executable hash drift")
    weights = model / dino.WEIGHTS_FILENAME
    if not weights.is_file() or _sha256_file(weights) != dino.WEIGHTS_SHA256:
        raise ProviderAdapterError("frozen Grounding DINO checkpoint is missing or changed")
    return {
        "schema_version": PROVIDER_LOCK_SCHEMA,
        "codex": {
            "executable": str(executable),
            "executable_sha256": executable_sha256,
            "cli_version": version,
            "authentication": "CHATGPT_LOGIN_PREFLIGHT_CONFIRMED",
            "model": CODEX_MODEL,
            "reasoning_effort": CODEX_REASONING_EFFORT,
        },
        "grounding_dino": {
            "model_dir": str(model),
            "model_repository": dino.MODEL_REPOSITORY,
            "model_revision": dino.MODEL_REVISION,
            "weights_sha256": dino.WEIGHTS_SHA256,
            "prompt": dino.PROMPT,
            "box_threshold": dino.BOX_THRESHOLD,
            "text_threshold": dino.TEXT_THRESHOLD,
            "nms_iou_threshold": dino.NMS_IOU_THRESHOLD,
            "max_proposals_per_image": dino.MAX_PROPOSALS_PER_IMAGE,
            "authority": dino.GENERATOR_AUTHORITY,
        },
        "brain_policy_id": DECISION_POLICY_ID,
        "claim_ceiling": "MECHANICAL_PROVIDER_REUSE_ONLY_NO_SCIENTIFIC_CONFIRMATION",
    }


def verify_provider_lock(value: Mapping[str, Any]) -> tuple[Path, Path]:
    if value.get("schema_version") != PROVIDER_LOCK_SCHEMA:
        raise ProviderAdapterError("provider lock schema mismatch")
    codex = value.get("codex")
    grounding = value.get("grounding_dino")
    if not isinstance(codex, Mapping) or not isinstance(grounding, Mapping):
        raise ProviderAdapterError("provider lock is incomplete")
    expected_codex = {
        "executable_sha256": CODEX_SHA256,
        "cli_version": CODEX_VERSION,
        "model": CODEX_MODEL,
        "reasoning_effort": CODEX_REASONING_EFFORT,
    }
    if any(codex.get(key) != expected for key, expected in expected_codex.items()):
        raise ProviderAdapterError("provider lock Codex identity drift")
    expected_dino = {
        "model_repository": dino.MODEL_REPOSITORY,
        "model_revision": dino.MODEL_REVISION,
        "weights_sha256": dino.WEIGHTS_SHA256,
        "prompt": dino.PROMPT,
        "box_threshold": dino.BOX_THRESHOLD,
        "text_threshold": dino.TEXT_THRESHOLD,
        "nms_iou_threshold": dino.NMS_IOU_THRESHOLD,
        "max_proposals_per_image": dino.MAX_PROPOSALS_PER_IMAGE,
        "authority": dino.GENERATOR_AUTHORITY,
    }
    if any(grounding.get(key) != expected for key, expected in expected_dino.items()):
        raise ProviderAdapterError("provider lock Grounding DINO identity drift")
    if value.get("brain_policy_id") != DECISION_POLICY_ID:
        raise ProviderAdapterError("provider lock Brain policy drift")
    executable = Path(str(codex.get("executable"))).resolve()
    model_dir = Path(str(grounding.get("model_dir"))).resolve()
    if not executable.is_file() or _sha256_file(executable) != CODEX_SHA256:
        raise ProviderAdapterError("locked Codex executable is unavailable or changed")
    weights = model_dir / dino.WEIGHTS_FILENAME
    if not weights.is_file() or _sha256_file(weights) != dino.WEIGHTS_SHA256:
        raise ProviderAdapterError("locked Grounding DINO checkpoint is unavailable or changed")
    return executable, model_dir


def _live_episode(
    *,
    episode_id: str,
    goal_name: str,
    image_path: Path,
    frame_id: str,
    captured_at_ms: int,
    proposals: list[Mapping[str, Any]],
    width: int,
    height: int,
) -> dict[str, Any]:
    candidates = []
    for rank, proposal in enumerate(proposals, start=1):
        box = [float(value) for value in proposal["bbox_xyxy"]]
        candidates.append(
            {
                "candidate_id": f"gdino-{frame_id}-{rank:03d}",
                "region": {
                    "frame_id": frame_id,
                    "coordinate_space": "NORMALIZED_XYXY",
                    "x_min": max(0.0, min(1.0, box[0] / width)),
                    "y_min": max(0.0, min(1.0, box[1] / height)),
                    "x_max": max(0.0, min(1.0, box[2] / width)),
                    "y_max": max(0.0, min(1.0, box[3] / height)),
                },
                "category_label": str(proposal["label"]),
                "proposal_score": float(proposal["score"]),
                "provider_rank": rank,
            }
        )
    return {
        "episode_id": episode_id,
        "goal_text": f"Find the clear, relatively unique entrance of {goal_name}.",
        "image_path": str(image_path),
        "candidates": candidates,
        "evaluator_episode": {
            "goal_spec": {
                "goal_type": "NAMED_BUILDING_ENTRANCE",
                "target_name": goal_name,
                "requested_relation": "entrance_of",
            },
            "observation_window": {
                "frame_ids": [frame_id],
                "start_timestamp_ms": captured_at_ms,
                "end_timestamp_ms": captured_at_ms,
            },
        },
    }


def ground_current_frame(
    *,
    provider_lock: Mapping[str, Any],
    call_dir: Path,
    episode_id: str,
    goal_name: str,
    image_path: Path,
    frame_id: str,
    observation_id: str,
    captured_at_ms: int,
) -> dict[str, Any]:
    """Run the frozen providers once and return a current-frame observation."""

    executable, model_dir = verify_provider_lock(provider_lock)
    image = image_path.resolve()
    if not image.is_file():
        raise ProviderAdapterError("current frame image is missing")
    output_dir = call_dir.resolve()
    if output_dir.exists():
        raise ProviderAdapterError("provider call directory already exists; refusing replay")

    with Image.open(image) as opened:
        width, height = opened.size
        opened.verify()
    metadata = [
        {
            "id": frame_id,
            "path": str(image),
            "width": width,
            "height": height,
            "image_sha256": _sha256_file(image),
        }
    ]
    # The imported function verifies the exact checkpoint and uses the frozen
    # prompt, thresholds, NMS, determinism settings, and device selection.
    try:
        inference, runtime_versions = dino.run_inference(model_dir, metadata)
    except dino.RunError as error:
        raise ProviderAdapterError(f"frozen Grounding DINO failed: {error}") from error
    episode = _live_episode(
        episode_id=episode_id,
        goal_name=goal_name,
        image_path=image,
        frame_id=frame_id,
        captured_at_ms=captured_at_ms,
        proposals=inference[0]["proposals"],
        width=width,
        height=height,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    rendered = output_dir / "brain-input.jpg"
    schema_path = output_dir / "brain-output-schema.json"
    brain._render_input(episode, "case-current", rendered)
    materializer.write_json(schema_path, brain._schema(DECISION_POLICY_ID))
    prompt = brain._prompt([("case-current", episode)], DECISION_POLICY_ID)
    (output_dir / "brain-prompt.txt").write_text(prompt, encoding="utf-8")

    validated: dict[str, Any] | None = None
    errors: list[str] = []
    successful_attempt: int | None = None
    for attempt in range(1, 3):
        raw_path = output_dir / f"attempt-{attempt}-last-message.json"
        command = [
            str(executable),
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-rules",
            "--json",
            "--color",
            "never",
            "--sandbox",
            "read-only",
            "--model",
            CODEX_MODEL,
            "-c",
            f'model_reasoning_effort="{CODEX_REASONING_EFFORT}"',
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(raw_path),
            "--image",
            str(rendered),
            "--",
            prompt,
        ]
        _atomic_json(
            output_dir / f"attempt-{attempt}-dispatch.json",
            {
                "schema_version": 1,
                "status": "DISPATCH_STARTED",
                "attempt": attempt,
                "episode_id": episode_id,
                "observation_id": observation_id,
                "frame_id": frame_id,
                "image_sha256": metadata[0]["image_sha256"],
            },
        )
        try:
            result = subprocess.run(
                command,
                cwd=output_dir,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
            )
        except subprocess.TimeoutExpired as error:
            _atomic_json(
                output_dir / "completion.json",
                {"schema_version": 1, "status": "IN_DOUBT", "attempt": attempt, "reason": "TIMEOUT"},
            )
            raise ProviderAdapterError("frozen Codex Brain call timed out and is in_doubt") from error
        (output_dir / f"attempt-{attempt}-stdout.jsonl").write_text(result.stdout, encoding="utf-8")
        (output_dir / f"attempt-{attempt}-stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0 or not raw_path.is_file():
            errors.append(f"attempt {attempt}: Codex exit {result.returncode}")
            continue
        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            validated = brain._validate_raw(raw, [("case-current", episode)], DECISION_POLICY_ID)[0]
            successful_attempt = attempt
            break
        except (json.JSONDecodeError, brain.BrainRunError) as error:
            errors.append(f"attempt {attempt}: {error}")
    if validated is None:
        _atomic_json(
            output_dir / "completion.json",
            {"schema_version": 1, "status": "RUN_FAILED", "errors": errors},
        )
        raise ProviderAdapterError(f"frozen Codex Brain failed twice: {errors}")

    p0_output = brain._frozen_output(episode, validated, DECISION_POLICY_ID)
    observation = {
        "schema_version": 1,
        "episode_id": episode_id,
        "observation_id": observation_id,
        "frame_id": frame_id,
        "frame_sha256": metadata[0]["image_sha256"],
        "captured_at_ms": captured_at_ms,
        "processed_at_ms": time.time_ns() // 1_000_000,
        "p0_output": p0_output,
    }
    _atomic_json(output_dir / "observation.json", observation)
    _atomic_json(
        output_dir / "completion.json",
        {
            "schema_version": 1,
            "status": "RUN_SUCCESS",
            "successful_attempt": successful_attempt,
            "p0_status": p0_output["decision"]["status"],
            "proposal_count": len(episode["candidates"]),
            "runtime_versions": runtime_versions,
            "observation_sha256": materializer.content_sha256(observation),
            "claim_ceiling": "MECHANICAL_PROVIDER_REUSE_ONLY_NO_SCIENTIFIC_CONFIRMATION",
        },
    )
    return observation
