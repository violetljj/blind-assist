"""Qualify the exact B1 provider transport without touching scientific state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .provider_transport import (
    DEFAULT_AUTH,
    DEFAULT_DOCKER,
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_PROXY_BIND,
    MODEL,
    docker_isolation_canary,
    provider_preflight_docker,
    run_provider_docker,
)


PROTOCOL_ID = "L10M-B1-I0-TRANSPORT-QUALIFICATION-V2-FORMAL-SHAPED"
ATTEMPT_COUNT = 10
PAYLOAD_ITEMS = 32
PAYLOAD_VALUE = "B1-I0-TRANSPORT-CANARY"
PASS_TERMINAL = "B1_TRANSPORT_QUALIFIED"
FAIL_TERMINAL = "B1_TRANSPORT_NOT_QUALIFIED"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_protocol_manifest(route: str) -> dict[str, object]:
    if route not in {"direct", "proxy"}:
        raise ValueError(f"unknown transport route: {route}")
    return {
        "protocol_id": PROTOCOL_ID,
        "role": "infrastructure_only_no_scientific_authority",
        "route": route,
        "request_count": ATTEMPT_COUNT,
        "execution": {
            "backend": "docker",
            "model": MODEL,
            "streaming_mode": "codex_responses_stream_with_responses_websocket_false",
            "provider_invocation": "exact shared _run_provider_docker path",
            "sequential_requests": True,
            "application_retries": 0,
        },
        "fixed_canary": {
            "research_content": False,
            "prompt_shape": "formal_b1_structured_candidate_and_feedback_envelope",
            "terminal_format": "strict JSON object",
            "payload_items": PAYLOAD_ITEMS,
            "payload_value": PAYLOAD_VALUE,
        },
        "pass_gate": {
            "successful_terminal_responses": ATTEMPT_COUNT,
            "nonempty_responses": ATTEMPT_COUNT,
            "parseable_schema_valid_responses": ATTEMPT_COUNT,
            "nonzero_provider_exit_count": 0,
            "websocket_reconnect_exhaustion_count": 0,
            "decode_failure_count": 0,
            "container_isolation_guard": "PASS",
        },
        "authority": {
            "scientific_instances_or_seeds_consumed": 0,
            "candidate_generation": False,
            "evaluator_access": False,
            "hidden_feedback_access": False,
            "direct_route_pass_only_diagnoses_transport_unless_direct_is_versioned_as_formal_b1_route": True,
            "only_the_route_bound_into_the_formal_b1_execution_protocol_can_authorize_b1": True,
        },
        "terminals": {
            "pass": PASS_TERMINAL,
            "fail": FAIL_TERMINAL,
            "failure_effect": "B1_REMAINS_UNAUTHORIZED",
        },
    }


def _prompt(attempt: int) -> str:
    return (
        "You are improving one transport-only canary candidate. You may change only "
        "the five displayed fields, and this has no research meaning. Do not use tools, "
        "inspect files, or discuss research. Return exactly one JSON object and nothing else.\n"
        "Allowed fields: action_selection.turn_threshold (0.10, 0.20, 0.30), "
        "fallback.min_quality (0.35, 0.50, 0.65), fallback.action (STOP, LEFT, RIGHT), "
        "stuck_response.on_confirmed_stuck (ENTER_RECOVERY, STOP), and "
        "recovery_transition.while_active (RECOVER, LEFT, RIGHT).\n"
        f"Paired seed: 0. Generation: {attempt}. Interface: component-grouped JSON.\n"
        "Current candidate:\n"
        '{"action_selection":{"turn_threshold":0.2},"fallback":{"action":"STOP","min_quality":0.35},'
        '"progress_contract":{"mode":"POSITIVE_PROGRESS|CONFIRMED_NO_PROGRESS|UNKNOWN_PROGRESS","mutable":false},'
        '"recovery_transition":{"while_active":"RECOVER"},"stuck_response":{"on_confirmed_stuck":"ENTER_RECOVERY"}}\n'
        "Latest evaluator feedback (canary-only, not scientific): "
        '{"behavioral_score":0.5,"behavioral_vector":{"action_agreement_rate":0.5,"arrival_success_rate":0.5,"false_arrival_rate":0.0,"oscillation_rate":0.0,"unsafe_action_rate":0.0},"best_score_so_far":0.5,"candidate_valid":true,"generation":0,"unsafe_candidate":false}\n'
        "Return the same displayed candidate with the canary marker and payload. The object must be "
        f'{{"canary_id":"B1-I0-V2","attempt":{attempt},"candidate":{{"action_selection":{{"turn_threshold":0.2}},"fallback":{{"action":"STOP","min_quality":0.35}},"progress_contract":{{"mode":"POSITIVE_PROGRESS|CONFIRMED_NO_PROGRESS|UNKNOWN_PROGRESS","mutable":false}},"recovery_transition":{{"while_active":"RECOVER"}},"stuck_response":{{"on_confirmed_stuck":"ENTER_RECOVERY"}}}},"payload":['
        + ",".join(f'"{PAYLOAD_VALUE}"' for _ in range(PAYLOAD_ITEMS))
        + "]}."
    )


def _validate_response(output: str, attempt: int) -> tuple[bool, str | None]:
    if not output.strip():
        return False, "empty_terminal_response"
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        return False, f"terminal_json_decode_failure:{exc.msg}"
    expected = {
        "canary_id": "B1-I0-V2",
        "attempt": attempt,
        "candidate": {
            "action_selection": {"turn_threshold": 0.2},
            "fallback": {"action": "STOP", "min_quality": 0.35},
            "progress_contract": {
                "mode": "POSITIVE_PROGRESS|CONFIRMED_NO_PROGRESS|UNKNOWN_PROGRESS",
                "mutable": False,
            },
            "recovery_transition": {"while_active": "RECOVER"},
            "stuck_response": {"on_confirmed_stuck": "ENTER_RECOVERY"},
        },
        "payload": [PAYLOAD_VALUE] * PAYLOAD_ITEMS,
    }
    if parsed != expected:
        return False, "terminal_schema_or_payload_mismatch"
    return True, None


def _diagnostic_flags(diagnostics: str) -> dict[str, bool]:
    lowered = diagnostics.lower()
    reconnect_exhaustion = any(
        marker in lowered
        for marker in (
            "reconnect attempts exhausted",
            "failed to reconnect",
            "reconnect exhausted",
        )
    )
    decode_failure = any(
        marker in lowered
        for marker in (
            "unicodedecodeerror",
            "decode failure",
            "decode error",
            "error decoding",
        )
    )
    return {
        "websocket_reconnect_exhaustion": reconnect_exhaustion,
        "decode_failure": decode_failure,
    }


def run_qualification(
    output_root: Path,
    *,
    route: str,
    docker: Path = DEFAULT_DOCKER,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    auth_path: Path = DEFAULT_AUTH,
    proxy_bind: str = DEFAULT_PROXY_BIND,
    timeout_seconds: int = 300,
) -> Path:
    protocol = build_protocol_manifest(route)
    provider = provider_preflight_docker(
        docker, docker_image, auth_path, proxy_bind
    )
    with tempfile.TemporaryDirectory(prefix="b1-i0-preflight-") as temporary:
        isolation = docker_isolation_canary(
            docker, docker_image, auth_path, Path(temporary)
        )

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"b1-i0-{route}-{datetime.now().strftime('%Y%m%dT%H%M%S')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(exist_ok=False)
    events_path = run_dir / "events.jsonl"
    manifest = {
        "protocol": protocol,
        "run_id": run_id,
        "status": "RUNNING",
        "started_at": _utc(),
        "provider": {**provider, "transport_route": route},
        "container_isolation_guard": isolation,
        "timeout_seconds": timeout_seconds,
    }
    _atomic_json(run_dir / "execution_manifest.json", manifest)

    completions: list[dict[str, object]] = []
    for attempt in range(1, ATTEMPT_COUNT + 1):
        request_id = str(uuid.uuid4())
        prompt = _prompt(attempt)
        _append_jsonl(
            events_path,
            {
                "kind": "dispatch",
                "infrastructure_attempt_id": f"{run_id}-{attempt:02d}",
                "request_id": request_id,
                "attempt": attempt,
                "dispatched_at": _utc(),
                "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            },
        )
        attempt_dir = run_dir / "attempts" / f"attempt-{attempt:02d}" / "workspace"
        attempt_dir.mkdir(parents=True)
        started = datetime.now(timezone.utc)
        try:
            output, returncode, diagnostics = run_provider_docker(
                docker,
                docker_image,
                auth_path,
                prompt,
                attempt_dir,
                timeout_seconds,
                proxy_bind,
                route,
            )
            schema_valid, response_error = _validate_response(output, attempt)
            flags = _diagnostic_flags(diagnostics)
        except subprocess.TimeoutExpired as exc:
            output = ""
            returncode = None
            diagnostics = str(exc)
            schema_valid = False
            response_error = "provider_timeout"
            flags = _diagnostic_flags(diagnostics)
        except Exception as exc:  # preserve an infrastructure receipt, never retry
            output = ""
            returncode = None
            diagnostics = f"{type(exc).__name__}: {exc}"
            schema_valid = False
            response_error = "provider_exception"
            flags = _diagnostic_flags(diagnostics)
        finished = datetime.now(timezone.utc)
        success = (
            returncode == 0
            and schema_valid
            and not flags["websocket_reconnect_exhaustion"]
            and not flags["decode_failure"]
        )
        completion = {
            "kind": "completion",
            "infrastructure_attempt_id": f"{run_id}-{attempt:02d}",
            "request_id": request_id,
            "attempt": attempt,
            "completed_at": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "returncode": returncode,
            "terminal_response_nonempty": bool(output.strip()),
            "terminal_response_bytes": len(output.encode("utf-8")),
            "terminal_response_sha256": _sha256_bytes(output.encode("utf-8")),
            "terminal_response_schema_valid": schema_valid,
            "response_error": response_error,
            **flags,
            "diagnostics_sha256": _sha256_bytes(diagnostics.encode("utf-8")),
            "diagnostics_tail": diagnostics[-4000:],
            "success": success,
        }
        _append_jsonl(events_path, completion)
        completions.append(completion)

    success_count = sum(bool(item["success"]) for item in completions)
    reconnect_exhaustion_count = sum(
        bool(item["websocket_reconnect_exhaustion"]) for item in completions
    )
    decode_failure_count = sum(bool(item["decode_failure"]) for item in completions)
    passed = (
        isolation.get("status") == "PASS"
        and success_count == ATTEMPT_COUNT
        and reconnect_exhaustion_count == 0
        and decode_failure_count == 0
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "route": route,
        "terminal": PASS_TERMINAL if passed else FAIL_TERMINAL,
        "b1_execution_authorized": passed and route == "proxy",
        "scientific_instances_or_seeds_consumed": 0,
        "scientific_verdict": "NO_SCIENTIFIC_VERDICT",
        "request_count": ATTEMPT_COUNT,
        "success_count": success_count,
        "failure_count": ATTEMPT_COUNT - success_count,
        "websocket_reconnect_exhaustion_count": reconnect_exhaustion_count,
        "decode_failure_count": decode_failure_count,
        "container_isolation_guard": isolation.get("status"),
        "completed_at": _utc(),
    }
    _atomic_json(run_dir / "result.json", result)
    manifest["status"] = "COMPLETE"
    manifest["completed_at"] = result["completed_at"]
    manifest["result_sha256"] = _sha256_bytes(
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    _atomic_json(run_dir / "execution_manifest.json", manifest)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts.local/evidence/l10m_b1/transport_qualification"),
    )
    parser.add_argument("--route", choices=("direct", "proxy"), required=True)
    parser.add_argument("--docker", type=Path, default=DEFAULT_DOCKER)
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--auth-path", type=Path, default=DEFAULT_AUTH)
    parser.add_argument("--proxy-bind", default=DEFAULT_PROXY_BIND)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    run_dir = run_qualification(
        args.output_root,
        route=args.route,
        docker=args.docker,
        docker_image=args.docker_image,
        auth_path=args.auth_path,
        proxy_bind=args.proxy_bind,
        timeout_seconds=args.timeout_seconds,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
