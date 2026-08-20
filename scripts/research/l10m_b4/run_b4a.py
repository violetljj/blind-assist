"""Freeze and execute the B4-A harder-cohort paired search comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from scripts.research.l10m_b1.policy_space import (
    INITIAL_SPEC,
    PolicySpec,
    canonical_spec,
    changed_components,
    parse_structured,
    render_structured,
)
from scripts.research.l10m_b1.provider_transport import (
    DEFAULT_AUTH,
    DEFAULT_DOCKER,
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_PROXY_BIND,
    docker_isolation_canary,
    provider_preflight_docker,
    run_provider_docker,
)
from scripts.research.l10m_b1.run_search import (
    _blind_violation,
    _prompt as b1_prompt,
    _provider_runtime_failure,
    _validate_transport_qualification,
)
from scripts.research.l10m_b3a.exploration import (
    BALANCED_EXPLORATION_INSTRUCTION,
    admit_balanced_proposal,
    render_move_ledger,
)

from .hard_benchmark import evaluate_instance, load_benchmark
from .protocol_b4a import (
    ARMS,
    GENERATIONS_PER_TRAJECTORY,
    PAIRED_IDENTITIES,
    PROTOCOL_ID,
    TRANSPORT_QUALIFICATION_RUN_ID,
    TRANSPORT_QUALIFICATION_SHA256,
    build_protocol_manifest,
    canonical_manifest_sha256,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_create_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _balanced_prompt(
    identity: int,
    generation: int,
    incumbent: PolicySpec,
    best_result: dict[str, object],
    ledger: list[dict[str, object]],
) -> str:
    base = b1_prompt(
        "structured",
        identity,
        generation,
        incumbent,
        best_result,
        float(best_result["behavioral_score"]),
    )
    suffix = "\nReturn exactly one replacement candidate."
    if not base.endswith(suffix):
        raise RuntimeError("frozen B1 prompt suffix changed")
    return (
        base[: -len(suffix)]
        + "\n\n"
        + BALANCED_EXPLORATION_INSTRUCTION
        + "\nAttempted canonical move ledger for this arm and paired identity:\n"
        + render_move_ledger(ledger)
        + suffix
    )


def _validate_protocol(path: Path, repo_root: Path) -> dict[str, Any]:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen != build_protocol_manifest(repo_root):
        raise RuntimeError("protocol receipt does not match frozen implementation")
    return frozen


def _validate_transport(path: Path) -> dict[str, Any]:
    qualification = _validate_transport_qualification(path)
    if qualification["sha256"] != TRANSPORT_QUALIFICATION_SHA256 or qualification["run_id"] != TRANSPORT_QUALIFICATION_RUN_ID:
        raise RuntimeError("transport qualification differs from frozen B4-A protocol")
    return qualification


def _seal_not_evaluable(run_dir: Path, reason: str) -> None:
    manifest_path = run_dir / "execution_manifest.json"
    if not manifest_path.exists():
        return
    events_path = run_dir / "events.jsonl"
    events = [] if not events_path.exists() else [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    completed_ids = {row["request_id"] for row in events if row.get("kind") == "completion"}
    for dispatch in [row for row in events if row.get("kind") == "dispatch"]:
        if dispatch["request_id"] in completed_ids:
            continue
        _append_jsonl(
            events_path,
            {
                "kind": "completion",
                "protocol_id": PROTOCOL_ID,
                "request_id": dispatch["request_id"],
                "instance_id": dispatch["instance_id"],
                "paired_identity": dispatch["paired_identity"],
                "replicate": dispatch["replicate"],
                "arm": dispatch["arm"],
                "generation": dispatch["generation"],
                "completed_at": _utc(),
                "returncode": None,
                "model_output": "",
                "candidate_output": "",
                "semantic_valid": False,
                "unsafe_candidate": False,
                "semantic_error": "in_doubt dispatch sealed; no resume or replacement authorized",
                "behavioral_score": None,
                "behavioral_vector": {},
                "transport_runtime_failure": True,
                "in_doubt": True,
            },
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "NOT_EVALUABLE",
            "terminal": "B4A_V2_NOT_EVALUABLE_RUNTIME",
            "scientific_verdict": "NO_SCIENTIFIC_VERDICT",
            "reason": reason,
            "completed_at": _utc(),
            "resume_authorized": False,
        }
    )
    _atomic_json(manifest_path, manifest)
    _write_create_once(
        run_dir / "attempt_closeout.json",
        {
            "run_id": run_dir.name,
            "terminal": "B4A_V2_NOT_EVALUABLE_RUNTIME",
            "scientific_verdict": "NO_SCIENTIFIC_VERDICT",
            "reason": reason,
            "resume_authorized": False,
            "sealed_at": _utc(),
        },
    )
    _atomic_json(run_dir / "progress.json", {"run_id": run_dir.name, "status": "not_evaluable", "eta": "unknown", "last_activity": _utc()})


def _run_trajectory(
    *,
    run_dir: Path,
    events_path: Path,
    progress_path: Path,
    instance: dict[str, Any],
    replicate: int,
    identity: int,
    arm: str,
    docker: Path,
    docker_image: str,
    auth_path: Path,
    proxy_bind: str,
    timeout_seconds: int,
    total_calls: int,
    protocol_id: str = PROTOCOL_ID,
    generations_per_trajectory: int = GENERATIONS_PER_TRAJECTORY,
    evaluate_fn: Callable[[PolicySpec, dict[str, Any]], dict[str, object]] = evaluate_instance,
) -> None:
    instance_id = str(instance["instance_id"])
    incumbent = INITIAL_SPEC
    best_result = evaluate_fn(INITIAL_SPEC, instance)
    attempted_moves: set[str] = set()
    move_ledger: list[dict[str, object]] = []
    for generation in range(1, generations_per_trajectory + 1):
        if arm == "structured_control":
            prompt = b1_prompt("structured", identity, generation, incumbent, best_result, float(best_result["behavioral_score"]))
        else:
            prompt = _balanced_prompt(identity, generation, incumbent, best_result, move_ledger)
        request_id = str(uuid.uuid4())
        started = _utc()
        common = {
            "protocol_id": protocol_id,
            "request_id": request_id,
            "instance_id": instance_id,
            "replicate": replicate,
            "paired_identity": identity,
            "arm": arm,
            "generation": generation,
        }
        _append_jsonl(
            events_path,
            {
                "kind": "dispatch",
                **common,
                "prompt_sha256": _sha256_bytes(prompt.encode()),
                "incumbent_canonical_sha256": _sha256_bytes(canonical_spec(incumbent).encode()),
                "attempted_moves_before": sorted(attempted_moves),
                "started_at": started,
                "status": "in_doubt",
            },
        )
        completed = sum('"kind": "completion"' in line for line in events_path.read_text(encoding="utf-8").splitlines())
        _atomic_json(
            progress_path,
            {
                "run_id": run_dir.name,
                "status": "running",
                "instance_id": instance_id,
                "replicate": replicate,
                "paired_identity": identity,
                "arm": arm,
                "generation": generation,
                "completed": completed,
                "total": total_calls,
                "last_activity": started,
                "eta": "unknown",
            },
        )
        workdir = run_dir / "workers" / instance_id / f"identity-{identity}" / arm / f"generation-{generation}"
        workdir.mkdir(parents=True, exist_ok=False)
        try:
            model_output, returncode, diagnostics = run_provider_docker(
                docker, docker_image, auth_path, prompt, workdir, timeout_seconds, proxy_bind, "proxy"
            )
        except subprocess.TimeoutExpired as error:
            _append_jsonl(
                events_path,
                {
                    "kind": "completion",
                    **common,
                    "completed_at": _utc(),
                    "returncode": None,
                    "model_output": "",
                    "candidate_output": "",
                    "semantic_valid": False,
                    "unsafe_candidate": False,
                    "semantic_error": f"TimeoutExpired: {error}",
                    "behavioral_score": None,
                    "behavioral_vector": {},
                    "transport_runtime_failure": True,
                },
            )
            raise RuntimeError(f"provider timeout at {instance_id}/{identity}/{arm}/{generation}") from error
        model_output = model_output.strip()
        if _blind_violation(diagnostics, "docker") is not None:
            raise RuntimeError(f"blind isolation violation at {instance_id}/{identity}/{arm}/{generation}")
        if _provider_runtime_failure(returncode, model_output):
            _append_jsonl(
                events_path,
                {
                    "kind": "completion",
                    **common,
                    "completed_at": _utc(),
                    "returncode": returncode,
                    "model_output": model_output,
                    "candidate_output": "",
                    "semantic_valid": False,
                    "unsafe_candidate": False,
                    "semantic_error": "ProviderRuntimeFailure",
                    "behavioral_score": None,
                    "behavioral_vector": {},
                    "transport_runtime_failure": True,
                    "diagnostics_tail": diagnostics,
                },
            )
            raise RuntimeError(f"provider runtime failure at {instance_id}/{identity}/{arm}/{generation}")
        admitted: PolicySpec | None = None
        model_proposal: PolicySpec | None = None
        move_token: str | None = None
        disposition = "CONTROL_MODEL_PROPOSAL"
        semantic_error = None
        try:
            model_proposal = parse_structured(model_output)
            if arm == "structured_balanced":
                admitted, move_token, disposition = admit_balanced_proposal(
                    incumbent, model_proposal, attempted_moves, seed=identity, generation=generation
                )
            else:
                admitted = model_proposal
            result = evaluate_fn(admitted, instance)
            valid = bool(result["semantic_valid"])
            unsafe = bool(result["unsafe_candidate"])
            score = float(result["behavioral_score"])
        except Exception as error:
            result = {"behavioral_vector": {}, "invariant_counts": {}}
            valid = False
            unsafe = False
            score = None
            semantic_error = f"{type(error).__name__}: {error}"
        previous_score = float(best_result["behavioral_score"])
        strict = bool(valid and not unsafe and score is not None and score > previous_score)
        admitted_output = "" if admitted is None else render_structured(admitted).strip()
        if move_token is not None:
            attempted_moves.add(move_token)
            move_ledger.append({"generation": generation, "move_token": move_token, "strict_improvement": strict})
        _append_jsonl(
            events_path,
            {
                "kind": "completion",
                **common,
                "completed_at": _utc(),
                "returncode": returncode,
                "model_output": model_output,
                "model_output_sha256": _sha256_bytes(model_output.encode()),
                "model_proposal_canonical": None if model_proposal is None else canonical_spec(model_proposal),
                "candidate_output": admitted_output,
                "candidate_output_sha256": _sha256_bytes(admitted_output.encode()),
                "operator_move_token": move_token,
                "operator_disposition": disposition,
                "attempted_moves_after": sorted(attempted_moves),
                "semantic_valid": valid,
                "unsafe_candidate": unsafe,
                "semantic_error": semantic_error,
                "behavioral_score": score,
                "behavioral_vector": result.get("behavioral_vector", {}),
                "invariant_counts": result.get("invariant_counts", {}),
                "changed_components": [] if admitted is None else changed_components(INITIAL_SPEC, admitted),
                "strict_improvement": strict,
                "diagnostics_tail": diagnostics,
            },
        )
        if strict:
            assert admitted is not None
            incumbent = admitted
            best_result = result


def run(
    *,
    repo_root: Path,
    output_root: Path,
    protocol_path: Path,
    transport_path: Path,
    docker: Path,
    docker_image: str,
    auth_path: Path,
    proxy_bind: str,
    timeout_seconds: int,
) -> Path:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    protocol_path = protocol_path.resolve()
    transport_path = transport_path.resolve()
    protocol = _validate_protocol(protocol_path, repo_root)
    qualification = _validate_transport(transport_path)
    provider = provider_preflight_docker(docker, docker_image, auth_path, proxy_bind)
    isolation = docker_isolation_canary(docker, docker_image, auth_path, output_root / "preflight")
    if isolation.get("status") != "PASS":
        raise RuntimeError("Docker isolation canary did not pass")
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"b4av2-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir(exist_ok=False)
    events_path = run_dir / "events.jsonl"
    progress_path = run_dir / "progress.json"
    total = len(PAIRED_IDENTITIES) * len(ARMS) * GENERATIONS_PER_TRAJECTORY
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _sha256(protocol_path),
        "protocol_manifest_sha256": canonical_manifest_sha256(protocol),
        "run_id": run_id,
        "status": "RUNNING",
        "started_at": _utc(),
        "planned_model_calls": total,
        "paired_identities": list(PAIRED_IDENTITIES),
        "arms": list(ARMS),
        "generations_per_trajectory": GENERATIONS_PER_TRAJECTORY,
        "provider": provider,
        "isolation": isolation,
        "transport_qualification": qualification,
        "resume_authorized": False,
        "worker_path_mode": "resolved_absolute_windows_path",
    }
    _write_create_once(run_dir / "execution_manifest.json", manifest)
    benchmark = {row["instance_id"]: row for row in load_benchmark()["instances"]}
    try:
        for pair_index, pair in enumerate(PAIRED_IDENTITIES):
            instance_id = str(pair["instance_id"])
            identity = int(pair["paired_identity"])
            replicate = int(pair["replicate"])
            order = ARMS if pair_index % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                _run_trajectory(
                    run_dir=run_dir,
                    events_path=events_path,
                    progress_path=progress_path,
                    instance=benchmark[instance_id],
                    replicate=replicate,
                    identity=identity,
                    arm=arm,
                    docker=docker,
                    docker_image=docker_image,
                    auth_path=auth_path,
                    proxy_bind=proxy_bind,
                    timeout_seconds=timeout_seconds,
                    total_calls=total,
                )
    except Exception as error:
        _seal_not_evaluable(run_dir, f"{type(error).__name__}: {error}")
        raise
    completion_count = sum('"kind": "completion"' in line for line in events_path.read_text(encoding="utf-8").splitlines())
    if completion_count != total:
        _seal_not_evaluable(run_dir, f"completion count {completion_count} differs from {total}")
        raise RuntimeError("formal completion count mismatch")
    manifest.update(
        {
            "status": "COMPLETE",
            "terminal": "B4A_V2_EXECUTION_COMPLETE",
            "completion_count": completion_count,
            "events_sha256": _sha256(events_path),
            "completed_at": _utc(),
        }
    )
    _atomic_json(run_dir / "execution_manifest.json", manifest)
    _atomic_json(progress_path, {"run_id": run_id, "status": "complete", "completed": total, "total": total, "last_activity": _utc(), "eta": "complete"})
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--repo-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--repo-root", type=Path, required=True)
    execute.add_argument("--output-root", type=Path, required=True)
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--transport-qualification", type=Path, required=True)
    execute.add_argument("--docker", type=Path, default=DEFAULT_DOCKER)
    execute.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    execute.add_argument("--auth-path", type=Path, default=DEFAULT_AUTH)
    execute.add_argument("--proxy-bind", default=DEFAULT_PROXY_BIND)
    execute.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.command == "freeze":
        _write_create_once(args.output, build_protocol_manifest(args.repo_root))
        print(json.dumps({"output": str(args.output), "status": "B4_A_V2_PROTOCOL_FROZEN_EXECUTION_NOT_STARTED"}))
        return
    run_dir = run(
        repo_root=args.repo_root,
        output_root=args.output_root,
        protocol_path=args.protocol,
        transport_path=args.transport_qualification,
        docker=args.docker,
        docker_image=args.docker_image,
        auth_path=args.auth_path,
        proxy_bind=args.proxy_bind,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({"run_dir": str(run_dir), "terminal": "B4A_V2_EXECUTION_COMPLETE"}))


if __name__ == "__main__":
    main()
