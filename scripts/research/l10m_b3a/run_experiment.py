"""Run the frozen two-arm B3-A balanced exploration causal test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.research.l10m_b1.evaluator import evaluate_spec
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
from scripts.research.l10m_b3a.protocol import (
    ARMS,
    EVALUATIONS_PER_ARM_PER_SEED,
    GENERATIONS_PER_ARM_PER_SEED,
    INITIAL_SCORE,
    MODEL,
    PAIRED_SEEDS,
    PROTOCOL_ID,
    TRANSPORT_QUALIFICATION_RUN_ID,
    TRANSPORT_QUALIFICATION_SHA256,
    build_protocol_manifest,
    canonical_manifest_sha256,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_create_once(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _treatment_prompt(
    seed: int,
    generation: int,
    incumbent: PolicySpec,
    best_result: dict[str, object],
    move_ledger: list[dict[str, object]],
) -> str:
    base = b1_prompt(
        "structured",
        seed,
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
        + "\nAttempted canonical move ledger for this arm and seed:\n"
        + render_move_ledger(move_ledger)
        + suffix
    )


def _validate_protocol(protocol_path: Path, repo_root: Path) -> dict[str, object]:
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = build_protocol_manifest(repo_root)
    if frozen != expected:
        raise RuntimeError("protocol receipt does not match the current frozen implementation")
    return frozen


def _validate_transport(path: Path) -> dict[str, object]:
    qualification = _validate_transport_qualification(path)
    if (
        qualification["sha256"] != TRANSPORT_QUALIFICATION_SHA256
        or qualification["run_id"] != TRANSPORT_QUALIFICATION_RUN_ID
    ):
        raise RuntimeError("transport qualification identity differs from the frozen B3-A protocol")
    return qualification


def _seal_not_evaluable(run_dir: Path, reason: str) -> None:
    manifest_path = run_dir / "execution_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events_path = run_dir / "events.jsonl"
    events = [] if not events_path.exists() else [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    dispatches = [event for event in events if event.get("kind") == "dispatch"]
    completions = [event for event in events if event.get("kind") == "completion"]
    completed_ids = {event["request_id"] for event in completions}
    for dispatch in dispatches:
        if dispatch["request_id"] in completed_ids:
            continue
        completion = {
            "kind": "completion",
            "protocol_id": PROTOCOL_ID,
            "request_id": dispatch["request_id"],
            "seed": dispatch["seed"],
            "arm": dispatch["arm"],
            "generation": dispatch["generation"],
            "completed_at": _utc(),
            "returncode": None,
            "model_output": "",
            "model_output_sha256": _sha256_bytes(b""),
            "candidate_output": "",
            "candidate_output_sha256": _sha256_bytes(b""),
            "semantic_valid": False,
            "unsafe_candidate": False,
            "semantic_error": "in_doubt dispatch sealed; no resume or replacement authorized",
            "behavioral_score": None,
            "behavioral_vector": {},
            "transport_runtime_failure": True,
            "in_doubt": True,
        }
        _append_jsonl(events_path, completion)
        completions.append(completion)
    closeout = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_dir.name,
        "terminal": "B3A_NOT_EVALUABLE_RUNTIME",
        "scientific_verdict": "NO_SCIENTIFIC_VERDICT",
        "reason": reason,
        "dispatch_count": len(dispatches),
        "completion_count": len(completions),
        "resume_authorized": False,
        "sealed_at": _utc(),
    }
    _write_create_once(run_dir / "attempt_closeout.json", closeout)
    manifest.update(
        {
            "status": "NOT_EVALUABLE",
            "terminal": closeout["terminal"],
            "scientific_verdict": closeout["scientific_verdict"],
            "completed_at": closeout["sealed_at"],
            "resume_authorized": False,
        }
    )
    _atomic_json(manifest_path, manifest)
    _atomic_json(run_dir / "progress.json", {"run_id": run_dir.name, "status": "not_evaluable", "completed": len(completions), "last_activity": _utc(), "eta": "unknown"})


def _run_arm(
    *,
    run_dir: Path,
    events_path: Path,
    progress_path: Path,
    seed: int,
    arm: str,
    docker: Path,
    docker_image: str,
    auth_path: Path,
    proxy_bind: str,
    timeout_seconds: int,
) -> None:
    incumbent = INITIAL_SPEC
    best_result = evaluate_spec(INITIAL_SPEC)
    if float(best_result["behavioral_score"]) != INITIAL_SCORE:
        raise RuntimeError("frozen initial evaluator score changed")
    attempted_moves: set[str] = set()
    move_ledger: list[dict[str, object]] = []

    for generation in range(1, GENERATIONS_PER_ARM_PER_SEED + 1):
        if arm == "structured_control":
            prompt = b1_prompt("structured", seed, generation, incumbent, best_result, float(best_result["behavioral_score"]))
        else:
            prompt = _treatment_prompt(seed, generation, incumbent, best_result, move_ledger)
        request_id = str(uuid.uuid4())
        started = _utc()
        _append_jsonl(
            events_path,
            {
                "kind": "dispatch",
                "protocol_id": PROTOCOL_ID,
                "request_id": request_id,
                "seed": seed,
                "arm": arm,
                "generation": generation,
                "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                "incumbent_canonical_sha256": _sha256_bytes(canonical_spec(incumbent).encode("utf-8")),
                "attempted_moves_before": sorted(attempted_moves),
                "started_at": started,
                "status": "in_doubt",
            },
        )
        completed_so_far = sum(1 for line in events_path.read_text(encoding="utf-8").splitlines() if '"kind": "completion"' in line)
        _atomic_json(
            progress_path,
            {
                "run_id": run_dir.name,
                "status": "running",
                "seed": seed,
                "arm": arm,
                "generation": generation,
                "completed": completed_so_far,
                "total": len(PAIRED_SEEDS) * len(ARMS) * GENERATIONS_PER_ARM_PER_SEED,
                "last_activity": started,
                "eta": "unknown",
            },
        )
        workdir = run_dir / "workers" / f"seed-{seed}" / arm / f"generation-{generation}"
        workdir.mkdir(parents=True, exist_ok=False)
        try:
            model_output, returncode, diagnostics = run_provider_docker(
                docker,
                docker_image,
                auth_path,
                prompt,
                workdir,
                timeout_seconds,
                proxy_bind,
                "proxy",
            )
        except subprocess.TimeoutExpired as error:
            completion = {
                "kind": "completion",
                "protocol_id": PROTOCOL_ID,
                "request_id": request_id,
                "seed": seed,
                "arm": arm,
                "generation": generation,
                "completed_at": _utc(),
                "returncode": None,
                "model_output": "",
                "model_output_sha256": _sha256_bytes(b""),
                "candidate_output": "",
                "candidate_output_sha256": _sha256_bytes(b""),
                "semantic_valid": False,
                "unsafe_candidate": False,
                "semantic_error": f"TimeoutExpired: {error}",
                "behavioral_score": None,
                "behavioral_vector": {},
                "transport_runtime_failure": True,
            }
            _append_jsonl(events_path, completion)
            raise RuntimeError(f"provider timeout at seed={seed} arm={arm} generation={generation}") from error

        model_output = model_output.strip()
        if _blind_violation(diagnostics, "docker") is not None:
            raise RuntimeError(f"blind isolation violation at seed={seed} arm={arm} generation={generation}")
        if _provider_runtime_failure(returncode, model_output):
            completion = {
                "kind": "completion",
                "protocol_id": PROTOCOL_ID,
                "request_id": request_id,
                "seed": seed,
                "arm": arm,
                "generation": generation,
                "completed_at": _utc(),
                "returncode": returncode,
                "model_output": model_output,
                "model_output_sha256": _sha256_bytes(model_output.encode("utf-8")),
                "candidate_output": "",
                "candidate_output_sha256": _sha256_bytes(b""),
                "semantic_valid": False,
                "unsafe_candidate": False,
                "semantic_error": "ProviderRuntimeFailure: nonzero exit or empty terminal response",
                "behavioral_score": None,
                "behavioral_vector": {},
                "transport_runtime_failure": True,
                "diagnostics_tail": diagnostics,
            }
            _append_jsonl(events_path, completion)
            raise RuntimeError(f"provider runtime failure at seed={seed} arm={arm} generation={generation}")

        admitted: PolicySpec | None = None
        model_proposal: PolicySpec | None = None
        move_token: str | None = None
        operator_disposition = "CONTROL_MODEL_PROPOSAL"
        semantic_error: str | None = None
        try:
            model_proposal = parse_structured(model_output)
            if arm == "structured_balanced":
                admitted, move_token, operator_disposition = admit_balanced_proposal(
                    incumbent,
                    model_proposal,
                    attempted_moves,
                    seed=seed,
                    generation=generation,
                )
            else:
                admitted = model_proposal
            result = evaluate_spec(admitted)
            valid = bool(result["semantic_valid"])
            unsafe = bool(result["unsafe_candidate"])
            score = float(result["behavioral_score"])
        except Exception as error:  # semantic invalidity is a measured outcome
            result = {"semantic_valid": False, "unsafe_candidate": False, "behavioral_score": None, "behavioral_vector": {}, "invariant_counts": {}}
            valid = False
            unsafe = False
            score = None
            semantic_error = f"{type(error).__name__}: {error}"

        previous_score = float(best_result["behavioral_score"])
        strict_improvement = bool(valid and not unsafe and score is not None and score > previous_score)
        admitted_output = "" if admitted is None else render_structured(admitted).strip()
        if move_token is not None:
            attempted_moves.add(move_token)
            move_ledger.append(
                {
                    "generation": generation,
                    "move_token": move_token,
                    "strict_improvement": strict_improvement,
                }
            )
        completion = {
            "kind": "completion",
            "protocol_id": PROTOCOL_ID,
            "request_id": request_id,
            "seed": seed,
            "arm": arm,
            "generation": generation,
            "completed_at": _utc(),
            "returncode": returncode,
            "model_output": model_output,
            "model_output_sha256": _sha256_bytes(model_output.encode("utf-8")),
            "model_proposal_canonical": None if model_proposal is None else canonical_spec(model_proposal),
            "candidate_output": admitted_output,
            "candidate_output_sha256": _sha256_bytes(admitted_output.encode("utf-8")),
            "admitted_canonical": None if admitted is None else canonical_spec(admitted),
            "operator_move_token": move_token,
            "operator_disposition": operator_disposition,
            "attempted_moves_after": sorted(attempted_moves),
            "semantic_valid": valid,
            "unsafe_candidate": unsafe,
            "semantic_error": semantic_error,
            "behavioral_score": score,
            "behavioral_vector": result.get("behavioral_vector", {}),
            "invariant_counts": result.get("invariant_counts", {}),
            "changed_components": [] if admitted is None else changed_components(INITIAL_SPEC, admitted),
            "strict_improvement": strict_improvement,
            "diagnostics_tail": diagnostics,
        }
        _append_jsonl(events_path, completion)
        if strict_improvement:
            assert admitted is not None
            incumbent = admitted
            best_result = result


def run_experiment(
    *,
    repo_root: Path,
    output_root: Path,
    protocol_path: Path,
    transport_qualification: Path,
    docker: Path,
    docker_image: str,
    auth_path: Path,
    proxy_bind: str,
    timeout_seconds: int,
) -> Path:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    protocol_path = protocol_path.resolve()
    protocol = _validate_protocol(protocol_path, repo_root)
    qualification = _validate_transport(transport_qualification)
    provider = provider_preflight_docker(docker, docker_image, auth_path, proxy_bind)
    isolation_root = output_root / "preflight"
    isolation = docker_isolation_canary(docker, docker_image, auth_path, isolation_root)
    if isolation.get("status") != "PASS":
        raise RuntimeError("Docker isolation canary did not pass")

    output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"b3a-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    events_path = run_dir / "events.jsonl"
    progress_path = run_dir / "progress.json"
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _sha256(protocol_path),
        "protocol_manifest_sha256": canonical_manifest_sha256(protocol),
        "run_id": run_id,
        "status": "RUNNING",
        "started_at": _utc(),
        "model": MODEL,
        "provider": provider,
        "transport_qualification": qualification,
        "isolation": isolation,
        "seeds": list(PAIRED_SEEDS),
        "arms": list(ARMS),
        "generations_per_arm_per_seed": GENERATIONS_PER_ARM_PER_SEED,
        "evaluations_per_arm_per_seed": EVALUATIONS_PER_ARM_PER_SEED,
        "planned_model_calls": len(PAIRED_SEEDS) * len(ARMS) * GENERATIONS_PER_ARM_PER_SEED,
        "retry_semantics": "no retry, replacement, or resume; any runtime failure seals whole cohort",
        "blind_boundary": "provider sees only prompt and read-only empty worker directory; evaluator and evidence remain host-local",
    }
    _atomic_json(run_dir / "execution_manifest.json", manifest)
    _atomic_json(progress_path, {"run_id": run_id, "status": "running", "completed": 0, "total": manifest["planned_model_calls"], "last_activity": _utc(), "eta": "unknown"})
    try:
        for seed_index, seed in enumerate(PAIRED_SEEDS):
            order = list(ARMS) if seed_index % 2 == 0 else list(reversed(ARMS))
            for arm in order:
                _run_arm(
                    run_dir=run_dir,
                    events_path=events_path,
                    progress_path=progress_path,
                    seed=seed,
                    arm=arm,
                    docker=docker,
                    docker_image=docker_image,
                    auth_path=auth_path,
                    proxy_bind=proxy_bind,
                    timeout_seconds=timeout_seconds,
                )
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        completions = [event for event in events if event.get("kind") == "completion"]
        expected = len(PAIRED_SEEDS) * len(ARMS) * EVALUATIONS_PER_ARM_PER_SEED
        if len(completions) != expected:
            raise RuntimeError(f"completion count mismatch: {len(completions)} != {expected}")
        manifest.update({"status": "COMPLETE", "terminal": "B3A_EXECUTION_COMPLETE_PENDING_ANALYSIS", "completed_at": _utc(), "model_calls": len(completions)})
        _atomic_json(run_dir / "execution_manifest.json", manifest)
        _atomic_json(progress_path, {"run_id": run_id, "status": "complete", "completed": expected, "total": expected, "last_activity": _utc(), "eta": "complete"})
        return run_dir
    except BaseException as error:
        _seal_not_evaluable(run_dir, f"{type(error).__name__}: {error}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--repo-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--transport-qualification", type=Path, required=True)
    run.add_argument("--docker", type=Path, default=DEFAULT_DOCKER)
    run.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    run.add_argument("--auth", type=Path, default=DEFAULT_AUTH)
    run.add_argument("--proxy-bind", default=DEFAULT_PROXY_BIND)
    run.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.command == "freeze":
        _write_create_once(args.output, build_protocol_manifest(args.repo_root))
        print(args.output.resolve())
        return
    run_dir = run_experiment(
        repo_root=args.repo_root,
        output_root=args.output_root,
        protocol_path=args.protocol,
        transport_qualification=args.transport_qualification,
        docker=args.docker,
        docker_image=args.docker_image,
        auth_path=args.auth,
        proxy_bind=args.proxy_bind,
        timeout_seconds=args.timeout_seconds,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
