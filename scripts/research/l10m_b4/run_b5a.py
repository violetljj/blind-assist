"""Freeze and execute the B5-A fresh generalization replication."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.research.l10m_b1.provider_transport import (
    DEFAULT_AUTH,
    DEFAULT_DOCKER,
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_PROXY_BIND,
    docker_isolation_canary,
    provider_preflight_docker,
)
from scripts.research.l10m_b1.run_search import _validate_transport_qualification

from .fresh_benchmark import evaluate_fresh_instance, load_fresh_benchmark
from .protocol_b5a import (
    ARMS,
    GENERATIONS_PER_TRAJECTORY,
    PAIRED_IDENTITIES,
    PROTOCOL_ID,
    TRANSPORT_QUALIFICATION_RUN_ID,
    TRANSPORT_QUALIFICATION_SHA256,
    build_protocol_manifest,
    canonical_manifest_sha256,
)
from .run_b4a import (
    _append_jsonl,
    _atomic_json,
    _run_trajectory,
    _sha256,
    _utc,
    _write_create_once,
)


def _validate_protocol(path: Path, repo_root: Path) -> dict[str, Any]:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen != build_protocol_manifest(repo_root):
        raise RuntimeError("protocol receipt does not match frozen B5-A implementation")
    return frozen


def _validate_transport(path: Path) -> dict[str, Any]:
    qualification = _validate_transport_qualification(path)
    if (
        qualification["sha256"] != TRANSPORT_QUALIFICATION_SHA256
        or qualification["run_id"] != TRANSPORT_QUALIFICATION_RUN_ID
    ):
        raise RuntimeError("transport qualification differs from frozen B5-A protocol")
    return qualification


def _seal_not_evaluable(run_dir: Path, reason: str) -> None:
    manifest_path = run_dir / "execution_manifest.json"
    if not manifest_path.exists():
        return
    events_path = run_dir / "events.jsonl"
    events = [] if not events_path.exists() else [
        json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
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
                "semantic_error": "in_doubt dispatch sealed; no retry, resume, or replacement authorized",
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
            "terminal": "B5A_NOT_EVALUABLE_RUNTIME",
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
            "terminal": "B5A_NOT_EVALUABLE_RUNTIME",
            "scientific_verdict": "NO_SCIENTIFIC_VERDICT",
            "reason": reason,
            "resume_authorized": False,
            "sealed_at": _utc(),
        },
    )
    _atomic_json(
        run_dir / "progress.json",
        {"run_id": run_dir.name, "status": "not_evaluable", "eta": "unknown", "last_activity": _utc()},
    )


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
    run_id = f"b5a-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
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
    benchmark = {row["instance_id"]: row for row in load_fresh_benchmark()["instances"]}
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
                    protocol_id=PROTOCOL_ID,
                    generations_per_trajectory=GENERATIONS_PER_TRAJECTORY,
                    evaluate_fn=evaluate_fresh_instance,
                )
    except Exception as error:
        _seal_not_evaluable(run_dir, f"{type(error).__name__}: {error}")
        raise
    completion_count = sum(
        '"kind": "completion"' in line for line in events_path.read_text(encoding="utf-8").splitlines()
    )
    if completion_count != total:
        _seal_not_evaluable(run_dir, f"completion count {completion_count} differs from {total}")
        raise RuntimeError("formal completion count mismatch")
    manifest.update(
        {
            "status": "COMPLETE",
            "terminal": "B5A_EXECUTION_COMPLETE",
            "completion_count": completion_count,
            "events_sha256": _sha256(events_path),
            "completed_at": _utc(),
        }
    )
    _atomic_json(run_dir / "execution_manifest.json", manifest)
    _atomic_json(
        progress_path,
        {"run_id": run_id, "status": "complete", "completed": total, "total": total, "last_activity": _utc(), "eta": "complete"},
    )
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
        print(json.dumps({"output": str(args.output), "status": "B5_A_PROTOCOL_FROZEN_EXECUTION_NOT_STARTED"}))
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
    print(json.dumps({"run_dir": str(run_dir), "terminal": "B5A_EXECUTION_COMPLETE"}))


if __name__ == "__main__":
    main()
