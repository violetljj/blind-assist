"""Execute the frozen, paired L10M-B1 search with conservative journaling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .evaluator import evaluate_spec
from .policy_space import (
    INITIAL_SPEC,
    PolicySpec,
    canonical_spec,
    changed_components,
    parse_raw,
    parse_structured,
    render_raw,
    render_structured,
)
from .protocol import (
    COMMON_SEARCH_INFORMATION,
    EVALUATIONS_PER_ARM_PER_SEED,
    GENERATIONS_PER_ARM_PER_SEED,
    PAIRED_SEEDS,
    PROTOCOL_ID,
    build_protocol_manifest,
)


DEFAULT_CLI = Path(r"E:\codex-tools\bin\codex.exe")
MODEL = "gpt-5.6-sol"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
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


def _load_events(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_output(arm: str, output: str) -> PolicySpec:
    return parse_raw(output) if arm == "raw" else parse_structured(output)


def _render_candidate(arm: str, spec: PolicySpec) -> str:
    return render_raw(spec) if arm == "raw" else render_structured(spec)


def _feedback(result: dict[str, object], *, generation: int, best_score: float) -> str:
    vector = result.get("behavioral_vector", {})
    return json.dumps(
        {
            "generation": generation,
            "candidate_valid": result.get("semantic_valid", False),
            "unsafe_candidate": result.get("unsafe_candidate", True),
            "behavioral_score": result.get("behavioral_score"),
            "behavioral_vector": vector,
            "best_score_so_far": best_score,
        },
        sort_keys=True,
    )


def _prompt(arm: str, seed: int, generation: int, candidate: PolicySpec, result: dict[str, object], best_score: float) -> str:
    interface = "raw source-level assignments" if arm == "raw" else "component-grouped JSON"
    return (
        COMMON_SEARCH_INFORMATION
        + f"\nPaired seed: {seed}. Generation: {generation}. Interface: {interface}.\n"
        + "Current candidate:\n"
        + _render_candidate(arm, candidate)
        + "\nLatest evaluator feedback (only this arm):\n"
        + _feedback(result, generation=generation - 1, best_score=best_score)
        + "\nReturn exactly one replacement candidate."
    )


def _provider_preflight(cli: Path) -> dict[str, str]:
    version_result = subprocess.run([str(cli), "--version"], check=True, capture_output=True)
    login_result = subprocess.run([str(cli), "login", "status"], check=True, capture_output=True)
    version = (version_result.stdout + version_result.stderr).decode("utf-8", errors="replace").strip()
    login = (login_result.stdout + login_result.stderr).decode("utf-8", errors="replace").strip()
    if not version.startswith("codex-cli ") or version.endswith("unknown"):
        raise RuntimeError(f"unexpected Codex CLI version: {version}")
    if "Logged in" not in login:
        raise RuntimeError(f"Codex login status did not confirm ChatGPT authentication: {login}")
    return {"path": str(cli), "version": version, "sha256": _sha256_file(cli), "login_status": login}


def _run_provider(cli: Path, prompt: str, workdir: Path, timeout_seconds: int) -> tuple[str, int, str]:
    output_path = workdir / "last_message.txt"
    command = [
        str(cli),
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--model",
        MODEL,
        "--cd",
        str(workdir),
        "--output-last-message",
        str(output_path),
        "-",
    ]
    completed = subprocess.run(
        command,
        input=prompt.encode("utf-8"),
        capture_output=True,
        cwd=str(workdir),
        timeout=timeout_seconds,
        shell=False,
    )
    output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    if not output:
        output = completed.stdout.decode("utf-8", errors="replace")
    diagnostics = completed.stderr.decode("utf-8", errors="replace")[-4000:]
    return output, completed.returncode, diagnostics


def _initial_result() -> dict[str, object]:
    return evaluate_spec(INITIAL_SPEC)


def _arm_events(events: list[dict[str, object]], seed: int, arm: str) -> list[dict[str, object]]:
    return [event for event in events if event.get("seed") == seed and event.get("arm") == arm and event.get("kind") == "completion"]


def _reconcile_in_doubt(events_path: Path, seed: int, arm: str) -> None:
    events = _load_events(events_path)
    dispatches = {
        str(event["request_id"]): event
        for event in events
        if event.get("kind") == "dispatch" and event.get("seed") == seed and event.get("arm") == arm
    }
    completions = {
        str(event["request_id"])
        for event in events
        if event.get("kind") == "completion" and event.get("seed") == seed and event.get("arm") == arm
    }
    for request_id, dispatch in dispatches.items():
        if request_id in completions:
            continue
        _append_jsonl(
            events_path,
            {
                "kind": "completion",
                "request_id": request_id,
                "protocol_id": PROTOCOL_ID,
                "seed": seed,
                "arm": arm,
                "generation": dispatch["generation"],
                "completed_at": _utc(),
                "returncode": None,
                "candidate_output": "",
                "candidate_output_sha256": _sha256_bytes(b""),
                "semantic_valid": False,
                "unsafe_candidate": False,
                "semantic_error": "in_doubt dispatch reconciled conservatively; counted against frozen budget",
                "behavioral_score": None,
                "behavioral_vector": {},
                "invariant_counts": {},
                "changed_components": [],
                "diagnostics_tail": "resume reconciliation",
            },
        )


def _run_arm(
    *,
    cli: Path,
    root: Path,
    run_dir: Path,
    seed: int,
    arm: str,
    timeout_seconds: int,
    events_path: Path,
    progress_path: Path,
) -> None:
    _reconcile_in_doubt(events_path, seed, arm)
    events = _load_events(events_path)
    completed = _arm_events(events, seed, arm)
    if len(completed) >= EVALUATIONS_PER_ARM_PER_SEED:
        return
    best = INITIAL_SPEC
    best_result = _initial_result()
    for prior in completed:
        if prior.get("semantic_valid") and not prior.get("unsafe_candidate") and prior.get("behavioral_score", -1.0) > best_result["behavioral_score"]:
            best = _parse_output(arm, str(prior["candidate_output"]))
            best_result = {
                "behavioral_score": prior["behavioral_score"],
                "behavioral_vector": prior.get("behavioral_vector", {}),
                "semantic_valid": True,
                "unsafe_candidate": False,
            }

    for generation in range(len(completed) + 1, GENERATIONS_PER_ARM_PER_SEED + 1):
        request_id = str(uuid.uuid4())
        prompt = _prompt(arm, seed, generation, best, best_result, float(best_result["behavioral_score"]))
        started = _utc()
        _append_jsonl(
            events_path,
            {
                "kind": "dispatch",
                "request_id": request_id,
                "protocol_id": PROTOCOL_ID,
                "seed": seed,
                "arm": arm,
                "generation": generation,
                "prompt_sha256": _sha256_bytes(prompt.encode()),
                "started_at": started,
                "status": "in_doubt",
            },
        )
        _atomic_json(
            progress_path,
            {"run_id": run_dir.name, "seed": seed, "arm": arm, "generation": generation, "completed": len(completed), "last_activity": started, "status": "running", "eta": "unknown"},
        )
        workdir = run_dir / "workers" / f"seed-{seed}" / arm / f"generation-{generation}"
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            output, returncode, diagnostics = _run_provider(cli, prompt, workdir, timeout_seconds)
            candidate_output = output.strip()
            try:
                candidate = _parse_output(arm, candidate_output)
                result = evaluate_spec(candidate)
                valid = bool(result["semantic_valid"])
                unsafe = bool(result["unsafe_candidate"])
                score = float(result["behavioral_score"])
                if valid and not unsafe and score > float(best_result["behavioral_score"]):
                    best = candidate
                    best_result = result
                semantic_error = None
            except Exception as exc:  # candidate invalidity is a measured outcome
                candidate = None
                result = {"semantic_valid": False, "unsafe_candidate": False, "behavioral_score": None, "behavioral_vector": {}}
                valid = False
                unsafe = False
                score = None
                semantic_error = f"{type(exc).__name__}: {exc}"
            completion = {
                "kind": "completion",
                "request_id": request_id,
                "protocol_id": PROTOCOL_ID,
                "seed": seed,
                "arm": arm,
                "generation": generation,
                "completed_at": _utc(),
                "returncode": returncode,
                "candidate_output": candidate_output,
                "candidate_output_sha256": _sha256_bytes(candidate_output.encode()),
                "semantic_valid": valid,
                "unsafe_candidate": unsafe,
                "semantic_error": semantic_error,
                "behavioral_score": score,
                "behavioral_vector": result.get("behavioral_vector", {}),
                "invariant_counts": result.get("invariant_counts", {}),
                "changed_components": changed_components(INITIAL_SPEC, candidate) if candidate is not None else [],
                "diagnostics_tail": diagnostics,
            }
            _append_jsonl(events_path, completion)
            completed.append(completion)
        except subprocess.TimeoutExpired as exc:
            completion = {
                "kind": "completion",
                "request_id": request_id,
                "protocol_id": PROTOCOL_ID,
                "seed": seed,
                "arm": arm,
                "generation": generation,
                "completed_at": _utc(),
                "returncode": None,
                "candidate_output": "",
                "candidate_output_sha256": _sha256_bytes(b""),
                "semantic_valid": False,
                "unsafe_candidate": False,
                "semantic_error": f"TimeoutExpired: {exc}",
                "behavioral_score": None,
                "behavioral_vector": {},
                "invariant_counts": {},
                "changed_components": [],
                "diagnostics_tail": "provider timeout; counted against frozen budget",
            }
            _append_jsonl(events_path, completion)
            completed.append(completion)
        _atomic_json(
            progress_path,
            {"run_id": run_dir.name, "seed": seed, "arm": arm, "generation": generation, "completed": len(completed), "last_activity": _utc(), "status": "running", "eta": "unknown"},
        )


def run_search(output_root: Path, cli: Path, timeout_seconds: int, resume_dir: Path | None = None) -> Path:
    protocol = build_protocol_manifest()
    provider = _provider_preflight(cli)
    if resume_dir is None:
        if protocol["execution_boundary"]["formal_search_started"]:
            raise RuntimeError("protocol unexpectedly marks formal search as already started")
        run_id = f"b1-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        events_path = run_dir / "events.jsonl"
        progress_path = run_dir / "progress.json"
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "status": "RUNNING",
            "started_at": _utc(),
            "provider": provider,
            "model": MODEL,
            "protocol_manifest_sha256": _sha256_bytes(json.dumps(protocol, sort_keys=True).encode()),
            "protocol_manifest": protocol,
            "seeds": list(PAIRED_SEEDS),
            "generations_per_arm_per_seed": GENERATIONS_PER_ARM_PER_SEED,
            "evaluations_per_arm_per_seed": EVALUATIONS_PER_ARM_PER_SEED,
            "retry_semantics": "no retry; timeout/in_doubt consumes one generation and evaluation",
            "blind_boundary": "provider workdirs contain no evaluator, cohort, truth, or repository files",
        }
        _atomic_json(run_dir / "execution_manifest.json", manifest)
        _atomic_json(progress_path, {"run_id": run_id, "status": "running", "completed": 0, "last_activity": _utc(), "eta": "unknown"})
    else:
        run_dir = resume_dir.resolve()
        manifest_path = run_dir / "execution_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("resume directory has no execution manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("status") == "COMPLETE":
            raise RuntimeError("resume target is not an unfinished B1 run for this protocol")
        if manifest.get("protocol_manifest_sha256") != _sha256_bytes(json.dumps(protocol, sort_keys=True).encode()):
            raise RuntimeError("resume protocol hash changed")
        if manifest.get("provider") != provider or manifest.get("model") != MODEL:
            raise RuntimeError("resume provider/model identity changed")
        events_path = run_dir / "events.jsonl"
        progress_path = run_dir / "progress.json"
        manifest["status"] = "RUNNING"
        manifest["resumed_at"] = _utc()
        _atomic_json(manifest_path, manifest)

    for seed_index, seed in enumerate(PAIRED_SEEDS):
        arm_order = ("raw", "structured") if seed_index % 2 == 0 else ("structured", "raw")
        for arm in arm_order:
            _run_arm(
                cli=cli,
                root=output_root,
                run_dir=run_dir,
                seed=seed,
                arm=arm,
                timeout_seconds=timeout_seconds,
                events_path=events_path,
                progress_path=progress_path,
            )
    manifest["status"] = "COMPLETE"
    manifest["completed_at"] = _utc()
    _atomic_json(run_dir / "execution_manifest.json", manifest)
    _atomic_json(progress_path, {"run_id": run_dir.name, "status": "complete", "completed": sum(event.get("kind") == "completion" for event in _load_events(events_path)), "last_activity": _utc(), "eta": "0s"})
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("artifacts.local/evidence/l10m_b1/runs"))
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    path = run_search(args.output_root, args.cli, args.timeout_seconds, args.resume)
    print(path)


if __name__ == "__main__":
    main()
