"""Execute the frozen, paired L10M-B1 search with conservative journaling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
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
DEFAULT_DOCKER = Path(r"E:\codex-tools\tools\docker-desktop\resources\bin\docker.exe")
DEFAULT_DOCKER_IMAGE = "l10m-b1-codex:0.148.0"
DEFAULT_AUTH = Path(r"C:\Users\26442\.codex\auth.json")
DEFAULT_PROXY_BIND = "172.31.224.1"
DOCKER_CA_CERT = Path(__file__).with_name("docker") / "we1.crt.pem"
MODEL = "gpt-5.6-sol"
BLIND_VIOLATION_MARKERS = (
    "MEMORY.md",
    "rollout_summaries",
    "Get-ChildItem",
    "rg -n",
    "git status",
    "git log",
)
DOCKER_BLIND_VIOLATION_MARKERS = (
    "MEMORY.md",
    "rollout_summaries",
    "C:\\Users\\",
    "E:\\linnan\\",
    "artifacts.local/evidence",
    "execution_manifest.json",
)


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
        + "\nDo not run shell commands, inspect files, or use tools. Use only the information in this prompt.\n"
        + "Return exactly one replacement candidate."
    )


def _provider_preflight_native(cli: Path) -> dict[str, str]:
    version_result = subprocess.run([str(cli), "--version"], check=True, capture_output=True)
    login_result = subprocess.run([str(cli), "login", "status"], check=True, capture_output=True)
    version = (version_result.stdout + version_result.stderr).decode("utf-8", errors="replace").strip()
    login = (login_result.stdout + login_result.stderr).decode("utf-8", errors="replace").strip()
    if not version.startswith("codex-cli ") or version.endswith("unknown"):
        raise RuntimeError(f"unexpected Codex CLI version: {version}")
    if "Logged in" not in login:
        raise RuntimeError(f"Codex login status did not confirm ChatGPT authentication: {login}")
    return {"path": str(cli), "version": version, "sha256": _sha256_file(cli), "login_status": login}


def _docker_run_base(docker: Path, image: str, *, proxy_url: str = "http://host.docker.internal:7890") -> list[str]:
    return [
        str(docker),
        "run",
        "--rm",
        "--interactive",
        "--init",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "0:0",
        "--pids-limit",
        "128",
        "--memory",
        "1g",
        "--cpus",
        "1.0",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--tmpfs",
        "/root/.codex:rw,nosuid,nodev,size=64m",
        "--env",
        "CODEX_HOME=/root/.codex",
        "--env",
        f"HTTPS_PROXY={proxy_url}",
        "--env",
        f"HTTP_PROXY={proxy_url}",
        "--env",
        "NO_PROXY=localhost,127.0.0.1",
        image,
    ]


class _LocalProxyForwarder:
    def __init__(self, bind_address: str, target_address: str = "127.0.0.1", target_port: int = 7890) -> None:
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((bind_address, 0))
        self._listener.listen(8)
        self._listener.settimeout(0.5)
        self.bind_address = bind_address
        self.port = int(self._listener.getsockname()[1])
        self.target_address = target_address
        self.target_port = target_port
        self._stop = False
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    @property
    def proxy_url(self) -> str:
        return f"http://{self.bind_address}:{self.port}"

    def _accept(self) -> None:
        while not self._stop:
            try:
                source, _ = self._listener.accept()
            except (socket.timeout, OSError):
                continue
            try:
                target = socket.create_connection((self.target_address, self.target_port), timeout=5)
            except OSError:
                source.close()
                continue
            threading.Thread(target=self._relay, args=(source, target), daemon=True).start()
            threading.Thread(target=self._relay, args=(target, source), daemon=True).start()

    @staticmethod
    def _relay(source: socket.socket, target: socket.socket) -> None:
        try:
            while True:
                data = source.recv(65536)
                if not data:
                    break
                target.sendall(data)
        except OSError:
            pass
        finally:
            try:
                source.close()
            finally:
                target.close()

    def close(self) -> None:
        self._stop = True
        try:
            self._listener.close()
        finally:
            self._thread.join(timeout=2)


def _docker_image_identity(docker: Path, image: str) -> dict[str, str]:
    inspected = subprocess.run(
        [str(docker), "image", "inspect", image, "--format", "{{.Id}}|{{json .RepoDigests}}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    image_id, repo_digests = inspected.stdout.strip().split("|", 1)
    return {"image": image, "image_id": image_id, "repo_digests": repo_digests}


def _provider_preflight_docker(docker: Path, image: str, auth_path: Path, proxy_bind: str = DEFAULT_PROXY_BIND) -> dict[str, object]:
    if not docker.exists():
        raise RuntimeError(f"Docker executable not found: {docker}")
    if not auth_path.is_file():
        raise RuntimeError(f"Codex auth file not found: {auth_path}")
    if not DOCKER_CA_CERT.is_file():
        raise RuntimeError(f"Docker CA certificate not found: {DOCKER_CA_CERT}")
    version_result = subprocess.run([str(docker), "version", "--format", "{{.Server.Version}}"], check=True, capture_output=True, text=True, encoding="utf-8")
    identity = _docker_image_identity(docker, image)
    common = _docker_run_base(docker, image)
    auth_mount = f"type=bind,source={auth_path},target=/root/.codex/auth.json,readonly"
    check_base = common[:-1] + ["--network", "none", "--mount", auth_mount, "--entrypoint", "codex", image]
    version = subprocess.run(check_base + ["--version"], check=True, capture_output=True, text=True, encoding="utf-8")
    login = subprocess.run(check_base + ["login", "status"], check=True, capture_output=True, text=True, encoding="utf-8")
    codex_version = (version.stdout + version.stderr).strip()
    login_status = (login.stdout + login.stderr).strip()
    if not codex_version.startswith("codex-cli ") or codex_version.endswith("unknown"):
        raise RuntimeError(f"unexpected container Codex version: {codex_version}")
    if "Logged in" not in login_status:
        raise RuntimeError(f"container Codex login status did not confirm ChatGPT authentication: {login_status}")
    return {
        "backend": "docker",
        "docker_path": str(docker),
        "docker_server_version": version_result.stdout.strip(),
        "image": identity,
        "codex_version": codex_version,
        "login_status": login_status,
        "auth_mount": "read_only_host_auth",
        "ca_cert_sha256": _sha256_file(DOCKER_CA_CERT),
        "proxy_bind": proxy_bind,
    }


def _provider_preflight(
    cli: Path,
    *,
    backend: str,
    docker: Path,
    image: str,
    auth_path: Path,
    proxy_bind: str = DEFAULT_PROXY_BIND,
) -> dict[str, object]:
    if backend == "native":
        return {"backend": "native", **_provider_preflight_native(cli)}
    if backend == "docker":
        return _provider_preflight_docker(docker, image, auth_path, proxy_bind)
    raise ValueError(f"unknown provider backend: {backend}")


def _run_provider(cli: Path, prompt: str, workdir: Path, timeout_seconds: int) -> tuple[str, int, str]:
    output_path = workdir / "last_message.txt"
    command = [
        str(cli),
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
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


def _run_provider_docker(
    docker: Path,
    image: str,
    auth_path: Path,
    prompt: str,
    workdir: Path,
    timeout_seconds: int,
    proxy_bind: str = DEFAULT_PROXY_BIND,
) -> tuple[str, int, str]:
    forwarder = _LocalProxyForwarder(proxy_bind)
    auth_mount = f"type=bind,source={auth_path},target=/root/.codex/auth.json,readonly"
    command = _docker_run_base(docker, image, proxy_url=forwarder.proxy_url)[:-1] + [
        "--mount",
        f"type=bind,source={workdir},target=/workspace,readonly",
        "--mount",
        auth_mount,
        "--workdir",
        "/workspace",
        "--network",
        "bridge",
        image,
        "/bin/sh",
        "-c",
        "codex -c responses_websocket=false exec --ephemeral --skip-git-repo-check --ignore-user-config --ignore-rules --sandbox danger-full-access --model "
        + MODEL
        + " --cd /workspace --output-last-message /tmp/last_message.txt - >/dev/null 2>/tmp/diagnostics.txt; rc=$?; cat /tmp/last_message.txt 2>/dev/null; cat /tmp/diagnostics.txt >&2; exit $rc",
    ]
    try:
        completed = subprocess.run(
            command,
            input=prompt.encode("utf-8"),
            capture_output=True,
            cwd=str(workdir),
            timeout=timeout_seconds,
            shell=False,
        )
        output = completed.stdout.decode("utf-8", errors="replace")
        diagnostics = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        return output, completed.returncode, diagnostics
    finally:
        forwarder.close()


def _docker_isolation_canary(docker: Path, image: str, auth_path: Path, output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="b1-docker-canary-", dir=output_root) as temporary:
        workdir = Path(temporary)
        (workdir / "worker_marker.txt").write_text("worker-visible\n", encoding="utf-8")
        host_marker = output_root / f"{workdir.name}-host-marker.txt"
        host_marker.write_text("host-hidden\n", encoding="utf-8")
        try:
            auth_mount = f"type=bind,source={auth_path},target=/root/.codex/auth.json,readonly"
            command = _docker_run_base(docker, image)[:-1] + [
                "--network",
                "none",
                "--mount",
                f"type=bind,source={workdir},target=/workspace,readonly",
                "--mount",
                auth_mount,
                "--workdir",
                "/workspace",
                image,
                "/bin/sh",
                "-c",
                "grep -q worker-visible /workspace/worker_marker.txt && ! test -e /host-hidden-marker && ! test -e /workspace/../host-hidden-marker",
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"Docker isolation canary failed: {completed.stderr.strip()}")
            return {"status": "PASS", "worker_mount": "read_only", "host_marker": "not_visible", "image": image}
        finally:
            host_marker.unlink(missing_ok=True)


def _blind_violation(diagnostics: str, backend: str) -> str | None:
    markers = DOCKER_BLIND_VIOLATION_MARKERS if backend == "docker" else BLIND_VIOLATION_MARKERS
    for marker in markers:
        if marker in diagnostics:
            return marker
    return None


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
    backend: str,
    docker: Path,
    docker_image: str,
    auth_path: Path,
    proxy_bind: str,
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
            if backend == "docker":
                output, returncode, diagnostics = _run_provider_docker(docker, docker_image, auth_path, prompt, workdir, timeout_seconds, proxy_bind)
            else:
                output, returncode, diagnostics = _run_provider(cli, prompt, workdir, timeout_seconds)
            candidate_output = output.strip()
            blind_marker = _blind_violation(diagnostics, backend)
            if blind_marker is not None:
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
                    "semantic_valid": False,
                    "unsafe_candidate": False,
                    "semantic_error": f"BlindIsolationViolation: provider diagnostics contained {blind_marker}",
                    "behavioral_score": None,
                    "behavioral_vector": {},
                    "invariant_counts": {},
                    "changed_components": [],
                    "diagnostics_tail": diagnostics,
                }
                _append_jsonl(events_path, completion)
                raise RuntimeError(f"B1 blind isolation violation: {blind_marker}")
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


def run_search(
    output_root: Path,
    cli: Path,
    timeout_seconds: int,
    resume_dir: Path | None = None,
    supersedes: str | None = None,
    backend: str = "docker",
    docker: Path = DEFAULT_DOCKER,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    auth_path: Path = DEFAULT_AUTH,
    proxy_bind: str = DEFAULT_PROXY_BIND,
) -> Path:
    output_root = output_root.resolve()
    protocol = build_protocol_manifest()
    provider = _provider_preflight(cli, backend=backend, docker=docker, image=docker_image, auth_path=auth_path, proxy_bind=proxy_bind)
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
            "supersedes_non_evaluable_attempt": supersedes,
            "provider_invocation": {
                "ignore_user_config": True,
                "ignore_rules": True,
                "sandbox": "danger-full-access_inside_hard_container",
                "outer_isolation": "read_only_rootfs_cap_drop_all_no_new_privileges_read_only_workspace",
                "skip_git_repo_check": True,
                "backend": backend,
                "docker_image": docker_image if backend == "docker" else None,
                "network": "bridge_via_host_proxy" if backend == "docker" else "native_default",
                "proxy_bind": proxy_bind if backend == "docker" else None,
                "host_mounts": ["worker_workspace_read_only", "codex_auth_read_only"] if backend == "docker" else ["none"],
                "cap_drop": ["ALL"] if backend == "docker" else [],
                "no_new_privileges": backend == "docker",
            },
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
                backend=backend,
                docker=docker,
                docker_image=docker_image,
                auth_path=auth_path,
                proxy_bind=proxy_bind,
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
    parser.add_argument("--backend", choices=("docker", "native"), default="docker")
    parser.add_argument("--docker", type=Path, default=DEFAULT_DOCKER)
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--auth-path", type=Path, default=DEFAULT_AUTH)
    parser.add_argument("--proxy-bind", default=DEFAULT_PROXY_BIND)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--supersedes")
    parser.add_argument("--docker-canary", action="store_true")
    args = parser.parse_args()
    if args.docker_canary:
        result = _docker_isolation_canary(args.docker, args.docker_image, args.auth_path, args.output_root.resolve())
        print(json.dumps(result, sort_keys=True))
        return
    path = run_search(args.output_root, args.cli, args.timeout_seconds, args.resume, args.supersedes, args.backend, args.docker, args.docker_image, args.auth_path, args.proxy_bind)
    print(path)


if __name__ == "__main__":
    main()
