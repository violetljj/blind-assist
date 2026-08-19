"""Docker Codex transport shared by B1 and infrastructure-only qualification."""

from __future__ import annotations

import hashlib
import socket
import subprocess
import tempfile
import threading
from pathlib import Path


DEFAULT_DOCKER = Path(r"E:\codex-tools\tools\docker-desktop\resources\bin\docker.exe")
DEFAULT_DOCKER_IMAGE = "l10m-b1-codex:0.148.0"
DEFAULT_AUTH = Path(r"C:\Users\26442\.codex\auth.json")
DEFAULT_PROXY_BIND = "172.31.224.1"
DOCKER_CA_CERT = Path(__file__).with_name("docker") / "we1.crt.pem"
MODEL = "gpt-5.6-sol"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def docker_run_base(
    docker: Path,
    image: str,
    *,
    proxy_url: str | None = "http://host.docker.internal:7890",
) -> list[str]:
    command = [
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
    ]
    if proxy_url is not None:
        command.extend(
            [
                "--env",
                f"HTTPS_PROXY={proxy_url}",
                "--env",
                f"HTTP_PROXY={proxy_url}",
                "--env",
                "NO_PROXY=localhost,127.0.0.1",
            ]
        )
    command.append(image)
    return command


class LocalProxyForwarder:
    def __init__(
        self,
        bind_address: str,
        target_address: str = "127.0.0.1",
        target_port: int = 7890,
    ) -> None:
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
                target = socket.create_connection(
                    (self.target_address, self.target_port), timeout=5
                )
            except OSError:
                source.close()
                continue
            threading.Thread(
                target=self._relay, args=(source, target), daemon=True
            ).start()
            threading.Thread(
                target=self._relay, args=(target, source), daemon=True
            ).start()

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


def docker_image_identity(docker: Path, image: str) -> dict[str, str]:
    inspected = subprocess.run(
        [
            str(docker),
            "image",
            "inspect",
            image,
            "--format",
            "{{.Id}}|{{json .RepoDigests}}",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    image_id, repo_digests = inspected.stdout.strip().split("|", 1)
    return {"image": image, "image_id": image_id, "repo_digests": repo_digests}


def provider_preflight_docker(
    docker: Path,
    image: str,
    auth_path: Path,
    proxy_bind: str = DEFAULT_PROXY_BIND,
) -> dict[str, object]:
    if not docker.exists():
        raise RuntimeError(f"Docker executable not found: {docker}")
    if not auth_path.is_file():
        raise RuntimeError(f"Codex auth file not found: {auth_path}")
    if not DOCKER_CA_CERT.is_file():
        raise RuntimeError(f"Docker CA certificate not found: {DOCKER_CA_CERT}")
    version_result = subprocess.run(
        [str(docker), "version", "--format", "{{.Server.Version}}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    identity = docker_image_identity(docker, image)
    auth_mount = (
        f"type=bind,source={auth_path},target=/root/.codex/auth.json,readonly"
    )
    check_base = docker_run_base(docker, image)[:-1] + [
        "--network",
        "none",
        "--mount",
        auth_mount,
        "--entrypoint",
        "codex",
        image,
    ]
    version = subprocess.run(
        check_base + ["--version"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    login = subprocess.run(
        check_base + ["login", "status"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    codex_version = (version.stdout + version.stderr).strip()
    login_status = (login.stdout + login.stderr).strip()
    if not codex_version.startswith("codex-cli ") or codex_version.endswith("unknown"):
        raise RuntimeError(f"unexpected container Codex version: {codex_version}")
    if "Logged in" not in login_status:
        raise RuntimeError(
            "container Codex login status did not confirm ChatGPT authentication: "
            + login_status
        )
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


def run_provider_docker(
    docker: Path,
    image: str,
    auth_path: Path,
    prompt: str,
    workdir: Path,
    timeout_seconds: int,
    proxy_bind: str = DEFAULT_PROXY_BIND,
    transport_route: str = "proxy",
) -> tuple[str, int, str]:
    if transport_route not in {"proxy", "direct"}:
        raise ValueError(f"unknown Docker transport route: {transport_route}")
    forwarder = (
        LocalProxyForwarder(proxy_bind) if transport_route == "proxy" else None
    )
    auth_mount = (
        f"type=bind,source={auth_path},target=/root/.codex/auth.json,readonly"
    )
    proxy_url = forwarder.proxy_url if forwarder is not None else None
    command = docker_run_base(docker, image, proxy_url=proxy_url)[:-1] + [
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
        if forwarder is not None:
            forwarder.close()


def docker_isolation_canary(
    docker: Path, image: str, auth_path: Path, output_root: Path
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="b1-docker-canary-", dir=output_root
    ) as temporary:
        workdir = Path(temporary)
        (workdir / "worker_marker.txt").write_text(
            "worker-visible\n", encoding="utf-8"
        )
        host_marker = output_root / f"{workdir.name}-host-marker.txt"
        host_marker.write_text("host-hidden\n", encoding="utf-8")
        try:
            auth_mount = (
                f"type=bind,source={auth_path},target=/root/.codex/auth.json,readonly"
            )
            command = docker_run_base(docker, image)[:-1] + [
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
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Docker isolation canary failed: {completed.stderr.strip()}"
                )
            return {
                "status": "PASS",
                "worker_mount": "read_only",
                "host_marker": "not_visible",
                "image": image,
            }
        finally:
            host_marker.unlink(missing_ok=True)
