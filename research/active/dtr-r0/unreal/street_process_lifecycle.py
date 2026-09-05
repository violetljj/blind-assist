"""Bounded cleanup for a task's observed subprocess descendants.

Register immediately after Popen, call capture regularly while the root is alive,
and call cleanup in finally. Ownership comes only from that Popen and observed
parent/child relationships, never executable names, a global PID list, or ports.
PID and creation time must still match before each signal. protected identities
exclude a known shared process and its branch from tracking and cleanup.

This is polling, not an OS job object: a process that spawns and becomes orphaned
entirely between captures can be missed. A clean receipt proves release of the
observed identities and requested ports, not complete OS containment. Calls on a
tree are serial. On Windows psutil terminate is a forced termination; the natural
exit interval (or a caller's application shutdown request) precedes that action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import errno
import ipaddress
import math
import socket
import subprocess
import time
from typing import Callable, Iterable

import psutil


@dataclass(frozen=True, order=True)
class ProcessIdentity:
    pid: int
    create_time: float

    @classmethod
    def of(cls, process: psutil.Process) -> ProcessIdentity:
        return cls(process.pid, process.create_time())


@dataclass
class _Tracked:
    identity: ProcessIdentity
    parent: ProcessIdentity | None
    depth: int
    name: str | None


class TaskProcessTree:
    """Track only a caller-owned Popen and its creation-checked descendants.

    wait() captures at a bounded interval and preserves Popen's exit/timeout
    semantics. on_poll can capture another owned tree (e.g. a serving worker
    while waiting for an editor). A timeout does not clean up: use finally.
    cleanup() returns a JSON-serializable verification receipt and is safe to
    repeat. A port is verification only; its listener is never adopted or killed.
    """

    def __init__(self, root: subprocess.Popen, *, owner: str,
                 protected: Iterable[ProcessIdentity] = ()):
        if not owner:
            raise ValueError("An explicit task owner is required")
        self.root = root
        self.owner = owner
        self.protected = frozenset(protected)
        self.root_identity: ProcessIdentity | None = None
        self._tracked: dict[ProcessIdentity, _Tracked] = {}
        self._errors: dict[tuple, dict] = {}
        self._excluded: set[ProcessIdentity] = set()
        self._actions: list[dict] = []
        self.capture_count = 0
        self._tracking_complete = True
        try:
            process = psutil.Process(root.pid)
            identity = ProcessIdentity.of(process)
            if identity in self.protected:
                raise ValueError("The explicitly owned root cannot be protected")
            self.root_identity = identity
            self._add(process, identity, parent=None, depth=0)
            if root.poll() is not None:
                self._error("register", identity, "root_already_exited", incomplete=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            self._error("register", None, type(exc).__name__, incomplete=True)
        self.capture()

    @property
    def identities(self) -> tuple[ProcessIdentity, ...]:
        return tuple(self._tracked)

    def _error(self, operation: str, identity: ProcessIdentity | None,
               reason: str, *, incomplete: bool = False) -> None:
        self._errors[(operation, identity, reason)] = {
            "operation": operation,
            "identity": asdict(identity) if identity else {"pid": self.root.pid},
            "reason": reason,
        }
        if incomplete:
            self._tracking_complete = False

    def _add(self, process: psutil.Process, identity: ProcessIdentity,
             *, parent: ProcessIdentity | None, depth: int) -> None:
        try:
            name = process.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            name = None
        self._tracked[identity] = _Tracked(identity, parent, depth, name)

    def _resolve(self, identity: ProcessIdentity) -> tuple[psutil.Process | None, str]:
        try:
            process = psutil.Process(identity.pid)
            if ProcessIdentity.of(process) != identity:
                return None, "PID_REUSED"
            if not process.is_running():
                return None, "EXITED"
            if process.status() == psutil.STATUS_ZOMBIE:
                return None, "EXITED_ZOMBIE"
            return process, "ALIVE"
        except psutil.NoSuchProcess:
            return None, "EXITED"
        except psutil.AccessDenied:
            self._error("identity", identity, "AccessDenied", incomplete=True)
            return None, "UNKNOWN"

    def capture(self) -> None:
        """Retain new descendants, including those of already orphaned children."""
        self.capture_count += 1
        pending = list(self._tracked.values())
        visited: set[ProcessIdentity] = set()
        while pending:
            tracked = pending.pop()
            identity = tracked.identity
            if identity in visited:
                continue
            visited.add(identity)
            parent, _ = self._resolve(identity)
            if parent is None:
                continue
            try:
                children = parent.children(recursive=False)
                for child in children:
                    child_identity = ProcessIdentity.of(child)
                    # Check the relation again after the snapshot. Do not adopt
                    # a reused PID or an older, unrelated/shared process.
                    if (not parent.is_running() or child.ppid() != identity.pid
                            or child_identity.create_time < identity.create_time):
                        continue
                    if child_identity in self.protected:
                        self._excluded.add(child_identity)
                        continue
                    if child_identity not in self._tracked:
                        self._add(child, child_identity, parent=identity,
                                  depth=tracked.depth + 1)
                        pending.append(self._tracked[child_identity])
            except psutil.NoSuchProcess:
                # Exiting during inspection is normal; existing identities
                # remain retained, including previously captured children.
                continue
            except psutil.AccessDenied:
                self._error("children", identity, "AccessDenied", incomplete=True)
        self.root.poll()  # reap the direct child without waiting

    @staticmethod
    def _duration(value: float, name: str, *, positive: bool = False) -> float:
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
        return float(value)

    def wait(self, timeout: float, *, poll_interval: float = .25,
             on_poll: Callable[[], object] | None = None) -> int:
        deadline = time.monotonic() + self._duration(timeout, "timeout")
        interval = self._duration(poll_interval, "poll_interval", positive=True)
        while True:
            self.capture()
            if on_poll is not None:
                on_poll()
            code = self.root.poll()
            if code is not None:
                return code
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(self.root.args, timeout)
            time.sleep(min(interval, remaining))

    def _states(self) -> dict[ProcessIdentity, str]:
        return {identity: self._resolve(identity)[1] for identity in self._tracked}

    @staticmethod
    def _exited(states: dict[ProcessIdentity, str]) -> bool:
        return all(state in {"EXITED", "EXITED_ZOMBIE", "PID_REUSED"}
                   for state in states.values())

    def _wait_for_exits(self, timeout: float, interval: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            self.capture()
            if self._exited(self._states()) and self.root.poll() is not None:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(interval, remaining))

    def _signal_remaining(self, action: str) -> None:
        self.capture()
        for tracked in sorted(self._tracked.values(), key=lambda row: row.depth, reverse=True):
            process, state = self._resolve(tracked.identity)
            if process is None:
                continue
            event = {"identity": asdict(tracked.identity), "action": action}
            try:
                # psutil itself also checks PID reuse immediately before these
                # operations; a new process with the old PID is never a target.
                getattr(process, action)()
                event["result"] = "SENT"
            except psutil.NoSuchProcess:
                event["result"] = "ALREADY_EXITED"
            except psutil.AccessDenied:
                event["result"] = "ACCESS_DENIED"
                self._error(action, tracked.identity, "AccessDenied")
            self._actions.append(event)

    @staticmethod
    def _ports(endpoints: tuple[tuple[str, int], ...]) -> list[dict]:
        result = []
        for host, port in endpoints:
            with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as probe:
                probe.settimeout(.1)
                code = probe.connect_ex((host, port))
            state = ("OPEN" if code == 0 else "CLOSED"
                     if code in {errno.ECONNREFUSED, 10061} else "UNKNOWN")
            verification = "LOOPBACK_CONNECT"
            if state == "UNKNOWN":
                # Windows can return WSAEWOULDBLOCK at the short connect
                # timeout even after every listener has exited. Inspect the
                # local listener table for this exact port instead of treating
                # a timeout as either a listener or proof of closure.
                try:
                    listeners = psutil.net_connections(kind="tcp")
                    state = "OPEN" if any(
                        row.status == psutil.CONN_LISTEN and row.laddr.port == port
                        and row.laddr.ip in {host, "0.0.0.0", "::"}
                        for row in listeners if row.laddr
                    ) else "CLOSED"
                    verification = "LOCAL_TCP_LISTENER_TABLE"
                except (psutil.AccessDenied, OSError):
                    pass
            result.append({"host": host, "port": port, "state": state,
                           "open": state == "OPEN" if state != "UNKNOWN" else None,
                           "probe_code": code,"verification":verification})
        return result

    def cleanup(self, *, natural_timeout: float = 2., terminate_timeout: float = 2.,
                kill_timeout: float = 2., port_timeout: float = 2.,
                poll_interval: float = .25,
                ports: Iterable[tuple[str, int]] = ()) -> dict:
        """Wait naturally, signal remaining owned identities, then verify ports.

        released is false on surviving/unknown identities, incomplete tracking,
        an unclosed requested port, or a root whose exit cannot be confirmed.
        The receipt includes only observed descendants; see the module boundary.
        """
        phases = [("natural", self._duration(natural_timeout, "natural_timeout")),
                  ("terminate", self._duration(terminate_timeout, "terminate_timeout")),
                  ("kill", self._duration(kill_timeout, "kill_timeout"))]
        interval = self._duration(poll_interval, "poll_interval", positive=True)
        port_timeout = self._duration(port_timeout, "port_timeout")
        endpoints = tuple(ports)
        for host, port in endpoints:
            if not ipaddress.ip_address(host).is_loopback or not 0 < port <= 65535:
                raise ValueError("Verification ports must be explicit loopback addresses and valid ports")
        started = time.monotonic()
        for phase, duration in phases:
            if phase != "natural":
                self._signal_remaining(phase)
            self._wait_for_exits(duration, interval)
            if self._exited(self._states()) and self.root.poll() is not None:
                break
        # Port closure can trail the root's exit and even its descendants' exit.
        # Wait for that observable separately; never kill a process by its port.
        deadline = time.monotonic() + port_timeout
        while True:
            self.capture()
            port_states = self._ports(endpoints)
            remaining = deadline - time.monotonic()
            if all(row["state"] == "CLOSED" for row in port_states) or remaining <= 0:
                break
            time.sleep(min(interval, remaining))
        states = self._states()
        root_code = self.root.poll()
        records = []
        for tracked in self._tracked.values():
            records.append({**asdict(tracked), "state": states[tracked.identity]})
        survivors = [asdict(identity) for identity, state in states.items()
                     if state in {"ALIVE", "UNKNOWN"}]
        return {
            "schema": "blindassist.task_process_tree.v1",
            "owner": self.owner,
            "scope": "observed_descendants",
            "root_pid": self.root.pid,
            "root_exit_code": root_code,
            "tracking_complete": self._tracking_complete,
            "capture_count": self.capture_count,
            "processes": records,
            "excluded": [asdict(identity) for identity in sorted(self._excluded)],
            "actions": list(self._actions),
            "errors": list(self._errors.values()),
            "survivors": survivors,
            "ports": port_states,
            "cleanup_seconds": time.monotonic() - started,
            "released": (self._tracking_complete and not survivors and root_code is not None
                         and all(row["state"] == "CLOSED" for row in port_states)),
        }
