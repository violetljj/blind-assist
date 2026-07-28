from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import ssl
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
import urllib.request


NETWORK_CHUNK = 8 << 20
SAFE_CONTENT_RANGE = re.compile(r"^[A-Za-z0-9 */-]{0,160}$")


class BudgetExceeded(RuntimeError):
    pass


class RangeContractError(RuntimeError):
    pass


class EvidencePersistenceFailure(RuntimeError):
    pass


class TransportFailure(RuntimeError):
    def __init__(self, category: str, snapshot: dict[str, Any]) -> None:
        super().__init__(category)
        self.category = category
        self.snapshot = snapshot


@dataclass
class ByteBudget:
    limit: int
    consumed: int = 0

    def reserve(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("NEGATIVE_BYTE_RESERVATION")
        if self.consumed + amount > self.limit:
            raise BudgetExceeded("BYTE_BUDGET_EXCEEDED")
        self.consumed += amount

    def release(self, amount: int) -> None:
        if amount < 0 or amount > self.consumed:
            raise ValueError("INVALID_BYTE_RELEASE")
        self.consumed -= amount


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_scalar_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def safe_content_range(value: Any) -> str | None:
    if not isinstance(value, str) or not SAFE_CONTENT_RANGE.fullmatch(value):
        return None
    return value


def sanitized_error(error: BaseException) -> dict[str, Any]:
    reason = error.reason if isinstance(error, URLError) else error
    category = "OTHER_IO"
    if isinstance(error, RangeContractError):
        category = "RANGE_CONTRACT"
    elif isinstance(error, HTTPError) and error.code == 407:
        category = "PROXY_AUTH"
    elif isinstance(error, HTTPError):
        category = "HTTP_ERROR"
    elif isinstance(reason, (TimeoutError, socket.timeout)):
        category = "TIMEOUT"
    elif isinstance(reason, ssl.SSLError):
        category = "TLS"
    elif isinstance(reason, socket.gaierror):
        category = "DNS"
    elif isinstance(reason, ConnectionResetError):
        category = "CONNECTION_RESET"
    elif isinstance(reason, ConnectionRefusedError):
        category = "CONNECTION_REFUSED"
    elif isinstance(error, http.client.IncompleteRead):
        category = "INCOMPLETE_READ"
    errno = safe_scalar_int(getattr(reason, "errno", None))
    http_status = safe_scalar_int(error.code) if isinstance(error, HTTPError) else None
    fingerprint_input = (
        f"{type(error).__name__}|{type(reason).__name__}|"
        f"{category}|{errno}|{http_status}"
    )
    return {
        "category": category,
        "outer_type": type(error).__name__[:80],
        "reason_type": type(reason).__name__[:80],
        "errno": errno,
        "http_status": http_status,
        "exception_chain_sha256": hashlib.sha256(
            fingerprint_input.encode("utf-8")
        ).hexdigest(),
    }


class AppendOnlyLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError("R2_LEDGER_MUST_BE_NEW")
        self.sequence = 0
        self.previous_row_sha256: str | None = None

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        chained = {
            **row,
            "ledger_sequence": self.sequence,
            "previous_row_sha256": self.previous_row_sha256,
        }
        canonical = json.dumps(
            chained, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        chained["row_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        serialized = json.dumps(chained, ensure_ascii=False, separators=(",", ":"))
        try:
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            with os.fdopen(descriptor, "a", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise EvidencePersistenceFailure("LEDGER_APPEND_FAILED") from error
        self.sequence += 1
        self.previous_row_sha256 = chained["row_sha256"]
        return chained


def write_json_atomic(path: Path, payload: dict[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if exclusive and path.exists():
        raise EvidencePersistenceFailure("TERMINAL_RECEIPT_ALREADY_EXISTS")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if exclusive and path.exists():
            raise EvidencePersistenceFailure("TERMINAL_RECEIPT_ALREADY_EXISTS")
        os.replace(temporary, path)
    except OSError as error:
        raise EvidencePersistenceFailure("EVIDENCE_ATOMIC_WRITE_FAILED") from error


class DiagnosticRemoteRange:
    def __init__(
        self,
        *,
        url: str,
        length: int,
        budget: int,
        ledger_path: Path,
        progress_path: Path,
        failure_receipt_path: Path,
        phase: str,
        user_agent: str,
        opener: Callable[..., Any] = urllib.request.urlopen,
        maximum_attempts: int = 3,
        timeout_seconds: int = 90,
        sleep: Callable[[float], None] = time.sleep,
        resource_probe: Callable[[], dict[str, int | None]] | None = None,
    ) -> None:
        if maximum_attempts != 3:
            raise ValueError("R2_MAXIMUM_ATTEMPTS_MUST_EQUAL_THREE")
        if failure_receipt_path.exists() or progress_path.exists():
            raise FileExistsError("R2_EVIDENCE_NAMESPACE_MUST_BE_NEW")
        self.url = url
        self.length = length
        self.position = 0
        self.budget = ByteBudget(budget)
        self.ledger = AppendOnlyLedger(ledger_path)
        self.progress_path = progress_path
        self.failure_receipt_path = failure_receipt_path
        self.phase = phase
        self.user_agent = user_agent
        self.opener = opener
        self.maximum_attempts = maximum_attempts
        self.timeout_seconds = timeout_seconds
        self.sleep = sleep
        self.resource_probe = resource_probe or (
            lambda: {"peak_rss_bytes": None, "peak_commit_bytes": None}
        )
        self.logical_request_id = 0
        self.successful_bytes = 0
        self.request_attempt_count = 0
        self.last_good_offset: int | None = None
        self.last_request: dict[str, Any] | None = None
        self.failed = False

    def seek(self, offset: int) -> int:
        if self.failed:
            raise RuntimeError("TRANSPORT_TERMINAL")
        if not 0 <= offset <= self.length:
            raise ValueError("SEEK_OUTSIDE_OBJECT")
        self.position = offset
        return offset

    def snapshot(self) -> dict[str, Any]:
        resources = self.resource_probe()
        return {
            "phase": self.phase,
            "position": self.position,
            "last_good_offset": self.last_good_offset,
            "budget_limit": self.budget.limit,
            "budget_consumed": self.budget.consumed,
            "budget_remaining": self.budget.limit - self.budget.consumed,
            "successful_bytes": self.successful_bytes,
            "logical_request_count": self.logical_request_id,
            "request_attempt_count": self.request_attempt_count,
            "last_request": self.last_request,
            "peak_rss_bytes": safe_scalar_int(resources.get("peak_rss_bytes")),
            "peak_commit_bytes": safe_scalar_int(resources.get("peak_commit_bytes")),
        }

    def read(self, size: int) -> bytes:
        if self.failed:
            raise RuntimeError("TRANSPORT_TERMINAL")
        size = min(size, self.length - self.position, NETWORK_CHUNK)
        if size <= 0:
            return b""
        start = self.position
        end = start + size - 1
        self.logical_request_id += 1
        request_id = self.logical_request_id
        expected_content_range = f"bytes {start}-{end}/{self.length}"
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": self.user_agent,
                "Connection": "close",
            },
        )
        for attempt in range(1, self.maximum_attempts + 1):
            attempt_started = time.monotonic_ns()
            read_cap = size + 1
            try:
                self.budget.reserve(read_cap)
            except BudgetExceeded as error:
                self.failed = True
                row = self._base_row(
                    request_id, start, end, size, attempt, attempt_started
                )
                row.update(
                    {
                        "status": "TERMINAL_BUDGET_STOP_BEFORE_REQUEST",
                        "successful_bytes": 0,
                        "accounted_bytes": 0,
                        "error": {
                            "category": "BYTE_BUDGET_EXCEEDED",
                            "outer_type": "BudgetExceeded",
                            "reason_type": "BudgetExceeded",
                            "errno": None,
                            "http_status": None,
                            "exception_chain_sha256": hashlib.sha256(
                                b"BudgetExceeded|BYTE_BUDGET_EXCEEDED"
                            ).hexdigest(),
                        },
                    }
                )
                self.last_request = self.ledger.append(row)
                self._write_terminal("BYTE_BUDGET_EXCEEDED")
                raise

            self.request_attempt_count += 1
            status: int | None = None
            content_range: str | None = None
            received_bytes = read_cap
            body: bytes | None = None
            network_error: BaseException | None = None
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    status = safe_scalar_int(response.status)
                    content_range = safe_content_range(
                        response.headers.get("Content-Range")
                    )
                    body = response.read(read_cap)
                    received_bytes = len(body)
                self.budget.release(read_cap - received_bytes)
                if (
                    status != 206
                    or received_bytes != size
                    or content_range != expected_content_range
                ):
                    raise RangeContractError("RANGE_CONTRACT")
            except (
                HTTPError,
                URLError,
                OSError,
                http.client.IncompleteRead,
                RangeContractError,
            ) as error:
                network_error = error

            if network_error is None:
                assert body is not None
                row = self._base_row(
                    request_id, start, end, size, attempt, attempt_started
                )
                row.update(
                    {
                        "status": "PASS",
                        "http_status": status,
                        "content_range": content_range,
                        "successful_bytes": received_bytes,
                        "accounted_bytes": received_bytes,
                        "body_sha256": hashlib.sha256(body).hexdigest(),
                    }
                )
                self.successful_bytes += received_bytes
                self.last_good_offset = end
                self.position += received_bytes
                try:
                    self.last_request = self.ledger.append(row)
                    self._write_progress("RUNNING")
                except EvidencePersistenceFailure:
                    self.failed = True
                    raise
                return body

            classification = sanitized_error(network_error)
            final_attempt = attempt == self.maximum_attempts
            row = self._base_row(
                request_id, start, end, size, attempt, attempt_started
            )
            row.update(
                {
                    "status": (
                        "TERMINAL_TRANSPORT_FAILURE"
                        if final_attempt
                        else "RETRYABLE_FAILURE"
                    ),
                    "http_status": status,
                    "content_range": content_range,
                    "successful_bytes": 0,
                    "accounted_bytes": (
                        received_bytes
                        if isinstance(network_error, RangeContractError)
                        else read_cap
                    ),
                    "error": classification,
                }
            )
            try:
                self.last_request = self.ledger.append(row)
                if final_attempt:
                    self.failed = True
                    self._write_terminal(classification["category"])
                else:
                    self._write_progress("RETRYABLE_FAILURE")
            except EvidencePersistenceFailure:
                self.failed = True
                raise
            if final_attempt:
                raise TransportFailure(
                    classification["category"], self.snapshot()
                ) from network_error
            self.sleep(0.25 * attempt)
        raise AssertionError("UNREACHABLE")

    def _base_row(
        self,
        request_id: int,
        start: int,
        end: int,
        size: int,
        attempt: int,
        attempt_started: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": "r2_range_attempt.v1",
            "logical_request_id": request_id,
            "phase": self.phase,
            "start": start,
            "end": end,
            "requested_bytes": size,
            "attempt": attempt,
            "attempt_started_monotonic_ns": attempt_started,
            "attempt_ended_monotonic_ns": time.monotonic_ns(),
        }

    def _write_progress(self, status: str) -> None:
        write_json_atomic(
            self.progress_path,
            {
                "schema_version": "r2_transport_progress.v1",
                "status": status,
                "updated_at_utc": utc_now(),
                **self.snapshot(),
            },
            exclusive=False,
        )

    def _write_terminal(self, category: str) -> None:
        terminal = {
            "schema_version": "r2_transport_failure_receipt.v1",
            "decision": "NOT_EVALUABLE",
            "terminal_category": category,
            "failed_at_utc": utc_now(),
            **self.snapshot(),
        }
        self._write_progress("TERMINAL_NOT_EVALUABLE")
        write_json_atomic(self.failure_receipt_path, terminal, exclusive=True)
