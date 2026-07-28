from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import traceback
from typing import Any, Iterable


SAFE_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_evidence_bindings(
    repo: Path,
    namespace: Path,
) -> list[dict[str, Any]]:
    names = {
        "range_ledger.jsonl",
        "stage_ledger.jsonl",
        "image_metadata_ledger.jsonl",
        "pairing_diagnostic_ledger.jsonl",
        "target_chunk_plan.json",
    }
    return [
        {
            "path": path.relative_to(repo).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(namespace.rglob("*"))
        if path.is_file() and path.name in names
    ]


def classify_exception(error: BaseException) -> dict[str, Any]:
    raw = str(error)
    code = (
        raw
        if SAFE_CODE.fullmatch(raw)
        else f"UNCLASSIFIED_{type(error).__name__.upper()}"
    )
    frames = traceback.extract_tb(error.__traceback__)
    site = None
    if frames:
        last = frames[-1]
        site = {
            "file_name": Path(last.filename).name,
            "line": last.lineno,
            "function": last.name,
        }
    result: dict[str, Any] = {
        "error_type": type(error).__name__,
        "error_code": code,
        "site": site,
    }
    errno = getattr(error, "errno", None)
    if isinstance(errno, int):
        result["errno"] = errno
    return result


class HashLedger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        self._stream = os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        )
        self.path = path
        self.rows = 0
        self.previous: str | None = None

    def _append(self, payload: dict[str, Any]) -> None:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload["row_sha256"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        self._stream.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self.previous = payload["row_sha256"]
        self.rows += 1

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()


class StageLedger(HashLedger):
    def append(
        self,
        *,
        stage: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._append(
            {
                "sequence": self.rows,
                "previous_row_sha256": self.previous,
                "stage": stage,
                "status": status,
                "details": details or {},
            }
        )

    def binding(self, repo: Path) -> dict[str, Any]:
        self.close()
        return {
            "path": self.path.relative_to(repo).as_posix(),
            "sha256": sha256_file(self.path),
            "rows": self.rows,
            "head_sha256": self.previous,
        }


class ImageMetadataLedger(HashLedger):
    def append(
        self,
        message: Any,
        *,
        chunk_index: int,
        chunk_pos: int,
        record_ordinal: int,
        record_offset: int,
        serialized_bytes: int,
    ) -> None:
        self._append(
            {
                "sequence_index": self.rows,
                "previous_row_sha256": self.previous,
                "chunk_index": chunk_index,
                "chunk_pos": chunk_pos,
                "record_ordinal": record_ordinal,
                "record_offset": record_offset,
                "header_timestamp_ns": message.header_timestamp_ns,
                "bag_timestamp_ns": message.bag_timestamp_ns,
                "sensor_sequence": message.sequence,
                "frame_id": message.frame_id,
                "encoding": message.encoding,
                "height": message.height,
                "width": message.width,
                "step": message.step,
                "is_bigendian": message.is_bigendian,
                "serialized_bytes": serialized_bytes,
                "payload_bytes": len(message.payload),
                "serialized_sha256": message.serialized_sha256,
                "payload_sha256": message.payload_sha256,
            }
        )

    def binding(self, repo: Path) -> dict[str, Any]:
        self.close()
        return {
            "path": self.path.relative_to(repo).as_posix(),
            "sha256": sha256_file(self.path),
            "rows": self.rows,
            "head_sha256": self.previous,
            "pixel_payload_materialized": False,
        }


def pairing_diagnostics(
    messages: Iterable[Any],
    *,
    geometry_timestamps_ns: list[int],
    maximum_delta_ns: int,
) -> dict[str, Any]:
    ordered = sorted(messages, key=lambda row: row.header_timestamp_ns)
    timestamps = [row.header_timestamp_ns for row in ordered]
    counts = Counter(timestamps)
    used: set[int] = set()
    selected_indices: list[int] = []
    ties = 0
    reuses = 0
    outside_tolerance = 0
    maximum_nearest_delta_ns = 0
    for geometry_timestamp_ns in geometry_timestamps_ns:
        distances = [
            abs(timestamp_ns - geometry_timestamp_ns)
            for timestamp_ns in timestamps
        ]
        if not distances:
            outside_tolerance += 1
            continue
        minimum = min(distances)
        maximum_nearest_delta_ns = max(maximum_nearest_delta_ns, minimum)
        nearest = [
            index
            for index, distance in enumerate(distances)
            if distance == minimum
        ]
        if minimum > maximum_delta_ns:
            outside_tolerance += 1
        elif len(nearest) != 1:
            ties += 1
        elif nearest[0] in used:
            reuses += 1
        else:
            used.add(nearest[0])
            selected_indices.append(nearest[0])
    guards_evaluable = (
        len(selected_indices) == len(geometry_timestamps_ns)
        and bool(selected_indices)
    )
    return {
        "image_count": len(ordered),
        "geometry_count": len(geometry_timestamps_ns),
        "image_timestamp_min_ns": min(timestamps) if timestamps else None,
        "image_timestamp_max_ns": max(timestamps) if timestamps else None,
        "geometry_timestamp_min_ns": (
            min(geometry_timestamps_ns) if geometry_timestamps_ns else None
        ),
        "geometry_timestamp_max_ns": (
            max(geometry_timestamps_ns) if geometry_timestamps_ns else None
        ),
        "duplicate_image_timestamp_count": sum(
            count - 1 for count in counts.values() if count > 1
        ),
        "image_timestamps_strict": all(
            left < right for left, right in zip(timestamps, timestamps[1:])
        ),
        "nearest_tie_count": ties,
        "nearest_reuse_count": reuses,
        "outside_tolerance_count": outside_tolerance,
        "maximum_nearest_delta_ns": maximum_nearest_delta_ns,
        "guard_before_available": (
            min(selected_indices) > 0 if guards_evaluable else False
        ),
        "guard_after_available": (
            max(selected_indices) + 1 < len(ordered)
            if guards_evaluable
            else False
        ),
    }


def write_pairing_diagnostic_ledger(
    path: Path,
    messages: Iterable[Any],
    *,
    geometry_timestamps_ns: list[int],
    maximum_delta_ns: int,
) -> dict[str, Any]:
    ordered = sorted(messages, key=lambda row: row.header_timestamp_ns)
    ledger = HashLedger(path)
    used: set[int] = set()
    for geometry_index, geometry_timestamp_ns in enumerate(
        geometry_timestamps_ns
    ):
        candidates = sorted(
            (
                abs(message.header_timestamp_ns - geometry_timestamp_ns),
                index,
                message.header_timestamp_ns,
            )
            for index, message in enumerate(ordered)
        )
        minimum = candidates[0][0] if candidates else None
        nearest = (
            [row for row in candidates if row[0] == minimum]
            if minimum is not None
            else []
        )
        unique_index = nearest[0][1] if len(nearest) == 1 else None
        reused = unique_index in used if unique_index is not None else False
        if (
            unique_index is not None
            and not reused
            and minimum is not None
            and minimum <= maximum_delta_ns
        ):
            used.add(unique_index)
        ledger._append(
            {
                "sequence_index": ledger.rows,
                "previous_row_sha256": ledger.previous,
                "geometry_index": geometry_index,
                "geometry_timestamp_ns": geometry_timestamp_ns,
                "minimum_delta_ns": minimum,
                "within_tolerance": (
                    minimum is not None and minimum <= maximum_delta_ns
                ),
                "nearest_candidate_count": len(nearest),
                "nearest_candidate_indices": [row[1] for row in nearest],
                "nearest_candidate_timestamps_ns": [
                    row[2] for row in nearest
                ],
                "unique_nearest_index": unique_index,
                "nearest_reused": reused,
            }
        )
    ledger.close()
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "rows": ledger.rows,
        "head_sha256": ledger.previous,
        "pixel_payload_materialized": False,
    }
