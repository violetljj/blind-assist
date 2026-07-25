"""Fetch only frozen JRDB labels and audit far-range denominators.

No PCD payload, in-box point count, support class, or centroid result is read.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import math
import os
import struct
import tempfile
import urllib.request
import zlib
from collections import Counter
from pathlib import Path
from typing import Any


LEDGER_SCHEMA = (
    "blindassist_ustrf_jrdb_person_3d_trajectory_far_range_denominator_"
    "adequacy_metadata_blind_replication_r0_denominator_ledger"
)
RECEIPT_SCHEMA = LEDGER_SCHEMA.replace("_ledger", "_receipt")


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: Any) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(value))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class RangeReader:
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.bytes_read = 0
        self.requests: list[dict[str, Any]] = []

    def get(self, url: str, start: int, end: int) -> bytes:
        require(0 <= start <= end, "invalid range")
        requested = end - start + 1
        require(self.bytes_read + requested <= self.budget, "network budget exceeded")
        request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "BlindAssist-source-fetch/1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            require(response.status == 206, f"range status {response.status}")
            content_range = response.headers.get("Content-Range", "")
            require(content_range.startswith(f"bytes {start}-{end}/"), "content-range drift")
            payload = response.read(requested + 1)
        require(len(payload) == requested, "range length drift")
        self.bytes_read += requested
        self.requests.append({"start": start, "end": end, "bytes": requested})
        return payload


def fetch_member(reader: RangeReader, url: str, member: dict[str, Any]) -> bytes:
    offset = int(member["local_header_offset"])
    header = reader.get(url, offset, offset + 29)
    values = struct.unpack("<4s5H3L2H", header)
    require(values[0] == b"PK\x03\x04", "local header signature")
    name_len, extra_len = values[-2], values[-1]
    tail = reader.get(url, offset + 30, offset + 30 + name_len + extra_len - 1)
    require(tail[:name_len].decode("utf-8") == member["name"], "local name drift")
    data_start = offset + 30 + name_len + extra_len
    compressed_size = int(member["compressed_size"])
    compressed = reader.get(url, data_start, data_start + compressed_size - 1)
    method = int(member["compression_method"])
    raw = compressed if method == 0 else zlib.decompress(compressed, -15)
    require(method in (0, 8), f"unsupported compression method {method}")
    require(len(raw) == int(member["uncompressed_size"]), "uncompressed size drift")
    require(
        binascii.crc32(raw) & 0xFFFFFFFF == int(member["crc32"], 16),
        "member CRC drift",
    )
    return raw


def load_or_fetch(
    reader: RangeReader,
    url: str,
    payload_root: Path,
    member: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    destination = payload_root.joinpath(*Path(member["name"]).parts)
    if destination.is_file():
        raw = destination.read_bytes()
        require(len(raw) == int(member["uncompressed_size"]), "cached member size drift")
        require(
            binascii.crc32(raw) & 0xFFFFFFFF == int(member["crc32"], 16),
            "cached member CRC drift",
        )
        mode = "REUSED_HASHED_PAYLOAD"
    else:
        require(not destination.exists(), "payload destination collision")
        raw = fetch_member(reader, url, member)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        mode = "RANGE_FETCHED_AFTER_FREEZE"
    return json.loads(raw), {
        "member": member["name"],
        "path": destination.as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mode": mode,
    }


def valid_box(item: dict[str, Any]) -> tuple[dict[str, float] | None, str | None]:
    source = item.get("box", {})
    keys = ("cx", "cy", "cz", "w", "l", "h", "rot_z")
    try:
        box = {key: float(source[key]) for key in keys}
    except (KeyError, TypeError, ValueError):
        return None, "invalid_3d_box"
    if not all(math.isfinite(value) for value in box.values()) or min(
        box["w"], box["l"], box["h"]
    ) <= 0:
        return None, "invalid_3d_box"
    return box, None


def range_band(distance: float) -> str:
    if distance < 10:
        return "0-10"
    if distance < 20:
        return "10-20"
    if distance < 40:
        return "20-40"
    return "40-plus"


def build(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    freeze_path = repo / config["outputs"]["sequence_freeze"]
    require(freeze_path.is_file(), "metadata freeze missing")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    require(
        freeze["status"] == "FROZEN_BEFORE_CANDIDATE_LABEL_OR_PCD_PAYLOAD",
        "freeze status drift",
    )
    require(freeze["config_sha256"] == sha256_file(config_path), "freeze config drift")
    require(freeze["blindness"]["candidate_label_payload_read"] is False, "freeze blindness drift")
    gate = config["denominator_adequacy"]
    payload_root = repo / config["outputs"]["label_payload_root"]
    reader = RangeReader(512 * 1024 * 1024)
    url = config["remote_archives"]["labels"]["url"]
    per_sequences = []
    payloads = []
    for selected in freeze["selected"]:
        sequence = selected["sequence"]
        members = {Path(row["name"]).parent.name: row for row in selected["label_members"]}
        labels_2d, receipt_2d = load_or_fetch(
            reader, url, payload_root, members["labels_2d_stitched"]
        )
        labels_3d, receipt_3d = load_or_fetch(reader, url, payload_root, members["labels_3d"])
        payloads.extend((receipt_2d, receipt_3d))
        first = int(selected["frame_first_stem"])
        stems = [f"{value:06d}" for value in range(first, first + selected["frame_count"])]
        require(stems[-1] == selected["frame_last_stem"], f"frozen stem continuity {sequence}")
        counters: Counter[str] = Counter()
        range_counts: Counter[str] = Counter()
        cross_modal: Counter[str] = Counter()
        occlusion: Counter[str] = Counter()
        invalid_reasons: Counter[str] = Counter()
        for stem in stems:
            objects_2d = labels_2d["labels"][f"{stem}.jpg"]
            objects_3d = labels_3d["labels"][f"{stem}.pcd"]
            index_2d = {item["label_id"]: item for item in objects_2d}
            index_3d = {item["label_id"]: item for item in objects_3d}
            require(len(index_2d) == len(objects_2d), f"duplicate 2d label {sequence}:{stem}")
            require(len(index_3d) == len(objects_3d), f"duplicate 3d label {sequence}:{stem}")
            counters["frame"] += 1
            for label_id in sorted(set(index_2d) | set(index_3d)):
                counters["union_object_frame"] += 1
                item_2d = index_2d.get(label_id)
                item_3d = index_3d.get(label_id)
                if item_3d is None:
                    counters["2d_only_object_frame"] += 1
                    continue
                box, reason = valid_box(item_3d)
                if box is None:
                    counters["invalid_3d_object_frame"] += 1
                    invalid_reasons[reason or "invalid_3d_box"] += 1
                    continue
                counters["valid_3d_object_frame"] += 1
                presence = "3d-and-2d" if item_2d is not None else "3d-only"
                cross_modal[presence] += 1
                occlusion[
                    (item_2d or {}).get("attributes", {}).get("occlusion") or "Unknown"
                ] += 1
                distance = math.sqrt(box["cx"] ** 2 + box["cy"] ** 2 + box["cz"] ** 2)
                range_counts[range_band(distance)] += 1
        far = range_counts["40-plus"]
        near = range_counts["0-10"] + range_counts["10-20"]
        adequate = (
            far >= int(gate["minimum_40_plus_object_frames_per_sequence"])
            and near >= int(gate["minimum_0_20_object_frames_per_sequence"])
        )
        per_sequences.append(
            {
                "sequence": sequence,
                "window_first_position": selected["window_first_position"],
                "window_last_position": selected["window_last_position"],
                "frame_first_stem": selected["frame_first_stem"],
                "frame_last_stem": selected["frame_last_stem"],
                "denominators": dict(sorted(counters.items())),
                "range_band_valid_3d_object_frames": dict(sorted(range_counts.items())),
                "cross_modal_valid_3d_object_frames": dict(sorted(cross_modal.items())),
                "occlusion_valid_3d_object_frames": dict(sorted(occlusion.items())),
                "invalid_reasons": dict(sorted(invalid_reasons.items())),
                "far_40_plus_object_frames": far,
                "near_0_20_object_frames": near,
                "adequate_for_far_direction": adequate,
            }
        )
    adequate_sequences = [
        row["sequence"] for row in per_sequences if row["adequate_for_far_direction"]
    ]
    status = (
        "ADEQUATE_FOR_FROZEN_PCD_SUPPORT"
        if len(adequate_sequences) >= int(gate["minimum_adequate_sequences"])
        else "DENOMINATOR_INSUFFICIENT"
    )
    ledger = {
        "schema": LEDGER_SCHEMA,
        "stage": config["stage"],
        "status": "COMPLETE",
        "config_sha256": sha256_file(config_path),
        "sequence_freeze_sha256": sha256_file(freeze_path),
        "audit_boundary": {
            "label_payload_read": True,
            "pcd_payload_read": False,
            "in_box_point_or_support_result_read": False,
            "centroid_or_motion_residual_read": False,
        },
        "gate": gate,
        "per_sequences": per_sequences,
        "adequate_sequences": adequate_sequences,
        "adequate_sequence_count": len(adequate_sequences),
        "selected_sequence_count": len(per_sequences),
        "payloads": payloads,
        "network": {
            "bytes_read": reader.bytes_read,
            "requests": reader.requests,
        },
        "authority": config["authority"],
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "stage": config["stage"],
        "status": "COMPLETE",
        "terminal_state": status,
        "validity": "PENDING_INDEPENDENT_VALIDATION",
        "config_sha256": sha256_file(config_path),
        "sequence_freeze_sha256": sha256_file(freeze_path),
        "denominator_ledger_sha256": hashlib.sha256(canonical_bytes(ledger)).hexdigest(),
        "adequate_sequences": adequate_sequences,
        "adequate_sequence_count": len(adequate_sequences),
        "pcd_support_authorized": status == "ADEQUATE_FOR_FROZEN_PCD_SUPPORT",
        "failure_action": gate["failure_action"] if status == "DENOMINATOR_INSUFFICIENT" else None,
        "authority": config["authority"],
    }
    return ledger, receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ledger, receipt = build(repo, config_path)
    atomic_write(repo / config["outputs"]["denominator_ledger"], ledger)
    atomic_write(repo / config["outputs"]["denominator_receipt"], receipt)
    print(
        json.dumps(
            {
                "terminal_state": receipt["terminal_state"],
                "adequate_sequence_count": receipt["adequate_sequence_count"],
                "adequate_sequences": receipt["adequate_sequences"],
                "network_bytes": ledger["network"]["bytes_read"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
