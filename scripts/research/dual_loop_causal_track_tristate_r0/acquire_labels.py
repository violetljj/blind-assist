from __future__ import annotations

import argparse
import binascii
import datetime as dt
import hashlib
import json
import struct
import urllib.request
import zlib
from pathlib import Path
from typing import Any

from .common import PROTOCOL_ID, read_json, sha256_file, write_exclusive


LABELS_URL = "https://jrdb.erc.monash.edu/static/downloads/train_labels.zip"


def _range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "BlindAssist-source-fetch/1",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"range request returned {response.status}")
        payload = response.read(end - start + 2)
    if len(payload) != end - start + 1:
        raise RuntimeError("range response length drift")
    return payload


def _fetch_member(member: dict[str, Any]) -> bytes:
    offset = int(member["local_header_offset"])
    header = _range(LABELS_URL, offset, offset + 29)
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError("local ZIP header signature drift")
    name_length, extra_length = fields[-2], fields[-1]
    tail = _range(
        LABELS_URL,
        offset + 30,
        offset + 30 + name_length + extra_length - 1,
    )
    if tail[:name_length].decode("utf-8") != member["name"]:
        raise RuntimeError("local ZIP member name drift")
    data_start = offset + 30 + name_length + extra_length
    compressed = _range(
        LABELS_URL,
        data_start,
        data_start + int(member["compressed_size"]) - 1,
    )
    method = int(member["compression_method"])
    if method not in (0, 8):
        raise RuntimeError(f"unsupported ZIP compression method {method}")
    raw = compressed if method == 0 else zlib.decompress(compressed, -15)
    if len(raw) != int(member["uncompressed_size"]):
        raise RuntimeError("uncompressed size drift")
    if f"{binascii.crc32(raw) & 0xFFFFFFFF:08x}" != str(member["crc32"]):
        raise RuntimeError("member CRC drift")
    json.loads(raw)
    return raw


def run(
    freeze_path: Path,
    role: str,
    output_root: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    if role not in {"source_2d", "truth_3d"}:
        raise ValueError("role must be source_2d or truth_3d")
    freeze = read_json(freeze_path)
    if freeze.get("status") != "FROZEN_BEFORE_SELECTED_LABEL_PAYLOAD_ACCESS":
        raise ValueError("confirmation freeze status drift")
    records = []
    for selected in freeze["selected"]:
        sequence = str(selected["sequence"])
        member = selected["members"][role]
        destination = output_root / role / f"{sequence}.json"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        raw = _fetch_member(member)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        records.append(
            {
                "sequence": sequence,
                "member": member["name"],
                "path": destination.as_posix(),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    receipt = {
        "schema": "blindassist.dual_loop_causal_track_tristate_acquisition.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "role": role,
        "freeze_sha256": sha256_file(freeze_path),
        "source_url": LABELS_URL,
        "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "records": records,
        "other_role_opened_by_this_command": False,
    }
    write_exclusive(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--role", choices=("source_2d", "truth_3d"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.freeze, args.role, args.output_root, args.receipt),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
