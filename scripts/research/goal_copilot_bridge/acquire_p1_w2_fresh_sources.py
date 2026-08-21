#!/usr/bin/env python3
"""Acquire the frozen P1-W2 ADT RGB/GT parents without running any model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


MANIFEST_URL = "https://explorer.projectaria.com/data/adt/download_links"
USER_AGENT = "BlindAssist-P1-W2-Materialization/1.0"
MAX_TOTAL_BYTES = 1_400_000_000


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fetch_manifest() -> dict[str, Any]:
    request = urllib.request.Request(MANIFEST_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"ADT manifest request failed: HTTP {response.status}")
        return json.load(response)


def acquire(member: dict[str, Any], target: Path) -> str:
    expected_size = int(member["file_size_bytes"])
    expected_sha1 = str(member["sha1sum"])
    if target.is_file():
        if target.stat().st_size != expected_size or digest(target, "sha1") != expected_sha1:
            raise RuntimeError(f"existing payload identity mismatch: {target}")
        return "reused_verified"

    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise RuntimeError(f"partial payload exceeds frozen size: {partial}")
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(str(member["download_url"]), headers=headers)
    mode = "ab" if offset else "xb"
    with urllib.request.urlopen(request, timeout=180) as response:
        if offset and response.status != 206:
            raise RuntimeError(f"server refused safe resume for {target.name}: HTTP {response.status}")
        with partial.open(mode) as out:
            written = offset
            while chunk := response.read(4 * 1024 * 1024):
                written += len(chunk)
                if written > expected_size:
                    raise RuntimeError(f"download exceeded frozen size: {target.name}")
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"download size mismatch for {target.name}: {partial.stat().st_size} != {expected_size}"
        )
    if digest(partial, "sha1") != expected_sha1:
        raise RuntimeError(f"download SHA-1 mismatch: {target.name}")
    partial.replace(target)
    return "resumed_verified" if offset else "downloaded_verified"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    freeze_bytes = args.freeze.read_bytes()
    freeze = json.loads(freeze_bytes)
    fresh = freeze["data"]["fresh_proxy"]
    if fresh["payload_downloaded"] or fresh["target_candidate_roster_frozen"]:
        raise RuntimeError("freeze does not authorize fresh materialization")
    if freeze["future_execution_budget"]["execution_authorized"]:
        raise RuntimeError("freeze unexpectedly authorizes model execution")
    parents = fresh["parents"]
    if len(parents) != 8 or len({row["sequence_id"] for row in parents}) != 8:
        raise RuntimeError("frozen parent roster identity is invalid")
    total_expected = sum(int(row[key][2]) for row in parents for key in ("rgb", "groundtruth"))
    if total_expected > MAX_TOTAL_BYTES:
        raise RuntimeError(f"frozen payload exceeds bounded budget: {total_expected}")

    live = fetch_manifest()["sequences"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for parent in parents:
        sequence_id = parent["sequence_id"]
        sequence = live.get(sequence_id)
        if sequence is None:
            raise RuntimeError(f"frozen ADT sequence absent from live manifest: {sequence_id}")
        for role, frozen_key, live_key in (
            ("RGB_PROVIDER_INPUT", "rgb", "video_main_rgb"),
            ("GT_EVALUATOR_ONLY", "groundtruth", "main_groundtruth"),
        ):
            frozen_member = parent[frozen_key]
            member = sequence[live_key]
            expected = (str(member["filename"]), str(member["sha1sum"]), int(member["file_size_bytes"]))
            if expected != (str(frozen_member[0]), str(frozen_member[1]), int(frozen_member[2])):
                raise RuntimeError(f"live ADT manifest drift: {sequence_id} {live_key}")
            filename = expected[0]
            if Path(filename).name != filename:
                raise RuntimeError(f"unsafe ADT filename: {filename}")
            target = args.output_dir / filename
            action = acquire(member, target)
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "stratum": parent["stratum"],
                    "role": role,
                    "manifest_key": live_key,
                    "path": target.resolve().as_posix(),
                    "filename": filename,
                    "bytes": target.stat().st_size,
                    "sha1": digest(target, "sha1"),
                    "sha256": digest(target, "sha256"),
                    "action": action,
                }
            )
            print(json.dumps({"sequence_id": sequence_id, "role": role, "action": action}))

    receipt = {
        "schema_version": "p1_w2_fresh_source_acquisition_v1",
        "protocol": "P1_W2_FRESH_SOURCE_MATERIALIZATION_AND_PRIVATE_ROSTER_FREEZE",
        "freeze_path": args.freeze.resolve().as_posix(),
        "freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "selection_identity_sha256": fresh["selection_identity_sha256"],
        "parent_count": 8,
        "member_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "members": rows,
        "firewall": {
            "rgb_provider_input_count": sum(row["role"] == "RGB_PROVIDER_INPUT" for row in rows),
            "gt_evaluator_only_count": sum(row["role"] == "GT_EVALUATOR_ONLY" for row in rows),
            "gt_may_enter_model_provider": False,
        },
        "model_matcher_identity_call_counts": {"model": 0, "matcher": 0, "identity": 0},
        "terminal": "P1_W2_FRESH_SOURCE_PAYLOAD_ACQUIRED",
        "execution_authorized": False,
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps({"terminal": receipt["terminal"], "total_bytes": receipt["total_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
