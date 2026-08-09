#!/usr/bin/env python3
"""Validate and record explicit D1 Development quality-screen activation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN"
SHA_PATTERN = re.compile(r"^[0-9A-F]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def validate(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    require(protocol.get("schema") == "blindassist_depthart_task_preserving_d1_quality_protocol_v1",
            "protocol schema drift")
    require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol id drift")
    require(protocol.get("status") == "FROZEN_PRE_OUTCOME_EXPLICIT_ACTIVATION_REQUIRED", "protocol status drift")
    require(protocol.get("owning_protocol_id") == "DEPTHART_TASK_PRESERVING_D1_FIXED_MIXED_DEVELOPMENT_SCREEN",
            "owning D1 protocol drift")
    require(protocol.get("strict_g4d_terminal_immutable") ==
            "CURRENT_QAIRT_2_47_SM8650_HTP_STANDARD_FLOAT_PATH_STRICT_G4D_NOT_SUPPORTED",
            "strict G4-D terminal drift")
    for name, binding in protocol["bindings"].items():
        require(isinstance(binding.get("sha256"), str) and SHA_PATTERN.fullmatch(binding["sha256"]),
                f"invalid binding SHA: {name}")
        path = Path(binding["path"])
        if not path.is_absolute():
            path = root / path
        require(path.is_file() and sha256(path) == binding["sha256"], f"binding drift: {name}")
        if "bytes" in binding:
            require(path.stat().st_size == int(binding["bytes"]), f"binding byte drift: {name}")
    sessions = protocol["cohort"]["ordered_sessions"]
    require(len(sessions) == 8 and len({row["visit_id"] for row in sessions}) == 8 and
            len({row["video_id"] for row in sessions}) == 8, "exact eight disjoint sessions required")
    require(all(row["frame_count"] == 300 and SHA_PATTERN.fullmatch(row["frame_stems_sha256"])
                for row in sessions), "session frame-plan drift")
    require(protocol["execution"].get("chunk_size_frames") == 50 and
            protocol["execution"].get("chunks_per_session") == 6 and
            protocol["execution"].get("total_chunks") == 48,
            "fixed 48x50 chunk schedule drift")
    require(protocol["metric_semantics"]["known_coverage_denominator"] == "TRUTH_KNOWN_CELLS",
            "known coverage semantics drift")
    require(protocol["authority"]["r2_cohort_access"] is False and
            protocol["authority"]["performance_before_quality_pass"] is False,
            "authority firewall drift")
    return {"bindings": len(protocol["bindings"]), "sessions": 8, "frames": 2400,
            "chunks": 48, "cells": 21600}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-quote", required=True)
    args = parser.parse_args()
    root, protocol_path = args.repo_root.resolve(), args.protocol.resolve()
    require(args.user_quote.strip() == "激活 D1 quality screen", "activation quote must match the explicit user instruction")
    checks = validate(load_json(protocol_path), root)
    receipt = {
        "schema": "blindassist_depthart_task_preserving_d1_quality_activation_receipt_v1",
        "protocol_id": PROTOCOL_ID, "status": "OUTCOME_ACCESS_ACTIVATED",
        "activated_at": "2026-08-10", "user_quote": args.user_quote.strip(),
        "interpretation": "Authorize the first and only frozen 8x300 D1 Development task-quality screen.",
        "protocol": {"path": str(protocol_path), "bytes": protocol_path.stat().st_size,
                     "sha256": sha256(protocol_path)},
        "protocol_sha256": sha256(protocol_path), "checks": checks,
        "execution_authorized": True, "development_outcome_access": "ACTIVATED_NOT_YET_ACCESSED",
        "r2_cohort_access": "NONE", "performance_authorized": False,
        "strict_g4d_terminal_immutable": True,
        "authority": "D1_DEVELOPMENT_TASK_QUALITY_SCREEN_ONLY",
    }
    output = args.output.resolve()
    require(not output.exists() and not output.with_suffix(output.suffix + ".partial").exists(),
            f"activation receipt already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(json.dumps({"status": receipt["status"], "protocol_sha256": receipt["protocol_sha256"],
                      "receipt_sha256": sha256(output), "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
