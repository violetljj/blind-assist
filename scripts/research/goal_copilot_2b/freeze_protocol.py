"""Validate and create-once seal the GC2-B design without authorizing search."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PROTOCOL = HERE / "protocol.json"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate(protocol: dict[str, Any], closeout: dict[str, Any], closeout_path: Path) -> None:
    predecessor = protocol["predecessor"]
    if sha256(closeout_path) != predecessor["closeout_sha256"]:
        raise RuntimeError("GC2-A closeout digest mismatch")
    if closeout["status"] != predecessor["required_status"]:
        raise RuntimeError("GC2-A terminal status mismatch")
    if closeout["protocol_seal_digest"] != predecessor["protocol_seal_digest"]:
        raise RuntimeError("GC2-A protocol seal mismatch")
    if closeout["gc2b_admission"] is not predecessor["required_admission"]:
        raise RuntimeError("GC2-A admission mismatch")
    if closeout["next_authorized_route"] != predecessor["required_route"]:
        raise RuntimeError("GC2-A route mismatch")
    if closeout["gc2b_model_calls_authorized"] is not False:
        raise RuntimeError("predecessor unexpectedly authorizes model calls")
    authorized = protocol["currently_authorized"]
    if authorized["model_calls"] or authorized["sky_or_evox_search"]:
        raise RuntimeError("design seal cannot authorize search or model calls")
    budget = protocol["search_budget"]
    if budget["generation_attempts_total"] != (
        len(budget["replicates"]) * budget["generation_attempts_per_replicate"]
    ):
        raise RuntimeError("search budget arithmetic mismatch")


def freeze(closeout_path: Path, sky_commit: str, output_root: Path) -> Path:
    if output_root.exists():
        raise FileExistsError(f"GC2-B design root already exists: {output_root}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    validate(protocol, closeout, closeout_path)
    payload = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "GOAL_COPILOT_2B_PROTOCOL_DESIGN_FROZEN_SEARCH_NOT_AUTHORIZED",
        "blindassist_commit": git_head(),
        "skydiscover_commit": sky_commit,
        "protocol_sha256": sha256(PROTOCOL),
        "gc2a_closeout_sha256": sha256(closeout_path),
        "frozen_winner_sha256": protocol["frozen_starting_policy"]["sha256"],
        "generation_attempts_total_if_later_authorized": protocol["search_budget"]["generation_attempts_total"],
        "model_calls_authorized": False,
        "search_authorized": False,
        "heldout_materialized": False,
        "claim_ceiling": protocol["claim_ceiling"],
    }
    payload["design_seal_digest"] = hashlib.sha256(canonical(payload)).hexdigest()
    output_root.mkdir(parents=True)
    path = output_root / "formal_design_seal.json"
    with path.open("xb") as stream:
        stream.write(canonical(payload))
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gc2a-closeout", type=Path, required=True)
    parser.add_argument("--sky-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.gc2a_closeout.resolve(), args.sky_commit, args.output_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
