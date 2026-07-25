#!/usr/bin/env python3
"""Snapshot the official THÖR source records used by the R1 authority audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "blindassist_ustrf_thor_source_native_id_time_transform_authority_recovery_r1_config"
ACQUISITION_SCHEMA = "blindassist_ustrf_thor_source_native_id_time_transform_authority_recovery_r1_acquisition"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "BlindAssist THOR source-authority audit/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def acquire(repo: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("config_identity")
    if config.get("status") != "frozen_before_source_audit":
        raise RuntimeError("config_not_frozen")
    snapshot_dir = repo / config["outputs"]["source_snapshot_dir"]
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for source in config["official_sources"]:
        if "record_api" in source:
            destination = snapshot_dir / f"zenodo-{source['record_id']}.json"
            payload = fetch(source["record_api"])
            parsed = json.loads(payload.decode("utf-8"))
            if int(parsed["id"]) != int(source["record_id"]):
                raise RuntimeError(f"record_identity:{source['record_id']}")
            payload = (
                json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
        elif source.get("kind") == "pdf":
            destination = snapshot_dir / "rudenko_ral2020_thor.pdf"
            payload = fetch(source["url"])
            if not payload.startswith(b"%PDF-"):
                raise RuntimeError("paper_not_pdf")
        else:
            destination = snapshot_dir / f"{source['source_id']}.html"
            payload = fetch(source["url"])
            if b"<html" not in payload.lower():
                raise RuntimeError(f"source_not_html:{source['source_id']}")
        destination.write_bytes(payload)
        artifacts.append(
            {
                "source_id": source["source_id"],
                "url": source.get("record_api", source.get("url")),
                "path": destination.relative_to(repo).as_posix(),
                "bytes": len(payload),
                "sha256": sha256_file(destination),
            }
        )
    return {
        "schema": ACQUISITION_SCHEMA,
        "stage": config["stage"],
        "status": "OFFICIAL_SOURCE_SNAPSHOTS_HASH_BOUND",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "config_sha256": sha256_file(config_path),
        "candidate_outputs_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    result = acquire(repo, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = repo / config["outputs"]["acquisition"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
