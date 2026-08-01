#!/usr/bin/env python3
"""Create or append the explicit F:\\ba-data project source index."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.research.candidate_event_mining.pipeline import (
    ContractError,
    PROJECT_INDEX_SCHEMA,
    read_json,
    refuse_overwrite,
    validate_project_index,
    write_json,
)


def _new_index(data_root: str) -> dict[str, object]:
    return {
        "schema": PROJECT_INDEX_SCHEMA,
        "index_version": "r0",
        "project_id": "blindassist-candidate-event-mining",
        "data_root": data_root,
        "project_root": str(Path(data_root) / "blindassist-candidate-event-mining"),
        "sources": [],
        "index_rules": {
            "source_id_unique": True,
            "session_id_required": True,
            "source_url_required": True,
            "retrieved_at_utc_required": True,
            "content_sha256_required": True,
            "media_must_remain_under_data_root": True,
            "public_download_only": True,
            "authentication_bypass": False,
            "payment_bypass": False,
            "access_control_evasion": False,
        },
        "authority": {
            "isolated_internal_research": True,
            "event_truth": False,
            "training": False,
            "confirmation": False,
            "production": False,
            "redistribution": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.resolve()
    if output.exists():
        index = validate_project_index(read_json(output))
        if args.source_record is None:
            raise ContractError(f"project index already exists: {output}")
    else:
        if args.source_record is not None and not output.parent.exists():
            output.parent.mkdir(parents=True)
        index = _new_index(args.data_root)
    if args.source_record is not None:
        record = read_json(args.source_record.resolve())
        for key in ("source_id", "session_id", "media_path", "source_url", "retrieved_at_utc", "content_sha256"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ContractError(f"source record requires {key}")
        if (
            record["content_sha256"] != record["content_sha256"].lower()
            or len(record["content_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in record["content_sha256"])
        ):
            raise ContractError("source record content_sha256 must be lowercase SHA-256")
        if record.get("retrieval_status", "declared") not in {"declared", "downloaded", "verified", "not_available"}:
            raise ContractError("invalid source record retrieval_status")
        existing = {item["source_id"] for item in index["sources"]}
        if record["source_id"] in existing:
            raise ContractError(f"source_id already indexed: {record['source_id']}")
        index["sources"].append(record)
        index["sources"].sort(key=lambda item: item["source_id"])
        index["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    validate_project_index(index)
    if output.exists() and args.source_record is None:
        raise ContractError(f"refusing to overwrite existing output: {output}")
    if output.exists() and args.source_record is not None:
        # An append is an intentional index mutation and remains fully revalidated above.
        pass
    else:
        refuse_overwrite(output)
    write_json(output, index)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(r"F:\ba-data\blindassist-candidate-event-mining\project_index.json"))
    parser.add_argument("--data-root", default=r"F:\ba-data")
    parser.add_argument("--source-record", type=Path)
    return parser.parse_args()


def main() -> int:
    try:
        index = run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "source_count": len(index["sources"]), "project_root": index["project_root"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
