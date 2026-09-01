#!/usr/bin/env python3
"""Download and screen one pre-frozen continuation row from the source queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_discriminative_view_source_queue as source  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-discriminative-view-source-queue-continue-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-discriminative-view-source-queue-result-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    base_protocol_row = protocol["base_protocol"]
    base_protocol_path = HERE / base_protocol_row["path"]
    pixel.require(pixel.sha256(base_protocol_path) == base_protocol_row["sha256"], "BASE_PROTOCOL_HASH")
    base_protocol = pixel.load_json(base_protocol_path)
    pixel.require(base_protocol["schema"] == base_protocol_row["required_schema"], "BASE_PROTOCOL_SCHEMA")
    for dependency in base_protocol["dependencies"]:
        pixel.require(
            pixel.sha256(HERE / dependency["path"]) == dependency["sha256"],
            f"BASE_DEPENDENCY_HASH:{dependency['path']}",
        )
    queue_row = protocol["queue"]
    queue_path = HERE / queue_row["path"]
    pixel.require(pixel.sha256(queue_path) == queue_row["sha256"], "QUEUE_HASH")
    queue = pixel.load_json(queue_path)
    pixel.require(queue["schema"] == queue_row["required_schema"], "QUEUE_SCHEMA")
    predecessor_row = protocol["predecessor"]
    predecessor_path = HERE / predecessor_row["path"]
    pixel.require(pixel.sha256(predecessor_path) == predecessor_row["sha256"], "PREDECESSOR_HASH")
    predecessor = pixel.load_json(predecessor_path)
    pixel.require(predecessor["schema"] == predecessor_row["required_schema"], "PREDECESSOR_SCHEMA")
    pixel.require(
        int(predecessor["selected_queue_index"]) == int(protocol["predecessor_selected_queue_index"]),
        "PREDECESSOR_QUEUE_INDEX",
    )
    wanted_index = int(protocol["selected_queue_index"])
    matches = [row for row in queue["ordered_candidates"] if int(row["queue_index"]) == wanted_index]
    pixel.require(len(matches) == 1, "QUEUE_INDEX_NOT_UNIQUE")
    queued = matches[0]
    pixel.require(
        wanted_index > int(protocol["predecessor_selected_queue_index"]),
        "NOT_A_CONTINUATION_ROW",
    )

    artifact_root = ROOT / base_protocol["source"]["artifact_root"]
    data_root = artifact_root / "datasets/3rscan"
    files: list[dict[str, Any]] = []
    template = base_protocol["source"]["official_url_template"]
    for requested in queued["download_plan"]:
        scan_id = str(requested["scan_id"])
        name = str(requested["file"])
        destination = data_root / scan_id / name
        url = template.format(scan_id=scan_id, file=name)
        transfer = source._download(url, destination)
        files.append(source._manifest(destination, artifact_root, transfer))
    screened = source._screen(queued["candidate"], data_root, base_protocol)
    selected = {
        "queue_index": wanted_index,
        "candidate": queued["candidate"],
        "source_files": files,
        **screened,
    }
    evaluable = bool(screened["source_evaluable"])
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "FRESH_PRE_FROZEN_QUEUE_CONTINUATION_SOURCE_SCREEN",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "queue": queue_row,
        "base_protocol": base_protocol_row,
        "selected_queue_index": wanted_index,
        "selected_source": selected,
        "source_evaluable": evaluable,
        "rows": [selected],
        "rgb_members_opened": 0,
        "model_calls": 0,
        "conclusion": (
            "L10_3RSCAN_DISCRIMINATIVE_VIEW_SOURCE_QUEUE_EVALUABLE"
            if evaluable
            else "L10_3RSCAN_DISCRIMINATIVE_VIEW_SOURCE_QUEUE_NOT_EVALUABLE"
        ),
        "next_action": protocol["next_action"] if evaluable else protocol["fallback_action"],
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
