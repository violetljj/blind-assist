#!/usr/bin/env python3
"""Download and screen a pre-frozen active-view source queue in order."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_temporal_scale_vacancy_confirmation_source as old  # noqa: E402
import l10_3rscan_discriminative_view_confirmation_source as active  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-discriminative-view-source-queue-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-discriminative-view-source-queue-result-v1"


def _validate_download(path: Path) -> None:
    if path.name.endswith(".json"):
        with path.open("r", encoding="utf-8") as stream:
            json.load(stream)
    elif path.name == "sequence.zip":
        with zipfile.ZipFile(path) as archive:
            pixel.require(archive.testzip() is None, f"CORRUPT_ZIP:{path}")
            names = archive.namelist()
            pixel.require(any(name.endswith(".pose.txt") for name in names), f"ZIP_NO_POSE:{path}")
            pixel.require(any(name.endswith(".depth.pgm") for name in names), f"ZIP_NO_DEPTH:{path}")
    pixel.require(path.stat().st_size > 0, f"EMPTY_SOURCE:{path}")


def _download(url: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        _validate_download(destination)
        print(f"REUSE {destination.name} {destination.stat().st_size}", flush=True)
        return {"downloaded_now": False, "url": url}
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    print(f"DOWNLOAD {url}", flush=True)
    try:
        request = Request(url, headers={"User-Agent": "BlindAssist-L10-source-queue/1.0"})
        with urlopen(request, timeout=120) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
        _validate_download(partial)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    print(f"DOWNLOADED {destination.name} {destination.stat().st_size}", flush=True)
    return {"downloaded_now": True, "url": url}


def _manifest(destination: Path, artifact_root: Path, transfer: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": destination.resolve().relative_to(artifact_root.resolve()).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": pixel.sha256(destination),
        **transfer,
    }


def _screen(
    candidate: dict[str, Any], data_root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    target_id = int(candidate["target_instance_id"])
    rules = protocol["candidate_view_rules"]
    reference_base = old.portfolio._portfolio(
        data_root,
        str(candidate["reference_scan_id"]),
        target_id,
        rules,
        int(protocol["reference_memory"]["base_positive_marginal_budget"]),
    )
    reference_points, reference_candidates, reference_opened = old.views._candidates(
        data_root, str(candidate["reference_scan_id"]), target_id, rules
    )
    reference_selected, vacancy_fills = old._fill_reference(
        reference_points,
        reference_candidates,
        reference_base["selected"],
        int(protocol["reference_memory"]["required_views"]),
    )
    _, query_candidates, query_opened = old.views._candidates(
        data_root, str(candidate["rescan_id"]), target_id, rules
    )
    query_selected, query_triplet = active._consecutive_triplet(query_candidates)

    sibling = None
    for sibling_id in candidate["rescan_door_instance_ids"]:
        sibling_id = int(sibling_id)
        if sibling_id == target_id:
            continue
        points, candidates, opened = old.views._candidates(
            data_root, str(candidate["rescan_id"]), sibling_id, rules
        )
        if not candidates:
            continue
        selected = max(
            (row for row, _, _ in candidates),
            key=lambda row: (
                int(row["visible_target_vertices"]),
                float(row["bbox_short_side_fraction"]),
                float(row["depth_visible_ratio"]),
                -int(row["frame"]),
            ),
        )
        sibling = {
            "instance_id": sibling_id,
            "label": "door_or_doorframe",
            "target_vertices": int(len(points)),
            "candidate_views": len(candidates),
            "selected": selected,
            "opened": opened,
        }
        break
    evaluable = (
        len(reference_base["selected"])
        >= int(protocol["reference_memory"]["minimum_positive_marginal_views"])
        and len(reference_selected) == int(protocol["reference_memory"]["required_views"])
        and len(query_selected) == 3
        and sibling is not None
    )
    return {
        "source_evaluable": evaluable,
        "reference_memory": {
            **reference_base,
            "eligible_admitted_views": len(reference_candidates),
            "required_views": int(protocol["reference_memory"]["required_views"]),
            "selected": reference_selected,
            "vacancy_fills": vacancy_fills,
            "opened": reference_opened,
        },
        "query_triplet": {
            "scan_id": str(candidate["rescan_id"]),
            "target_instance_id": target_id,
            "eligible_admitted_views": len(query_candidates),
            "selected": [deepcopy(row) for row in query_selected],
            "triplet_receipt": query_triplet,
            "opened": query_opened,
        },
        "same_scene_sibling": sibling,
        "rgb_members_opened": 0,
        "model_calls": 0,
    }


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    queue_row = protocol["queue"]
    queue_path = HERE / queue_row["path"]
    pixel.require(pixel.sha256(queue_path) == queue_row["sha256"], "QUEUE_HASH")
    queue = pixel.load_json(queue_path)
    pixel.require(queue["schema"] == queue_row["required_schema"], "QUEUE_SCHEMA")

    artifact_root = ROOT / protocol["source"]["artifact_root"]
    data_root = artifact_root / "datasets/3rscan"
    rows: list[dict[str, Any]] = []
    selected_index = None
    selected_source = None
    template = protocol["source"]["official_url_template"]
    for queued in queue["ordered_candidates"]:
        queue_index = int(queued["queue_index"])
        candidate = queued["candidate"]
        files: list[dict[str, Any]] = []
        for requested in queued["download_plan"]:
            scan_id = str(requested["scan_id"])
            name = str(requested["file"])
            destination = data_root / scan_id / name
            url = template.format(scan_id=scan_id, file=name)
            transfer = _download(url, destination)
            files.append(_manifest(destination, artifact_root, transfer))
        screened = _screen(candidate, data_root, protocol)
        row = {
            "queue_index": queue_index,
            "candidate": candidate,
            "source_files": files,
            "conclusion": (
                "L10_3RSCAN_DISCRIMINATIVE_VIEW_QUEUE_ROW_SOURCE_EVALUABLE"
                if screened["source_evaluable"]
                else "L10_3RSCAN_DISCRIMINATIVE_VIEW_QUEUE_ROW_SOURCE_NOT_EVALUABLE"
            ),
            **screened,
        }
        rows.append(row)
        print(
            f"SCREENED queue={queue_index} evaluable={screened['source_evaluable']} "
            f"reference_views={screened['reference_memory']['eligible_admitted_views']} "
            f"query_views={screened['query_triplet']['eligible_admitted_views']} "
            f"triplet={screened['query_triplet']['triplet_receipt']}",
            flush=True,
        )
        if screened["source_evaluable"]:
            selected_index = queue_index
            selected_source = row
            break

    evaluable = selected_source is not None
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "PRE_FROZEN_ORDERED_MULTI_FAMILY_POST_DOWNLOAD_POSE_DEPTH_ONLY_SOURCE",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "queue": queue_row,
        "conclusion": (
            "L10_3RSCAN_DISCRIMINATIVE_VIEW_SOURCE_QUEUE_EVALUABLE"
            if evaluable
            else "L10_3RSCAN_DISCRIMINATIVE_VIEW_SOURCE_QUEUE_NOT_EVALUABLE"
        ),
        "source_evaluable": evaluable,
        "rows_screened": rows,
        "selected_queue_index": selected_index,
        "selected_source": selected_source,
        "unused_queue_indices": [
            int(row["queue_index"])
            for row in queue["ordered_candidates"]
            if selected_index is not None and int(row["queue_index"]) > selected_index
        ],
        "rgb_members_opened": 0,
        "model_calls": 0,
        "next_action": protocol["next_action"]["evaluable" if evaluable else "not_evaluable"],
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
