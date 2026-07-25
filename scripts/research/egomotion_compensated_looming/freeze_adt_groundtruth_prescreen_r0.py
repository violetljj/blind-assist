#!/usr/bin/env python3
"""Freeze a metadata-only ADT ground-truth prescreen without reading payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DATA_URL = "https://explorer.projectaria.com/data/adt"
LINKS_URL = "https://explorer.projectaria.com/data/adt/download_links"
DEVICE_SUFFIX = re.compile(r"_(?:M1292|71292|61283)$")
SELECTION_SALT = "egomotion-compensated-looming-adt-prescreen-r0-v1"

PROXY_STRATA = {
    "PURE_EGO_ROTATION_NO_CLOSING_PROXY": {
        "activities": {"ObjectExamination", "ExampleSequence"},
        "multi_person": None,
    },
    "EGO_APPROACH_STATIC_SURFACE_PROXY": {
        "activities": {"Cleaning", "RoomDecoration", "Working"},
        "multi_person": False,
    },
    "STATIONARY_EGO_ACTIVE_TARGET_APPROACH_PROXY": {
        "activities": {"MealPreparation", "Partying"},
        "multi_person": True,
    },
    "LATERAL_PASS_NO_SUSTAINED_CLOSING_PROXY": {
        "activities": {"Partying"},
        "multi_person": True,
    },
}


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"metadata request failed: {response.status} {url}")
        return json.load(response)


def _cluster_base(sequence_id: str) -> str:
    return DEVICE_SUFFIX.sub("", sequence_id)


def _candidate_rank(stratum: str, sequence_id: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SALT}|{stratum}|{sequence_id}".encode("utf-8")
    ).hexdigest()


def build_freeze(per_stratum: int) -> dict[str, Any]:
    data = _get_json(DATA_URL)
    links_response = _get_json(LINKS_URL)
    links = links_response["sequences"]
    if set(data) != set(links):
        raise AssertionError("ADT data/download-link sequence sets differ")
    if len(data) != 236:
        raise AssertionError(f"unexpected ADT sequence count: {len(data)}")

    cluster_counts = Counter(_cluster_base(sequence_id) for sequence_id in data)
    selected_ids: set[str] = set()
    selections: list[dict[str, Any]] = []
    for stratum, rule in PROXY_STRATA.items():
        eligible: list[str] = []
        for sequence_id, metadata in data.items():
            if metadata["duration_s"] < 20:
                continue
            if cluster_counts[_cluster_base(sequence_id)] != 1:
                continue
            if metadata["activity"] not in rule["activities"]:
                continue
            if (
                rule["multi_person"] is not None
                and metadata["is_multi_person"] is not rule["multi_person"]
            ):
                continue
            if sequence_id in selected_ids:
                continue
            member = links[sequence_id].get("main_groundtruth")
            if not member:
                continue
            eligible.append(sequence_id)
        ranked = sorted(eligible, key=lambda item: (_candidate_rank(stratum, item), item))
        chosen = ranked[:per_stratum]
        if len(chosen) != per_stratum:
            raise AssertionError(
                f"insufficient metadata-only candidates for {stratum}: {len(chosen)}"
            )
        for sequence_id in chosen:
            selected_ids.add(sequence_id)
            metadata = data[sequence_id]
            member = links[sequence_id]["main_groundtruth"]
            selections.append(
                {
                    "proxy_stratum": stratum,
                    "sequence_id": sequence_id,
                    "capture_cluster_base_inferred": _cluster_base(sequence_id),
                    "capture_cluster_inference_authoritative": False,
                    "scene": metadata["scene"],
                    "activity": metadata["activity"],
                    "is_multi_person": metadata["is_multi_person"],
                    "duration_s": metadata["duration_s"],
                    "trajectory_length_m": metadata.get(
                        "computed.trajectory_length_m"
                    ),
                    "speed_mps_mean": metadata.get("computed.speed_mps_mean"),
                    "selection_rank_sha256": _candidate_rank(stratum, sequence_id),
                    "main_groundtruth": {
                        "filename": member["filename"],
                        "sha1sum": member["sha1sum"],
                        "file_size_bytes": member["file_size_bytes"],
                    },
                }
            )

    identity_lines = [
        "\t".join(
            [
                row["proxy_stratum"],
                row["sequence_id"],
                row["main_groundtruth"]["filename"],
                row["main_groundtruth"]["sha1sum"],
                str(row["main_groundtruth"]["file_size_bytes"]),
            ]
        )
        for row in selections
    ]
    return {
        "schema_version": "adt_groundtruth_prescreen_freeze_r0",
        "source_id": "ARIA_DIGITAL_TWIN",
        "selection_stage": "metadata_only_before_groundtruth_payload",
        "selection_salt": SELECTION_SALT,
        "sequence_inventory_count": len(data),
        "per_proxy_stratum": per_stratum,
        "selected_sequence_count": len(selections),
        "selected_total_groundtruth_bytes": sum(
            row["main_groundtruth"]["file_size_bytes"] for row in selections
        ),
        "selection_identity_sha256": hashlib.sha256(
            "\n".join(identity_lines).encode("utf-8")
        ).hexdigest(),
        "selections": selections,
        "role_split_frozen": False,
        "counterfactual_cell_labels_frozen": False,
        "proxy_strata_are_cell_truth": False,
        "rgb_payload_requested": False,
        "groundtruth_payload_requested": False,
        "authority": {
            "may_download_selected_main_groundtruth_only": True,
            "may_download_rgb_or_vrs": False,
            "may_run_candidate_signal": False,
            "may_freeze_r0_admission": False,
        },
        "terminal": "ADT_GROUNDTRUTH_PRESCREEN_SELECTION_FROZEN",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.per_stratum <= 8:
        raise SystemExit("--per-stratum must be in [1, 8]")
    receipt = build_freeze(args.per_stratum)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "VALID",
                "terminal": receipt["terminal"],
                "selected_sequence_count": receipt["selected_sequence_count"],
                "selected_total_groundtruth_bytes": receipt[
                    "selected_total_groundtruth_bytes"
                ],
                "rgb_payload_requested": receipt["rgb_payload_requested"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
