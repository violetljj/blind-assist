#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


OFFICIAL_PAGE_SHA256 = "2bd8df16acad79c70e1021f1da039c78510034fd9091fd706f8a3f480ea5c186"
SELECTION_SALT = "looming-r1-bonn-claim-scoped-source-freeze-v1"
PRIOR_INSPECTED = {
    "rgbd_bonn_moving_obstructing_box": {
        "archive_sha256": "d0fc8c33d896b6a4c4a506508f44ddec63bf7dc37c8f58524826cbca1bbba689",
        "review_receipt_sha256": "42032693643d8b26a5cbe3e289c6645b21d67cc87f5b8b13ac31ec5f655597c2",
    },
    "rgbd_bonn_person_tracking": {
        "archive_sha256": "a4810fd91ef2ea1d630b53fe0df5d76144c1b18d86ca91fb3a035debd0c9c5f5",
        "review_receipt_sha256": "323627cee037393b9f8e7493cca8d497cd22035ca0a6338daee37292c6f49c40",
    },
    "rgbd_bonn_crowd": {
        "archive_sha256": "76b30c86e4ccb78ab668dd427dab696faef4885633dba436bc391b3a8142df70",
        "review_receipt_sha256": None,
    },
}
MAX_DISPLAY_SIZE_MB = 550.0
ROLE_COUNTS = {"discovery": 2, "validation": 2, "sealed_holdout": 2}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(sequence_id: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SALT}\t{sequence_id}".encode("utf-8")
    ).hexdigest()


def parse_inventory(html: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"Name:\s*(rgbd_bonn_[a-z0-9_]+)<br\s*/>\s*"
        r"Size:\s*([0-9.]+)\s*(MB|GB)<br\s*/>\s*"
        r'<a href="(https://www\.ipb\.uni-bonn\.de/html/projects/'
        r'rgbd_dynamic2019/\1\.zip)">Download</a>',
        re.IGNORECASE,
    )
    inventory: list[dict[str, Any]] = []
    for sequence_id, size_text, unit, url in pattern.findall(html):
        size_mb = float(size_text) * (1024.0 if unit.upper() == "GB" else 1.0)
        inventory.append(
            {
                "sequence_id": sequence_id,
                "display_size": f"{size_text} {unit.upper()}",
                "display_size_mb_normalized": size_mb,
                "url": url,
            }
        )
    inventory.sort(key=lambda item: item["sequence_id"])
    if len(inventory) != 26:
        raise ValueError(f"expected 26 sequence entries, got {len(inventory)}")
    if len({item["sequence_id"] for item in inventory}) != 26:
        raise ValueError("duplicate sequence identity")
    return inventory


def build_receipt(inventory: list[dict[str, Any]], page_sha256: str) -> dict[str, Any]:
    if page_sha256 != OFFICIAL_PAGE_SHA256:
        raise ValueError("official Bonn page SHA-256 mismatch")
    inventory_ids = {item["sequence_id"] for item in inventory}
    if not PRIOR_INSPECTED.keys() <= inventory_ids:
        raise ValueError("prior-inspected Bonn sequence missing from inventory")

    eligible = [
        item
        for item in inventory
        if item["sequence_id"] not in PRIOR_INSPECTED
        and item["display_size_mb_normalized"] <= MAX_DISPLAY_SIZE_MB
    ]
    ranked = sorted(eligible, key=lambda item: stable_hash(item["sequence_id"]))
    required_count = sum(ROLE_COUNTS.values())
    if len(ranked) < required_count:
        raise ValueError("insufficient bounded unseen sequence candidates")

    selected: list[dict[str, Any]] = []
    cursor = 0
    for role, count in ROLE_COUNTS.items():
        for item in ranked[cursor : cursor + count]:
            selected.append(
                {
                    **item,
                    "role": role,
                    "claims": [
                        "C1_ROTATION_LEAKAGE_SUPPRESSION",
                        "C2_STATIC_SURFACE_CLOSING_RETENTION",
                    ],
                    "selection_hash": stable_hash(item["sequence_id"]),
                    "permanent_if_downloaded": True,
                }
            )
        cursor += count

    identity_lines = [
        "\t".join(
            [
                item["role"],
                item["sequence_id"],
                item["display_size"],
                item["url"],
            ]
        )
        for item in selected
    ]
    cohort_sha256 = hashlib.sha256(
        ("\n".join(identity_lines) + "\n").encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": "bonn_claim_scoped_inventory_and_role_freeze_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1",
        "source_family": "BONN_RGBD_DYNAMIC",
        "authority": {
            "official_page_url": (
                "https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/"
            ),
            "official_page_sha256": page_sha256,
            "official_claims": {
                "dynamic_sequence_count": 24,
                "static_sequence_count": 2,
                "optitrack_sensor_pose": True,
                "registered_rgb_depth": True,
                "static_environment_laser_scan_available": True,
            },
            "published_per_archive_cryptographic_checksum": False,
            "required_after_download": [
                "HTTP_STATUS_AND_CONTENT_LENGTH_RECEIPT",
                "LOCAL_ARCHIVE_SHA256",
                "ARCHIVE_MEMBER_INVENTORY_SHA256",
            ],
        },
        "observed_inventory": {
            "sequence_count": len(inventory),
            "entries": inventory,
        },
        "prior_inspected_denylist": [
            {
                "sequence_id": sequence_id,
                **evidence,
                "disposition": "PRIOR_INSPECTED_ENGINEERING_ONLY",
            }
            for sequence_id, evidence in sorted(PRIOR_INSPECTED.items())
        ],
        "selection_contract": {
            "selection_salt": SELECTION_SALT,
            "maximum_display_size_mb": MAX_DISPLAY_SIZE_MB,
            "rule": (
                "Exclude all three prior-inspected sequences; retain official "
                "entries at or below 550 displayed MB; rank by SHA256(salt, "
                "sequence_id); assign the first two to discovery, next two to "
                "validation, and next two to sealed holdout."
            ),
            "role_counts": ROLE_COUNTS,
            "cohort_identity_sha256": cohort_sha256,
            "selected": selected,
        },
        "claim_interpretation": {
            "sequence_id_is_session_id": True,
            "parent_capture_cluster_published": False,
            "cell_truth_proven_before_payload": False,
            "unsupported_units_abstain": True,
            "dynamic_object_truth_claimed": False,
        },
        "read_firewall": {
            "old_15_pair_window_selection_tuning_acceptance_reads": 0,
            "prior_bonn_outcome_reads": 0,
            "selected_archive_download_count": 0,
            "selected_rgb_decode_count": 0,
            "candidate_signal_computed": False,
        },
        "terminal": "BONN_CLAIM_SCOPED_ROLE_FREEZE_READY_PAYLOAD_NOT_ACQUIRED",
        "status": "VALID",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--official-page", type=Path)
    source.add_argument("--prior-receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.official_page is not None:
        page_sha256 = sha256_path(args.official_page)
        html = args.official_page.read_text(encoding="utf-8")
        inventory = parse_inventory(html)
    else:
        prior = json.loads(args.prior_receipt.read_text(encoding="utf-8"))
        if prior.get("schema_version") != "bonn_claim_scoped_inventory_and_role_freeze_r0":
            raise ValueError("prior receipt schema mismatch")
        page_sha256 = prior["authority"]["official_page_sha256"]
        inventory = prior["observed_inventory"]["entries"]
    receipt = build_receipt(inventory, page_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "terminal": receipt["terminal"],
                "selected": [
                    f"{item['role']}:{item['sequence_id']}"
                    for item in receipt["selection_contract"]["selected"]
                ],
                "cohort_identity_sha256": receipt["selection_contract"][
                    "cohort_identity_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
