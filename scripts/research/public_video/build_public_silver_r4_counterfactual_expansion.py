#!/usr/bin/env python3
"""Build an isolated r4 public-video silver package with one same-source pair.

The builder copies only validated public-video provisional-training packages
from an r3 parent. It does not read or write any independent model-direction
experiment. The sole label change is the prereviewed SANPO gie8 timeline:
an early passable-corridor episode and a late narrowing-corridor episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_public_video_silver_labels import load_json, sha256_file, validate


SCHEMA = "blindassist_public_silver_counterfactual_expansion_v1"
GIE8_PACKAGE = "sanpo-gie8-cafe-sidewalk-gpt5-20260716"
PAIR_ID = "sanpo-gie8-clear-to-cafe-furniture-narrowing-0000-0135"
PROMPT_TEXT = """Review a temporally ordered public RGB episode for navigation risk.
Judge the intended near-field walking corridor, not object presence alone.
Use candidate_no_alert only when heading and near field are visible, the
corridor remains passable, and no near lateral intrusion is hidden. Use
candidate_alert when an approaching obstacle materially narrows or occupies
the intended corridor. Abstain when those facts cannot be established.
These are model-provisional labels, never human event truth."""


def prompt_sha256() -> str:
    return hashlib.sha256(PROMPT_TEXT.encode("utf-8")).hexdigest()


def reject_independent_direction(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if "secondary-corridor-causal" in normalized:
        raise ValueError(f"independent model direction is outside this builder's scope: {path}")


def expanded_gie8_manifest(parent: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(parent))
    result["labeler"] = {
        "provider": "openai",
        "model": "gpt-5",
        "prompt_id": "blindassist-public-risk-silver-counterfactual-v2",
        "prompt_sha256": prompt_sha256(),
        "review_mode": "multiframe_temporal",
    }
    result["episodes"] = [
        {
            "episode_id": "sanpo-gie8-cafe-passable-side-corridor-0000-0030",
            "evidence_frame_sha256": [
                "c28ef43d5651fe6fa2bed876e65e1e4d6764678f7fead88a82670fd5d8f86ac1",
                "fad813768622cb26b7ff5ab3ed20a2b3c262e86e30b6633d12b91de1f7a939e4",
                "21a994421e87ad4b4f64fa0bd32904c43a08428a36140facf65e98048b98b5a6",
            ],
            "silver_should_alert": "candidate_no_alert",
            "confidence": 0.72,
            "counterfactual_pair_id": PAIR_ID,
            "risk_profile": {
                "primary_hazard_type": "static_sidewalk_furniture",
                "corridor_relation": "lateral_and_passable",
                "lifecycle": "no_alert",
                "counterfactual_pair_id": PAIR_ID,
            },
            "negative_decision_quality": {
                "corridor_heading_stability": 0.86,
                "near_field_visibility": 0.84,
                "corridor_clearance": 0.74,
                "near_field_lateral_intrusion_absent": 0.72,
            },
            "uncertainty_reasons": [
                "Cafe chairs remain visible on the right edge, so the label means passable corridor rather than an empty scene.",
                "The intended route is inferred from camera heading and the open center-left sidewalk.",
            ],
        },
        {
            "episode_id": "sanpo-gie8-cafe-furniture-narrowing-0105-0135",
            "evidence_frame_sha256": [
                "000c59891d83966aba0c6b7416da3de8ae6f464793c20fd7d4efc91683202a02",
                "f77d37e66ba2510258ce309f491f880ec71870a7db6bb1abfbcd54e418a065fe",
                "9839ece0cc3952399cefc0b36455c2eb6da5f2e669bb0c2ce64c5bd3e5573700",
            ],
            "silver_should_alert": "candidate_alert",
            "confidence": 0.70,
            "counterfactual_pair_id": PAIR_ID,
            "risk_profile": {
                "primary_hazard_type": "static_sidewalk_obstruction",
                "corridor_relation": "near_field_corridor_narrowing",
                "lifecycle": "approach_alertable",
                "counterfactual_pair_id": PAIR_ID,
            },
            "uncertainty_reasons": [
                "A left-side bypass remains visible, so this is corridor narrowing rather than complete blockage.",
                "One-second sampling cannot establish the exact first safe alert frame.",
            ],
        },
    ]
    return result


def build(*, parent_root: Path, output_root: Path) -> dict[str, Any]:
    parent_root = parent_root.resolve()
    output_root = output_root.resolve()
    reject_independent_direction(parent_root)
    reject_independent_direction(output_root)
    if output_root.exists():
        raise ValueError(f"refusing to overwrite output root: {output_root}")

    silver_paths = sorted(parent_root.glob("*/silver_labels_v2.json"))
    if not silver_paths:
        raise ValueError(f"no v2 public-silver packages found under {parent_root}")
    if not any(path.parent.name == GIE8_PACKAGE for path in silver_paths):
        raise ValueError(f"required gie8 package is missing under {parent_root}")

    package_rows: list[dict[str, Any]] = []
    output_root.mkdir(parents=True)
    try:
        for parent_silver_path in silver_paths:
            parent_package = parent_silver_path.parent
            reject_independent_direction(parent_package)
            package_name = parent_package.name
            destination = output_root / package_name
            shutil.copytree(parent_package, destination)

            silver_path = destination / "silver_labels_v2.json"
            source_path = destination / "source_manifest_v2.json"
            parent_silver_sha256 = sha256_file(parent_silver_path)
            silver = load_json(silver_path)
            change = "copied_without_label_change"
            if package_name == GIE8_PACKAGE:
                silver = expanded_gie8_manifest(silver)
                silver_path.write_text(json.dumps(silver, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                change = "replaced_single_alert_with_same_source_counterfactual_pair"

            validation = validate(silver, source_manifest_path=source_path)
            receipt = {
                "schema": SCHEMA,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "package_name": package_name,
                "change": change,
                "parent_package": str(parent_package.resolve()),
                "parent_silver_labels_v2_sha256": parent_silver_sha256,
                "source_manifest_v2_sha256": sha256_file(source_path),
                "silver_labels_v2_sha256": sha256_file(silver_path),
                "validation": validation,
                "independent_model_directions_used": False,
                "human_event_truth_present": False,
                "calibration_authorized": False,
                "blind_evaluation_authorized": False,
                "production_model_replacement_authorized": False,
            }
            (destination / "promotion_receipt.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            package_rows.append(receipt)

        root_receipt = {
            "schema": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "parent_root": str(parent_root),
            "output_root": str(output_root),
            "package_count": len(package_rows),
            "changed_packages": [row["package_name"] for row in package_rows if row["change"] != "copied_without_label_change"],
            "packages": package_rows,
            "isolation_contract": {
                "public_video_mainline_only": True,
                "independent_model_direction_data_used": False,
                "independent_model_direction_metrics_used_as_gate": False,
            },
        }
        receipt_path = output_root / "r4_build_receipt.json"
        receipt_path.write_text(json.dumps(root_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        Path(str(receipt_path) + ".sha256").write_text(sha256_file(receipt_path) + "\n", encoding="ascii")
        return root_receipt
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = build(parent_root=args.parent_root, output_root=args.output_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "package_count": receipt["package_count"], "changed_packages": receipt["changed_packages"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
