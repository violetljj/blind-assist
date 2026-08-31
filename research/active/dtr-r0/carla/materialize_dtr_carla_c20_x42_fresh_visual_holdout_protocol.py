"""Materialize the C20 fresh visual holdout for one frozen X42 score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17
import materialize_dtr_carla_c19_x42_visual_transfer_protocol as c19


COHORT_ID = "DTR_CARLA_C20_X42_FRESH_VISUAL_HOLDOUT_V1"
CAPTURE_SEED = 200941
TARGET_TEMPLATES = {
    "c8_l01": "w22",
    "c8_l02": "v_model3",
    "c8_l03": "v_c8_t2",
    "c8_l04": "v_c8_charger_police",
}
C19_PROTOCOL_SHA256 = (
    "E156AF47BCE5CA858C63AD45F432B2FB94280E2BBA64C327BC431778A4D9C044"
)
C19_INSTANCE_RESULT_SHA256 = (
    "DCEB2AC66A8EE3F1771E0AC1F8C17C5783966733117F43B12F6CDA2994532E70"
)


def materialize(base: dict) -> dict:
    protocol = c19.materialize(base)
    protocol["schema_version"] = 20
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Evaluate frozen X42 once on fresh pixels and weather with exact-"
        "blueprint-disjoint targets in two of four layouts, while restoring "
        "the two target blueprints whose replacements failed frozen raster "
        "source admission in C19."
    )
    for layout_id, layout in protocol["layouts"].items():
        template = TARGET_TEMPLATES[layout_id]
        for asset in layout["assets"]:
            key = str(asset["asset_key"])
            if "target" in key or "alias" in key:
                asset["template"] = template

    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c20-x42-fresh-visual-holdout-contract-v1",
        "parent_development_cohort_id": base.get("cohort_id"),
        "parent_protocol_sha256": c17.BASE_PROTOCOL_SHA256,
        "new_capture_seed": CAPTURE_SEED,
        "fresh_pixels": True,
        "changed_weather_by_layout": c17.WEATHERS,
        "target_blueprints_by_layout": TARGET_TEMPLATES,
        "target_blueprint_disjoint_layouts": ["c8_l01", "c8_l02"],
        "target_blueprint_inherited_layouts": ["c8_l03", "c8_l04"],
        "preserved_admitted_motion_route_occlusion_geometry": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
        "c19_design_evidence_only": {
            "protocol_sha256": C19_PROTOCOL_SHA256,
            "instance_result_sha256": C19_INSTANCE_RESULT_SHA256,
            "status": "SOURCE_NOT_EVALUABLE_5_OF_8_OCCLUSION_CONTRACTS",
        },
    }
    protocol["claim_boundary"] = [
        "C20 uses fresh pixels and changed weather in all layouts, but only two of four target blueprints are exact-blueprint-disjoint from C16.",
        "C20 is a fresh synthetic visual holdout, not full target-appearance or geometric scenario confirmation.",
        "Motion, route, occlusion, and near-miss geometry remain inherited from C16.",
        "C20 cannot establish deployment, product, user-benefit, reliability, or safety authority.",
        "The frozen X42 chain is scored once; failed source admission is NOT_EVALUABLE and cannot be repaired after pixels are opened.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(c17.BASE_PROTOCOL) != c17.BASE_PROTOCOL_SHA256:
        raise RuntimeError("C16 base protocol drift")
    protocol = materialize(c17._read_json(c17.BASE_PROTOCOL))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "C20_X42_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "x42_sha256": c17.X42_PREDICTOR_SHA256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
