"""Materialize C18 after C17 exposed a source-view reachability failure."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17


COHORT_ID = "DTR_CARLA_C18_X42_COUNTERFACTUAL_FALSIFIER_V1"
CAPTURE_SEED = 180913
C17_TERMINAL_PROTOCOL_SHA256 = (
    "C1E4FAA28C7FCDFB656AEDFA44EF44F1502A2A80FCA9A23BE3ADF72F4C557CE3"
)
C17_TERMINAL_INSTANCE_RESULT_SHA256 = (
    "4327CE2D5361F7D557A9866FFBE82E2C72DE20907EAE66BEBCD845976B001A7E"
)


def materialize(base: dict) -> dict:
    protocol = c17.materialize(base)
    protocol["schema_version"] = 18
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Falsify frozen X42 once on new paired lateral near-miss and temporarily "
        "occluded true-contact twins while retaining the C16 camera-reachable "
        "scene orientation exposed as necessary by terminal C17."
    )

    # C17 reversed the whole world-facing scene and made 7/8 target sequences
    # fail the frozen pre/post visibility gate.  C18 is a new capture cohort:
    # restore only the known camera-reachable axes and witness pose, while
    # retaining C17's new appearance, weather, motion scale, lateral mirror,
    # opposite occluder direction, tighter safe twins, and frozen X42 gate.
    for layout_id, layout in protocol["layouts"].items():
        parent = base["layouts"][layout_id]
        layout["anchor"] = copy.deepcopy(parent["anchor"])
        layout["witness"] = copy.deepcopy(parent["witness"])

    protocol["source_disjoint_contract"] = {
        **protocol["source_disjoint_contract"],
        "schema": "dtr-carla-c18-x42-source-disjoint-contract-v1",
        "new_capture_seed": CAPTURE_SEED,
        "changed_scene_orientation": False,
        "retained_camera_reachable_parent_axes": True,
        "changed_local_lateral_geometry": True,
        "c17_terminal_design_evidence": {
            "protocol_sha256": C17_TERMINAL_PROTOCOL_SHA256,
            "instance_result_sha256": C17_TERMINAL_INSTANCE_RESULT_SHA256,
            "status": "SOURCE_NOT_EVALUABLE_1_OF_8_OCCLUSION_CONTRACTS",
        },
        "c17_pixels_reused": False,
    }
    protocol["claim_boundary"] = [
        "C18 is a fresh capture cohort; no C17 pixels are reused.",
        "C17 selected the camera-reachable orientation only. X42, thresholds, target appearances, weather, motion scale, lateral mirror, opposite occluder direction, and tighter safe twins remain unopened for C18.",
        "C18 remains synthetic Development and cannot establish deployment, product, user-benefit, reliability, or safety authority.",
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
                "status": "C18_X42_PROTOCOL_STATIC_VALID_PREREGISTERED",
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
