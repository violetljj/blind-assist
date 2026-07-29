from __future__ import annotations

import json
from pathlib import Path

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_r3_rotation_leakage_source_localization_contract_preflight_r0
    as validator,
)


ROOT = Path(__file__).resolve().parents[4]


def test_independent_preflight_validates_without_execution_authority() -> None:
    receipt, decision = validator.validate(ROOT)
    assert receipt["protocol_status"] == "VALID"
    assert receipt["execution_authorized"] is False
    assert receipt["stage_b_response_payload_read"] is False
    assert receipt["source_localization_workload_run"] is False
    assert decision["decision"].startswith("HOLD_")
    assert decision["r3_modification_authorized"] is False


def test_routes_are_exact_and_cluster_level() -> None:
    contract = validator.load_json(ROOT / validator.CONTRACT)
    assert set(contract["per_cluster_routing"]["routes"]) == validator.EXPECTED_ROUTES
    assert contract["estimand"]["analysis_unit"] == "cluster"
    assert contract["estimand"]["longitudinal_repeats"].startswith("601 ")


def test_signed_and_absolute_reductions_remain_separate() -> None:
    contract = validator.load_json(ROOT / validator.CONTRACT)
    assert (
        contract["aggregation_and_coverage"]["pair_signed"]
        == "median(evaluable common-cell signed expansion)"
    )
    assert (
        contract["aggregation_and_coverage"]["pair_absolute"]
        == "median(abs(evaluable common-cell signed expansion))"
    )
    assert "abs(median(signed)) is forbidden" in contract["estimand"][
        "signed_absolute_separation"
    ]


def test_rotation_boundary_and_memory_gate_are_frozen() -> None:
    contract = validator.load_json(ROOT / validator.CONTRACT)
    assert (
        contract["unchanged_r3_parameters"]["final"][
            "rotation_absolute_leakage_boundary_per_s"
        ]
        == 0.01
    )
    assert (
        contract["resource_gate"][
            "launch_and_refill_minimum_available_ram_bytes"
        ]
        == 6 * 1024**3
    )
    assert (
        contract["resource_gate"]["in_flight_emergency_floor_bytes"]
        == 4 * 1024**3
    )


def test_identity_lock_contains_only_sealed_rotation_sequences() -> None:
    lock = validator.load_json(ROOT / validator.LOCK)
    assert len(lock["clusters"]) == 8
    assert len({item["sequence_id"] for item in lock["clusters"]}) == 8
    assert all(
        item["sequence_id"].endswith("__EGO_ROTATION_STATIC_SCENE__CLEAN")
        for item in lock["clusters"]
    )
    assert lock["stage_b_response_payload_read_during_preflight"] is False


def test_json_files_do_not_contain_nonfinite_tokens() -> None:
    for path in (ROOT / validator.CONTRACT, ROOT / validator.LOCK):
        text = path.read_text(encoding="utf-8")
        json.loads(text, parse_constant=lambda token: (_ for _ in ()).throw(
            AssertionError(token)
        ))
