from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation import (
    independent_validator,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.contract import (
    PROTOCOL_PATH,
    canonical_sha256,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.evidence import (
    EvidenceWriter,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.metrics import (
    score,
)
from scripts.research.assistive_geometry.tests_ag_r2_cross_sensor_confirmation.test_metrics import (
    synthetic_inputs,
)


def _seal(value: dict) -> dict:
    result = dict(value)
    result["content_sha256"] = canonical_sha256(result)
    return result


def _write_phase(writer: EvidenceWriter, relative: str, value: dict) -> dict:
    sealed = _seal(value)
    return writer.write_json(relative, sealed) | {"content_sha256": sealed["content_sha256"]}


def _prediction_arrays(frame: dict) -> dict[str, np.ndarray]:
    prediction = frame["prediction"]
    return {
        "parent_id": np.asarray(frame["parent_id"]),
        "frame_id": np.asarray(frame["frame_id"]),
        "source_hw": np.asarray(prediction["depth_m"].shape, dtype=np.int64),
        "output_hw": np.asarray(prediction["depth_m"].shape, dtype=np.int64),
        "intrinsics": np.asarray(
            [[frame["fx"], 0.0, 4.5], [0.0, frame["fy"], 4.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        **{key: np.asarray(value) for key, value in prediction.items()},
    }


def _truth_arrays(frame: dict) -> dict[str, np.ndarray]:
    truth = frame["truth"]
    return {
        "parent_id": np.asarray(frame["parent_id"]),
        "frame_id": np.asarray(frame["frame_id"]),
        "fx": np.asarray(frame["fx"], dtype=np.float64),
        "fy": np.asarray(frame["fy"], dtype=np.float64),
        "camera_height_m": np.asarray(1.5, dtype=np.float64),
        **{key: np.asarray(value) for key, value in truth.items()},
    }


def build_evidence(root: Path) -> None:
    protocol, source, frames = synthetic_inputs()
    writer = EvidenceWriter(
        root,
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_start_receipt.v1",
            "synthetic_fixture": True,
        },
    )
    roster_receipt = _write_phase(
        writer,
        "roster.json",
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_roster.v1",
            "phase": "ROSTER_SEALED",
            "parent_count": 3,
            "score_count": 36,
        },
    )
    raw_records = []
    conditioned_records = []
    truth_records = []
    for frame in frames:
        parent = frame["parent_id"]
        frame_id = frame["frame_id"]
        raw = writer.write_npz(f"phase-a/raw/{parent}/{frame_id}.npz", _prediction_arrays(frame))
        raw_records.append({"parent_id": parent, "frame_id": frame_id, **raw})
        conditioned = writer.write_npz(f"phase-c/conditioned/{parent}/{frame_id}.npz", _prediction_arrays(frame))
        conditioned_records.append({"parent_id": parent, "frame_id": frame_id, **conditioned})
        truth = writer.write_npz(f"phase-d/truth/{parent}/{frame_id}.npz", _truth_arrays(frame))
        truth_records.append({"parent_id": parent, "frame_id": frame_id, **truth})
    raw_receipt = _write_phase(
        writer,
        "phase-a/raw-prediction-completion.json",
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_phase_completion.v1",
            "phase": "RAW_RGBK_PREDICTIONS_SEALED",
            "record_count": 36,
            "records": raw_records,
            "predecessor": {"path": "roster.json", "sha256": roster_receipt["sha256"]},
        },
    )
    context_receipt = _write_phase(
        writer,
        "phase-b/session-contexts.json",
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_phase_completion.v1",
            "phase": "CALIBRATION_SOURCE_CONTEXT_SEALED",
            "parent_count": 3,
            "contexts": [{"parent_id": parent, "camera_height_m": 1.5} for parent in ("plant_scene_2", "motion_1", "mannequin_5")],
            "predecessor": {"path": "phase-a/raw-prediction-completion.json", "sha256": raw_receipt["sha256"]},
        },
    )
    conditioned_receipt = _write_phase(
        writer,
        "phase-c/conditioned-factor-completion.json",
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_phase_completion.v1",
            "phase": "CONDITIONED_FACTORS_SEALED_BEFORE_SCORE_TRUTH",
            "record_count": 36,
            "records": conditioned_records,
            "predecessor": {"path": "phase-b/session-contexts.json", "sha256": context_receipt["sha256"]},
        },
    )
    truth_receipt = _write_phase(
        writer,
        "phase-d/truth-completion.json",
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_phase_completion.v1",
            "phase": "SCORE_SOURCE_TRUTH_SEALED",
            "record_count": 36,
            "records": truth_records,
            "predecessor": {"path": "phase-c/conditioned-factor-completion.json", "sha256": conditioned_receipt["sha256"]},
        },
    )
    _write_phase(
        writer,
        "source-summary.json",
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_source_summary.v1",
            "phase": "SOURCE_SUMMARY_SEALED",
            "parents": source["parents"],
            "predecessor": {"path": "phase-d/truth-completion.json", "sha256": truth_receipt["sha256"]},
        },
    )
    summary = score(protocol, source, frames)
    writer.write_json("summary.json", summary)
    result = _seal(
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_result.v1",
            "terminal": summary["terminal"],
            "execution_valid": True,
            "summary_sha256": summary["content_sha256"],
            "training_steps": 0,
            "reducer_calls": 0,
            "network_requests": 0,
        }
    )
    writer.write_json("result.json", result)
    writer.finalize(summary["terminal"])


def test_independent_validator_replays_complete_evidence(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    build_evidence(root)
    result = independent_validator.verify(root, PROTOCOL_PATH)
    assert result["passed"] is True
    assert result["scientific_terminal"] == "CONFIRM_PASS"
    assert result["summary_exact_replay"] is True
    assert result["frame_count"] == 36


def test_manifest_and_summary_mutations_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    build_evidence(root)
    path = root / "phase-c/conditioned/plant_scene_2/plant_scene_2-00.npz"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(independent_validator.ValidationError, match="F2V_MANIFEST_FILE_DRIFT"):
        independent_validator.verify(root, PROTOCOL_PATH)


def test_phase_predecessor_mutation_fails_even_if_manifest_is_not_consulted(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    build_evidence(root)
    phase = root / "phase-c/conditioned-factor-completion.json"
    value = __import__("json").loads(phase.read_text(encoding="utf-8"))
    value["predecessor"]["sha256"] = "0" * 64
    value.pop("content_sha256")
    value = _seal(value)
    phase.write_text(__import__("json").dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(independent_validator.ValidationError):
        independent_validator.verify(root, PROTOCOL_PATH)


def test_validator_import_firewall_is_ast_enforced() -> None:
    path = Path(independent_validator.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = ("producer", "eth3d_source", "source_geometry", "model_only", "metrics", "evidence", "recipe", "reducer")
    assert not [name for name in imports if any(token in name.lower() for token in forbidden)]


def test_partial_source_not_evaluable_is_hash_and_firewall_verified(tmp_path: Path) -> None:
    root = tmp_path / "partial"
    writer = EvidenceWriter(root, {"schema": "fixture-start"})
    _write_phase(
        writer,
        "roster.json",
        {"schema": "fixture-roster", "phase": "ROSTER_SEALED"},
    )
    failure = _seal(
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_failure.v1",
            "terminal": "NOT_EVALUABLE",
            "reason_code": "F2_CALIBRATION_FORMAT",
            "message": "fixture",
            "firewall_stage": "RAW_SCORE_PREDICTION",
            "access_events": [
                {
                    "event_index": 0,
                    "firewall_stage": "ROSTER",
                    "parent_id": "plant_scene_2",
                    "archive_kind": "RGBD_TRAINING_ARCHIVE",
                    "source_phase": "ROSTER_METADATA",
                    "purpose": "fixture",
                    "member": "plant_scene_2/associated.txt",
                    "bytes": 1,
                }
            ],
        }
    )
    failure_receipt = writer.write_json("failure.json", failure)
    result = _seal(
        {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_result.v1",
            "terminal": "NOT_EVALUABLE",
            "execution_valid": True,
            "failure_sha256": failure["content_sha256"],
            "failure_file_sha256": failure_receipt["sha256"],
            "summary_sha256": None,
            "model_inference_calls": 0,
            "training_steps": 0,
            "reducer_calls": 0,
            "network_requests": 0,
        }
    )
    writer.write_json("result.json", result)
    writer.finalize("NOT_EVALUABLE")
    validation = independent_validator.verify(root, PROTOCOL_PATH)
    assert validation["passed"] is True
    assert validation["scientific_terminal"] == "NOT_EVALUABLE"
    assert validation["completed_phase_count"] == 1
