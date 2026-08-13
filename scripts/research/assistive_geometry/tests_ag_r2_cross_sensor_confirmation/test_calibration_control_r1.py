from __future__ import annotations

import ast
import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation import (
    PROTOCOL_ID,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation import (
    calibration_control_r1 as control_module,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation import (
    validate_calibration_control_r1_repair_lock as repair_validator,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.calibration_control_r1 import (
    CONTROL_AUTHORITY,
    CONTROL_BUDGET,
    CONTROL_LOCK_ID,
    CONTROL_LOCK_SCHEMA,
    CONTROL_STATUS,
    execute_control_preflight,
    validate_control_lock,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.contract import (
    ContractError,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.control_format_r1 import (
    discover_kalibr_camera_controls,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.validate_calibration_control_r1 import (
    validate as independently_validate,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _binding(role: str, path: Path) -> dict:
    return {"role": role, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _camera(node: str, rostopic: str | None, tx: float = 0.0) -> str:
    topic = "" if rostopic is None else f"  rostopic: {rostopic}\n"
    return (
        f"{node}:\n"
        "  camera_model: pinhole\n"
        f"{topic}"
        "  T_cam_imu:\n"
        f"  - [1.0, 0.0, 0.0, {tx}]\n"
        "  - [0.0, 0.0, -1.0, 0.0]\n"
        "  - [0.0, 1.0, 0.0, 0.0]\n"
        "  - [0.0, 0.0, 0.0, 1.0]\n"
    )


def _yaml(*cameras: str) -> bytes:
    return "".join(cameras).encode()


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    members: list[tuple[str, bytes]],
) -> tuple[Path, dict]:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive = archive_root / "camera_imu_calib_radtan.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as container:
        for name, payload in members:
            container.writestr(name, payload)
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "archives": [
                    {
                        "parent_id": "ALL_THREE_SESSIONS",
                        "kind": "CAMERA_IMU_CALIBRATION_ARCHIVE",
                        "url": "https://www.eth3d.net/data/slam/camera_imu_calib_radtan.zip",
                        "bytes": archive.stat().st_size,
                        "sha256": _sha(archive),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    repair = tmp_path / "repair-lock.json"
    repair.write_text("{}", encoding="utf-8")
    official = tmp_path / "official.json"
    official.write_text(
        json.dumps(
            {
                "schema": (
                    "blindassist.ag.r2.cross_sensor_factor_confirmation_"
                    "calibration_control_r1_official_camera_selection_evidence.v1"
                ),
                "selection_contract": control_module._expected_selection_contract(),
            }
        ),
        encoding="utf-8",
    )
    amendment = tmp_path / "amendment.json"
    amendment.write_text(
        json.dumps(
            {
                "status": "R1_PROTOCOL_REPAIR_FROZEN_NOT_AUTHORIZED_NOT_RUN",
                "prior_access_disclosure": {"r0_calibration_archive_member_enumerated": True},
                "unchanged_scientific_contract": {"scientific_status": "NOT_RUN"},
            }
        ),
        encoding="utf-8",
    )
    r0_terminal = tmp_path / "r0-terminal.json"
    r0_terminal.write_text(
        json.dumps(
            {
                "status": "CALIBRATION_CONTROL_FAIL_CLOSED_AMBIGUOUS_OR_MISSING_MATRIX_ONE_SHOT_CONSUMED",
                "control_outcome": {"one_shot_consumed": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(control_module, "DATA_IDENTITY_PATH", identity)
    monkeypatch.setattr(control_module, "R1_REPAIR_LOCK_PATH", repair)
    monkeypatch.setattr(control_module, "R1_OFFICIAL_EVIDENCE_PATH", official)
    monkeypatch.setattr(control_module, "R1_AMENDMENT_PATH", amendment)
    monkeypatch.setattr(control_module, "R0_TERMINAL_PATH", r0_terminal)
    monkeypatch.setattr(repair_validator, "validate_lock_file", lambda _path, _root: {"valid": True})
    lock = {
        "schema": CONTROL_LOCK_SCHEMA,
        "lock_id": CONTROL_LOCK_ID,
        "protocol_id": PROTOCOL_ID,
        "status": CONTROL_STATUS,
        "repair_implementation_lock": _binding("R1_REPAIR_IMPLEMENTATION_LOCK", repair),
        "data_identity": _binding("DATA_IDENTITY_PRE_R0_SNAPSHOT", identity),
        "official_camera_selection_evidence": _binding("R1_OFFICIAL_CAMERA_SELECTION_EVIDENCE", official),
        "protocol_amendment": _binding("R1_PROTOCOL_AMENDMENT", amendment),
        "r0_terminal": _binding("R0_CONSUMED_CONTROL_TERMINAL", r0_terminal),
        "archive_root": str(archive_root.resolve()),
        "output_root": str((tmp_path / "ag-r2-cross-sensor-calibration-control-r1").resolve()),
        "budget": CONTROL_BUDGET,
        "authority": CONTROL_AUTHORITY,
        "one_shot": {
            "exclusive_r1_control_root": True,
            "producer_runs": 1,
            "independent_validator_replays": 1,
            "r0_rerun": False,
            "r0_resume": False,
            "r0_replacement": False,
        },
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return lock_path, lock


def test_parser_preserves_every_matrix_and_same_node_rostopic() -> None:
    controls = discover_kalibr_camera_controls(
        _yaml(
            _camera("cam0", "/uvc_camera/cam_3/image_raw"),
            _camera("cam1", "'/uvc_camera/cam_2/image_raw'", 0.1),
        )
    )
    assert [(item.camera_node_key, item.rostopic) for item in controls] == [
        ("cam0", "/uvc_camera/cam_3/image_raw"),
        ("cam1", "/uvc_camera/cam_2/image_raw"),
    ]


def test_r1_selects_right_rgb_namespace_not_camchain_order_and_independent_validator_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, lock = _fixture(
        tmp_path,
        monkeypatch,
        members=[
            (
                "calibration/camchain-imucam.yaml",
                _yaml(
                    _camera("cam0", "/uvc_camera/cam_3/image_raw"),
                    _camera("cam1", "/uvc_camera/cam_2/image_raw", 0.1),
                ),
            )
        ],
    )
    assert validate_control_lock(lock_path)["status"] == CONTROL_STATUS
    result = execute_control_preflight(lock_path)
    assert result["selected_member"]["camera_node_key"] == "cam1"
    assert result["selected_member"]["rostopic_namespace"] == "/uvc_camera/cam_2"
    assert result["inventory"]["matrix_discovery_count"] == 2
    replay = independently_validate(Path(lock["output_root"]), lock_path)
    assert replay["valid"] is True
    assert replay["target_namespace_match_count"] == 1


@pytest.mark.parametrize(
    ("members", "expected_discoveries", "expected_matches"),
    [
        (
            [("one.yaml", _yaml(_camera("cam0", "/uvc_camera/cam_3/image_raw")))],
            1,
            0,
        ),
        (
            [
                (
                    "two.yaml",
                    _yaml(
                        _camera("cam0", "/uvc_camera/cam_2/image_raw"),
                        _camera("cam1", "/uvc_camera/cam_2/image_rect", 0.1),
                    ),
                )
            ],
            2,
            2,
        ),
    ],
)
def test_r1_failure_preserves_exact_counts_and_independent_validator_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    members: list[tuple[str, bytes]],
    expected_discoveries: int,
    expected_matches: int,
) -> None:
    lock_path, lock = _fixture(tmp_path, monkeypatch, members=members)
    with pytest.raises(ContractError, match="F2_R1_CALIBRATION_CONTROL_TARGET_CAMERA_AMBIGUOUS_OR_MISSING"):
        execute_control_preflight(lock_path)
    root = Path(lock["output_root"])
    failure = json.loads((root / "failure.json").read_text())
    observed = failure["observability"]
    assert observed["yaml_candidate_count"] == len(members)
    assert observed["yaml_members_read"] == len(members)
    assert observed["matrix_discovery_count"] == expected_discoveries
    assert observed["target_namespace_match_count"] == expected_matches
    assert observed["all_yaml_candidates_read"] is True
    assert failure["selection_receipt"]["first_or_best_selected"] is False
    replay = independently_validate(root, lock_path)
    assert replay["valid"] is True
    assert replay["matrix_discovery_count"] == expected_discoveries
    assert replay["target_namespace_match_count"] == expected_matches


def test_r1_control_lock_mutations_fail_before_archive_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _path, original = _fixture(
        tmp_path,
        monkeypatch,
        members=[("one.yaml", _yaml(_camera("cam0", "/uvc_camera/cam_2/image_raw")))],
    )
    for name, mutate, code in (
        (
            "authority",
            lambda value: value["authority"].__setitem__("model_or_checkpoint_access", True),
            "F2_R1_CONTROL_AUTHORITY_DRIFT",
        ),
        (
            "r0-rerun",
            lambda value: value["one_shot"].__setitem__("r0_rerun", True),
            "F2_R1_CONTROL_ONE_SHOT_DRIFT",
        ),
        (
            "official",
            lambda value: value["official_camera_selection_evidence"].__setitem__("sha256", "0" * 64),
            "F2_R1_CONTROL_OFFICIAL_CAMERA_SELECTION_EVIDENCE_BINDING_FILE_DRIFT",
        ),
    ):
        changed = deepcopy(original)
        mutate(changed)
        path = tmp_path / f"lock-{name}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ContractError, match=code):
            validate_control_lock(path)


def test_r1_independent_validator_does_not_import_producer_source_or_parser() -> None:
    from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation import (
        validate_calibration_control_r1,
    )

    tree = ast.parse(Path(validate_calibration_control_r1.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(
        forbidden in imported
        for imported in imports
        for forbidden in (
            "calibration_control_r1",
            "eth3d_source",
            "control_format_r1",
            "evidence",
        )
    )


def test_tracked_r1_repair_implementation_lock_validates() -> None:
    lock = REPO_ROOT / (
        "docs/research/assistive-geometry/"
        "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
        "CONFIRMATION_CALIBRATION_CONTROL_R0_FAILURE_AUDIT_AND_R1_PROTOCOL_"
        "REPAIR_IMPLEMENTATION_LOCK_2026-08-13.json"
    )
    result = repair_validator.validate_lock_file(lock, REPO_ROOT)
    assert result["valid"] is True
    assert result["implementation_binding_count"] == 5
