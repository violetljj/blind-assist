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
    calibration_control as control_module,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.calibration_control import (
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
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.validate_calibration_control import (
    validate as independently_validate,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OFFICIAL = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_OFFICIAL_CONTROL_EVIDENCE_2026-08-12.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _binding(role: str, path: Path) -> dict:
    return {"role": role, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _yaml(node: str = "cam0") -> bytes:
    return (
        f"{node}:\n"
        "  camera_model: pinhole\n"
        "  T_cam_imu:\n"
        "  - [1.0, 0.0, 0.0, 0.0]\n"
        "  - [0.0, 0.0, -1.0, 0.0]\n"
        "  - [0.0, 1.0, 0.0, 0.0]\n"
        "  - [0.0, 0.0, 0.0, 1.0]\n"
    ).encode()


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    members: list[tuple[str, bytes]] | None = None,
) -> tuple[Path, dict]:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive = archive_root / "camera_imu_calib_radtan.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as container:
        for name, payload in members or [("calibration/camchain-imucam.yaml", _yaml())]:
            container.writestr(name, payload)
    implementation = control_module.IMPLEMENTATION_LOCK_PATH
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "payload_access_receipt": {"calibration_payload_opened": False},
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
    monkeypatch.setattr(control_module, "DATA_IDENTITY_PATH", identity)
    lock = {
        "schema": CONTROL_LOCK_SCHEMA,
        "lock_id": CONTROL_LOCK_ID,
        "protocol_id": PROTOCOL_ID,
        "status": CONTROL_STATUS,
        "implementation_lock": _binding("REPAIR_IMPLEMENTATION_LOCK", implementation),
        "data_identity": _binding("DATA_IDENTITY", identity),
        "official_control_evidence": _binding("OFFICIAL_FORMAT_AND_IMU_CONVENTION", OFFICIAL),
        "archive_root": str(archive_root.resolve()),
        "output_root": str((tmp_path / "control-evidence").resolve()),
        "budget": CONTROL_BUDGET,
        "authority": CONTROL_AUTHORITY,
        "one_shot": {"exclusive_control_root": True, "rerun": False, "resume": False, "replacement": False},
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return lock_path, lock


def test_synthetic_control_discovers_one_exact_kalibr_member_and_independent_validator_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, lock = _fixture(tmp_path, monkeypatch)
    assert validate_control_lock(lock_path)["status"] == CONTROL_STATUS
    result = execute_control_preflight(lock_path)
    assert result["status"] == "CALIBRATION_CONTROL_PASS_EXACT_MEMBER_BOUND"
    assert result["selected_member"]["name"] == "calibration/camchain-imucam.yaml"
    assert result["selected_member"]["camera_node_key"] == "cam0"
    assert result["access_receipt"]["session_rgbd_archive_reads"] == 0
    assert result["access_receipt"]["model_or_checkpoint_reads"] == 0
    replay = independently_validate(Path(lock["output_root"]), lock_path)
    assert replay["valid"] is True
    assert replay["yaml_member_reads"] == 1


def test_ambiguous_kalibr_controls_consume_only_control_root_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, lock = _fixture(
        tmp_path,
        monkeypatch,
        members=[("one.yaml", _yaml("cam0")), ("two.yaml", _yaml("cam1"))],
    )
    with pytest.raises(ContractError, match="F2_CALIBRATION_CONTROL_AMBIGUOUS_OR_MISSING_MATRIX"):
        execute_control_preflight(lock_path)
    root = Path(lock["output_root"])
    assert json.loads((root / "failure.json").read_text())["one_shot_consumed"] is True
    assert json.loads((root / "manifest.json").read_text())["terminal"] == "CALIBRATION_CONTROL_FAIL_CLOSED"


def test_control_lock_authority_and_binding_mutations_fail_before_archive_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _lock_path, original = _fixture(tmp_path, monkeypatch)
    for name, mutate, code in (
        ("authority", lambda value: value["authority"].__setitem__("model_or_checkpoint_access", True), "F2_CONTROL_AUTHORITY_DRIFT"),
        ("rerun", lambda value: value["one_shot"].__setitem__("rerun", True), "F2_CONTROL_ONE_SHOT_DRIFT"),
        ("identity", lambda value: value["data_identity"].__setitem__("sha256", "0" * 64), "F2_CONTROL_DATA_IDENTITY_BINDING_FILE_DRIFT"),
    ):
        changed = deepcopy(original)
        mutate(changed)
        path = tmp_path / f"lock-{name}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ContractError, match=code):
            validate_control_lock(path)


def test_independent_control_validator_does_not_import_producer_or_source_adapter() -> None:
    from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation import (
        validate_calibration_control,
    )

    tree = ast.parse(Path(validate_calibration_control.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(
        forbidden in imported
        for imported in imports
        for forbidden in ("calibration_control", "eth3d_source", "control_format", "evidence")
    )
