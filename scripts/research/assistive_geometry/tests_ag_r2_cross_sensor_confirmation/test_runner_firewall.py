from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation import runner
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.contract import (
    ContractError,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.eth3d_source import (
    ReadEvent,
    SourcePhase,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.runner import (
    PhaseFirewall,
)


def _event(phase: SourcePhase) -> ReadEvent:
    return ReadEvent(
        parent_id="plant_scene_2",
        archive_kind="RGBD_TRAINING_ARCHIVE",
        phase=phase,
        purpose="fixture",
        member="plant_scene_2/fixture",
        bytes=1,
    )


def test_prediction_calibration_conditioned_truth_order_is_fail_closed() -> None:
    firewall = PhaseFirewall()
    firewall.observe(_event(SourcePhase.ROSTER_METADATA))
    with pytest.raises(ContractError, match="F2_FIREWALL_SOURCE_PHASE_VIOLATION"):
        firewall.observe(_event(SourcePhase.RAW_SCORE_PREDICTION))
    firewall.advance("ROSTER")
    firewall.observe(_event(SourcePhase.RAW_SCORE_PREDICTION))
    with pytest.raises(ContractError, match="F2_FIREWALL_SOURCE_PHASE_VIOLATION"):
        firewall.observe(_event(SourcePhase.CALIBRATION_SOURCE))
    firewall.advance("RAW_SCORE_PREDICTION")
    firewall.observe(_event(SourcePhase.CALIBRATION_SOURCE))
    with pytest.raises(ContractError, match="F2_FIREWALL_SOURCE_PHASE_VIOLATION"):
        firewall.observe(_event(SourcePhase.SCORE_SOURCE))
    firewall.advance("CALIBRATION_SOURCE")
    with pytest.raises(ContractError, match="F2_FIREWALL_SOURCE_PHASE_VIOLATION"):
        firewall.observe(_event(SourcePhase.SCORE_SOURCE))
    firewall.advance("CONDITIONED_SEALED")
    firewall.observe(_event(SourcePhase.SCORE_SOURCE))
    firewall.advance("SCORE_SOURCE")
    assert firewall.stage == "COMPLETE"
    assert [row["source_phase"] for row in firewall.events] == [
        "ROSTER_METADATA",
        "RAW_SCORE_PREDICTION",
        "CALIBRATION_SOURCE",
        "SCORE_SOURCE",
    ]


def test_runner_requires_external_lock_and_has_no_reducer_import() -> None:
    path = Path(runner.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("reducer" in name.lower() for name in imports)
    execute = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "execute")
    assert [argument.arg for argument in execute.args.args] == ["execution_lock_path"]
    calls = {
        node.func.id
        for node in ast.walk(execute)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "validate_execution_lock" in calls
    assert "EvidenceWriter" in calls


def test_prediction_arrays_are_reloaded_before_calibration_and_conditioned_before_truth() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    raw_reload = source.index("raw_records = _reload_records")
    calibration_read = source.index("calibration_frames = [")
    conditioned_reload = source.index("conditioned_records = _reload_records")
    score_read = source.index("SourcePhase.SCORE_SOURCE,", conditioned_reload)
    assert raw_reload < calibration_read < conditioned_reload < score_read
