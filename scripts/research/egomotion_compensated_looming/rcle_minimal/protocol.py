from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = MODULE_ROOT / "configs" / "phase_a_synthetic_signal_audit_r0.json"
LOCK_PATH = MODULE_ROOT / "configs" / "phase_a_synthetic_signal_audit_r0.lock.json"
PROTOCOL_SHA256 = "d20e77f3ea5f7ac55376006f1d14feb0ffb5daffd10a42792912fb89cdb1b502"


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    split: str
    motion_family: str
    axis: str
    angular_velocity_deg_per_s: float
    scale_rate_per_s: float
    fps: int
    degradation: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    if sha256_file(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError("PHASE_A_PROTOCOL_HASH_MISMATCH")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("PHASE_A_PROTOCOL_LOCK_HASH_MISMATCH")
    validate_protocol(protocol)
    return protocol


def _trial_id(
    split: str,
    family: str,
    axis: str,
    angular_velocity: float,
    scale_rate: float,
    fps: int,
    degradation: str,
    seed: int,
) -> str:
    angular = f"{angular_velocity:+05.1f}".replace("+", "p").replace("-", "m")
    scale = f"{scale_rate:+0.2f}".replace("+", "p").replace("-", "m")
    return (
        f"{split}_{family}_{axis}_w{angular}_c{scale}_"
        f"fps{fps}_{degradation}_seed{seed}"
    )


def _spec(
    split: str,
    family: str,
    axis: str,
    angular_velocity: float,
    scale_rate: float,
    fps: int,
    degradation: str,
    seed: int,
) -> TrialSpec:
    return TrialSpec(
        trial_id=_trial_id(
            split,
            family,
            axis,
            angular_velocity,
            scale_rate,
            fps,
            degradation,
            seed,
        ),
        split=split,
        motion_family=family,
        axis=axis,
        angular_velocity_deg_per_s=angular_velocity,
        scale_rate_per_s=scale_rate,
        fps=fps,
        degradation=degradation,
        seed=seed,
    )


def enumerate_trials(protocol: dict[str, Any]) -> list[TrialSpec]:
    seeds = protocol["trials"]["seeds"]
    fps_groups = protocol["rendering"]["fps_groups"]
    clean = protocol["trials"]["clean_matrix"]
    trials: list[TrialSpec] = []

    rotation = clean["pure_rotation"]
    for axis in rotation["axes"]:
        for direction in rotation["directions"]:
            for speed in rotation["angular_speeds_deg_per_s"]:
                for fps in fps_groups:
                    for seed in seeds:
                        trials.append(
                            _spec(
                                "clean",
                                "pure_rotation",
                                axis,
                                direction * speed,
                                0.0,
                                fps,
                                "clean",
                                seed,
                            )
                        )

    scale = clean["scale"]
    for direction in scale["directions"]:
        for magnitude in scale["scale_rate_magnitudes_per_s"]:
            for fps in fps_groups:
                for seed in seeds:
                    trials.append(
                        _spec(
                            "clean",
                            "scale",
                            "none",
                            0.0,
                            direction * magnitude,
                            fps,
                            "clean",
                            seed,
                        )
                    )

    mixed = clean["rotation_plus_scale_up"]
    for axis in mixed["axes"]:
        for direction in mixed["directions"]:
            for speed in mixed["angular_speeds_deg_per_s"]:
                for scale_rate in mixed["scale_rates_per_s"]:
                    for fps in fps_groups:
                        for seed in seeds:
                            trials.append(
                                _spec(
                                    "clean",
                                    "rotation_plus_scale_up",
                                    axis,
                                    direction * speed,
                                    scale_rate,
                                    fps,
                                    "clean",
                                    seed,
                                )
                            )

    stress = protocol["trials"]["stress_sentinel"]
    fps = stress["fps"]
    angular_speed = stress["angular_speed_deg_per_s"]
    scale_rate = stress["scale_rate_magnitude_per_s"]
    for degradation in stress["profiles"]:
        for axis in rotation["axes"]:
            for direction in rotation["directions"]:
                for seed in seeds:
                    trials.append(
                        _spec(
                            "stress",
                            "pure_rotation",
                            axis,
                            direction * angular_speed,
                            0.0,
                            fps,
                            degradation,
                            seed,
                        )
                    )
        for direction in scale["directions"]:
            for seed in seeds:
                trials.append(
                    _spec(
                        "stress",
                        "scale",
                        "none",
                        0.0,
                        direction * scale_rate,
                        fps,
                        degradation,
                        seed,
                    )
                )
        for axis in rotation["axes"]:
            for direction in rotation["directions"]:
                for seed in seeds:
                    trials.append(
                        _spec(
                            "stress",
                            "rotation_plus_scale_up",
                            axis,
                            direction * angular_speed,
                            scale_rate,
                            fps,
                            degradation,
                            seed,
                        )
                    )

    return trials


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol["schema_version"] != "rcle.phase_a.protocol.v1":
        raise ValueError("UNSUPPORTED_PHASE_A_PROTOCOL_SCHEMA")
    if protocol["status"] != "PREREGISTERED_BEFORE_FORMAL_RUN":
        raise ValueError("PHASE_A_PROTOCOL_NOT_PREREGISTERED")
    if protocol["metrics"]["statistical_unit"] != "trial":
        raise ValueError("PHASE_A_STATISTICAL_UNIT_MUST_BE_TRIAL")
    if protocol["local_affine"]["grid_rows"] != 3:
        raise ValueError("PHASE_A_GRID_ROWS_MUST_BE_3")
    if protocol["local_affine"]["grid_cols"] != 3:
        raise ValueError("PHASE_A_GRID_COLS_MUST_BE_3")
    if protocol["local_affine"]["expansion_definition"] != "0.5_times_trace_A":
        raise ValueError("PHASE_A_EXPANSION_DEFINITION_DRIFT")
    if protocol["metrics"]["ratios_are_diagnostic_only"] is not True:
        raise ValueError("PHASE_A_RATIO_AUTHORITY_DRIFT")
    fps = protocol["rendering"]["fps_groups"]
    if fps != [15, 30, 60]:
        raise ValueError("PHASE_A_FPS_GROUP_DRIFT")
    seeds = protocol["trials"]["seeds"]
    if len(seeds) != len(set(seeds)) or len(seeds) != 20:
        raise ValueError("PHASE_A_SEED_INVENTORY_DRIFT")
    trials = enumerate_trials(protocol)
    ids = [trial.trial_id for trial in trials]
    if len(ids) != len(set(ids)):
        raise ValueError("PHASE_A_DUPLICATE_TRIAL_ID")
    expected = protocol["trials"]["expected_total_trials"]
    if len(trials) != expected:
        raise ValueError(
            f"PHASE_A_TRIAL_COUNT_DRIFT expected={expected} actual={len(trials)}"
        )
    clean = sum(trial.split == "clean" for trial in trials)
    stress = sum(trial.split == "stress" for trial in trials)
    if clean != protocol["trials"]["expected_clean_trials"]:
        raise ValueError("PHASE_A_CLEAN_TRIAL_COUNT_DRIFT")
    if stress != protocol["trials"]["expected_stress_trials"]:
        raise ValueError("PHASE_A_STRESS_TRIAL_COUNT_DRIFT")
    if any(trial.fps <= 0 for trial in trials):
        raise ValueError("PHASE_A_NON_POSITIVE_FPS")
