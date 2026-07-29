"""P3 synthetic transport adapter and response-hidden equivalence lock."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..rgb_algorithm_development_canary_cid_sims_r0 import producer as r3
from . import generator_geometry as geometry


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
LOCK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "R3_TRANSPORT_EQUIVALENCE_LOCK_R0"
)
EXPECTED_R3_IMPLEMENTATION = "ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3"
PAIR_CORE_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "rgb_algorithm_development_canary_cid_sims_r0/producer.py"
)
R3_RUNNER_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "ecological_response_discovery_r0/runner.py"
)
THREE_PAIR_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "rcle_low_reference_false_trigger_r1/temporal_confirmation.py"
)
AMENDMENT_RELATIVE = (
    "docs/research/rcle/"
    "RCLE_ROTATION_COMPENSATION_MECHANISM_AUDIT_R1_"
    "IMPLEMENTATION_AMENDMENT_R3_2026-07-28.json"
)
CONTRACT_RELATIVE = (
    "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_CONTRACT_2026-07-28.json"
)
PROTOCOL_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/configs/"
    "phase_a_synthetic_signal_audit_r0.json"
)


class InvalidTransport(ValueError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())


def rotation_homography(
    previous_world_from_camera: np.ndarray,
    current_world_from_camera: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    previous = np.asarray(previous_world_from_camera, dtype=np.float64)
    current = np.asarray(current_world_from_camera, dtype=np.float64)
    native_k = np.asarray(intrinsic, dtype=np.float64)
    if previous.shape != (3, 3) or current.shape != (3, 3):
        raise InvalidTransport("ROTATION_SHAPE")
    if native_k.shape != (3, 3):
        raise InvalidTransport("INTRINSIC_SHAPE")
    if not (
        np.isfinite(previous).all()
        and np.isfinite(current).all()
        and np.isfinite(native_k).all()
    ):
        raise InvalidTransport("NONFINITE_GEOMETRY")
    return native_k @ (current.T @ previous) @ np.linalg.inv(native_k)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise InvalidTransport("RGB_UINT8_HWC3_REQUIRED")
    return np.ascontiguousarray(cv2.cvtColor(value, cv2.COLOR_RGB2GRAY))


def valid_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask)
    if value.shape != shape:
        raise InvalidTransport("VALID_MASK_SHAPE")
    if value.dtype == np.bool_:
        result = value.astype(np.uint8) * 255
    elif value.dtype == np.uint8 and set(np.unique(value)).issubset({0, 255}):
        result = value
    else:
        raise InvalidTransport("VALID_MASK_BINARY")
    return np.ascontiguousarray(result)


def evaluate_pair(
    *,
    pair_index: int,
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    previous_valid: np.ndarray,
    current_valid: np.ndarray,
    previous_timestamp_s: float,
    current_timestamp_s: float,
    previous_world_from_camera: np.ndarray,
    current_world_from_camera: np.ndarray,
    intrinsic: np.ndarray,
    protocol: dict[str, Any],
    state: r3.PairState,
) -> dict[str, Any]:
    previous_gray = rgb_to_gray(previous_rgb)
    current_gray = rgb_to_gray(current_rgb)
    previous_mask = valid_mask(previous_valid, previous_gray.shape)
    current_mask = valid_mask(current_valid, current_gray.shape)
    left = Decimal(str(previous_timestamp_s))
    right = Decimal(str(current_timestamp_s))
    if not Decimal("0") < right - left <= Decimal("0.1"):
        raise InvalidTransport("PAIR_DT")
    homography = rotation_homography(
        previous_world_from_camera,
        current_world_from_camera,
        intrinsic,
    )
    return r3._evaluate_pair(
        pair_index,
        previous_gray,
        current_gray,
        left,
        right,
        homography,
        protocol,
        state,
        previous_mask,
        current_mask,
    )


def _state_digest(state: r3.PairState) -> str:
    survivor = state.survivors
    payload: dict[str, Any] = {"dt_seconds": state.dt_seconds, "survivors": None}
    if survivor is not None:
        payload["survivors"] = {
            key: value
            for key, value in vars(survivor).items()
        }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _reference_decode(path: Path) -> np.ndarray:
    # This intentionally follows the frozen file transport's Pillow RGB decode.
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


def _fixture_frames() -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    scene = geometry.build_scene("ADVIO_14", 0, "PREFLIGHT_EQUIVALENCE")
    trajectory_path = (
        repo_root()
        / "artifacts.local/evidence/"
        "rcle_periodic_self_motion_counterfactual_r2/"
        "p1_geometry_r2_keyset_repair_r0/trajectory_manifest.json"
    )
    manifest = json.loads(trajectory_path.read_text(encoding="utf-8"))
    poses = manifest["ADVIO_14"]["poses"][:5]
    rgbs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    for pose in poses:
        rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
        translation = np.asarray(pose["translation_m"], dtype=np.float64)
        rendered = geometry.render(scene, rotation, translation)
        rgbs.append(rendered["rgb"])
        mask = np.isfinite(rendered["depth"])
        # A deterministic partial-mask sentinel proves that the adapter does not
        # silently replace source-native validity with an all-valid mask.
        mask[:3, :7] = False
        masks.append(mask)
        rotations.append(rotation)
    # Channel sentinel: Pillow RGB and in-memory RGB paths must agree while a
    # BGR mutation would not.
    rgbs[0] = rgbs[0].copy()
    rgbs[0][10:20, 10:20] = np.array([255, 0, 31], dtype=np.uint8)
    return rgbs, masks, rotations


def run_equivalence(output: Path | None = None) -> dict[str, Any]:
    root = repo_root()
    contract = json.loads((root / CONTRACT_RELATIVE).read_text(encoding="utf-8"))
    if contract["unchanged_algorithm_lock"]["implementation"] != EXPECTED_R3_IMPLEMENTATION:
        raise InvalidTransport("R3_IMPLEMENTATION_DRIFT")
    bindings = {
        item["path"]: item["sha256"]
        for item in contract["frozen_dependencies"].values()
        if isinstance(item, dict) and "path" in item and "sha256" in item
    }
    required = {
        R3_RUNNER_RELATIVE,
        PAIR_CORE_RELATIVE,
        THREE_PAIR_RELATIVE,
        AMENDMENT_RELATIVE,
    }
    if not required.issubset(bindings):
        raise InvalidTransport("R3_BINDING_SET")
    for relative in required:
        if sha256_file(root / relative) != bindings[relative]:
            raise InvalidTransport(f"R3_BINDING_DRIFT:{relative}")
    protocol = json.loads((root / PROTOCOL_RELATIVE).read_text(encoding="utf-8"))
    rgbs, masks, rotations = _fixture_frames()
    timestamps = [index / 60.0 for index in range(5)]
    adapter_state = r3.PairState()
    reference_state = r3.PairState()
    adapter_digests: list[str] = []
    reference_digests: list[str] = []
    context = (
        tempfile.TemporaryDirectory(prefix="rcle-p3-transport-")
        if output is None
        else tempfile.TemporaryDirectory(
            prefix="rcle-p3-transport-", dir=os.fspath(output.parent)
        )
    )
    with context as temporary:
        directory = Path(temporary)
        for index, rgb in enumerate(rgbs):
            Image.fromarray(rgb, mode="RGB").save(
                directory / f"{index:03d}.png",
                format="PNG",
                compress_level=9,
            )
        for pair_index in range(4):
            adapter_row = evaluate_pair(
                pair_index=pair_index,
                previous_rgb=rgbs[pair_index],
                current_rgb=rgbs[pair_index + 1],
                previous_valid=masks[pair_index],
                current_valid=masks[pair_index + 1],
                previous_timestamp_s=timestamps[pair_index],
                current_timestamp_s=timestamps[pair_index + 1],
                previous_world_from_camera=rotations[pair_index],
                current_world_from_camera=rotations[pair_index + 1],
                intrinsic=geometry.K,
                protocol=protocol,
                state=adapter_state,
            )
            previous_gray = _reference_decode(directory / f"{pair_index:03d}.png")
            current_gray = _reference_decode(directory / f"{pair_index + 1:03d}.png")
            reference_row = r3._evaluate_pair(
                pair_index,
                previous_gray,
                current_gray,
                Decimal(str(timestamps[pair_index])),
                Decimal(str(timestamps[pair_index + 1])),
                geometry.K
                @ (rotations[pair_index + 1].T @ rotations[pair_index])
                @ np.linalg.inv(geometry.K),
                protocol,
                reference_state,
                masks[pair_index].astype(np.uint8) * 255,
                masks[pair_index + 1].astype(np.uint8) * 255,
            )
            adapter_digest = hashlib.sha256(canonical_bytes(adapter_row)).hexdigest()
            reference_digest = hashlib.sha256(canonical_bytes(reference_row)).hexdigest()
            if adapter_digest != reference_digest:
                raise InvalidTransport(f"PAIR_NUMERIC_MISMATCH:{pair_index}")
            if _state_digest(adapter_state) != _state_digest(reference_state):
                raise InvalidTransport(f"PAIR_STATE_MISMATCH:{pair_index}")
            adapter_digests.append(adapter_digest)
            reference_digests.append(reference_digest)
    payload = {
        "schema": "rcle.periodic_self_motion_counterfactual.p3_transport_lock.v1",
        "protocol_id": PROTOCOL_ID,
        "lock_id": LOCK_ID,
        "terminal": "TRANSPORT_EQUIVALENCE_PASS / VALID / PREFLIGHT_ONLY",
        "formal_execution_authorized": False,
        "p4_activated": False,
        "scientific_outcome_interpreted": False,
        "fixture": {
            "role": "DETERMINISTIC_TRANSPORT_EQUIVALENCE_ONLY",
            "pair_count": 4,
            "identity_rotation_covered": bool(
                np.array_equal(rotations[0], np.eye(3))
            ),
            "nonzero_rotation_covered": any(
                not np.array_equal(item, np.eye(3)) for item in rotations[1:]
            ),
            "rgb_channel_sentinel": True,
            "partial_valid_mask": True,
            "continuous_pair_state": True,
            "pair_row_sha256": adapter_digests,
            "reference_pair_row_sha256": reference_digests,
            "rows_equal": True,
            "state_equal": True,
        },
        "bindings": [
            {"path": relative, "sha256": sha256_file(root / relative)}
            for relative in sorted(required)
        ]
        + [
            {"path": CONTRACT_RELATIVE, "sha256": sha256_file(root / CONTRACT_RELATIVE)},
            {"path": PROTOCOL_RELATIVE, "sha256": sha256_file(root / PROTOCOL_RELATIVE)},
            {
                "path": Path(__file__).resolve().relative_to(root).as_posix(),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        ],
        "forbidden_changes": {
            "r3_or_threshold_or_three_pair": False,
            "sequence16_android_realtime": False,
            "formal_seed_access": False,
            "formal_480_plus_16": False,
        },
    }
    if output is not None:
        write_exclusive(output.resolve(), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_equivalence(args.output)
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "pair_count": result["fixture"]["pair_count"],
                "rows_equal": result["fixture"]["rows_equal"],
                "state_equal": result["fixture"]["state_equal"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
