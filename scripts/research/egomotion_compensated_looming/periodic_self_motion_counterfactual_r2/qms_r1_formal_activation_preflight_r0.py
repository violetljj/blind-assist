"""QMS-R1 successor formal activation preflight.

This module can freeze the operator, successor 480+16 identities, and exactly
eight fresh PREFLIGHT identities.  Its only executable workload is the guarded
W8 preflight.  It cannot launch either the predecessor or successor formal run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from unittest import mock

# This import must precede every numeric library.  In W8 mode the guarded
# module removes generic thread caps before NumPy initializes OpenBLAS.
from . import p3_runtime_preflight_r0 as guarded

import numpy as np

from . import generator_geometry as geometry
from . import material_residual_contraction_r1 as qms
from . import quality_interventions_r0 as quality


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_FORMAL_ACTIVATION_PREFLIGHT_R0"
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
PREFLIGHT_BLOCK = "ADVIO_14"
MAIN_ARMS = guarded.FACTORIAL_ARMS
GUARD_ARMS = guarded.GUARDRAIL_ARMS
FRAME_COUNT = 602
PAIR_COUNT = 601
OLD_FORMAL = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "P4_FORMAL_IDENTITY_LOCK_R0_2026-07-29.json"
)
OLD_PREFLIGHT = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "PREFLIGHT_IDENTITY_LOCK_R0_2026-07-29.json"
)
DEV_MANIFEST = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1/predecessor_dev/identity_manifest.json"
)
CAL_MANIFEST = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1/new_cal/identity_manifest.json"
)
TRAJECTORY_MANIFEST = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p1_geometry_r2_keyset_repair_r0/trajectory_manifest.json"
)


class InvalidActivationPreflight(ValueError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    rows = [
        {
            "path": item.relative_to(path).as_posix(),
            "sha256": sha256_file(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value))


def _seed_record(domain: str, block: str, kind: str, ordinal: int) -> dict[str, Any]:
    token = f"{TASK_ID}|{domain}|{block}|{kind}|{ordinal:02d}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return {
        "token": token,
        "token_sha256": digest,
        "numeric_seed_uint64": int.from_bytes(bytes.fromhex(digest)[:8], "big"),
    }


def operator_lock(root: Path) -> dict[str, Any]:
    source = (
        root
        / "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/"
        "material_residual_contraction_r1.py"
    )
    identity = qms.frozen_operator_identity()
    qms.assert_frozen_operator_identity()
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r1_operator_lock.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "operator_id": identity["operator_id"],
        "operator_identity": identity,
        "operator_identity_sha256": hashlib.sha256(canonical_bytes(identity)).hexdigest(),
        "operator_source_path": source.relative_to(root).as_posix(),
        "operator_source_sha256": sha256_file(source),
        "formal_execution_authorized": False,
        "terminal": "QMS_R1_OPERATOR_FROZEN / PREFLIGHT_ONLY",
    }


def _collect_values(value: Any, key: str) -> set[Any]:
    found: set[Any] = set()
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key and isinstance(child, (str, int)):
                found.add(child)
            found.update(_collect_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_values(child, key))
    return found


def exclusion_sources(root: Path) -> tuple[list[dict[str, Any]], dict[str, set[Any]]]:
    sources = []
    union = {
        key: set()
        for key in (
            "numeric_seed_uint64",
            "token",
            "token_sha256",
            "cluster_id",
            "sequence_id",
            "scene_geometry_sha256",
        )
    }
    for label, relative in (
        ("OLD_FORMAL", OLD_FORMAL),
        ("QMS_R1_DEV", DEV_MANIFEST),
        ("QMS_R1_CAL", CAL_MANIFEST),
        ("OLD_PREFLIGHT", OLD_PREFLIGHT),
    ):
        path = root / relative
        value = load_json(path)
        counts = {}
        for key in union:
            values = _collect_values(value, key)
            union[key].update(values)
            counts[key] = len(values)
        sources.append(
            {
                "label": label,
                "path": relative,
                "sha256": sha256_file(path),
                "unique_counts": counts,
            }
        )
    old_w8 = (
        root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "p3_transport_analysis_runtime_preflight_r3/w8/sequences"
    )
    if old_w8.is_dir():
        for receipt in old_w8.glob("*/receipt.json"):
            union["scene_geometry_sha256"].update(
                _collect_values(load_json(receipt), "scene_geometry_sha256")
            )
        sources.append(
            {
                "label": "OLD_PREFLIGHT_W8_SCENE_RECEIPTS",
                "path": old_w8.relative_to(root).as_posix(),
                "receipt_count": len(list(old_w8.glob("*/receipt.json"))),
                "scene_hash_count_after_union": len(union["scene_geometry_sha256"]),
            }
        )
    return sources, union


def _scene(block: str, ordinal: int, domain: str, kind: str) -> dict[str, Any]:
    seed = _seed_record(domain, block, kind, ordinal)
    generator_namespace = "GUARD" if kind == "GUARD" else "MAIN"
    with mock.patch.object(
        geometry, "derive_seed", return_value=seed["numeric_seed_uint64"]
    ):
        scene = geometry.build_scene(block, ordinal, generator_namespace)
    return scene


def _preflight_scene_core(kind: str) -> dict[str, Any]:
    if kind not in {"FACTORIAL", "GUARDRAIL"}:
        raise InvalidActivationPreflight("PREFLIGHT_KIND")
    seed = _seed_record("PREFLIGHT", PREFLIGHT_BLOCK, kind, 0)
    generator_namespace = "GUARD" if kind == "GUARDRAIL" else "MAIN"
    with mock.patch.object(
        geometry, "derive_seed", return_value=seed["numeric_seed_uint64"]
    ):
        return geometry.build_scene(PREFLIGHT_BLOCK, 0, generator_namespace)


def _assert_fresh(candidate: dict[str, Any], excluded: dict[str, set[Any]]) -> None:
    for key in excluded:
        value = candidate.get(key)
        if value is not None and value in excluded[key]:
            raise InvalidActivationPreflight(f"IDENTITY_OVERLAP:{key}:{value}")


def formal_identity_lock(root: Path, operator_path: Path) -> dict[str, Any]:
    sources, excluded = exclusion_sources(root)
    trajectories = load_json(root / TRAJECTORY_MANIFEST)
    identities: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    for block in BLOCKS:
        for kind, count, arms in (
            ("MAIN", 20, MAIN_ARMS),
            ("GUARD", 2, GUARD_ARMS),
        ):
            for ordinal in range(count):
                seed = _seed_record("SUCCESSOR_FORMAL", block, kind, ordinal)
                cluster_id = f"QMSR1_SUCCESSOR_{block}_{kind}_{ordinal:02d}"
                scene = _scene(block, ordinal, "SUCCESSOR_FORMAL", kind)
                latent = {
                    **seed,
                    "cluster_id": cluster_id,
                    "scene_geometry_sha256": scene["scene_geometry_sha256"],
                }
                _assert_fresh(latent, excluded)
                seeds.append({"block": block, "kind": kind, "ordinal": ordinal, **latent})
                trajectory = trajectories[block]
                source_arms = (
                    MAIN_ARMS
                    if kind == "MAIN"
                    else (
                        "MONOTONIC_APPROACH",
                        "MONOTONIC_APPROACH_PLUS_PERIODIC",
                    )
                )
                for arm_ordinal, (arm, source_arm) in enumerate(zip(arms, source_arms)):
                    sequence_id = (
                        f"QMSR1_SUCCESSOR_FORMAL_{block}_{kind}_{ordinal:02d}__{arm}"
                    )
                    identity = {
                        "sequence_id": sequence_id,
                        "cluster_id": cluster_id,
                        "block": block,
                        "kind": kind,
                        "ordinal": ordinal,
                        "role": (
                            "MAIN_FACTORIAL" if kind == "MAIN" else "POSITIVE_GUARDRAIL"
                        ),
                        "arm": arm,
                        "source_arm_id": source_arm,
                        "arm_ordinal": arm_ordinal,
                        **seed,
                        "scene_geometry_sha256": scene["scene_geometry_sha256"],
                        "trajectory_sha256": (
                            (
                                trajectory["periodic_pose_sha256"]
                                if arm.startswith("PERIODIC")
                                else hashlib.sha256(
                                    geometry.canonical_bytes({"static": FRAME_COUNT})
                                ).hexdigest()
                            )
                            if kind == "MAIN"
                            else geometry._guard_trajectory(
                                trajectory,
                                source_arm == "MONOTONIC_APPROACH_PLUS_PERIODIC",
                            )["pose_sha256"]
                        ),
                        "frame_count": FRAME_COUNT,
                        "pair_count": PAIR_COUNT,
                    }
                    _assert_fresh(identity, excluded)
                    identities.append(identity)
    if len(identities) != 496 or len(seeds) != 88:
        raise InvalidActivationPreflight("FORMAL_CARDINALITY")
    payload = {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r1_successor_formal_identity_lock.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "phase": "SUCCESSOR_FORMAL_480_PLUS_16_IDENTITY_FREEZE_ONLY",
        "counts": {
            "main_sequences": 480,
            "guardrail_sequences": 16,
            "total_sequences": 496,
            "latent_clusters": 88,
            "total_frames": 298592,
            "total_pairs": 298096,
        },
        "operator_lock": {
            "path": operator_path.relative_to(root).as_posix(),
            "sha256": sha256_file(operator_path),
        },
        "exclusion_sources": sources,
        "seeds": seeds,
        "identities": identities,
        "zero_overlap_fields": [
            "numeric_seed_uint64",
            "token",
            "token_sha256",
            "cluster_id",
            "sequence_id",
            "scene_geometry_sha256",
        ],
        "formal_execution_authorized": False,
        "formal_sequences_run": 0,
        "terminal": "SUCCESSOR_FORMAL_IDENTITY_LOCK_VALID / ACTIVATION_PREFLIGHT_REQUIRED",
    }
    payload["identity_set_sha256"] = hashlib.sha256(
        canonical_bytes(identities)
    ).hexdigest()
    return payload


def preflight_identity_lock(
    root: Path, operator_path: Path, formal_path: Path
) -> dict[str, Any]:
    sources, excluded = exclusion_sources(root)
    formal = load_json(formal_path)
    for key in excluded:
        excluded[key].update(_collect_values(formal, key))
    identities = []
    seeds = []
    for kind, arms in (("FACTORIAL", MAIN_ARMS), ("GUARDRAIL", GUARD_ARMS)):
        seed = _seed_record("PREFLIGHT", PREFLIGHT_BLOCK, kind, 0)
        scene = _preflight_scene_core(kind)
        cluster_id = f"QMSR1_PREFLIGHT_{PREFLIGHT_BLOCK}_{kind}_00"
        latent = {
            **seed,
            "cluster_id": cluster_id,
            "scene_geometry_sha256": scene["scene_geometry_sha256"],
        }
        _assert_fresh(latent, excluded)
        seeds.append({"kind": kind, **latent})
        for arm_ordinal, arm in enumerate(arms):
            identity = {
                "sequence_id": f"QMSR1_PREFLIGHT_{PREFLIGHT_BLOCK}_{kind}_00__{arm}",
                "cluster_kind": kind,
                "cluster_id": cluster_id,
                "cluster_token_sha256": seed["token_sha256"],
                **seed,
                "scene_geometry_sha256": scene["scene_geometry_sha256"],
                "arm": arm,
                "arm_ordinal": arm_ordinal,
                "frame_count": FRAME_COUNT,
                "pair_count": PAIR_COUNT,
            }
            _assert_fresh(identity, excluded)
            identities.append(identity)
    payload = {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r1_activation_preflight_identity_lock.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "role": "W8_GUARDED_HOST_ONLY_NO_SCIENTIFIC_INTERPRETATION",
        "block": PREFLIGHT_BLOCK,
        "seeds": seeds,
        "identities": identities,
        "identity_count": 8,
        "workers": 8,
        "native_threads_per_worker": 18,
        "operator_lock_sha256": sha256_file(operator_path),
        "successor_formal_identity_lock_sha256": sha256_file(formal_path),
        "exclusion_sources": sources,
        "formal_execution_authorized": False,
        "formal_seed_execution": False,
        "terminal": "PREFLIGHT_IDENTITY_LOCK_VALID / W8_NOT_RUN",
    }
    payload["identity_set_sha256"] = hashlib.sha256(
        canonical_bytes(identities)
    ).hexdigest()
    return payload


_ORIGINAL_EVALUATE = guarded._evaluate_identity


def _preflight_seed(kind: str) -> dict[str, Any]:
    return _seed_record("PREFLIGHT", PREFLIGHT_BLOCK, kind, 0)


def _preflight_scene(kind: str) -> dict[str, Any]:
    scene = _preflight_scene_core(kind)
    scene["preflight_kind"] = kind
    scene["preflight_seed_token_sha256"] = _preflight_seed(kind)["token_sha256"]
    return scene


def _qms_render(
    scene: dict[str, Any], pose: dict[str, Any], arm: str
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(pose["translation_m"], dtype=np.float64)
    paired = qms.render_pair(scene, rotation, translation)
    rgb = paired["rgb_pair"]["clean"]
    if arm.endswith("__LOW_TEXTURE"):
        rgb = paired["rgb_pair"]["low"]
    elif arm.endswith("__BLUR"):
        rgb = quality.apply_blur(rgb, 0.475)
    return rgb, paired["valid_mask"]


def _evaluate_successor(identity: dict[str, Any]) -> dict[str, Any]:
    guarded._seed_record = _preflight_seed
    guarded._build_scene = _preflight_scene
    guarded._render_frame = _qms_render
    receipt = _ORIGINAL_EVALUATE(identity)
    receipt["schema"] = (
        "rcle.periodic_self_motion_counterfactual."
        "qms_r1_activation_preflight_sequence_receipt.v1"
    )
    receipt["task_id"] = TASK_ID
    receipt["qms_r1_operator_source_sha256"] = sha256_file(Path(qms.__file__))
    return receipt


def run_w8(identity_lock: Path, output_dir: Path) -> dict[str, Any]:
    manifest = load_json(identity_lock)
    if (
        manifest.get("task_id") != TASK_ID
        or manifest.get("identity_count") != 8
        or manifest.get("workers") != 8
        or manifest.get("formal_execution_authorized") is not False
    ):
        raise InvalidActivationPreflight("PREFLIGHT_LOCK")
    root = repo_root()
    predecessor_formal = (
        root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "p4_formal"
    )
    successor_formal = (
        root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "qms_r1_successor_formal"
    )
    if not predecessor_formal.is_dir() or successor_formal.exists():
        raise InvalidActivationPreflight("FORMAL_PATH_FIREWALL")
    predecessor_before = tree_sha256(predecessor_formal)
    guarded.validate_identity_manifest = lambda value: (
        None
        if canonical_bytes(value) == canonical_bytes(manifest)
        else (_ for _ in ()).throw(InvalidActivationPreflight("PREFLIGHT_LOCK_DRIFT"))
    )
    guarded._evaluate_identity = _evaluate_successor
    guarded.P3_ID = TASK_ID
    receipt = guarded.run_profile(identity_lock, output_dir, 8, 18)
    predecessor_after = tree_sha256(predecessor_formal)
    if predecessor_before != predecessor_after or successor_formal.exists():
        raise InvalidActivationPreflight("FORMAL_PATH_FIREWALL_DRIFT")
    receipt["task_id"] = TASK_ID
    receipt["formal_path_firewall"] = {
        "predecessor_formal_tree_sha256_before": predecessor_before,
        "predecessor_formal_tree_sha256_after": predecessor_after,
        "predecessor_formal_unchanged": True,
        "successor_formal_path": successor_formal.relative_to(root).as_posix(),
        "successor_formal_path_absent": True,
        "formal_sequences_run": 0,
    }
    guarded._write_json(output_dir / "success.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--repo-root", type=Path, required=True)
    freeze.add_argument("--operator-lock", type=Path, required=True)
    freeze.add_argument("--formal-identity-lock", type=Path, required=True)
    freeze.add_argument("--preflight-identity-lock", type=Path, required=True)
    run = sub.add_parser("run-w8")
    run.add_argument("--identity-lock", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "freeze":
        root = args.repo_root.resolve()
        operator_path = (root / args.operator_lock).resolve()
        formal_path = (root / args.formal_identity_lock).resolve()
        preflight_path = (root / args.preflight_identity_lock).resolve()
        write_exclusive(operator_path, operator_lock(root))
        write_exclusive(formal_path, formal_identity_lock(root, operator_path))
        write_exclusive(
            preflight_path,
            preflight_identity_lock(root, operator_path, formal_path),
        )
        value = load_json(preflight_path)
    else:
        value = run_w8(args.identity_lock.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "terminal": value.get("terminal"),
                "formal_execution_authorized": value.get(
                    "formal_execution_authorized", False
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
