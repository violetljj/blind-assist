from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dtr_carla_c2_rich_scene import (  # noqa: E402
    MODEL_TOP_LEVEL_ALLOWLIST,
    build_plan_receipt,
    build_rgbd_alignment_receipt,
    camera_intrinsics,
    plan_waypoints_world,
    sha256_file,
)
from join_dtr_carla_c4_multimap import (  # noqa: E402
    ALLOWED_MAP_STARTUP_ARGUMENTS,
    C2_EXPERIMENT_ID,
    C2_RESULT_NOT_EVALUABLE_STATUS,
    C2_RESULT_STATUS,
    C2_SHARD_STATUS,
    C4ContractError,
    C4_EXPERIMENT_ID,
    C4_INDEX_SCHEMA,
    C4_OCCLUSION_AUDIT_SCHEMA,
    C4_RESULT_STATUS,
    FORMAL_SENSORS,
    GROUP_EXACT_KEYS,
    INDEX_EXACT_KEYS,
    _episode_dynamic_audit,
    _seal_tree,
    join_multimap,
)


ASSET_REGISTRY = HERE / "dtr_carla_c4_asset_registry.json"
SCENE_REGISTRY = HERE / "dtr_carla_c4_scene_registry.json"
DEPTH_CODEC = {
    "name": "CARLA_RGB24_NORMALIZED_DEPTH",
    "maximum_depth_m": 1000.0,
    "formula": "meters=1000*(R+256*G+65536*B)/(16777215)",
}
RIGID_EXTRINSIC = {
    "x_m": 0.08,
    "y_m": 0.0,
    "z_m": 1.62,
    "roll_degrees": 0.0,
    "pitch_degrees": 0.0,
    "yaw_degrees": 0.0,
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected object fixture: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


class C4JoinFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.bundle_root = root / "bundle"
        self.index_path = self.bundle_root / "compiled-c4-protocol.json"
        self.output_root = root / "joined"
        self.asset_registry = _read_json(ASSET_REGISTRY)
        self.scene_registry = _read_json(SCENE_REGISTRY)
        self.assets_by_id = {
            str(value["asset_id"]): value
            for value in self.asset_registry["assets"]  # type: ignore[index]
        }
        self.scenes = self.scene_registry["scenes"]  # type: ignore[index]
        self.groups: list[dict[str, object]] = []

    def build(self) -> Path:
        registry_root = self.bundle_root / "registries"
        registry_root.mkdir(parents=True)
        shutil.copy2(ASSET_REGISTRY, registry_root / "asset_registry.json")
        shutil.copy2(SCENE_REGISTRY, registry_root / "scene_registry.json")

        layouts_by_map: dict[str, list[str]] = {}
        for layout_id, scene in self.scenes.items():  # type: ignore[union-attr]
            layouts_by_map.setdefault(str(scene["map"]), []).append(str(layout_id))
        for map_name, layout_ids in layouts_by_map.items():
            group_id = f"{map_name.rsplit('/', 1)[-1].lower()}_group"
            protocol = self._protocol(map_name, layout_ids)
            protocol_relative = Path("groups") / f"{group_id}.json"
            protocol_path = self.bundle_root / protocol_relative
            _write_json(protocol_path, protocol)
            evidence_relative = Path("evidence") / group_id
            evidence_root = self.bundle_root / evidence_relative
            result_sha256 = self._child_root(
                evidence_root, protocol, map_name, layout_ids
            )
            self.groups.append(
                {
                    "group_id": group_id,
                    "map": map_name,
                    "startup_map_argument": ALLOWED_MAP_STARTUP_ARGUMENTS[map_name],
                    "layout_ids": layout_ids,
                    "protocol_path": protocol_relative.as_posix(),
                    "protocol_sha256": sha256_file(protocol_path),
                    "evidence_path": evidence_relative.as_posix(),
                    "evidence_result_sha256": result_sha256,
                }
            )

        index = {
            "schema_version": C4_INDEX_SCHEMA,
            "experiment_id": C4_EXPERIMENT_ID,
            "registries": {
                "asset_registry": {
                    "path": "registries/asset_registry.json",
                    "sha256": sha256_file(registry_root / "asset_registry.json"),
                },
                "scene_registry": {
                    "path": "registries/scene_registry.json",
                    "sha256": sha256_file(registry_root / "scene_registry.json"),
                },
            },
            "capture": {
                "resolution": [1280, 720],
                "sensor_order": list(FORMAL_SENSORS),
            },
            "admission": {
                "expected_map_count": len(layouts_by_map),
                "expected_protocol_count": len(layouts_by_map),
                "expected_layout_count": len(self.scenes),  # type: ignore[arg-type]
                "expected_episode_count": sum(
                    len(scene["episodes"])
                    for scene in self.scenes.values()  # type: ignore[union-attr]
                ),
                "expected_sensor_count": len(FORMAL_SENSORS),
                "expected_shard_count": len(layouts_by_map) * len(FORMAL_SENSORS),
            },
            "map_layout_groups": self.groups,
        }
        _write_json(self.index_path, index)
        return self.index_path

    def _protocol(self, map_name: str, layout_ids: list[str]) -> dict[str, object]:
        layouts: dict[str, object] = {}
        scenarios: list[dict[str, object]] = []
        occlusion_contracts: list[dict[str, object]] = []
        for layout_id in layout_ids:
            scene = self.scenes[layout_id]  # type: ignore[index]
            layouts[layout_id] = {
                "duration_seconds": 0.5,
                "anchor": scene["anchor"],
                "assets": [
                    {"asset_key": str(actor["instance_id"])}
                    for actor in scene["actors"]
                ],
            }
            for episode in scene["episodes"]:
                scenarios.append(
                    {
                        "episode_id": str(episode["episode_id"]),
                        "layout_id": layout_id,
                        "navigation_session_id": str(
                            episode["navigation_session_id"]
                        ),
                        "issued_plan": episode["issued_plan"],
                    }
                )
            occlusion_contracts.append(
                {
                    "contract_id": f"{layout_id}_occlusion_pair",
                    "episodes": [
                        str(episode["episode_id"]) for episode in scene["episodes"]
                    ],
                    "minimum_pre_track_frames": 10,
                    "minimum_complete_occlusion_seconds": 0.30,
                    "maximum_complete_occlusion_seconds": 0.60,
                    "minimum_post_reappearance_frames": 10,
                }
            )
        return {
            "experiment_id": C2_EXPERIMENT_ID,
            "environment": {"map": map_name, "sample_seconds": 0.05},
            "capture": {
                "resolution": [1280, 720],
                "sensor_order": list(FORMAL_SENSORS),
            },
            "layouts": layouts,
            "scenarios": scenarios,
            "occlusion_contracts": occlusion_contracts,
        }

    def _child_root(
        self,
        evidence_root: Path,
        protocol: dict[str, object],
        map_name: str,
        layout_ids: list[str],
    ) -> str:
        evidence_root.mkdir(parents=True)
        frozen_protocol_path = evidence_root / "frozen_protocol.json"
        _write_json(frozen_protocol_path, protocol)
        protocol_sha256 = sha256_file(frozen_protocol_path)
        episodes = [
            (layout_id, episode)
            for layout_id in layout_ids
            for episode in self.scenes[layout_id]["episodes"]  # type: ignore[index]
        ]
        self._model_root(evidence_root / "model", episodes)
        occlusion_reports, outcomes = self._evaluator_root(
            evidence_root / "evaluator", episodes
        )
        self._shards(evidence_root, episodes, map_name, protocol_sha256)

        model_root = evidence_root / "model"
        evaluator_root = evidence_root / "evaluator"
        sealed_model = _seal_tree(model_root, [model_root])
        sealed_evidence = _seal_tree(
            evidence_root,
            [evidence_root / "shards", model_root, evaluator_root],
        )
        _write_json(evidence_root / "sealed_model_manifest.json", sealed_model)
        _write_json(evidence_root / "sealed_evidence_manifest.json", sealed_evidence)
        checks = {
            "fixture_complete": True,
            "track_then_complete_physical_occlusion_contract_met": all(
                bool(value["passed"]) for value in occlusion_reports
            ),
            "contact_safe_outcome_pair_matches": all(
                value["expected_outcome"] == value["observed_outcome"]
                and value["expected_responsible_assets"]
                == value["observed_responsible_assets"]
                for value in outcomes
            ),
        }
        result = {
            "schema_version": "dtr-carla-c2-rich-scene-result-v2",
            "experiment_id": C2_EXPERIMENT_ID,
            "status": (
                C2_RESULT_STATUS
                if all(checks.values())
                else C2_RESULT_NOT_EVALUABLE_STATUS
            ),
            "checks": checks,
            "protocol_sha256": protocol_sha256,
            "outcomes": outcomes,
            "occlusion_reports": occlusion_reports,
            "sealed_model_manifest_sha256": sha256_file(
                evidence_root / "sealed_model_manifest.json"
            ),
            "sealed_evidence_manifest_sha256": sha256_file(
                evidence_root / "sealed_evidence_manifest.json"
            ),
        }
        _write_json(evidence_root / "result.json", result)
        return sha256_file(evidence_root / "result.json")

    def _model_root(
        self,
        model_root: Path,
        episodes: list[tuple[str, dict[str, object]]],
    ) -> None:
        model_root.mkdir(parents=True)
        raw_wearable: dict[str, list[dict[str, object]]] = {}
        raw_depth: dict[str, list[dict[str, object]]] = {}
        for ordinal, (_, episode) in enumerate(episodes):
            episode_id = str(episode["episode_id"])
            transform = {
                "x": float(ordinal),
                "y": 0.0,
                "z": 1.62,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            }
            raw_wearable[episode_id] = [
                {
                    "sample_index": 0,
                    "time_s": 0.0,
                    "camera_transform": transform,
                    "wearer_transform": transform,
                    "world_frame": 1000 + ordinal,
                }
            ]
            raw_depth[episode_id] = [
                {
                    "sample_index": 0,
                    "time_s": 0.0,
                    "camera_transform": transform,
                    "wearer_transform": transform,
                    "world_frame": 2000 + ordinal,
                }
            ]
        alignment = build_rgbd_alignment_receipt(raw_wearable, raw_depth)
        alignment_path = model_root / "rgbd_alignment_receipt.json"
        _write_json(alignment_path, alignment)
        alignment_file_sha = sha256_file(alignment_path)
        alignment_by_episode = {
            str(value["episode_id"]): value for value in alignment["episodes"]
        }
        calibration = {
            "schema_version": "dtr-c2-model-camera-contract-v1",
            "resolution": {"width": 1280, "height": 720},
            "fov_degrees": 90.0,
            "K": camera_intrinsics(1280, 720, 90.0),
            "depth_codec": DEPTH_CODEC,
            "wearable_rigid_extrinsic": RIGID_EXTRINSIC,
            "sensor_tick_seconds": 0.05,
        }
        calibration_path = model_root / "camera_calibration.json"
        _write_json(calibration_path, calibration)
        contract = {
            "schema_version": "dtr-c2-model-contract-v2",
            "current_actors_enabled": False,
            "dense_modalities": ["wearable_rgb", "metric_depth"],
            "evaluator_sibling_not_required": True,
            "rgbd_alignment": {
                "authority": alignment["authority"],
                "receipt_path": "rgbd_alignment_receipt.json",
                "receipt_sha256": alignment["receipt_sha256"],
                "file_sha256": alignment_file_sha,
                "world_frame_rule": alignment["world_frame_rule"],
            },
            "record_top_level_allowlist": sorted(MODEL_TOP_LEVEL_ALLOWLIST),
        }
        contract_path = model_root / "model_contract.json"
        _write_json(contract_path, contract)

        episode_links: list[dict[str, object]] = []
        for layout_id, episode in episodes:
            episode_id = str(episode["episode_id"])
            scene = self.scenes[layout_id]  # type: ignore[index]
            anchor = scene["anchor"]
            receipt = build_plan_receipt(episode["issued_plan"])
            if receipt is None:
                raise AssertionError("C4 fixture requires a plan")
            plan_relative = f"plans/{episode_id}.json"
            plan = {
                "schema_version": "dtr-c2-model-plan-v1",
                "episode_id": episode_id,
                "navigation_session_id": str(episode["navigation_session_id"]),
                "layout_anchor": {
                    "world_center_xy_m": anchor["center_xy_m"],
                    "world_forward_xy": anchor["forward_xy"],
                    "world_right_xy": anchor["right_xy"],
                },
                "issued_plan": {
                    "authority": "VALID",
                    "receipt": receipt,
                    "receipt_sha256": receipt["receipt_sha256"],
                    "world_coordinate_frame": "CARLA_WORLD_XY",
                    "time_parameterized_waypoints_world": plan_waypoints_world(
                        receipt, anchor
                    ),
                },
            }
            plan_path = model_root / plan_relative
            _write_json(plan_path, plan)
            episode_root = model_root / "episodes" / episode_id
            rgb_path = episode_root / "rgb" / "000000.png"
            depth_path = episode_root / "depth" / "000000.png"
            rgb_path.parent.mkdir(parents=True, exist_ok=True)
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_path.write_bytes(b"fixture-rgb")
            depth_path.write_bytes(b"fixture-depth")
            wearable = raw_wearable[episode_id][0]
            depth = raw_depth[episode_id][0]
            observation = {
                "schema_version": "dtr-c2-model-observation-v2",
                "episode_id": episode_id,
                "sample_index": 0,
                "world_frame": int(wearable["world_frame"]),
                "time_s": 0.0,
                "timestamp_s": 0.0,
                "wearable_rgb": {
                    "path": rgb_path.relative_to(model_root).as_posix(),
                    "bytes": rgb_path.stat().st_size,
                    "sha256": sha256_file(rgb_path),
                    "width": 1280,
                    "height": 720,
                    "source_world_frame": int(wearable["world_frame"]),
                },
                "metric_depth": {
                    "path": depth_path.relative_to(model_root).as_posix(),
                    "bytes": depth_path.stat().st_size,
                    "sha256": sha256_file(depth_path),
                    "width": 1280,
                    "height": 720,
                    "codec": DEPTH_CODEC,
                    "source_world_frame": int(depth["world_frame"]),
                },
                "camera": {
                    "world_transform": wearable["camera_transform"],
                    "rigid_extrinsic": RIGID_EXTRINSIC,
                    "width": 1280,
                    "height": 720,
                    "fov_degrees": 90.0,
                    "K": camera_intrinsics(1280, 720, 90.0),
                },
                "wearer_pose_current": wearable["wearer_transform"],
                "navigation": {
                    "navigation_session_id": str(episode["navigation_session_id"]),
                    "issued_plan": {
                        "authority": "VALID",
                        "path": plan_relative,
                        "receipt_sha256": receipt["receipt_sha256"],
                    },
                },
                "frame_alignment": {
                    "authority": alignment["authority"],
                    "reference_modality": "wearable_rgb",
                    "receipt_path": "rgbd_alignment_receipt.json",
                    "receipt_sha256": alignment["receipt_sha256"],
                    "depth_minus_wearable_source_world_frame_offset": int(
                        alignment_by_episode[episode_id][
                            "depth_minus_wearable_source_world_frame_offset"
                        ]
                    ),
                },
            }
            observations_path = episode_root / "observations.jsonl"
            _write_jsonl(observations_path, [observation])
            episode_manifest = {
                "schema_version": "dtr-c2-model-episode-manifest-v2",
                "episode_id": episode_id,
                "frames": 1,
                "observations_sha256": sha256_file(observations_path),
                "rgb_payloads": 1,
                "depth_payloads": 1,
                "navigation_session_id": str(episode["navigation_session_id"]),
                "rgbd_alignment": {
                    "authority": alignment["authority"],
                    "receipt_path": "rgbd_alignment_receipt.json",
                    "receipt_sha256": alignment["receipt_sha256"],
                    "depth_minus_wearable_source_world_frame_offset": int(
                        alignment_by_episode[episode_id][
                            "depth_minus_wearable_source_world_frame_offset"
                        ]
                    ),
                },
                "issued_plan": {
                    "authority": "VALID",
                    "path": plan_relative,
                    "receipt_sha256": receipt["receipt_sha256"],
                    "file_sha256": sha256_file(plan_path),
                },
            }
            episode_manifest_path = episode_root / "manifest.json"
            _write_json(episode_manifest_path, episode_manifest)
            episode_links.append(
                {
                    "episode_id": episode_id,
                    "manifest_path": episode_manifest_path.relative_to(
                        model_root
                    ).as_posix(),
                    "manifest_sha256": sha256_file(episode_manifest_path),
                }
            )
        _write_json(
            model_root / "manifest.json",
            {
                "schema_version": "dtr-c2-model-root-manifest-v2",
                "experiment_id": C2_EXPERIMENT_ID,
                "camera_calibration": {
                    "path": "camera_calibration.json",
                    "sha256": sha256_file(calibration_path),
                },
                "model_contract": {
                    "path": "model_contract.json",
                    "sha256": sha256_file(contract_path),
                },
                "rgbd_alignment_receipt": {
                    "path": "rgbd_alignment_receipt.json",
                    "receipt_sha256": alignment["receipt_sha256"],
                    "sha256": alignment_file_sha,
                },
                "episodes": episode_links,
            },
        )

    def _evaluator_root(
        self,
        evaluator_root: Path,
        episodes: list[tuple[str, dict[str, object]]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        for layout_id, episode in episodes:
            episode_id = str(episode["episode_id"])
            scene = self.scenes[layout_id]  # type: ignore[index]
            dynamic_ids = [
                str(actor["instance_id"])
                for actor in scene["actors"]
                if bool(self.assets_by_id[str(actor["asset_id"])]["risk_participation"])
            ]
            rows: list[dict[str, object]] = []
            for frame in range(10):
                actors: dict[str, object] = {
                    "wearer": {
                        "transform": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "roll": 0.0,
                            "pitch": 0.0,
                            "yaw": 0.0,
                        }
                    }
                }
                visibility: dict[str, object] = {}
                polygons: dict[str, object] = {}
                for target_index, target_id in enumerate(dynamic_ids):
                    x = 1.0 + frame * 0.02 + target_index * 0.01
                    y = target_index * 0.01
                    actors[target_id] = {
                        "transform": {
                            "x": x,
                            "y": y,
                            "z": 0.0,
                            "roll": 0.0,
                            "pitch": 0.0,
                            "yaw": 0.0,
                        }
                    }
                    visibility[target_id] = {"visible": True}
                    polygons[target_id] = [
                        [x - 0.1, y - 0.1],
                        [x + 0.1, y - 0.1],
                        [x + 0.1, y + 0.1],
                        [x - 0.1, y + 0.1],
                    ]
                rows.append(
                    {
                        "schema_version": "dtr-c2-evaluator-frame-v1",
                        "episode_id": episode_id,
                        "layout_id": layout_id,
                        "sample_index": frame,
                        "time_s": frame * 0.05,
                        "actors": actors,
                        "instance_visibility": visibility,
                        "truth": {"collision_polygons_xy": polygons},
                    }
                )
            _write_jsonl(
                evaluator_root / "episodes" / episode_id / "frames.jsonl",
                rows,
            )

        episodes_by_layout: dict[str, list[dict[str, object]]] = {}
        for layout_id, episode in episodes:
            episodes_by_layout.setdefault(layout_id, []).append(episode)
        occlusion_reports: list[dict[str, object]] = []
        outcomes: list[dict[str, object]] = []
        for layout_id, layout_episodes in episodes_by_layout.items():
            report_passes = layout_id == "c4_layout_01"
            episode_reports: dict[str, object] = {}
            selected_indices: dict[str, list[int]] = {}
            for episode in layout_episodes:
                episode_id = str(episode["episode_id"])
                selected = None
                runs: list[dict[str, object]] = []
                if report_passes:
                    selected = {
                        "sample_indices": list(range(10, 16)),
                        "duration_seconds": 0.30,
                        "pre_track_sample_indices": list(range(10)),
                        "post_reappearance_sample_indices": list(range(16, 26)),
                        "pre_track_frames": 10,
                        "post_reappearance_frames": 10,
                        "passed": True,
                    }
                    runs.append(selected)
                episode_reports[episode_id] = {
                    "runs": runs,
                    "selected": selected,
                    "passed": report_passes,
                }
                selected_indices[episode_id] = (
                    list(selected["sample_indices"]) if selected is not None else []
                )
                expected_outcome = str(episode["expected_outcome"])
                expected_responsible = sorted(
                    str(value) for value in episode["expected_responsible_assets"]
                )
                outcomes.append(
                    {
                        "episode_id": episode_id,
                        "layout_id": layout_id,
                        "expected_outcome": expected_outcome,
                        "observed_outcome": expected_outcome,
                        "expected_responsible_assets": expected_responsible,
                        "observed_responsible_assets": expected_responsible,
                        "first_contact_time_s": (
                            1.0 if expected_outcome == "CONTACT" else None
                        ),
                        "frames": 26,
                        "active_assets_excluding_wearer": len(
                            self.scenes[layout_id]["actors"]  # type: ignore[index]
                        ),
                    }
                )
            occlusion_reports.append(
                {
                    "contract_id": f"{layout_id}_occlusion_pair",
                    "episodes": episode_reports,
                    "pair_occlusion_indices_identical": True,
                    "selected_indices": selected_indices,
                    "passed": report_passes,
                }
            )
        _write_json(
            evaluator_root / "physical_occlusion_report.json", occlusion_reports
        )
        _write_json(evaluator_root / "outcome_summary.json", outcomes)
        return occlusion_reports, outcomes

    def _shards(
        self,
        evidence_root: Path,
        episodes: list[tuple[str, dict[str, object]]],
        map_name: str,
        protocol_sha256: str,
    ) -> None:
        for sensor in FORMAL_SENSORS:
            shard_root = evidence_root / "shards" / sensor
            inventory = [
                {
                    "episode_id": str(episode["episode_id"]),
                    "width": 1280,
                    "height": 720,
                }
                for _, episode in episodes
            ]
            inventory_path = shard_root / "payload_inventory.json"
            _write_json(inventory_path, inventory)
            calibration_path = shard_root / "camera_calibration.json"
            _write_json(
                calibration_path,
                {
                    "schema_version": "dtr-c2-camera-calibration-v1",
                    "sensor": sensor,
                    "width": 1280,
                    "height": 720,
                },
            )
            for _, episode in episodes:
                episode_id = str(episode["episode_id"])
                _write_jsonl(
                    shard_root / "episodes" / episode_id / "frames.jsonl",
                    [{"episode_id": episode_id, "sensor": sensor}],
                )
            _write_json(
                shard_root / "result.json",
                {
                    "schema_version": "dtr-carla-c2-raw-shard-result-v1",
                    "experiment_id": C2_EXPERIMENT_ID,
                    "status": C2_SHARD_STATUS,
                    "sensor": sensor,
                    "map": map_name,
                    "protocol_sha256": protocol_sha256,
                    "calibration_sha256": sha256_file(calibration_path),
                    "payload_inventory_sha256": sha256_file(inventory_path),
                    "payload_count": len(inventory),
                    "checks": {"all_formal_payloads_are_1280x720": True},
                },
            )


class C4MultimapJoinTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = C4JoinFixture(Path(self.temp.name))
        self.index_path = self.fixture.build()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _reseal_child_and_index(
        self, index: dict[str, object], group: dict[str, object]
    ) -> None:
        child_root = self.fixture.bundle_root / str(group["evidence_path"])
        model_root = child_root / "model"
        evaluator_root = child_root / "evaluator"
        _write_json(
            child_root / "sealed_model_manifest.json",
            _seal_tree(model_root, [model_root]),
        )
        _write_json(
            child_root / "sealed_evidence_manifest.json",
            _seal_tree(
                child_root,
                [child_root / "shards", model_root, evaluator_root],
            ),
        )
        result = _read_json(child_root / "result.json")
        result["sealed_model_manifest_sha256"] = sha256_file(
            child_root / "sealed_model_manifest.json"
        )
        result["sealed_evidence_manifest_sha256"] = sha256_file(
            child_root / "sealed_evidence_manifest.json"
        )
        _write_json(child_root / "result.json", result)
        group["evidence_result_sha256"] = sha256_file(child_root / "result.json")
        _write_json(self.index_path, index)

    def _group_with_passing_report(
        self, index: dict[str, object]
    ) -> dict[str, object]:
        for group in index["map_layout_groups"]:  # type: ignore[index]
            child_root = self.fixture.bundle_root / str(group["evidence_path"])
            reports = json.loads(
                (child_root / "evaluator" / "physical_occlusion_report.json").read_text(
                    encoding="utf-8"
                )
            )
            if any(bool(value["passed"]) for value in reports):
                return group
        raise AssertionError("fixture has no passing physical occlusion report")

    def test_final_index_exact_shape_joins_all_eight_layout_families(self) -> None:
        index = _read_json(self.index_path)
        self.assertEqual(INDEX_EXACT_KEYS, set(index))
        self.assertEqual(
            {"asset_registry", "scene_registry"}, set(index["registries"])
        )
        self.assertTrue(
            all(
                set(link) == {"path", "sha256"}
                for link in index["registries"].values()
            )
        )
        self.assertTrue(
            all(set(group) == GROUP_EXACT_KEYS for group in index["map_layout_groups"])
        )
        child_results_before = {
            str(group["group_id"]): sha256_file(
                self.fixture.bundle_root
                / str(group["evidence_path"])
                / "result.json"
            )
            for group in index["map_layout_groups"]
        }
        child_statuses = {
            _read_json(
                self.fixture.bundle_root
                / str(group["evidence_path"])
                / "result.json"
            )["status"]
            for group in index["map_layout_groups"]
        }
        self.assertEqual({C2_RESULT_NOT_EVALUABLE_STATUS}, child_statuses)

        result = join_multimap(self.index_path, self.fixture.output_root)

        self.assertEqual(C4_RESULT_STATUS, result["status"])
        self.assertEqual(8, result["layout_family_count"])
        self.assertEqual(8, result["layout_count"])
        self.assertEqual(16, result["episode_count"])
        self.assertTrue(all(result["checks"].values()), result)
        audit = _read_json(
            self.fixture.output_root / "evaluator" / "layout_coverage_audit.json"
        )
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(8, len(audit["layout_families"]))
        self.assertTrue(
            all(value["episodes"] for value in audit["layout_families"].values())
        )
        occlusion_audit = _read_json(
            self.fixture.output_root / "evaluator" / "pack_occlusion_audit.json"
        )
        self.assertEqual(C4_OCCLUSION_AUDIT_SCHEMA, occlusion_audit["schema_version"])
        self.assertTrue(occlusion_audit["passed"], occlusion_audit)
        self.assertEqual(8, occlusion_audit["layout_report_count"])
        self.assertEqual(6, len(occlusion_audit["child_sources"]))
        self.assertTrue(
            all(
                value["child_result_status"] == C2_RESULT_NOT_EVALUABLE_STATUS
                and value["physical_occlusion_report"]["sealed_and_result_bound"]
                for value in occlusion_audit["child_sources"]
            )
        )
        self.assertEqual(1, len(occlusion_audit["qualifying_pairs"]))
        self.assertEqual(
            "c4_layout_01", occlusion_audit["qualifying_pairs"][0]["layout_id"]
        )
        self.assertEqual(
            ["CONTACT", "SAFE"],
            next(
                value
                for value in occlusion_audit["reports"]
                if value["passed"]
            )["observed_outcome_set"],
        )
        self.assertEqual(
            "evaluator/pack_occlusion_audit.json",
            result["pack_occlusion_audit"]["path"],
        )
        evaluator_manifest = _read_json(
            self.fixture.output_root / "evaluator" / "manifest.json"
        )
        self.assertEqual(
            "pack_occlusion_audit.json",
            evaluator_manifest["pack_occlusion_audit"]["path"],
        )
        sealed_evidence = json.loads(
            (self.fixture.output_root / "sealed_evidence_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn(
            "evaluator/pack_occlusion_audit.json",
            {value["path"] for value in sealed_evidence},
        )
        model_manifest = _read_json(self.fixture.output_root / "model" / "manifest.json")
        self.assertEqual(
            {"schema_version", "experiment_id", "groups"}, set(model_manifest)
        )
        self.assertFalse((self.fixture.output_root / "model" / "registries").exists())
        self.assertEqual([], result["model_truth_failures"])
        first_group = index["map_layout_groups"][0]
        child_model = (
            self.fixture.bundle_root
            / str(first_group["evidence_path"])
            / "model"
        )
        child_payload = next(child_model.glob("episodes/*/rgb/*.png"))
        packaged_payload = (
            self.fixture.output_root
            / "model"
            / "groups"
            / str(first_group["group_id"])
            / child_payload.relative_to(child_model)
        )
        self.assertTrue(os.path.samefile(child_payload, packaged_payload))
        for group in index["map_layout_groups"]:
            child_result = (
                self.fixture.bundle_root
                / str(group["evidence_path"])
                / "result.json"
            )
            self.assertEqual(
                child_results_before[str(group["group_id"])], sha256_file(child_result)
            )

    def test_rejects_redundant_evaluator_truth_in_group(self) -> None:
        index = _read_json(self.index_path)
        index["map_layout_groups"][0]["episodes"] = []
        _write_json(self.index_path, index)
        with self.assertRaisesRegex(ValueError, "keys differ"):
            join_multimap(self.index_path, self.fixture.output_root)
        self.assertFalse(self.fixture.output_root.exists())

    def test_rejects_child_live_tree_tamper_before_output(self) -> None:
        index = _read_json(self.index_path)
        group = index["map_layout_groups"][0]
        child_root = self.fixture.bundle_root / str(group["evidence_path"])
        payload = next((child_root / "model").glob("episodes/*/rgb/*.png"))
        payload.write_bytes(b"tampered")
        with self.assertRaisesRegex(C4ContractError, "sealed model entries differ"):
            join_multimap(self.index_path, self.fixture.output_root)
        self.assertFalse(self.fixture.output_root.exists())

    def test_rejects_sealed_physical_report_that_differs_from_child_result(self) -> None:
        index = _read_json(self.index_path)
        group = self._group_with_passing_report(index)
        child_root = self.fixture.bundle_root / str(group["evidence_path"])
        report_path = child_root / "evaluator" / "physical_occlusion_report.json"
        reports = json.loads(report_path.read_text(encoding="utf-8"))
        reports[0]["passed"] = not bool(reports[0]["passed"])
        _write_json(report_path, reports)
        self._reseal_child_and_index(index, group)

        with self.assertRaisesRegex(
            C4ContractError, "differs from child result.occlusion_reports"
        ):
            join_multimap(self.index_path, self.fixture.output_root)
        self.assertFalse(self.fixture.output_root.exists())

    def test_rejects_physical_report_missing_from_sealed_evidence(self) -> None:
        index = _read_json(self.index_path)
        group = self._group_with_passing_report(index)
        child_root = self.fixture.bundle_root / str(group["evidence_path"])
        (child_root / "evaluator" / "physical_occlusion_report.json").unlink()
        self._reseal_child_and_index(index, group)

        with self.assertRaisesRegex(
            C4ContractError, "missing from sealed evidence"
        ):
            join_multimap(self.index_path, self.fixture.output_root)
        self.assertFalse(self.fixture.output_root.exists())

    def test_pack_gate_is_not_met_when_no_report_has_valid_geometry(self) -> None:
        index = _read_json(self.index_path)
        group = self._group_with_passing_report(index)
        child_root = self.fixture.bundle_root / str(group["evidence_path"])
        report_path = child_root / "evaluator" / "physical_occlusion_report.json"
        reports = json.loads(report_path.read_text(encoding="utf-8"))
        for report in reports:
            if not bool(report["passed"]):
                continue
            report["passed"] = False
            for episode_id, episode_report in report["episodes"].items():
                episode_report["runs"] = []
                episode_report["selected"] = None
                episode_report["passed"] = False
                report["selected_indices"][episode_id] = []
        _write_json(report_path, reports)
        child_result_path = child_root / "result.json"
        child_result = _read_json(child_result_path)
        child_result["occlusion_reports"] = reports
        _write_json(child_result_path, child_result)
        self._reseal_child_and_index(index, group)

        result = join_multimap(self.index_path, self.fixture.output_root)

        self.assertEqual("DTR_CARLA_C4_MULTIMAP_GATE_NOT_MET", result["status"])
        self.assertFalse(
            result["checks"]["pack_level_contact_safe_physical_occlusion_pair_met"]
        )
        audit = _read_json(
            self.fixture.output_root / "evaluator" / "pack_occlusion_audit.json"
        )
        self.assertFalse(audit["passed"])
        self.assertEqual([], audit["qualifying_pairs"])

    def test_pack_gate_uses_child_result_contact_safe_outcomes(self) -> None:
        index = _read_json(self.index_path)
        group = self._group_with_passing_report(index)
        child_root = self.fixture.bundle_root / str(group["evidence_path"])
        child_result_path = child_root / "result.json"
        child_result = _read_json(child_result_path)
        passing_episode_ids = {
            episode_id
            for report in child_result["occlusion_reports"]
            if report["passed"]
            for episode_id in report["episodes"]
        }
        for outcome in child_result["outcomes"]:
            if (
                outcome["episode_id"] in passing_episode_ids
                and outcome["observed_outcome"] == "CONTACT"
            ):
                outcome["observed_outcome"] = "SAFE"
        child_result["checks"]["contact_safe_outcome_pair_matches"] = False
        child_result["status"] = C2_RESULT_NOT_EVALUABLE_STATUS
        _write_json(child_result_path, child_result)
        self._reseal_child_and_index(index, group)

        result = join_multimap(self.index_path, self.fixture.output_root)

        self.assertEqual("DTR_CARLA_C4_MULTIMAP_GATE_NOT_MET", result["status"])
        audit = _read_json(
            self.fixture.output_root / "evaluator" / "pack_occlusion_audit.json"
        )
        passing_geometry = next(
            value for value in audit["reports"] if value["layout_id"] == "c4_layout_01"
        )
        self.assertFalse(
            passing_geometry["checks"]["observed_outcomes_are_contact_and_safe"]
        )

    def test_rejects_non_pack_child_check_failure(self) -> None:
        index = _read_json(self.index_path)
        group = index["map_layout_groups"][0]
        child_root = self.fixture.bundle_root / str(group["evidence_path"])
        child_result_path = child_root / "result.json"
        child_result = _read_json(child_result_path)
        child_result["checks"]["fixture_complete"] = False
        child_result["status"] = C2_RESULT_NOT_EVALUABLE_STATUS
        _write_json(child_result_path, child_result)
        group["evidence_result_sha256"] = sha256_file(child_result_path)
        _write_json(self.index_path, index)

        with self.assertRaisesRegex(C4ContractError, "non-pack checks failed"):
            join_multimap(self.index_path, self.fixture.output_root)
        self.assertFalse(self.fixture.output_root.exists())

    def test_rejects_actor_outcome_twin_and_scenario_semantics_in_model(self) -> None:
        index = _read_json(self.index_path)
        group = index["map_layout_groups"][0]
        child_root = self.fixture.bundle_root / str(group["evidence_path"])
        _write_json(
            child_root / "model" / "neutral_notes.json",
            {
                "actor_hint": "hidden",
                "outcome_hint": "unknown",
                "twin_hint": "paired",
                "scenario_hint": "layout",
            },
        )
        self._reseal_child_and_index(index, group)
        with self.assertRaisesRegex(C4ContractError, "semantic truth audit"):
            join_multimap(self.index_path, self.fixture.output_root)
        self.assertFalse(self.fixture.output_root.exists())

    def test_stationary_dynamic_target_fails_episode_gate(self) -> None:
        evaluator_root = Path(self.temp.name) / "stationary" / "evaluator"
        episode_id = "c4_l01_e01"
        rows = []
        for frame in range(10):
            rows.append(
                {
                    "episode_id": episode_id,
                    "actors": {
                        "wearer": {"transform": {"x": 0.0, "y": 0.0}},
                        "moving_one": {"transform": {"x": 1.0, "y": 0.0}},
                    },
                    "instance_visibility": {"moving_one": {"visible": True}},
                    "truth": {
                        "collision_polygons_xy": {
                            "moving_one": [[0.9, -0.1], [1.1, -0.1], [1.1, 0.1], [0.9, 0.1]]
                        }
                    },
                }
            )
        _write_jsonl(
            evaluator_root / "episodes" / episode_id / "frames.jsonl", rows
        )
        audit = _episode_dynamic_audit(
            evaluator_root,
            episode_id,
            ["moving_one"],
            minimum_visible_frames=10,
            risk_corridor_threshold_m=3.0,
            wearer_radius_m=0.45,
        )
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["targets"]["moving_one"]["checks"]["observed_transform_motion"])

    def test_rejects_registry_hash_drift_before_output(self) -> None:
        index = _read_json(self.index_path)
        registry_path = self.fixture.bundle_root / str(
            index["registries"]["scene_registry"]["path"]
        )
        registry = _read_json(registry_path)
        registry["description"] = "drift"
        _write_json(registry_path, registry)
        with self.assertRaisesRegex(C4ContractError, "hash differs"):
            join_multimap(self.index_path, self.fixture.output_root)
        self.assertFalse(self.fixture.output_root.exists())


if __name__ == "__main__":
    unittest.main()
