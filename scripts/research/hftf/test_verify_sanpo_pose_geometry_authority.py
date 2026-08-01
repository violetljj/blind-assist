from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_sanpo_pose_geometry_authority import audit


def _md5_base64(path: Path) -> str:
    return base64.b64encode(
        hashlib.md5(path.read_bytes(), usedforsecurity=False).digest()
    ).decode("ascii")


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> list[float]:
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.tolist()


class SanpoPoseGeometryAuthorityTest(unittest.TestCase):
    def _make_official_repo(self, root: Path) -> tuple[Path, str, str]:
        repo = root / "official"
        common = repo / "sanpo_dataset" / "lib" / "common.py"
        common.parent.mkdir(parents=True)
        common.write_text(
            "\n".join(
                [
                    "FILENAME_RGB = '{frame_num:06d}.png'",
                    "FILENAME_DEPTH = '{frame_num:06d}.float16.gz'",
                    (
                        "FEATURE_CAMERA_TRANSLATIONS = "
                        "'camera_translation_in_m'"
                    ),
                    (
                        "FEATURE_CAMERA_QUATERNIONS = "
                        "'camera_quaternions_right_handed_y_up'"
                    ),
                    "for frame_num in range(self.n_frames(sensor_name)):",
                    "    pose = self.camera_poses(sensor_name)[frame_num]",
                    (
                        "    rgb = FILENAME_RGB.format("
                        "frame_num=self.frame_num)"
                    ),
                    (
                        "    depth = FILENAME_DEPTH.format("
                        "frame_num=self.frame_num)"
                    ),
                    (
                        "    quaternion = [line['q_x'], line['q_y'], "
                        "line['q_z'], line['q_w']]"
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "remote",
                "add",
                "origin",
                "https://github.com/google-research-datasets/"
                "sanpo_dataset.git",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "add", "sanpo_dataset/lib/common.py"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-m",
                "fixture",
            ],
            check=True,
            capture_output=True,
        )
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        common_sha = hashlib.sha256(common.read_bytes()).hexdigest()
        return repo, commit, common_sha

    def _make_replay(
        self,
        root: Path,
        *,
        include_ground: bool = True,
        bad_timestamp: bool = False,
    ) -> tuple[Path, dict[str, object], dict[str, object]]:
        replay = root / "replay"
        for relative in ("images", "masks", "depth", "source_metadata"):
            (replay / relative).mkdir(parents=True, exist_ok=True)

        width, height = 160, 120
        fx = fy = 125.0
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        frame_count = 6
        rotations: list[np.ndarray] = []
        translations: list[np.ndarray] = []
        base = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        for index in range(frame_count):
            yaw = math.radians(-35.0 + 14.0 * index)
            pitch = math.radians(-15.0 + 6.0 * index)
            roll = math.radians(-10.0 + 4.0 * index)
            yaw_rotation = np.asarray(
                [
                    [math.cos(yaw), -math.sin(yaw), 0.0],
                    [math.sin(yaw), math.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            pitch_rotation = np.asarray(
                [
                    [math.cos(pitch), 0.0, math.sin(pitch)],
                    [0.0, 1.0, 0.0],
                    [-math.sin(pitch), 0.0, math.cos(pitch)],
                ]
            )
            roll_rotation = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, math.cos(roll), -math.sin(roll)],
                    [0.0, math.sin(roll), math.cos(roll)],
                ]
            )
            rotations.append(
                yaw_rotation @ pitch_rotation @ roll_rotation @ base
            )
            translations.append(
                np.asarray(
                    [0.4 * index, 0.08 * index * index, 1.4]
                )
            )

        pose_path = replay / "source_metadata" / "camera_poses.csv"
        with pose_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "tracking_state",
                "pos_x",
                "pos_y",
                "pos_z",
                "q_x",
                "q_y",
                "q_z",
                "q_w",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for rotation, translation in zip(rotations, translations):
                quaternion = _matrix_to_quaternion_xyzw(rotation)
                writer.writerow(
                    {
                        "tracking_state": "TrackingState.READY",
                        "pos_x": translation[0],
                        "pos_y": translation[1],
                        "pos_z": translation[2],
                        "q_x": quaternion[0],
                        "q_y": quaternion[1],
                        "q_z": quaternion[2],
                        "q_w": quaternion[3],
                    }
                )

        description_path = (
            replay / "source_metadata" / "source_session_description.json"
        )
        description_path.write_text(
            json.dumps(
                {
                    "session_camera_details": [
                        {"camera_name": "camera_chest", "fps": 20.0}
                    ]
                }
            ),
            encoding="utf-8",
        )

        def inventory(path: Path, name: str) -> dict[str, object]:
            return {
                "name": name,
                "generation": "1",
                "size": path.stat().st_size,
                "md5_base64": _md5_base64(path),
                "crc32c_base64": "fixture-crc",
            }

        description_inventory = inventory(
            description_path, "fixture/session/description.json"
        )
        pose_inventory = inventory(
            pose_path, "fixture/session/camera_poses.csv"
        )

        u, v = np.meshgrid(np.arange(width), np.arange(height))
        rays_camera = np.stack(
            (
                (u - cx) / fx,
                (v - cy) / fy,
                np.ones_like(u, dtype=np.float64),
            ),
            axis=0,
        ).reshape(3, -1)
        manifest_rows: list[dict[str, object]] = []
        for index, (rotation, translation) in enumerate(
            zip(rotations, translations)
        ):
            directions = rotation @ rays_camera
            candidates: list[tuple[np.ndarray, int]] = []
            if include_ground:
                ground_scale = np.where(
                    directions[2] < -1e-8,
                    -translation[2] / directions[2],
                    np.inf,
                )
                candidates.append((ground_scale, 3))
            for axis, coordinate in ((0, 15.0), (1, -10.0), (1, 10.0)):
                scale = np.where(
                    np.abs(directions[axis]) > 1e-8,
                    (coordinate - translation[axis]) / directions[axis],
                    np.inf,
                )
                scale = np.where(scale > 0.0, scale, np.inf)
                candidates.append((scale, 7))
            scales = np.stack([item[0] for item in candidates], axis=0)
            nearest = np.argmin(scales, axis=0)
            depth = np.min(scales, axis=0)
            depth = np.where(np.isfinite(depth), depth, 30.0)
            semantic = np.asarray(
                [candidates[item][1] for item in nearest], dtype=np.uint8
            )
            depth = depth.reshape(height, width).astype("<f2")
            semantic = semantic.reshape(height, width)

            image_path = replay / "images" / f"{index:06d}.png"
            mask_path = replay / "masks" / f"{index:06d}.png"
            depth_path = replay / "depth" / f"{index:06d}.float16.gz"
            Image.new("RGB", (width, height), (10, 20, 30)).save(image_path)
            mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            mask_rgb[..., 0] = semantic
            Image.fromarray(mask_rgb, mode="RGB").save(mask_path)
            payload = np.concatenate(
                (
                    np.asarray([height, width], dtype="<f2"),
                    depth.reshape(-1),
                )
            ).tobytes()
            depth_path.write_bytes(gzip.compress(payload))
            manifest_rows.append(
                {
                    "id": f"fixture-{index}",
                    "width": width,
                    "height": height,
                    "session_id": "fixture-session",
                    "sequence_id": "fixture-sequence",
                    "source_frame_index": index,
                    "source_timestamp_ms": (
                        999 if bad_timestamp and index == 0 else index * 50
                    ),
                    "source_mask_path": f"masks/{index:06d}.png",
                    "source_depth_path": f"depth/{index:06d}.float16.gz",
                    "modalities": {
                        "rgb": {"name": f"frames/{index:06d}.png"},
                        "panoptic_mask": {
                            "name": f"masks/{index:06d}.png"
                        },
                        "metric_depth": {
                            "name": f"depth/{index:06d}.float16.gz"
                        },
                    },
                }
            )

        (replay / "manifest.replay.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in manifest_rows),
            encoding="utf-8",
        )
        (replay / "dataset_spec.json").write_text(
            json.dumps(
                {
                    "schema": "blindassist_sanpo_synthetic_replay_v1",
                    "sampling": {
                        "source_fps": 20.0,
                        "selected_source_frames": list(range(frame_count)),
                    },
                    "camera": {
                        "fx": fx,
                        "fy": fy,
                        "cx": cx,
                        "cy": cy,
                        "image_width": width,
                        "image_height": height,
                    },
                    "source_inventory": {
                        "description": description_inventory,
                        "camera_poses": pose_inventory,
                    },
                }
            ),
            encoding="utf-8",
        )
        return replay, description_inventory, pose_inventory

    def _audit(
        self,
        root: Path,
        *,
        include_ground: bool = True,
        bad_timestamp: bool = False,
        corrupt_live_pose: bool = False,
        corrupt_common_hash: bool = False,
        evaluation_mode: str = "discovery",
    ) -> dict[str, object]:
        repo, commit, common_sha = self._make_official_repo(root)
        replay, description, poses = self._make_replay(
            root,
            include_ground=include_ground,
            bad_timestamp=bad_timestamp,
        )
        live_description = {
            "name": description["name"],
            "generation": description["generation"],
            "size": description["size"],
            "md5Hash": description["md5_base64"],
            "crc32c": description["crc32c_base64"],
        }
        live_poses = {
            "name": poses["name"],
            "generation": poses["generation"],
            "size": poses["size"],
            "md5Hash": (
                "wrong" if corrupt_live_pose else poses["md5_base64"]
            ),
            "crc32c": poses["crc32c_base64"],
        }
        return audit(
            replay,
            repo,
            expected_official_commit=commit,
            expected_common_sha256=(
                "0" * 64 if corrupt_common_hash else common_sha
            ),
            live_description_metadata=live_description,
            live_pose_metadata=live_poses,
            reprojection_stride=5,
            ground_stride=4,
            evaluation_mode=evaluation_mode,
        )

    def test_source_derived_proxy_frame_is_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = self._audit(Path(temp_dir))

        self.assertEqual(
            report["terminal"], "HFTF_H0_1_SANPO_PROXY_FRAME_ADMITTED"
        )
        transform = report["transform_direction_canary"]
        self.assertEqual(transform["best"]["orientation_hypothesis"], "R")
        self.assertEqual(
            transform["best"]["camera_basis_rows"], np.eye(3, dtype=int).tolist()
        )
        self.assertEqual(
            report["ground_and_body_proxy_canary"]["vertical_axis"], "+Z"
        )
        self.assertEqual(
            report["capability_decisions"][
                "physical_camera_to_person_calibration"
            ],
            "NOT_EVALUABLE",
        )

    def test_live_pose_identity_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = self._audit(Path(temp_dir), corrupt_live_pose=True)

        self.assertEqual(
            report["terminal"],
            "HFTF_H0_1_SOURCE_AUTHORITY_NOT_EVALUABLE",
        )
        self.assertFalse(report["source_pose_authority"]["ok"])

    def test_frozen_canonical_mode_reports_h0_2_replication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = self._audit(
                Path(temp_dir),
                evaluation_mode="frozen_canonical_replication",
            )

        self.assertEqual(
            report["terminal"],
            "HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED",
        )
        self.assertTrue(
            report["transform_direction_canary"][
                "frozen_canonical_replication_admitted"
            ]
        )
        self.assertEqual(
            report["allowed_next_step"], "H0_2_COHORT_AGGREGATION"
        )

    def test_manifest_timestamp_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = self._audit(Path(temp_dir), bad_timestamp=True)

        self.assertEqual(
            report["terminal"],
            "HFTF_H0_1_SOURCE_AUTHORITY_NOT_EVALUABLE",
        )

    def test_official_loader_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = self._audit(Path(temp_dir), corrupt_common_hash=True)

        self.assertEqual(
            report["terminal"],
            "HFTF_H0_1_SOURCE_AUTHORITY_NOT_EVALUABLE",
        )
        self.assertFalse(report["official_loader_authority"]["ok"])

    def test_absent_ground_supports_mapping_but_not_proxy_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = self._audit(Path(temp_dir), include_ground=False)

        self.assertEqual(report["terminal"], "HFTF_H0_1_POSE_MAPPING_ONLY")
        self.assertEqual(
            report["capability_decisions"]["official_frame_pose_row_mapping"],
            "ELIGIBLE",
        )
        self.assertEqual(
            report["capability_decisions"][
                "standard_body_proxy_for_h1_geometry_mechanics"
            ],
            "NOT_EVALUABLE",
        )


if __name__ == "__main__":
    unittest.main()
