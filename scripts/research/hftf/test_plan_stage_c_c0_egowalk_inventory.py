from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_c0_egowalk_inventory import plan


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StageCC0EgoWalkInventoryTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, dict[str, object]]:
        docs = root / "docs/research/hftf"
        docs.mkdir(parents=True)
        parent = docs / "parent.md"
        parent.write_text("parent", encoding="utf-8")
        metadata = root / "metadata"
        meta = metadata / "meta"
        data = metadata / "data"
        meta.mkdir(parents=True)
        data.mkdir()
        trajectories = ["2024_01_01__00_00_00", "2024_01_02__00_00_00"]
        (meta / "info.json").write_text('{"fps":5}', encoding="utf-8")
        (meta / "camera_rgb.json").write_text(
            '{"fx":1}', encoding="utf-8"
        )
        (meta / "heights.json").write_text(
            json.dumps({name: 1.3 for name in trajectories}),
            encoding="utf-8",
        )
        (meta / "trajectories.json").write_text(
            json.dumps(trajectories), encoding="utf-8"
        )
        repo_files: dict[str, object] = {}
        for rank, name in enumerate(trajectories):
            rows = 320
            frame = list(range(rows))
            table = pl.DataFrame(
                {
                    "timestamp": [1000 + 200 * value for value in frame],
                    "trajectory": [name] * rows,
                    "frame": frame,
                    "cart_x": [0.02 * value for value in frame],
                    "cart_y": [0.0] * rows,
                    "cart_z": [0.0] * rows,
                    "quat_x": [0.0] * rows,
                    "quat_y": [0.0] * rows,
                    "quat_z": [0.0] * rows,
                    "quat_w": [1.0] * rows,
                }
            )
            parquet = data / f"{name}.parquet"
            table.write_parquet(parquet)
            sizes = {"pose": 10 + rank, "rgb": 20 + rank, "depth": 30 + rank}
            paths = {
                "pose": f"data/{name}.parquet",
                "rgb": f"video/rgb/{name}__rgb.mp4",
                "depth": f"video/depth/{name}__depth.mkv",
            }
            for role, path in paths.items():
                repo_files[path] = {
                    "size_bytes": sizes[role],
                    "sha256": (
                        _sha256(parquet) if role == "pose" else role * 64
                    )[:64],
                }
        protocol = {
            "schema": "blindassist_hftf_stage_c_source_feasibility_c0",
            "status": "FROZEN_BEFORE_C0_MEDIA_CONTENT_OR_GEOMETRY_OUTCOME",
            "workflow_profile": "DEVELOPMENT_STANDARD",
            "parent_result_path": "parent.md",
            "parent_result_sha256": _sha256(parent),
            "canonical_temporal_contract": {"timeline_hz": 5},
            "source_roles": {
                "egowalk": {
                    "dataset_repo": "test/repo",
                    "repo_type": "dataset",
                    "revision": "abc",
                    "metadata_bindings": {
                        "trajectory_count": 2,
                        "info_json_sha256": _sha256(meta / "info.json"),
                        "camera_rgb_json_sha256": _sha256(
                            meta / "camera_rgb.json"
                        ),
                        "heights_json_sha256": _sha256(
                            meta / "heights.json"
                        ),
                        "trajectories_json_sha256": _sha256(
                            meta / "trajectories.json"
                        ),
                    },
                    "metadata_health_gates": {
                        "minimum_rows": 320,
                        "median_timestamp_delta_ms_inclusive": [195, 205],
                        "every_timestamp_delta_ms_inclusive": [150, 250],
                        "maximum_quaternion_abs_norm_error": 0.001,
                        "minimum_max_distance_from_start_m": 5.0,
                        "maximum_single_step_translation_m": 1.5,
                    },
                    "selection_rule": {
                        "required_trajectories": 2,
                        "order": "ascending_total_bytes_pose_plus_rgb_plus_depth_then_trajectory_id",
                    },
                    "expected_metadata_only_selected_cohort": trajectories,
                }
            },
        }
        protocol_path = docs / "protocol.json"
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        return protocol_path, metadata, repo_files

    def test_selection_is_size_ordered_and_date_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            protocol, metadata, files = self._fixture(Path(temp))
            report = plan(protocol, metadata, files)
            self.assertEqual(
                "C0_EGOWALK_METADATA_COHORT_LOCKED",
                report["terminal"],
            )
            self.assertEqual(2, report["metadata_healthy_count"])
            self.assertFalse(report["rgb_or_depth_media_content_read"])

    def test_pose_null_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            protocol, metadata, files = self._fixture(Path(temp))
            target = metadata / "data/2024_01_01__00_00_00.parquet"
            frame = pl.read_parquet(target).with_columns(
                pl.when(pl.col("frame") == 7)
                .then(None)
                .otherwise(pl.col("cart_x"))
                .alias("cart_x")
            )
            frame.write_parquet(target)
            files["data/2024_01_01__00_00_00.parquet"]["sha256"] = _sha256(
                target
            )
            report = plan(protocol, metadata, files)
            self.assertEqual(
                "C0_EGOWALK_METADATA_SELECTION_NOT_EVALUABLE",
                report["terminal"],
            )
            first = next(
                item
                for item in report["inventory_ledger"]
                if item["trajectory"] == "2024_01_01__00_00_00"
            )
            self.assertIn("pose_null_frames", first["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
