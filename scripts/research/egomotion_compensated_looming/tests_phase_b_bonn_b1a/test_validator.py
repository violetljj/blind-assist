from __future__ import annotations

from decimal import Decimal
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
from PIL import Image

from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_b1a import (
    validator,
)


def pose_row(
    timestamp: str,
    translation: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
    rank: int,
) -> validator.PoseRow:
    return validator.PoseRow(
        Decimal(timestamp),
        np.asarray(translation, dtype=np.float64),
        np.asarray(quaternion, dtype=np.float64),
        rank,
    )


def depth_png(value: int = 5000) -> bytes:
    array = np.full((480, 640), value, dtype=np.uint16)
    buffer = BytesIO()
    Image.fromarray(array, mode="I;16").save(buffer, format="PNG")
    return buffer.getvalue()


def base_receipt() -> dict[str, object]:
    windows = validator._canonical_windows()
    window_results = [
        {
            "sequence_id": row["sequence_id"],
            "window_rank": row["window_rank"],
            "candidate_pair_count": 1,
            "truth_covered_pair_count": 1,
            "role": (
                validator.ROTATION_ROLE
                if index in (0, 2)
                else validator.NO_ROLE
            ),
        }
        for index, row in enumerate(windows)
    ]
    ledger: dict[str, object] = {
        "schema_version": "rcle.phase_b.bonn_b1a.ledger.v1",
        "windows": windows,
        "pairs": [],
        "window_results": window_results,
    }
    claim = {
        "argv": [],
        "design_lock_sha256": validator.DESIGN_LOCK_SHA256,
        "preregistration_sha256": validator.PREREGISTRATION_SHA256,
        "b0_receipt_sha256": validator.B0_RECEIPT_SHA256,
        "window_denominator_sha256": validator.WINDOW_DENOMINATOR_SHA256,
        "implementation_lock_sha256": "1" * 64,
        "bootstrap_runner_sha256": "2" * 64,
        "delete_replace_or_rewrite_claim": "FORBIDDEN",
        "exclusive_create": True,
        "maximum_claims": 1,
        "application_data_operations_before_claim": 0,
        "claim_permanently_retained": True,
        "success_failure_exception_or_interrupt_consumes_claim": True,
    }
    archive_hashes = {
        sequence_id: f"{rank:064x}"
        for rank, sequence_id in enumerate(
            {
                row["sequence_id"] for row in windows
            },
            start=1,
        )
    }
    return {
        "schema_version": "rcle.phase_b.bonn_b1a.receipt.v1",
        "protocol_id": validator.PROTOCOL_ID,
        "created_at": "fixture",
        "environment": {
            "python": "3.11.9",
            "numpy": "2.1.3",
            "pillow": "12.2.0",
        },
        "design_lock_sha256": validator.DESIGN_LOCK_SHA256,
        "preregistration_sha256": validator.PREREGISTRATION_SHA256,
        "implementation_lock_sha256": "1" * 64,
        "bootstrap_runner_sha256": "2" * 64,
        "b0_receipt_sha256": validator.B0_RECEIPT_SHA256,
        "window_denominator_sha256": validator.WINDOW_DENOMINATOR_SHA256,
        "cohort_identity_sha256": validator.COHORT_IDENTITY_SHA256,
        "source_authority_sha256_by_id": dict(
            validator.SOURCE_AUTHORITY_SHA256
        ),
        "archive_sha256_by_sequence": archive_hashes,
        "run_claim_sha256": hashlib.sha256(claim_bytes(claim)).hexdigest(),
        "claim": claim,
        "read_firewall": {
            "rgb_member_bytes_read": 0,
            "rgb_decode_operations": 0,
            "phase_b_metric_operations": 0,
            "static_map_reads": 0,
            "legacy_bonn_outcome_reads": 0,
            "network_operations": 0,
            "depth_decode_operations": 0,
            "pose_numeric_rows_parsed": 0,
        },
        "ledger_identity_sha256": validator.canonical_sha256(ledger),
        "ledger": ledger,
        "branch_distinct_sequence_counts": {
            validator.ROTATION_ROLE: 2,
            validator.APPROACH_ROLE: 0,
        },
        "gate_pass": True,
        "terminal_state": validator.PASS_TERMINAL,
        "b1b_branch_scope_may_be_reviewed": True,
        "execution_authority_consumed": True,
        "b1b_implementation_authorized": False,
    }


def claim_bytes(claim: dict[str, object]) -> bytes:
    return (validator.canonical_json(claim) + "\n").encode("utf-8")


class B1AIndependentValidatorTest(unittest.TestCase):
    def test_validate_existing_accepts_only_verifiable_failure_receipt_as_invalid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            claim = {
                "bootstrap_runner_sha256": "a" * 64,
                "exclusive_create": True,
            }
            raw = claim_bytes(claim)
            (output / "run_claim.json").write_bytes(raw)
            failure = {
                "bootstrap_runner_sha256": "a" * 64,
                "claim": claim,
                "run_claim_sha256": hashlib.sha256(raw).hexdigest(),
                "schema_version": "rcle.phase_b.bonn.b1a.failure_receipt.v1",
                "terminal_state": validator.INVALID_TERMINAL,
            }
            (output / "failure_receipt.json").write_text(
                validator.canonical_json(failure) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                validator,
                "_canonical_paths",
                return_value={"output": output},
            ):
                result = validator.validate_existing(root)
            self.assertEqual(result["status"], "INVALID")
            self.assertEqual(
                result["errors"], ["FORMAL_EXECUTION_FAILURE_RECEIPT"]
            )

    def test_text_contract_and_member_paths(self) -> None:
        rows = validator.parse_index_text(
            b"# comment\r\n 1.0 rgb/1.png \n1.03 rgb/2.png\n"
        )
        self.assertEqual([row.source_row_rank for row in rows], [0, 1])
        self.assertEqual(rows[1].timestamp, Decimal("1.03"))
        poses = validator.parse_pose_text(
            b"1 0 0 0 0 0 0 1\n1.03 1 2 3 0 0 0 1\n"
        )
        self.assertEqual(len(poses), 2)
        for bad in (
            "../x.png",
            "/x.png",
            "C:/x.png",
            "a\\b.png",
            "a/./b.png",
            "a\u00a0b.png",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validator.validate_member_reference(bad)

    def test_text_rejects_token_count_nonfinite_and_order(self) -> None:
        for payload in (
            b"1 only extra\n",
            b"NaN rgb/x.png\n",
            b"1 rgb/a.png\n1 rgb/b.png\n",
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validator.parse_index_text(payload)
        with self.assertRaisesRegex(ValueError, "POSE_TOKEN_COUNT"):
            validator.parse_pose_text(b"1 0 0 0 0 0 1\n")

    def test_adjacent_candidate_denominator_is_frozen_before_joins(self) -> None:
        rows = validator.parse_index_text(
            b"0 rgb/0.png\n.019 rgb/1.png\n.039 rgb/2.png\n.089 rgb/3.png\n"
        )
        pairs = validator.adjacent_pairs(rows, Decimal("0"), Decimal("0.1"))
        self.assertEqual([pair["candidate"] for pair in pairs], [False, True, True])
        self.assertEqual([pair["dt"] for pair in pairs], [
            Decimal(".019"),
            Decimal(".020"),
            Decimal(".050"),
        ])

    def test_depth_join_is_window_local_greedy_and_one_to_one(self) -> None:
        rgb = validator.parse_index_text(
            b"0.100 rgb/a.png\n0.110 rgb/b.png\n0.200 rgb/c.png\n"
        )
        depth = validator.parse_index_text(
            b"0.090 depth/a.png\n0.105 depth/b.png\n0.205 depth/c.png\n"
        )
        assignments = validator.assign_depth_rows(
            rgb, depth, Decimal("0.09"), Decimal("0.20")
        )
        self.assertEqual(assignments, {0: 1, 1: 0})
        self.assertNotIn(2, assignments)

    def test_slerp_shortest_arc_q_negative_q_and_mid_yaw(self) -> None:
        identity = np.asarray((0.0, 0.0, 0.0, 1.0))
        self.assertTrue(np.allclose(
            validator.slerp(identity, -identity, 0.5), identity
        ))
        yaw_90 = np.asarray((0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)))
        midpoint = validator.slerp(identity, yaw_90, 0.5)
        rotated = validator.quaternion_rotation(midpoint) @ np.asarray((1.0, 0.0, 0.0))
        self.assertTrue(np.allclose(rotated[:2], (math.sqrt(0.5), math.sqrt(0.5))))

    def test_pose_bracket_transform_angular_and_translation_direction(self) -> None:
        identity = (0.0, 0.0, 0.0, 1.0)
        rows = [
            pose_row("0", (0.0, 0.0, 0.0), identity, 0),
            pose_row("0.04", (0.04, 0.0, 0.0), identity, 1),
        ]
        middle = validator.interpolate_pose(rows, Decimal("0.02"))
        self.assertTrue(np.allclose(middle[0], (0.02, 0.0, 0.0)))
        geometry = validator.relative_geometry(
            (rows[0].translation, rows[0].quaternion_xyzw),
            (rows[1].translation, rows[1].quaternion_xyzw),
            0.04,
        )
        self.assertTrue(np.allclose(geometry["t_current_from_previous"], (-0.04, 0, 0)))
        self.assertAlmostEqual(geometry["translation_speed"], 1.0)
        self.assertAlmostEqual(geometry["angular_rate_rad_s"], 0.0)

    def test_angular_rate_is_trace_radians_per_second(self) -> None:
        identity = (np.zeros(3), np.asarray((0.0, 0.0, 0.0, 1.0)))
        yaw_90 = (
            np.zeros(3),
            np.asarray((0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))),
        )
        geometry = validator.relative_geometry(identity, yaw_90, 0.5)
        self.assertAlmostEqual(geometry["angular_rate_rad_s"], math.pi)
        self.assertGreater(
            geometry["angular_rate_rad_s"], 5.0 * math.pi / 180.0
        )

    def test_rotation_homography_uses_previous_to_current_direction(self) -> None:
        identity = validator.rotation_homography(np.eye(3))
        self.assertTrue(np.allclose(identity, np.eye(3)))
        yaw = validator.quaternion_rotation(
            np.asarray((0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)))
        )
        expected = validator.K @ yaw @ np.linalg.inv(validator.K)
        self.assertTrue(np.allclose(validator.rotation_homography(yaw), expected))

    def test_depth_decoder_is_native_uint16_no_convert(self) -> None:
        decoded = validator.decode_depth_png(depth_png())
        self.assertEqual(decoded.dtype, np.uint16)
        self.assertEqual(decoded.shape, (480, 640))
        buffer = BytesIO()
        Image.new("RGB", (640, 480)).save(buffer, format="PNG")
        with self.assertRaisesRegex(ValueError, "DEPTH_PNG_ARRAY"):
            validator.decode_depth_png(buffer.getvalue())

    def test_global_zbuffer_uses_min_z_then_raster_rank(self) -> None:
        candidates = [
            {"pixel_x": 10, "pixel_y": 20, "z_predicted": 2.0, "raster_rank": 8},
            {"pixel_x": 10, "pixel_y": 20, "z_predicted": 1.0, "raster_rank": 9},
            {"pixel_x": 11, "pixel_y": 20, "z_predicted": 1.0, "raster_rank": 7},
            {"pixel_x": 11, "pixel_y": 20, "z_predicted": 1.0, "raster_rank": 6},
        ]
        winners = validator.zbuffer_winners(candidates)
        self.assertEqual(winners[(10, 20)]["raster_rank"], 9)
        self.assertEqual(winners[(11, 20)]["raster_rank"], 6)

    def test_identity_depth_truth_counts_and_euclidean_range_rate(self) -> None:
        depth = np.full((480, 640), 5000, dtype=np.uint16)
        grids = validator.evaluate_truth(
            depth, depth, np.eye(3), np.zeros(3), 0.04
        )
        self.assertEqual(len(grids), 9)
        self.assertTrue(all(grid["truth_eligible"] for grid in grids))
        self.assertTrue(all(grid["N_previous"] >= grid["N_projected"] for grid in grids))
        self.assertTrue(all(grid["N_projected"] == grid["N_static"] for grid in grids))
        self.assertTrue(all(abs(grid["c_truth_grid"]) <= 1e-12 for grid in grids))

    def test_roles_follow_priority_and_full_denominator(self) -> None:
        rotation_pairs = [
            {"truth": 0.0, "absolute_truth": 0.01, "angular": 0.10, "translation": 0.01}
        ] * 8
        rotation = validator.classify_window(10, rotation_pairs)
        self.assertEqual(rotation["role"], validator.ROTATION_ROLE)
        approach_pairs = [
            {"truth": 0.06, "absolute_truth": 0.06, "angular": 0.0, "translation": 0.01}
        ] * 8
        approach = validator.classify_window(10, approach_pairs)
        self.assertEqual(approach["role"], validator.APPROACH_ROLE)
        self.assertEqual(
            validator.classify_window(10, approach_pairs[:7])["role"],
            validator.NO_COVERAGE,
        )

    def test_receipt_validates_full_ten_roles_hashes_and_terminal(self) -> None:
        receipt = base_receipt()
        result = validator.validate_receipt_data(
            receipt, claim_raw=claim_bytes(receipt["claim"])
        )
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["terminal_state"], validator.PASS_TERMINAL)

    def test_claim_hash_is_raw_file_hash_and_object_must_match(self) -> None:
        receipt = base_receipt()
        raw = claim_bytes(receipt["claim"])
        self.assertNotEqual(
            receipt["run_claim_sha256"],
            validator.canonical_sha256(receipt["claim"]),
        )
        self.assertEqual(
            validator.validate_receipt_data(receipt, claim_raw=raw)["status"],
            "VALID",
        )
        changed_raw = raw.replace(b'"maximum_claims":1', b'"maximum_claims":2')
        result = validator.validate_receipt_data(
            receipt, claim_raw=changed_raw
        )
        self.assertIn("CLAIM_OBJECT_MISMATCH", result["errors"])
        self.assertIn("CLAIM_FILE_SHA256", result["errors"])

    def test_tamper_firewall_claim_first_and_window_cardinality_fail_closed(self) -> None:
        mutations = []
        tampered = base_receipt()
        tampered["ledger"]["windows"][0]["start"] = "0"
        mutations.append(tampered)
        firewall = base_receipt()
        firewall["read_firewall"]["rgb_member_bytes_read"] = 1
        mutations.append(firewall)
        claim = base_receipt()
        claim["claim"]["application_data_operations_before_claim"] = 1
        claim["run_claim_sha256"] = hashlib.sha256(
            claim_bytes(claim["claim"])
        ).hexdigest()
        mutations.append(claim)
        short = base_receipt()
        short["ledger"]["windows"].pop()
        short["ledger_identity_sha256"] = validator.canonical_sha256(short["ledger"])
        mutations.append(short)
        for receipt in mutations:
            raw = claim_bytes(receipt["claim"])
            with self.subTest(
                errors=validator.validate_receipt_data(
                    receipt, claim_raw=raw
                )["errors"]
            ):
                result = validator.validate_receipt_data(
                    receipt, claim_raw=raw
                )
                self.assertEqual(result["status"], "INVALID")
                self.assertEqual(result["terminal_state"], validator.INVALID_TERMINAL)

    def test_validate_existing_independently_replays_and_is_read_only(self) -> None:
        receipt = base_receipt()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = (
                root
                / "artifacts.local"
                / "evidence"
                / "rcle_phase_b_bonn_b1"
                / "b1a_geometry_admission"
            )
            output.mkdir(parents=True)
            path = output / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            (output / "run_claim.json").write_bytes(
                claim_bytes(receipt["claim"])
            )
            (output / "ledger.json").write_text(
                json.dumps(receipt["ledger"]), encoding="utf-8"
            )
            before = sorted(item.relative_to(root) for item in root.rglob("*"))
            replay_metadata = {
                "archive_sha256_by_sequence": receipt[
                    "archive_sha256_by_sequence"
                ],
                "depth_decode_operations": 0,
                "pose_numeric_rows_parsed": 0,
            }
            with mock.patch.object(
                validator,
                "_canonical_paths",
                return_value={"output": output},
            ), mock.patch.object(
                validator,
                "_recompute_ledger",
                return_value=(receipt["ledger"], replay_metadata),
            ):
                result = validator.validate_existing(root)
            after = sorted(item.relative_to(root) for item in root.rglob("*"))
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(before, after)

    def test_validate_existing_replay_mismatch_is_invalid(self) -> None:
        receipt = base_receipt()
        replayed = json.loads(json.dumps(receipt["ledger"]))
        replayed["windows"][0]["start"] = "0"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = (
                root
                / "artifacts.local"
                / "evidence"
                / "rcle_phase_b_bonn_b1"
                / "b1a_geometry_admission"
            )
            output.mkdir(parents=True)
            (output / "receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            (output / "run_claim.json").write_bytes(
                claim_bytes(receipt["claim"])
            )
            (output / "ledger.json").write_text(
                json.dumps(receipt["ledger"]), encoding="utf-8"
            )
            metadata = {
                "archive_sha256_by_sequence": receipt[
                    "archive_sha256_by_sequence"
                ],
                "depth_decode_operations": 0,
                "pose_numeric_rows_parsed": 0,
            }
            with mock.patch.object(
                validator,
                "_canonical_paths",
                return_value={"output": output},
            ), mock.patch.object(
                validator,
                "_recompute_ledger",
                return_value=(replayed, metadata),
            ):
                result = validator.validate_existing(root)
        self.assertEqual(result["status"], "INVALID")
        self.assertTrue(
            any(error.startswith("REPLAY_MISMATCH:") for error in result["errors"])
        )

    def test_independent_replay_orchestration_uses_fixture_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            preregistration = root / "prereg.md"
            design_lock = root / "design.json"
            implementation_lock = root / "implementation.json"
            official = root / "official.html"
            tum = root / "tum.html"
            fixture_source = root / "fixture_source.py"
            b0_receipt = root / "b0.json"
            for path, content in (
                (preregistration, b"prereg fixture"),
                (design_lock, b"design fixture"),
                (official, b"official fixture"),
                (tum, b"tum fixture"),
                (fixture_source, b"# fixture source\n"),
            ):
                path.write_bytes(content)
            archive_hashes: dict[str, str] = {}
            sequence_results: list[dict[str, str]] = []
            for sequence_id in sorted(
                {window[1] for window in validator.WINDOWS}
            ):
                archive = root / f"{sequence_id}.zip"
                sequence_windows = [
                    window
                    for window in validator.WINDOWS
                    if window[1] == sequence_id
                ]
                rgb_lines: list[str] = []
                depth_lines: list[str] = []
                pose_lines: list[str] = []
                with zipfile.ZipFile(
                    archive, "w", zipfile.ZIP_STORED
                ) as bundle:
                    for rank, window in enumerate(sequence_windows):
                        start = Decimal(window[3])
                        for offset_index, offset in enumerate(
                            (Decimal("0"), Decimal("0.04"))
                        ):
                            timestamp = str(start + offset)
                            name = f"{rank}_{offset_index}.png"
                            rgb_lines.append(f"{timestamp} rgb/{name}\n")
                            depth_lines.append(
                                f"{timestamp} depth/{name}\n"
                            )
                            pose_lines.append(
                                f"{timestamp} 0 0 0 0 0 0 1\n"
                            )
                            bundle.writestr(
                                f"{sequence_id}/rgb/{name}", b"rgb-fixture"
                            )
                            bundle.writestr(
                                f"{sequence_id}/depth/{name}", depth_png()
                            )
                    bundle.writestr(
                        f"{sequence_id}/rgb.txt", "".join(rgb_lines)
                    )
                    bundle.writestr(
                        f"{sequence_id}/depth.txt", "".join(depth_lines)
                    )
                    bundle.writestr(
                        f"{sequence_id}/groundtruth.txt",
                        "".join(pose_lines),
                    )
                archive_sha = validator.sha256_file(archive)
                archive_hashes[sequence_id] = archive_sha
                sequence_results.append(
                    {
                        "sequence_id": sequence_id,
                        "archive_path": str(archive),
                        "archive_sha256": archive_sha,
                    }
                )
            b0_receipt.write_text(
                json.dumps(
                    {
                        "window_denominator_sha256": (
                            validator.WINDOW_DENOMINATOR_SHA256
                        ),
                        "cohort_identity_sha256": (
                            validator.COHORT_IDENTITY_SHA256
                        ),
                        "sequence_results": sequence_results,
                    }
                ),
                encoding="utf-8",
            )
            prereg_sha = validator.sha256_file(preregistration)
            design_sha = validator.sha256_file(design_lock)
            authority = {
                "bonn_official_page": validator.sha256_file(official),
                "tum_file_formats": validator.sha256_file(tum),
            }
            implementation = {
                "design_lock_sha256": design_sha,
                "preregistration_sha256": prereg_sha,
                "canonical_execution_authorized": True,
                "source_files": {
                    "fixture_source.py": validator.sha256_file(
                        fixture_source
                    )
                },
            }
            implementation_lock.write_text(
                json.dumps(implementation), encoding="utf-8"
            )
            implementation_sha = validator.sha256_file(
                implementation_lock
            )
            receipt = {
                "implementation_lock_sha256": implementation_sha,
                "bootstrap_runner_sha256": validator.sha256_file(
                    fixture_source
                ),
                "implementation_lock": implementation,
                "claim": {
                    "canonical_output": str(root / "output"),
                    "canonical_run_claim": str(
                        root / "output" / "run_claim.json"
                    ),
                    "implementation_lock_sha256": implementation_sha,
                    "bootstrap_runner_sha256": validator.sha256_file(
                        fixture_source
                    ),
                    "source_authority_sha256": authority,
                    "archive_sha256_by_sequence": archive_hashes,
                },
            }
            paths = {
                "output": root / "output",
                "preregistration": preregistration,
                "design_lock": design_lock,
                "implementation_lock": implementation_lock,
                "bootstrap_runner": fixture_source,
                "bonn_official_page": official,
                "tum_file_formats": tum,
                "b0_receipt": b0_receipt,
            }
            with mock.patch.multiple(
                validator,
                PREREGISTRATION_SHA256=prereg_sha,
                DESIGN_LOCK_SHA256=design_sha,
                B0_RECEIPT_SHA256=validator.sha256_file(b0_receipt),
                SOURCE_AUTHORITY_SHA256=authority,
            ):
                ledger, metadata = validator._recompute_ledger(
                    root, receipt, paths
                )
        self.assertEqual(len(ledger["windows"]), 10)
        self.assertEqual(len(ledger["window_results"]), 10)
        self.assertEqual(len(ledger["pairs"]), 10)
        self.assertTrue(all(pair["candidate"] for pair in ledger["pairs"]))
        self.assertEqual(metadata["depth_decode_operations"], 20)
        self.assertEqual(metadata["pose_numeric_rows_parsed"], 20)

    def test_schema_fixes_authority_and_terminal_contract(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "rcle_phase_b_bonn_b1a_receipt.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertTrue(
            schema["properties"]["execution_authority_consumed"]["const"]
        )
        self.assertFalse(
            schema["properties"]["b1b_implementation_authorized"]["const"]
        )
        self.assertEqual(
            schema["$defs"]["ledger"]["properties"]["windows"]["minItems"], 10
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), validator.RECEIPT_KEYS)
        self.assertEqual(
            set(schema["properties"]), validator.RECEIPT_KEYS
        )

    def test_validator_does_not_import_producer(self) -> None:
        source = Path(validator.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import protocol", source)
        self.assertNotIn("import producer", source)
        self.assertNotIn("from .protocol", source)


if __name__ == "__main__":
    unittest.main()
