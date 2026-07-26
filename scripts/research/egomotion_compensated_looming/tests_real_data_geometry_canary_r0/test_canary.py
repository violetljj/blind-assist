from __future__ import annotations

import ast
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image

from egomotion_compensated_looming.real_data_geometry_canary_r0.producer import (
    canonical_bytes,
    load_json,
    produce_archive,
)
from egomotion_compensated_looming.real_data_geometry_canary_r0.validator import (
    validate_materialized,
    validate_receipt,
)
from egomotion_compensated_looming.run_real_data_geometry_canary_r0 import (
    _exclusive_write,
    _verify_implementation_lock,
)


MODULE = (
    Path(__file__).resolve().parents[1]
    / "real_data_geometry_canary_r0"
)
RUNNER = (
    Path(__file__).resolve().parents[1]
    / "run_real_data_geometry_canary_r0.py"
)


def _png(value: int) -> bytes:
    array = np.full((480, 640), value, dtype=np.uint16)
    stream = BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def _add(bundle: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    bundle.addfile(info, BytesIO(payload))


def _fixture_archive(
    path: Path,
    *,
    invalid_pose_time: str | None = None,
) -> None:
    rgb_times = ("0.10", "0.14", "0.18", "1.10", "1.14", "1.18")
    depth_times = ("0.10", "0.14", "0.18", "1.10", "1.14")
    rgb_text = "\n".join(
        f"{time} rgb/{index:06d}.png"
        for index, time in enumerate(rgb_times)
    ).encode("utf-8")
    depth_text = "\n".join(
        f"{time} depth/{index:06d}.png"
        for index, time in enumerate(depth_times)
    ).encode("utf-8")
    pose_text = "\n".join(
        (
            f"{time} {float(time) * 0.01:.9f} 0 0 0 0 0 0"
            if time == invalid_pose_time
            else f"{time} {float(time) * 0.01:.9f} 0 0 0 0 0 1"
        )
        for time in rgb_times
    ).encode("utf-8")
    with tarfile.open(path, "w:gz") as bundle:
        root = "rgbd_dataset_freiburg2_rpy"
        _add(bundle, f"{root}/rgb.txt", rgb_text)
        _add(bundle, f"{root}/depth.txt", depth_text)
        _add(bundle, f"{root}/groundtruth.txt", pose_text)
        for index in range(len(depth_times)):
            _add(
                bundle,
                f"{root}/depth/{index:06d}.png",
                _png(25000),
            )


def _fixture_contract() -> dict[str, object]:
    return {
        "protocol_id": "RCLE-PHASE-B-REAL-DATA-GEOMETRY-CANARY-R0",
        "canary_cohort": {
            "window_identity": [
                {
                    "window_index": 0,
                    "start_unix_s": "0",
                    "end_unix_s": "1",
                    "candidate_pair_count": 2,
                    "prior_evaluable_pair_count": 2,
                    "prior_window_disposition": "EVALUABLE",
                },
                {
                    "window_index": 4,
                    "start_unix_s": "1",
                    "end_unix_s": "2",
                    "candidate_pair_count": 2,
                    "prior_evaluable_pair_count": 1,
                    "prior_window_disposition": "PAIR_COVERAGE_LT_0P80",
                },
            ],
            "candidate_pair_denominator": 4,
        },
        "result_model": {
            "pass_terminal": (
                "VALID_IMPLEMENTATION_DEBUGGED_GEOMETRY_INTERFACE_ONLY"
            ),
            "nonpass_terminal": (
                "VALID_NOT_EVALUABLE_GEOMETRY_IMPLEMENTATION_VERSION_CLOSED"
            ),
        },
    }


class CanaryFixtureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.archive = Path(self.temporary.name) / "fixture.tgz"
        _fixture_archive(self.archive)
        self.config = load_json(MODULE / "runtime_config_r0.json")
        self.schema = load_json(MODULE / "output_schema_r0.json")
        self.contract = _fixture_contract()
        self.rows, self.summaries = produce_archive(
            self.archive,
            self.contract,
            self.config,
            self.schema,
        )

    def validate(
        self,
        rows: list[dict[str, object]] | None = None,
        summaries: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return validate_materialized(
            self.archive,
            self.contract,
            self.config,
            self.schema,
            rows if rows is not None else self.rows,
            summaries if summaries is not None else self.summaries,
            enforce_frozen_counts=False,
        )

    def test_success_and_abstention_rows_share_exact_schema(self) -> None:
        self.assertEqual(4, len(self.rows))
        expected = set(self.schema["pair_record_keys"])
        for row in self.rows:
            self.assertEqual(expected, set(row))
        success = self.rows[0]
        self.assertTrue(success["evaluable"])
        self.assertIsNone(success["abstention_reason"])
        for metric in (
            "raw_translation_speed_m_s",
            "median_angular_speed_deg_s",
            "median_signed_radial_expansion_per_s",
            "median_absolute_radial_expansion_per_s",
            "radial_expansion_positive_fraction",
            "q90_time_normalized_parallax_rad_per_s",
            "valid_depth_fraction",
        ):
            self.assertIsInstance(success[metric], float)
        abstention = self.rows[-1]
        self.assertFalse(abstention["evaluable"])
        self.assertEqual(
            "RGB_DEPTH_UNMATCHED_OR_REUSED",
            abstention["abstention_reason"],
        )
        for metric in (
            "raw_translation_speed_m_s",
            "median_angular_speed_deg_s",
            "median_signed_radial_expansion_per_s",
            "median_absolute_radial_expansion_per_s",
            "radial_expansion_positive_fraction",
            "q90_time_normalized_parallax_rad_per_s",
            "valid_depth_fraction",
        ):
            self.assertIsNone(abstention[metric])

    def test_independent_replay_has_identity_schema_and_float64_parity(self) -> None:
        validation = self.validate()
        self.assertTrue(validation["gate_pass"], validation)
        self.assertEqual(0, validation["pair_identity_mismatch_count"])
        self.assertEqual(
            0,
            validation["pair_record_key_set_mismatch_count"],
        )
        self.assertEqual(
            0,
            validation["numeric_metric_parity_violation_count"],
        )

    def test_key_add_drop_and_identity_mutation_fail_closed(self) -> None:
        for mutation in ("add", "drop", "identity"):
            with self.subTest(mutation=mutation):
                rows = deepcopy(self.rows)
                if mutation == "add":
                    rows[0]["unexpected"] = 1
                elif mutation == "drop":
                    del rows[0]["valid_depth_fraction"]
                else:
                    rows[0]["current_rgb_timestamp"] = "999"
                validation = self.validate(rows)
                self.assertFalse(validation["gate_pass"])
                if mutation == "identity":
                    self.assertGreater(
                        validation["pair_identity_mismatch_count"],
                        0,
                    )
                else:
                    self.assertGreater(
                        validation["pair_record_key_set_mismatch_count"],
                        0,
                    )

    def test_relaxed_only_numeric_difference_cannot_pass_r0(self) -> None:
        rows = deepcopy(self.rows)
        rows[0]["raw_translation_speed_m_s"] += 5e-11
        validation = self.validate(rows)
        self.assertFalse(validation["gate_pass"])
        self.assertGreater(
            validation["numeric_metric_parity_violation_count"],
            0,
        )
        self.assertEqual(
            0,
            validation[
                "relaxed_numeric_metric_parity_violation_count"
            ],
        )

    def test_absolute_difference_below_primary_tolerance_passes(self) -> None:
        rows = deepcopy(self.rows)
        rows[0]["raw_translation_speed_m_s"] += 5e-13
        validation = self.validate(rows)
        self.assertTrue(validation["gate_pass"], validation)

    def test_window_disposition_mutation_fails_closed(self) -> None:
        for field, value in (
            ("start_unix_s", "999"),
            ("end_unix_s", "999"),
            ("candidate_pair_count", 999),
            ("evaluable", True),
            ("disposition", "EVALUABLE"),
        ):
            with self.subTest(field=field):
                summaries = deepcopy(self.summaries)
                summaries[1][field] = value
                validation = self.validate(summaries=summaries)
                self.assertFalse(validation["gate_pass"])

    def test_window_distribution_mutation_fails_float_parity(self) -> None:
        summaries = deepcopy(self.summaries)
        summaries[0]["distributions"][
            "raw_translation_speed_m_s"
        ]["median"] += 5e-11
        validation = self.validate(summaries=summaries)
        self.assertFalse(validation["gate_pass"])
        self.assertGreater(
            validation["numeric_metric_parity_violation_count"],
            0,
        )

    def test_nested_window_schema_type_mutations_fail_closed(self) -> None:
        mutations = (
            (
                "quantile_string",
                lambda summaries: summaries[0]["distributions"][
                    "median_angular_speed_deg_s"
                ].__setitem__("median", "0.0"),
            ),
            (
                "quantile_integer",
                lambda summaries: summaries[0]["distributions"][
                    "median_angular_speed_deg_s"
                ].__setitem__("median", 0),
            ),
            (
                "count_float",
                lambda summaries: summaries[0]["distributions"][
                    "median_angular_speed_deg_s"
                ].__setitem__("count", 2.0),
            ),
            (
                "abstention_count_float",
                lambda summaries: summaries[1][
                    "abstention_counts"
                ].__setitem__(
                    "RGB_DEPTH_UNMATCHED_OR_REUSED",
                    1.0,
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                summaries = deepcopy(self.summaries)
                mutate(summaries)
                validation = self.validate(summaries=summaries)
                self.assertFalse(validation["gate_pass"])
                self.assertGreater(
                    validation["pair_record_key_set_mismatch_count"],
                    0,
                )

    def test_boolean_integer_type_confusion_is_schema_mismatch(self) -> None:
        rows = deepcopy(self.rows)
        rows[0]["evaluable"] = 1
        validation = self.validate(rows)
        self.assertFalse(validation["gate_pass"])
        self.assertGreater(
            validation["pair_record_key_set_mismatch_count"],
            0,
        )
        rows = deepcopy(self.rows)
        rows[0]["valid_depth_fraction"] = 1
        validation = self.validate(rows)
        self.assertFalse(validation["gate_pass"])
        self.assertGreater(
            validation["pair_record_key_set_mismatch_count"],
            0,
        )

    def test_abstention_nonnull_metric_and_wrong_reason_fail(self) -> None:
        for field, value in (
            ("raw_translation_speed_m_s", 0.0),
            ("abstention_reason", "UNKNOWN_REASON"),
        ):
            with self.subTest(field=field):
                rows = deepcopy(self.rows)
                rows[-1][field] = value
                validation = self.validate(rows)
                self.assertFalse(validation["gate_pass"])

    def test_validator_import_firewall_uses_ast(self) -> None:
        validator_path = MODULE / "validator.py"
        tree = ast.parse(validator_path.read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        forbidden = (
            "producer",
            "tum_fr2_rpy_geometry_audit",
            "pb_h1_role_proxy",
        )
        self.assertFalse(
            [
                name
                for name in imports
                if any(token in name for token in forbidden)
            ]
        )

    def test_producer_imports_pb_h1_but_not_old_audit(self) -> None:
        producer_source = (MODULE / "producer.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(producer_source)
        modules = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        self.assertTrue(
            any("pb_h1_role_proxy.geometry" in name for name in modules)
        )
        self.assertFalse(
            any("tum_fr2_rpy_geometry_audit" in name for name in modules)
        )
        self.assertNotIn("source_audit_result", producer_source)
        self.assertNotIn("result.json", producer_source)

    def test_receipt_exact_schema_and_hash_bindings(self) -> None:
        pair_bytes = b'{"fixture":true}\n'
        summary_bytes = canonical_bytes(self.summaries)
        bindings = {
            "contract_sha256": "a" * 64,
            "archive_sha256": "b" * 64,
            "source_audit_contract_sha256": "c" * 64,
            "source_audit_result_sha256": "d" * 64,
            "pb_h1_geometry_sha256": "e" * 64,
        }
        receipt = {
            "schema_version": (
                "rcle.real_data_geometry_canary.receipt.v1"
            ),
            "protocol_id": self.contract["protocol_id"],
            **bindings,
            "implementation_lock_sha256": "f" * 64,
            "pair_ledger_sha256": sha256(pair_bytes).hexdigest(),
            "window_summary_sha256": sha256(summary_bytes).hexdigest(),
            "pair_record_count": len(self.rows),
            "window_count": len(self.summaries),
        }
        self.assertEqual(
            [],
            validate_receipt(
                receipt,
                pair_bytes,
                summary_bytes,
                self.schema,
                bindings,
                "f" * 64,
                self.contract["protocol_id"],
                len(self.rows),
                len(self.summaries),
            ),
        )
        receipt["pair_ledger_sha256"] = "0" * 64
        self.assertIn(
            "RECEIPT_BINDING:pair_ledger_sha256",
            validate_receipt(
                receipt,
                pair_bytes,
                summary_bytes,
                self.schema,
                bindings,
                "f" * 64,
                self.contract["protocol_id"],
                len(self.rows),
                len(self.summaries),
            ),
        )
        for field, value in (
            ("schema_version", "WRONG"),
            ("protocol_id", "WRONG"),
            ("pair_record_count", 999),
            ("window_count", 999),
        ):
            with self.subTest(field=field):
                mutated = deepcopy(receipt)
                mutated["pair_ledger_sha256"] = sha256(
                    pair_bytes
                ).hexdigest()
                mutated[field] = value
                self.assertTrue(
                    validate_receipt(
                        mutated,
                        pair_bytes,
                        summary_bytes,
                        self.schema,
                        bindings,
                        "f" * 64,
                        self.contract["protocol_id"],
                        len(self.rows),
                        len(self.summaries),
                    )
                )

    def test_runner_requires_external_activation_lock(self) -> None:
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }
        self.assertIn(
            "FORMAL_EXECUTION_ACTIVATION_LOCK_MISSING",
            strings,
        )
        self.assertIn("CANONICAL_OUTPUT_ALREADY_EXISTS", strings)

    def test_empty_implementation_control_manifest_is_rejected(self) -> None:
        lock = Path(self.temporary.name) / "lock.json"
        lock.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "rcle.real_data_geometry_canary."
                        "implementation_lock.v1"
                    ),
                    "implementation_review_status": "PASS",
                    "formal_execution_authorized": False,
                    "control_files": [],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ValueError,
            "IMPLEMENTATION_CONTROL_MANIFEST_INVALID",
        ):
            _verify_implementation_lock(
                Path(self.temporary.name),
                lock,
            )

    def test_run_claim_is_exclusive_and_not_overwritable(self) -> None:
        claim = Path(self.temporary.name) / "claim.json"
        _exclusive_write(claim, b"first")
        with self.assertRaises(FileExistsError):
            _exclusive_write(claim, b"second")
        self.assertEqual(b"first", claim.read_bytes())

    def test_invalid_quaternion_is_pair_abstention_not_global_crash(
        self,
    ) -> None:
        archive = Path(self.temporary.name) / "invalid-pose.tgz"
        _fixture_archive(archive, invalid_pose_time="0.14")
        rows, summaries = produce_archive(
            archive,
            self.contract,
            self.config,
            self.schema,
        )
        self.assertEqual(
            "POSE_INVALID_QUATERNION",
            rows[0]["abstention_reason"],
        )
        validation = validate_materialized(
            archive,
            self.contract,
            self.config,
            self.schema,
            rows,
            summaries,
            enforce_frozen_counts=False,
        )
        self.assertTrue(validation["gate_pass"], validation)

    def test_unknown_geometry_error_is_not_mapped_to_abstention(
        self,
    ) -> None:
        config = deepcopy(self.config)
        config["intrinsic"] = [[1.0]]
        with self.assertRaisesRegex(ValueError, "PB_H1_MATRIX_SHAPE"):
            produce_archive(
                self.archive,
                self.contract,
                config,
                self.schema,
            )


if __name__ == "__main__":
    unittest.main()
