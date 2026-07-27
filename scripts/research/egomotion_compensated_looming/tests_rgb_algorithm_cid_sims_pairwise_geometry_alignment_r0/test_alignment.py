from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.rgb_algorithm_cid_sims_pairwise_geometry_alignment_r0 import (
    producer,
    run,
    validator,
)


class GeometryBandTests(unittest.TestCase):
    def test_frozen_boundaries(self) -> None:
        self.assertEqual(
            producer.geometry_band(0.009999999),
            "BELOW_TRIGGER_REFERENCE",
        )
        self.assertEqual(producer.geometry_band(0.01), "WEAK_POSITIVE_RADIAL")
        self.assertEqual(
            producer.geometry_band(0.049999999), "WEAK_POSITIVE_RADIAL"
        )
        self.assertEqual(
            producer.geometry_band(0.05), "POSITIVE_APPROACH_GEOMETRY"
        )


class StatisticTests(unittest.TestCase):
    def test_average_ranks_preserve_ties(self) -> None:
        self.assertEqual(
            producer.average_ranks([30.0, 10.0, 20.0, 20.0]).tolist(),
            [4.0, 1.0, 2.5, 2.5],
        )

    def test_run_stats_use_pair_time_span(self) -> None:
        rows = [
            {
                "window_index": 0,
                "pair_index": 0,
                "previous_timestamp_s": 1.0,
                "current_timestamp_s": 1.1,
                "flag": True,
            },
            {
                "window_index": 0,
                "pair_index": 1,
                "previous_timestamp_s": 1.1,
                "current_timestamp_s": 1.3,
                "flag": True,
            },
            {
                "window_index": 0,
                "pair_index": 2,
                "previous_timestamp_s": 1.3,
                "current_timestamp_s": 1.6,
                "flag": False,
            },
            {
                "window_index": 0,
                "pair_index": 3,
                "previous_timestamp_s": 1.6,
                "current_timestamp_s": 2.0,
                "flag": True,
            },
        ]
        summary = producer.run_stats(rows, lambda row: row["flag"])
        self.assertEqual(summary["pair_count"], 2)
        self.assertAlmostEqual(summary["duration_s"], 0.3)

    def test_run_stats_do_not_merge_windows(self) -> None:
        rows = [
            {
                "window_index": 0,
                "pair_index": 298,
                "previous_timestamp_s": 9.9,
                "current_timestamp_s": 10.0,
                "flag": True,
            },
            {
                "window_index": 1,
                "pair_index": 0,
                "previous_timestamp_s": 10.0,
                "current_timestamp_s": 10.1,
                "flag": True,
            },
        ]
        summary = producer.run_stats(rows, lambda row: row["flag"])
        self.assertEqual(summary["pair_count"], 1)

    def test_jaccard_zero_union_is_none(self) -> None:
        rows = [
            {
                "geometry_evaluable": True,
                "geometry_band": "WEAK_POSITIVE_RADIAL",
                "rgb_trigger": False,
            }
        ]
        summary = producer.summarize_window(
            {
                "window_index": 0,
                "role": "TEST",
                "start_timestamp_s": "0",
            },
            [
                {
                    **rows[0],
                    "window_index": 0,
                    "pair_index": 0,
                    "previous_timestamp_s": 0.0,
                    "current_timestamp_s": 0.1,
                    "geometry_signed_radial_expansion_per_s": 0.02,
                    "rgb_compensated_expansion_per_s": 0.0,
                }
            ],
        )
        self.assertIsNone(summary["positive_trigger_jaccard"])

    def test_producer_validator_aggregate_parity(self) -> None:
        contract = {
            "windows": [
                {
                    "window_index": 0,
                    "role": "CONTROL",
                    "start_timestamp_s": "0",
                },
                {
                    "window_index": 1,
                    "role": "POSITIVE",
                    "start_timestamp_s": "1",
                },
            ]
        }
        rows = [
            {
                "window_index": 0,
                "pair_index": 0,
                "previous_timestamp_s": 0.0,
                "current_timestamp_s": 0.1,
                "dt_s": 0.1,
                "rgb_compensated_expansion_per_s": 0.0,
                "rgb_trigger": False,
                "geometry_evaluable": True,
                "geometry_abstention_reason": None,
                "geometry_signed_radial_expansion_per_s": -0.01,
                "geometry_radial_expansion_positive_fraction": 0.4,
                "geometry_q90_time_normalized_parallax_rad_per_s": 0.2,
                "geometry_band": "BELOW_TRIGGER_REFERENCE",
            },
            {
                "window_index": 1,
                "pair_index": 0,
                "previous_timestamp_s": 1.0,
                "current_timestamp_s": 1.1,
                "dt_s": 0.1,
                "rgb_compensated_expansion_per_s": 0.02,
                "rgb_trigger": True,
                "geometry_evaluable": True,
                "geometry_abstention_reason": None,
                "geometry_signed_radial_expansion_per_s": 0.08,
                "geometry_radial_expansion_positive_fraction": 0.8,
                "geometry_q90_time_normalized_parallax_rad_per_s": 0.3,
                "geometry_band": "POSITIVE_APPROACH_GEOMETRY",
            },
        ]
        self.assertEqual(
            producer.summarize_all(contract, rows),
            validator._summarize_all(contract, rows),
        )


class MutationTests(unittest.TestCase):
    def test_exact_compare_rejects_numeric_and_order_mutations(self) -> None:
        errors: list[str] = []
        validator._compare_exact(
            [{"value": 1.0000000000000002}, {"value": 2.0}],
            [{"value": 1.0}, {"value": 2.0}],
            "ROW",
            errors,
        )
        self.assertIn("ROW[0].value:FLOAT_HEX", errors)
        errors = []
        validator._compare_exact([2, 1], [1, 2], "ORDER", errors)
        self.assertEqual(errors, ["ORDER[0]:INTEGER", "ORDER[1]:INTEGER"])

    def test_runner_lock_and_activation_mutations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo_root = Path(raw)
            contract_relative = next(
                path for path in run.EXPECTED_LOCK_PATHS if "CONTRACT" in path
            )
            for relative in run.EXPECTED_LOCK_PATHS:
                path = repo_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            contract_path = repo_root / contract_relative
            contract_path.write_text(
                json.dumps({"source": {"archive_sha256": "archive"}}),
                encoding="utf-8",
            )
            files = [
                {
                    "path": relative,
                    "sha256": hashlib.sha256(
                        (repo_root / relative).read_bytes()
                    ).hexdigest(),
                }
                for relative in sorted(run.EXPECTED_LOCK_PATHS)
            ]
            lock_path = repo_root / "implementation_lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "schema_version": "rcle.rgb_algorithm.pairwise_geometry_alignment.implementation_lock.v1",
                        "protocol_id": run.PROTOCOL_ID,
                        "status": "LOCKED_BEFORE_FULL_PAIR_GEOMETRY_ACCESS",
                        "files": files,
                    }
                ),
                encoding="utf-8",
            )
            output_dir = repo_root / "evidence" / "run_r0"
            activation = {
                "schema_version": "rcle.rgb_algorithm.pairwise_geometry_alignment.activation.v1",
                "protocol_id": run.PROTOCOL_ID,
                "status": "AUTHORIZED_FOR_ONE_PAIRWISE_ALIGNMENT_RUN",
                "implementation_lock_sha256": run.digest_file(lock_path),
                "contract_sha256": run.digest_file(contract_path),
                "archive_sha256": "archive",
                "output_dir": "evidence/run_r0",
                "maximum_authority": "POSTHOC_REAL_DATA_MECHANISM_ALIGNMENT_ONLY",
                **{
                    field: False
                    for field in (
                        "algorithm_reexecution_authorized",
                        "threshold_tuning_authorized",
                        "outcome_blind_claim_authorized",
                        "independent_confirmation_authorized",
                        "performance_qualification_authorized",
                        "product_or_safety_claim_authorized",
                        "network_access_authorized",
                        "download_authorized",
                    )
                },
            }
            activation_path = repo_root / "activation.json"
            activation_path.write_text(json.dumps(activation), encoding="utf-8")
            self.assertEqual(
                run.verify_locks(
                    repo_root,
                    contract_path,
                    lock_path,
                    activation_path,
                    output_dir,
                ),
                [],
            )
            activation["performance_qualification_authorized"] = True
            activation_path.write_text(json.dumps(activation), encoding="utf-8")
            errors = run.verify_locks(
                repo_root,
                contract_path,
                lock_path,
                activation_path,
                output_dir,
            )
            self.assertIn(
                "ACTIVATION_FORBIDDEN:performance_qualification_authorized",
                errors,
            )


if __name__ == "__main__":
    unittest.main()
