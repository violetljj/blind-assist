from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_d2_official_train_metadata import MetadataApi
from plan_stage_c_d4_m0_metadata_census import (
    CANONICAL_ROOT,
    CONTRACT_RELATIVE_PATH,
    M0_INVALID,
    allocation_parameters,
    audit_post_q0_local_evidence,
    derive_exclusions,
    execute_with_failure_closure,
    freeze_existing_partial,
    parse_split_ids,
    qualify_candidate,
    rank_pool,
    repo_root,
    require_canonical_root,
    scan_all,
    validate_existing_terminal,
    validate_contract,
    write_json_exclusive_fsync,
)


def sid(index: int) -> str:
    return f"{index:064x}"


def receipt(name: str) -> dict[str, str]:
    return {
        "name": name,
        "generation": "11",
        "metageneration": "1",
        "size": "7",
        "md5Hash": "YWJj",
        "crc32c": "ZGVm",
    }


class StrictMetadataApi:
    def __init__(self, fps: float = 5.0) -> None:
        self.fps = fps
        self.listed: list[str] = []
        self.fetched: list[str] = []

    def get_object(self, name: str, retries: int) -> dict[str, str]:
        return receipt(name)

    def fetch_text(self, url: str, retries: int) -> str:
        raise AssertionError("Unexpected text read")

    def fetch_json(self, url: str, retries: int) -> dict[str, object]:
        self.fetched.append(url)
        if "description.json" not in url:
            raise AssertionError("Only description bytes are allowed")
        return {
            "session_type": "synthetic",
            "session_camera_location": ["camera_chest"],
            "session_camera_details": [
                {
                    "fps": self.fps,
                    "left_camera_params": {
                        "fx": 10.0,
                        "fy": 11.0,
                        "cx": 5.0,
                        "cy": 6.0,
                        "image_width": 20,
                        "image_height": 12,
                    },
                }
            ],
        }

    def list_objects(
        self, prefix: str, retries: int
    ) -> list[dict[str, str]]:
        self.listed.append(prefix)
        if "/video_frames/" in prefix:
            raise AssertionError("RGB listing is forbidden")
        suffix = ".png" if "/segmentation_masks/" in prefix else ".float16.gz"
        step = 1 if self.fps == 5.0 else 4
        return [
            receipt(f"{prefix}{index:06d}{suffix}")
            for index in range(0, 13 * step, step)
        ]

    def value(self) -> MetadataApi:
        return MetadataApi(
            self.get_object,
            self.fetch_text,
            self.fetch_json,
            self.list_objects,
        )


class D4M0MetadataCensusTest(unittest.TestCase):
    def test_real_exclusion_union_is_exact_124(self) -> None:
        d2 = json.loads(
            (
                repo_root()
                / "artifacts.local/evidence/hftf/"
                "stage-c-d2-official-train-metadata-qualification-20260802/"
                "qualification.json"
            ).read_text()
        )
        q0 = json.loads(
            (
                repo_root()
                / "artifacts.local/evidence/hftf/"
                "stage-c-d3-q0-metadata-roster-20260802/roster.json"
            ).read_text()
        )
        design = json.loads(
            (
                repo_root()
                / "docs/research/hftf/"
                "HFTF_STAGE_C_D4_OPPORTUNITY_ECOLOGY_AND_"
                "RECRUITABILITY_R0_2026-08-02.json"
            ).read_text()
        )
        prior, q0_ids, _ = derive_exclusions(d2, q0, design)
        self.assertEqual((84, 40, 124), (len(prior), len(q0_ids), len(prior | q0_ids)))

    def test_scan_covers_1560_and_calls_exact_1442(self) -> None:
        ids = [sid(index) for index in range(1560)]
        excluded = set(ids[:118])
        calls: list[str] = []

        def qualify(value: str) -> dict[str, object]:
            calls.append(value)
            return {"session_id": value, "classification": "metadata_eligible_20hz"}

        ledger, five, twenty = scan_all(ids, excluded, qualify)
        self.assertEqual((1560, 1442, 0, 1442), (len(ledger), len(calls), len(five), len(twenty)))
        self.assertFalse(any(row["session_id"] in calls for row in ledger[:118]))

    def test_scan_closes_ineligible_without_replacement(self) -> None:
        ids = [sid(index) for index in range(1560)]
        calls = 0

        def qualify(value: str) -> dict[str, object]:
            nonlocal calls
            calls += 1
            from plan_stage_c_d2_official_train_metadata import CandidateIneligible
            raise CandidateIneligible("closed")

        ledger, five, twenty = scan_all(ids, set(ids[:118]), qualify)
        self.assertEqual(1442, calls)
        self.assertEqual(1442, sum(row["classification"] == "metadata_ineligible_closed" for row in ledger))
        self.assertEqual(([], []), (five, twenty))

    def test_parse_split_preserves_source_order(self) -> None:
        ids = [sid(index) for index in reversed(range(1560))]
        self.assertEqual(ids, parse_split_ids("\n".join(ids) + "\n"))

    def test_parse_split_rejects_duplicate(self) -> None:
        ids = [sid(index) for index in range(1559)] + [sid(0)]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_split_ids("\n".join(ids))

    def test_qualifier_reads_description_and_mask_depth_only(self) -> None:
        api = StrictMetadataApi(5.0)
        row = qualify_candidate(sid(1), 3, api.value())
        self.assertEqual("metadata_eligible_5hz", row["classification"])
        self.assertEqual(2, len(api.listed))
        self.assertTrue(all("video_frames" not in item for item in api.listed))
        self.assertFalse(row["pose_content_read"])

    def test_qualifier_accepts_exact_20hz_timeline(self) -> None:
        api = StrictMetadataApi(20.0)
        row = qualify_candidate(sid(2), 3, api.value())
        self.assertEqual(list(range(0, 49, 4)), row["selected_source_frames"])
        self.assertEqual("metadata_eligible_20hz", row["classification"])

    def test_qualifier_rejects_other_fps(self) -> None:
        api = StrictMetadataApi(10.0)
        with self.assertRaisesRegex(Exception, "fps"):
            qualify_candidate(sid(3), 3, api.value())

    def test_transport_exhaustion_propagates_and_invalidates_census(self) -> None:
        api = StrictMetadataApi(5.0)
        api.get_object = mock.Mock(side_effect=OSError("timeout"))
        with self.assertRaisesRegex(OSError, "timeout"):
            qualify_candidate(sid(4), 3, api.value())

    def test_rank_known_answer_and_reproducible(self) -> None:
        seed = bytes(range(32))
        value = sid(7)
        expected = hashlib.sha256(
            b"HFTF_D4_R0_ALLOC|" + seed + b"|" + value.encode("ascii")
        ).hexdigest()
        first = rank_pool([value, sid(8)], seed)
        second = rank_pool([value, sid(8)], seed)
        self.assertEqual(first, second)
        self.assertEqual(expected, next(row["rank_digest"] for row in first if row["session_id"] == value))

    def test_rank_collision_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "collision"):
            rank_pool([sid(1), sid(2)], bytes(32), digest=lambda _: "0" * 64)

    def test_allocation_boundary_values(self) -> None:
        self.assertEqual({"N_five_hz": 63, "C": 63, "n_ecology": 23, "B_effect_reserve": 40}, allocation_parameters(63))
        self.assertEqual((64, 24, 40), tuple(allocation_parameters(64)[key] for key in ("C", "n_ecology", "B_effect_reserve")))
        self.assertEqual((127, 47, 80), tuple(allocation_parameters(127)[key] for key in ("C", "n_ecology", "B_effect_reserve")))
        self.assertEqual((128, 48, 80), tuple(allocation_parameters(200)[key] for key in ("C", "n_ecology", "B_effect_reserve")))

    def test_exclusive_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "value.json"
            with mock.patch("plan_stage_c_d4_m0_metadata_census.fsync_directory"):
                write_json_exclusive_fsync(path, {"value": 1})
                with self.assertRaises(FileExistsError):
                    write_json_exclusive_fsync(path, {"value": 2})

    def test_canonical_root_rejects_alternate(self) -> None:
        self.assertEqual((repo_root() / CANONICAL_ROOT).resolve(), require_canonical_root(repo_root() / CANONICAL_ROOT))
        with self.assertRaisesRegex(ValueError, "Noncanonical"):
            require_canonical_root(repo_root() / "wrong")

    def test_partial_root_freezes_invalid_without_seed_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "attempt.json").write_text("{}")
            with mock.patch("plan_stage_c_d4_m0_metadata_census.fsync_directory"):
                code = freeze_existing_partial(root, {"attempt.json"})
            failure = json.loads((root / "failure.json").read_text())
        self.assertEqual(2, code)
        self.assertEqual(M0_INVALID, failure["terminal"])
        self.assertFalse(failure["seed_generated_by_this_freeze"])

    def test_corrupt_result_is_not_accepted_and_freezes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "result.json").write_text("{}")
            names = {"result.json"}
            self.assertFalse(validate_existing_terminal(root, names))
            with mock.patch(
                "plan_stage_c_d4_m0_metadata_census.fsync_directory"
            ):
                code = freeze_existing_partial(root, names)
            failure = json.loads((root / "failure.json").read_text())
        self.assertEqual(2, code)
        self.assertEqual(M0_INVALID, failure["terminal"])

    def test_valid_locked_terminal_hash_chain_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = {
                "attempt.json", "preflight.json", "census.json", "pool.json",
                "alloc-attempt.json", "seed.json", "result.json",
            }
            for name in names - {"result.json"}:
                (root / name).write_text(name)
            bindings = {
                "attempt_sha256": hashlib.sha256((root / "attempt.json").read_bytes()).hexdigest(),
                "preflight_sha256": hashlib.sha256((root / "preflight.json").read_bytes()).hexdigest(),
                "census_sha256": hashlib.sha256((root / "census.json").read_bytes()).hexdigest(),
                "pool_sha256": hashlib.sha256((root / "pool.json").read_bytes()).hexdigest(),
                "allocation_attempt_sha256": hashlib.sha256((root / "alloc-attempt.json").read_bytes()).hexdigest(),
                "seed_receipt_sha256": hashlib.sha256((root / "seed.json").read_bytes()).hexdigest(),
            }
            (root / "result.json").write_text(json.dumps({
                "schema": "blindassist_hftf_stage_c_d4_m0_recruitability_pool_result",
                "terminal": "D4_M0_FRESH_METADATA_RECRUITABILITY_POOL_CENSUS_LOCKED",
                "bindings": bindings,
            }))
            self.assertTrue(validate_existing_terminal(root, names))
            (root / "preflight.json").write_text("tampered")
            self.assertFalse(validate_existing_terminal(root, names))

    def test_valid_insufficient_terminal_hash_chain_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = {"attempt.json", "preflight.json", "census.json", "pool.json", "result.json"}
            for name in names - {"result.json"}:
                (root / name).write_text(name)
            (root / "result.json").write_text(json.dumps({
                "schema": "blindassist_hftf_stage_c_d4_m0_recruitability_pool_result",
                "terminal": "D4_M0_FRESH_5HZ_METADATA_RECRUITABILITY_POOL_INSUFFICIENT_STOP",
                "bindings": {
                    "attempt_sha256": hashlib.sha256((root / "attempt.json").read_bytes()).hexdigest(),
                    "preflight_sha256": hashlib.sha256((root / "preflight.json").read_bytes()).hexdigest(),
                    "census_sha256": hashlib.sha256((root / "census.json").read_bytes()).hexdigest(),
                    "pool_sha256": hashlib.sha256((root / "pool.json").read_bytes()).hexdigest(),
                },
            }))
            self.assertTrue(validate_existing_terminal(root, names))
            (root / "attempt.json").write_text("tampered")
            self.assertFalse(validate_existing_terminal(root, names))

    def test_local_audit_reads_only_allowlisted_attempt_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            roots = []
            union: set[str] = set()
            for index, schema in enumerate(
                (
                    "blindassist_hftf_stage_c_d3_q0_slot_attempt",
                    "blindassist_hftf_stage_c_d3_q0_1_slot_attempt",
                )
            ):
                evidence = root / f"evidence-{index}"
                slot = evidence / f"slot-0{index + 1}-token"
                slot.mkdir(parents=True)
                value = sid(index + 1)
                union.add(value)
                (slot / "attempt.json").write_text(
                    json.dumps(
                        {
                            "schema": schema,
                            "status": (
                                "D3_Q0_SLOT_ATTEMPT_FSYNCED_BEFORE_"
                                "FIRST_POSE_OR_MEDIA_REQUEST"
                            ),
                            "session_id": value,
                        }
                    )
                )
                (slot / "sealed_payload.json").write_bytes(
                    b"must not be opened: not JSON"
                )
                roots.append(
                    {
                        "path": str(evidence),
                        "expected_attempt_count": 1,
                        "allowed_attempt_schemas": [schema],
                    }
                )
            result = audit_post_q0_local_evidence(union, roots)
        self.assertEqual(2, result["json_file_count"])
        self.assertFalse(
            result["sealed_payload_selector_truth_or_frame_json_read"]
        )

    def test_pre_network_drift_writes_durable_invalid_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract = base / "contract.json"
            design = base / "design.json"
            contract.write_text("{}")
            design.write_text("{}")
            root = base / "run"
            api = mock.Mock()
            seed = mock.Mock()
            context = {
                "retries": 3,
                "design_path": design,
                "excluded": set(),
                "local_evidence_audit_roots": [],
            }
            with mock.patch(
                "plan_stage_c_d4_m0_metadata_census.validate_contract",
                return_value=context,
            ), mock.patch(
                "plan_stage_c_d4_m0_metadata_census."
                "audit_post_q0_local_evidence",
                side_effect=ValueError("outside parent"),
            ):
                with self.assertRaisesRegex(ValueError, "outside parent"):
                    execute_with_failure_closure(
                        contract,
                        root,
                        3,
                        api=api,
                        token_bytes=seed,
                        verify_git=False,
                    )
            failure = json.loads((root / "failure.json").read_text())
        self.assertEqual(M0_INVALID, failure["terminal"])
        self.assertIn("outside parent", failure["reason"])
        api.get_object.assert_not_called()
        seed.assert_not_called()

    def test_real_contract_validates_without_git_gate(self) -> None:
        context = validate_contract(
            repo_root() / CONTRACT_RELATIVE_PATH, verify_git=False
        )
        self.assertEqual((84, 40, 124), (len(context["prior"]), len(context["q0_ids"]), len(context["excluded"])))
        self.assertEqual(3, context["retries"])


if __name__ == "__main__":
    unittest.main()
