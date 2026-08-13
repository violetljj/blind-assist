from __future__ import annotations

import ast
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.research.taro_o0r_factor_headroom_runtime.evidence import (
    FactorEvidenceWriter,
)
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r11_abstention_runtime import (
    run_top24_selection as runner,
)


def feature(eligible: bool) -> dict:
    return {
        "query_receipt": {} if eligible else None,
        "r6_state": "UNKNOWN",
        "positive_obstacle_veto": False if eligible else None,
        "occupied_hits": [[[False]]] if eligible else None,
        "far_valid_anchor_count": 6 if eligible else 0,
        "far_fractions": [0.0, 0.0, 0.0] if eligible else None,
        "observed_support_points": 0,
    }


def synthetic_sources() -> list[dict]:
    rows: list[dict] = []
    for parent_index, ((parent_id, video_id), frame_count) in enumerate(
        zip(runner.EXPECTED_PARENT_IDENTITIES, runner.FROZEN_FRAME_COUNTS, strict=True)
    ):
        eligible_count = parent_index % 31
        emitted = 0
        for frame_index in range(frame_count):
            queries = []
            for _query_index in range(9):
                allowed = emitted < eligible_count
                queries.append(feature(allowed))
                emitted += int(allowed)
            token = f"{frame_index:06d}"
            rows.append(
                {
                    "parent_id": parent_id,
                    "video_id": video_id,
                    "timestamp_token": token,
                    "physical_frame_id": f"{video_id}:{token}",
                    "query_features": queries,
                    "source_phase_has_label_input": False,
                    "training_steps": 0,
                    "network_requests": 0,
                    "content_sha256": adapter.canonical_sha256([parent_id, video_id, token]),
                }
            )
    return rows


def completion_for(sources: list[dict]) -> dict:
    return {
        "content_sha256": adapter.canonical_sha256("synthetic-completion"),
        "source_frame_hash_sequence_sha256": adapter.canonical_sha256(
            [row["content_sha256"] for row in sources]
        ),
    }


class R11Top24SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = synthetic_sources()
        cls.completion = completion_for(cls.sources)

    def test_exact_contract_and_repaired_phase_a_predecessor(self) -> None:
        self.assertEqual(runner.PARENT_COUNT, 48)
        self.assertEqual(runner.SELECTED_PARENT_COUNT, 24)
        self.assertEqual(runner.FRAME_COUNT, 1043)
        self.assertEqual(runner.QUERY_COUNT, 9387)
        self.assertEqual(sum(runner.FROZEN_FRAME_COUNTS), 1043)
        self.assertEqual(runner.SUCCESS_PRE_TERMINAL_FILE_COUNT, 3)
        self.assertEqual(runner.SUCCESS_FINAL_FILE_COUNT, 4)
        audit = runner._validate_phase_a_audit()
        terminal, completion = runner._validate_phase_a_terminal_light()
        self.assertEqual(audit["status"], runner.PHASE_A_AUDIT_PASS)
        self.assertEqual(terminal["terminal"], runner.PHASE_A_PASS_TERMINAL)
        self.assertEqual(completion["content_sha256"], runner.PHASE_A_COMPLETION_CONTENT_SHA256)

    def test_frozen_selector_rule_and_protocol_reject_mutation(self) -> None:
        runner._validate_protocol_and_r9()
        selector = runner._load_json(runner._repo_path(runner.EXPECTED_BINDINGS["R9_SELECTOR_ARTIFACT"]))
        self.assertEqual(
            runner.validate_frozen_selector(selector)["content_sha256"],
            runner.FROZEN_SELECTOR_CONTENT_SHA256,
        )
        mutated = copy.deepcopy(selector)
        mutated["chosen_rule"]["maximum_far_fraction"] = 0.05
        with self.assertRaises(runner.FreshTop24SelectionError):
            runner.validate_frozen_selector(mutated)
        rule = copy.deepcopy(runner.FROZEN_RULE)
        rule["rule_id"] = "0" * 16
        with self.assertRaises(runner.FreshTop24SelectionError):
            runner.validate_frozen_rule(rule)

    def test_deterministic_48_to_24_and_exact_zero_read_ledger(self) -> None:
        scores, selection = runner.build_selection(self.completion, self.sources)
        forward = runner.rank_parent_scores(scores["parent_scores"])
        reverse = runner.rank_parent_scores(list(reversed(scores["parent_scores"])))
        expected = [(row["parent_id"], row["video_id"]) for row in forward]
        self.assertEqual(expected, [(row["parent_id"], row["video_id"]) for row in reverse])
        self.assertEqual(selection["selected_parent_count"], 24)
        self.assertEqual(
            selection["selected_parent_identities"],
            [[row["parent_id"], row["video_id"]] for row in forward[:24]],
        )
        for record in (scores, selection, *scores["parent_scores"]):
            self.assertEqual(record["source_zip_member_payload_reads"], 0)
            self.assertEqual(record["highres_depth_member_payload_reads"], 0)
            self.assertEqual(record["faro_reads"], 0)
            self.assertEqual(record["truth_reads"], 0)
            self.assertEqual(record["label_reads"], 0)
            self.assertEqual(record["outcome_reads"], 0)
            self.assertEqual(record["model_executions"], 0)
            self.assertEqual(record["training_steps"], 0)
            self.assertEqual(record["network_requests"], 0)
        self.assertTrue(scores["all_48_source_records_sealed_before_scoring"])
        self.assertTrue(scores["all_48_parent_scores_sealed_before_faro"])
        self.assertTrue(selection["selection_sealed_before_faro"])
        self.assertFalse(selection["read_unselected_faro"])
        self.assertFalse(selection["unknown_is_negative"])

    def test_score_and_selection_mutations_fail_even_when_resealed(self) -> None:
        scores, selection = runner.build_selection(self.completion, self.sources)
        mutated_scores = copy.deepcopy(scores)
        mutated_scores["parent_scores"][0]["tie_break_sha256"] = "0" * 64
        mutated_scores.pop("content_sha256")
        mutated_scores["content_sha256"] = adapter.canonical_sha256(mutated_scores)
        with self.assertRaises(runner.FreshTop24SelectionError):
            runner.validate_parent_scores(mutated_scores)

        mutated_selection = copy.deepcopy(selection)
        mutated_selection["selected_parent_identities"] = list(
            reversed(mutated_selection["selected_parent_identities"])
        )
        mutated_selection.pop("content_sha256")
        mutated_selection["content_sha256"] = adapter.canonical_sha256(mutated_selection)
        with self.assertRaises(runner.FreshTop24SelectionError):
            runner.validate_selection(mutated_selection, scores)

    def test_ties_use_only_canonical_parent_video_sha(self) -> None:
        scores, _selection = runner.build_selection(self.completion, self.sources)
        equal = copy.deepcopy(scores["parent_scores"])
        for row in equal:
            row["eligible_query_count"] = 0
            row["eligible_fraction_of_available"] = 0.0
        ranked = runner.rank_parent_scores(equal)
        self.assertEqual(
            [(row["parent_id"], row["video_id"]) for row in ranked],
            sorted(runner.EXPECTED_PARENT_IDENTITIES, key=lambda identity: adapter.canonical_sha256(list(identity))),
        )

    def test_source_firewall_and_public_api_fail_closed(self) -> None:
        runner.assert_public_api_source_only()
        source = copy.deepcopy(self.sources[0])
        source["source_phase_has_label_input"] = True
        with self.assertRaises(runner.FreshTop24SelectionError):
            runner.score_parent([source], runner.FROZEN_RULE)
        self.assertFalse(runner.EXPECTED_AUTHORITY["faro_read"])
        self.assertFalse(runner.EXPECTED_AUTHORITY["highres_depth_member_payload_read"])
        self.assertFalse(runner.EXPECTED_AUTHORITY["model_execution"])

    def test_phase_a_reload_calls_r7_and_r11_validators_and_checks_subset(self) -> None:
        parent_id, video_id, token = "visit", "video", "000001"
        physical_frame_id = f"{video_id}:{token}"
        source = {
            "parent_id": parent_id,
            "video_id": video_id,
            "timestamp_token": token,
            "physical_frame_id": physical_frame_id,
            "source_frame_receipt_sha256": "RECEIPT",
            "content_sha256": "SOURCE",
        }
        base_rows = [
            {"query_id": f"q{index}", "grid_index": index, "state": "UNKNOWN"}
            for index in range(9)
        ]
        candidate_rows = copy.deepcopy(base_rows)
        base = {
            "parent_id": parent_id,
            "video_id": video_id,
            "timestamp_token": token,
            "physical_frame_id": physical_frame_id,
            "source_frame_record_sha256": "SOURCE",
            "query_results": base_rows,
        }
        candidate = {
            **{key: base[key] for key in ("parent_id", "video_id", "timestamp_token", "physical_frame_id")},
            "source_frame_record_sha256": "SOURCE",
            "query_results": candidate_rows,
        }
        receipt = runner._seal(
            {
                "schema": "blindassist.taro.o1r.r11_fresh_pool_source_frame_receipt.v1",
                "parent_id": parent_id,
                "video_id": video_id,
                "timestamp_token": token,
                "physical_frame_id": physical_frame_id,
                "highres_depth_member_payload_read": False,
                "faro_payload_read": False,
                "truth_payload_read": False,
            }
        )
        source["source_frame_receipt_sha256"] = receipt["content_sha256"]
        lineage = runner._seal(
            {
                "schema": "blindassist.taro.o1r.r11_fresh_pool_phase_a_lineage.v1",
                "physical_frame_id": physical_frame_id,
                "source_frame_receipt_sha256": receipt["content_sha256"],
                "r7_source_frame_record": {"fixture": "source"},
                "r7_positive_factor_bundle": {"fixture": "base"},
                "r11_abstention_bundle": {"fixture": "candidate"},
                "highres_depth_member_payload_read": False,
                "faro_payload_read": False,
                "truth_inputs": 0,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / runner._source_relative(parent_id, video_id, token)
            lineage_path = root / runner._lineage_relative(parent_id, video_id, token)
            source_path.parent.mkdir(parents=True)
            lineage_path.parent.mkdir(parents=True)
            source_path.write_bytes(adapter.canonical_json_bytes(receipt) + b"\n")
            import gzip

            with gzip.open(lineage_path, "wt", encoding="utf-8") as stream:
                stream.write(adapter.canonical_json_bytes(lineage).decode("utf-8"))
            completion = {
                "source_frame_hash_sequence_sha256": adapter.canonical_sha256([source["content_sha256"]])
            }
            with (
                mock.patch.object(runner, "FRAME_COUNT", 1),
                mock.patch.object(runner.r7_canary, "validate_source_frame_record", return_value=source) as source_validator,
                mock.patch.object(runner.r7_positive, "validate_positive_occupancy_factor", return_value=base) as base_validator,
                mock.patch.object(runner.abstention_candidate, "validate_abstention_bundle", return_value=candidate) as candidate_validator,
            ):
                records, ledger = runner._load_phase_a_sources(
                    root, [(parent_id, video_id, token)], completion
                )
                self.assertEqual(records, [source])
                self.assertEqual(ledger["query_features_scored"], 9)
                source_validator.assert_called_once()
                base_validator.assert_called_once()
                candidate_validator.assert_called_once()

                candidate_rows[0]["state"] = "OCCUPIED_OBSERVED"
                with self.assertRaises(runner.FreshTop24SelectionError):
                    runner._load_phase_a_sources(root, [(parent_id, video_id, token)], completion)

    def test_missing_execution_lock_fails_before_formal_root_creation(self) -> None:
        formal_root = runner._repo_path(runner.OUTPUT_ROOT)
        self.assertFalse(formal_root.exists())
        with (
            mock.patch.object(
                runner,
                "_load_json",
                side_effect=FileNotFoundError("fixture missing lock"),
            ),
            self.assertRaises(FileNotFoundError),
        ):
            runner.validate_execution_lock(runner._repo_path(runner.LOCK_RELATIVE))
        self.assertFalse(formal_root.exists())

    def test_actual_argv_and_implementation_commit_binding_fail_closed(self) -> None:
        with mock.patch.object(runner.sys, "orig_argv", ["python.exe", *runner.EXPECTED_ARGV]):
            runner._validate_actual_argv()
        with (
            mock.patch.object(runner.sys, "orig_argv", ["python.exe", *runner.EXPECTED_ARGV, "--extra"]),
            self.assertRaises(runner.FreshTop24SelectionError),
        ):
            runner._validate_actual_argv()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"CODE": root / "code.py", "ARTIFACT": root / "artifact.json"}
            payloads = {"CODE": b"print('locked')\n", "ARTIFACT": b"{}\n"}
            for role, path in paths.items():
                path.write_bytes(payloads[role])
            bindings = [
                {
                    "role": role,
                    "path": f"fixture/{role.lower()}",
                    "bytes": len(payloads[role]),
                    "sha256": runner.materializer.sha256_bytes(payloads[role]),
                }
                for role in ("CODE", "ARTIFACT")
            ]
            path_by_relative = {row["path"]: paths[row["role"]] for row in bindings}
            with (
                mock.patch.object(runner, "EXPECTED_BINDINGS", {row["role"]: row["path"] for row in bindings}),
                mock.patch.object(runner, "ARTIFACT_BINDING_ROLES", {"ARTIFACT"}),
                mock.patch.object(runner, "_repo_path", side_effect=lambda relative: path_by_relative[relative]),
                mock.patch.object(runner, "_git_bytes", return_value=payloads["CODE"]),
            ):
                runner._verify_binding_rows({"bindings": bindings}, "a" * 40)
            with (
                mock.patch.object(runner, "EXPECTED_BINDINGS", {row["role"]: row["path"] for row in bindings}),
                mock.patch.object(runner, "ARTIFACT_BINDING_ROLES", {"ARTIFACT"}),
                mock.patch.object(runner, "_repo_path", side_effect=lambda relative: path_by_relative[relative]),
                mock.patch.object(runner, "_git_bytes", return_value=b"drift"),
                self.assertRaises(runner.FreshTop24SelectionError),
            ):
                runner._verify_binding_rows({"bindings": bindings}, "a" * 40)

    def test_failure_terminal_is_single_atomic_terminal_and_releases_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "selection"
            writer = FactorEvidenceWriter(root, runner.EXPECTED_RESOURCE_BUDGET["maximum_evidence_bytes"])
            writer.activate({"schema": "fixture.execution.v1"})
            runner._allocate_terminal_reserve(writer)
            (root / "selection.json.partial").write_bytes(b"partial")
            runner._write_failure(writer, runner.FreshTop24SelectionError("FIXTURE", "failure"))
            self.assertFalse((root / runner.TERMINAL_RESERVE_NAME).exists())
            self.assertFalse((root / "result.json").exists())
            self.assertFalse((root / "manifest.json").exists())
            terminal = runner._validate_seal(
                runner._load_json(root / "terminal.json"),
                runner.TERMINAL_SCHEMA,
            )
            self.assertEqual(terminal["terminal"], runner.FAIL_TERMINAL)
            self.assertFalse(terminal["passed"])
            self.assertIn("selection.json.partial", terminal["files"])

    def test_resource_guard_uses_os_peak_wset_and_final_wall_reserve(self) -> None:
        process = mock.Mock()
        process.memory_info.return_value = SimpleNamespace(peak_wset=1234)
        budget = {
            "maximum_wall_seconds": 100,
            "maximum_peak_rss_bytes": 2000,
            "maximum_evidence_bytes": 4096,
        }
        with mock.patch.object(runner.time, "monotonic", return_value=50):
            self.assertEqual(runner._resource_snapshot(process, 10, budget)["peak_rss_bytes"], 1234)
            with self.assertRaises(runner.FreshTop24SelectionError):
                runner._resource_snapshot(process, 10, budget, reserve_seconds=61)
        process.memory_info.return_value = SimpleNamespace(rss=100)
        with self.assertRaises(runner.FreshTop24SelectionError):
            runner._resource_snapshot(process, 10, budget)

    def test_runner_has_no_faro_model_network_or_phase_b_import(self) -> None:
        tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
        forbidden = (
            "r6_confirmation_io",
            "run_selected_phase_b",
            "depthart_runner",
            "requests",
            "urllib",
        )
        self.assertFalse(any(any(token in module for token in forbidden) for module in imports))


if __name__ == "__main__":
    unittest.main()
