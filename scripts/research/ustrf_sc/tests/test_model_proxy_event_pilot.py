#!/usr/bin/env python3
"""Adversarial tests for the frozen USTRF model-proxy pilot contract."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import validate_ustrf_sc_model_proxy_event_pilot as adapter  # noqa: E402
from research.ustrf_sc.validate_model_proxy_event_pilot import (  # noqa: E402
    ContractError,
    EXPECTED_ROLES,
    EXPECTED_SCENES,
    EXPECTED_SESSION_ID,
    sha256,
    validate,
)


CONFIG_PATH = REPO_ROOT / "configs" / "ustrf_sc_model_proxy_route_event_pilot_v1.json"
ZERO_SHA = "0" * 64
ONE_SHA = "1" * 64


def _write_bytes(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> str:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return _write_bytes(path, encoded)


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> str:
    encoded = ("\n".join(json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values) + "\n").encode(
        "utf-8"
    )
    return _write_bytes(path, encoded)


class PilotFixture:
    """Build a valid ten-episode manifest without invoking real media tools."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.config_sha256 = _write_json(self.root / "frozen_config.json", self.config)
        self.toolchain = {
            "ffmpeg_sha256": "a" * 64,
            "ffmpeg_version": "ffmpeg test-double 1",
            "ffprobe_sha256": "b" * 64,
            "ffprobe_version": "ffprobe test-double 1",
        }
        self.frames_by_video: dict[Path, list[dict[str, object]]] = {}
        self.media_by_video: dict[Path, dict[str, object]] = {}
        self.image_dimensions: dict[Path, dict[str, int]] = {}
        self.manifest: dict[str, object] = {
            "schema": "blindassist_ustrf_sc_model_proxy_route_event_manifest_v1",
            "contract_id": self.config["contract_id"],
            "frozen_config_path": "frozen_config.json",
            "frozen_config_sha256": self.config_sha256,
            "toolchain": copy.deepcopy(self.toolchain),
            "benchmark_only": True,
            "human_truth": False,
            "training_eligible": False,
            "android_runtime_authorized": False,
            "production_authorized": False,
            "proxy_full_matrix_expansion_eligible": True,
            "proxy_u0_evaluation_eligible": False,
            "status": "review_complete",
            "episodes": [],
        }
        prompt_hashes: dict[str, str] = {}
        for role in self.config["review"]["required_reviewer_roles"]:
            prompt_hashes[role] = _write_bytes(
                self.root / "prompts" / f"{role}.txt", f"frozen prompt for {role}\n".encode("utf-8")
            )
        for scene in EXPECTED_SCENES:
            for pair_role in EXPECTED_ROLES:
                self.manifest["episodes"].append(self._episode(scene=scene, pair_role=pair_role))
        self._attach_generation_provenance()
        self._attach_independent_reviews(prompt_hashes)
        self.manifest_path = self.root / "manifest.json"
        self.manifest_sha256 = _write_json(self.manifest_path, self.manifest)

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _episode(self, *, scene: str, pair_role: str) -> dict[str, object]:
        pair_id = f"{EXPECTED_SESSION_ID}__{scene}__pair_01"
        episode_id = f"{pair_id}__{pair_role}"
        stem = f"{scene}__{pair_role}"

        sheet_path = self.root / "media" / f"{stem}__contact.png"
        sheet_sha = _write_bytes(sheet_path, f"contact:{episode_id}".encode("utf-8"))
        self.image_dimensions[sheet_path.resolve()] = {"width": 1536, "height": 1024}

        panels: list[dict[str, object]] = []
        for panel_index, stage in enumerate(("first_visible", "approach", "alertable", "cleared"), start=1):
            panel_path = self.root / "media" / f"{stem}__panel_{panel_index}.png"
            panel_sha = _write_bytes(panel_path, f"panel:{episode_id}:{panel_index}".encode("utf-8"))
            self.image_dimensions[panel_path.resolve()] = {"width": 1280, "height": 720}
            panels.append(
                {
                    "panel_index": panel_index,
                    "lifecycle_stage": stage,
                    "path": self._relative(panel_path),
                    "sha256": panel_sha,
                }
            )

        video_path = self.root / "media" / f"{stem}.mp4"
        video_sha = _write_bytes(video_path, f"video:{episode_id}".encode("utf-8"))
        media = {
            "width": 1280,
            "height": 720,
            "avg_frame_rate": "10/1",
            "time_base": "1/10240",
            "duration_ms": 10000,
        }
        self.media_by_video[video_path.resolve()] = copy.deepcopy(media)
        frames = [
            {
                "frame_index": frame_index,
                "stream_index": 0,
                "dts": frame_index * 1024,
                "pts": frame_index * 1024,
                "duration": 1024,
                "decoded_size_bytes": 1280 * 720 * 3,
                "decoded_frame_sha256": hashlib.sha256(
                    f"{episode_id}:{frame_index}".encode("utf-8")
                ).hexdigest(),
            }
            for frame_index in range(100)
        ]
        self.frames_by_video[video_path.resolve()] = copy.deepcopy(frames)
        ledger_path = self.root / "ledgers" / f"{stem}.json"
        ledger = {
            "schema": "blindassist_ustrf_sc_model_proxy_frame_ledger_v1",
            "episode_id": episode_id,
            "video_sha256": video_sha,
            "time_base": media["time_base"],
            "frames": frames,
        }
        ledger_sha = _write_json(ledger_path, ledger)

        review_input_path = self.root / "review_inputs" / f"{stem}.json"
        review_input = {
            "schema": "blindassist_ustrf_sc_model_review_input_v1",
            "episode_id": episode_id,
            "contact_sheet_sha256": sheet_sha,
            "video_sha256": video_sha,
            "frame_ledger_sha256": ledger_sha,
            "candidate_output_included": False,
            "panel_sha256s": [panel["sha256"] for panel in panels],
        }
        review_input_sha = _write_json(review_input_path, review_input)
        return {
            "episode_id": episode_id,
            "session_id": EXPECTED_SESSION_ID,
            "scene_id": scene,
            "matched_pair_id": pair_id,
            "pair_role": pair_role,
            "expected_should_alert": pair_role == "positive",
            "authority": "synthetic_or_model_proxy_only",
            "contact_sheet_path": self._relative(sheet_path),
            "contact_sheet_sha256": sheet_sha,
            "panels": panels,
            "video_path": self._relative(video_path),
            "video_sha256": video_sha,
            "media": media,
            "frame_ledger_path": self._relative(ledger_path),
            "frame_ledger_sha256": ledger_sha,
            "review_input_path": self._relative(review_input_path),
            "review_input_sha256": review_input_sha,
            "independent_model_reviews": [],
            "model_adjudication": None,
        }

    def _attach_generation_provenance(self) -> None:
        dataset_spec_path = self.root / "dataset_spec.json"
        dataset_spec_sha = _write_json(
            dataset_spec_path,
            {"schema": "test_model_proxy_dataset_spec_v1", "episode_count": len(self.episodes)},
        )
        records_path = self.root / "generation_records.jsonl"
        records_sha = _write_jsonl(
            records_path,
            [
                {
                    "file": Path(episode["contact_sheet_path"]).name,
                    "sha256": episode["contact_sheet_sha256"],
                    "accepted": True,
                }
                for episode in self.episodes
            ],
        )
        preview_path = self.root / "qa" / "preview.html"
        preview_sha = _write_bytes(preview_path, b"<!doctype html><title>model proxy QA</title>\n")
        self.manifest["model_generation_provenance"] = {
            "dataset_spec": {"path": "dataset_spec.json", "sha256": dataset_spec_sha},
            "generation_records": {"path": "generation_records.jsonl", "sha256": records_sha},
            "qa_preview": {"path": "qa/preview.html", "sha256": preview_sha},
        }

    def _attach_independent_reviews(self, prompt_hashes: dict[str, str]) -> None:
        sources: list[dict[str, str]] = []
        raw_rows_by_role: dict[str, list[dict[str, object]]] = {}
        for review_index, reviewer_role in enumerate(
            self.config["review"]["required_reviewer_roles"], start=1
        ):
            reviewer_id = f"model_proxy_global_reviewer_{review_index}"
            raw_rows = []
            for episode in self.episodes:
                relations = (
                    ["outside", "boundary", "inside", "clear"]
                    if episode["pair_role"] == "positive"
                    else ["outside", "outside", "outside", "outside"]
                )
                raw_rows.append(
                    {
                        "file": Path(episode["contact_sheet_path"]).name,
                        "accept": True,
                        "observed_role": episode["pair_role"],
                        "panel_relations": relations,
                        "temporal_coherence": True,
                        "issue_tags": [],
                        "rationale": "Bound synthetic receipt for validator tests.",
                    }
                )
            raw_path = self.root / "raw_reviews" / f"reviewer_{review_index}.json"
            raw_sha = _write_json(raw_path, raw_rows)
            prompt_path = self.root / "prompts" / f"{reviewer_role}.txt"
            sources.append(
                {
                    "reviewer_id": reviewer_id,
                    "raw_output_path": self._relative(raw_path),
                    "raw_output_sha256": raw_sha,
                    "prompt_path": self._relative(prompt_path),
                    "prompt_sha256": prompt_hashes[reviewer_role],
                }
            )
            raw_rows_by_role[reviewer_role] = raw_rows
        self.manifest["independent_review_run_sources"] = sources

        for episode in self.episodes:
            stem = f"{episode['scene_id']}__{episode['pair_role']}"
            bindings: list[dict[str, str]] = []
            for review_index, (reviewer_role, source) in enumerate(
                zip(self.config["review"]["required_reviewer_roles"], sources), start=1
            ):
                raw_row = next(
                    row
                    for row in raw_rows_by_role[reviewer_role]
                    if row["file"] == Path(episode["contact_sheet_path"]).name
                )
                review_path = self.root / "reviews" / f"{stem}__review_{review_index}.json"
                review = {
                    "schema": self.config["review"]["review_schema"],
                    "episode_id": episode["episode_id"],
                    "reviewer_id": source["reviewer_id"],
                    "reviewer_type": "ai_model",
                    "reviewer_role": reviewer_role,
                    "provider": "test-double",
                    "model": "test-model",
                    "model_version": "v1",
                    "review_run_id": f"{episode['episode_id']}__run_{review_index}",
                    "workflow_id": self.config["review"]["workflow_id"],
                    "prompt_path": source["prompt_path"],
                    "prompt_sha256": source["prompt_sha256"],
                    "raw_output_path": source["raw_output_path"],
                    "raw_output_sha256": source["raw_output_sha256"],
                    "input_sha256": episode["review_input_sha256"],
                    "isolated_context": True,
                    "other_review_visible_before_submission": False,
                    "candidate_output_visible": False,
                    "confidence": 0.99,
                    "abstained": False,
                    "abstain_reasons": [],
                    "contact_sheet_sha256": episode["contact_sheet_sha256"],
                    "video_sha256": episode["video_sha256"],
                    "frame_ledger_sha256": episode["frame_ledger_sha256"],
                    "verdict": "accept",
                    "accept": raw_row["accept"],
                    "observed_role": raw_row["observed_role"],
                    "panel_relations": list(raw_row["panel_relations"]),
                    "raw_panel_relations": list(raw_row["panel_relations"]),
                    "temporal_coherence": raw_row["temporal_coherence"],
                    "issue_tags": list(raw_row["issue_tags"]),
                    "rationale": raw_row["rationale"],
                }
                bindings.append(
                    {"path": self._relative(review_path), "sha256": _write_json(review_path, review)}
                )
            episode["independent_model_reviews"] = bindings

    @property
    def episodes(self) -> list[dict[str, object]]:
        return self.manifest["episodes"]

    def read_bound_json(self, relative: str) -> dict[str, object]:
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def rewrite_review(self, episode_index: int, review_index: int, review: dict[str, object]) -> None:
        binding = self.episodes[episode_index]["independent_model_reviews"][review_index]
        binding["sha256"] = _write_json(self.root / binding["path"], review)

    def run(
        self,
        *,
        config: dict[str, object] | None = None,
        manifest: dict[str, object] | None = None,
        config_sha256: str | None = None,
        manifest_sha256: str | None = None,
        toolchain: dict[str, str] | None = None,
    ) -> dict[str, object]:
        return validate(
            self.config if config is None else config,
            self.manifest if manifest is None else manifest,
            root=self.root,
            config_sha256=self.config_sha256 if config_sha256 is None else config_sha256,
            manifest_sha256=self.manifest_sha256 if manifest_sha256 is None else manifest_sha256,
            actual_toolchain=self.toolchain if toolchain is None else toolchain,
            frame_provider=lambda path: copy.deepcopy(self.frames_by_video[path.resolve()]),
            media_provider=lambda path: copy.deepcopy(self.media_by_video[path.resolve()]),
            image_provider=lambda path: copy.deepcopy(self.image_dimensions[path.resolve()]),
        )


class ModelProxyEventPilotAdversarialTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.fixture = PilotFixture(Path(self._temporary.name))

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def assert_rejected(self, *, pattern: str | None = None, **kwargs: object) -> None:
        with self.assertRaisesRegex(ContractError, pattern or ".+"):
            self.fixture.run(**kwargs)

    def test_valid_fixture_passes_and_reports_hash_bound_receipts(self) -> None:
        report = self.fixture.run()
        self.assertTrue(report["model_proxy_pipeline_audit_passed"])
        self.assertEqual(self.fixture.config_sha256, report["config_sha256"])
        self.assertEqual(self.fixture.manifest_sha256, report["manifest_sha256"])
        self.assertEqual(self.fixture.toolchain, report["toolchain"])
        self.assertEqual(20, len(report["review_receipt_sha256s"]))
        self.assertTrue(report["proxy_full_matrix_expansion_eligible"])
        self.assertFalse(report["proxy_u0_evaluation_eligible"])

    def test_pilot_manifest_can_never_self_authorize_proxy_u0(self) -> None:
        self.fixture.manifest["proxy_u0_evaluation_eligible"] = True
        self.assert_rejected(pattern="proxy_u0_evaluation_eligible=false")

    def test_empty_or_weakened_arbitrary_config_is_rejected(self) -> None:
        for candidate in ({}, {"schema": "arbitrary"}, copy.deepcopy(self.fixture.config)):
            if candidate.get("matrix"):
                candidate["matrix"]["scene_ids"] = []
                candidate["matrix"]["expected_episode_count"] = 0
            with self.subTest(candidate=candidate):
                self.assert_rejected(config=candidate, pattern="config|schema|matrix")

    def test_absolute_evidence_path_is_rejected(self) -> None:
        episode = self.fixture.episodes[0]
        episode["contact_sheet_path"] = str((self.fixture.root / episode["contact_sheet_path"]).resolve())
        self.assert_rejected(pattern="relative path")

    def test_duplicate_episode_id_is_rejected(self) -> None:
        self.fixture.episodes[1]["episode_id"] = self.fixture.episodes[0]["episode_id"]
        self.assert_rejected(pattern="episode IDs")

    def test_duplicate_video_media_is_rejected(self) -> None:
        first = self.fixture.episodes[0]
        second = self.fixture.episodes[1]
        second["video_path"] = first["video_path"]
        second["video_sha256"] = first["video_sha256"]
        self.assert_rejected(pattern="reuses a video")

    def test_wrong_lifecycle_stage_is_rejected(self) -> None:
        self.fixture.episodes[0]["panels"][2]["lifecycle_stage"] = "approach"
        self.assert_rejected(pattern="index/stage")

    def test_malformed_frame_object_fails_closed(self) -> None:
        episode = self.fixture.episodes[0]
        ledger_path = self.fixture.root / episode["frame_ledger_path"]
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["frames"][0] = []
        episode["frame_ledger_sha256"] = _write_json(ledger_path, ledger)
        self.assert_rejected(pattern=r"frames\[0\] must be an object")

    def test_review_role_bypass_is_rejected(self) -> None:
        episode = self.fixture.episodes[0]
        first = self.fixture.read_bound_json(episode["independent_model_reviews"][0]["path"])
        second = self.fixture.read_bound_json(episode["independent_model_reviews"][1]["path"])
        second["reviewer_role"] = first["reviewer_role"]
        self.fixture.rewrite_review(0, 1, second)
        self.assert_rejected(pattern="reviewer roles")

    def test_review_run_reuse_is_rejected(self) -> None:
        episode = self.fixture.episodes[0]
        first = self.fixture.read_bound_json(episode["independent_model_reviews"][0]["path"])
        second = self.fixture.read_bound_json(episode["independent_model_reviews"][1]["path"])
        second["review_run_id"] = first["review_run_id"]
        self.fixture.rewrite_review(0, 1, second)
        self.assert_rejected(pattern="identities/runs")

    def test_review_input_hash_bypass_is_rejected(self) -> None:
        episode = self.fixture.episodes[0]
        review = self.fixture.read_bound_json(episode["independent_model_reviews"][0]["path"])
        review["input_sha256"] = ONE_SHA
        self.fixture.rewrite_review(0, 0, review)
        self.assert_rejected(pattern="workflow/input binding")

    def test_positive_panel4_must_be_canonical_clear_not_raw_outside(self) -> None:
        episode_index = next(
            index for index, episode in enumerate(self.fixture.episodes) if episode["pair_role"] == "positive"
        )
        episode = self.fixture.episodes[episode_index]
        review = self.fixture.read_bound_json(episode["independent_model_reviews"][0]["path"])
        review["panel_relations"][3] = "outside"
        review["raw_panel_relations"][3] = "outside"
        self.fixture.rewrite_review(episode_index, 0, review)
        self.assert_rejected(pattern="intrusion then clearance")

    def test_accepted_generation_record_hash_cannot_be_rebound_to_different_media(self) -> None:
        provenance = self.fixture.manifest["model_generation_provenance"]["generation_records"]
        records_path = self.fixture.root / provenance["path"]
        records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
        records[0]["sha256"] = ONE_SHA
        provenance["sha256"] = _write_jsonl(records_path, records)
        self.assert_rejected(pattern="accepted model-generation record")

    def test_raw_review_output_semantics_remain_bound_after_rehash(self) -> None:
        source = self.fixture.manifest["independent_review_run_sources"][0]
        raw_path = self.fixture.root / source["raw_output_path"]
        raw_rows = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_rows[0]["rationale"] = "Tampered raw model rationale."
        source["raw_output_sha256"] = _write_json(raw_path, raw_rows)
        for episode_index, episode in enumerate(self.fixture.episodes):
            review = self.fixture.read_bound_json(episode["independent_model_reviews"][0]["path"])
            review["raw_output_sha256"] = source["raw_output_sha256"]
            self.fixture.rewrite_review(episode_index, 0, review)
        self.assert_rejected(pattern="differs from raw output field rationale")

    def test_frozen_config_hash_and_toolchain_are_bound(self) -> None:
        self.assert_rejected(config_sha256=ONE_SHA, pattern="frozen config")
        changed_toolchain = copy.deepcopy(self.fixture.toolchain)
        changed_toolchain["ffmpeg_sha256"] = ZERO_SHA
        self.assert_rejected(toolchain=changed_toolchain, pattern="toolchain receipt")

    def test_hash_arguments_fail_closed_when_not_sha256(self) -> None:
        self.assert_rejected(config_sha256="not-a-sha", pattern="config_sha256")
        self.assert_rejected(manifest_sha256="not-a-sha", pattern="manifest_sha256")

    def test_adapter_hashes_current_manifest_and_binds_executing_toolchain(self) -> None:
        ffmpeg = self.fixture.root / "ffmpeg.exe"
        ffprobe = self.fixture.root / "ffprobe.exe"
        _write_bytes(ffmpeg, b"fake ffmpeg")
        _write_bytes(ffprobe, b"fake ffprobe")
        output = self.fixture.root / "audit.json"
        fresh_toolchain = {
            "ffmpeg_sha256": sha256(ffmpeg),
            "ffmpeg_version": "fake ffmpeg version",
            "ffprobe_sha256": sha256(ffprobe),
            "ffprobe_version": "fake ffprobe version",
        }
        expected_report = {
            "proxy_full_matrix_expansion_eligible": True,
            "proxy_u0_evaluation_eligible": False,
            "model_proxy_pipeline_audit_passed": True,
        }
        argv = [
            "validate_ustrf_sc_model_proxy_event_pilot.py",
            "--config", str(CONFIG_PATH),
            "--manifest", str(self.fixture.manifest_path),
            "--ffmpeg", str(ffmpeg),
            "--ffprobe", str(ffprobe),
            "--output", str(output),
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(adapter, "toolchain_receipt", return_value=fresh_toolchain),
            mock.patch.object(adapter, "validate", return_value=expected_report) as mocked_validate,
        ):
            self.assertEqual(0, adapter.main())
        call = mocked_validate.call_args
        self.assertEqual(sha256(CONFIG_PATH), call.kwargs["config_sha256"])
        self.assertEqual(sha256(self.fixture.manifest_path), call.kwargs["manifest_sha256"])
        self.assertEqual(fresh_toolchain, call.kwargs["actual_toolchain"])
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
