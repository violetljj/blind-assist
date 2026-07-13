from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPTS = Path(__file__).resolve().parent


def module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


freeze = module("freeze_sanpo_v3_regression")
validator = module("validate_sanpo_v3_dataset")
views = module("prepare_sanpo_v3_dataset_views")
model_review = module("review_sanpo_sequence_with_model")
gate = module("sanpo_training_gate")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class SanpoV3DatasetControlsTest(unittest.TestCase):
    def test_expanded_coverage_requires_session_held_out_scenes_and_official_splits(self) -> None:
        buckets = (
            "parallel_boundary", "step_curb", "center_obstacle", "lateral_pedestrian_or_ebike",
        )
        sequences = []
        for bucket_index, bucket in enumerate(buckets):
            for session_index in range(2):
                sequences.append({
                    "sequence_id": f"train-{bucket_index}-{session_index}", "split": "train",
                    "scene_bucket": bucket, "session_id": f"train-session-{bucket_index}-{session_index}",
                    "official_split": "train", "frame_count": 50,
                })
            sequences.append({
                "sequence_id": f"dev-{bucket_index}", "split": "dev", "scene_bucket": bucket,
                "session_id": f"dev-session-{bucket_index}", "official_split": "train", "frame_count": 50,
            })
        train = {
            "row_count": 600, "sequences": sequences,
            "class_presence_frame_count": {name: 1 for name in validator.SEMANTIC_CLASSES},
        }
        blind = {"row_count": 120, "sequences": [
            {"sequence_id": f"blind-{index}", "split": "blind", "scene_bucket": "center_obstacle",
             "session_id": f"blind-session-{index}", "official_split": "test", "frame_count": 60}
            for index in range(2)
        ]}
        policy = {
            "format": validator.EXPANDED_COVERAGE_FORMAT,
            "sequence_frame_count": 50, "blind_sequence_count": 2,
            "blind_sequence_frame_count": 60, "min_train_sessions": 8, "min_dev_sessions": 4,
            "required_scene_sessions": {
                bucket: {"train": 2, "dev": 1, "total": 3} for bucket in buckets
            },
            "official_split_by_target_split": {"train": "train", "dev": "train", "blind": "test"},
        }
        self.assertEqual([], validator.validate_v3_coverage(train, blind, policy))
        sequences[-1]["official_split"] = "test"
        errors = validator.validate_v3_coverage(train, blind, policy)
        self.assertTrue(any("official split" in error for error in errors), errors)
        sequences[-1]["official_split"] = "train"
        sequences[-1]["scene_bucket"] = "parallel_boundary"
        errors = validator.validate_v3_coverage(train, blind, policy)
        self.assertTrue(any("lateral_pedestrian_or_ebike" in error and "dev" in error for error in errors), errors)

    def test_equal_four_class_projection_is_allowed_when_raw_masks_differ(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.make_v3_rows(root)
            first, second = rows[0], rows[1]
            first_mask = root / first["semantic_mask_path"]
            second_mask = root / second["semantic_mask_path"]
            shutil.copyfile(first_mask, second_mask)
            projected_sha = digest(second_mask)
            second["semantic_mask_sha256"] = projected_sha
            second["label_provenance"]["mapped_mask_sha256"] = projected_sha
            second["label_provenance"]["source_mask_sha256"] = "b" * 64
            errors, summary = validator.validate_rows(rows, root, {"train", "dev", "blind"})
            self.assertFalse(any("duplicate" in error and "mask" in error for error in errors), errors)
            self.assertGreater(summary["duplicate_mask_observation"]["semantic_duplicate_row_count"], 0)
            second["label_provenance"]["source_mask_sha256"] = first["label_provenance"]["source_mask_sha256"]
            errors, summary = validator.validate_rows(rows, root, {"train", "dev", "blind"})
            self.assertFalse(any("duplicate" in error and "mask" in error for error in errors), errors)
            self.assertGreater(summary["duplicate_mask_observation"]["raw_duplicate_row_count"], 0)

    def test_model_review_allows_a_high_confidence_no_alert_negative(self) -> None:
        request = {
            "evidence_frames": [{"frame_index": 0}, {"frame_index": 25}, {"frame_index": 49}],
        }
        response = {
            "reviewer": {"type": "model", "model": "test-model", "version_or_date": "2026-07-12"},
            "decision": "accept_for_dense_annotation",
            "primary_scene_bucket": "parallel_boundary",
            "corridor_event_present": False,
            "expected_alert_outcome": "no_alert",
            "confidence": 0.9,
            "evidence_frame_indexes": [0, 25, 49],
            "rationale": "Parallel curb remains outside the walking corridor.",
            "limitations": "Sample fixture only.",
            "selection_evidence_agrees": True,
        }
        self.assertEqual([], model_review.validate_response(request, response))
        response["evidence_frame_indexes"] = [0, 25]
        self.assertTrue(model_review.validate_response(request, response))

    def make_v3_rows(self, root: Path) -> list[dict]:
        buckets = list(validator.SCENE_BUCKETS)
        sequences = [("train", bucket, 50) for bucket in buckets[:3]]
        sequences += [("dev", bucket, 50) for bucket in buckets[3:]]
        sequences += [("blind", "parallel_boundary", 60), ("blind", "low_light", 60)]
        rows: list[dict] = []
        counter = 0
        for seq_index, (split, bucket, frame_count) in enumerate(sequences):
            sequence_id = f"sequence_{seq_index}"
            for frame_index in range(frame_count):
                sample_id = f"sample_{counter:04d}"
                image_rel = Path("images") / f"{sample_id}.png"
                mask_rel = Path("semantic_masks") / f"{sample_id}.png"
                image = root / image_rel
                mask = root / mask_rel
                image.parent.mkdir(parents=True, exist_ok=True)
                mask.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (5, 5), (counter % 256, counter // 256, seq_index)).save(image)
                # A unique 5x5 base-4 pattern; aggregate coverage includes all classes.
                values = []
                value = counter
                for _ in range(25):
                    values.append(value % 4)
                    value //= 4
                Image.frombytes("L", (5, 5), bytes(values)).save(mask)
                mask_sha = digest(mask)
                rows.append({
                    "id": sample_id,
                    "split": split,
                    "session_id": f"session_{seq_index}",
                    "sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "scene_bucket": bucket,
                    "risk_event_id": f"event_{seq_index}",
                    "expected_event_phase": "ALERTED" if frame_index == 2 else "APPROACHING",
                    "expected_should_alert": frame_index >= 2,
                    "image_path": image_rel.as_posix(),
                    "semantic_mask_path": mask_rel.as_posix(),
                    "image_sha256": digest(image),
                    "semantic_mask_sha256": mask_sha,
                    "label_authority": "source_ground_truth",
                    "label_provenance": {
                        "annotation_kind": "pixel_semantic",
                        "source_mask_sha256": mask_sha,
                        "mapped_mask_sha256": mask_sha,
                        "mapping_sha256": "a" * 64,
                        "source_assets": [
                            {"role": "sanpo_rgb", "source_id": "sanpo_fixture", "path": image_rel.as_posix(), "sha256": digest(image)},
                            {"role": "sanpo_raw_mask", "source_id": "sanpo_fixture", "path": mask_rel.as_posix(), "sha256": mask_sha},
                        ],
                    },
                    "source_asset_ids": [f"{sample_id}:sanpo_rgb:0", f"{sample_id}:sanpo_raw_mask:1"],
                    "source": {
                        "source_id": "sanpo_fixture",
                        "dataset": "SANPO-Real",
                        "license": "CC BY 4.0",
                        "license_url": "https://creativecommons.org/licenses/by/4.0/",
                        "privacy_review_status": "approved_no_identifiable_subjects",
                    },
                })
                counter += 1
        return rows

    def write_source_attestation(self, root: Path, rows: list[dict]) -> None:
        evidence = root / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in ("license", "privacy", "inventory"):
            path = evidence / f"{name}.json"
            path.write_text(json.dumps({"kind": name, "fixture": True}), encoding="utf-8")
            files[name] = path
        recipe = root / gate.ASSEMBLY_RECIPE
        recipe.write_text(json.dumps({"fixture": True}), encoding="utf-8")
        inventory = root / gate.ASSET_INVENTORY
        assets = []
        for row in rows:
            for index, item in enumerate(row["label_provenance"]["source_assets"]):
                assets.append({
                    "entry_id": row["source_asset_ids"][index], "sample_id": row["id"],
                    "source_id": item["source_id"], "session_id": row["session_id"],
                    "frame_index": row["frame_index"], "role": item["role"],
                    "path": item["path"], "sha256": item["sha256"],
                })
        inventory.write_text(json.dumps({"schema": gate.ASSET_INVENTORY_SCHEMA, "assets": assets}), encoding="utf-8")
        payload = {
            "schema": gate.ATTESTATION_SCHEMA,
            "recipe_path": recipe.relative_to(root).as_posix(),
            "recipe_sha256": digest(recipe),
            "asset_inventory_path": inventory.relative_to(root).as_posix(),
            "asset_inventory_sha256": digest(inventory),
            "sources": [{
                "source_id": "sanpo_fixture",
                "adapter_id": "sanpo_v0",
                "dataset": "SANPO-Real",
                "dataset_version": "fixture-v0",
                "license": "CC BY 4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "privacy_review_status": "approved_no_identifiable_subjects",
                "license_evidence_path": files["license"].relative_to(root).as_posix(),
                "license_evidence_sha256": digest(files["license"]),
                "privacy_evidence_path": files["privacy"].relative_to(root).as_posix(),
                "privacy_evidence_sha256": digest(files["privacy"]),
                "inventory_path": files["inventory"].relative_to(root).as_posix(),
                "inventory_sha256": digest(files["inventory"]),
            }],
        }
        (root / "source_attestation.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_v3_coverage_and_blind_lock_accepts_separated_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.make_v3_rows(root)
            self.write_source_attestation(root, rows)
            source = root / "reviewed.jsonl"
            write_jsonl(source, rows)
            original_argv = sys.argv
            try:
                sys.argv = ["prepare", "--source-manifest", str(source), "--dataset-root", str(root)]
                self.assertEqual(0, views.main())
            finally:
                sys.argv = original_argv
            train_rows = validator.load_jsonl(root / "training_manifest.jsonl")
            blind_rows = validator.load_jsonl(root / "blind_holdout" / "manifest.jsonl")
            errors, train_summary = validator.validate_rows(train_rows, root, {"train", "dev"})
            blind_errors, blind_summary = validator.validate_rows(blind_rows, root, {"blind"})
            self.assertEqual([], errors + blind_errors)
            self.assertEqual([], validator.validate_access_lock(root, train_rows, blind_rows))
            self.assertEqual([], validator.validate_v3_coverage(train_summary, blind_summary))

    def test_blind_metadata_and_session_leakage_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.make_v3_rows(root)
            self.write_source_attestation(root, rows)
            source = root / "reviewed.jsonl"
            write_jsonl(source, rows)
            original_argv = sys.argv
            try:
                sys.argv = ["prepare", "--source-manifest", str(source), "--dataset-root", str(root)]
                views.main()
            finally:
                sys.argv = original_argv
            train = validator.load_jsonl(root / "training_manifest.jsonl")
            blind = validator.load_jsonl(root / "blind_holdout" / "manifest.jsonl")
            train.append(blind[0])
            errors = validator.validate_access_lock(root, train, blind)
            self.assertTrue(any("blind metadata" in error or "overlap" in error for error in errors))
            blind[0]["session_id"] = train[0]["session_id"]
            self.assertTrue(any("session" in error and "leakage" in error for error in validator.validate_access_lock(root, train[:-1], blind)))
            errors, _ = validator.validate_rows(train, root, {"train", "dev"})
            blind_errors, _ = validator.validate_rows(blind, root, {"blind"})
            self.assertTrue(errors or blind_errors)

    def test_label_authority_is_split_locked_and_teacher_consensus_is_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.make_v3_rows(root)
            train = next(row for row in rows if row["split"] == "train")
            train["label_authority"] = "teacher_consensus_pseudo_label"
            train["label_provenance"] = {
                "teachers": [
                    {"model_id": "teacher-a", "weights_sha256": "1" * 64, "output_sha256": "2" * 64},
                    {"model_id": "teacher-b", "weights_sha256": "3" * 64, "output_sha256": "4" * 64},
                ],
                "agreement_iou": 0.95,
                "temporal_consistency": 0.90,
                "consensus_mask_sha256": train["semantic_mask_sha256"],
            }
            self.assertEqual([], validator.validate_label_authority(train, train["id"], "train"))
            self.assertTrue(any("forbids teacher/pseudo" in item for item in validator.validate_label_authority(train, train["id"], "dev")))
            train["label_provenance"]["agreement_iou"] = 0.50
            self.assertTrue(any("consensus IoU" in item for item in validator.validate_label_authority(train, train["id"], "train")))

    def test_dev_and_blind_accept_only_fully_bound_procedural_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.make_v3_rows(root)
            sample = next(row for row in rows if row["split"] == "blind")
            evidence = root / "procedural_evidence"
            evidence.mkdir()
            tactile = evidence / "tactile.png"
            obstacle = evidence / "obstacle.png"
            code = evidence / "generator.py"
            config = evidence / "config.json"
            Image.new("L", (5, 5), 1).save(tactile)
            Image.new("L", (5, 5), 2).save(obstacle)
            code.write_text("# tactile_occupied_compositor_v1\n", encoding="utf-8")
            config.write_text('{"version":1}\n', encoding="utf-8")
            matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            sample["label_authority"] = "procedural_ground_truth"
            sample["label_provenance"] = {
                "schema": validator.PROCEDURAL_PROVENANCE_SCHEMA,
                "generator_id": "tactile_occupied_compositor_v1",
                "generator_code_path": code.relative_to(root).as_posix(),
                "generator_code_sha256": digest(code),
                "generator_config_path": config.relative_to(root).as_posix(),
                "generator_config_sha256": digest(config),
                "seed": 7113,
                "transform_matrix": matrix,
                "transform_sha256": validator._canonical_json_sha256(matrix),
                "source_masks": [
                    {"role": "tactile_ground_truth", "source_id": "guidetwsi_fixture", "path": tactile.relative_to(root).as_posix(), "sha256": digest(tactile)},
                    {"role": "obstacle_ground_truth", "source_id": "sanpo_fixture", "path": obstacle.relative_to(root).as_posix(), "sha256": digest(obstacle)},
                ],
                "source_assets": [
                    {"role": "guide_rgb", "source_id": "guidetwsi_fixture", "path": tactile.relative_to(root).as_posix(), "sha256": digest(tactile)},
                    {"role": "guide_polygon", "source_id": "guidetwsi_fixture", "path": config.relative_to(root).as_posix(), "sha256": digest(config)},
                    {"role": "sanpo_rgb", "source_id": "sanpo_fixture", "path": sample["image_path"], "sha256": sample["image_sha256"]},
                    {"role": "sanpo_raw_mask", "source_id": "sanpo_fixture", "path": obstacle.relative_to(root).as_posix(), "sha256": digest(obstacle)},
                ],
                "output_mask_sha256": sample["semantic_mask_sha256"],
            }
            self.assertEqual([], validator.validate_label_authority(sample, sample["id"], "blind", root))
            sample["label_provenance"]["source_masks"][0]["sha256"] = "0" * 64
            errors = validator.validate_label_authority(sample, sample["id"], "blind", root)
            self.assertTrue(any("provenance SHA256 mismatch" in item for item in errors))
            sample["label_authority"] = "teacher_consensus_pseudo_label"
            errors = validator.validate_label_authority(sample, sample["id"], "blind", root)
            self.assertTrue(any("forbids teacher/pseudo" in item for item in errors))

    def test_semantic_only_rows_omit_and_cannot_smuggle_event_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.make_v3_rows(root)
            for row in rows:
                row["benchmark_kind"] = "semantic_segmentation_only"
                row.pop("risk_event_id")
                row.pop("expected_event_phase")
                row.pop("expected_should_alert")
            train_dev = [row for row in rows if row["split"] in {"train", "dev"}]
            errors, _ = validator.validate_rows(train_dev, root, {"train", "dev"})
            self.assertEqual([], errors)
            train_dev[0]["expected_should_alert"] = True
            errors, _ = validator.validate_rows(train_dev, root, {"train", "dev"})
            self.assertTrue(any("must not carry risk-event labels" in item for item in errors))

    def test_gate_requires_sha_bound_source_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.make_v3_rows(root)
            self.write_source_attestation(root, rows)
            source = root / "reviewed.jsonl"
            write_jsonl(source, rows)
            original_argv = sys.argv
            try:
                sys.argv = ["prepare", "--source-manifest", str(source), "--dataset-root", str(root)]
                self.assertEqual(0, views.main())
            finally:
                sys.argv = original_argv
            report = gate.build_report(root)
            self.assertEqual("green", report["overall_status"], report["errors"])
            inventory = root / "evidence" / "inventory.json"
            inventory.write_text('{"tampered":true}', encoding="utf-8")
            report = gate.build_report(root)
            self.assertEqual("red", report["overall_status"])
            self.assertTrue(any("inventory" in item and "SHA256" in item for item in report["errors"]))

    def test_gate_rejects_semantically_incomplete_and_cross_split_raw_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.make_v3_rows(root)
            self.write_source_attestation(root, rows)
            source = root / "reviewed.jsonl"
            write_jsonl(source, rows)
            original_argv = sys.argv
            try:
                sys.argv = ["prepare", "--source-manifest", str(source), "--dataset-root", str(root)]
                self.assertEqual(0, views.main())
            finally:
                sys.argv = original_argv

            inventory_path = root / gate.ASSET_INVENTORY
            attestation_path = root / "source_attestation.json"
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            removed = inventory["assets"].pop()
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            attestation["asset_inventory_sha256"] = digest(inventory_path)
            attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
            report = gate.build_report(root)
            self.assertEqual("red", report["overall_status"])
            self.assertTrue(any("missing inventory" in item or "unreferenced" in item for item in report["errors"]))

            inventory["assets"].append(removed)
            train_path = root / gate.CANONICAL_TRAINING_MANIFEST
            train_rows = validator.load_jsonl(train_path)
            train_row = next(row for row in train_rows if row["split"] == "train")
            dev_row = next(row for row in train_rows if row["split"] == "dev")
            train_asset = next(
                item for item in inventory["assets"]
                if item["sample_id"] == train_row["id"] and item["role"] == "sanpo_rgb"
            )
            dev_asset = next(
                item for item in inventory["assets"]
                if item["sample_id"] == dev_row["id"] and item["role"] == "sanpo_rgb"
            )
            for field in ("path", "sha256"):
                dev_asset[field] = train_asset[field]
            dev_declared = next(
                item for item in dev_row["label_provenance"]["source_assets"] if item["role"] == "sanpo_rgb"
            )
            for field in ("path", "sha256"):
                dev_declared[field] = train_asset[field]
            write_jsonl(train_path, train_rows)
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            attestation["asset_inventory_sha256"] = digest(inventory_path)
            attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
            report = gate.build_report(root)
            self.assertEqual("red", report["overall_status"])
            self.assertTrue(any("crosses splits" in item for item in report["errors"]))

    def test_training_authorization_does_not_need_blind_manifest_or_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.make_v3_rows(root)
            self.write_source_attestation(root, rows)
            source = root / "reviewed.jsonl"
            write_jsonl(source, rows)
            original_argv = sys.argv
            try:
                sys.argv = ["prepare", "--source-manifest", str(source), "--dataset-root", str(root)]
                self.assertEqual(0, views.main())
            finally:
                sys.argv = original_argv
            report_path = root / "qa" / "training_gate_report.json"
            report = gate.run_gate(root, report_path)
            self.assertEqual("green", report["overall_status"], report["errors"])
            (root / "blind_holdout").rename(root / "blind_holdout.inaccessible")
            verified = gate.consume_training_authorization(root, report_path)
            self.assertEqual(report["report_sha256"], verified["report_sha256"])
            manifest = root / gate.CANONICAL_TRAINING_MANIFEST
            manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                gate.consume_training_authorization(root, report_path)

    def test_final_root_report_must_be_regenerated_after_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            staging = base / "canonical.building"
            published = base / "canonical"
            staging.mkdir()
            rows = self.make_v3_rows(staging)
            self.write_source_attestation(staging, rows)
            source = staging / "reviewed.jsonl"
            write_jsonl(source, rows)
            original_argv = sys.argv
            try:
                sys.argv = ["prepare", "--source-manifest", str(source), "--dataset-root", str(staging)]
                self.assertEqual(0, views.main())
            finally:
                sys.argv = original_argv
            gate.run_gate(staging, staging / "qa" / "training_gate_report.json")
            shutil.copytree(staging, published)
            with self.assertRaisesRegex(ValueError, "different dataset root"):
                gate.consume_training_authorization(published, published / "qa" / "training_gate_report.json")
            final_report = gate.run_gate(published, published / "qa" / "training_gate_report.json")
            verified = gate.consume_training_authorization(published, published / "qa" / "training_gate_report.json")
            self.assertEqual(final_report["report_sha256"], verified["report_sha256"])

    def test_90_frame_regression_lock_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            for directory in (source / "images" / "test", source / "source_masks" / "test"):
                directory.mkdir(parents=True, exist_ok=True)
            rows: list[dict] = []
            for index in range(90):
                sample_id = f"frame_{index:03d}"
                image = source / "images" / "test" / f"{sample_id}.png"
                mask = source / "source_masks" / "test" / f"{sample_id}.png"
                Image.new("RGB", (2, 2), (index, 0, 0)).save(image)
                Image.new("L", (2, 2), index % 4).save(mask)
                rows.append({"id": sample_id, "image_path": f"images/test/{sample_id}.png"})
            write_jsonl(source / "manifest.jsonl", rows)
            config = root / "benchmark-config.json"
            config.write_text('{"runs_per_frame":3}\n', encoding="utf-8")
            output = root / "sanpo-v3-regression-90f"
            result = subprocess.run([
                sys.executable, str(SCRIPTS / "freeze_sanpo_v3_regression.py"),
                "--source-root", str(source), "--output-root", str(output),
                "--benchmark-config", str(config), "--device-report", "device-report-20260711",
            ], text=True, capture_output=True)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertEqual([], freeze.verify_lock(output))
            Image.new("RGB", (2, 2), "white").save(output / "images" / "test" / "frame_000.png")
            self.assertTrue(any("SHA256 differs" in error for error in freeze.verify_lock(output)))


if __name__ == "__main__":
    unittest.main()
