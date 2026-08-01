from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_stage_c_t0_sanpo_short_path_equivalence as subject


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _md5(payload: bytes) -> str:
    return base64.b64encode(
        hashlib.md5(payload, usedforsecurity=False).digest()
    ).decode("ascii")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class EquivalenceFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.session_id = "c" * 64
        self.canonical = root / "canonical"
        self.candidate = root / "candidate"
        self.evidence = root / "evidence"
        docs_root = subject._repo_root() / "docs/research/hftf"
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix=".t0-equivalence-test-",
            dir=docs_root,
            delete=False,
        )
        handle.close()
        self.contract_path = Path(handle.name)
        self.source_indices = [0, 2]
        self.acquisition_token = "0123456789abcdef"
        self._make_package(self.canonical, short=False)
        self._make_package(self.candidate, short=True)
        self._make_reports()
        self.contract = self._contract()
        _write_json(self.contract_path, self.contract)
        self._make_transport_receipt()

    @staticmethod
    def _inventory(name: str, payload: bytes) -> dict[str, object]:
        return {
            "name": name,
            "generation": "7",
            "metageneration": "1",
            "size": len(payload),
            "md5_base64": _md5(payload),
            "crc32c_base64": "fixture-crc",
        }

    def _make_package(self, package: Path, *, short: bool) -> None:
        (package / "source_licenses.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (package / "source_licenses.md").write_text(
            "fixture license\n", encoding="utf-8"
        )
        metadata_payloads = {
            "description": b'{"session_type":"synthetic"}\n',
            "labelmap": b'{"1":"ground"}\n',
            "annotation_types": b'{"0":"HIGH","2":"HIGH"}\n',
            "camera_poses": (
                b"tracking_state,pos_x,pos_y,pos_z,q_x,q_y,q_z,q_w\n"
                b"TrackingState.READY,0,0,1.7,0,0,0,1\n"
            ),
        }
        metadata_names = {
            "description": "sanpo/session/description.json",
            "labelmap": "sanpo/labelmap.json",
            "annotation_types": "sanpo/session/annotation_types.json",
            "camera_poses": "sanpo/session/camera_poses.csv",
        }
        inventory: dict[str, object] = {}
        for key, relative in subject.METADATA_PATHS.items():
            payload = metadata_payloads[key]
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            inventory[key] = self._inventory(metadata_names[key], payload)
        split_payload = (self.session_id + "\n").encode()
        if short:
            split_path = (
                package
                / "source_metadata/official_split_session_ids.txt"
            )
            split_path.write_bytes(split_payload)
        inventory["official_split_receipt"] = self._inventory(
            "sanpo/splits/train_session_ids.txt", split_payload
        )

        rows: list[dict[str, object]] = []
        modality_inventories: dict[str, list[dict[str, object]]] = {
            "rgb": [],
            "masks": [],
            "depth": [],
        }
        sequence_id = (
            f"sanpo_synthetic_{self.session_id}_camera_chest_left_"
            "000000_10fps"
        )
        for timeline_index, source_index in enumerate(self.source_indices):
            alias = f"{timeline_index:02x}"
            if short:
                paths = {
                    "rgb": f"i/train/{alias}.png",
                    "masks": f"m/train/{alias}.png",
                    "depth": f"d/train/{alias}.f16.gz",
                }
            else:
                sample = f"{sequence_id}_{timeline_index:06d}"
                paths = {
                    "rgb": f"images/train/{sample}.png",
                    "masks": f"source_masks/train/{sample}.png",
                    "depth": f"source_depth/train/{sample}.float16.gz",
                }
            payloads = {
                "rgb": f"rgb-{source_index}".encode(),
                "masks": f"mask-{source_index}".encode(),
                "depth": f"depth-{source_index}".encode(),
            }
            objects = {
                "rgb": self._inventory(
                    f"sanpo/video_frames/{source_index:06d}.png",
                    payloads["rgb"],
                ),
                "masks": self._inventory(
                    f"sanpo/segmentation_masks/{source_index:06d}.png",
                    payloads["masks"],
                ),
                "depth": self._inventory(
                    f"sanpo/depth_maps/{source_index:06d}.float16.gz",
                    payloads["depth"],
                ),
            }
            for key in ("rgb", "masks", "depth"):
                path = package / paths[key]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payloads[key])
                modality_inventories[key].append(objects[key])
            sample_id = f"{sequence_id}_{timeline_index:06d}"
            rows.append(
                {
                    "id": sample_id,
                    "image_path": paths["rgb"],
                    "image_sha256": _sha256(package / paths["rgb"]),
                    "source_mask_path": paths["masks"],
                    "source_mask_sha256": _sha256(package / paths["masks"]),
                    "source_depth_path": paths["depth"],
                    "source_depth_sha256": _sha256(package / paths["depth"]),
                    "width": 8,
                    "height": 6,
                    "session_id": self.session_id,
                    "sequence_id": sequence_id,
                    "frame_index": timeline_index,
                    "source_frame_index": source_index,
                    "source_timestamp_ms": source_index * 50,
                    "label_authority": (
                        "official_panoptic_ground_truth_pretraining_only"
                    ),
                    "event_truth": None,
                    "source": {
                        "source_id": "sanpo_synthetic_v0",
                        "dataset": "SANPO-Synthetic v0",
                        "official_split": "train",
                        "session_id": self.session_id,
                        "camera": "camera_chest",
                        "lens": "left",
                    },
                    "modalities": {
                        "rgb": objects["rgb"],
                        "panoptic_mask": objects["masks"],
                        "metric_depth": objects["depth"],
                    },
                }
            )
        inventory.update(modality_inventories)
        spec = {
            "schema": subject.REPLAY_SCHEMA,
            "source": {
                "source_id": "sanpo_synthetic_v0",
                "dataset": "SANPO-Synthetic v0",
                "official_split": "train",
                "session_id": self.session_id,
            },
            "sampling": {
                "source_fps": 20.0,
                "target_fps": 10.0,
                "selected_source_frames": self.source_indices,
            },
            "camera": {
                "image_width": 8,
                "image_height": 6,
                "fx": 4.0,
                "fy": 4.0,
                "cx": 3.5,
                "cy": 2.5,
            },
            "source_inventory": inventory,
        }
        if short:
            spec["local_transport"] = {
                "schema": subject.TRANSPORT_SCHEMA,
                "identity_in_manifest_not_local_filename": True,
            }
        _write_json(package / "dataset_spec.json", spec)
        (package / "manifest.replay.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _path_report(self, maximum: int = 180) -> dict[str, object]:
        return {
            "maximum_content_path_length": maximum,
            "content_path_count_including_tmp": 100,
            "limit_exclusive": 240,
            "all_content_paths_under_limit": True,
            "session_id_present_in_any_content_path": False,
        }

    def _make_reports(self) -> None:
        self.canary_path = self.evidence / "canary.json"
        _write_json(
            self.canary_path,
            {
                "schema": subject.CANARY_SCHEMA,
                "terminal": subject.CANARY_READY,
                "long_session_id_excluded_from_content_paths": True,
                "path_preflight": self._path_report(),
                "generation_bound_url_observed": True,
                "size_and_md5_verified": True,
                "network_opened": False,
                "source_opened": False,
                "fresh_or_reserved_source_opened": False,
            },
        )
        self.preflight_path = self.evidence / "preflight.json"
        _write_json(
            self.preflight_path,
            {
                "schema": subject.TRANSPORT_SCHEMA,
                "terminal": subject.PREFLIGHT_READY,
                "acquisition_token": self.acquisition_token,
                "planned_output_root": str(self.candidate.resolve()),
                "path_preflight": self._path_report(),
                "network_opened": False,
                "source_opened": False,
                "output_created": False,
            },
        )

    def _contract(self) -> dict[str, object]:
        implementations = {}
        for key, relative in subject.IMPLEMENTATIONS.items():
            implementations[key] = {
                "path": relative,
                "sha256": _sha256(subject._repo_root() / relative),
            }
        implementations["acquirer"]["network_execution_authorized"] = True
        return {
            "schema": subject.CONTRACT_SCHEMA,
            "status": subject.CONTRACT_STATUS,
            "implementations": implementations,
            "source": {
                "session_id": self.session_id,
                "official_split": "train",
                "role": subject.EXECUTION_ROLE,
                "outcome_open_before_t0": True,
                "fresh_evidence_credit": False,
                "reserved_source": False,
                "canonical_consumed_package": {
                    "root": str(self.canonical.resolve()),
                    "manifest_sha256": _sha256(
                        self.canonical / "manifest.replay.jsonl"
                    ),
                    "dataset_spec_sha256": _sha256(
                        self.canonical / "dataset_spec.json"
                    ),
                },
            },
            "acquisition_config": {
                "session_id": self.session_id,
                "camera": "camera_chest",
                "lens": "left",
                "official_split": "train",
                "start_frame": 0,
                "target_fps": 10.0,
                "frame_count": 2,
            },
            "transport_root": str((self.root / "transport").resolve()),
            "short_path_package": {
                "root": str(self.candidate.resolve()),
                "post_open_local_hashes_are_transport_receipts_only": True,
            },
            "evidence": {
                "filesystem_canary": {
                    "path": str(self.canary_path.resolve()),
                    "sha256": _sha256(self.canary_path),
                },
                "path_preflight": {
                    "path": str(self.preflight_path.resolve()),
                    "sha256": _sha256(self.preflight_path),
                },
            },
            "authorization": {
                "consumed_development_transport_execution": True,
                "fresh_or_reserved_source_open": False,
                "teacher_or_student_execution": False,
            },
        }

    def _make_transport_receipt(self) -> None:
        qa_path = self.candidate / "qa/replay_validation.json"
        _write_json(qa_path, {"ok": True})
        _write_json(
            self.candidate / "qa/transport_receipt.json",
            {
                "schema": subject.TRANSPORT_SCHEMA,
                "terminal": subject.TRANSPORT_READY,
                "acquisition_token": self.acquisition_token,
                "output_root": str(self.candidate.resolve()),
                "session_id": self.session_id,
                "official_split": "train",
                "selected_source_frames": self.source_indices,
                "execution_contract_path": str(self.contract_path.resolve()),
                "execution_contract_sha256": _sha256(self.contract_path),
                "path_preflight": self._path_report(),
                "dataset_spec_sha256": _sha256(
                    self.candidate / "dataset_spec.json"
                ),
                "manifest_sha256": _sha256(
                    self.candidate / "manifest.replay.jsonl"
                ),
                "replay_validation_sha256": _sha256(qa_path),
                "all_downloads_generation_bound": True,
                "all_downloads_size_and_md5_verified": True,
                "authorization": {
                    "pose_geometry_authority_verification_authorized": True,
                    "teacher_label_or_corpus_authorized": False,
                    "student_training_authorized": False,
                    "fresh_or_reserved_evaluation_authorized": False,
                    "research_mainline_changed": False,
                    "default_app_changed": False,
                },
            },
        )


class StageCT0SanpoShortPathEquivalenceTest(unittest.TestCase):
    def test_governance_failure_terminal_is_no_source_replacement(self) -> None:
        self.assertEqual(
            (
                "T0_SANPO_SHORT_PATH_TRANSPORT_NOT_EVALUABLE_"
                "NO_SOURCE_REPLACEMENT"
            ),
            subject.NOT_EVALUABLE,
        )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = EquivalenceFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.fixture.contract_path.unlink(missing_ok=True)
        self.temporary.cleanup()

    def test_exact_consumed_packages_are_equivalent(self) -> None:
        result = subject.validate(
            self.fixture.contract_path, self.fixture.candidate
        )
        self.assertEqual(subject.READY, result["terminal"])
        self.assertEqual(2, result["frame_count"])
        self.assertTrue(all(result["gates"].values()))
        self.assertFalse(result["network_opened"])

    def test_local_content_sha_drift_fails(self) -> None:
        (self.fixture.candidate / "i/train/00.png").write_bytes(b"changed")
        with self.assertRaisesRegex(subject.EquivalenceError, "SHA-256"):
            subject.validate(
                self.fixture.contract_path, self.fixture.candidate
            )

    def test_remote_generation_identity_drift_fails(self) -> None:
        manifest = self.fixture.candidate / "manifest.replay.jsonl"
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
        ]
        rows[0]["modalities"]["rgb"]["generation"] = "8"
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        _write_json(self.fixture.contract_path, self.fixture.contract)
        self.fixture._make_transport_receipt()
        with self.assertRaisesRegex(subject.EquivalenceError, "inventory mismatch"):
            subject.validate(
                self.fixture.contract_path, self.fixture.candidate
            )

    def test_canary_report_hash_drift_fails(self) -> None:
        self.fixture.canary_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(subject.EquivalenceError, "report SHA-256"):
            subject.validate(
                self.fixture.contract_path, self.fixture.candidate
            )

    def test_transport_receipt_path_or_md5_claim_drift_fails(self) -> None:
        receipt_path = (
            self.fixture.candidate / "qa/transport_receipt.json"
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["path_preflight"]["maximum_content_path_length"] = 240
        _write_json(receipt_path, receipt)
        with self.assertRaisesRegex(subject.EquivalenceError, "path report"):
            subject.validate(
                self.fixture.contract_path, self.fixture.candidate
            )
        self.fixture._make_transport_receipt()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["all_downloads_size_and_md5_verified"] = False
        _write_json(receipt_path, receipt)
        with self.assertRaisesRegex(subject.EquivalenceError, "receipt mismatch"):
            subject.validate(
                self.fixture.contract_path, self.fixture.candidate
            )

    def test_validator_implementation_hash_must_be_frozen(self) -> None:
        self.fixture.contract["implementations"]["equivalence_validator"][
            "sha256"
        ] = "0" * 64
        _write_json(self.fixture.contract_path, self.fixture.contract)
        self.fixture._make_transport_receipt()
        with self.assertRaisesRegex(
            subject.EquivalenceError, "implementation SHA-256"
        ):
            subject.validate(
                self.fixture.contract_path, self.fixture.candidate
            )

    def test_candidate_hashes_cannot_be_prefilled_in_contract(self) -> None:
        self.fixture.contract["short_path_package"][
            "manifest_sha256"
        ] = _sha256(self.fixture.candidate / "manifest.replay.jsonl")
        _write_json(self.fixture.contract_path, self.fixture.contract)
        self.fixture._make_transport_receipt()
        with self.assertRaisesRegex(
            subject.EquivalenceError, "freeze only root"
        ):
            subject.validate(
                self.fixture.contract_path, self.fixture.candidate
            )


if __name__ == "__main__":
    unittest.main()
