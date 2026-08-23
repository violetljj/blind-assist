from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import contract as sut


def freeze_bundle(*, modality: str = "REFERENCE_IMAGE_INSTANCE", mode: str = "UNIQUE") -> dict:
    if mode == "UNIQUE":
        ids = ["private-instance-a"]
    elif mode == "SET_VALUED":
        ids = ["private-instance-a", "private-instance-b"]
    else:
        ids = []
    anchors = [
        {
            "physical_instance_id": instance_id,
            "coordinate_frame_id": "scene-native-world",
            "position_xyz_m": [float(index), 1.0, 2.0],
            "authority": "SOURCE_NATIVE_METRIC_ANCHOR",
        }
        for index, instance_id in enumerate(ids)
    ]
    return {
        "schema_version": sut.FREEZE_SCHEMA,
        "protocol_id": sut.PROTOCOL_ID,
        "contract_id": "contract-001",
        "frozen_at_utc": "2026-08-24T12:00:00Z",
        "freeze_states": {
            "episode_observation_pixels": "NOT_CAPTURED",
            "provider_output": "NOT_CREATED",
            "candidate_output": "NOT_CREATED",
            "outcome_access": "NONE",
        },
        "public_goal": {
            "goal_text": "Find the same door shown in this reference image" if modality == "REFERENCE_IMAGE_INSTANCE" else "Find the two equivalent public entry doors",
            "modality": modality,
            "reference_mode": mode,
            "reference_anchor_id": "public-anchor-001",
            "reference_image": (
                {
                    "reference_image_id": "reference-001",
                    "image_sha256": "a" * 64,
                    "width": 640,
                    "height": 480,
                    "target_selector": "PUBLIC_TARGET_REGION",
                    "public_target_region_xyxy": [0.2, 0.1, 0.8, 0.9],
                }
                if modality == "REFERENCE_IMAGE_INSTANCE"
                else None
            ),
            "language_description": "the door with a brass handle" if modality == "REFERENCE_IMAGE_INSTANCE" else "the two equivalent public entry doors",
        },
        "private_binding": {
            "binding_authority": "SOURCE_NATIVE_REFERENCE_LINK",
            "binding_created_at_utc": "2026-08-24T11:59:00Z",
            "binding_created_before_episode_observations": True,
            "binding_created_before_provider_output": True,
            "model_or_teacher_used_for_binding": False,
            "source_record_sha256": "b" * 64,
            "bound_reference_image_sha256": "a" * 64 if modality == "REFERENCE_IMAGE_INSTANCE" else None,
            "legal_physical_instance_ids": ids,
            "world_anchors": anchors,
        },
    }


def truth(public: dict, private: dict, rows: list[dict]) -> dict:
    return {
        "schema_version": sut.TRUTH_SCHEMA,
        "contract_id": public["contract_id"],
        "public_contract_body_sha256": public["body_sha256"],
        "private_identity_lock_body_sha256": private["body_sha256"],
        "provider_access_to_truth": False,
        "truth_created_after_contract_freeze": True,
        "observations": rows,
    }


def row(visibility: str, regions: list[dict] | None = None) -> dict:
    return {
        "observation_id": f"obs-{visibility.lower()}",
        "frame_sha256": "c" * 64,
        "visibility": visibility,
        "target_regions": regions or [],
    }


class PublicIdentifiableReferentContractTest(unittest.TestCase):
    def test_schema_is_machine_readable_and_frozen_to_protocol(self) -> None:
        path = Path(__file__).with_name("public_identifiable_referent_contract_v1.schema.json")
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual("https://json-schema.org/draft/2020-12/schema", schema["$schema"])
        self.assertEqual(sut.PROTOCOL_ID, schema["properties"]["protocol_id"]["const"])

    def test_reference_image_freezes_unique_public_contract_and_private_lock(self) -> None:
        public, private = sut.freeze_contract(freeze_bundle())
        self.assertEqual("REFERENCE_IMAGE_INSTANCE", public["modality"])
        self.assertEqual("UNIQUE", public["reference_mode"])
        self.assertEqual("TARGET_INSTANCE_REFERENCE", public["reference_image"]["role"])
        self.assertFalse(public["provider_may_access_private_identity"])
        self.assertFalse(public["cohort_freeze_authorized"])
        self.assertFalse(public["algorithm_authorized"])
        self.assertNotIn("private-instance-a", json.dumps(public, sort_keys=True))
        self.assertEqual(
            public["private_identity_commitment_sha256"],
            private["private_identity_commitment_sha256"],
        )
        self.assertEqual(public["body_sha256"], private["public_contract_body_sha256"])

    def test_reference_image_cannot_masquerade_as_set_valued(self) -> None:
        with self.assertRaisesRegex(sut.ContractError, "must be UNIQUE"):
            sut.freeze_contract(freeze_bundle(mode="SET_VALUED"))

    def test_contract_must_precede_pixels_candidates_and_provider(self) -> None:
        for field, value in (
            ("episode_observation_pixels", "CAPTURED"),
            ("candidate_output", "CREATED"),
            ("provider_output", "CREATED"),
            ("outcome_access", "READ"),
        ):
            with self.subTest(field=field):
                candidate = freeze_bundle()
                candidate["freeze_states"][field] = value
                with self.assertRaises(sut.ContractError):
                    sut.freeze_contract(candidate)

    def test_model_or_teacher_cannot_create_identity_authority(self) -> None:
        candidate = freeze_bundle()
        candidate["private_binding"]["model_or_teacher_used_for_binding"] = True
        with self.assertRaisesRegex(sut.ContractError, "cannot be identity authority"):
            sut.freeze_contract(candidate)

    def test_private_lock_must_bind_the_exact_public_reference_image(self) -> None:
        candidate = freeze_bundle()
        candidate["private_binding"]["bound_reference_image_sha256"] = "d" * 64
        with self.assertRaisesRegex(sut.ContractError, "not bound to the public reference image"):
            sut.freeze_contract(candidate)

    def test_identity_binding_cannot_postdate_contract_freeze(self) -> None:
        candidate = freeze_bundle()
        candidate["private_binding"]["binding_created_at_utc"] = "2026-08-24T12:01:00Z"
        with self.assertRaisesRegex(sut.ContractError, "cannot postdate"):
            sut.freeze_contract(candidate)

    def test_unique_requires_exactly_one_private_instance_and_world_anchor(self) -> None:
        candidate = freeze_bundle()
        candidate["private_binding"]["legal_physical_instance_ids"].append("private-instance-b")
        with self.assertRaisesRegex(sut.ContractError, "exactly one"):
            sut.freeze_contract(candidate)

    def test_public_target_region_must_be_positive_and_normalized(self) -> None:
        candidate = freeze_bundle()
        candidate["public_goal"]["reference_image"]["public_target_region_xyxy"] = [0.8, 0.1, 0.2, 0.9]
        with self.assertRaisesRegex(sut.ContractError, "positive normalized"):
            sut.freeze_contract(candidate)

    def test_unique_observation_truth_binds_visible_and_not_visible_without_leakage(self) -> None:
        public, private = sut.freeze_contract(freeze_bundle())
        visible = row(
            "VISIBLE",
            [{"physical_instance_id": "private-instance-a", "bbox_xyxy_normalized": [0.1, 0.2, 0.4, 0.8]}],
        )
        not_visible = row("NOT_VISIBLE")
        not_visible["observation_id"] = "obs-not-visible"
        report = sut.validate_observation_truth(public, private, truth(public, private, [visible, not_visible]))
        self.assertEqual(2, report["primary_evaluable_count"])
        self.assertFalse(report["passive_baseline_authorized"])
        self.assertFalse(report["algorithm_authorized"])

    def test_unique_visible_truth_rejects_wrong_or_missing_instance(self) -> None:
        public, private = sut.freeze_contract(freeze_bundle())
        wrong = row(
            "VISIBLE",
            [{"physical_instance_id": "private-instance-b", "bbox_xyxy_normalized": [0.1, 0.2, 0.4, 0.8]}],
        )
        with self.assertRaisesRegex(sut.ContractError, "illegal"):
            sut.validate_observation_truth(public, private, truth(public, private, [wrong]))
        with self.assertRaisesRegex(sut.ContractError, "VISIBLE requires"):
            sut.validate_observation_truth(public, private, truth(public, private, [row("VISIBLE")]))

    def test_set_valued_language_contract_preserves_legal_set(self) -> None:
        public, private = sut.freeze_contract(
            freeze_bundle(modality="LANGUAGE_REFERRING_EXPRESSION", mode="SET_VALUED")
        )
        self.assertEqual("SET_VALUED", public["reference_mode"])
        self.assertIsNone(public["reference_image"])
        visible = row(
            "VISIBLE",
            [{"physical_instance_id": "private-instance-b", "bbox_xyxy_normalized": [0.2, 0.2, 0.5, 0.9]}],
        )
        report = sut.validate_observation_truth(public, private, truth(public, private, [visible]))
        self.assertEqual(1, report["primary_evaluable_count"])

    def test_ambiguous_contract_cannot_fabricate_a_target(self) -> None:
        public, private = sut.freeze_contract(
            freeze_bundle(modality="LANGUAGE_REFERRING_EXPRESSION", mode="AMBIGUOUS")
        )
        report = sut.validate_observation_truth(public, private, truth(public, private, [row("UNKNOWN")]))
        self.assertEqual(0, report["primary_evaluable_count"])
        fabricated = row(
            "VISIBLE",
            [{"physical_instance_id": "private-instance-a", "bbox_xyxy_normalized": [0.1, 0.1, 0.2, 0.2]}],
        )
        with self.assertRaises(sut.ContractError):
            sut.validate_observation_truth(public, private, truth(public, private, [fabricated]))

    def test_hash_binding_fails_closed(self) -> None:
        public, private = sut.freeze_contract(freeze_bundle())
        tampered = copy.deepcopy(private)
        tampered["private_binding"]["legal_physical_instance_ids"] = ["private-instance-x"]
        with self.assertRaisesRegex(sut.ContractError, "body hash mismatch"):
            sut.validate_observation_truth(public, tampered, truth(public, private, [row("NOT_VISIBLE")]))


if __name__ == "__main__":
    unittest.main()
