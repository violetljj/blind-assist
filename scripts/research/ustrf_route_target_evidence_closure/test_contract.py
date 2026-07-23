from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).with_name("contract.py")
SPEC = importlib.util.spec_from_file_location("route_target_contract", MODULE_PATH)
contract = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(contract)

FUSION_PATH = Path(__file__).with_name("fuse_seen_person_proposals.py")
FUSION_SPEC = importlib.util.spec_from_file_location("route_target_fusion", FUSION_PATH)
fusion = importlib.util.module_from_spec(FUSION_SPEC)
assert FUSION_SPEC.loader is not None
FUSION_SPEC.loader.exec_module(fusion)

CANDIDATE_SPEC = importlib.util.spec_from_file_location("route_candidates", Path(__file__).with_name("candidates.py"))
assert CANDIDATE_SPEC and CANDIDATE_SPEC.loader
candidates = importlib.util.module_from_spec(CANDIDATE_SPEC)
CANDIDATE_SPEC.loader.exec_module(candidates)

ROUTE_SPEC = importlib.util.spec_from_file_location(
    "crowdbot_causal_route",
    Path(__file__).with_name("build_crowdbot_causal_route_ledger.py"),
)
assert ROUTE_SPEC and ROUTE_SPEC.loader
causal_route = importlib.util.module_from_spec(ROUTE_SPEC)
ROUTE_SPEC.loader.exec_module(causal_route)

ROLE_PROPOSAL_SPEC = importlib.util.spec_from_file_location(
    "crowdbot_role_proposal",
    Path(__file__).with_name("build_crowdbot_projected_track_role_proposal.py"),
)
assert ROLE_PROPOSAL_SPEC and ROLE_PROPOSAL_SPEC.loader
role_proposal = importlib.util.module_from_spec(ROLE_PROPOSAL_SPEC)
ROLE_PROPOSAL_SPEC.loader.exec_module(role_proposal)

FUSION_TRUTH_SPEC = importlib.util.spec_from_file_location(
    "crowdbot_truth_fusion",
    Path(__file__).with_name("fuse_crowdbot_holdout_person_role_truth.py"),
)
assert FUSION_TRUTH_SPEC and FUSION_TRUTH_SPEC.loader
truth_fusion = importlib.util.module_from_spec(FUSION_TRUTH_SPEC)
FUSION_TRUTH_SPEC.loader.exec_module(truth_fusion)

WINDOW_SPEC = importlib.util.spec_from_file_location(
    "crowdbot_truth_windows",
    Path(__file__).with_name("freeze_crowdbot_holdout_truth_windows.py"),
)
assert WINDOW_SPEC and WINDOW_SPEC.loader
truth_windows = importlib.util.module_from_spec(WINDOW_SPEC)
WINDOW_SPEC.loader.exec_module(truth_windows)

CANDIDATE_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "crowdbot_candidate_runner",
    Path(__file__).with_name("run_crowdbot_holdout_candidates.py"),
)
assert CANDIDATE_RUNNER_SPEC and CANDIDATE_RUNNER_SPEC.loader
candidate_runner = importlib.util.module_from_spec(CANDIDATE_RUNNER_SPEC)
CANDIDATE_RUNNER_SPEC.loader.exec_module(candidate_runner)

MATERIALIZE_SPEC = importlib.util.spec_from_file_location(
    "crowdbot_materialize_sequence",
    Path(__file__).with_name("materialize_crowdbot_rgbd_sequence.py"),
)
assert MATERIALIZE_SPEC and MATERIALIZE_SPEC.loader
materialize_sequence = importlib.util.module_from_spec(MATERIALIZE_SPEC)
MATERIALIZE_SPEC.loader.exec_module(materialize_sequence)

AUDIT_SPEC = importlib.util.spec_from_file_location(
    "crowdbot_holdout_audit",
    Path(__file__).with_name("audit_materialized_holdout.py"),
)
assert AUDIT_SPEC and AUDIT_SPEC.loader
holdout_audit = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(holdout_audit)


class RouteTargetContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.config_path = cls.repo / "configs/ustrf_route_target_evidence_closure_r1.json"
        cls.config = json.loads(cls.config_path.read_text(encoding="utf-8"))

    def test_live_preregistration_passes(self) -> None:
        result = contract.validate_prereg(self.config, repo=self.repo)
        self.assertEqual(len(result["oracle_arms"]), 3)
        self.assertEqual(len(result["candidate_roster"]), 3)

    def test_fourth_candidate_fails_closed(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["candidate_roster"].append(copy.deepcopy(changed["candidate_roster"][-1]))
        with self.assertRaisesRegex(contract.ContractError, "one to three"):
            contract.validate_prereg(changed, repo=self.repo)

    def test_open_holdout_state_fails_before_materialization(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["sealed_holdout"]["state"] = "opened"
        with self.assertRaisesRegex(contract.ContractError, "sealed_holdout.state"):
            contract.validate_prereg(changed, repo=self.repo)

    def test_preadmission_candidate_output_leak_fails_closed(self) -> None:
        changed = copy.deepcopy(self.config)
        binding = changed["sealed_holdout"]["source_preadmission_receipt"]
        payload = json.loads((self.repo / binding["path"]).read_text(encoding="utf-8"))
        payload["candidate_outputs_executed_on_sources"] = True
        artifact_root = self.repo / "artifacts.local"
        with tempfile.TemporaryDirectory(dir=artifact_root) as temporary:
            receipt = Path(temporary) / "leaked-preadmission.json"
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            binding["path"] = receipt.relative_to(self.repo).as_posix()
            binding["sha256"] = contract.sha256_file(receipt)
            with self.assertRaisesRegex(contract.ContractError, "cannot inspect candidate outputs"):
                contract.validate_prereg(changed, repo=self.repo)

    def test_threshold_drift_fails_closed(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["frozen_axes"]["person_confidence_threshold"] = 0.34
        with self.assertRaisesRegex(contract.ContractError, "thresholds drifted"):
            contract.validate_prereg(changed, repo=self.repo)

    def test_holdout_causal_route_cannot_access_future_pose(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["sealed_holdout"]["causal_route_input_contract"]["future_pose_access_for_candidate"] = True
        with self.assertRaisesRegex(contract.ContractError, "causal route input contract drifted"):
            contract.validate_prereg(changed, repo=self.repo)

    def test_holdout_candidate_output_cannot_resolve_truth(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["sealed_holdout"]["holdout_truth_freeze_contract"][
            "candidate_output_used_to_resolve_truth_or_quarantine"
        ] = True
        with self.assertRaisesRegex(contract.ContractError, "candidate output can influence truth"):
            contract.validate_prereg(changed, repo=self.repo)

    def test_illegal_role_transition_fails_closed(self) -> None:
        def frame(frame_id: str, role: str) -> dict:
            return {
                "frame_id": frame_id,
                "visibility": "visible",
                "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
                "source_capture_timestamp_ns": int(frame_id) * 1_000_000,
                "route_status": "known",
                "route_receipt_id": "route-r1",
                "route_evidence_age_ms": 0.0,
                "role": role,
            }
        payload = {
            "schema": contract.ROLE_TRUTH_SCHEMA,
            "split": "seen_diagnostic_only",
            "roles": list(contract.ROLES),
            "sources": [
                {"source_id": "lilocbench_dynamics_0_front", "person_episodes": [{
                    "person_id": "person-1",
                    "risk_event_id": "event-1",
                    "event_truth": {
                        "first_visible_frame": "000001",
                        "alertable_start_frame": "000001",
                        "clear_frame": "000002",
                        "should_alert": True,
                        "critical": False
                    },
                    "frames": [frame("000001", "cleared"), frame("000002", "route_intersecting")]
                }]},
                {"source_id": "lilocbench_lt_changes_dynamics_0_front", "person_episodes": []}
            ]
        }
        with self.assertRaisesRegex(contract.ContractError, "illegal role transition"):
            contract.validate_role_truth(payload, config=self.config)

    def test_parent_hash_drift_fails_closed(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["parent_bindings"]["association_only"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(contract.ContractError, "hash mismatch"):
            contract.validate_prereg(changed, repo=self.repo)

    def test_dual_model_fusion_is_one_to_one(self) -> None:
        first = [
            {"bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
            {"bbox_xyxy": [20.0, 0.0, 30.0, 10.0]},
        ]
        second = [
            {"bbox_xyxy": [1.0, 0.0, 11.0, 10.0]},
            {"bbox_xyxy": [19.0, 0.0, 29.0, 10.0]},
        ]
        matches = fusion.greedy_matches(first, second, 0.30)
        self.assertEqual({(left, right) for left, right, _ in matches}, {(0, 0), (1, 1)})

    def test_single_model_node_requires_adjudication(self) -> None:
        frames = [{
            "source_id": "source",
            "blind_window_id": "blind",
            "frame_id": "000001",
            "image_sha256": "a" * 64,
            "person_nodes": [{
                "frame_node_id": "pass_a_only_000",
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                "evidence": "pass_a_only",
                "person_identity_hint": None,
            }],
        }]
        tracklets, _ = fusion.build_tracklets(frames)
        self.assertEqual(tracklets[0]["status"], "third_model_adjudication_required")

    def test_identity_lineage_never_crosses_blind_windows(self) -> None:
        frames = []
        for index, blind_window_id in enumerate(("blind_a", "blind_b")):
            frames.append({
                "source_id": "source",
                "blind_window_id": blind_window_id,
                "frame_id": f"{index:06d}",
                "image_sha256": str(index) * 64,
                "person_nodes": [{
                    "frame_node_id": "seed_000",
                    "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                    "evidence": "frozen_seed_truth",
                    "person_identity_hint": "reused_hint",
                    "pass_a_track_id": "reused_pass_a_track",
                    "pass_b_track_id": "reused_pass_b_track",
                }],
            })
        tracklets, _ = fusion.build_tracklets(frames)
        self.assertEqual(len(tracklets), 2)
        self.assertEqual({tracklet["blind_window_id"] for tracklet in tracklets}, {"blind_a", "blind_b"})

    def test_c2_delivers_once_across_lineage_fragmentation(self) -> None:
        fsm = candidates.C2RouteOccupancyEpisodeFSM(min_alert=2, min_clear=3)
        deliveries = []
        deliveries += fsm.update(1, True, {1: "route_intersecting"})["deliveries"]
        deliveries += fsm.update(2, True, {1: "route_intersecting"})["deliveries"]
        deliveries += fsm.update(3, True, {2: "route_intersecting"})["deliveries"]
        deliveries += fsm.update(4, True, {2: "route_intersecting"})["deliveries"]
        self.assertEqual(deliveries, [1])

    def test_c3_missing_lineage_cannot_clear(self) -> None:
        fsm = candidates.C3DualKeyClearanceFSM(min_alert=2, min_clear=3)
        fsm.update(1, True, {1: "route_intersecting"})
        fsm.update(2, True, {1: "route_intersecting"})
        for frame in range(3, 8):
            result = fsm.update(frame, True, {})
        self.assertTrue(fsm.active)
        self.assertEqual(result["closures"], [])

    def test_crowdbot_static_tf_path_maps_camera_into_qolo(self) -> None:
        sample = {
            "timestamp_ns": 1,
            "translation_xyz": [1.0, 2.0, 3.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        inventory = {"frame_pairs": [{
            "parent_frame": "tf_qolo",
            "child_frame": "camera_color_optical_frame",
            "message_count": 2,
            "first": sample,
            "last": sample,
        }]}
        matrix, path = causal_route.find_static_transform(
            inventory,
            target="tf_qolo",
            source="camera_color_optical_frame",
        )
        self.assertEqual(path, ["camera_color_optical_frame", "tf_qolo"])
        self.assertEqual(matrix[:3, 3].tolist(), [1.0, 2.0, 3.0])

    def test_crowdbot_dynamic_tf_edge_cannot_supply_camera_extrinsic(self) -> None:
        inventory = {"frame_pairs": [{
            "parent_frame": "tf_qolo",
            "child_frame": "camera_color_optical_frame",
            "message_count": 2,
            "first": {
                "timestamp_ns": 1,
                "translation_xyz": [0.0, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "last": {
                "timestamp_ns": 2,
                "translation_xyz": [0.1, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }]}
        with self.assertRaisesRegex(RuntimeError, "no static TF path"):
            causal_route.find_static_transform(
                inventory,
                target="tf_qolo",
                source="camera_color_optical_frame",
            )

    def test_projected_role_event_is_critical_and_clears_once(self) -> None:
        roles = ["route_intersecting"] * 3 + ["receding"] * 3
        frames = [
            {"projected_tracks": [{"published_track_id": 7, "role_proposal": role}]}
            for role in roles
        ]
        events = role_proposal.derive_event_proposals("sequence", frames)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["critical"])
        self.assertEqual(events[0]["clear_frame"], 5)
        self.assertEqual(frames[5]["projected_tracks"][0]["role_proposal"], "cleared")

    def test_projected_role_missing_gap_cannot_clear(self) -> None:
        frames = [
            {"projected_tracks": [{"published_track_id": 7, "role_proposal": "route_intersecting"}]},
            {"projected_tracks": [{"published_track_id": 7, "role_proposal": "route_intersecting"}]},
            {"projected_tracks": [{"published_track_id": 7, "role_proposal": "receding"}]},
            {"projected_tracks": []},
            {"projected_tracks": [{"published_track_id": 7, "role_proposal": "receding"}]},
            {"projected_tracks": [{"published_track_id": 7, "role_proposal": "receding"}]},
        ]
        events = role_proposal.derive_event_proposals("sequence", frames)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["censored_without_clear"])

    def test_holdout_truth_fusion_accepts_two_visual_signals_with_unique_track(self) -> None:
        common = {
            "source_id": "source",
            "sequence_id": "sequence",
            "frame_id": "000001",
            "source_capture_timestamp_ns": 1,
            "image_sha256": "a" * 64,
        }
        left = {**common, "person_proposals": [{"bbox_xyxy": [0.0, 0.0, 10.0, 20.0]}]}
        right = {**common, "person_proposals": [{"bbox_xyxy": [1.0, 0.0, 11.0, 20.0]}]}
        projected = {
            **common,
            "status": "known",
            "projected_tracks": [{
                "published_track_id": 9,
                "projected_track_center_uv": [5.0, 10.0],
                "projected_ground_uv": [5.0, 19.0],
                "route_distance_m": 0.1,
                "future_track_route_distance_m": 0.0,
                "role_proposal": "route_intersecting",
                "event_id": "event-1",
            }],
        }
        fused = truth_fusion.fuse_frame(left, right, projected, 0.3)
        self.assertTrue(fused["full_person_role_complete"])
        self.assertEqual(fused["persons"][0]["published_track_id"], 9)

    def test_holdout_truth_fusion_quarantines_single_visual_without_track(self) -> None:
        common = {
            "source_id": "source",
            "sequence_id": "sequence",
            "frame_id": "000001",
            "source_capture_timestamp_ns": 1,
            "image_sha256": "a" * 64,
        }
        left = {**common, "person_proposals": [{"bbox_xyxy": [0.0, 0.0, 10.0, 20.0]}]}
        right = {**common, "person_proposals": []}
        projected = {**common, "status": "known", "projected_tracks": []}
        fused = truth_fusion.fuse_frame(left, right, projected, 0.3)
        self.assertFalse(fused["full_person_role_complete"])
        self.assertEqual(len(fused["quarantined_visual_candidates"]), 1)

    def test_holdout_truth_fusion_merges_role_consistent_projected_track_aliases(self) -> None:
        common = {
            "source_id": "source",
            "sequence_id": "sequence",
            "frame_id": "000001",
            "source_capture_timestamp_ns": 1,
            "image_sha256": "a" * 64,
        }
        left = {**common, "person_proposals": [{"bbox_xyxy": [0.0, 0.0, 20.0, 30.0]}]}
        right = {**common, "person_proposals": [{"bbox_xyxy": [0.0, 0.0, 20.0, 30.0]}]}
        projected_tracks = []
        for track_id, u in ((7, 8.0), (9, 12.0)):
            projected_tracks.append({
                "published_track_id": track_id,
                "projected_track_center_uv": [u, 15.0],
                "projected_ground_uv": [u, 29.0],
                "route_distance_m": 1.0,
                "future_track_route_distance_m": 1.0,
                "role_proposal": "adjacent_safe",
                "event_id": None,
            })
        fused = truth_fusion.fuse_frame(
            left,
            right,
            {**common, "status": "known", "projected_tracks": projected_tracks},
            0.3,
        )
        self.assertTrue(fused["full_person_role_complete"])
        self.assertEqual(fused["persons"][0]["published_track_alias_ids"], [7, 9])

    def test_visual_consensus_outside_truth_route_gets_adjacent_safe_role(self) -> None:
        common = {
            "source_id": "source",
            "sequence_id": "sequence",
            "frame_id": "000001",
            "source_capture_timestamp_ns": 1,
            "image_sha256": "a" * 64,
        }
        proposal = {"bbox_xyxy": [0.0, 0.0, 20.0, 30.0], "proposal_confidence": 0.9}
        fused = truth_fusion.fuse_frame(
            {**common, "person_proposals": [proposal]},
            {**common, "person_proposals": [proposal]},
            {**common, "status": "known", "projected_tracks": []},
            0.3,
            0.35,
            {"status": "known", "uv": [500.0, 400.0]},
            0.08,
        )
        self.assertTrue(fused["full_person_role_complete"])
        self.assertEqual(fused["persons"][0]["role"], "adjacent_safe")

    def test_visual_truth_uses_future_route_polyline_not_only_endpoint(self) -> None:
        box = [100.0, 100.0, 140.0, 220.0]
        route = {
            "status": "known",
            "uv": [500.0, 400.0],
            "uv_polyline": [[80.0, 210.0], [160.0, 210.0], [500.0, 400.0]],
        }
        self.assertTrue(truth_fusion.truth_route_hit(box, route, 0.0))

    def test_visual_truth_polyline_outside_box_is_safe(self) -> None:
        box = [100.0, 100.0, 140.0, 220.0]
        route = {
            "status": "known",
            "uv": [500.0, 400.0],
            "uv_polyline": [[200.0, 50.0], [300.0, 60.0], [500.0, 400.0]],
        }
        self.assertFalse(truth_fusion.truth_route_hit(box, route, 0.0))

    def test_single_visual_with_stable_annotation_track_is_evaluable(self) -> None:
        common = {
            "source_id": "source",
            "sequence_id": "sequence",
            "frame_id": "000001",
            "source_capture_timestamp_ns": 1,
            "image_sha256": "a" * 64,
        }
        candidate = {
            "bbox_xyxy": [0.0, 0.0, 20.0, 30.0],
            "visual_signals": ["pass_b"],
            "visual_confidences": [0.9],
            "proposal_confidence": 0.9,
            "visual_iou": None,
            "proposal_track_id": "visual-track-1",
        }
        fused = truth_fusion.fuse_frame(
            {**common, "person_proposals": []},
            {**common, "person_proposals": []},
            {**common, "status": "known", "projected_tracks": []},
            0.3,
            0.35,
            {"status": "known", "uv": [500.0, 400.0]},
            0.08,
            [candidate],
            {"visual-track-1"},
        )
        self.assertTrue(fused["full_person_role_complete"])
        self.assertEqual(fused["persons"][0]["person_id"], "visual-track-1")

    def test_visual_route_event_clears_only_after_three_adjacent_frames(self) -> None:
        frames = []
        for index, role in enumerate(["route_intersecting"] * 3 + ["adjacent_safe"] * 3):
            frames.append({
                "source_id": "source",
                "sequence_id": "sequence",
                "frame_id": f"{index:06d}",
                "persons": [{
                    "person_id": "visual-track-1",
                    "published_track_id": None,
                    "visual_proposal_track_id": "visual-track-1",
                    "role": role,
                    "event_id": None,
                }],
            })
        events = truth_fusion.derive_visual_event_proposals(frames)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["critical"])
        self.assertEqual(events[0]["clear_frame"], 5)
        self.assertEqual(frames[5]["persons"][0]["role"], "cleared")

    def test_visible_metric_event_starts_from_camera_confirmed_roles(self) -> None:
        frames = []
        for index, role in enumerate(
            ["route_intersecting", "route_intersecting", "route_intersecting", "receding", "cleared"]
        ):
            frames.append({
                "source_id": "source",
                "sequence_id": "sequence",
                "frame_id": f"{index:06d}",
                "persons": [{
                    "person_id": "published-track-7",
                    "published_track_id": 7,
                    "role": role,
                    "event_id": "raw-lidar-event-must-be-replaced",
                }],
            })
        events = truth_fusion.derive_visible_metric_event_proposals(frames)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["onset_frame"], 0)
        self.assertEqual(events[0]["alertable_start_frame"], 1)
        self.assertEqual(events[0]["clear_frame"], 4)
        self.assertTrue(events[0]["critical"])
        self.assertTrue(events[0]["identity_continuous"])
        self.assertNotEqual(frames[0]["persons"][0]["event_id"], "raw-lidar-event-must-be-replaced")

    def test_route_external_quarantined_visual_does_not_invalidate_negative_frame(self) -> None:
        common = {
            "source_id": "source",
            "sequence_id": "sequence",
            "frame_id": "000001",
            "source_capture_timestamp_ns": 1,
            "image_sha256": "a" * 64,
        }
        candidate = {
            "bbox_xyxy": [0.0, 0.0, 20.0, 30.0],
            "visual_signals": ["pass_b"],
            "visual_confidences": [0.9],
            "proposal_confidence": 0.9,
            "visual_iou": None,
            "proposal_track_id": None,
        }
        fused = truth_fusion.fuse_frame(
            {**common, "person_proposals": []},
            {**common, "person_proposals": []},
            {**common, "status": "known", "projected_tracks": []},
            0.3,
            0.35,
            {"status": "known", "uv": [500.0, 400.0]},
            0.08,
            [candidate],
            set(),
        )
        self.assertFalse(fused["full_person_role_complete"])
        self.assertTrue(fused["route_relevant_person_truth_complete"])
        self.assertFalse(fused["quarantined_visual_candidates"][0]["truth_route_hit"])

    def test_route_intersecting_quarantined_visual_invalidates_negative_frame(self) -> None:
        common = {
            "source_id": "source",
            "sequence_id": "sequence",
            "frame_id": "000001",
            "source_capture_timestamp_ns": 1,
            "image_sha256": "a" * 64,
        }
        candidate = {
            "bbox_xyxy": [0.0, 0.0, 20.0, 30.0],
            "visual_signals": ["pass_b"],
            "visual_confidences": [0.9],
            "proposal_confidence": 0.9,
            "visual_iou": None,
            "proposal_track_id": None,
        }
        fused = truth_fusion.fuse_frame(
            {**common, "person_proposals": []},
            {**common, "person_proposals": []},
            {**common, "status": "known", "projected_tracks": []},
            0.3,
            0.35,
            {"status": "known", "uv": [10.0, 20.0]},
            0.0,
            [candidate],
            set(),
        )
        self.assertFalse(fused["route_relevant_person_truth_complete"])
        self.assertTrue(fused["quarantined_visual_candidates"][0]["truth_route_hit"])

    def test_truth_window_negative_runs_do_not_bridge_unknown_frames(self) -> None:
        self.assertEqual(
            truth_windows.consecutive_true_runs([True, True, False, True, False, True, True, True]),
            [(0, 1), (3, 3), (5, 7)],
        )

    def test_candidate_truth_attribution_is_unique(self) -> None:
        matches = candidate_runner.truth_matches(
            {1: [0.0, 0.0, 20.0, 20.0], 2: [2.0, 0.0, 22.0, 20.0]},
            [{"person_id": "p", "bbox_xyxy": [0.0, 0.0, 20.0, 20.0]}],
            0.3,
        )
        self.assertEqual(len(matches), 1)
        self.assertIn(1, matches)

    def test_candidate_winner_uses_metric_order_before_id(self) -> None:
        def row(candidate_id: str, recall: float) -> dict:
            metrics = {
                "critical_miss": 0,
                "event_recall": recall,
                "false_alerts_per_minute": 0.0,
                "clearance_rate": 1.0,
                "clearance_p95_ms": 0.0,
                "repeat_alert_count": 0,
                "evidence_age_p95_ms": 0.0,
            }
            return {"candidate_id": candidate_id, "sources": [{"metrics": metrics}, {"metrics": metrics}]}
        self.assertLess(
            candidate_runner.winner_sort_key(row("C3", 1.0)),
            candidate_runner.winner_sort_key(row("C1", 0.9)),
        )

    def test_candidate_unknown_person_alert_is_hard_source_failure(self) -> None:
        truth = {
            "accepted_events": [],
            "sources": [{"source_id": "source", "scorable_negative_exposure_minutes": 1.0}],
        }
        runs = {
            "sequence": {
                "deliveries": [{
                    "frame": 1,
                    "delivery_key": 7,
                    "truth_event_ids": [],
                    "matched_person_ids": [],
                    "unknown_person_overlap": True,
                    "unknown_person_track_ids": [7],
                    "route_known": True,
                    "window_types": ["negative"],
                }],
                "closures": [],
                "evidence_ages_ms": [10.0],
                "active_alert_on_unknown_route_frame_count": 0,
            }
        }
        gate = {
            "event_recall_min": 1.0,
            "critical_miss_max": 0,
            "false_alerts_per_minute_max": 0.0,
            "clearance_rate_min": 1.0,
            "clearance_p95_ms_max": 1000.0,
            "repeat_alert_count_max": 0,
            "event_regeneration_count_max": 0,
            "evidence_age_p95_ms_max": 200.0,
            "active_alert_on_unknown_route_frames_max": 0,
        }
        scored = candidate_runner.score_source("source", truth, runs, gate)
        self.assertEqual(scored["metrics"]["unknown_person_active_alert_count"], 1)
        self.assertFalse(scored["gate_checks"]["unknown_person_active_alert_count"])
        self.assertFalse(scored["gate_pass"])

    def test_bgr8_source_is_losslessly_normalized_to_rgb(self) -> None:
        bgr_pixel = bytes([10, 20, 30])
        message = SimpleNamespace(
            encoding="bgr8",
            width=640,
            height=480,
            step=1920,
            data=bgr_pixel * (640 * 480),
        )
        rgb = materialize_sequence.decode_rgb_message(message)
        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertEqual(rgb[0, 0].tolist(), [30, 20, 10])

    def test_false_alert_rate_counts_full_sequence_route_known_delivery(self) -> None:
        truth = {
            "accepted_events": [],
            "sources": [{"source_id": "source", "scorable_negative_exposure_minutes": 2.0}],
        }
        runs = {
            "sequence": {
                "deliveries": [{
                    "frame": 100,
                    "delivery_key": 7,
                    "truth_event_ids": [],
                    "matched_person_ids": [],
                    "unknown_person_overlap": False,
                    "unknown_person_track_ids": [],
                    "route_known": True,
                    "window_types": [],
                }],
                "closures": [],
                "evidence_ages_ms": [10.0],
                "active_alert_on_unknown_route_frame_count": 0,
            }
        }
        gate = {
            "event_recall_min": 1.0,
            "critical_miss_max": 0,
            "false_alerts_per_minute_max": 1.0,
            "clearance_rate_min": 1.0,
            "clearance_p95_ms_max": 1000.0,
            "repeat_alert_count_max": 0,
            "event_regeneration_count_max": 0,
            "evidence_age_p95_ms_max": 200.0,
            "active_alert_on_unknown_route_frames_max": 0,
        }
        scored = candidate_runner.score_source("source", truth, runs, gate)
        self.assertEqual(scored["false_alert_count"], 1)
        self.assertEqual(scored["metrics"]["false_alerts_per_minute"], 0.5)

    def test_replacement_audit_accepts_per_source_sequence_counts(self) -> None:
        self.assertEqual(
            holdout_audit.expected_sequence_counts(
                ["crowdbot_0410_mds", "crowdbot_1203_shared_control"],
                ["crowdbot_0410_mds=10", "crowdbot_1203_shared_control=13"],
                8,
            ),
            {"crowdbot_0410_mds": 10, "crowdbot_1203_shared_control": 13},
        )


if __name__ == "__main__":
    unittest.main()
