from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import content_sha256
from scripts.research.goal_copilot_bridge.p1_prospective_capture.materialize_capture import (
    CAPTURE_INSTRUCTION,
    CAPTURE_RECEIPT_SCHEMA,
    FRAME_OFFSETS_FROM_END_SECONDS,
    ProspectiveCaptureError,
    SOURCE_ROLE,
    _default_extract,
    _default_probe,
    materialize_capture,
)
from scripts.research.goal_copilot_bridge.p1_prospective_capture.arm_capture import arm_capture


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashed(value: dict, key: str) -> dict:
    value[key] = content_sha256(value)
    return value


class ProspectiveCaptureMaterializationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.capture_root = self.root / "captures"
        self.capture_root.mkdir()
        base = datetime(2026, 8, 22, 8, 0, tzinfo=timezone.utc)
        episodes = []
        captures = []
        for index in range(5):
            episode_id = f"prospective-{index + 1:02d}"
            media = self.capture_root / f"{episode_id}.mp4"
            media.write_bytes(f"physical-video-{index}".encode())
            goal_at = base + timedelta(seconds=index)
            episodes.append({
                "episode_id": episode_id,
                "goal_text_original": "帮我找入口",
                "canonical_prompt": "building entrance",
                "goal_contract": {"goal_type": "NAMED_BUILDING_ENTRANCE", "reference_mode": "SET_VALUED", "task_semantics": "find the public entrance"},
                "goal_provenance": {"goal_recorded_at_utc": goal_at.isoformat(), "source_authority": "PRODUCT_TASK_INPUT", "source_record_sha256": "a" * 64, "capture_state_at_goal_recording": "NOT_STARTED", "truth_state_at_goal_recording": "NOT_CREATED"},
            })
            captures.append({
                "episode_id": episode_id,
                "capture_started_at_utc": (base + timedelta(seconds=60 + index * 15)).isoformat(),
                "capture_completed_at_utc": (base + timedelta(seconds=70 + index * 15)).isoformat(),
                "media_path": media.name,
                "media_sha256": file_sha(media),
                "width": 1280,
                "height": 720,
                "duration_seconds": 10.0,
                "camera_view": "FIRST_PERSON_FORWARD",
                "continuous_capture": True,
            })
        self.c0 = hashed({
            "schema_version": "blindassist_p1_pa3_c0_public_goal_cohort_v1",
            "private_truth_access": False,
            "pa3_inference_authorized": False,
            "episodes": episodes,
        }, "receipt_body_sha256")
        self.plan = arm_capture(self.c0, (base + timedelta(seconds=30)).isoformat())
        self.receipt = hashed({
            "schema_version": CAPTURE_RECEIPT_SCHEMA,
            "goal_receipt_body_sha256": self.c0["receipt_body_sha256"],
            "capture_plan_body_sha256": self.plan["capture_plan_body_sha256"],
            "source_role": SOURCE_ROLE,
            "recorder_authority": "DEVICE_OWNED_CONTINUOUS_VIDEO_RECORDER",
            "capture_instruction_id": CAPTURE_INSTRUCTION,
            "truth_state_at_capture": "NOT_CREATED",
            "provider_state_at_capture": "NOT_STARTED",
            "continuous_capture_required": True,
            "frame_selection_rule": "FIXED_OFFSETS_FROM_VIDEO_END_NO_PIXEL_OR_TRUTH_SELECTION",
            "frame_offsets_from_end_seconds": list(FRAME_OFFSETS_FROM_END_SECONDS),
            "receipt_created_at_utc": (base + timedelta(minutes=3)).isoformat(),
            "episode_count": len(captures),
            "episodes": captures,
        }, "capture_body_sha256")

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def probe(_: Path) -> dict:
        return {"width": 1280, "height": 720, "duration_seconds": 10.0}

    @staticmethod
    def extract(_: Path, timestamps, outputs):
        for output in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"deterministic-jpeg")
        return list(timestamps)

    def run_materialize(self, receipt=None, c0=None, capture_plan=None, output="out") -> Path:
        return materialize_capture(c0=c0 or self.c0, capture_plan=capture_plan or self.plan, capture_receipt=receipt or self.receipt,
                                   capture_root=self.capture_root, output_dir=self.root / output,
                                   probe_media=self.probe, extract_frames=self.extract)

    def test_materializes_full_roster_with_fixed_frames_and_zero_provider_calls(self) -> None:
        path = self.run_materialize()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(15, manifest["case_count"])
        self.assertEqual(5, manifest["episode_count"])
        self.assertEqual(0, manifest["provider_model_calls"])
        self.assertFalse(manifest["private_truth_access"])
        self.assertEqual([7.5, 8.5, 9.5], [row["source_video_timestamp_seconds"] for row in manifest["cases"][:3]])

    def test_rejects_capture_before_goal(self) -> None:
        receipt = deepcopy(self.receipt)
        receipt.pop("capture_body_sha256")
        receipt["episodes"][0]["capture_started_at_utc"] = "2026-08-22T07:59:59+00:00"
        hashed(receipt, "capture_body_sha256")
        with self.assertRaisesRegex(ProspectiveCaptureError, "precedence failed"):
            self.run_materialize(receipt=receipt)

    def test_rejects_truth_or_outcome_fields(self) -> None:
        receipt = deepcopy(self.receipt)
        receipt.pop("capture_body_sha256")
        receipt["episodes"][0]["selected_frame"] = 9
        hashed(receipt, "capture_body_sha256")
        with self.assertRaisesRegex(ProspectiveCaptureError, "truth/outcome-derived"):
            self.run_materialize(receipt=receipt)

    def test_rejects_partial_roster(self) -> None:
        receipt = deepcopy(self.receipt)
        receipt.pop("capture_body_sha256")
        receipt["episodes"].pop()
        receipt["episode_count"] = len(receipt["episodes"])
        hashed(receipt, "capture_body_sha256")
        with self.assertRaisesRegex(ProspectiveCaptureError, "entire frozen C0 roster"):
            self.run_materialize(receipt=receipt)

    def test_rejects_media_outside_capture_root(self) -> None:
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"outside")
        plan = deepcopy(self.plan)
        plan.pop("capture_plan_body_sha256")
        plan["episodes"][0]["media_relative_path"] = str(outside)
        hashed(plan, "capture_plan_body_sha256")
        receipt = deepcopy(self.receipt)
        receipt.pop("capture_body_sha256")
        receipt["capture_plan_body_sha256"] = plan["capture_plan_body_sha256"]
        receipt["episodes"][0]["media_path"] = str(outside)
        receipt["episodes"][0]["media_sha256"] = file_sha(outside)
        hashed(receipt, "capture_body_sha256")
        with self.assertRaisesRegex(ProspectiveCaptureError, "escapes capture root"):
            self.run_materialize(receipt=receipt, capture_plan=plan)

    def test_rejects_short_capture(self) -> None:
        with self.assertRaisesRegex(ProspectiveCaptureError, "duration outside frozen range"):
            materialize_capture(c0=self.c0, capture_plan=self.plan, capture_receipt=self.receipt, capture_root=self.capture_root,
                                output_dir=self.root / "short", probe_media=lambda _: {"width": 1280, "height": 720, "duration_seconds": 2.0},
                                extract_frames=self.extract)

    def test_rejects_reused_physical_capture(self) -> None:
        plan = deepcopy(self.plan)
        plan.pop("capture_plan_body_sha256")
        plan["episodes"][1]["media_relative_path"] = plan["episodes"][0]["media_relative_path"]
        hashed(plan, "capture_plan_body_sha256")
        receipt = deepcopy(self.receipt)
        receipt.pop("capture_body_sha256")
        receipt["capture_plan_body_sha256"] = plan["capture_plan_body_sha256"]
        receipt["episodes"][1]["media_path"] = receipt["episodes"][0]["media_path"]
        receipt["episodes"][1]["media_sha256"] = receipt["episodes"][0]["media_sha256"]
        hashed(receipt, "capture_body_sha256")
        with self.assertRaisesRegex(ProspectiveCaptureError, "reuses another episode"):
            self.run_materialize(receipt=receipt, capture_plan=plan)

    def test_rejects_capture_receipt_bound_to_another_plan(self) -> None:
        receipt = deepcopy(self.receipt)
        receipt.pop("capture_body_sha256")
        receipt["capture_plan_body_sha256"] = "0" * 64
        hashed(receipt, "capture_body_sha256")
        with self.assertRaisesRegex(ProspectiveCaptureError, "not bound to capture plan"):
            self.run_materialize(receipt=receipt)

    def test_pyav_probe_and_fixed_offset_extraction(self) -> None:
        import av
        from PIL import Image

        video = self.capture_root / "pyav-integration.mp4"
        with av.open(str(video), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width = 64
            stream.height = 48
            stream.pix_fmt = "yuv420p"
            for index in range(40):
                image = Image.new("RGB", (64, 48), color=(index * 5 % 255, 20, 40))
                frame = av.VideoFrame.from_image(image)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        probed = _default_probe(video)
        self.assertEqual((64, 48), (probed["width"], probed["height"]))
        targets = [probed["duration_seconds"] + offset for offset in FRAME_OFFSETS_FROM_END_SECONDS]
        outputs = [self.root / f"frame-{index}.jpg" for index in range(3)]
        actual = _default_extract(video, targets, outputs)
        self.assertTrue(all(output.is_file() for output in outputs))
        self.assertTrue(all(abs(left - right) <= 0.25 for left, right in zip(targets, actual, strict=True)))


if __name__ == "__main__":
    unittest.main()
