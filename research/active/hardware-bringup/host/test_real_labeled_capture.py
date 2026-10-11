"""Independent offline flow checks; no serial device or real geometry labels."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

from capture import Decoder
import real_labeled_capture as guided
import real_labeled_annotation as annotation

REPO = Path(__file__).resolve().parents[4]
PAYLOAD = REPO / "artifacts.local/work/real-labeled-capture-prep-20261011"
RAW = REPO / "artifacts.local/hardware-bringup/captures/cnh-h3-1mhz-20260927/raw.bin"


class FakeClock:
    def __init__(self):
        self.value = 0.

    def sleep(self, seconds):
        self.value += seconds

    def monotonic(self):
        return self.value

    def monotonic_ns(self):
        return int(self.value * 1e9)

    def utcnow(self):
        return (datetime(2026, 10, 11, tzinfo=timezone.utc) + timedelta(seconds=self.value)).isoformat()


class Answers:
    def __init__(self, stop_after=1, redo_first=False):
        self.prompts, self.finished = [], 0
        self.stop_after, self.redo_first = stop_after, redo_first

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if len(self.prompts) > 400:
            raise RuntimeError(f"mock answer cannot satisfy prompt: {prompt}")
        if "无物体" in prompt or "开阔填" in prompt:
            return "5.2"
        if "实测板宽" in prompt:
            return "1.5"
        if "实测板高" in prompt:
            return "1.4"
        if "前照" in prompt and "路径" in prompt:
            return "phone-before.jpg"
        if "后照" in prompt and "路径" in prompt:
            return "phone-after.jpg"
        if "输入 y 开始" in prompt:
            return "y"
        if "重录本段" in prompt:
            self.finished += 1
            if self.redo_first and self.finished == 1:
                return "y"
            return "q" if self.stop_after and self.finished >= self.stop_after else ""
        if "材质/颜色" in prompt:
            return "matte wood"
        if "支架位置" in prompt:
            return "right outside x=0.8m; photo review pending"
        if "传感器离地高度" in prompt:
            return "1.65"
        if "俯角" in prompt and "误差" not in prompt:
            return "10"
        if "距离/高度测量误差" in prompt:
            return "0.02"
        if "俯角读数误差" in prompt:
            return "2"
        suggested = re.search(r"\[(?:建议\s*)?(-?\d+(?:\.\d+)?)(?:，[^\]]*)?\]", prompt)
        if suggested:
            return suggested[1]
        return ""


class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        PAYLOAD.mkdir(parents=True, exist_ok=True)
        cls.workspace = tempfile.TemporaryDirectory(prefix="independent-replay-", dir=PAYLOAD)
        cls.root = Path(cls.workspace.name)
        cls.source_sha256 = hashlib.sha256(RAW.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        # Preserve receipts, raw replay outputs and failure evidence under artifacts.local.
        cls.workspace._finalizer.detach()

    def make(self, name, answers=None, duration=3., camera=None, interrupt_at=None):
        clock, devices, console = FakeClock(), [], []
        def factory():
            device = guided.ReplaySerial(RAW, clock)
            if interrupt_at:
                original = device.read
                def read(size):
                    if clock.monotonic() >= interrupt_at:
                        raise KeyboardInterrupt("injected replay interruption")
                    return original(size)
                device.read = read
            devices.append(device)
            return device
        app = guided.GuidedCapture(self.root/name, factory, clock=clock,
                                   input_fn=answers or Answers(), print_fn=console.append,
                                   segment_seconds=duration, countdown_seconds=5,
                                   camera_url=camera, replay=True)
        return app, devices, console

    def assert_integrity(self, directory):
        meta = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
        data = (directory/"raw.bin").read_bytes()
        self.assertEqual(len(data), meta["raw_bytes"])
        self.assertEqual(hashlib.sha256(data).hexdigest(), meta["raw_sha256"])
        rows = [json.loads(x) for x in (directory/"frames.jsonl").read_text(encoding="utf-8").splitlines()]
        decoded = []
        decoder = Decoder(lambda kind, row: decoded.append(row) if kind == "frames" else None)
        decoder.feed(data, 0)
        self.assertEqual([r["sensor"] for r in decoded], [r["sensor"] for r in rows])
        self.assertEqual(len(rows), meta["summary"]["frames"])
        self.assertTrue(all(r["host_received_monotonic_ns"] is not None and r["host_received_utc"] for r in rows))
        self.assertEqual(sorted(r["host_received_monotonic_ns"] for r in rows), [r["host_received_monotonic_ns"] for r in rows])
        return meta

    def test_real_raw_preflight_guidance_and_integrity(self):
        answers = Answers()
        app, devices, console = self.make("guided", answers)
        manifest = app.run_session(1)
        self.assertEqual("paused", manifest["status"])
        self.assertTrue(manifest["preflight"]["passed"])
        self.assertTrue(devices[0].closed)
        attempt = app.root/"session-01/open-dark/attempt-01"
        meta = self.assert_integrity(attempt)
        self.assertEqual("SIMULATED_SERIAL_REPLAY", meta["evidence_scope"])
        self.assertEqual("phone-before.jpg", meta["photo_before"])
        self.assertEqual("phone-after.jpg", meta["photo_after"])
        self.assertEqual(5.2, meta["target_distance_m"])
        self.assertEqual(1.65, meta["sensor_height_m"])
        self.assertEqual(10., meta["tilt_deg"])
        self.assertIn("逐zone固件距离mm", "\n".join(console))
        self.assertIn("ambient", "\n".join(console))
        self.assertTrue((app.root/"session-01/inventory.tsv").is_file())

    def test_preflight_rejects_identity_config_layout_rate_loss(self):
        app, _, _ = self.make("gate")
        manifest = app.run_session(1)
        events = [json.loads(x) for x in (app.root/"session-01/preflight-01/events.jsonl").read_text(encoding="utf-8").splitlines()]
        summary = manifest["preflight"]["summary"]
        self.assertTrue(guided.check_preflight(events, summary)["passed"])
        self.assertFalse(guided.check_preflight([], summary)["passed"])
        wrong_events = copy.deepcopy(events)
        wrong_events[0]["sensor"]["firmware"] = "not-h3"
        self.assertFalse(guided.check_preflight(wrong_events, summary)["passed"])
        for key, value in (("frames", 2), ("cnh_layout_counts", {"4x4x24": 40}),
                           ("observed_sensor_hz", 3.81), ("missing_sequences", 20),
                           ("device_errors", 1), ("sequence_resets_or_reorders", 1),
                           ("duplicate_sequences", 1),
                           ("issues", {"invalid_frame": 1})):
            with self.subTest(key=key):
                bad = dict(summary, **{key: value})
                self.assertFalse(guided.check_preflight(events, bad)["passed"])
        bad_config = copy.deepcopy(events)
        for event in bad_config:
            if event["sensor"].get("type") == "config":
                event["sensor"]["bins"] = 24
        self.assertFalse(guided.check_preflight(bad_config, summary)["passed"])

    def test_actual_thirty_second_window_preserves_recorded_bytes(self):
        app, devices, _ = self.make("thirty-second", duration=30.)
        app.device = app.serial_factory()
        try:
            app.collect(app.root/"single-30s", 30., {"condition": "offline-existing-raw"})
        finally:
            app.device.close()
        meta = self.assert_integrity(app.root/"single-30s")
        self.assertGreaterEqual(meta["summary"]["frames"], 150)
        self.assertTrue(30. <= meta["host_elapsed_seconds"] < 30.3)
        self.assertEqual(self.source_sha256, hashlib.sha256(RAW.read_bytes()).hexdigest())

    def test_stale_frames_before_manual_reset_are_retained_but_not_gated(self):
        data = RAW.read_bytes()
        old_frames = [line for line in data.splitlines(keepends=True) if b'"cnh_frame"' in line][-5:]
        fixture = self.root/"synthetic-stale-before-boot.bin"
        fixture.write_bytes(b"".join(old_frames) + data)
        clock = FakeClock()
        device = guided.ReplaySerial(fixture, clock)
        app = guided.GuidedCapture(self.root/"stale", lambda: device, clock=clock,
                                   input_fn=Answers(), print_fn=lambda x: None, replay=True)
        app.device = device
        try:
            result = app.collect(app.root/"preflight", 12., {"phase": "preflight"}, show=False)
        finally:
            device.close()
        self.assertGreater(result["summary"]["sequence_resets_or_reorders"], 0)
        self.assertTrue(guided.check_preflight(result["preflight_events"], result["preflight_summary"])["passed"])
        self.assert_integrity(app.root/"preflight")

    def test_countdown_interruption_does_not_leave_recording_attempt(self):
        app, devices, _ = self.make("countdown-interruption")
        original = app.clock.sleep
        def interrupt_sleep(seconds):
            if seconds == 1 and app.clock.monotonic() >= 12.:
                raise KeyboardInterrupt("injected interruption during countdown")
            original(seconds)
        app.clock.sleep = interrupt_sleep
        with self.assertRaises(KeyboardInterrupt):
            app.run_session(1)
        state = json.loads((app.root/"session-01/session.json").read_text(encoding="utf-8"))
        self.assertEqual("interrupted", state["status"])
        self.assertEqual("interrupted_before_recording", state["attempts"][0]["status"])
        self.assertTrue(devices[0].closed)

    def test_interruption_preservation_resume_and_redo(self):
        app, devices, _ = self.make("interrupt", interrupt_at=19.)
        with self.assertRaises(KeyboardInterrupt):
            app.run_session(1)
        directory = app.root/"session-01"
        old = self.assert_integrity(directory/"open-dark/attempt-01")
        self.assertEqual("interrupted", old["status"])
        self.assertTrue(devices[0].closed)
        resumed, _, _ = self.make("interrupt")
        state = resumed.run_session(1, resume=True)
        self.assertEqual("interrupted", state["attempts"][0]["status"])
        self.assertEqual("complete", state["attempts"][1]["status"])
        original_raw = (directory/"open-dark/attempt-02/raw.bin").read_bytes()
        redo, _, _ = self.make("interrupt")
        state = redo.run_session(1, redo=["open-dark"])
        self.assertEqual("superseded", state["attempts"][1]["status"])
        self.assertEqual(original_raw, (directory/"open-dark/attempt-02/raw.bin").read_bytes())
        self.assertEqual("complete", state["attempts"][2]["status"])
        self.assert_integrity(directory/"open-dark/attempt-03")

    def test_full_twelve_segment_flow_and_camera_unavailable(self):
        app, devices, _ = self.make("full", Answers(stop_after=None), duration=1., camera="http://127.0.0.1:81/stream")
        unavailable = lambda url, path: {"status": "unavailable", "error": "injected missing Wi-Fi", "synchronized": False}
        with patch.object(guided, "save_optional_jpeg", unavailable):
            state = app.run_session(1)
        self.assertEqual("complete", state["status"])
        self.assertEqual(12, len(state["attempts"]))
        self.assertEqual({s["segment_id"] for s in guided.segment_plan()}, {a["segment_id"] for a in state["attempts"]})
        for attempt in state["attempts"]:
            meta = self.assert_integrity(app.root/"session-01"/attempt["path"])
            self.assertEqual(2, len(meta["camera"]))
            self.assertTrue(all(x["status"] == "unavailable" for x in meta["camera"]))
        self.assertTrue(devices[0].closed)

    def test_optional_camera_never_accepts_external_upload_or_redirect_target(self):
        for url in ("https://example.com/stream", "http://8.8.8.8:81/stream", "http://localhost:81/stream", "http://192.168.1.2:81/upload"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                guided.local_camera_url(url)

    def test_measured_numbers_require_explicit_entry_and_open_unknown_is_none(self):
        app, _, messages = self.make("explicit-measurement")
        values = iter(("", "1.52"))
        app.ask = lambda prompt: next(values)
        self.assertEqual(1.52, app.number("measured height", 1.65, .3, 2.5))
        self.assertTrue(messages)
        app.ask = lambda prompt: ""
        self.assertIsNone(app.number("open distance unknown", optional=True))

    def test_annotation_synthetic_envelope_keeps_seq_and_host_timestamps(self):
        directory = self.root/"synthetic-annotation-interface"
        directory.mkdir()
        metadata_path = directory/"metadata.json"
        metadata_path.write_text('{"evidence_scope":"SYNTHETIC_TEST_ONLY"}', encoding="utf-8")
        review = annotation.review_template()
        review.update(capture_metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                      photo_before="synthetic-before.jpg", photo_after="synthetic-after.jpg",
                      photos_before_after_reviewed=True, stationary_verified=True,
                      measurement_reviewed=True, axis_alignment_reviewed=True,
                      query_completeness_reviewed={"HEAD": True, "BODY": True})
        review_path = directory/"review.json"
        review_path.write_text(json.dumps(review), encoding="utf-8")
        frames_path = directory/"frames.jsonl"
        frames_path.write_text(json.dumps({"sensor": {"seq": 7, "ms": 1234},
                                           "host_received_monotonic_ns": 999,
                                           "host_received_utc": "2026-10-11T00:00:00Z"})+"\n", encoding="utf-8")
        output = directory/"synthetic-labels.jsonl"
        annotation.annotate(frames_path, metadata_path, review_path, output)
        rows = [json.loads(x) for x in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(2, len(rows))
        for row in rows:
            self.assertEqual((7, 1234, 999, "2026-10-11T00:00:00Z"),
                             (row["seq"], row["ms"], row["host_received_monotonic_ns"], row["host_received_utc"]))
            self.assertEqual("NOT_EVALUABLE", row["timeliness"])

    def test_annotation_unknown_contact_pass_clear_and_uncertainty(self):
        review = annotation.review_template()
        self.assertTrue(all(x["label"] == "UNKNOWN" for x in annotation.classify_scene(review).values()))
        review.update(photo_before="before.jpg", photo_after="after.jpg", photos_before_after_reviewed=True,
                      stationary_verified=True, measurement_reviewed=True, axis_alignment_reviewed=True,
                      query_completeness_reviewed={"HEAD": True, "BODY": True})
        def rod(x, y, dx=.001):
            return {"type": "cylinder_y", "center_m": [x, y, 1.515],
                    "radius_interval_m": [.014, .016], "length_interval_m": [.198, .202],
                    "translation_uncertainty_m": [dx, .005, .005]}
        for x, y, expected in ((0, .1, ("POS", "clear")), (0, .65, ("clear", "POS")),
                               (.50, .1, ("clear", "clear")), (.35, .1, ("pass", "clear"))):
            review["objects"] = [rod(x, y)]
            result = annotation.classify_scene(review)
            self.assertEqual(expected, tuple(result[q]["label"] for q in ("HEAD", "BODY")))
        review["objects"] = [rod(.31, .1, .02)]
        self.assertEqual("UNKNOWN", annotation.classify_scene(review)["HEAD"]["label"])
        review["objects"] = []
        review["query_completeness_reviewed"]["HEAD"] = False
        self.assertEqual("UNKNOWN", annotation.classify_scene(review)["HEAD"]["label"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
