from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import socket
import statistics
import struct
import threading
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Full, Queue
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

FRAME_SCHEMA = "blindassist_atoms3r_e2e_frame_r1"
SUMMARY_SCHEMA = "blindassist_atoms3r_e2e_summary_r1"
SYNC_SCHEMA = "blindassist_atoms3r_host_clock_sync_r1"
REQUIRED_FRAME_HEADERS = {
    "content-length",
    "x-sequence-id",
    "x-clock-domain",
    "x-frame-sequence",
    "x-capture-timestamp-us",
    "x-capture-timestamp-semantics",
    "x-jpeg-ready-timestamp-us",
    "x-jpeg-ready-timestamp-semantics",
    "x-device-send-start-timestamp-us",
    "x-frame-ready-interval-us",
    "x-frame-acquire-duration-us",
    "x-jpeg-metadata-prepare-duration-us",
    "x-previous-response-write-valid",
    "x-previous-frame-sequence",
    "x-previous-response-write-duration-us",
    "x-tof-timestamp-us",
    "x-tof-timestamp-semantics",
    "x-tof-minus-capture-us",
    "x-tof-age-at-jpeg-ready-us",
    "x-tof-during-acquire",
    "x-tof-updates-during-acquire",
    "x-tof-updates-since-previous-frame",
    "x-tof-sampling-enabled",
    "x-tof-valid",
    "x-tof-range-mm",
    "x-tof-status",
    "x-tof-range-status-code",
    "x-jpeg-size-bytes",
    "x-width",
    "x-height",
    "x-jpeg-quality",
    "x-auto-exposure",
    "x-camera-psram-dma-enabled",
    "x-stream-tcp-nodelay",
    "x-stream-preamble-coalesced",
    "x-stream-handler-core",
    "x-stream-handler-priority",
    "x-exposure-value",
    "x-wifi-rssi-dbm",
    "x-free-heap-bytes",
}


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[min(max(index, 0), len(ordered) - 1)]


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "stdev": statistics.pstdev(values)
        if len(values) > 1
        else 0.0
        if values
        else None,
    }


@dataclass(frozen=True)
class ClockSync:
    sample_id: int
    host_midpoint_us: float
    device_midpoint_us: float
    device_minus_host_us: float
    round_trip_us: float
    error_bound_us: float
    sequence_id: str
    clock_domain: str
    method: str


def synchronize_clock(
    base_url: str, attempts: int, starting_id: int, udp_port: int
) -> tuple[ClockSync, list[dict[str, Any]]]:
    candidates: list[ClockSync] = []
    rows: list[dict[str, Any]] = []
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise ValueError(f"Clock sync requires an http base URL, got {base_url!r}")
    sequence_id = ""
    clock_domain = ""
    with urllib.request.urlopen(f"{base_url}/api/time", timeout=3) as response:
        identity = json.load(response)
        sequence_id = str(identity["sequence_id"])
        clock_domain = str(identity["clock_domain"])
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.settimeout(1.0)
        destination = (parsed.hostname, udp_port)
        for attempt in range(attempts):
            request_id = starting_id + attempt
            host_start_ns = time.perf_counter_ns()
            connection.sendto(
                struct.pack("<4sIQ", b"BAT0", request_id, host_start_ns), destination
            )
            payload, source = connection.recvfrom(64)
            host_end_ns = time.perf_counter_ns()
            if source[0] != socket.gethostbyname(parsed.hostname):
                raise RuntimeError(f"Unexpected UDP timing source: {source}")
            if len(payload) != 24:
                raise RuntimeError(
                    f"Unexpected UDP timing response length: {len(payload)}"
                )
            magic, response_id, device_received_us, device_ready_us = struct.unpack(
                "<4sIQQ", payload
            )
            if magic != b"BAT1" or response_id != request_id:
                raise RuntimeError("Mismatched UDP timing response")
            host_start_us = host_start_ns / 1000.0
            host_end_us = host_end_ns / 1000.0
            host_midpoint_us = (host_start_us + host_end_us) / 2.0
            device_midpoint_us = (device_received_us + device_ready_us) / 2.0
            round_trip_us = (host_end_us - host_start_us) - (
                device_ready_us - device_received_us
            )
            sync = ClockSync(
                sample_id=starting_id + attempt,
                host_midpoint_us=host_midpoint_us,
                device_midpoint_us=device_midpoint_us,
                device_minus_host_us=device_midpoint_us - host_midpoint_us,
                round_trip_us=max(round_trip_us, 0.0),
                error_bound_us=max(round_trip_us, 0.0) / 2.0,
                sequence_id=sequence_id,
                clock_domain=clock_domain,
                method=f"udp_midpoint_port_{udp_port}",
            )
            candidates.append(sync)
            rows.append({"schema": SYNC_SCHEMA, **asdict(sync)})
    return min(candidates, key=lambda item: item.round_trip_us), rows


class SharedMonitor:
    def __init__(
        self,
        base_url: str,
        status_interval_s: float,
        sync_interval_s: float,
        timing_udp_port: int,
        initial_sync: ClockSync,
    ):
        self.base_url = base_url
        self.status_interval_s = status_interval_s
        self.sync_interval_s = sync_interval_s
        self.timing_udp_port = timing_udp_port
        self.current_sync = initial_sync
        self.status_rows: list[dict[str, Any]] = []
        self.sync_rows: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="atoms3r-status-monitor", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(self.status_interval_s, 2.0) + 3.0)

    def sync_snapshot(self) -> ClockSync:
        with self.lock:
            return self.current_sync

    def _run(self) -> None:
        next_status = 0.0
        next_sync = time.monotonic() + self.sync_interval_s
        sync_id = self.current_sync.sample_id + 1
        while not self.stop_event.wait(0.1):
            now = time.monotonic()
            if now >= next_status:
                try:
                    host_received_ns = time.perf_counter_ns()
                    with urllib.request.urlopen(
                        f"{self.base_url}/api/status", timeout=3
                    ) as response:
                        payload = json.load(response)
                    payload["host_status_received_monotonic_ns"] = host_received_ns
                    with self.lock:
                        self.status_rows.append(payload)
                except Exception as error:  # noqa: BLE001 - sustained monitor must retain transient failures
                    with self.lock:
                        self.errors.append(f"status:{type(error).__name__}:{error}")
                next_status = now + self.status_interval_s
            if now >= next_sync:
                try:
                    selected, rows = synchronize_clock(
                        self.base_url, 5, sync_id, self.timing_udp_port
                    )
                    sync_id += len(rows)
                    with self.lock:
                        self.current_sync = selected
                        self.sync_rows.extend(rows)
                except Exception as error:  # noqa: BLE001 - sustained monitor must retain transient failures
                    with self.lock:
                        self.errors.append(f"sync:{type(error).__name__}:{error}")
                next_sync = now + self.sync_interval_s


class NoPipeline:
    identity = "NOT_CONFIGURED"

    @staticmethod
    def infer(_image: np.ndarray, _metadata: dict[str, Any]) -> Any:
        raise RuntimeError("No inference pipeline configured")

    @staticmethod
    def calculate_risk(_inference: Any, _metadata: dict[str, Any]) -> Any:
        raise RuntimeError("No risk pipeline configured")

    @staticmethod
    def emit_feedback(_risk: Any, _metadata: dict[str, Any]) -> Any:
        raise RuntimeError("No feedback pipeline configured")


def load_pipeline(
    module_path: Path | None,
    model_path: Path | None,
    feedback_mode: str,
    num_threads: int,
) -> Any:
    if module_path is None:
        return NoPipeline()
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location("atoms3r_e2e_pipeline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    spec.loader.exec_module(module)
    pipeline = module.build_pipeline(
        model_path=model_path,
        feedback_mode=feedback_mode,
        num_threads=num_threads,
    )
    for method in ("infer", "calculate_risk", "emit_feedback"):
        if not callable(getattr(pipeline, method, None)):
            raise TypeError(f"Pipeline is missing callable {method}()")
    return pipeline


def read_mjpeg_frames(
    stream_url: str,
) -> Iterator[tuple[dict[str, str], bytes, int, int, int]]:
    with urllib.request.urlopen(stream_url, timeout=10) as response:
        while True:
            boundary = response.readline()
            if not boundary:
                return
            if not boundary.startswith(b"--") and not boundary.startswith(b"\r\n--"):
                continue
            headers: dict[str, str] = {}
            while True:
                line = response.readline()
                if not line:
                    return
                if line in (b"\r\n", b"\n"):
                    break
                name, separator, value = line.decode("ascii").partition(":")
                if not separator:
                    raise ValueError(f"Malformed MJPEG part header: {line!r}")
                headers[name.strip().lower()] = value.strip()
            missing = sorted(REQUIRED_FRAME_HEADERS - headers.keys())
            if missing:
                raise ValueError(f"MJPEG frame headers missing: {missing}")
            content_length = int(headers["content-length"])
            host_read_start_ns = time.perf_counter_ns()
            first_byte = response.read(1)
            host_first_byte_received_ns = time.perf_counter_ns()
            if not first_byte:
                raise EOFError("Missing first JPEG byte")
            jpeg = first_byte + response.read(content_length - 1)
            host_jpeg_complete_ns = time.perf_counter_ns()
            if len(jpeg) != content_length:
                raise EOFError(f"Short JPEG body: {len(jpeg)} != {content_length}")
            yield (
                headers,
                jpeg,
                host_read_start_ns,
                host_first_byte_received_ns,
                host_jpeg_complete_ns,
            )


@dataclass(frozen=True)
class FramePacket:
    headers: dict[str, str]
    jpeg: bytes
    host_read_start_ns: int
    host_first_byte_received_ns: int
    host_jpeg_complete_ns: int
    connection_index: int


class LatestFrameReader:
    def __init__(self, stream_url: str) -> None:
        self.stream_url = stream_url
        self.queue: Queue[FramePacket] = Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="atoms3r-mjpeg-reader", daemon=True
        )
        self.latest_queue_overwrite_count = 0
        self.overwritten_frames: list[dict[str, Any]] = []
        self.reconnect_count = 0
        self.errors: list[str] = []

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=12.0)
        if self.thread.is_alive():
            self.errors.append("reader_stop:thread_did_not_stop_within_12s")

    def get(self, timeout_s: float) -> FramePacket:
        return self.queue.get(timeout=timeout_s)

    def offer(self, packet: FramePacket) -> None:
        while not self.stop_event.is_set():
            try:
                self.queue.put_nowait(packet)
                return
            except Full:
                try:
                    stale = self.queue.get_nowait()
                    self.latest_queue_overwrite_count += 1
                    overwritten_ns = time.perf_counter_ns()
                    self.overwritten_frames.append(
                        {
                            "schema": "blindassist_atoms3r_overwritten_frame_r0",
                            "sequence_id": stale.headers.get("x-sequence-id"),
                            "frame_sequence": int(
                                stale.headers.get("x-frame-sequence", "-1")
                            ),
                            "host_first_byte_received_monotonic_ns": (
                                stale.host_first_byte_received_ns
                            ),
                            "host_full_frame_received_monotonic_ns": (
                                stale.host_jpeg_complete_ns
                            ),
                            "host_queue_overwritten_monotonic_ns": overwritten_ns,
                            "queue_wait_before_overwrite_ms": (
                                overwritten_ns - stale.host_jpeg_complete_ns
                            )
                            / 1_000_000.0,
                            "frame_overwritten": True,
                        }
                    )
                except Empty:
                    continue

    def _run(self) -> None:
        connection_index = 0
        while not self.stop_event.is_set():
            connection_index += 1
            try:
                for (
                    headers,
                    jpeg,
                    read_start_ns,
                    first_byte_received_ns,
                    jpeg_complete_ns,
                ) in read_mjpeg_frames(self.stream_url):
                    if self.stop_event.is_set():
                        return
                    self.offer(
                        FramePacket(
                            headers=headers,
                            jpeg=jpeg,
                            host_read_start_ns=read_start_ns,
                            host_first_byte_received_ns=first_byte_received_ns,
                            host_jpeg_complete_ns=jpeg_complete_ns,
                            connection_index=connection_index,
                        )
                    )
                if not self.stop_event.is_set():
                    raise EOFError("MJPEG stream ended")
            except Exception as error:  # noqa: BLE001 - retain and recover stream failures
                if self.stop_event.is_set():
                    return
                self.reconnect_count += 1
                self.errors.append(f"stream:{type(error).__name__}:{error}")
                self.stop_event.wait(1.0)


def decode_jpeg(jpeg: bytes) -> np.ndarray:
    encoded = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV rejected JPEG")
    return image


def create_output_dir(root: Path) -> Path:
    output = root / utc_stamp()
    output.mkdir(parents=True, exist_ok=False)
    return output


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Expected boolean header, got {value!r}")


def frame_row(
    headers: dict[str, str],
    jpeg: bytes,
    host_read_start_ns: int,
    host_first_byte_received_ns: int,
    host_jpeg_complete_ns: int,
    sync: ClockSync,
    pipeline: Any,
    connection_index: int,
) -> dict[str, Any]:
    decode_start_ns = time.perf_counter_ns()
    image = decode_jpeg(jpeg)
    decode_complete_ns = time.perf_counter_ns()
    capture_us = int(headers["x-capture-timestamp-us"])
    jpeg_ready_us = int(headers["x-jpeg-ready-timestamp-us"])
    device_send_start_us = int(headers["x-device-send-start-timestamp-us"])
    frame_acquire_duration_us = int(headers["x-frame-acquire-duration-us"])
    frame_acquire_started_us = jpeg_ready_us - frame_acquire_duration_us
    mapped_capture_host_us = capture_us - sync.device_minus_host_us
    mapped_jpeg_ready_host_us = jpeg_ready_us - sync.device_minus_host_us
    mapped_send_start_host_us = device_send_start_us - sync.device_minus_host_us

    row: dict[str, Any] = {
        "schema": FRAME_SCHEMA,
        "stream_connection_index": connection_index,
        "sequence_id": headers["x-sequence-id"],
        "clock_domain": headers["x-clock-domain"],
        "clock_sync_sample_id": sync.sample_id,
        "clock_sync_rtt_us": sync.round_trip_us,
        "clock_sync_error_bound_us": sync.error_bound_us,
        "frame_sequence": int(headers["x-frame-sequence"]),
        "frame_overwritten": False,
        "capture_timestamp_us": capture_us,
        "capture_timestamp_semantics": headers["x-capture-timestamp-semantics"],
        "jpeg_ready_timestamp_us": jpeg_ready_us,
        "jpeg_ready_timestamp_semantics": headers["x-jpeg-ready-timestamp-semantics"],
        "device_send_start_timestamp_us": device_send_start_us,
        "device_frame_ready_interval_us": int(headers["x-frame-ready-interval-us"]),
        "device_frame_acquire_duration_us": frame_acquire_duration_us,
        "device_capture_to_fb_return_us": jpeg_ready_us - capture_us,
        "device_capture_minus_acquire_start_us": (
            capture_us - frame_acquire_started_us
        ),
        "device_jpeg_metadata_prepare_duration_us": int(
            headers["x-jpeg-metadata-prepare-duration-us"]
        ),
        "reported_previous_response_write_valid": parse_bool(
            headers["x-previous-response-write-valid"]
        ),
        "reported_previous_frame_sequence": int(headers["x-previous-frame-sequence"]),
        "reported_previous_response_write_duration_us": int(
            headers["x-previous-response-write-duration-us"]
        ),
        "preceding_response_write_duration_us": (
            int(headers["x-previous-response-write-duration-us"])
            if parse_bool(headers["x-previous-response-write-valid"])
            else None
        ),
        "device_response_write_duration_us": None,
        "tof_timestamp_us": int(headers["x-tof-timestamp-us"]),
        "tof_timestamp_semantics": headers["x-tof-timestamp-semantics"],
        "tof_minus_capture_us": (
            int(headers["x-tof-minus-capture-us"])
            if int(headers["x-tof-timestamp-us"]) > 0
            else None
        ),
        "tof_age_at_jpeg_ready_us": int(headers["x-tof-age-at-jpeg-ready-us"]),
        "tof_during_acquire": parse_bool(headers["x-tof-during-acquire"]),
        "tof_updates_during_acquire": int(headers["x-tof-updates-during-acquire"]),
        "tof_updates_since_previous_frame": int(
            headers["x-tof-updates-since-previous-frame"]
        ),
        "tof_sampling_enabled": parse_bool(headers["x-tof-sampling-enabled"]),
        "tof_valid": parse_bool(headers["x-tof-valid"]),
        "tof_range_mm": int(headers["x-tof-range-mm"]),
        "tof_status": headers["x-tof-status"],
        "tof_range_status_code": int(headers["x-tof-range-status-code"]),
        "width": int(headers["x-width"]),
        "height": int(headers["x-height"]),
        "jpeg_quality": int(headers["x-jpeg-quality"]),
        "jpeg_bytes": len(jpeg),
        "device_jpeg_size_bytes": int(headers["x-jpeg-size-bytes"]),
        "auto_exposure": parse_bool(headers["x-auto-exposure"]),
        "camera_psram_dma_enabled": parse_bool(headers["x-camera-psram-dma-enabled"]),
        "stream_tcp_nodelay": parse_bool(headers["x-stream-tcp-nodelay"]),
        "stream_preamble_coalesced": parse_bool(
            headers["x-stream-preamble-coalesced"]
        ),
        "stream_handler_core": int(headers["x-stream-handler-core"]),
        "stream_handler_priority": int(headers["x-stream-handler-priority"]),
        "exposure_value": int(headers["x-exposure-value"]),
        "device_wifi_rssi_dbm": int(headers["x-wifi-rssi-dbm"]),
        "device_free_heap_bytes": int(headers["x-free-heap-bytes"]),
        "host_jpeg_read_start_monotonic_ns": host_read_start_ns,
        "host_first_byte_received_monotonic_ns": host_first_byte_received_ns,
        "host_jpeg_complete_monotonic_ns": host_jpeg_complete_ns,
        "decode_start_monotonic_ns": decode_start_ns,
        "decode_complete_monotonic_ns": decode_complete_ns,
        "device_capture_to_jpeg_ready_ms": (jpeg_ready_us - capture_us) / 1000.0,
        "device_jpeg_ready_to_send_start_ms": (device_send_start_us - jpeg_ready_us)
        / 1000.0,
        "capture_to_host_read_start_ms": (
            host_read_start_ns / 1000.0 - mapped_capture_host_us
        )
        / 1000.0,
        "capture_to_host_jpeg_complete_ms": (
            host_jpeg_complete_ns / 1000.0 - mapped_capture_host_us
        )
        / 1000.0,
        "jpeg_ready_to_host_read_start_ms": (
            host_read_start_ns / 1000.0 - mapped_jpeg_ready_host_us
        )
        / 1000.0,
        "send_start_to_host_jpeg_complete_ms": (
            host_jpeg_complete_ns / 1000.0 - mapped_send_start_host_us
        )
        / 1000.0,
        "host_jpeg_read_ms": (host_jpeg_complete_ns - host_read_start_ns) / 1_000_000.0,
        "host_first_byte_wait_ms": (host_first_byte_received_ns - host_read_start_ns)
        / 1_000_000.0,
        "host_first_byte_to_full_frame_ms": (
            host_jpeg_complete_ns - host_first_byte_received_ns
        )
        / 1_000_000.0,
        "host_latest_queue_wait_ms": (decode_start_ns - host_jpeg_complete_ns)
        / 1_000_000.0,
        "host_jpeg_decode_ms": (decode_complete_ns - decode_start_ns) / 1_000_000.0,
        "capture_to_decode_complete_ms": (
            decode_complete_ns / 1000.0 - mapped_capture_host_us
        )
        / 1000.0,
        "pipeline_identity": pipeline.identity,
        "pipeline_num_threads": getattr(pipeline, "num_threads", None),
        "inference_complete_monotonic_ns": None,
        "risk_complete_monotonic_ns": None,
        "feedback_complete_monotonic_ns": None,
        "host_inference_ms": None,
        "host_risk_ms": None,
        "host_feedback_dispatch_ms": None,
        "capture_to_inference_complete_ms": None,
        "capture_to_risk_complete_ms": None,
        "capture_to_feedback_complete_ms": None,
        "risk_result": None,
        "feedback_result": None,
        "slow_frame": None,
    }
    if pipeline.identity != "NOT_CONFIGURED":
        inference_start_ns = time.perf_counter_ns()
        inference = pipeline.infer(image, row)
        inference_complete_ns = time.perf_counter_ns()
        risk = pipeline.calculate_risk(inference, row)
        risk_complete_ns = time.perf_counter_ns()
        feedback = pipeline.emit_feedback(risk, row)
        feedback_complete_ns = time.perf_counter_ns()
        row.update(
            {
                "inference_complete_monotonic_ns": inference_complete_ns,
                "risk_complete_monotonic_ns": risk_complete_ns,
                "feedback_complete_monotonic_ns": feedback_complete_ns,
                "host_inference_ms": (inference_complete_ns - inference_start_ns)
                / 1_000_000.0,
                "host_risk_ms": (risk_complete_ns - inference_complete_ns)
                / 1_000_000.0,
                "host_feedback_dispatch_ms": (feedback_complete_ns - risk_complete_ns)
                / 1_000_000.0,
                "capture_to_inference_complete_ms": (
                    inference_complete_ns / 1000.0 - mapped_capture_host_us
                )
                / 1000.0,
                "capture_to_risk_complete_ms": (
                    risk_complete_ns / 1000.0 - mapped_capture_host_us
                )
                / 1000.0,
                "capture_to_feedback_complete_ms": (
                    feedback_complete_ns / 1000.0 - mapped_capture_host_us
                )
                / 1000.0,
                "risk_result": risk,
                "feedback_result": feedback,
            }
        )
    return row


def finalize_device_attribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sequence = {(row["sequence_id"], row["frame_sequence"]): row for row in rows}
    for reporter in rows:
        if not reporter["reported_previous_response_write_valid"]:
            continue
        target = by_sequence.get(
            (
                reporter["sequence_id"],
                reporter["reported_previous_frame_sequence"],
            )
        )
        if target is not None:
            target["device_response_write_duration_us"] = reporter[
                "reported_previous_response_write_duration_us"
            ]

    intervals = [
        float(row["device_frame_ready_interval_us"])
        for row in rows
        if row["device_frame_ready_interval_us"] > 0
    ]
    if not intervals:
        for row in rows:
            row["slow_frame_evaluable"] = False
            row["slow_frame"] = False
        return {
            "definition": "interval_gt_median_plus_3_mad_or_gt_2_median",
            "median_interval_us": None,
            "mad_interval_us": None,
            "median_plus_3_mad_us": None,
            "twice_median_us": None,
        }
    median_interval_us = statistics.median(intervals)
    mad_interval_us = statistics.median(
        [abs(value - median_interval_us) for value in intervals]
    )
    median_mad_threshold_us = median_interval_us + 3.0 * mad_interval_us
    twice_median_us = 2.0 * median_interval_us
    for row in rows:
        interval_us = float(row["device_frame_ready_interval_us"])
        row["slow_frame_evaluable"] = interval_us > 0
        row["slow_frame"] = bool(
            interval_us > 0
            and (interval_us > median_mad_threshold_us or interval_us > twice_median_us)
        )
    return {
        "definition": "interval_gt_median_plus_3_mad_or_gt_2_median",
        "median_interval_us": median_interval_us,
        "mad_interval_us": mad_interval_us,
        "median_plus_3_mad_us": median_mad_threshold_us,
        "twice_median_us": twice_median_us,
    }


def attribution_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "device_frame_ready_interval_us",
        "device_frame_acquire_duration_us",
        "device_capture_to_fb_return_us",
        "device_capture_minus_acquire_start_us",
        "device_jpeg_metadata_prepare_duration_us",
        "device_response_write_duration_us",
        "preceding_response_write_duration_us",
        "device_jpeg_size_bytes",
        "exposure_value",
        "device_wifi_rssi_dbm",
        "device_free_heap_bytes",
        "tof_age_at_jpeg_ready_us",
        "tof_updates_during_acquire",
        "tof_updates_since_previous_frame",
        "host_first_byte_wait_ms",
        "host_first_byte_to_full_frame_ms",
        "host_latest_queue_wait_ms",
        "host_jpeg_decode_ms",
    )

    def finite_values(key: str) -> list[float]:
        return [
            float(row[key])
            for row in rows
            if row.get(key) is not None and math.isfinite(float(row[key]))
        ]

    return {
        "count": len(rows),
        "metrics": {key: metric_summary(finite_values(key)) for key in keys},
        "auto_exposure_fraction": (
            sum(1 for row in rows if row["auto_exposure"]) / len(rows) if rows else None
        ),
        "tof_during_acquire_fraction": (
            sum(1 for row in rows if row["tof_during_acquire"]) / len(rows)
            if rows
            else None
        ),
    }


def summarize(
    rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    reconnects: int,
    latest_queue_overwrites: int,
    errors: list[str],
    slow_frame_contract: dict[str, Any],
) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [
            float(row[key])
            for row in rows
            if row.get(key) is not None and math.isfinite(float(row[key]))
        ]

    host_intervals_ms = [
        (
            rows[index]["host_jpeg_complete_monotonic_ns"]
            - rows[index - 1]["host_jpeg_complete_monotonic_ns"]
        )
        / 1_000_000.0
        for index in range(1, len(rows))
        if rows[index]["stream_connection_index"]
        == rows[index - 1]["stream_connection_index"]
    ]
    device_intervals_ms = [
        (rows[index]["capture_timestamp_us"] - rows[index - 1]["capture_timestamp_us"])
        / 1000.0
        for index in range(1, len(rows))
        if rows[index]["sequence_id"] == rows[index - 1]["sequence_id"]
        and rows[index]["capture_timestamp_us"]
        > rows[index - 1]["capture_timestamp_us"]
    ]
    sequence_gaps = [
        rows[index]["frame_sequence"] - rows[index - 1]["frame_sequence"] - 1
        for index in range(1, len(rows))
        if rows[index]["sequence_id"] == rows[index - 1]["sequence_id"]
        and rows[index]["frame_sequence"] > rows[index - 1]["frame_sequence"] + 1
    ]
    heaps = [
        float(row["free_heap_bytes"]) for row in status_rows if "free_heap_bytes" in row
    ]
    temperatures = [
        float(row["chip_temperature_c"])
        for row in status_rows
        if "chip_temperature_c" in row
    ]
    rssis = [
        float(row["wifi"]["rssi_dbm"])
        for row in status_rows
        if row.get("wifi", {}).get("connected")
    ]
    slow_rows = [row for row in rows if row.get("slow_frame")]
    normal_rows = [
        row
        for row in rows
        if row.get("slow_frame_evaluable") and not row.get("slow_frame")
    ]
    return {
        "schema": SUMMARY_SCHEMA,
        "frame_count": len(rows),
        "stream_reconnect_count": reconnects,
        "host_latest_queue_overwrite_count": latest_queue_overwrites,
        "errors": errors,
        "pipeline_identity": rows[0]["pipeline_identity"] if rows else "UNKNOWN",
        "pipeline_num_threads": (rows[0].get("pipeline_num_threads") if rows else None),
        "host_logical_cpu_count": os.cpu_count(),
        "slow_frame_contract": slow_frame_contract,
        "slow_frame_count": len(slow_rows),
        "slow_frame_fraction": (
            len(slow_rows) / (len(slow_rows) + len(normal_rows))
            if slow_rows or normal_rows
            else None
        ),
        "slow_frame_profile": attribution_profile(slow_rows),
        "normal_frame_profile": attribution_profile(normal_rows),
        "capture_to_host_jpeg_complete_ms": metric_summary(
            values("capture_to_host_jpeg_complete_ms")
        ),
        "capture_to_host_read_start_ms": metric_summary(
            values("capture_to_host_read_start_ms")
        ),
        "device_capture_to_jpeg_ready_ms": metric_summary(
            values("device_capture_to_jpeg_ready_ms")
        ),
        "device_capture_to_fb_return_ms": metric_summary(
            [value / 1000.0 for value in values("device_capture_to_fb_return_us")]
        ),
        "device_capture_minus_acquire_start_ms": metric_summary(
            [
                value / 1000.0
                for value in values("device_capture_minus_acquire_start_us")
            ]
        ),
        "jpeg_ready_to_host_read_start_ms": metric_summary(
            values("jpeg_ready_to_host_read_start_ms")
        ),
        "send_start_to_host_jpeg_complete_ms": metric_summary(
            values("send_start_to_host_jpeg_complete_ms")
        ),
        "host_jpeg_read_ms": metric_summary(values("host_jpeg_read_ms")),
        "host_latest_queue_wait_ms": metric_summary(
            values("host_latest_queue_wait_ms")
        ),
        "host_jpeg_decode_ms": metric_summary(values("host_jpeg_decode_ms")),
        "capture_to_decode_complete_ms": metric_summary(
            values("capture_to_decode_complete_ms")
        ),
        "host_inference_ms": metric_summary(values("host_inference_ms")),
        "host_risk_ms": metric_summary(values("host_risk_ms")),
        "host_feedback_dispatch_ms": metric_summary(
            values("host_feedback_dispatch_ms")
        ),
        "capture_to_inference_complete_ms": metric_summary(
            values("capture_to_inference_complete_ms")
        ),
        "capture_to_risk_complete_ms": metric_summary(
            values("capture_to_risk_complete_ms")
        ),
        "capture_to_feedback_complete_ms": metric_summary(
            values("capture_to_feedback_complete_ms")
        ),
        "clock_sync_rtt_ms": metric_summary(
            [value / 1000.0 for value in values("clock_sync_rtt_us")]
        ),
        "clock_sync_error_bound_ms": metric_summary(
            [value / 1000.0 for value in values("clock_sync_error_bound_us")]
        ),
        "host_interarrival_ms": metric_summary(host_intervals_ms),
        "device_capture_interval_ms": metric_summary(device_intervals_ms),
        "tof_minus_capture_ms": metric_summary(
            [value / 1000.0 for value in values("tof_minus_capture_us")]
        ),
        "absolute_tof_capture_skew_ms": metric_summary(
            [abs(value) / 1000.0 for value in values("tof_minus_capture_us")]
        ),
        "tof_valid_fraction": (
            sum(1 for row in rows if row["tof_valid"]) / len(rows) if rows else None
        ),
        "tof_sampling_enabled_values": sorted(
            {
                row["tof_sampling_enabled"]
                for row in rows
                if row.get("tof_sampling_enabled") is not None
            }
        ),
        "camera_psram_dma_enabled_values": sorted(
            {
                row["camera_psram_dma_enabled"]
                for row in rows
                if row.get("camera_psram_dma_enabled") is not None
            }
        ),
        "stream_tcp_nodelay_values": sorted(
            {
                row["stream_tcp_nodelay"]
                for row in rows
                if row.get("stream_tcp_nodelay") is not None
            }
        ),
        "stream_preamble_coalesced_values": sorted(
            {
                row["stream_preamble_coalesced"]
                for row in rows
                if row.get("stream_preamble_coalesced") is not None
            }
        ),
        "stream_handler_core_values": sorted(
            {
                row["stream_handler_core"]
                for row in rows
                if row.get("stream_handler_core") is not None
            }
        ),
        "stream_handler_priority_values": sorted(
            {
                row["stream_handler_priority"]
                for row in rows
                if row.get("stream_handler_priority") is not None
            }
        ),
        "sequence_gap_event_count": len(sequence_gaps),
        "sequence_gap_total_frames": sum(sequence_gaps),
        "free_heap_bytes": metric_summary(heaps),
        "free_heap_first_to_last_bytes": heaps[-1] - heaps[0] if heaps else None,
        "chip_temperature_c": metric_summary(temperatures),
        "chip_temperature_first_to_last_c": (
            temperatures[-1] - temperatures[0] if temperatures else None
        ),
        "wifi_rssi_dbm": metric_summary(rssis),
        "status_sample_count": len(status_rows),
        "run_accepted": bool(rows) and reconnects == 0 and not errors,
        "run_acceptance_failures": (
            (["NO_FRAMES"] if not rows else [])
            + (["STREAM_RECONNECTS_PRESENT"] if reconnects else [])
            + (["ERRORS_PRESENT"] if errors else [])
        ),
        "limitations": [
            "capture_timestamp_us is first DMA buffer since boot, not exposure start",
            "clock mapping uses minimum-RTT UDP midpoint and retains its error bound",
            "host receive timestamps are parser read boundaries, not NIC hardware timestamps",
            "sequence gaps are delivery observations and cannot prove sensor-internal frame drops",
            "latest-queue overwrites intentionally discard stale host frames and contribute sequence gaps",
            "chip temperature is the ESP32 internal sensor, not enclosure or camera temperature",
            "voice or vibration latency is absent unless a real pipeline module emits that output",
        ],
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure AtomS3R MJPEG/ToF end-to-end timing."
    )
    parser.add_argument("--base-url", default="http://192.168.5.11")
    parser.add_argument("--duration-seconds", type=float, default=300.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts.local/evidence/atoms3r-e2e"),
    )
    parser.add_argument("--clock-sync-attempts", type=int, default=20)
    parser.add_argument("--timing-udp-port", type=int, default=3333)
    parser.add_argument("--status-interval-seconds", type=float, default=5.0)
    parser.add_argument("--resync-interval-seconds", type=float, default=60.0)
    parser.add_argument("--pipeline-module", type=Path)
    parser.add_argument("--pipeline-model", type=Path)
    parser.add_argument("--pipeline-num-threads", type=int, default=4)
    parser.add_argument(
        "--feedback-mode", choices=("record_only", "console"), default="record_only"
    )
    args = parser.parse_args()
    if (
        args.duration_seconds <= 0
        or args.clock_sync_attempts < 3
        or not 1 <= args.pipeline_num_threads <= 64
    ):
        parser.error(
            "duration must be positive, clock sync attempts must be at least 3, "
            "and pipeline threads must be in 1..64"
        )

    output_dir = create_output_dir(args.output_root)
    base_url = args.base_url.rstrip("/")
    stream_url = (
        f"{base_url.rsplit(':', 1)[0]}:81/stream"
        if base_url.count(":") == 2
        else f"{base_url}:81/stream"
    )
    initial_sync, initial_sync_rows = synchronize_clock(
        base_url, args.clock_sync_attempts, 0, args.timing_udp_port
    )
    pipeline = load_pipeline(
        args.pipeline_module,
        args.pipeline_model,
        args.feedback_mode,
        args.pipeline_num_threads,
    )
    monitor = SharedMonitor(
        base_url,
        args.status_interval_seconds,
        args.resync_interval_seconds,
        args.timing_udp_port,
        initial_sync,
    )
    monitor.sync_rows.extend(initial_sync_rows)
    monitor.start()

    rows: list[dict[str, Any]] = []
    reader = LatestFrameReader(stream_url)
    started = time.monotonic()
    reader.start()
    try:
        while time.monotonic() - started < args.duration_seconds:
            try:
                packet = reader.get(timeout_s=1.0)
            except Empty:
                continue
            sync = monitor.sync_snapshot()
            rows.append(
                frame_row(
                    packet.headers,
                    packet.jpeg,
                    packet.host_read_start_ns,
                    packet.host_first_byte_received_ns,
                    packet.host_jpeg_complete_ns,
                    sync,
                    pipeline,
                    packet.connection_index,
                )
            )
            if args.max_frames > 0 and len(rows) >= args.max_frames:
                break
    finally:
        reader.stop()
        monitor.stop()

    slow_frame_contract = finalize_device_attribution(rows)
    with (output_dir / "frames.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    with monitor.lock:
        status_rows = list(monitor.status_rows)
        sync_rows = list(monitor.sync_rows)
        monitor_errors = list(monitor.errors)
    with (output_dir / "overwritten_frames.jsonl").open(
        "x", encoding="utf-8"
    ) as handle:
        for row in reader.overwritten_frames:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    with (output_dir / "status.jsonl").open("x", encoding="utf-8") as handle:
        for row in status_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    with (output_dir / "clock_sync.jsonl").open("x", encoding="utf-8") as handle:
        for row in sync_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    summary = summarize(
        rows,
        status_rows,
        reader.reconnect_count,
        reader.latest_queue_overwrite_count,
        reader.errors + monitor_errors,
        slow_frame_contract,
    )
    summary["started_at_utc"] = output_dir.name
    summary["requested_duration_seconds"] = args.duration_seconds
    summary["actual_duration_seconds"] = time.monotonic() - started
    summary["effective_received_fps"] = (
        len(rows) / summary["actual_duration_seconds"]
        if summary["actual_duration_seconds"] > 0
        else None
    )
    summary["output_dir"] = str(output_dir.resolve())
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["run_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
