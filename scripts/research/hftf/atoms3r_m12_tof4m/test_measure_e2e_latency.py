import unittest

import cv2
import numpy as np
from measure_e2e_latency import ClockSync, NoPipeline, frame_row, summarize


def jpeg_bytes() -> bytes:
    accepted, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    if not accepted:
        raise RuntimeError("OpenCV JPEG fixture encoding failed")
    return encoded.tobytes()


def frame_headers(sequence: int = 7) -> dict[str, str]:
    return {
        "x-sequence-id": "device-boot",
        "x-clock-domain": "esp32_boot_monotonic:device-boot",
        "x-frame-sequence": str(sequence),
        "x-capture-timestamp-us": "100000",
        "x-capture-timestamp-semantics": "esp32_camera_first_dma_buffer_since_boot",
        "x-jpeg-ready-timestamp-us": "140000",
        "x-jpeg-ready-timestamp-semantics": "camera_frame_buffer_available",
        "x-device-send-start-timestamp-us": "145000",
        "x-tof-timestamp-us": "110000",
        "x-tof-timestamp-semantics": "sensor_read_complete",
        "x-tof-minus-capture-us": "10000",
        "x-tof-valid": "true",
        "x-tof-range-mm": "750",
        "x-tof-status": "VALID",
        "x-tof-range-status-code": "0",
        "x-width": "8",
        "x-height": "8",
        "x-jpeg-quality": "10",
    }


class MeasureE2eLatencyTest(unittest.TestCase):
    def test_cross_clock_values_are_converted_from_microseconds_to_milliseconds(self):
        sync = ClockSync(
            sample_id=2,
            host_midpoint_us=1_000_000.0,
            device_midpoint_us=100_000.0,
            device_minus_host_us=-900_000.0,
            round_trip_us=2_000.0,
            error_bound_us=1_000.0,
            sequence_id="device-boot",
            clock_domain="esp32_boot_monotonic:device-boot",
            method="udp_midpoint_port_3333",
        )
        row = frame_row(
            frame_headers(),
            jpeg_bytes(),
            host_read_start_ns=1_020_000_000,
            host_jpeg_complete_ns=1_030_000_000,
            sync=sync,
            pipeline=NoPipeline(),
            connection_index=1,
        )

        self.assertAlmostEqual(row["capture_to_host_read_start_ms"], 20.0)
        self.assertAlmostEqual(row["capture_to_host_jpeg_complete_ms"], 30.0)
        self.assertAlmostEqual(row["device_capture_to_jpeg_ready_ms"], 40.0)
        self.assertAlmostEqual(row["tof_minus_capture_us"], 10_000)

    def test_summary_reports_jitter_skew_gaps_and_clock_uncertainty(self):
        rows = []
        for index, complete_ns in enumerate(
            (1_000_000_000, 1_040_000_000, 1_100_000_000)
        ):
            rows.append(
                {
                    "pipeline_identity": "NOT_CONFIGURED",
                    "stream_connection_index": 1,
                    "sequence_id": "device-boot",
                    "frame_sequence": (0, 1, 3)[index],
                    "capture_timestamp_us": (100_000, 140_000, 200_000)[index],
                    "host_jpeg_complete_monotonic_ns": complete_ns,
                    "tof_minus_capture_us": (-10_000, 20_000, -30_000)[index],
                    "tof_valid": True,
                    "clock_sync_rtt_us": 2_000.0,
                    "clock_sync_error_bound_us": 1_000.0,
                }
            )

        summary = summarize(rows, [], reconnects=0, errors=[])

        self.assertEqual(summary["host_interarrival_ms"]["p50"], 40.0)
        self.assertEqual(summary["absolute_tof_capture_skew_ms"]["max"], 30.0)
        self.assertEqual(summary["sequence_gap_event_count"], 1)
        self.assertEqual(summary["sequence_gap_total_frames"], 1)
        self.assertEqual(summary["clock_sync_error_bound_ms"]["p50"], 1.0)


if __name__ == "__main__":
    unittest.main()
