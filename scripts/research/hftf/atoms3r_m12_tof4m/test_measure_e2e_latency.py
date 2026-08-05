import unittest

import cv2
import numpy as np

from measure_e2e_latency import (
    ClockSync,
    FramePacket,
    LatestFrameReader,
    NoPipeline,
    finalize_device_attribution,
    frame_row,
    summarize,
)


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
        "x-frame-ready-interval-us": "50000",
        "x-frame-acquire-duration-us": "40000",
        "x-jpeg-metadata-prepare-duration-us": "5000",
        "x-previous-response-write-valid": "true",
        "x-previous-frame-sequence": "6",
        "x-previous-response-write-duration-us": "12000",
        "x-tof-timestamp-us": "110000",
        "x-tof-timestamp-semantics": "sensor_read_complete",
        "x-tof-minus-capture-us": "10000",
        "x-tof-age-at-jpeg-ready-us": "30000",
        "x-tof-during-acquire": "true",
        "x-tof-updates-during-acquire": "1",
        "x-tof-updates-since-previous-frame": "2",
        "x-tof-sampling-enabled": "true",
        "x-tof-valid": "true",
        "x-tof-range-mm": "750",
        "x-tof-status": "VALID",
        "x-tof-range-status-code": "0",
        "x-jpeg-size-bytes": str(len(jpeg_bytes())),
        "x-width": "8",
        "x-height": "8",
        "x-jpeg-quality": "10",
        "x-auto-exposure": "true",
        "x-camera-psram-dma-enabled": "false",
        "x-stream-tcp-nodelay": "false",
        "x-stream-preamble-coalesced": "false",
        "x-exposure-value": "321",
        "x-wifi-rssi-dbm": "-37",
        "x-free-heap-bytes": "150000",
    }


class MeasureE2eLatencyTest(unittest.TestCase):
    def test_latest_frame_reader_overwrites_stale_packet(self):
        reader = LatestFrameReader("http://unused")
        first = FramePacket({}, b"first", 1, 2, 3, 1)
        latest = FramePacket({}, b"latest", 4, 5, 6, 1)

        reader.offer(first)
        reader.offer(latest)

        self.assertEqual(reader.latest_queue_overwrite_count, 1)
        self.assertEqual(reader.get(timeout_s=0.01), latest)

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
            host_first_byte_received_ns=1_022_000_000,
            host_jpeg_complete_ns=1_030_000_000,
            sync=sync,
            pipeline=NoPipeline(),
            connection_index=1,
        )

        self.assertAlmostEqual(row["capture_to_host_read_start_ms"], 20.0)
        self.assertAlmostEqual(row["capture_to_host_jpeg_complete_ms"], 30.0)
        self.assertAlmostEqual(row["device_capture_to_jpeg_ready_ms"], 40.0)
        self.assertAlmostEqual(row["tof_minus_capture_us"], 10_000)
        self.assertTrue(row["tof_sampling_enabled"])
        self.assertFalse(row["camera_psram_dma_enabled"])
        self.assertEqual(row["device_capture_to_fb_return_us"], 40_000)
        self.assertEqual(row["device_capture_minus_acquire_start_us"], 0)
        self.assertIsNone(row["pipeline_num_threads"])
        self.assertFalse(row["stream_tcp_nodelay"])
        self.assertFalse(row["stream_preamble_coalesced"])

    def test_absent_tof_timestamp_makes_skew_not_evaluable(self):
        headers = frame_headers()
        headers["x-tof-timestamp-us"] = "0"
        headers["x-tof-minus-capture-us"] = "-100000"
        headers["x-tof-sampling-enabled"] = "false"
        headers["x-tof-valid"] = "false"
        row = frame_row(
            headers,
            jpeg_bytes(),
            host_read_start_ns=1_020_000_000,
            host_first_byte_received_ns=1_022_000_000,
            host_jpeg_complete_ns=1_030_000_000,
            sync=ClockSync(
                sample_id=1,
                host_midpoint_us=1_000_000.0,
                device_midpoint_us=100_000.0,
                device_minus_host_us=-900_000.0,
                round_trip_us=2_000.0,
                error_bound_us=1_000.0,
                sequence_id="device-boot",
                clock_domain="esp32_boot_monotonic:device-boot",
                method="udp_midpoint_port_3333",
            ),
            pipeline=NoPipeline(),
            connection_index=1,
        )

        self.assertIsNone(row["tof_minus_capture_us"])
        self.assertFalse(row["tof_sampling_enabled"])

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

        summary = summarize(
            rows,
            [],
            reconnects=0,
            latest_queue_overwrites=0,
            errors=[],
            slow_frame_contract={},
        )

        self.assertEqual(summary["host_interarrival_ms"]["p50"], 40.0)
        self.assertEqual(summary["absolute_tof_capture_skew_ms"]["max"], 30.0)
        self.assertEqual(summary["sequence_gap_event_count"], 1)
        self.assertEqual(summary["sequence_gap_total_frames"], 1)
        self.assertEqual(summary["host_latest_queue_overwrite_count"], 0)
        self.assertEqual(summary["clock_sync_error_bound_ms"]["p50"], 1.0)
        self.assertTrue(summary["run_accepted"])
        self.assertEqual(summary["run_acceptance_failures"], [])

        rejected = summarize(
            rows,
            [],
            reconnects=1,
            latest_queue_overwrites=0,
            errors=["stream:EOFError"],
            slow_frame_contract={},
        )
        self.assertFalse(rejected["run_accepted"])
        self.assertEqual(
            rejected["run_acceptance_failures"],
            ["STREAM_RECONNECTS_PRESENT", "ERRORS_PRESENT"],
        )

    def test_slow_frame_contract_uses_frozen_median_mad_or_ratio_rule(self):
        rows = [
            {
                "sequence_id": "boot",
                "frame_sequence": index,
                "device_frame_ready_interval_us": interval,
                "reported_previous_response_write_valid": index > 0,
                "reported_previous_frame_sequence": index - 1,
                "reported_previous_response_write_duration_us": 10_000 + index,
                "device_response_write_duration_us": None,
            }
            for index, interval in enumerate((0, 40_000, 41_000, 42_000, 100_000))
        ]

        contract = finalize_device_attribution(rows)

        self.assertEqual(contract["median_interval_us"], 41_500.0)
        self.assertFalse(rows[3]["slow_frame"])
        self.assertTrue(rows[4]["slow_frame"])
        self.assertEqual(rows[3]["device_response_write_duration_us"], 10_004)


if __name__ == "__main__":
    unittest.main()
