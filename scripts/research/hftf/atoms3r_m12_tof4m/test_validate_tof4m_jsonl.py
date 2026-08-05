import unittest

from validate_tof4m_jsonl import EVENT_SCHEMA, SAMPLE_SCHEMA, validate_rows


def sample(index: int, timestamp_ns: int, **overrides):
    row = {
        "schema": SAMPLE_SCHEMA,
        "firmware_version": "atoms3r_m12_tof4m_r0",
        "sequence_id": "device-boot",
        "sample_index": index,
        "timestamp_ns": timestamp_ns,
        "timestamp_semantics": "sensor_read_complete",
        "clock_domain": "esp32_boot_monotonic:device-boot",
        "sensor_id": "m5stack_unit_tof4m_vl53l1x",
        "measurement_status": "VALID",
        "timeout": False,
        "range_status_code": 0,
        "range_mm": 1250,
        "range_m": 1.25,
        "peak_signal_rate_mcps": 2.5,
        "ambient_rate_mcps": 0.2,
    }
    row.update(overrides)
    return row


class ValidateTof4mJsonlTest(unittest.TestCase):
    def test_accepts_events_and_monotonic_samples(self):
        rows = [
            {
                "schema": EVENT_SCHEMA,
                "sequence_id": "device-boot",
                "timestamp_ns": 1000,
                "clock_domain": "esp32_boot_monotonic:device-boot",
                "event": "boot",
                "status": "READY_FOR_TOF_INIT",
            },
            sample(0, 2000),
            sample(
                1,
                3000,
                measurement_status="INVALID_SENSOR_STATUS",
                range_status_code=2,
                range_m=None,
            ),
        ]
        result = validate_rows(rows)
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["event_count"], 1)

    def test_rejects_non_monotonic_timestamp(self):
        with self.assertRaisesRegex(ValueError, "timestamp is not strictly increasing"):
            validate_rows([sample(0, 2000), sample(1, 2000)])

    def test_rejects_missing_sample_index(self):
        with self.assertRaisesRegex(ValueError, "expected sample_index 1"):
            validate_rows([sample(0, 2000), sample(2, 3000)])

    def test_rejects_valid_range_m_disagreement(self):
        with self.assertRaisesRegex(ValueError, "range_m does not match range_mm"):
            validate_rows([sample(0, 2000, range_m=1.5)])

    def test_rejects_invalid_row_with_metric_range(self):
        with self.assertRaisesRegex(ValueError, "invalid row must use null range_m"):
            validate_rows(
                [
                    sample(
                        0,
                        2000,
                        measurement_status="INVALID_RANGE",
                        range_mm=2,
                        range_m=0.002,
                    )
                ]
            )


if __name__ == "__main__":
    unittest.main()
