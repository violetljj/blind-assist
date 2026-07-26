from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_b0_r1 import (
    protocol,
    validator,
)


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        url: str,
        *,
        status: int = 200,
        content_type: str = "application/zip",
    ) -> None:
        self._payload = BytesIO(payload)
        self._url = url
        self.status = status
        self.headers = {
            "Content-Length": str(len(payload)),
            "Content-Type": content_type,
        }

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._payload.read(size)

    def geturl(self) -> str:
        return self._url


class FailingReadResponse(FakeResponse):
    def read(self, size: int = -1) -> bytes:
        raise ConnectionError("fixture transport interruption")


def zip_payload(
    *,
    rgb: str = "0 a\n10 b\n20 c\n",
    depth: str = "1 a\n11 b\n21 c\n",
    pose: str = "2 1 2 3 4 5 6 7\n12 9 8 7\n22 6 5 4\n",
    extra: dict[str, bytes] | None = None,
) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("sequence/rgb.txt", rgb)
        archive.writestr("sequence/depth.txt", depth)
        archive.writestr("sequence/groundtruth.txt", pose)
        archive.writestr("sequence/rgb/0.png", b"not-decoded-rgb")
        archive.writestr("sequence/depth/0.png", b"not-decoded-depth")
        for name, payload in (extra or {}).items():
            archive.writestr(name, payload)
    return buffer.getvalue()


class PhaseBBonnB0R1Test(unittest.TestCase):
    def test_fixed_cohort_and_urls(self) -> None:
        self.assertEqual(len(protocol.SEQUENCE_IDS), 6)
        self.assertEqual(len(set(protocol.SEQUENCE_IDS)), 6)
        for sequence in protocol.SEQUENCE_IDS:
            self.assertEqual(
                protocol.URL_TEMPLATE.format(sequence_id=sequence),
                (
                    "https://www.ipb.uni-bonn.de/html/projects/"
                    f"rgbd_dynamic2019/{sequence}.zip"
                ),
            )

    def test_acquire_streams_hashes_and_atomically_finishes(self) -> None:
        payload = zip_payload()
        sequence = protocol.SEQUENCE_IDS[0]
        url = protocol.URL_TEMPLATE.format(sequence_id=sequence)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / f"{sequence}.zip"
            transport, attempts = protocol.acquire_archive(
                sequence,
                destination,
                opener=lambda requested, timeout: FakeResponse(
                    payload, requested
                ),
            )
            self.assertTrue(destination.exists())
            self.assertFalse(destination.with_suffix(".zip.part").exists())
            self.assertEqual(transport["archive_bytes"], len(payload))
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["outcome"], "COMPLETE")
            self.assertFalse(attempts[0]["range_resume_used"])
            self.assertEqual(attempts[0]["final_url"], url)

    def test_redirect_and_html_are_rejected_three_times(self) -> None:
        sequence = protocol.SEQUENCE_IDS[0]
        url = protocol.URL_TEMPLATE.format(sequence_id=sequence)
        calls = 0

        def redirect(_requested: str, _timeout: float) -> FakeResponse:
            nonlocal calls
            calls += 1
            return FakeResponse(b"<html>error</html>", url + "?mirror=1")

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                RuntimeError, "TRANSPORT_ATTEMPTS_EXHAUSTED"
            ):
                protocol.acquire_archive(
                    sequence,
                    Path(temp) / f"{sequence}.zip",
                    opener=redirect,
                )
        self.assertEqual(calls, 3)

    def test_part_is_truncated_before_each_network_attempt_and_ledger_is_immediate(
        self,
    ) -> None:
        payload = zip_payload()
        sequence = protocol.SEQUENCE_IDS[0]
        url = protocol.URL_TEMPLATE.format(sequence_id=sequence)
        observed_part_sizes: list[int] = []
        persisted: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / f"{sequence}.zip"
            part = destination.with_suffix(".zip.part")
            part.write_bytes(b"stale-partial-data")
            calls = 0

            def opener(_requested: str, _timeout: float) -> FakeResponse:
                nonlocal calls
                calls += 1
                observed_part_sizes.append(part.stat().st_size)
                if calls == 1:
                    return FailingReadResponse(payload, url)
                return FakeResponse(payload, url)

            _transport, attempts = protocol.acquire_archive(
                sequence,
                destination,
                opener=opener,
                on_attempt=lambda attempt: persisted.append(dict(attempt)),
            )
        self.assertEqual(observed_part_sizes, [0, 0])
        self.assertEqual(
            [attempt["outcome"] for attempt in attempts],
            ["FAILED", "COMPLETE"],
        )
        self.assertEqual(persisted, attempts)
        self.assertTrue(
            all(
                attempt["part_truncated_before_network"]
                for attempt in attempts
            )
        )

    def test_failed_attempt_ledger_persistence_error_is_fail_stop(self) -> None:
        sequence = protocol.SEQUENCE_IDS[0]
        url = protocol.URL_TEMPLATE.format(sequence_id=sequence)

        def fail_persistence(_attempt: dict[str, object]) -> None:
            raise OSError("ledger persistence fixture")

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(protocol.DurableLedgerError):
                protocol.acquire_archive(
                    sequence,
                    Path(temp) / f"{sequence}.zip",
                    opener=lambda requested, timeout: FailingReadResponse(
                        b"PKfixture", url
                    ),
                    on_attempt=fail_persistence,
                )

    def test_attempt_override_and_sequence_override_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                ValueError, "SEQUENCE_OUTSIDE_FROZEN_COHORT"
            ):
                protocol.acquire_archive(
                    "other",
                    Path(temp) / "other.zip",
                    opener=lambda url, timeout: None,
                )
            with self.assertRaisesRegex(
                ValueError, "ATTEMPT_COUNT_OVERRIDE_FORBIDDEN"
            ):
                protocol.acquire_archive(
                    protocol.SEQUENCE_IDS[0],
                    Path(temp) / "x.zip",
                    opener=lambda url, timeout: None,
                    maximum_attempts=1,
                )

    def test_archive_inventory_crc_timestamp_and_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive_path = Path(temp) / "fixture.zip"
            archive_path.write_bytes(zip_payload())
            result = protocol.inspect_archive("fixture", archive_path)
        self.assertEqual(result["status"], "EVALUABLE_ARCHIVE_AUTHORITY")
        self.assertEqual(result["timestamp_firewall"], "PASS_FIRST_TOKEN_ONLY")
        self.assertEqual(result["pose_tokens_parsed_after_first"], 0)
        self.assertEqual(result["window_count"], 1)
        self.assertEqual(
            result["windows"],
            [
                {
                    "window_rank": 0,
                    "start": "2",
                    "end": "12",
                    "interval": "HALF_OPEN",
                },
            ],
        )
        self.assertEqual(
            result["crc_only_stream"]["file_members_streamed"], 5
        )
        self.assertEqual(
            result["crc_only_stream"]["persisted_extracted_bytes"], 0
        )

    def test_short_tail_is_discarded_and_zero_window_retained(self) -> None:
        payload = zip_payload(
            rgb="0 a\n9 b\n",
            depth="0 a\n9 b\n",
            pose="0 1 2\n9 3 4\n",
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "short.zip"
            path.write_bytes(payload)
            result = protocol.inspect_archive("fixture", path)
        self.assertEqual(result["window_count"], 0)
        self.assertEqual(result["windows"], [])

    def test_nonfinite_and_nonmonotonic_timestamp_rejected(self) -> None:
        cases = ("0 a\nNaN b\n", "1 a\n1 b\n")
        for rgb in cases:
            with self.subTest(rgb=rgb):
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / "bad.zip"
                    path.write_bytes(zip_payload(rgb=rgb))
                    with self.assertRaises(ValueError):
                        protocol.inspect_archive("fixture", path)

    def test_pose_tokens_after_timestamp_are_never_parsed(self) -> None:
        payload = zip_payload(
            pose="0 THIS_IS_NOT_A_NUMBER\n10 STILL_NOT_NUMERIC\n20 !!!\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pose-firewall.zip"
            path.write_bytes(payload)
            result = protocol.inspect_archive("fixture", path)
        self.assertEqual(result["pose_tokens_parsed_after_first"], 0)
        self.assertEqual(result["timestamps"]["pose"]["count"], 3)

    def test_decimal_normalization_and_exact_window_boundary(self) -> None:
        payload = zip_payload(
            rgb="-0 ignored\n1e1 ignored\n",
            depth="+0 ignored\n10.0 ignored\n",
            pose=".0 ignored\n10 ignored\n",
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "decimal.zip"
            path.write_bytes(payload)
            result = protocol.inspect_archive("fixture", path)
        self.assertEqual(result["timestamps"]["rgb"]["first"], "0")
        self.assertEqual(result["timestamps"]["rgb"]["last"], "10")
        self.assertEqual(result["window_count"], 1)

    def test_duplicate_required_member_rejected(self) -> None:
        payload = zip_payload(extra={"other/rgb.txt": b"0 a\n10 b\n"})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.zip"
            path.write_bytes(payload)
            with self.assertRaisesRegex(
                ValueError, "REQUIRED_MEMBER_CARDINALITY"
            ):
                protocol.inspect_archive("fixture", path)

    def test_path_traversal_and_backslash_rejected(self) -> None:
        for name in ("../escape.txt",):
            with self.subTest(name=name):
                payload = zip_payload(extra={name: b"x"})
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / "unsafe.zip"
                    path.write_bytes(payload)
                    with self.assertRaises(ValueError):
                        protocol.inspect_archive("fixture", path)
        with self.assertRaises(ValueError):
            protocol._normalized_member_name("sequence\\escape.txt")
        with self.assertRaises(ValueError):
            protocol._normalized_member_name("root/C:/escape.txt")

    def test_casefold_member_collision_rejected(self) -> None:
        payload = zip_payload(extra={"sequence/RGB/0.png": b"collision"})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "casefold.zip"
            path.write_bytes(payload)
            with self.assertRaisesRegex(
                ValueError, "DUPLICATE_NORMALIZED_ZIP_MEMBER"
            ):
                protocol.inspect_archive("fixture", path)

    def test_crc_corruption_is_rejected(self) -> None:
        payload = bytearray(zip_payload())
        marker = b"not-decoded-rgb"
        index = payload.find(marker)
        self.assertGreaterEqual(index, 0)
        payload[index] ^= 0x01
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "corrupt.zip"
            path.write_bytes(bytes(payload))
            with self.assertRaises(zipfile.BadZipFile):
                protocol.inspect_archive("fixture", path)

    def test_each_file_member_is_opened_once_per_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "single-stream.zip"
            path.write_bytes(zip_payload())
            original = zipfile.ZipFile.open
            opened: list[str] = []

            def counted(
                archive: zipfile.ZipFile,
                member: object,
                *args: object,
                **kwargs: object,
            ) -> object:
                opened.append(
                    member.filename
                    if isinstance(member, zipfile.ZipInfo)
                    else str(member)
                )
                return original(archive, member, *args, **kwargs)

            with mock.patch.object(zipfile.ZipFile, "open", new=counted):
                result = protocol.inspect_archive("fixture", path)
        self.assertEqual(len(opened), result["file_member_count"])
        self.assertEqual(len(opened), len(set(opened)))

    def test_claim_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = protocol.canonical_paths(root)
            paths["implementation_lock"].parent.mkdir(
                parents=True, exist_ok=True
            )
            paths["implementation_lock"].write_text("{}\n", encoding="utf-8")
            first = protocol.create_run_claim(root, ["fixture"])
            self.assertEqual(first["network_operations_before_claim"], 0)
            self.assertEqual(
                first["canonical_run_claim"], str(paths["run_claim"])
            )
            self.assertEqual(
                first["pre_r1_head_disclosure"],
                {
                    "request_count": 6,
                    "response_body_bytes": 0,
                    "reported_content_length_total_bytes": 2262988443,
                    "authority": (
                        "NON_AUTHORITATIVE_TRANSPORT_DISCOVERY_"
                        "EXCLUDED_FROM_ALL_GATES"
                    ),
                },
            )
            with self.assertRaises(FileExistsError):
                protocol.create_run_claim(root, ["fixture"])

    def test_runner_has_no_path_or_network_probe_override(self) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "run_phase_b_bonn_formal_entry_b0_r1.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "--output",
            "--archive",
            "--url",
            "--sequence",
            "--head",
            "--probe",
        ):
            self.assertNotIn(forbidden, runner)
        self.assertLess(
            runner.index("create_run_claim("),
            runner.index("run_b0("),
        )

    def test_receipt_schema_declares_six_item_denominator(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "rcle_phase_b_bonn_b0_r1_receipt.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["window_denominator"]["minItems"], 6
        )
        self.assertEqual(
            schema["properties"]["phase_b_metrics_authorized"]["const"],
            False,
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            set(schema["properties"]),
            validator.RECEIPT_TOP_LEVEL_KEYS,
        )

    def test_implementation_lock_hashes_validate(self) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        lock = protocol.validate_implementation_lock(
            repo_root, protocol.canonical_paths(repo_root)
        )
        self.assertEqual(lock["offline_contract_tests_expected"], 22)
        self.assertTrue(lock["canonical_execution_authorized"])

    def test_independent_validator_recomputes_same_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "independent.zip"
            path.write_bytes(zip_payload())
            produced = protocol.inspect_archive("fixture", path)
            checked = validator._inspect("fixture", path)
        self.assertEqual(
            validator._comparison(produced),
            validator._comparison(checked),
        )

    def test_failed_attempt_validator_is_fail_closed(self) -> None:
        url = protocol.URL_TEMPLATE.format(
            sequence_id=protocol.SEQUENCE_IDS[0]
        )
        valid = {
            "error_type": "RetryableTransportError",
            "error": "TRANSPORT_OPEN_FAILED",
            "bytes_written": 0,
            "status": None,
            "final_url": None,
            "headers": {},
        }
        self.assertTrue(validator._failed_attempt_contract(valid, url))
        for mutation in (
            {"error_type": None},
            {"error": ""},
            {"error": "UNFROZEN_FAILURE"},
            {"bytes_written": -1},
        ):
            self.assertFalse(
                validator._failed_attempt_contract(
                    {**valid, **mutation}, url
                )
            )


if __name__ == "__main__":
    unittest.main()
