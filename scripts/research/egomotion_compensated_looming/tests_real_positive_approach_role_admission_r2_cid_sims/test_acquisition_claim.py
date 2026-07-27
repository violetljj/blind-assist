"""Focused offline tests for R2 selection, exclusive claim, and acquisition."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import (
    acquire,
    bootstrap_claim,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture_zip(*, extra_member: str | None = None) -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("floor3_1/pose.txt", "0 0 0 0 0 0 0 1\n")
        archive.writestr("floor3_1/color/000000.png", b"RGB_NOT_READ")
        archive.writestr("floor3_1/depth/000000.png", b"DEPTH_NOT_READ")
        if extra_member is not None:
            archive.writestr(extra_member, b"x")
    return stream.getvalue()


def _documents(root: Path) -> tuple[dict[str, Path], dict[str, str], str]:
    descriptor = json.dumps(
        {
            "candidate_id": bootstrap_claim.CANDIDATE_ID,
            "sequence_id": bootstrap_claim.SEQUENCE_ID,
            "file_id": bootstrap_claim.OFFICIAL_FILE_ID,
            "official_payload_url": bootstrap_claim.OFFICIAL_PAYLOAD_URL,
            "bytes": bootstrap_claim.OFFICIAL_PAYLOAD_BYTES,
            "official_md5": bootstrap_claim.OFFICIAL_PAYLOAD_MD5,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    descriptor_sha256 = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()
    authority = {
        "protocol_id": bootstrap_claim.PROTOCOL_ID,
        "candidate_count": 1,
        "candidate": {
            "candidate_id": bootstrap_claim.CANDIDATE_ID,
            "sequence_id": bootstrap_claim.SEQUENCE_ID,
            "file_id": bootstrap_claim.OFFICIAL_FILE_ID,
            "official_payload_url": bootstrap_claim.OFFICIAL_PAYLOAD_URL,
            "bytes": bootstrap_claim.OFFICIAL_PAYLOAD_BYTES,
            "official_md5": bootstrap_claim.OFFICIAL_PAYLOAD_MD5,
        },
        "selection_authority": {
            "eligible_sequence_ids": list(
                reversed(bootstrap_claim.FROZEN_RUN_UNIVERSE)
            ),
            "deterministic_selection_rule": bootstrap_claim.SELECTION_RULE,
            "selected": bootstrap_claim.SEQUENCE_ID,
            "outcome_blind": True,
        },
        "identity": {
            "source_descriptor_canonical_json": descriptor,
            "source_descriptor_sha256": descriptor_sha256,
        },
    }
    values: dict[str, object] = {
        "contract": {"protocol_id": bootstrap_claim.PROTOCOL_ID},
        "source_authority": authority,
        "burned_manifest": {"protocol_id": bootstrap_claim.PROTOCOL_ID},
    }
    paths = {
        name: root / f"{name}.json"
        for name in (
            "contract",
            "source_authority",
            "burned_manifest",
            "implementation_lock",
        )
    }
    for name, value in values.items():
        _write_json(paths[name], value)
    _write_json(
        paths["implementation_lock"],
        {
            "protocol_id": bootstrap_claim.PROTOCOL_ID,
            "contract_sha256": bootstrap_claim.sha256_file(paths["contract"]),
            "source_authority_sha256": bootstrap_claim.sha256_file(
                paths["source_authority"]
            ),
            "burned_manifest_sha256": bootstrap_claim.sha256_file(
                paths["burned_manifest"]
            ),
            "source_descriptor_sha256": descriptor_sha256,
        },
    )
    hashes = {
        name: bootstrap_claim.sha256_file(path)
        for name, path in paths.items()
    }
    return paths, hashes, descriptor_sha256


def _claim(
    root: Path,
) -> tuple[dict[str, Path], Path]:
    paths, hashes, descriptor_sha256 = _documents(root)
    claim_path = root / "claim.json"
    bootstrap_claim.create_claim(
        paths["contract"],
        paths["source_authority"],
        paths["burned_manifest"],
        paths["implementation_lock"],
        claim_path,
        expected_hashes=hashes,
        expected_source_descriptor_sha256=descriptor_sha256,
        claim_created_by_runner_only=True,
        verify_implementation_files=False,
    )
    return paths, claim_path


class _FakeResponse:
    status = 200

    def __init__(
        self,
        payload: bytes,
        *,
        final_url: str = acquire.OFFICIAL_PAYLOAD_URL,
    ):
        self._stream = BytesIO(payload)
        self._final_url = final_url

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def geturl(self) -> str:
        return self._final_url

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None


class AcquisitionClaimTest(unittest.TestCase):
    def test_selection_is_lexicographic_over_exact_frozen_universe(self) -> None:
        self.assertEqual(
            bootstrap_claim.select_frozen_candidate(
                ("floor3_3", "floor3_1", "floor3_2")
            ),
            "floor3_1",
        )
        with self.assertRaisesRegex(ValueError, "FROZEN_RUN_UNIVERSE"):
            bootstrap_claim.select_frozen_candidate(
                ("floor3_1", "floor3_2")
            )

    def test_claim_is_exactly_bound_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, hashes, descriptor_sha256 = _documents(root)
            claim_path = root / "claim.json"
            claim = bootstrap_claim.create_claim(
                paths["contract"],
                paths["source_authority"],
                paths["burned_manifest"],
                paths["implementation_lock"],
                claim_path,
                expected_hashes=hashes,
                expected_source_descriptor_sha256=descriptor_sha256,
                claim_created_by_runner_only=True,
                verify_implementation_files=False,
            )
            self.assertEqual(
                claim["official_file_id"],
                "c595882daafe788a29d687872cc1fc2a",
            )
            self.assertEqual(claim["official_payload_bytes"], 2_211_008_069)
            self.assertEqual(
                claim["official_payload_md5"],
                "585d38855ad7d04817991cdbbb72016b",
            )
            with self.assertRaises(FileExistsError):
                bootstrap_claim.create_claim(
                    paths["contract"],
                    paths["source_authority"],
                    paths["burned_manifest"],
                    paths["implementation_lock"],
                    claim_path,
                    expected_hashes=hashes,
                    expected_source_descriptor_sha256=descriptor_sha256,
                    claim_created_by_runner_only=True,
                    verify_implementation_files=False,
                )

    def test_single_get_streams_both_hashes_without_rgb_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, claim_path = _claim(root)
            payload = _fixture_zip()
            calls: list[object] = []

            def opener(request: object, timeout: int) -> _FakeResponse:
                calls.append((request, timeout))
                return _FakeResponse(payload)

            receipt = acquire.acquire_once(
                claim_path,
                paths["contract"],
                paths["source_authority"],
                paths["burned_manifest"],
                paths["implementation_lock"],
                root / acquire.ARCHIVE_NAME,
                root / "source_receipt.json",
                opener=opener,
                expected_payload_bytes=len(payload),
                expected_payload_md5=hashlib.md5(
                    payload, usedforsecurity=False
                ).hexdigest(),
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                receipt["archive_sha256"], hashlib.sha256(payload).hexdigest()
            )
            self.assertEqual(receipt["rgb_members_read"], 0)
            self.assertEqual(receipt["redirect_count"], 0)
            self.assertEqual(receipt["retry_count"], 0)

    def test_size_or_md5_mismatch_is_valid_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, claim_path = _claim(root)
            payload = _fixture_zip()
            with self.assertRaises(acquire.SourceAccessHold) as caught:
                acquire.acquire_once(
                    claim_path,
                    paths["contract"],
                    paths["source_authority"],
                    paths["burned_manifest"],
                    paths["implementation_lock"],
                    root / acquire.ARCHIVE_NAME,
                    root / "source_receipt.json",
                    opener=lambda request, timeout: _FakeResponse(payload),
                    expected_payload_bytes=len(payload) + 1,
                    expected_payload_md5=hashlib.md5(
                        payload, usedforsecurity=False
                    ).hexdigest(),
                )
            self.assertEqual(
                caught.exception.code,
                "R2_OFFICIAL_PAYLOAD_IDENTITY_MISMATCH",
            )
            self.assertEqual(caught.exception.receipt["terminal"], acquire.HOLD)
            self.assertEqual(caught.exception.receipt["retry_count"], 0)

    def test_unsafe_and_multi_root_archives_are_invalid(self) -> None:
        for extra_member, code in (
            ("../escape", "R2_ARCHIVE_UNSAFE_MEMBER"),
            ("floor3_2/pose.txt", "R2_ARCHIVE_MULTI_SEQUENCE_ROOT"),
        ):
            with self.subTest(extra_member=extra_member):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    paths, claim_path = _claim(root)
                    payload = _fixture_zip(extra_member=extra_member)
                    with self.assertRaisesRegex(ValueError, code):
                        acquire.acquire_once(
                            claim_path,
                            paths["contract"],
                            paths["source_authority"],
                            paths["burned_manifest"],
                            paths["implementation_lock"],
                            root / acquire.ARCHIVE_NAME,
                            root / "source_receipt.json",
                            opener=lambda request, timeout: _FakeResponse(
                                payload
                            ),
                            expected_payload_bytes=len(payload),
                            expected_payload_md5=hashlib.md5(
                                payload, usedforsecurity=False
                            ).hexdigest(),
                        )

    def test_any_redirect_is_rejected(self) -> None:
        handler = acquire._NoRedirectHandler()
        request = urllib.request.Request(acquire.OFFICIAL_PAYLOAD_URL)
        with self.assertRaisesRegex(ValueError, "R2_REDIRECT_FORBIDDEN"):
            handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                acquire.OFFICIAL_PAYLOAD_URL,
            )


if __name__ == "__main__":
    unittest.main()
