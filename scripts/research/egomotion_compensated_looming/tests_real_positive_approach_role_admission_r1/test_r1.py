from __future__ import annotations

import ast
from io import BytesIO
import hashlib
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r1 import (
    acquire,
    bootstrap_claim,
    formal_runner,
    pilot,
    producer,
    validator,
)


DESCRIPTOR = (
    '{"candidate_id":"ETH3D_SLAM_SOFA_3_RGBD",'
    '"dataset_page_sha256":"e66b2664b64c13bed891e9c43f7cf142686f84ab4aa902b6551eac3df7bdcea9",'
    '"license":"CC_BY_NC_SA_4_0",'
    '"official_dataset_page":"https://www.eth3d.net/slam_datasets",'
    '"official_format_documentation":"https://www.eth3d.net/slam_documentation",'
    '"official_payload_url":"https://www.eth3d.net/data/slam/datasets/sofa_3_rgbd.zip",'
    '"selection":"single frozen real RGB-D training sequence; no sofa_4, alternate ETH3D sequence, '
    'mirror, archive variant, retry, or replacement after claim"}'
)


def json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def png_bytes(value: int = 10000) -> bytes:
    depth = np.full((48, 64), value, dtype=np.uint16)
    stream = BytesIO()
    Image.fromarray(depth).save(stream, format="PNG")
    return stream.getvalue()


def fixture_zip(
    *,
    frame_count: int = 102,
    omit_depth_member_index: int | None = None,
    unsafe_member: str | None = None,
) -> bytes:
    stream = BytesIO()
    timestamps = [index * 0.1 for index in range(frame_count)]
    associated = []
    depth_rows = []
    rgb_rows = []
    poses = []
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sofa_3/calibration.txt", "50 50 31.5 23.5\n")
        for index, timestamp in enumerate(timestamps):
            rgb_name = f"rgb/{index:06d}.png"
            depth_name = f"depth/{index:06d}.png"
            associated.append(
                f"{timestamp:.6f} {rgb_name} {timestamp:.6f} {depth_name}\n"
            )
            depth_rows.append(f"{timestamp:.6f} {depth_name}\n")
            rgb_rows.append(f"{timestamp:.6f} {rgb_name}\n")
            poses.append(
                f"{timestamp:.6f} 0 0 {index * 0.02:.6f} 0 0 0 1\n"
            )
            if index != omit_depth_member_index:
                archive.writestr(f"sofa_3/{depth_name}", png_bytes())
            # Deliberately invalid image bytes: any RGB decode would fail.
            archive.writestr(f"sofa_3/{rgb_name}", b"RGB_MUST_NOT_BE_DECODED")
        archive.writestr("sofa_3/associated.txt", "".join(associated))
        archive.writestr("sofa_3/depth.txt", "".join(depth_rows))
        archive.writestr("sofa_3/rgb.txt", "".join(rgb_rows))
        archive.writestr("sofa_3/groundtruth.txt", "".join(poses))
        if unsafe_member is not None:
            archive.writestr(unsafe_member, b"unsafe")
    return stream.getvalue()


def package_documents(root: Path) -> tuple[dict[str, Path], dict[str, str]]:
    authority = {
        "schema_version": "test",
        "protocol_id": producer.PROTOCOL_ID,
        "candidate_count": 1,
        "candidate": {
            "candidate_id": producer.CANDIDATE_ID,
            "official_payload_url": acquire.OFFICIAL_PAYLOAD_URL,
        },
        "identity": {
            "source_descriptor_canonical_json": DESCRIPTOR,
            "source_descriptor_sha256": hashlib.sha256(
                DESCRIPTOR.encode("utf-8")
            ).hexdigest(),
        },
        "ancestry_and_independence": {
            "ancestry": producer.EXPECTED_ANCESTRY,
            "independence_group": producer.INDEPENDENCE_GROUP,
            "required_non_overlap": sorted(producer.REQUIRED_BURNED_GROUPS),
            "reuse_policy_after_access": producer.REUSE_POLICY,
        },
    }
    manifest = {
        "schema_version": "test",
        "protocol_id": producer.PROTOCOL_ID,
        "r1_candidate_must_not_descend_from": sorted(
            producer.REQUIRED_BURNED_GROUPS
        ),
    }
    contract = {
        "schema_version": "test",
        "protocol_id": producer.PROTOCOL_ID,
        "legal_terminals": [producer.ADMITTED, producer.HOLD],
        "source_selection": {
            "candidate_id": producer.CANDIDATE_ID,
            "candidate_count": 1,
            "no_replacement": True,
            "terminal_on_source_failure": producer.HOLD,
        },
        "geometry_rule": {
            "gates": {
                "candidate_pair_coverage_min": "0.80",
                "median_signed_radial_expansion_per_s_min": "0.05",
                "median_radial_expansion_positive_fraction_min": "0.75",
                "minimum_evaluable_pairs": 8,
            }
        },
        "access_contract": {
            "algorithm_outcome_access": False,
            "forbidden_payload_reads": [
                "RGB image pixels",
                "RCLE algorithm output",
            ],
        },
        "performance_qualification_gate": {
            "only_unlock_terminal": producer.ADMITTED,
        },
    }
    paths = {
        "contract": root / "contract.json",
        "source_authority": root / "source_authority.json",
        "burned_manifest": root / "burned_manifest.json",
        "implementation_lock": root / "implementation_lock.json",
    }
    json_write(paths["contract"], contract)
    json_write(paths["source_authority"], authority)
    json_write(paths["burned_manifest"], manifest)
    json_write(
        paths["implementation_lock"],
        {
            "schema_version": "test",
            "protocol_id": producer.PROTOCOL_ID,
            "contract_sha256": bootstrap_claim.sha256_file(paths["contract"]),
            "source_authority_sha256": bootstrap_claim.sha256_file(
                paths["source_authority"]
            ),
            "burned_manifest_sha256": bootstrap_claim.sha256_file(
                paths["burned_manifest"]
            ),
            "source_descriptor_sha256": bootstrap_claim.SOURCE_DESCRIPTOR_SHA256,
        },
    )
    hashes = {name: bootstrap_claim.sha256_file(path) for name, path in paths.items()}
    return paths, hashes


class FakeResponse:
    status = 200

    def __init__(self, payload: bytes, *, final_url: str = acquire.OFFICIAL_PAYLOAD_URL):
        self.stream = BytesIO(payload)
        self.final_url = final_url

    def read(self, size: int = -1) -> bytes:
        return self.stream.read(size)

    def geturl(self) -> str:
        return self.final_url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        pass


class AdmissionR1Test(unittest.TestCase):
    def test_descriptor_and_claim_are_hash_bound_and_exclusive(self) -> None:
        self.assertEqual(
            hashlib.sha256(DESCRIPTOR.encode("utf-8")).hexdigest(),
            bootstrap_claim.SOURCE_DESCRIPTOR_SHA256,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, hashes = package_documents(root)
            claim = root / "claim.json"
            value = bootstrap_claim.create_claim(
                paths["contract"],
                paths["source_authority"],
                paths["burned_manifest"],
                paths["implementation_lock"],
                claim,
                expected_hashes=hashes,
                claim_created_by_runner_only=True,
                verify_implementation_files=False,
            )
            self.assertTrue(value["claim_created_by_runner_only"])
            self.assertEqual(
                set(value["bindings"]),
                {
                    "contract",
                    "source_authority",
                    "burned_manifest",
                    "implementation_lock",
                },
            )
            with self.assertRaises(FileExistsError):
                bootstrap_claim.create_claim(
                    paths["contract"],
                    paths["source_authority"],
                    paths["burned_manifest"],
                    paths["implementation_lock"],
                    claim,
                    expected_hashes=hashes,
                    claim_created_by_runner_only=True,
                    verify_implementation_files=False,
                )

    def test_formal_runner_synthetic_zip_validates_without_rgb_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, hashes = package_documents(root)
            calls = []

            def opener(request: object, timeout: int) -> FakeResponse:
                calls.append((request, timeout))
                return FakeResponse(fixture_zip())

            run_dir = root / "run"
            terminal = formal_runner.run_formal(
                contract=paths["contract"],
                source_authority=paths["source_authority"],
                burned_manifest=paths["burned_manifest"],
                implementation_lock=paths["implementation_lock"],
                run_dir=run_dir,
                workers=1,
                expected_hashes=hashes,
                opener=opener,
                verify_implementation_files=False,
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(terminal["validation_terminal"], "VALID")
            self.assertEqual(
                terminal["scientific_terminal"],
                "REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID",
            )
            result = json.loads(
                (run_dir / "formal/result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(result["access"]["rgb_txt_content_read"])
            self.assertFalse(result["access"]["rgb_pixels_read"])
            self.assertEqual(result["pair_record_count"], 99)
            self.assertEqual(result["admitted_window_count"], 1)
            progress = json.loads(
                (run_dir / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["status"], "SUCCEEDED")
            self.assertEqual(progress["completed_units"], progress["total_units"])
            self.assertTrue((run_dir / "SUCCESS.json").is_file())
            self.assertFalse((run_dir / "FAILURE.json").exists())

    def test_incomplete_ten_second_window_is_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, hashes = package_documents(root)

            def opener(request: object, timeout: int) -> FakeResponse:
                return FakeResponse(fixture_zip(frame_count=10))

            terminal = formal_runner.run_formal(
                contract=paths["contract"],
                source_authority=paths["source_authority"],
                burned_manifest=paths["burned_manifest"],
                implementation_lock=paths["implementation_lock"],
                run_dir=root / "run",
                expected_hashes=hashes,
                opener=opener,
                verify_implementation_files=False,
            )
            self.assertEqual(terminal["scientific_terminal"], producer.HOLD)
            result = json.loads(
                (root / "run/formal/result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(result["windows"][0]["window_complete"])

    def test_missing_depth_member_abstains_without_denominator_drop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, hashes = package_documents(root)

            def opener(request: object, timeout: int) -> FakeResponse:
                return FakeResponse(fixture_zip(omit_depth_member_index=10))

            terminal = formal_runner.run_formal(
                contract=paths["contract"],
                source_authority=paths["source_authority"],
                burned_manifest=paths["burned_manifest"],
                implementation_lock=paths["implementation_lock"],
                run_dir=root / "run",
                expected_hashes=hashes,
                opener=opener,
                verify_implementation_files=False,
            )
            self.assertEqual(terminal["validation_terminal"], "VALID")
            rows = [
                json.loads(line)
                for line in (root / "run/formal/pair_ledger.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 99)
            self.assertTrue(
                any(row.get("reason") == "ASSOCIATED_MEMBER_MISSING" for row in rows)
            )

    def test_unsafe_zip_member_is_invalid_not_source_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, hashes = package_documents(root)

            def opener(request: object, timeout: int) -> FakeResponse:
                return FakeResponse(fixture_zip(unsafe_member="../escape"))

            with self.assertRaisesRegex(ValueError, "R1_ARCHIVE_UNSAFE_MEMBER"):
                formal_runner.run_formal(
                    contract=paths["contract"],
                    source_authority=paths["source_authority"],
                    burned_manifest=paths["burned_manifest"],
                    implementation_lock=paths["implementation_lock"],
                    run_dir=root / "run",
                    expected_hashes=hashes,
                    opener=opener,
                    verify_implementation_files=False,
                )
            self.assertTrue((root / "run/FAILURE.json").is_file())
            self.assertFalse((root / "run/SUCCESS.json").exists())

    def test_multi_sequence_root_is_invalid_not_source_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, hashes = package_documents(root)

            def opener(request: object, timeout: int) -> FakeResponse:
                return FakeResponse(fixture_zip(unsafe_member="other/file.txt"))

            with self.assertRaisesRegex(ValueError, "MULTI_SEQUENCE_ROOT"):
                formal_runner.run_formal(
                    contract=paths["contract"],
                    source_authority=paths["source_authority"],
                    burned_manifest=paths["burned_manifest"],
                    implementation_lock=paths["implementation_lock"],
                    run_dir=root / "run",
                    expected_hashes=hashes,
                    opener=opener,
                    verify_implementation_files=False,
                )
            failure = json.loads(
                (root / "run/FAILURE.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                failure["scientific_terminal"], "INVALID_R1_EVIDENCE / INVALID"
            )
            self.assertEqual(failure["validation_terminal"], "INVALID")
            self.assertTrue(failure["claim_consumed"])

    def test_redirect_handler_rejects_unapproved_hop_before_following(self) -> None:
        handler = acquire._StrictOfficialRedirectHandler()
        request = urllib.request.Request(acquire.OFFICIAL_PAYLOAD_URL)
        with self.assertRaisesRegex(ValueError, "R1_OFFICIAL_GET_RESPONSE"):
            handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "https://invalid.example/sofa_3_rgbd.zip",
            )
        self.assertEqual(handler.redirect_chain, [])

    def test_implementation_lock_checks_every_required_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, _ = package_documents(root)
            locked_rows = []
            for relative in sorted(bootstrap_claim.REQUIRED_IMPLEMENTATION_PATHS):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"locked\n")
                locked_rows.append(
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(b"locked\n").hexdigest(),
                    }
                )
            json_write(
                paths["implementation_lock"],
                {
                    "schema_version": "test",
                    "protocol_id": producer.PROTOCOL_ID,
                    "contract_sha256": bootstrap_claim.sha256_file(
                        paths["contract"]
                    ),
                    "source_authority_sha256": bootstrap_claim.sha256_file(
                        paths["source_authority"]
                    ),
                    "burned_manifest_sha256": bootstrap_claim.sha256_file(
                        paths["burned_manifest"]
                    ),
                    "source_descriptor_sha256": (
                        bootstrap_claim.SOURCE_DESCRIPTOR_SHA256
                    ),
                    "files": locked_rows,
                },
            )
            hashes = {
                name: bootstrap_claim.sha256_file(path)
                for name, path in paths.items()
            }
            bootstrap_claim.create_claim(
                paths["contract"],
                paths["source_authority"],
                paths["burned_manifest"],
                paths["implementation_lock"],
                root / "claim.json",
                expected_hashes=hashes,
                claim_created_by_runner_only=True,
                repo_root=root,
            )
            tampered = root / sorted(bootstrap_claim.REQUIRED_IMPLEMENTATION_PATHS)[0]
            tampered.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(ValueError, "IMPLEMENTATION_FILE_MISMATCH"):
                bootstrap_claim.create_claim(
                    paths["contract"],
                    paths["source_authority"],
                    paths["burned_manifest"],
                    paths["implementation_lock"],
                    root / "claim2.json",
                    expected_hashes=hashes,
                    claim_created_by_runner_only=True,
                    repo_root=root,
                )

    def test_failed_official_source_is_valid_hold_and_no_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, hashes = package_documents(root)
            calls = []

            def opener(request: object, timeout: int) -> FakeResponse:
                calls.append((request, timeout))
                raise urllib.error.URLError("synthetic source unavailable")

            run_dir = root / "run"
            terminal = formal_runner.run_formal(
                contract=paths["contract"],
                source_authority=paths["source_authority"],
                burned_manifest=paths["burned_manifest"],
                implementation_lock=paths["implementation_lock"],
                run_dir=run_dir,
                expected_hashes=hashes,
                opener=opener,
                verify_implementation_files=False,
            )
            self.assertEqual(len(calls), 1)
            self.assertTrue((run_dir / "SUCCESS.json").is_file())
            self.assertFalse((run_dir / "FAILURE.json").exists())
            self.assertEqual(terminal["scientific_terminal"], producer.HOLD)
            self.assertEqual(terminal["validation_terminal"], "VALID")
            self.assertFalse(terminal["performance_qualification_may_be_created"])
            receipt = json.loads(
                (run_dir / "source_receipt.json").read_text(encoding="utf-8")
            )
            self.assertFalse(receipt["retry_authorized"])

    def test_validator_has_no_producer_import(self) -> None:
        tree = ast.parse(Path(validator.__file__).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(any("producer" in name for name in imported))

    def test_source_fence_has_no_rgb_read_api(self) -> None:
        source = Path(producer.__file__).read_text(encoding="utf-8")
        self.assertNotIn('read_control("rgb.txt")', source)
        self.assertNotIn("cv2.imread", source)
        self.assertNotIn("Image.open(BytesIO(raw_rgb", source)

    def test_pilot_refuses_sofa_3_before_path_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "SOFA_3_FORBIDDEN"):
            pilot.run_pilot(
                Path("missing/sofa_3"),
                pair_limit=2,
                workers=1,
            )


if __name__ == "__main__":
    unittest.main()
