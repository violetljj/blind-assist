from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_d5_s0a_tartanground_catalog import (
    CANONICAL_ROOT,
    CATALOG_INVALID,
    CATALOG_LOCKED,
    CONTRACT_RELATIVE_PATH,
    TOOLKIT_DIRNAME,
    execute_with_failure_closure,
    freeze_existing_partial,
    parse_gitmodules,
    parse_ls_tree_gitlink,
    parse_manifest,
    repo_root,
    require_canonical_root,
    validate_contract,
    validate_existing_terminal,
    write_json_exclusive_fsync,
)


COMMIT = "1" * 40
GITLINKS = {
    "tartanair/image_resampling/image_sampler": "2" * 40,
    "tartanair/image_resampling/mvs_utils": "3" * 40,
    "tartanair/data_cacher": "4" * 40,
}
URLS = {
    "tartanair/image_resampling/image_sampler":
        "https://github.com/castacks/image_sampler.git",
    "tartanair/image_resampling/mvs_utils":
        "https://github.com/castacks/mvs_utils.git",
    "tartanair/data_cacher":
        "https://github.com/Amigoshan/data_cacher.git",
}
REQUIRED = (
    "image_lcam_front.zip",
    "depth_lcam_front.zip",
    "seg_lcam_front.zip",
    "metadata.zip",
)


def fixture_manifest(parent_count: int = 64, environment_count: int = 8) -> bytes:
    rows = ["Other/Data_car/P0000/image_lcam_front.zip 0.001 G"]
    for index in range(parent_count):
        environment = f"Env{index % environment_count:02d}"
        trajectory = f"P1{index:03d}"
        for archive in REQUIRED:
            rows.append(
                f"{environment}/Data_diff/{trajectory}/{archive} "
                f"0.{100000000 + index} G"
            )
    return ("\n".join(rows) + "\n").encode()


def fixture_gitmodules() -> bytes:
    rows = []
    for index, (path, url) in enumerate(URLS.items()):
        rows.extend(
            [
                f'[submodule "module-{index}"]',
                f"\tpath = {path}",
                f"\turl = {url}",
            ]
        )
    return ("\n".join(rows) + "\n").encode()


class FakeGit:
    def __init__(self, manifest: bytes | None = None) -> None:
        self.manifest = manifest or fixture_manifest()
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args: list[str], cwd: Path) -> bytes:
        self.calls.append(tuple(args))
        if args[:2] == ["init", "."]:
            (cwd / ".git").mkdir()
            return b""
        if args[:3] == ["remote", "add", "origin"]:
            return b""
        if "fetch" in args:
            return b""
        if args == ["rev-parse", "FETCH_HEAD"]:
            return (COMMIT + "\n").encode()
        if args[:2] == ["cat-file", "blob"]:
            if args[2].endswith(":.gitmodules"):
                return fixture_gitmodules()
            if args[2].endswith(":tartanair/download_ground_files.txt"):
                return self.manifest
        if args[:2] == ["ls-tree", COMMIT]:
            path = args[-1]
            return f"160000 commit {GITLINKS[path]}\t{path}\n".encode()
        raise AssertionError(f"Unexpected git call: {args}")


def contract_context(base: Path) -> tuple[Path, dict[str, object]]:
    contract_path = base / "contract.json"
    design_path = base / "design.json"
    planner_path = base / "planner.py"
    mirror_path = base / "mirror.md"
    test_path = base / "test.py"
    design_path.write_text('{"status":"design"}')
    planner_path.write_text("planner")
    mirror_path.write_text("mirror")
    test_path.write_text(
        "\n".join(f"    def test_{index}(self): pass" for index in range(18))
    )
    contract = {
        "schema": (
            "blindassist_hftf_stage_c_d5_s0a_tartanground_catalog_"
            "execution_contract"
        ),
        "status": (
            "FROZEN_AFTER_D5_S0_DESIGN_BEFORE_EXACT_COMMIT_FETCH_OR_"
            "CATALOG_READ"
        ),
        "parents": {
            "d5_s0_design": {
                "path": str(design_path),
                "sha256": hashlib.sha256(design_path.read_bytes()).hexdigest(),
                "required_status": "design",
            }
        },
        "implementations": {
            "planner": {
                "path": str(planner_path),
                "sha256": hashlib.sha256(planner_path.read_bytes()).hexdigest(),
            },
            "narrative_mirror": {
                "path": str(mirror_path),
                "sha256": hashlib.sha256(mirror_path.read_bytes()).hexdigest(),
            },
        },
        "implementation_tests": {
            "planner_test": {
                "path": str(test_path),
                "sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
            },
            "test_count": 18,
        },
        "source_lock": {
            "toolkit_repository": "https://github.com/castacks/tartanairpy.git",
            "toolkit_commit": COMMIT,
            "manifest_path": "tartanair/download_ground_files.txt",
            "gitmodules_path": ".gitmodules",
            "submodule_gitlinks": [
                {"path": path, "url": URLS[path], "commit": commit}
                for path, commit in GITLINKS.items()
            ],
        },
        "network": {"git_fetch_attempts": 1},
        "catalog_gate": {
            "required_archives": list(REQUIRED),
            "minimum_distinct_diff_trajectory_parents": 64,
            "minimum_distinct_environments": 8,
        },
        "authorization": {
            "exact_commit_fetch_authorized": True,
            "dataset_host_request_authorized": False,
        },
    }
    contract_path.write_text(json.dumps(contract))
    return contract_path, contract


class D5S0ATartanGroundCatalogTest(unittest.TestCase):
    def test_parse_manifest_counts_exact_complete_parents(self) -> None:
        result = parse_manifest(
            fixture_manifest(), required_archives=REQUIRED
        )
        self.assertEqual(64, result["target_diff_parent_count"])
        self.assertEqual(64, result["required_catalog_complete_parent_count"])
        self.assertEqual(
            8, result["required_catalog_complete_environment_count"]
        )

    def test_regex_does_not_generate_unlisted_parent(self) -> None:
        result = parse_manifest(
            fixture_manifest(parent_count=1, environment_count=1),
            required_archives=REQUIRED,
        )
        self.assertEqual(
            ["Env00/Data_diff/P1000"],
            [row["parent_id"] for row in result["parents"]],
        )

    def test_non_diff_and_non_p1_rows_are_not_members(self) -> None:
        value = (
            b"A/Data_car/P1000/image_lcam_front.zip 0.1 G\n"
            b"A/Data_diff/P0000/image_lcam_front.zip 0.1 G\n"
        )
        result = parse_manifest(value, required_archives=REQUIRED)
        self.assertEqual(0, result["target_diff_parent_count"])

    def test_missing_archive_marks_parent_incomplete(self) -> None:
        lines = fixture_manifest(parent_count=1).decode().splitlines()
        value = ("\n".join(lines[:-1]) + "\n").encode()
        result = parse_manifest(value, required_archives=REQUIRED)
        self.assertEqual(0, result["required_catalog_complete_parent_count"])
        self.assertEqual(
            ["metadata.zip"],
            result["parents"][0]["missing_required_catalog_archives"],
        )

    def test_duplicate_manifest_path_fails_closed(self) -> None:
        row = b"A/Data_diff/P1000/metadata.zip 0.1 G\n"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_manifest(row + row, required_archives=REQUIRED)

    def test_unsafe_manifest_path_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            parse_manifest(
                b"../A/Data_diff/P1000/metadata.zip 0.1 G\n",
                required_archives=REQUIRED,
            )

    def test_real_manifest_size_shape_is_retained(self) -> None:
        value = (
            b"AbandonedCable/Data_diff/P1000/"
            b"depth_lcam_front.zip 0.151895107 G\n"
        )
        result = parse_manifest(value, required_archives=REQUIRED)
        observed = result["parents"][0]["archives"]["depth_lcam_front.zip"]
        self.assertEqual("0.151895107 G", observed["declared_size"])

    def test_blank_or_malformed_size_manifest_is_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "Empty"):
            parse_manifest(b"\n \t\n", required_archives=REQUIRED)
        with self.assertRaisesRegex(ValueError, "size format"):
            parse_manifest(
                b"A/Data_diff/P1000/metadata.zip 123\n",
                required_archives=REQUIRED,
            )

    def test_gitmodules_exact_mapping(self) -> None:
        self.assertEqual(URLS, parse_gitmodules(fixture_gitmodules()))

    def test_gitmodules_missing_url_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            parse_gitmodules(b'[submodule "x"]\n path = x\n')

    def test_gitlink_parser_requires_mode_commit_and_path(self) -> None:
        path = next(iter(GITLINKS))
        line = f"160000 commit {GITLINKS[path]}\t{path}"
        self.assertEqual(GITLINKS[path], parse_ls_tree_gitlink(line, path))
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            parse_ls_tree_gitlink(line.replace("160000", "100644"), path)

    def test_exclusive_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "value.json"
            write_json_exclusive_fsync(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                write_json_exclusive_fsync(path, {"value": 2})

    def test_canonical_root_rejects_alternate(self) -> None:
        expected = (repo_root() / CANONICAL_ROOT).resolve()
        self.assertEqual(expected, require_canonical_root(expected))
        with self.assertRaisesRegex(ValueError, "Noncanonical"):
            require_canonical_root(repo_root() / "wrong")

    def test_fetch_reads_only_two_commit_objects_and_no_submodule(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, contract = contract_context(base)
            fake = FakeGit()
            with mock.patch(
                "plan_stage_c_d5_s0a_tartanground_catalog.validate_contract",
                return_value={
                    "contract": contract,
                    "contract_path": contract_path,
                    "design_path": base / "design.json",
                    "implementation_paths": [
                        (base / "planner.py", "planner"),
                        (base / "mirror.md", "narrative_mirror"),
                    ],
                    "test_path": base / "test.py",
                },
            ):
                result = execute_with_failure_closure(
                    contract_path,
                    base / "run",
                    git_runner=fake,
                    verify_git=False,
                )
        self.assertEqual(CATALOG_LOCKED, result["terminal"])
        cat_calls = [call for call in fake.calls if call[:2] == ("cat-file", "blob")]
        self.assertEqual(2, len(cat_calls))
        fetch_calls = [call for call in fake.calls if "fetch" in call]
        self.assertEqual(
            [
                (
                    "-c",
                    "protocol.version=2",
                    "fetch",
                    "--no-tags",
                    "--depth=1",
                    "--recurse-submodules=no",
                    "origin",
                    COMMIT,
                )
            ],
            fetch_calls,
        )
        forbidden_commands = {"checkout", "switch", "reset", "submodule"}
        self.assertFalse(
            any(call and call[0] in forbidden_commands for call in fake.calls)
        )

    def test_attempt_and_preflight_precede_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, contract = contract_context(base)

            def checking_git(args: list[str], cwd: Path) -> bytes:
                if "fetch" in args:
                    root = cwd.parent
                    self.assertTrue((root / "attempt.json").is_file())
                    self.assertTrue((root / "preflight.json").is_file())
                return FakeGit()(args, cwd)

            with mock.patch(
                "plan_stage_c_d5_s0a_tartanground_catalog.validate_contract",
                return_value={
                    "contract": contract,
                    "contract_path": contract_path,
                    "design_path": base / "design.json",
                    "implementation_paths": [
                        (base / "planner.py", "planner"),
                        (base / "mirror.md", "narrative_mirror"),
                    ],
                    "test_path": base / "test.py",
                },
            ):
                execute_with_failure_closure(
                    contract_path,
                    base / "run",
                    git_runner=checking_git,
                    verify_git=False,
                )

    def test_fetch_failure_writes_invalid_and_never_writes_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, contract = contract_context(base)

            def failing_git(args: list[str], cwd: Path) -> bytes:
                if args[:2] == ["init", "."]:
                    (cwd / ".git").mkdir()
                    return b""
                if "fetch" in args:
                    raise OSError("transport")
                return b""

            with mock.patch(
                "plan_stage_c_d5_s0a_tartanground_catalog.validate_contract",
                return_value={
                    "contract": contract,
                    "contract_path": contract_path,
                    "design_path": base / "design.json",
                    "implementation_paths": [
                        (base / "planner.py", "planner"),
                        (base / "mirror.md", "narrative_mirror"),
                    ],
                    "test_path": base / "test.py",
                },
            ):
                with self.assertRaisesRegex(OSError, "transport"):
                    execute_with_failure_closure(
                        contract_path,
                        base / "run",
                        git_runner=failing_git,
                        verify_git=False,
                    )
            failure = json.loads((base / "run/failure.json").read_text())
        self.assertEqual(CATALOG_INVALID, failure["terminal"])
        self.assertFalse((base / "run/catalog.json").exists())

    def test_incomplete_catalog_reaches_capacity_stop_not_source_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, contract = contract_context(base)
            fake = FakeGit(fixture_manifest(parent_count=63))
            with mock.patch(
                "plan_stage_c_d5_s0a_tartanground_catalog.validate_contract",
                return_value={
                    "contract": contract,
                    "contract_path": contract_path,
                    "design_path": base / "design.json",
                    "implementation_paths": [
                        (base / "planner.py", "planner"),
                        (base / "mirror.md", "narrative_mirror"),
                    ],
                    "test_path": base / "test.py",
                },
            ):
                result = execute_with_failure_closure(
                    contract_path,
                    base / "run",
                    git_runner=fake,
                    verify_git=False,
                )
        self.assertIn("CAPACITY_INSUFFICIENT", result["terminal"])
        self.assertFalse(result["d5_s0_source_feasibility_terminal_reached"])

    def test_partial_root_freezes_invalid_without_git(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "attempt.json").write_text("{}")
            code = freeze_existing_partial(root, {"attempt.json"})
            failure = json.loads((root / "failure.json").read_text())
        self.assertEqual(2, code)
        self.assertEqual(CATALOG_INVALID, failure["terminal"])

    def test_valid_terminal_requires_hash_chain_and_exact_fetch_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, contract = contract_context(base)
            root = base / "run"
            with mock.patch(
                "plan_stage_c_d5_s0a_tartanground_catalog.validate_contract",
                return_value={
                    "contract": contract,
                    "contract_path": contract_path,
                    "design_path": base / "design.json",
                    "implementation_paths": [
                        (base / "planner.py", "planner"),
                        (base / "mirror.md", "narrative_mirror"),
                    ],
                    "test_path": base / "test.py",
                },
            ):
                execute_with_failure_closure(
                    contract_path,
                    root,
                    git_runner=FakeGit(),
                    verify_git=False,
                )
            names = {path.name for path in root.iterdir()}
            with mock.patch(
                "plan_stage_c_d5_s0a_tartanground_catalog.git_local",
                return_value=COMMIT,
            ):
                self.assertTrue(
                    validate_existing_terminal(
                        root, names, contract_path=contract_path
                    )
                )
                (root / "catalog.json").write_text("tampered")
                self.assertFalse(
                    validate_existing_terminal(
                        root, names, contract_path=contract_path
                    )
                )

    def test_terminal_rejects_cross_artifact_source_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, contract = contract_context(base)
            root = base / "run"
            with mock.patch(
                "plan_stage_c_d5_s0a_tartanground_catalog.validate_contract",
                return_value={
                    "contract": contract,
                    "contract_path": contract_path,
                    "design_path": base / "design.json",
                    "implementation_paths": [
                        (base / "planner.py", "planner"),
                        (base / "mirror.md", "narrative_mirror"),
                    ],
                    "test_path": base / "test.py",
                },
            ):
                execute_with_failure_closure(
                    contract_path,
                    root,
                    git_runner=FakeGit(),
                    verify_git=False,
                )
            names = {path.name for path in root.iterdir()}
            result_path = root / "result.json"
            attempt_path = root / "attempt.json"
            catalog_path = root / "catalog.json"
            originals = {
                path: path.read_bytes()
                for path in (result_path, attempt_path, catalog_path)
            }

            result = json.loads(result_path.read_text())
            result["source_identity"]["toolkit_commit"] = "9" * 40
            result_path.write_text(json.dumps(result))
            with mock.patch(
                "plan_stage_c_d5_s0a_tartanground_catalog.git_local",
                return_value="9" * 40,
            ):
                self.assertFalse(
                    validate_existing_terminal(
                        root, names, contract_path=contract_path
                    )
                )
            result_path.write_bytes(originals[result_path])

            attempt = json.loads(attempt_path.read_text())
            attempt["toolkit_commit"] = "9" * 40
            attempt_path.write_text(json.dumps(attempt))
            with mock.patch(
                "plan_stage_c_d5_s0a_tartanground_catalog.git_local",
                return_value=COMMIT,
            ):
                self.assertFalse(
                    validate_existing_terminal(
                        root, names, contract_path=contract_path
                    )
                )
            attempt_path.write_bytes(originals[attempt_path])

            catalog = json.loads(catalog_path.read_text())
            catalog["source_identity"]["toolkit_repository"] = "foreign"
            catalog_path.write_text(json.dumps(catalog))
            with mock.patch(
                "plan_stage_c_d5_s0a_tartanground_catalog.git_local",
                return_value=COMMIT,
            ):
                self.assertFalse(
                    validate_existing_terminal(
                        root, names, contract_path=contract_path
                    )
                )

    def test_failure_terminal_binds_observed_names_and_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, contract = contract_context(base)
            root = base / "run"
            root.mkdir()
            attempt = {
                "schema": (
                    "blindassist_hftf_stage_c_d5_s0a_tartanground_"
                    "catalog_attempt"
                ),
                "status": "ATTEMPT_FSYNCED_BEFORE_FIRST_GIT_NETWORK_REQUEST",
                "execution_contract_sha256": hashlib.sha256(
                    contract_path.read_bytes()
                ).hexdigest(),
                "toolkit_repository": contract["source_lock"][
                    "toolkit_repository"
                ],
                "toolkit_commit": COMMIT,
            }
            (root / "attempt.json").write_text(json.dumps(attempt))
            failure = {
                "schema": (
                    "blindassist_hftf_stage_c_d5_s0a_tartanground_"
                    "catalog_failure"
                ),
                "terminal": CATALOG_INVALID,
                "observed_top_level_names": ["attempt.json"],
                "attempt_sha256": hashlib.sha256(
                    (root / "attempt.json").read_bytes()
                ).hexdigest(),
                "preflight_sha256": None,
                "resume_or_rerun_authorized": False,
            }
            (root / "failure.json").write_text(json.dumps(failure))
            names = {path.name for path in root.iterdir()}
            self.assertTrue(
                validate_existing_terminal(
                    root, names, contract_path=contract_path
                )
            )
            (root / "late-unknown").write_text("drift")
            names = {path.name for path in root.iterdir()}
            self.assertFalse(
                validate_existing_terminal(
                    root, names, contract_path=contract_path
                )
            )

    def test_real_contract_validates_without_git_gate(self) -> None:
        context = validate_contract(
            repo_root() / CONTRACT_RELATIVE_PATH,
            verify_git=False,
        )
        self.assertEqual(
            "158a6844d782942110967325ca3082f50ab2bfc7",
            context["contract"]["source_lock"]["toolkit_commit"],
        )


if __name__ == "__main__":
    unittest.main()
