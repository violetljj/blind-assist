from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_d5_s0a1_tartanground_catalog import (
    CANONICAL_ROOT,
    CATALOG_INVALID,
    CATALOG_LOCKED,
    CONTRACT_RELATIVE_PATH,
    execute_with_failure_closure,
    freeze_existing_partial,
    parse_manifest_path_tokens,
    repo_root,
    require_canonical_root,
    validate_contract,
    validate_existing_terminal,
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


def manifest(
    parent_count: int = 64,
    environment_count: int = 8,
    suffix: str = "0.1 G",
) -> bytes:
    rows = ["Other/Data_car/P0000/image_lcam_front.zip arbitrary suffix"]
    for index in range(parent_count):
        environment = f"Env{index % environment_count:02d}"
        trajectory = f"P1{index:03d}"
        for archive in REQUIRED:
            tail = f" {suffix}" if suffix else ""
            rows.append(
                f"{environment}/Data_diff/{trajectory}/{archive}{tail}"
            )
    return ("\n".join(rows) + "\n").encode()


def gitmodules() -> bytes:
    rows: list[str] = []
    for index, (path, url) in enumerate(URLS.items()):
        rows += [
            f'[submodule "module-{index}"]',
            f"\tpath = {path}",
            f"\turl = {url}",
        ]
    return ("\n".join(rows) + "\n").encode()


class FakeGit:
    def __init__(self, value: bytes | None = None) -> None:
        self.manifest = value or manifest()
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args: list[str], cwd: Path) -> bytes:
        self.calls.append(tuple(args))
        if args[:2] == ["init", "."]:
            (cwd / ".git").mkdir()
            return b""
        if args[:3] == ["remote", "add", "origin"] or "fetch" in args:
            return b""
        if args == ["rev-parse", "FETCH_HEAD"]:
            return (COMMIT + "\n").encode()
        if args[:2] == ["cat-file", "blob"]:
            return gitmodules() if args[2].endswith(":.gitmodules") else self.manifest
        if args[:2] == ["ls-tree", COMMIT]:
            path = args[-1]
            return f"160000 commit {GITLINKS[path]}\t{path}\n".encode()
        raise AssertionError(f"Unexpected Git call: {args}")


def make_context(base: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    contract_path = base / "contract.json"
    design = base / "design.json"
    invalid = base / "invalid.json"
    planner = base / "planner.py"
    helper = base / "helper.py"
    mirror = base / "mirror.md"
    test = base / "test.py"
    design.write_text('{"status":"design"}')
    invalid.write_text('{"terminal":"invalid"}')
    planner.write_text("planner")
    helper.write_text("helper")
    mirror.write_text("mirror")
    test.write_text(
        "\n".join(f"    def test_{index}(self): pass" for index in range(20))
    )
    paths = {
        "design": design,
        "invalid": invalid,
        "planner": planner,
        "helper": helper,
        "mirror": mirror,
        "test": test,
    }
    contract: dict[str, object] = {
        "schema": (
            "blindassist_hftf_stage_c_d5_s0a1_tartanground_catalog_"
            "execution_contract"
        ),
        "status": (
            "FROZEN_AFTER_S0A1_DESIGN_BEFORE_NEW_EXACT_COMMIT_FETCH_OR_"
            "MANIFEST_READ"
        ),
        "parents": {
            "design": {
                "path": str(design),
                "sha256": hashlib.sha256(design.read_bytes()).hexdigest(),
                "required_status": "design",
            },
            "invalid": {
                "path": str(invalid),
                "sha256": hashlib.sha256(invalid.read_bytes()).hexdigest(),
                "required_terminal": "invalid",
            },
        },
        "implementations": {
            "planner": {
                "path": str(planner),
                "sha256": hashlib.sha256(planner.read_bytes()).hexdigest(),
            },
            "helper": {
                "path": str(helper),
                "sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
            },
            "mirror": {
                "path": str(mirror),
                "sha256": hashlib.sha256(mirror.read_bytes()).hexdigest(),
            },
        },
        "implementation_tests": {
            "planner_test": {
                "path": str(test),
                "sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
            },
            "test_count": 20,
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
        "manifest_parser": {
            "first_token_only_is_path_identity": True,
            "all_suffix_tokens_discarded": True,
            "suffix_retained_or_used": False,
        },
        "catalog_gate": {
            "required_archives": list(REQUIRED),
            "minimum_distinct_diff_trajectory_parents": 64,
            "minimum_distinct_environments": 8,
        },
        "authorization": {
            "read_failed_s0a_root_authorized": False,
            "dataset_host_request_authorized": False,
        },
    }
    contract_path.write_text(json.dumps(contract))
    context = {
        "contract": contract,
        "contract_path": contract_path,
        "parent_paths": [(design, "design"), (invalid, "invalid")],
        "implementation_paths": [
            (planner, "planner"),
            (helper, "helper"),
            (mirror, "mirror"),
        ],
        "test_path": test,
    }
    return contract_path, contract, context


class D5S0A1CatalogTest(unittest.TestCase):
    def test_suffix_shapes_do_not_change_catalog(self) -> None:
        variants = (
            "0.151895107 G",
            "0 G",
            "words",
            "",
            "+?!",
            "odd\t\f\v tokens",
        )
        results = [
            parse_manifest_path_tokens(
                manifest(parent_count=1, environment_count=1, suffix=value),
                required_archives=REQUIRED,
            )
            for value in variants
        ]
        self.assertTrue(
            all(result["parents"] == results[0]["parents"] for result in results)
        )

    def test_lf_and_crlf_have_identical_catalog(self) -> None:
        lf = manifest(parent_count=2, environment_count=1)
        crlf = lf.replace(b"\n", b"\r\n")
        first = parse_manifest_path_tokens(lf, required_archives=REQUIRED)
        second = parse_manifest_path_tokens(crlf, required_archives=REQUIRED)
        self.assertEqual(first, second)

    def test_suffix_is_not_retained_or_derived(self) -> None:
        secret = "MUST_NOT_SURVIVE_978"
        result = parse_manifest_path_tokens(
            manifest(parent_count=1, suffix=secret),
            required_archives=REQUIRED,
        )
        serialized = json.dumps(result)
        self.assertNotIn(secret, serialized)
        self.assertFalse(result["suffix_tokens_read_validated_retained_or_used"])

    def test_blank_manifest_is_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "Blank-only"):
            parse_manifest_path_tokens(b"\n\t \n", required_archives=REQUIRED)

    def test_duplicate_path_is_invalid(self) -> None:
        row = b"A/Data_diff/P1000/metadata.zip x\n"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_manifest_path_tokens(row + row, required_archives=REQUIRED)

    def test_unsafe_paths_are_invalid(self) -> None:
        for value in (
            b"../A/Data_diff/P1000/metadata.zip x\n",
            b"A\\Data_diff\\P1000\\metadata.zip x\n",
        ):
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                parse_manifest_path_tokens(value, required_archives=REQUIRED)

    def test_regex_does_not_generate_parent(self) -> None:
        result = parse_manifest_path_tokens(
            b"A/Data_diff/P1000/metadata.zip anything\n",
            required_archives=REQUIRED,
        )
        self.assertEqual(["A/Data_diff/P1000"], [row["parent_id"] for row in result["parents"]])

    def test_non_p1_and_non_diff_are_not_members(self) -> None:
        value = (
            b"A/Data_diff/P0000/metadata.zip x\n"
            b"A/Data_car/P1000/metadata.zip y\n"
        )
        result = parse_manifest_path_tokens(value, required_archives=REQUIRED)
        self.assertEqual(0, result["target_diff_parent_count"])

    def test_missing_archive_is_incomplete(self) -> None:
        rows = manifest(parent_count=1).decode().splitlines()
        result = parse_manifest_path_tokens(
            ("\n".join(rows[:-1]) + "\n").encode(),
            required_archives=REQUIRED,
        )
        self.assertEqual(0, result["required_catalog_complete_parent_count"])
        self.assertEqual(["metadata.zip"], result["parents"][0]["missing_required_catalog_archives"])

    def test_capacity_counts_parent_and_environment(self) -> None:
        result = parse_manifest_path_tokens(manifest(), required_archives=REQUIRED)
        self.assertEqual(64, result["required_catalog_complete_parent_count"])
        self.assertEqual(8, result["required_catalog_complete_environment_count"])

    def test_canonical_root_rejects_old_and_alternate(self) -> None:
        expected = (repo_root() / CANONICAL_ROOT).resolve()
        self.assertEqual(expected, require_canonical_root(expected))
        with self.assertRaisesRegex(ValueError, "Noncanonical"):
            require_canonical_root(repo_root() / "wrong")

    def test_execute_uses_exact_fetch_and_two_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            fake = FakeGit()
            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog.validate_contract",
                return_value=context,
            ):
                result = execute_with_failure_closure(
                    contract_path, base / "run", git_runner=fake, verify_git=False
                )
        self.assertEqual(CATALOG_LOCKED, result["terminal"])
        self.assertEqual(2, sum(call[:2] == ("cat-file", "blob") for call in fake.calls))
        fetch_calls = [call for call in fake.calls if "fetch" in call]
        self.assertEqual(
            [(
                "-c", "protocol.version=2", "fetch", "--no-tags",
                "--depth=1", "--recurse-submodules=no", "origin", COMMIT,
            )],
            fetch_calls,
        )
        self.assertEqual(
            {
                ("ls-tree", COMMIT, "--", path)
                for path in GITLINKS
            },
            {call for call in fake.calls if call[:2] == ("ls-tree", COMMIT)},
        )
        self.assertFalse(any(call and call[0] in {"checkout", "submodule", "reset", "switch"} for call in fake.calls))

    def test_execute_catalog_observation_is_suffix_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            observations = []
            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog.validate_contract",
                return_value=context,
            ):
                for index, suffix in enumerate(("0.1 G", "opaque words 978")):
                    root = base / f"run-{index}"
                    execute_with_failure_closure(
                        contract_path,
                        root,
                        git_runner=FakeGit(manifest(suffix=suffix)),
                        verify_git=False,
                    )
                    observations.append(
                        json.loads((root / "catalog.json").read_text())[
                            "catalog_observation"
                        ]
                    )
        self.assertEqual(observations[0], observations[1])

    def test_attempt_and_preflight_exist_before_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            fake = FakeGit()

            def checking(args: list[str], cwd: Path) -> bytes:
                if "fetch" in args:
                    self.assertTrue((cwd.parent / "attempt.json").is_file())
                    self.assertTrue((cwd.parent / "preflight.json").is_file())
                return fake(args, cwd)

            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog.validate_contract",
                return_value=context,
            ):
                execute_with_failure_closure(
                    contract_path, base / "run", git_runner=checking, verify_git=False
                )

    def test_transport_failure_writes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)

            def failing(args: list[str], cwd: Path) -> bytes:
                if args[:2] == ["init", "."]:
                    (cwd / ".git").mkdir()
                    return b""
                if "fetch" in args:
                    raise OSError("transport")
                return b""

            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog.validate_contract",
                return_value=context,
            ):
                with self.assertRaisesRegex(OSError, "transport"):
                    execute_with_failure_closure(
                        contract_path, base / "run", git_runner=failing, verify_git=False
                    )
            failure = json.loads((base / "run/failure.json").read_text())
        self.assertEqual(CATALOG_INVALID, failure["terminal"])
        self.assertFalse(failure["failed_s0a_root_read"])

    def test_capacity_insufficient_is_not_source_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog.validate_contract",
                return_value=context,
            ):
                result = execute_with_failure_closure(
                    contract_path,
                    base / "run",
                    git_runner=FakeGit(manifest(parent_count=63)),
                    verify_git=False,
                )
        self.assertIn("CAPACITY_INSUFFICIENT", result["terminal"])
        self.assertFalse(result["d5_s0_source_feasibility_terminal_reached"])

    def test_partial_root_freezes_without_git(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "attempt.json").write_text("{}")
            self.assertEqual(2, freeze_existing_partial(root, {"attempt.json"}))
            failure = json.loads((root / "failure.json").read_text())
        self.assertEqual(CATALOG_INVALID, failure["terminal"])

    def test_failure_terminal_validates_preflight_and_fetch_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            root = base / "run"

            def failing(args: list[str], cwd: Path) -> bytes:
                if args[:2] == ["init", "."]:
                    (cwd / ".git").mkdir()
                    return b""
                if "fetch" in args:
                    raise OSError("transport")
                return b""

            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog.validate_contract",
                return_value=context,
            ):
                with self.assertRaises(OSError):
                    execute_with_failure_closure(
                        contract_path, root, git_runner=failing, verify_git=False
                    )
            names = {path.name for path in root.iterdir()}
            self.assertTrue(
                validate_existing_terminal(
                    root, names, contract_path=contract_path
                )
            )
            preflight_path = root / "preflight.json"
            original = preflight_path.read_bytes()
            preflight = json.loads(original)
            preflight["status"] = "tampered"
            preflight_path.write_text(json.dumps(preflight))
            self.assertFalse(
                validate_existing_terminal(
                    root, names, contract_path=contract_path
                )
            )
            preflight_path.write_bytes(original)
            failure_path = root / "failure.json"
            failure = json.loads(failure_path.read_text())
            failure["fetched_commit"] = COMMIT
            failure_path.write_text(json.dumps(failure))
            self.assertFalse(
                validate_existing_terminal(
                    root, names, contract_path=contract_path
                )
            )

    def test_post_fetch_failure_binds_exact_fetch_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            root = base / "run"
            fake = FakeGit()

            def fail_manifest(args: list[str], cwd: Path) -> bytes:
                if (
                    args[:2] == ["cat-file", "blob"]
                    and args[2].endswith(
                        ":tartanair/download_ground_files.txt"
                    )
                ):
                    raise ValueError("manifest")
                return fake(args, cwd)

            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog.validate_contract",
                return_value=context,
            ), mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog."
                "local_fetch_head_or_none",
                return_value=COMMIT,
            ):
                with self.assertRaisesRegex(ValueError, "manifest"):
                    execute_with_failure_closure(
                        contract_path,
                        root,
                        git_runner=fail_manifest,
                        verify_git=False,
                    )
            names = {path.name for path in root.iterdir()}
            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog."
                "local_fetch_head_or_none",
                return_value=COMMIT,
            ):
                self.assertTrue(
                    validate_existing_terminal(
                        root, names, contract_path=contract_path
                    )
                )
            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog."
                "local_fetch_head_or_none",
                return_value="9" * 40,
            ):
                self.assertFalse(
                    validate_existing_terminal(
                        root, names, contract_path=contract_path
                    )
                )

    def test_valid_terminal_requires_contract_and_fetch_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            root = base / "run"
            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog.validate_contract",
                return_value=context,
            ):
                execute_with_failure_closure(
                    contract_path, root, git_runner=FakeGit(), verify_git=False
                )
            names = {path.name for path in root.iterdir()}
            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog.git_local",
                return_value=COMMIT,
            ):
                self.assertTrue(
                    validate_existing_terminal(root, names, contract_path=contract_path)
                )
                result = json.loads((root / "result.json").read_text())
                result["source_identity"]["toolkit_commit"] = "9" * 40
                (root / "result.json").write_text(json.dumps(result))
                self.assertFalse(
                    validate_existing_terminal(root, names, contract_path=contract_path)
                )

    def test_terminal_rejects_terminal_gate_and_claim_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            root = base / "run"
            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog.validate_contract",
                return_value=context,
            ):
                execute_with_failure_closure(
                    contract_path, root, git_runner=FakeGit(), verify_git=False
                )
            names = {path.name for path in root.iterdir()}
            result_path = root / "result.json"
            original = json.loads(result_path.read_text())
            mutations = [
                ("terminal", "D5_S0A1_TARTANGROUND_DIFF_CATALOG_CAPACITY_INSUFFICIENT_STOP"),
                ("d5_s0_source_feasibility_terminal_reached", True),
                ("s0b_execution_authorized_automatically", True),
            ]
            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog.git_local",
                return_value=COMMIT,
            ):
                for key, value in mutations:
                    changed = json.loads(json.dumps(original))
                    changed[key] = value
                    result_path.write_text(json.dumps(changed))
                    self.assertFalse(
                        validate_existing_terminal(
                            root, names, contract_path=contract_path
                        ),
                        key,
                    )
                changed = json.loads(json.dumps(original))
                changed["catalog_gate"][
                    "observed_required_catalog_complete_parent_count"
                ] += 1
                result_path.write_text(json.dumps(changed))
                self.assertFalse(
                    validate_existing_terminal(
                        root, names, contract_path=contract_path
                    )
                )

    def test_terminal_rejects_catalog_count_and_firewall_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, _, context = make_context(base)
            root = base / "run"
            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog.validate_contract",
                return_value=context,
            ):
                execute_with_failure_closure(
                    contract_path, root, git_runner=FakeGit(), verify_git=False
                )
            names = {path.name for path in root.iterdir()}
            catalog_path = root / "catalog.json"
            original = json.loads(catalog_path.read_text())
            with mock.patch(
                "plan_stage_c_d5_s0a1_tartanground_catalog.git_local",
                return_value=COMMIT,
            ):
                changed = json.loads(json.dumps(original))
                changed["catalog_observation"][
                    "required_catalog_complete_parent_count"
                ] += 1
                catalog_path.write_text(json.dumps(changed))
                self.assertFalse(
                    validate_existing_terminal(
                        root, names, contract_path=contract_path
                    )
                )
                changed = json.loads(json.dumps(original))
                changed["failed_s0a_root_read"] = True
                catalog_path.write_text(json.dumps(changed))
                self.assertFalse(
                    validate_existing_terminal(
                        root, names, contract_path=contract_path
                    )
                )

    def test_contract_rejects_suffix_authority_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            contract_path, contract, _ = make_context(base)
            contract["manifest_parser"]["suffix_retained_or_used"] = True
            contract_path.write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, "suffix"):
                validate_contract(contract_path, verify_git=False)

    def test_real_contract_validates_without_git_gate(self) -> None:
        context = validate_contract(
            repo_root() / CONTRACT_RELATIVE_PATH, verify_git=False
        )
        self.assertEqual(
            "158a6844d782942110967325ca3082f50ab2bfc7",
            context["contract"]["source_lock"]["toolkit_commit"],
        )


if __name__ == "__main__":
    unittest.main()
