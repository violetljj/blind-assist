from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
RUNNER_PATH = REPO_ROOT / "tools" / "run_dtr_carla_c4_multimap.ps1"
C2_RUNNER_PATH = REPO_ROOT / "tools" / "run_dtr_carla_c2_rich_scene.ps1"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dtr_carla_c4_runner_plan import (  # noqa: E402
    C2_EXPERIMENT_ID,
    C4_EXPERIMENT_ID,
    COMPILED_SCHEMA_VERSION,
    RunnerPlanError,
    SENSOR_ORDER,
    build_runner_plan,
)


MAPS = ("Town01", "Town02", "Town03_Opt", "Town04")


class C4RunnerPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.index_path = self._write_bundle()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest().upper()

    def _group_protocol(
        self, map_name: str, group_id: str, layouts: list[str]
    ) -> dict[str, object]:
        object_path = f"/Game/Carla/Maps/{map_name}.{map_name}"
        return {
            "experiment_id": C2_EXPERIMENT_ID,
            "environment": {"map": f"Carla/Maps/{map_name}"},
            "capture": {
                "resolution": [1280, 720],
                "sensor_order": list(SENSOR_ORDER),
            },
            "layouts": {
                layout_id: {"duration_seconds": 1.0} for layout_id in layouts
            },
            "scenarios": [
                {"episode_id": f"episode_{layout_id}", "layout_id": layout_id}
                for layout_id in layouts
            ],
            "admission": {
                "expected_layout_count": len(layouts),
                "expected_episode_count": len(layouts),
            },
            "c4_compatibility": {
                "schema_version": COMPILED_SCHEMA_VERSION,
                "experiment_id": C4_EXPERIMENT_ID,
                "protocol_id": group_id,
                "carla_map": f"Carla/Maps/{map_name}",
                "engine_ini_map_object_path": object_path,
                "cold_start_status": "UNPROBED",
            },
        }

    def _write_bundle(self) -> Path:
        asset_registry = self.root / "dtr_carla_c4_asset_registry.json"
        scene_registry = self.root / "dtr_carla_c4_scene_registry.json"
        self._write_json(asset_registry, {"schema_version": "dtr-c4-asset-registry-v1"})
        self._write_json(scene_registry, {"schema_version": "dtr-c4-scene-registry-v1"})
        protocols = []
        for map_index, map_name in enumerate(MAPS, start=1):
            group_id = f"town{map_index:02d}"
            layouts = [f"layout_{map_index:02d}_a", f"layout_{map_index:02d}_b"]
            relative_path = Path("groups") / f"{group_id}.c2-protocol.json"
            protocol_path = self.root / relative_path
            self._write_json(
                protocol_path,
                self._group_protocol(map_name, group_id, layouts),
            )
            protocols.append(
                {
                    "protocol_id": group_id,
                    "protocol_path": relative_path.as_posix(),
                    "protocol_sha256": self._sha256(protocol_path),
                    "carla_map": f"Carla/Maps/{map_name}",
                    "startup_map_argument": f"/Game/Carla/Maps/{map_name}.{map_name}",
                    "engine_ini_map_object_path": f"/Game/Carla/Maps/{map_name}.{map_name}",
                    "cold_start_status": "UNPROBED",
                    "layout_ids": layouts,
                    "episodes": [
                        {
                            "episode_id": f"episode_{layout_id}",
                            "layout_id": layout_id,
                        }
                        for layout_id in layouts
                    ],
                    "layout_count": len(layouts),
                    "episode_count": len(layouts),
                }
            )
        index = {
            "schema_version": COMPILED_SCHEMA_VERSION,
            "experiment_id": C4_EXPERIMENT_ID,
            "capture": {
                "resolution": [1280, 720],
                "sensor_order": list(SENSOR_ORDER),
            },
            "registries": {
                "asset_registry": {
                    "path": asset_registry.name,
                    "sha256": self._sha256(asset_registry),
                },
                "scene_registry": {
                    "path": scene_registry.name,
                    "sha256": self._sha256(scene_registry),
                },
            },
            "protocols": protocols,
            "admission": {
                "expected_map_count": 4,
                "expected_protocol_count": 4,
                "expected_layout_count": 8,
                "expected_episode_count": 8,
                "expected_sensor_count": 4,
                "expected_shard_count": 16,
            },
        }
        index_path = self.root / "compiled-index.json"
        self._write_json(index_path, index)
        return index_path

    def _read_index(self) -> dict[str, object]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _rewrite_protocol_entry(
        self, index: dict[str, object], entry: dict[str, object], protocol: dict[str, object]
    ) -> None:
        protocol_path = self.root / entry["protocol_path"]
        self._write_json(protocol_path, protocol)
        entry["protocol_sha256"] = self._sha256(protocol_path)
        self._write_json(self.index_path, index)

    def test_valid_bundle_builds_four_serial_map_groups_and_sixteen_shards(self) -> None:
        plan = build_runner_plan(
            self.index_path,
            base_rpc_port=24000,
            port_group_stride=3,
        )
        self.assertEqual([1280, 720], plan["resolution"])
        self.assertEqual(4, plan["map_count"])
        self.assertEqual(8, plan["layout_count"])
        self.assertEqual(16, plan["shard_count"])
        self.assertEqual(12, len(plan["all_ports"]))
        first = plan["map_layout_groups"][0]
        last = plan["map_layout_groups"][-1]
        self.assertEqual([24000, 24001, 24002], first["ports"])
        self.assertEqual([24009, 24010, 24011], last["ports"])
        self.assertEqual(
            "/Game/Carla/Maps/Town01.Town01",
            first["startup_map_argument"],
        )
        self.assertEqual(
            first["startup_map_argument"], first["engine_ini_map_object_path"]
        )

    def test_rejects_fewer_than_four_distinct_maps(self) -> None:
        index = self._read_index()
        index["protocols"] = index["protocols"][:3]
        self._write_json(self.index_path, index)
        with self.assertRaisesRegex(RunnerPlanError, "at least four"):
            build_runner_plan(self.index_path, base_rpc_port=24000, port_group_stride=3)

    def test_rejects_fewer_than_eight_distinct_layouts(self) -> None:
        index = self._read_index()
        entry = index["protocols"][-1]
        protocol_path = self.root / entry["protocol_path"]
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        retained = entry["layout_ids"][0]
        entry["layout_ids"] = [retained]
        entry["episodes"] = [entry["episodes"][0]]
        entry["layout_count"] = 1
        entry["episode_count"] = 1
        protocol["layouts"] = {retained: protocol["layouts"][retained]}
        protocol["scenarios"] = [protocol["scenarios"][0]]
        protocol["admission"]["expected_layout_count"] = 1
        protocol["admission"]["expected_episode_count"] = 1
        index["admission"]["expected_layout_count"] = 7
        index["admission"]["expected_episode_count"] = 7
        self._rewrite_protocol_entry(index, entry, protocol)
        with self.assertRaisesRegex(RunnerPlanError, "at least eight"):
            build_runner_plan(self.index_path, base_rpc_port=24000, port_group_stride=3)

    def test_rejects_resolution_drift(self) -> None:
        index = self._read_index()
        index["capture"]["resolution"] = [1920, 1080]
        self._write_json(self.index_path, index)
        with self.assertRaisesRegex(RunnerPlanError, "exactly 1280x720"):
            build_runner_plan(self.index_path, base_rpc_port=24000, port_group_stride=3)

    def test_rejects_startup_map_not_bound_to_protocol_map(self) -> None:
        index = self._read_index()
        index["protocols"][0]["startup_map_argument"] = (
            "/Game/Carla/Maps/Town02.Town02"
        )
        self._write_json(self.index_path, index)
        with self.assertRaisesRegex(RunnerPlanError, "startup_map_argument does not bind"):
            build_runner_plan(self.index_path, base_rpc_port=24000, port_group_stride=3)

    def test_rejects_engine_ini_object_path_mismatch(self) -> None:
        index = self._read_index()
        index["protocols"][0]["engine_ini_map_object_path"] = (
            "/Game/Carla/Maps/Town02.Town02"
        )
        self._write_json(self.index_path, index)
        with self.assertRaisesRegex(RunnerPlanError, "engine_ini_map_object_path does not bind"):
            build_runner_plan(self.index_path, base_rpc_port=24000, port_group_stride=3)

    def test_rejects_protocol_environment_map_mismatch(self) -> None:
        index = self._read_index()
        entry = index["protocols"][0]
        protocol_path = self.root / entry["protocol_path"]
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["environment"]["map"] = "Carla/Maps/Town05"
        self._rewrite_protocol_entry(index, entry, protocol)
        with self.assertRaisesRegex(RunnerPlanError, "environment.map differs"):
            build_runner_plan(self.index_path, base_rpc_port=24000, port_group_stride=3)

    def test_rejects_registry_hash_drift(self) -> None:
        index = self._read_index()
        registry_path = self.root / index["registries"]["asset_registry"]["path"]
        registry_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(RunnerPlanError, "asset_registry hash differs"):
            build_runner_plan(self.index_path, base_rpc_port=24000, port_group_stride=3)

    def test_rejects_legacy_shared_port_range(self) -> None:
        with self.assertRaisesRegex(RunnerPlanError, "2000..2022"):
            build_runner_plan(self.index_path, base_rpc_port=2000, port_group_stride=3)

    def test_rejects_overlapping_port_groups(self) -> None:
        with self.assertRaisesRegex(RunnerPlanError, "at least 3"):
            build_runner_plan(self.index_path, base_rpc_port=24000, port_group_stride=2)

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 is unavailable")
    def test_powershell_runner_parses_without_execution(self) -> None:
        escaped_path = str(RUNNER_PATH).replace("'", "''")
        command = (
            "$tokens=$null; $errors=$null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile('{escaped_path}', "
            "[ref]$tokens, [ref]$errors) | Out-Null; "
            "if ($errors.Count -ne 0) { $errors | ForEach-Object { $_.Message }; exit 2 }"
        )
        completed = subprocess.run(
            ["pwsh", "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 is unavailable")
    def test_powershell_plan_only_uses_helper_without_reserving_evidence(self) -> None:
        evidence_root = self.root / "must-not-be-created"
        completed = subprocess.run(
            [
                "pwsh",
                "-NoProfile",
                "-File",
                str(RUNNER_PATH),
                "-RunId",
                "static_plan",
                "-CompiledProtocol",
                str(self.index_path),
                "-CarlaPython",
                sys.executable,
                "-RawEvidenceRoot",
                str(evidence_root),
                "-PlanOnly",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        plan = json.loads(completed.stdout)
        self.assertEqual(4, plan["map_count"])
        self.assertEqual(16, plan["shard_count"])
        self.assertFalse(evidence_root.exists())

    @unittest.skipUnless(
        sys.platform == "win32" and shutil.which("pwsh"),
        "Windows PowerShell 7 networking cmdlets are unavailable",
    )
    def test_mock_children_emit_exact_runtime_index_and_complete_resume_skips(self) -> None:
        evidence_root = self.root / "mock-evidence"
        mock_c2 = self.root / "mock-c2.ps1"
        mock_join = self.root / "mock-join.py"
        mock_c2.write_text(
            """[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$RunId,
    [string]$CarlaRoot,
    [string]$CarlaPython,
    [string]$RawEvidenceRoot,
    [string]$Protocol,
    [int]$RpcPort,
    [string]$StartupEngineIni,
    [int]$CaptureTimeoutSeconds,
    [double]$MinimumFreePhysicalGB,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
if ($MinimumFreePhysicalGB -ne 3.0) {
    throw "C4 did not forward MinimumFreePhysicalGB: $MinimumFreePhysicalGB"
}
$child = Join-Path $RawEvidenceRoot $RunId
[IO.Directory]::CreateDirectory($child) | Out-Null
$frozen = Join-Path $child 'frozen_protocol.json'
Copy-Item -LiteralPath $Protocol -Destination $frozen
$hash = (Get-FileHash -LiteralPath $frozen -Algorithm SHA256).Hash
$result = [ordered]@{
    experiment_id = 'DTR_CARLA_C2_RICH_MULTILAYOUT_OCCLUSION_SOURCE_V2'
    status = 'DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE'
    protocol_sha256 = $hash
    checks = [ordered]@{ mock_child_contract = $true }
}
[IO.File]::WriteAllText(
    (Join-Path $child 'result.json'),
    (($result | ConvertTo-Json -Depth 10) + [Environment]::NewLine),
    [Text.UTF8Encoding]::new($false)
)
""",
            encoding="utf-8",
        )
        mock_join.write_text(
            """from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--compiled-protocol", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
args = parser.parse_args()
index = json.loads(args.compiled_protocol.read_text(encoding="utf-8"))
assert set(index) == {
    "schema_version", "experiment_id", "capture", "registries", "admission",
    "map_layout_groups",
}
assert index["schema_version"] == "dtr-carla-c4-multimap-compiled-v1"
assert set(index["registries"]) == {"asset_registry", "scene_registry"}
for link in index["registries"].values():
    assert set(link) == {"path", "sha256"}
    target = (args.compiled_protocol.parent / link["path"]).resolve()
    assert hashlib.sha256(target.read_bytes()).hexdigest().upper() == link["sha256"]
assert index["admission"] == {
    "expected_map_count": 4,
    "expected_protocol_count": 4,
    "expected_layout_count": 8,
    "expected_episode_count": 8,
    "expected_sensor_count": 4,
    "expected_shard_count": 16,
}
assert len(index["map_layout_groups"]) == 4
for group in index["map_layout_groups"]:
    assert set(group) == {
        "group_id", "map", "startup_map_argument", "layout_ids", "protocol_path",
        "protocol_sha256", "evidence_path", "evidence_result_sha256",
    }
    leaf = group["map"].rsplit("/", 1)[-1]
    assert group["startup_map_argument"] == f"/Game/Carla/Maps/{leaf}.{leaf}"
    result_path = (
        args.compiled_protocol.parent / group["evidence_path"] / "result.json"
    ).resolve()
    assert hashlib.sha256(result_path.read_bytes()).hexdigest().upper() == group[
        "evidence_result_sha256"
    ]
args.output_root.mkdir(parents=True)
(args.output_root / "result.json").write_text(
    json.dumps({
        "experiment_id": "DTR_CARLA_C4_MULTIMAP_WORLD_PACK_V1",
        "status": "DTR_CARLA_C4_MULTIMAP_SOURCE_COMPLETE",
        "checks": {"mock_join_contract": True},
        "index_sha256": hashlib.sha256(args.compiled_protocol.read_bytes()).hexdigest().upper(),
    }) + "\\n",
    encoding="utf-8",
)
""",
            encoding="utf-8",
        )
        base_command = [
            "pwsh",
            "-NoProfile",
            "-File",
            str(RUNNER_PATH),
            "-RunId",
            "mock_run",
            "-CompiledProtocol",
            str(self.index_path),
            "-CarlaRoot",
            str(self.root / "unused-carla"),
            "-CarlaPython",
            sys.executable,
            "-RawEvidenceRoot",
            str(evidence_root),
            "-C2Runner",
            str(mock_c2),
            "-JoinScript",
            str(mock_join),
            "-BaseRpcPort",
            "48000",
            "-MinimumFreePhysicalGB",
            "3.0",
            "-CooldownSeconds",
            "0",
        ]
        completed = subprocess.run(
            base_command,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        runtime_path = (
            evidence_root / "mock_run" / "frozen-inputs" / "runtime-compiled-protocol.json"
        )
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        self.assertEqual(
            {
                "schema_version",
                "experiment_id",
                "capture",
                "registries",
                "admission",
                "map_layout_groups",
            },
            set(runtime),
        )
        self.assertEqual(6, len(runtime["admission"]))

        mock_c2.write_text(
            """[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$RunId,
    [string]$CarlaRoot,
    [string]$CarlaPython,
    [string]$RawEvidenceRoot,
    [string]$Protocol,
    [int]$RpcPort,
    [string]$StartupEngineIni,
    [int]$CaptureTimeoutSeconds,
    [double]$MinimumFreePhysicalGB
)
throw 'complete child must have been skipped'
""",
            encoding="utf-8",
        )
        resumed = subprocess.run(
            [*base_command, "-Resume"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, resumed.returncode, resumed.stdout + resumed.stderr)
        self.assertIn("SKIP complete C4 final package", resumed.stdout)

        reuse_command = list(base_command)
        reuse_command[reuse_command.index("-RunId") + 1] = "mock_reuse"
        reuse_command.extend(
            ["-ReuseChildEvidenceRoot", str(evidence_root / "mock_run")]
        )
        reused = subprocess.run(
            reuse_command,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, reused.returncode, reused.stdout + reused.stderr)
        self.assertEqual(4, reused.stdout.count("REUSE verified C4 child"))
        reuse_receipt = json.loads(
            (
                evidence_root
                / "mock_reuse"
                / "reused_children_receipt.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(4, reuse_receipt["reused_group_count"])
        self.assertTrue(
            all(value["copy_verified_complete"] for value in reuse_receipt["groups"])
        )

    def test_runner_source_has_required_resource_and_resume_boundaries(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        lower = source.lower()
        self.assertNotIn("client.load_world", lower)
        self.assertNotIn("capture_dtr_carla_c2_rich_scene.py", source)
        self.assertIn("run_dtr_carla_c2_rich_scene.ps1", source)
        self.assertIn("-StartupEngineIni", source)
        self.assertIn("-MinimumFreePhysicalGB", source)
        self.assertIn("GameDefaultMap=", source)
        self.assertIn("startup_engine_ini_sha256", source)
        self.assertIn("Refusing evidence overwrite", source)
        self.assertIn("Test-CompletedChild", source)
        self.assertIn("-Resume", source)
        self.assertIn("evidence_result_sha256", source)
        self.assertIn("runtime-compiled-protocol.json", source)
        self.assertIn("ReuseChildEvidenceRoot", source)
        self.assertIn("Import-ReusableChildren", source)
        self.assertIn("REUSE verified C4 child", source)
        self.assertIn("reused_children_receipt.json", source)
        self.assertIn("--compiled-protocol", source)
        self.assertIn("DTR_CARLA_C4_MULTIMAP_SOURCE_COMPLETE", source)

        c2_source = C2_RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("Get-OwnedCarlaProcesses", c2_source)
        self.assertIn("New-RuntimeStartupEngineIni", c2_source)
        self.assertIn("Remove-RuntimeStartupEngineIni", c2_source)
        self.assertIn('-EngineIni=$runtimeStartupEngineIniPath', c2_source)
        self.assertIn("Clear-NonTerminalJoinArtifacts", c2_source)
        self.assertIn("immutable sensor shards preserved", c2_source)
        self.assertIn("Stop-OwnedPython", c2_source)
        self.assertIn("Stop-OwnedCarla", c2_source)
        self.assertIn("Assert-CarlaIdle", c2_source)
        self.assertIn("fresh-server shard", c2_source)


if __name__ == "__main__":
    unittest.main()
