#!/usr/bin/env python3
"""Event-driven asset lifecycle coordinator for BlindAssist experiments.

The coordinator resolves and records catalog inputs before execution, runs one
native argv without a shell, then registers reusable caches, hard cases, thin
results, output assets, and reports.  It never performs a full artifact-tree
walk; only run-touched asset units and the small resource-fabric indexes are
reconciled.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import asset_catalog as catalog  # noqa: E402
import resource_fabric as fabric  # noqa: E402


RUN_SCHEMA = "blindassist-asset-run-v1"
TERMINAL_STATES = {"succeeded", "failed", "interrupted", "management_failed"}
TOKEN_PATTERN = re.compile(r"\{\{(input|cache|output):([A-Za-z0-9_.-]+)\}\}")


class RuntimeError_(RuntimeError):
    """A user-correctable lifecycle coordination failure."""


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError_(f"Run spec requires non-empty {label}")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError_(f"Run spec {label} must be a JSON array")
    return value


def resolve_repo_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def ensure_artifact_path(path: Path, artifact_root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(artifact_root.resolve())
    except ValueError as exc:
        raise RuntimeError_(f"{label} escapes artifacts.local: {resolved}") from exc
    return resolved


def portable_path(path: Path, artifact_root: Path) -> str:
    return path.resolve().relative_to(artifact_root.resolve()).as_posix()


def alias_map(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError_(f"Run spec {label} entries must be JSON objects")
        alias = require_text(record.get("alias"), f"{label}.alias")
        if alias in mapped:
            raise RuntimeError_(f"Duplicate {label} alias: {alias}")
        mapped[alias] = record
    return mapped


def load_spec(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError_(f"Cannot read run spec {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != RUN_SCHEMA:
        raise RuntimeError_(f"Unsupported run spec schema: {value.get('schema') if isinstance(value, dict) else None}")
    require_text(value.get("id"), "id")
    require_text(value.get("route"), "route")
    require_text(value.get("question"), "question")
    require_text(value.get("evaluator"), "evaluator")
    require_text(value.get("evidence_boundary"), "evidence_boundary")
    command = require_list(value.get("command"), "command")
    if not command or not all(isinstance(item, str) and item for item in command):
        raise RuntimeError_("Run spec command must be a non-empty string argv")
    alias_map(require_list(value.get("inputs"), "inputs"), "inputs")
    alias_map(require_list(value.get("cache_inputs"), "cache_inputs"), "cache_inputs")
    alias_map(require_list(value.get("outputs"), "outputs"), "outputs")
    alias_map(require_list(value.get("cache_outputs"), "cache_outputs"), "cache_outputs")
    parameters = value.get("parameters", {})
    if not isinstance(parameters, dict):
        raise RuntimeError_("Run spec parameters must be a JSON object")
    return value


def journal_path(artifact_root: Path, route: str, run_id: str) -> Path:
    return (
        fabric.fabric_root(artifact_root)
        / "runs"
        / fabric.slug(route)
        / f"{fabric.slug(run_id)}.json"
    )


def write_journal(path: Path, journal: dict[str, Any], state: str, **updates: Any) -> None:
    journal.update(updates)
    journal["state"] = state
    journal["updated_at"] = fabric.utc_now()
    fabric.atomic_write_json(path, journal)


def refuse_existing_journal(path: Path) -> None:
    if not path.exists():
        return
    existing = fabric.read_json(path)
    state = existing.get("state", "unknown")
    if state in TERMINAL_STATES:
        raise RuntimeError_(
            f"Run id already has terminal state {state}; use a new id: {path}"
        )
    raise RuntimeError_(
        f"Run id has non-terminal state {state}; automatic replay is refused as in_doubt: {path}"
    )


def reconcile_input_path(
    path: Path,
    *,
    artifact_root: Path,
    repo_root: Path,
    database: Path,
    policy: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    connection = catalog.open_catalog(database)
    try:
        with connection:
            return catalog.reconcile_asset_path(
                connection,
                path=path,
                artifact_root=artifact_root,
                repo_root=repo_root,
                policy=policy,
                scan_id=f"run-input:{fabric.slug(run_id)}:{uuid.uuid4().hex[:8]}",
                now=catalog.utc_now(),
            )
    finally:
        connection.close()


def resolve_inputs(
    spec: dict[str, Any],
    *,
    artifact_root: Path,
    repo_root: Path,
    database: Path,
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    snapshots: list[dict[str, Any]] = []
    paths: dict[str, str] = {}
    run_id = spec["id"]
    for alias, item in alias_map(require_list(spec.get("inputs"), "inputs"), "inputs").items():
        has_asset = bool(item.get("asset"))
        has_path = bool(item.get("path"))
        if has_asset == has_path:
            raise RuntimeError_(f"Input {alias} requires exactly one of asset or path")
        requested_path: Path | None = None
        if has_path:
            requested_path = ensure_artifact_path(
                resolve_repo_path(require_text(item.get("path"), f"input {alias}.path"), repo_root),
                artifact_root,
                f"Input {alias}",
            )
            if not requested_path.exists():
                raise RuntimeError_(f"Input {alias} is missing: {requested_path}")
            reconciled = reconcile_input_path(
                requested_path,
                artifact_root=artifact_root,
                repo_root=repo_root,
                database=database,
                policy=policy,
                run_id=run_id,
            )
            selector = reconciled["asset_id"]
        else:
            selector = require_text(item.get("asset"), f"input {alias}.asset")

        snapshot, resolved = fabric.resolve_master_asset_input(
            artifact_root,
            selector,
            consumer=run_id,
            purpose=item.get("purpose") or "experiment-input",
            experiment_id=run_id,
            event_id=f"fabric-experiment:{fabric.slug(run_id)}:{selector.split('#', 1)[0]}",
        )
        actual_path = requested_path or resolved
        snapshot = dict(snapshot)
        snapshot["alias"] = alias
        snapshot["purpose"] = item.get("purpose") or "experiment-input"
        if requested_path is not None:
            snapshot["requested_path"] = portable_path(requested_path, artifact_root)
            snapshot["requested_relative_path"] = reconciled["requested_relative_path"]
            if requested_path.is_file():
                snapshot["requested_content_id"] = f"sha256:{fabric.sha256_file(requested_path)}"
                snapshot["requested_bytes"] = requested_path.stat().st_size
        snapshots.append(snapshot)
        paths[alias] = str(actual_path)
    return snapshots, paths


def resolve_cache_inputs(
    spec: dict[str, Any], artifact_root: Path
) -> tuple[list[str], dict[str, str], list[dict[str, Any]]]:
    cache_keys: list[str] = []
    paths: dict[str, str] = {}
    events: list[dict[str, Any]] = []
    run_id = spec["id"]
    for alias, item in alias_map(
        require_list(spec.get("cache_inputs"), "cache_inputs"), "cache_inputs"
    ).items():
        key = require_text(item.get("cache_key"), f"cache input {alias}.cache_key")
        event = fabric.cache_use_command(
            argparse.Namespace(
                artifact_root=artifact_root,
                cache_key=key,
                event_id=f"{run_id}-{alias}-cache-hit",
                consumer=run_id,
                purpose=item.get("purpose") or "experiment-cache-input",
                experiment_id=run_id,
            )
        )
        cache_keys.append(key)
        paths[alias] = event["resolved_payload"]
        events.append(event)
    return sorted(set(cache_keys)), paths, events


def resolve_outputs(
    spec: dict[str, Any], repo_root: Path, artifact_root: Path
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    paths: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    for alias, item in alias_map(require_list(spec.get("outputs"), "outputs"), "outputs").items():
        path = ensure_artifact_path(
            resolve_repo_path(require_text(item.get("path"), f"output {alias}.path"), repo_root),
            artifact_root,
            f"Output {alias}",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        paths[alias] = path
        records.append(
            {
                "alias": alias,
                "path": portable_path(path, artifact_root),
                "role": item.get("role") or "output",
                "required": bool(item.get("required", True)),
            }
        )
    result_alias = spec.get("result_output")
    if result_alias is not None and result_alias not in paths:
        raise RuntimeError_(f"Unknown result_output alias: {result_alias}")
    return paths, records


def substitute_command(
    command: list[str],
    input_paths: dict[str, str],
    cache_paths: dict[str, str],
    output_paths: dict[str, Path],
) -> list[str]:
    groups: dict[str, dict[str, str]] = {
        "input": input_paths,
        "cache": cache_paths,
        "output": {key: str(value) for key, value in output_paths.items()},
    }

    def replace(match: re.Match[str]) -> str:
        kind, alias = match.groups()
        try:
            return groups[kind][alias]
        except KeyError as exc:
            raise RuntimeError_(f"Command references unknown {kind} alias: {alias}") from exc

    resolved = [TOKEN_PATTERN.sub(replace, item) for item in command]
    unresolved = [item for item in resolved if "{{" in item or "}}" in item]
    if unresolved:
        raise RuntimeError_(f"Command contains unresolved template tokens: {unresolved}")
    return resolved


def output_receipts(
    output_paths: dict[str, Path], output_records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    receipts: list[dict[str, Any]] = []
    missing: list[str] = []
    by_alias = {item["alias"]: item for item in output_records}
    for alias, path in output_paths.items():
        if not path.exists():
            if by_alias[alias]["required"]:
                missing.append(alias)
            receipts.append({**by_alias[alias], "state": "missing"})
            continue
        scan = fabric.scan_payload(path)
        receipts.append(
            {
                **by_alias[alias],
                "state": "present",
                "payload_type": scan["payload_type"],
                "content_id": scan["resource_id"],
                "bytes": scan["bytes"],
                "file_count": scan["file_count"],
            }
        )
    return receipts, missing


def create_declared_caches(
    spec: dict[str, Any],
    *,
    artifact_root: Path,
    output_paths: dict[str, Path],
    asset_inputs: list[dict[str, Any]],
    cache_input_keys: list[str],
) -> list[dict[str, Any]]:
    created: list[dict[str, Any]] = []
    input_by_alias = {item["alias"]: item for item in asset_inputs}
    for alias, item in alias_map(
        require_list(spec.get("cache_outputs"), "cache_outputs"), "cache_outputs"
    ).items():
        output_alias = require_text(
            item.get("output"), f"cache output {alias}.output"
        )
        if output_alias not in output_paths:
            raise RuntimeError_(
                f"Cache output {alias} references unknown output alias: {output_alias}"
            )
        payload = output_paths[output_alias]
        if not payload.exists():
            raise RuntimeError_(f"Cache output payload is missing: {payload}")
        selected_aliases = item.get("source_inputs")
        if selected_aliases is None:
            selected = asset_inputs
        else:
            if not isinstance(selected_aliases, list):
                raise RuntimeError_(f"Cache output {alias}.source_inputs must be an array")
            try:
                selected = [input_by_alias[value] for value in selected_aliases]
            except KeyError as exc:
                raise RuntimeError_(
                    f"Cache output {alias} references unknown input alias: {exc.args[0]}"
                ) from exc
        parent_keys = item.get("parent_cache_keys", cache_input_keys)
        if not isinstance(parent_keys, list):
            raise RuntimeError_(f"Cache output {alias}.parent_cache_keys must be an array")
        cache = fabric.create_cache(
            artifact_root,
            layer=item.get("layer") or "features",
            source_ids=[],
            parent_cache_keys=parent_keys,
            model_ids=[],
            transform=require_text(item.get("transform"), f"cache output {alias}.transform"),
            transform_version=require_text(
                item.get("transform_version"),
                f"cache output {alias}.transform_version",
            ),
            parameters=item.get("parameters", {}),
            producer=item.get("producer") or spec["evaluator"],
            payload_source=payload,
            mode=item.get("mode") or "hardlink",
            code_sha256=item.get("code_sha256"),
            config_sha256=item.get("config_sha256"),
            asset_inputs=selected,
        )
        created.append({"alias": alias, **cache})
    return created


def result_value(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        return None
    try:
        value = fabric.read_json(path)
    except fabric.FabricError:
        return None
    return value if isinstance(value, dict) else None


def register_result_hard_cases(
    result: dict[str, Any],
    *,
    spec: dict[str, Any],
    artifact_root: Path,
    asset_inputs: list[dict[str, Any]],
    cache_keys: list[str],
) -> list[dict[str, Any]]:
    lifecycle = result.get("asset_lifecycle", {})
    declarations = lifecycle.get("hard_cases", []) if isinstance(lifecycle, dict) else []
    if declarations is None:
        return []
    if not isinstance(declarations, list):
        raise RuntimeError_("result.asset_lifecycle.hard_cases must be an array")
    input_by_alias = {item["alias"]: item for item in asset_inputs}
    records: list[dict[str, Any]] = []
    for index, declaration in enumerate(declarations):
        if not isinstance(declaration, dict):
            raise RuntimeError_("Each result hard-case declaration must be an object")
        aliases = declaration.get("input_aliases")
        selected_assets = asset_inputs
        if aliases is not None:
            if not isinstance(aliases, list):
                raise RuntimeError_("hard-case input_aliases must be an array")
            try:
                selected_assets = [input_by_alias[value] for value in aliases]
            except KeyError as exc:
                raise RuntimeError_(f"Hard case references unknown input alias: {exc.args[0]}") from exc
        records.append(
            fabric.create_hard_case(
                artifact_root,
                hard_case_id=declaration.get("id") or f"{spec['id']}-hard-case-{index + 1}",
                route=spec["route"],
                case_kind=declaration.get("case_kind") or "hard_case",
                failure_layer=require_text(
                    declaration.get("failure_layer"), "hard-case failure_layer"
                ),
                evidence_split=declaration.get("evidence_split") or "development",
                source_ids=[],
                cache_keys=cache_keys,
                asset_inputs=selected_assets,
                selector=declaration.get("selector", {}),
                truth_authority=declaration.get("truth_authority") or "evaluator-output",
                selected_by=declaration.get("selected_by") or spec["evaluator"],
                observed_outcome=require_text(
                    declaration.get("observed_outcome"), "hard-case observed_outcome"
                ),
                claim_ceiling=declaration.get("claim_ceiling") or spec["evidence_boundary"],
                allowed_uses=declaration.get("allowed_uses", []),
                forbidden_uses=declaration.get("forbidden_uses", []),
            )
        )
    return records


def register_execution_failure(
    *,
    spec: dict[str, Any],
    artifact_root: Path,
    asset_inputs: list[dict[str, Any]],
    cache_keys: list[str],
    exit_code: int,
    failure_layer: str,
) -> dict[str, Any]:
    return fabric.create_hard_case(
        artifact_root,
        hard_case_id=f"{spec['id']}-{failure_layer}-failure",
        route=spec["route"],
        case_kind="failure",
        failure_layer=failure_layer,
        evidence_split="development",
        source_ids=[],
        cache_keys=cache_keys,
        asset_inputs=asset_inputs,
        selector={"run_id": spec["id"], "exit_code": exit_code},
        truth_authority="process-and-output-contract-receipt",
        selected_by="tools/data/asset_runtime.py",
        observed_outcome=f"Run terminated at {failure_layer} with exit code {exit_code}",
        claim_ceiling="Execution failure only; no task-domain conclusion.",
        allowed_uses=[],
        forbidden_uses=[],
    )


def refresh_governance(
    *,
    artifact_root: Path,
    repo_root: Path,
    database: Path,
    policy: dict[str, Any],
    output_paths: dict[str, Path],
    run_id: str,
) -> dict[str, Any]:
    connection = catalog.open_catalog(database)
    reconciled = []
    fabric_root = fabric.fabric_root(artifact_root).resolve()
    try:
        with connection:
            for path in output_paths.values():
                if not path.exists():
                    continue
                try:
                    path.resolve().relative_to(fabric_root)
                    continue
                except ValueError:
                    pass
                reconciled.append(
                    catalog.reconcile_asset_path(
                        connection,
                        path=path,
                        artifact_root=artifact_root,
                        repo_root=repo_root,
                        policy=policy,
                        scan_id=f"run-output:{fabric.slug(run_id)}:{uuid.uuid4().hex[:8]}",
                        now=catalog.utc_now(),
                    )
                )
            synced = catalog.sync_resource_fabric(
                connection,
                artifact_root,
                f"run-fabric:{fabric.slug(run_id)}",
                catalog.utc_now(),
            )
    finally:
        connection.close()
    master_report = catalog.report_command(
        argparse.Namespace(
            artifact_root=artifact_root,
            database=database,
            output_dir=artifact_root / catalog.DEFAULT_REPORT_RELATIVE,
        )
    )
    master_verify = catalog.verify_command(
        argparse.Namespace(
            artifact_root=artifact_root,
            database=database,
            repo_root=repo_root,
            deep=False,
        )
    )
    fabric_report = fabric.report_command(
        argparse.Namespace(
            artifact_root=artifact_root,
            output_dir=None,
            inventory_root=None,
        )
    )
    fabric_verify = fabric.verify_command(
        argparse.Namespace(artifact_root=artifact_root, deep=False)
    )
    return {
        "reconciled_outputs": reconciled,
        "fabric_sync": synced,
        "master_report": master_report,
        "master_verify": master_verify,
        "fabric_report": fabric_report,
        "fabric_verify": fabric_verify,
    }


def run_spec(
    spec_path: Path,
    *,
    repo_root: Path,
    artifact_root: Path,
    policy_path: Path,
) -> tuple[dict[str, Any], int]:
    spec = load_spec(spec_path)
    repo_root = repo_root.resolve()
    artifact_root = artifact_root.resolve()
    policy = catalog.load_policy(policy_path.resolve())
    database = artifact_root / catalog.DEFAULT_DATABASE_RELATIVE
    if not database.is_file():
        raise RuntimeError_(
            f"Master asset catalog is missing: {database}; run asset_catalog.py discover first"
        )
    run_id = spec["id"]
    route = spec["route"]
    log_path = journal_path(artifact_root, route, run_id)
    refuse_existing_journal(log_path)

    journal = {
        "schema": "blindassist-asset-run-journal-v1",
        "run_schema": RUN_SCHEMA,
        "id": fabric.slug(run_id),
        "route": fabric.slug(route),
        "spec_sha256": fabric.sha256_file(spec_path),
        "declared_command": spec["command"],
        "created_at": fabric.utc_now(),
    }
    write_journal(log_path, journal, "preparing")
    try:
        asset_inputs, input_paths = resolve_inputs(
            spec,
            artifact_root=artifact_root,
            repo_root=repo_root,
            database=database,
            policy=policy,
        )
        cache_keys, cache_paths, cache_events = resolve_cache_inputs(spec, artifact_root)
        output_paths, output_records = resolve_outputs(spec, repo_root, artifact_root)
        command = substitute_command(spec["command"], input_paths, cache_paths, output_paths)
        fabric.create_experiment(
            artifact_root,
            experiment_id=run_id,
            route=route,
            question=spec["question"],
            evaluator=spec["evaluator"],
            status="prepared",
            source_ids=[],
            cache_keys=cache_keys,
            asset_inputs=asset_inputs,
            hard_case_ids=[],
            parameters=spec.get("parameters", {}),
            boundary=spec["evidence_boundary"],
        )
        write_journal(
            log_path,
            journal,
            "prepared",
            command=command,
            asset_inputs=asset_inputs,
            cache_inputs=cache_events,
            outputs=output_records,
        )
    except Exception as exc:
        write_journal(
            log_path,
            journal,
            "management_failed",
            finished_at=fabric.utc_now(),
            native_exit_code=None,
            management_error=str(exc),
        )
        raise
    write_journal(log_path, journal, "running", started_at=fabric.utc_now())

    try:
        completed = subprocess.run(command, cwd=repo_root, shell=False, check=False)
        native_exit = int(completed.returncode)
    except KeyboardInterrupt:
        write_journal(
            log_path,
            journal,
            "interrupted",
            finished_at=fabric.utc_now(),
            native_exit_code=130,
        )
        raise
    except Exception as exc:
        write_journal(
            log_path,
            journal,
            "management_failed",
            finished_at=fabric.utc_now(),
            native_exit_code=None,
            management_error=str(exc),
        )
        raise

    temporary_dir = artifact_root / "tmp" / "asset-runtime" / f"{fabric.slug(run_id)}-{uuid.uuid4().hex[:8]}"
    try:
        receipts, missing_outputs = output_receipts(output_paths, output_records)
        contract_ok = not missing_outputs
        success = native_exit == 0 and contract_ok
        created_caches = (
            create_declared_caches(
                spec,
                artifact_root=artifact_root,
                output_paths=output_paths,
                asset_inputs=asset_inputs,
                cache_input_keys=cache_keys,
            )
            if success
            else []
        )
        produced_cache_keys = [item["cache_key"] for item in created_caches]
        result_alias = spec.get("result_output")
        native_result_path = output_paths.get(result_alias) if result_alias else None
        result = result_value(native_result_path)
        if result is None:
            temporary_dir.mkdir(parents=True, exist_ok=True)
            result_path = temporary_dir / "terminal-result.json"
            result = {
                "status": (
                    "PASS"
                    if success
                    else f"FAILED_EXIT_{native_exit}"
                    if native_exit != 0
                    else "FAILED_OUTPUT_CONTRACT"
                ),
                "run_id": run_id,
                "native_exit_code": native_exit,
                "missing_required_outputs": missing_outputs,
                "outputs": receipts,
            }
            fabric.atomic_write_json(result_path, result)
        else:
            result_path = native_result_path

        hard_cases = register_result_hard_cases(
            result,
            spec=spec,
            artifact_root=artifact_root,
            asset_inputs=asset_inputs,
            cache_keys=cache_keys + produced_cache_keys,
        )
        if not success:
            hard_cases.append(
                register_execution_failure(
                    spec=spec,
                    artifact_root=artifact_root,
                    asset_inputs=asset_inputs,
                    cache_keys=cache_keys,
                    exit_code=native_exit,
                    failure_layer="execution" if native_exit != 0 else "output-contract",
                )
            )
        experiment_status = (
            str(result.get("status"))
            if result.get("status") is not None
            else "PASS" if success else "FAILED"
        )
        finalized = fabric.finalize_experiment(
            artifact_root,
            route=route,
            experiment_id=run_id,
            result_json=result_path,
            status=experiment_status,
            produced_cache_keys=produced_cache_keys,
        )
        receipts, _ = output_receipts(output_paths, output_records)
        governance = refresh_governance(
            artifact_root=artifact_root,
            repo_root=repo_root,
            database=database,
            policy=policy,
            output_paths=output_paths,
            run_id=run_id,
        )
        state = "succeeded" if success else "failed"
        final = {
            "status": "PASS" if success else "FAIL",
            "run_id": fabric.slug(run_id),
            "route": fabric.slug(route),
            "state": state,
            "native_exit_code": native_exit,
            "missing_required_outputs": missing_outputs,
            "experiment": finalized,
            "created_caches": created_caches,
            "hard_cases": hard_cases,
            "outputs": receipts,
            "journal": portable_path(log_path, artifact_root),
            "governance": governance,
        }
        write_journal(
            log_path,
            journal,
            state,
            finished_at=fabric.utc_now(),
            native_exit_code=native_exit,
            missing_required_outputs=missing_outputs,
            experiment=finalized,
            created_cache_keys=produced_cache_keys,
            hard_cases=[item["hard_case"] for item in hard_cases],
            output_receipts=receipts,
        )
        if success:
            return final, 0
        return final, native_exit if native_exit != 0 else 2
    except Exception as exc:
        write_journal(
            log_path,
            journal,
            "management_failed",
            finished_at=fabric.utc_now(),
            native_exit_code=native_exit,
            management_error=str(exc),
        )
        raise
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Execute one governed run specification")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, default=catalog.DEFAULT_REPO_ROOT)
    run.add_argument("--artifact-root", type=Path, default=catalog.DEFAULT_ARTIFACT_ROOT)
    run.add_argument("--policy", type=Path, default=catalog.DEFAULT_POLICY_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result, code = run_spec(
            args.spec.resolve(),
            repo_root=args.repo_root,
            artifact_root=args.artifact_root,
            policy_path=args.policy,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return code
    except KeyboardInterrupt:
        print(json.dumps({"status": "INTERRUPTED"}, ensure_ascii=False), file=sys.stderr)
        return 130
    except (RuntimeError_, catalog.CatalogError, fabric.FabricError, OSError, ValueError) as exc:
        print(
            json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
