from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from pb_h1_role_proxy.experiment import build_result


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_paths(repo_root: Path) -> dict[str, Path]:
    module = (
        repo_root
        / "scripts/research/egomotion_compensated_looming"
    )
    return {
        "runner": Path(__file__).resolve(),
        "geometry": module / "pb_h1_role_proxy/geometry.py",
        "experiment": module / "pb_h1_role_proxy/experiment.py",
    }


def _validate_existing(repo_root: Path, output_root: Path) -> dict[str, Any]:
    result_path = output_root / "result.json"
    receipt_path = output_root / "receipt.json"
    result_payload = result_path.read_bytes()
    result = json.loads(result_payload)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if sha256(result_payload).hexdigest() != receipt.get("result_sha256"):
        errors.append("RESULT_SHA256")
    if result.get("terminal") != receipt.get("terminal"):
        errors.append("TERMINAL")
    if receipt.get("execution_validity") != "VALID":
        errors.append("EXECUTION_VALIDITY")
    fixture = result["controlled_fixture"]
    if fixture.get("physical_calibration_pass") is not bool(
        all(fixture["checks"].values())
    ):
        errors.append("FIXTURE_CHECK_AGGREGATION")
    bonn = result["burned_bonn_window"]
    expected_bonn_evaluable = bool(
        bonn["summary"]["pair_coverage"] >= 0.80
        and bonn["summary"]["median_valid_fraction"] >= 0.50
    )
    if (
        result["conclusions"]["bonn_diagnostic_evaluable"]
        is not expected_bonn_evaluable
    ):
        errors.append("BONN_EVALUABILITY")
    ledger_path = (
        repo_root
        / "artifacts.local/evidence/rcle_phase_b_bonn_b1/"
        "b1a_geometry_admission/ledger.json"
    )
    archive_path = (
        repo_root
        / "artifacts.local/datasets/rcle_phase_b_bonn_b0_r1/archives/"
        "rgbd_bonn_crowd2.zip"
    )
    if _sha256_file(ledger_path) != bonn["ledger_sha256"]:
        errors.append("BONN_LEDGER_SHA256")
    if _sha256_file(archive_path) != bonn["archive_sha256"]:
        errors.append("BONN_ARCHIVE_SHA256")
    for name, path in _implementation_paths(repo_root).items():
        if _sha256_file(path) != result["implementation_sha256"].get(name):
            errors.append(f"IMPLEMENTATION_SHA256:{name}")
    return {
        "schema_version": "rcle.pb_h1_role_proxy.validation.v1",
        "status": "VALID" if not errors else "INVALID",
        "errors": errors,
        "result_sha256": sha256(result_payload).hexdigest(),
        "terminal": result["terminal"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts.local/evidence/rcle_pb_h1_role_proxy_r0/discovery_r0"
        ),
    )
    parser.add_argument("--validate-existing", action="store_true")
    arguments = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    if arguments.validate_existing:
        validation = _validate_existing(repo_root, arguments.output_root)
        print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
        return 0 if validation["status"] == "VALID" else 1
    result = build_result(repo_root)
    result["implementation_sha256"] = {
        name: _sha256_file(path)
        for name, path in _implementation_paths(repo_root).items()
    }
    result_payload = _canonical_json(result)
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    result_path = arguments.output_root / "result.json"
    receipt_path = arguments.output_root / "receipt.json"
    result_path.write_bytes(result_payload)
    receipt = {
        "schema_version": "rcle.pb_h1_role_proxy.receipt.v1",
        "result_sha256": sha256(result_payload).hexdigest(),
        "result_path": str(result_path.resolve()),
        "terminal": result["terminal"],
        "execution_validity": "VALID",
        "scientific_outcome": (
            "SUPPORT" if result["conclusions"][
                "tum_fr2_rpy_audit_worthwhile"
            ] else "STOP"
        ),
    }
    receipt_path.write_bytes(_canonical_json(receipt))
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
