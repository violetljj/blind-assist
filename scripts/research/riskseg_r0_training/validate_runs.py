"""Freeze and cross-check the three fixed RISKSEG-R0 training runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


EXPECTED_SEEDS = (20260801, 20260802, 20260803)
DECISION_SEED = 20260801
EXPECTED_STATUS = "TRAINING_COMPLETE_DEV_SELECTED"
ALLOWED_STOP_REASONS = {"DEV_MIOU_EARLY_STOPPING", "MAX_EPOCHS"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def finite_number(value: Any, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def validate_run(run_dir: Path, expected_seed: int) -> dict[str, Any]:
    report_path = run_dir / "training_report.json"
    report = read_object(report_path)
    if (
        report.get("schema_version") != "blindassist.riskseg_r0.pidnet_training.v1"
        or report.get("protocol_id") != "RISKSEG-R0"
        or int(report.get("seed", -1)) != expected_seed
        or report.get("status") != EXPECTED_STATUS
    ):
        raise ValueError(f"{report_path}: training identity/status mismatch")
    if bool(report.get("decision_seed")) != (expected_seed == DECISION_SEED):
        raise ValueError(f"{report_path}: decision_seed mismatch")
    if report.get("event_eval_outcome_accessed_by_trainer") is not False:
        raise ValueError(f"{report_path}: event-eval firewall was not preserved")
    if report.get("stop_reason") not in ALLOWED_STOP_REASONS:
        raise ValueError(f"{report_path}: unexpected stop reason")

    recipe = report.get("recipe", {})
    epochs = int(report.get("epochs_completed", -1))
    minimum_epochs = int(recipe.get("minimum_epochs", -1))
    maximum_epochs = int(recipe.get("max_epochs", -1))
    if not minimum_epochs <= epochs <= maximum_epochs:
        raise ValueError(f"{report_path}: epoch count is outside the frozen recipe")
    if int(report.get("best_epoch", -1)) > epochs:
        raise ValueError(f"{report_path}: best epoch exceeds completed epochs")

    data = report.get("data", {})
    train_sessions = set(data.get("train_sessions", []))
    dev_sessions = set(data.get("dev_sessions", []))
    if train_sessions & dev_sessions or data.get("session_overlap") != []:
        raise ValueError(f"{report_path}: train/dev session leakage")

    checkpoint_path = run_dir / str(report["checkpoint_path"])
    history_path = run_dir / str(report["history_path"])
    checkpoint_sha256 = sha256_file(checkpoint_path)
    history_sha256 = sha256_file(history_path)
    if checkpoint_sha256 != report.get("checkpoint_sha256"):
        raise ValueError(f"{report_path}: checkpoint SHA-256 mismatch")
    if history_sha256 != report.get("history_sha256"):
        raise ValueError(f"{report_path}: history SHA-256 mismatch")

    history_rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(history_rows) != epochs:
        raise ValueError(f"{report_path}: history row count mismatch")
    if [int(row["epoch"]) for row in history_rows] != list(range(1, epochs + 1)):
        raise ValueError(f"{report_path}: history epoch sequence mismatch")

    metrics = report.get("best_dev_metrics", {})
    per_class = metrics.get("per_class_iou", {})
    class_order = tuple(recipe.get("class_order", []))
    if class_order != (
        "walkable",
        "blocking_obstacle",
        "boundary_level_change",
        "unknown_nonwalkable",
    ) or set(per_class) != set(class_order):
        raise ValueError(f"{report_path}: four-class contract mismatch")

    return {
        "seed": expected_seed,
        "decision_seed": expected_seed == DECISION_SEED,
        "run_dir": str(run_dir.resolve()),
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "history_path": str(history_path.resolve()),
        "history_sha256": history_sha256,
        "epochs_completed": epochs,
        "best_epoch": int(report["best_epoch"]),
        "stop_reason": report["stop_reason"],
        "dev_mean_iou": finite_number(metrics["mean_iou"], field="dev mean IoU"),
        "boundary_f1_tolerance_1px": finite_number(
            metrics["boundary_f1_tolerance_1px"], field="boundary F1"
        ),
        "worst_session_mean_iou": finite_number(
            metrics["worst_session_mean_iou"], field="worst-session mean IoU"
        ),
        "per_class_iou": {
            name: finite_number(per_class[name], field=f"{name} IoU")
            for name in class_order
        },
        "contract": {
            "manifest_sha256": data["manifest_sha256"],
            "implementation_sha256": report["implementation_sha256"],
            "pretrained_sha256": report["pretrained_sha256"],
            "official_repo_commit": report["official_repo_commit"],
            "train_sessions": sorted(train_sessions),
            "dev_sessions": sorted(dev_sessions),
            "recipe": recipe,
        },
    }


def require_common_contract(runs: list[dict[str, Any]]) -> None:
    reference = runs[0]["contract"]
    for run in runs[1:]:
        if run["contract"] != reference:
            raise ValueError(
                f"seed {run['seed']}: frozen data/implementation/recipe contract drift"
            )


def describe(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "population_stddev": statistics.pstdev(values),
    }


def build_receipt(run_dirs: list[Path]) -> dict[str, Any]:
    if len(run_dirs) != len(EXPECTED_SEEDS):
        raise ValueError("exactly three run directories are required")
    runs = [
        validate_run(run_dir, seed)
        for run_dir, seed in zip(run_dirs, EXPECTED_SEEDS, strict=True)
    ]
    require_common_contract(runs)
    class_order = list(runs[0]["per_class_iou"])
    return {
        "schema_version": "blindassist.riskseg_r0.training_freeze.v1",
        "protocol_id": "RISKSEG-R0",
        "status": "THREE_FIXED_SEED_TRAINING_ARTIFACTS_FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_order": list(EXPECTED_SEEDS),
        "decision_seed": DECISION_SEED,
        "event_eval_outcome_accessed_by_trainer": False,
        "runs": runs,
        "descriptive_pixel_stability_not_promotion_gate": {
            "dev_mean_iou": describe([run["dev_mean_iou"] for run in runs]),
            "boundary_f1_tolerance_1px": describe(
                [run["boundary_f1_tolerance_1px"] for run in runs]
            ),
            "worst_session_mean_iou": describe(
                [run["worst_session_mean_iou"] for run in runs]
            ),
            "per_class_iou": {
                name: describe([run["per_class_iou"][name] for run in runs])
                for name in class_order
            },
        },
        "interpretation": (
            "This receipt freezes provenance and reports descriptive dev-pixel "
            "stability only. It does not select a seed or authorize promotion."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Repeat exactly three times in fixed seed order 20260801/2/3.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_receipt([Path(value) for value in args.run_dir])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
