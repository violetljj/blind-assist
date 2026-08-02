#!/usr/bin/env python3
"""Evaluate HFTF Development checkpoints across environment clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from train_stage_c_d5_tartanground_development_student import (
    HftfDataset,
    TemporalStudent,
    decode_labels,
    evaluate,
    load_jsonl,
    sha256,
)


def image_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    luminance_means = []
    luminance_contrasts = []
    temporal_l1 = []
    cache: dict[str, np.ndarray] = {}
    for record in records:
        frames = []
        for item in record["history_rgb"]:
            path = item["image_path"]
            value = cache.get(path)
            if value is None:
                with Image.open(path) as source:
                    value = np.asarray(
                        source.convert("RGB"),
                        dtype=np.uint8,
                    )
                cache[path] = value
            frames.append(value.astype(np.float32) / 255.0)
        current = frames[-1]
        luminance = (
            0.2126 * current[:, :, 0]
            + 0.7152 * current[:, :, 1]
            + 0.0722 * current[:, :, 2]
        )
        luminance_means.append(float(luminance.mean()))
        luminance_contrasts.append(float(luminance.std()))
        temporal_l1.append(
            float(
                np.mean(
                    [
                        np.mean(np.abs(frames[index] - frames[index - 1]))
                        for index in range(1, len(frames))
                    ]
                )
            )
        )
    return {
        "current_luminance_mean": float(np.mean(luminance_means)),
        "current_luminance_p10": float(
            np.quantile(luminance_means, 0.1)
        ),
        "current_luminance_p90": float(
            np.quantile(luminance_means, 0.9)
        ),
        "current_luminance_contrast_mean": float(
            np.mean(luminance_contrasts)
        ),
        "history_adjacent_rgb_l1_mean": float(np.mean(temporal_l1)),
    }


def label_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    risks = []
    knowns = []
    for record in records:
        risk, known = decode_labels(record)
        risks.append(risk.numpy()[1:, 1:])
        knowns.append(known.numpy()[1:, 1:])
    risk_array = np.stack(risks)
    known_array = np.stack(knowns)
    mask = known_array.astype(bool)
    return {
        "sample_count": len(records),
        "future_body_head_known_cells": int(mask.sum()),
        "future_body_head_known_fraction": float(known_array.mean()),
        "future_body_head_positive_rate": float(
            np.mean(risk_array[mask] >= 0.5)
        ),
        "future_body_head_risk_score_mean": float(
            np.mean(risk_array[mask])
        ),
    }


def data_loader(
    records: list[dict[str, Any]],
    input_arm: str,
) -> DataLoader:
    return DataLoader(
        HftfDataset(
            records,
            input_arm,
            train=False,
            seed=0,
        ),
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def metric_projection(metrics: dict[str, Any]) -> dict[str, float]:
    body_head = metrics["risk_future_body_head"]
    projection = {
        "future_body_head_macro_f1": float(
            metrics["future_body_head_macro_f1"]
        ),
        "future_body_head_micro_f1": float(body_head["f1"]),
        "future_body_head_precision": float(body_head["precision"]),
        "future_body_head_recall": float(body_head["recall"]),
        "future_body_head_fpr": float(body_head["false_positive_rate"]),
        "future_body_head_mae": float(body_head["risk_score_mae"]),
    }
    for source, destination in (
        ("auroc", "future_body_head_auroc"),
        ("average_precision", "future_body_head_average_precision"),
    ):
        if body_head.get(source) is not None:
            projection[destination] = float(body_head[source])
    return projection


def comparisons(
    models: dict[str, Any],
    reference: str,
) -> dict[str, Any]:
    reference_row = models[reference]
    output = {}
    for name, row in models.items():
        if name == reference:
            continue
        environment_deltas = {}
        for environment, metrics in row["by_environment"].items():
            candidate = metric_projection(metrics)
            baseline = metric_projection(
                reference_row["by_environment"][environment]
            )
            environment_deltas[environment] = {
                key: candidate[key] - baseline[key]
                for key in candidate
            }
        macro_deltas = [
            row["future_body_head_macro_f1"]
            for row in environment_deltas.values()
        ]
        output[name] = {
            "aggregate": {
                key: metric_projection(row["overall"])[key]
                - metric_projection(reference_row["overall"])[key]
                for key in metric_projection(row["overall"])
            },
            "by_environment": environment_deltas,
            "environment_macro_f1_wins": sum(
                value > 0.0 for value in macro_deltas
            ),
            "environment_macro_f1_losses": sum(
                value < 0.0 for value in macro_deltas
            ),
            "environment_macro_f1_ties": sum(
                value == 0.0 for value in macro_deltas
            ),
            "environment_macro_f1_mean_delta": float(
                np.mean(macro_deltas)
            ),
            "environment_macro_f1_worst_delta": float(
                np.min(macro_deltas)
            ),
        }
    return output


def parse_models(
    rows: list[list[str]],
) -> list[tuple[str, str, Path]]:
    parsed = []
    seen = set()
    for name, arm, checkpoint in rows:
        if name in seen:
            raise ValueError(f"Duplicate model name: {name}")
        if arm not in {"single", "history"}:
            raise ValueError(f"Unknown input arm: {arm}")
        seen.add(name)
        parsed.append((name, arm, Path(checkpoint)))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        metavar=("NAME", "INPUT_ARM", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="Evaluate only records with this role; repeat for multiple roles.",
    )
    args = parser.parse_args()

    model_specs = parse_models(args.model)
    names = {name for name, _, _ in model_specs}
    if args.reference not in names:
        parser.error("--reference must name one of --model")
    records = load_jsonl(args.samples)
    if args.role:
        records = [
            record for record in records if record["role"] in args.role
        ]
    if not records:
        raise ValueError("No samples")
    environments = sorted({record["environment"] for record in records})
    environment_records = {
        environment: [
            record
            for record in records
            if record["environment"] == environment
        ]
        for environment in environments
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_results = {}
    for name, input_arm, checkpoint_path in model_specs:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        architecture = checkpoint.get("architecture", "pooled")
        temporal_mode = checkpoint.get("temporal_mode", "joint")
        model = TemporalStudent(
            args.pretrained,
            architecture=architecture,
            temporal_mode=temporal_mode,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        overall = evaluate(
            model,
            data_loader(records, input_arm),
            device,
        )
        by_environment = {
            environment: evaluate(
                model,
                data_loader(rows, input_arm),
                device,
            )
            for environment, rows in environment_records.items()
        }
        model_results[name] = {
            "input_arm": input_arm,
            "architecture": architecture,
            "temporal_mode": temporal_mode,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            "overall": overall,
            "by_environment": by_environment,
        }
        print(
            json.dumps(
                {
                    "model": name,
                    "future_body_head_macro_f1": overall[
                        "future_body_head_macro_f1"
                    ],
                }
            ),
            flush=True,
        )

    report = {
        "schema": (
            "blindassist_hftf_stage_c_d5_tartanground_"
            "development_checkpoint_evaluation"
        ),
        "status": "DEVELOPMENT_CHECKPOINT_EVALUATION_COMPLETE",
        "policy": {
            "outcome_open": True,
            "repairable": True,
            "one_shot": False,
            "heldout": False,
            "promotion_evidence": False,
        },
        "samples": {
            "path": str(args.samples.resolve()),
            "bytes": args.samples.stat().st_size,
            "sha256": sha256(args.samples),
            "sample_count": len(records),
        },
        "pretrained_sha256": sha256(args.pretrained),
        "reference": args.reference,
        "roles": sorted({record["role"] for record in records}),
        "environments": {
            environment: {
                "labels": label_summary(rows),
                "images": image_summary(rows),
            }
            for environment, rows in environment_records.items()
        },
        "models": model_results,
    }
    report["comparisons"] = comparisons(
        model_results,
        args.reference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "bytes": args.output.stat().st_size,
                "sha256": sha256(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
