#!/usr/bin/env python3
"""Train all r7.63 route-auxiliary frames and freeze the r7.66 diagnostic candidate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import train_public_video_temporal_route_head as training


SCHEMA = "blindassist_public_video_temporal_risk_profile_candidate_v1"


def run(args: argparse.Namespace) -> dict:
    for path in (args.contract, args.training_contract, args.training_report, args.audit_report,
                 args.cache_report, args.cache, args.output_weights, args.output_report):
        mil.reject_independent_direction(path)
    if args.output_weights.exists() or args.output_report.exists() or Path(str(args.output_report) + ".sha256").exists():
        raise ValueError("refusing to overwrite frozen temporal risk-profile candidate")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    training_contract = json.loads(args.training_contract.read_text(encoding="utf-8"))
    lifecycle.verify_json_sidecar(args.training_report)
    audit = lifecycle.verify_json_sidecar(args.audit_report)
    cache_report = lifecycle.verify_json_sidecar(args.cache_report)
    bound = contract["bound_inputs"]
    for path, key in ((args.training_contract, "training_contract_sha256"),
                      (args.training_report, "training_report_sha256"),
                      (args.audit_report, "uncertainty_audit_sha256"),
                      (args.cache_report, "feature_cache_report_sha256"),
                      (args.cache, "feature_cache_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input mismatch: {path}")
    if audit.get("strict_event_separation", {}).get("obstacle_peak_ratio") is not True:
        raise ValueError("r7.65 did not support the frozen relative-peak hypothesis")
    if common.sha256_file(args.cache) != cache_report["cache"]["sha256"]:
        raise ValueError("cache differs from extraction report")
    cache = np.load(args.cache)
    train_x = cache["train_x"].astype(np.float32)
    train_y = cache["train_y"].astype(np.float32)
    train_sources = cache["train_sources"].astype(str)
    probe = dict(training_contract["single_seed_probe"])
    candidate = contract["candidate_training"]
    for key, candidate_key in (("seed", "seed"), ("epochs", "epochs"), ("batch_size", "batch_size"),
                               ("learning_rate", "learning_rate"), ("weight_decay", "weight_decay")):
        if float(probe[key]) != float(candidate[candidate_key]):
            raise ValueError(f"candidate training differs from r7.64: {key}")
    model, losses = training.train_fold(train_x, train_y, train_sources, probe, train_x.shape[1])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_channels": int(train_x.shape[1]),
                "contract_sha256": common.sha256_file(args.contract)}, args.output_weights)
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract), "cache_sha256": common.sha256_file(args.cache),
                   "audit_report_sha256": common.sha256_file(args.audit_report)},
        "training": {"frame_count": len(train_x), "source_count": len(set(train_sources.tolist())),
                     "input_channels": int(train_x.shape[1]), "trainable_parameter_count": parameter_count,
                     "epoch_losses": losses, "finite": bool(np.isfinite(losses).all())},
        "weights": {"path": str(args.output_weights), "sha256": common.sha256_file(args.output_weights),
                    "role": contract["candidate_training"]["weights_role"]},
        "risk_profile": contract["risk_profile"],
        "prospective_acceptance_credit_present": False,
        "authorization": contract["authorization"],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output_report) + ".sha256").write_text(common.sha256_file(args.output_report) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-weights", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, "weights_sha256": value["weights"]["sha256"],
                      "output_sha256": common.sha256_file(args.output_report)}, ensure_ascii=False))
