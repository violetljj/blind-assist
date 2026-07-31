"""Immutable scientific constants for FP-aware DDRNet R0."""

from __future__ import annotations

from typing import Any

from . import CANDIDATE_ID, PROTOCOL_ID


EXPECTED_SEEDS = [20260711, 20260712, 20260713]
EXPECTED_CONFIG_SHA256 = "4261fdb486a49c77b77287f24e597dba6b65c72725b85e7ebe9c288f1c5ccfa6"
EXPECTED_GATES = {
    "min_fp_pixel_reduction": 0.3,
    "min_overall_recall_retention": 0.9,
    "min_session_recall_retention": 0.8,
    "min_boundary_recall_retention": 0.8,
    "min_obstacle_recall_retention": 0.8,
    "min_delta_recall_C_minus_A": 0.05,
    "max_delta_false_positive_area_fraction_C_minus_A": 0.05,
    "min_candidate_component_recall": 0.5,
    "max_false_activation_components_per_frame": 3.0,
}
EXPECTED_TERMINALS = {
    "supported": "FP_WEIGHTED_SAMPLING_SUPPORTED_DEVELOPMENT_ONLY",
    "not_supported": "FP_WEIGHTED_SAMPLING_NOT_SUPPORTED",
    "not_evaluable": "FP_WEIGHTED_SAMPLING_NOT_EVALUABLE",
}
PENDING_VALIDATION_TERMINAL = "FP_WEIGHTED_SAMPLING_DECISION_PENDING_VALIDATION"


def validate_config_sha256(actual: str) -> None:
    if actual != EXPECTED_CONFIG_SHA256:
        raise ValueError(f"config SHA256 drifted: {actual}")


def validate_config_contract(config: dict[str, Any]) -> None:
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("config protocol identity mismatch")
    if config.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("config candidate identity mismatch")
    if config.get("stage") != "DEVELOPMENT":
        raise ValueError("config stage must remain DEVELOPMENT")
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("training contract missing")
    expected_training = {
        "optimizer": "Adam",
        "optimizer_steps": 1200,
        "head_warmup_steps": 100,
        "eval_every_steps": 50,
        "batch_size": 12,
        "seeds": EXPECTED_SEEDS,
        "head_learning_rate": 0.0003,
        "finetune_learning_rate": 0.00005,
        "finetune_final_lr_ratio": 0.1,
        "freeze_batchnorm_statistics": True,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training contract drifted: {key}")
    sampling = training.get("sampling")
    expected_sampling = {
        "session_selection": "uniform",
        "positive_guided_fraction": 0.7,
        "fp_weighted_full_frame_fraction": 0.3,
        "boundary_probability_within_positive_branch": 0.65,
        "crop_min_fraction": 0.55,
        "crop_max_fraction": 0.85,
        "horizontal_flip_probability": 0.5,
        "zero_session_fp_weight": "NOT_EVALUABLE_BEFORE_TRAINING",
        "full_frame_transform_unchanged": True,
    }
    if sampling != expected_sampling:
        raise ValueError("sampling contract drifted")
    expected_loss = {
        "weighted_cross_entropy": 0.5,
        "weighted_soft_dice": 0.4,
        "weighted_focal": 0.1,
        "focal_gamma": 2.0,
        "maximum_class_weight": 4.0,
        "class_weight_source": "train_only",
    }
    if training.get("loss") != expected_loss:
        raise ValueError("loss contract drifted")
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation contract missing")
    if evaluation.get("roles") != ["consumed_old_blind", "r1_consumed_fresh"]:
        raise ValueError("evaluation roles drifted")
    if evaluation.get("role_counts") != {
        "consumed_old_blind": 120,
        "r1_consumed_fresh": 200,
    }:
        raise ValueError("evaluation role counts drifted")
    if evaluation.get("batch_size") != 12:
        raise ValueError("evaluation batch size drifted")
    if evaluation.get("connectivity") != 8:
        raise ValueError("component connectivity drifted")
    if evaluation.get("component_hit_rule") != "positive_intersection":
        raise ValueError("component hit rule drifted")
    if evaluation.get("same_seed_pairing") is not True:
        raise ValueError("same-seed pairing disabled")
    if evaluation.get("all_three_seeds_required") is not True:
        raise ValueError("all-three-seed rule disabled")
    if evaluation.get("best_seed_selection_forbidden") is not True:
        raise ValueError("best-seed selection was enabled")
    if evaluation.get("gates") != EXPECTED_GATES:
        raise ValueError("evaluation gates drifted")
    if config.get("terminals") != EXPECTED_TERMINALS:
        raise ValueError("terminal map drifted")
