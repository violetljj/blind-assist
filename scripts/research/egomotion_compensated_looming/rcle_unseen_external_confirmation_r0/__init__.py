"""Pure data-layer metrics for RCLE unseen external confirmation R0."""

from .metrics import (
    BELOW_RELATIVE_REDUCTION_MIN,
    BELOW_ROLE,
    MAX_PAIR_DT_S,
    POSITIVE_FIRST_TRIGGER_DELAY_MAX_S,
    POSITIVE_RETENTION_MIN,
    POSITIVE_ROLE,
    REQUIRED_CONSECUTIVE_PAIRS,
    THRESHOLD_PER_S,
    derive_confirmation_rows,
    evaluate_confirmation,
    summarize_confirmation,
)

__all__ = [
    "BELOW_RELATIVE_REDUCTION_MIN",
    "BELOW_ROLE",
    "MAX_PAIR_DT_S",
    "POSITIVE_FIRST_TRIGGER_DELAY_MAX_S",
    "POSITIVE_RETENTION_MIN",
    "POSITIVE_ROLE",
    "REQUIRED_CONSECUTIVE_PAIRS",
    "THRESHOLD_PER_S",
    "derive_confirmation_rows",
    "evaluate_confirmation",
    "summarize_confirmation",
]
