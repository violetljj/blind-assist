"""Outcome-blind canonical move coverage for the B3-A treatment arm."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, replace

from scripts.research.l10m_b1.policy_space import (
    FALLBACK_ACTIONS,
    QUALITY_FLOORS,
    RECOVERY_ACTIONS,
    STUCK_RESPONSES,
    TURN_THRESHOLDS,
    PolicySpec,
)


BALANCED_EXPLORATION_INSTRUCTION = """
Balanced exploration rule for this arm only:
- Treat a canonical semantic move as one parameter, its current value, and one adjacent direction or categorical destination.
- Prefer a legal move not listed in the attempted-move ledger.
- While an untried legal move remains, do not repeat an already attempted no-improvement move.
- Do not infer that any parameter or direction is preferred. Coverage, not a known answer, is the rule.
Return the same complete component-grouped JSON candidate as in the control arm.
""".strip()

NUMERIC_FIELDS = {
    "action_selection_turn_threshold": TURN_THRESHOLDS,
    "fallback_min_quality": QUALITY_FLOORS,
}
CATEGORICAL_FIELDS = {
    "fallback_action": FALLBACK_ACTIONS,
    "stuck_response": STUCK_RESPONSES,
    "recovery_transition_action": RECOVERY_ACTIONS,
}


def _value_text(value: object) -> str:
    if isinstance(value, float):
        return format(value, ".2f")
    return str(value)


def move_token(field: str, before: object, after: object) -> str:
    if field in NUMERIC_FIELDS:
        direction = "UP" if float(after) > float(before) else "DOWN"
    elif field in CATEGORICAL_FIELDS:
        direction = f"TO_{after}"
    else:
        raise ValueError(f"unknown policy field: {field}")
    return f"{field}|{_value_text(before)}|{direction}|{_value_text(after)}"


def legal_adjacent_moves(spec: PolicySpec) -> dict[str, PolicySpec]:
    """Return every legal one-step move from an incumbent, keyed canonically."""
    spec.validate()
    values = asdict(spec)
    moves: dict[str, PolicySpec] = {}
    for field, domain in NUMERIC_FIELDS.items():
        current = values[field]
        index = domain.index(current)
        for next_index in (index - 1, index + 1):
            if 0 <= next_index < len(domain):
                target = domain[next_index]
                token = move_token(field, current, target)
                moves[token] = replace(spec, **{field: target})
    for field, domain in CATEGORICAL_FIELDS.items():
        current = values[field]
        for target in domain:
            if target != current:
                token = move_token(field, current, target)
                moves[token] = replace(spec, **{field: target})
    return dict(sorted(moves.items()))


def proposal_move_tokens(incumbent: PolicySpec, proposal: PolicySpec) -> set[str]:
    """Map a full proposal to adjacent canonical directions from the incumbent."""
    incumbent.validate()
    proposal.validate()
    before = asdict(incumbent)
    after = asdict(proposal)
    legal = legal_adjacent_moves(incumbent)
    tokens: set[str] = set()
    for field in before:
        if before[field] == after[field]:
            continue
        if field in NUMERIC_FIELDS:
            domain = NUMERIC_FIELDS[field]
            current_index = domain.index(before[field])
            direction = 1 if domain.index(after[field]) > current_index else -1
            adjacent = domain[current_index + direction]
            token = move_token(field, before[field], adjacent)
        else:
            token = move_token(field, before[field], after[field])
        if token not in legal:
            raise RuntimeError(f"proposal produced a non-legal canonical move: {token}")
        tokens.add(token)
    return tokens


def _rank(seed: int, generation: int, source: str, token: str) -> str:
    payload = f"L10M-B3-A|{seed}|{generation}|{source}|{token}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def admit_balanced_proposal(
    incumbent: PolicySpec,
    model_proposal: PolicySpec,
    attempted_moves: set[str],
    *,
    seed: int,
    generation: int,
) -> tuple[PolicySpec, str | None, str]:
    """Admit one untried legal move without consulting scores or target identity."""
    legal = legal_adjacent_moves(incumbent)
    untried = set(legal) - attempted_moves
    proposed = proposal_move_tokens(incumbent, model_proposal)
    proposed_untried = proposed & untried
    if proposed_untried:
        token = min(proposed_untried, key=lambda item: _rank(seed, generation, "model", item))
        return legal[token], token, "MODEL_UNTRIED_DIRECTION"
    if untried:
        token = min(untried, key=lambda item: _rank(seed, generation, "coverage", item))
        return legal[token], token, "COVERAGE_PROJECTION"
    return model_proposal, None, "MODEL_AFTER_LOCAL_COVERAGE_EXHAUSTED"


def render_move_ledger(entries: list[dict[str, object]]) -> str:
    if not entries:
        return "(none)"
    lines = []
    for entry in entries:
        outcome = "improved" if entry["strict_improvement"] else "no_improvement"
        lines.append(f"- {entry['move_token']}: {outcome}")
    return "\n".join(lines)
