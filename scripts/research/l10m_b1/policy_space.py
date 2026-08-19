"""One finite policy space exposed through raw-source and structured interfaces."""

from __future__ import annotations

import ast
import itertools
import json
from dataclasses import asdict, dataclass

from scripts.research.l10m_b0.evaluation import Action, Belief, Evidence


FROZEN_PROGRESS_CONTRACT = "POSITIVE_PROGRESS|CONFIRMED_NO_PROGRESS|UNKNOWN_PROGRESS"
TURN_THRESHOLDS = (0.10, 0.20, 0.30)
QUALITY_FLOORS = (0.35, 0.50, 0.65)
FALLBACK_ACTIONS = (Action.STOP.value, Action.LEFT.value, Action.RIGHT.value)
STUCK_RESPONSES = ("ENTER_RECOVERY", "STOP")
RECOVERY_ACTIONS = (Action.RECOVER.value, Action.LEFT.value, Action.RIGHT.value)
RAW_NAMES = (
    "ACTION_SELECTION_TURN_THRESHOLD",
    "FALLBACK_MIN_QUALITY",
    "FALLBACK_ACTION",
    "STUCK_RESPONSE",
    "RECOVERY_TRANSITION_ACTION",
)


@dataclass(frozen=True)
class PolicySpec:
    action_selection_turn_threshold: float = 0.20
    fallback_min_quality: float = 0.35
    fallback_action: str = Action.STOP.value
    stuck_response: str = "ENTER_RECOVERY"
    recovery_transition_action: str = Action.RECOVER.value

    def validate(self) -> None:
        if self.action_selection_turn_threshold not in TURN_THRESHOLDS:
            raise ValueError("turn threshold outside the matched finite space")
        if self.fallback_min_quality not in QUALITY_FLOORS:
            raise ValueError("quality floor outside the matched finite space")
        if self.fallback_action not in FALLBACK_ACTIONS:
            raise ValueError("fallback action outside the matched finite space")
        if self.stuck_response not in STUCK_RESPONSES:
            raise ValueError("stuck response outside the matched finite space")
        if self.recovery_transition_action not in RECOVERY_ACTIONS:
            raise ValueError("recovery action outside the matched finite space")

    def propose(self, evidence: Evidence, belief: Belief) -> Action:
        self.validate()
        if not evidence.target_visible or evidence.quality < self.fallback_min_quality:
            return Action(self.fallback_action)
        if abs(evidence.alignment) > self.action_selection_turn_threshold:
            return Action.LEFT if evidence.alignment < 0 else Action.RIGHT
        if belief.stuck_count >= 2:
            if self.stuck_response == "STOP":
                return Action.STOP
            return Action(self.recovery_transition_action)
        return Action.FORWARD


INITIAL_SPEC = PolicySpec()


def all_specs() -> tuple[PolicySpec, ...]:
    return tuple(
        PolicySpec(*values)
        for values in itertools.product(
            TURN_THRESHOLDS,
            QUALITY_FLOORS,
            FALLBACK_ACTIONS,
            STUCK_RESPONSES,
            RECOVERY_ACTIONS,
        )
    )


def canonical_spec(spec: PolicySpec) -> str:
    spec.validate()
    return json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))


def render_raw(spec: PolicySpec) -> str:
    spec.validate()
    return (
        "# L10M-B1 raw source-level policy surface. Frozen semantics live outside this file.\n"
        f"ACTION_SELECTION_TURN_THRESHOLD = {spec.action_selection_turn_threshold:.2f}\n"
        f"FALLBACK_MIN_QUALITY = {spec.fallback_min_quality:.2f}\n"
        f"FALLBACK_ACTION = {spec.fallback_action!r}\n"
        f"STUCK_RESPONSE = {spec.stuck_response!r}\n"
        f"RECOVERY_TRANSITION_ACTION = {spec.recovery_transition_action!r}\n"
    )


def parse_raw(source: str) -> PolicySpec:
    """Parse a deliberately small source surface without executing candidate code."""
    tree = ast.parse(source, mode="exec")
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise ValueError("raw candidate may contain only literal policy assignments")
        name = node.targets[0].id
        if name not in RAW_NAMES or name in values:
            raise ValueError("raw candidate contains an unknown or duplicate policy field")
        values[name] = ast.literal_eval(node.value)
    if set(values) != set(RAW_NAMES):
        raise ValueError("raw candidate must assign every policy field exactly once")
    spec = PolicySpec(
        action_selection_turn_threshold=float(values["ACTION_SELECTION_TURN_THRESHOLD"]),
        fallback_min_quality=float(values["FALLBACK_MIN_QUALITY"]),
        fallback_action=str(values["FALLBACK_ACTION"]),
        stuck_response=str(values["STUCK_RESPONSE"]),
        recovery_transition_action=str(values["RECOVERY_TRANSITION_ACTION"]),
    )
    spec.validate()
    return spec


def render_structured(spec: PolicySpec) -> str:
    spec.validate()
    payload = {
        "progress_contract": {"mode": FROZEN_PROGRESS_CONTRACT, "mutable": False},
        "stuck_response": {"on_confirmed_stuck": spec.stuck_response},
        "recovery_transition": {"while_active": spec.recovery_transition_action},
        "action_selection": {"turn_threshold": spec.action_selection_turn_threshold},
        "fallback": {"min_quality": spec.fallback_min_quality, "action": spec.fallback_action},
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def parse_structured(source: str) -> PolicySpec:
    payload = json.loads(source)
    if payload.get("progress_contract") != {"mode": FROZEN_PROGRESS_CONTRACT, "mutable": False}:
        raise ValueError("structured candidate modified the frozen progress contract")
    if set(payload) != {"progress_contract", "stuck_response", "recovery_transition", "action_selection", "fallback"}:
        raise ValueError("structured candidate fields changed")
    spec = PolicySpec(
        action_selection_turn_threshold=float(payload["action_selection"]["turn_threshold"]),
        fallback_min_quality=float(payload["fallback"]["min_quality"]),
        fallback_action=str(payload["fallback"]["action"]),
        stuck_response=str(payload["stuck_response"]["on_confirmed_stuck"]),
        recovery_transition_action=str(payload["recovery_transition"]["while_active"]),
    )
    spec.validate()
    return spec


def changed_components(before: PolicySpec, after: PolicySpec) -> list[str]:
    mapping = {
        "action_selection_turn_threshold": "action_selection",
        "fallback_min_quality": "fallback",
        "fallback_action": "fallback",
        "stuck_response": "stuck_response",
        "recovery_transition_action": "recovery_transition",
    }
    before_values = asdict(before)
    after_values = asdict(after)
    return sorted({mapping[name] for name in mapping if before_values[name] != after_values[name]})
