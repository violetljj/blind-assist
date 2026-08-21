"""Current-frame-only control loop for BLINDASSIST_LAST_10M_REGROUNDING_V0.

The engine intentionally consumes one already-produced P0 output at a time.  It
never calls, selects, or modifies a provider, and it never carries a candidate,
region, image feature, or identity assertion across observations.  Prior
candidate details live only in the append-only audit event returned to the
caller; they are not part of :class:`EpisodeState` and cannot influence a later
decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


FAIL_CLOSED_MESSAGE = "没有可靠找到入口，请停下并缓慢重新扫描。"
HUMAN_ASSISTANCE_MESSAGE = "持续无法确认入口，已停止自动引导。请联系现场工作人员或可信任的人协助。"
CLAIM_CEILING = "REAL_SITE_MECHANICAL_TASK_ONLY_NO_SCIENTIFIC_CONFIRMATION_NO_SAFETY_CLAIM"


class ContractError(ValueError):
    """Raised when an observation or episode violates the mechanical contract."""


class State(str, Enum):
    SCAN = "SCAN"
    CURRENT_CANDIDATE = "CURRENT_CANDIDATE"
    ALIGN = "ALIGN"
    ADVANCE_AND_REOBSERVE = "ADVANCE_AND_REOBSERVE"
    ARRIVAL_CONFIRM = "ARRIVAL_CONFIRM"
    COMPLETE = "COMPLETE"
    RESCAN = "RESCAN"
    ABSTAIN = "ABSTAIN"


class Attribution(str, Enum):
    CURRENT_FRAME_GROUNDING_BOTTLENECK = "CURRENT_FRAME_GROUNDING_BOTTLENECK"
    INTERACTION_OR_CONTROL_BOTTLENECK = "INTERACTION_OR_CONTROL_BOTTLENECK"
    REGROUNDING_LOOP_MECHANICALLY_USEFUL = "REGROUNDING_LOOP_MECHANICALLY_USEFUL"


@dataclass(frozen=True)
class Policy:
    center_left: float = 0.42
    center_right: float = 0.58
    arrival_min_height: float = 0.55
    max_consecutive_unreliable: int = 3
    max_instructions: int = 12

    def validate(self) -> None:
        if not 0.0 < self.center_left < self.center_right < 1.0:
            raise ContractError("center bounds must be ordered inside (0, 1)")
        if not 0.0 < self.arrival_min_height <= 1.0:
            raise ContractError("arrival_min_height must be in (0, 1]")
        if self.max_consecutive_unreliable < 1:
            raise ContractError("max_consecutive_unreliable must be positive")
        if self.max_instructions < 1:
            raise ContractError("max_instructions must be positive")


@dataclass
class EpisodeState:
    schema_version: int
    episode_id: str
    location_id: str
    goal_name: str
    state: str
    started_at_ms: int
    last_frame_id: str | None = None
    last_observation_id: str | None = None
    observation_count: int = 0
    reliable_observation_count: int = 0
    consecutive_unreliable: int = 0
    first_discovery_at_ms: int | None = None
    completed_at_ms: int | None = None
    instruction_count: int = 0
    rescan_count: int = 0

    @classmethod
    def start(cls, *, episode_id: str, location_id: str, goal_name: str, started_at_ms: int) -> "EpisodeState":
        for label, value in (("episode_id", episode_id), ("location_id", location_id), ("goal_name", goal_name)):
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{label} must be non-empty")
        if not isinstance(started_at_ms, int) or started_at_ms < 0:
            raise ContractError("started_at_ms must be a non-negative integer")
        return cls(1, episode_id, location_id, goal_name, State.SCAN.value, started_at_ms)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeState":
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ContractError("episode state keys do not match schema")
        state = cls(**dict(value))
        if state.schema_version != 1 or state.state not in {item.value for item in State}:
            raise ContractError("invalid episode state")
        return state

    def to_dict(self) -> dict[str, Any]:
        # Deliberately contains no candidate id, region, image, feature, score,
        # handoff token, or identity field from any observation.
        return asdict(self)


@dataclass(frozen=True)
class CurrentCandidate:
    candidate_id: str
    frame_id: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def audit_dict(self) -> dict[str, Any]:
        return asdict(self) | {"center_x": self.center_x, "height": self.height}


@dataclass(frozen=True)
class AdaptedGrounding:
    reliable: bool
    p0_status: str
    abstention_reason: str | None
    provider_failure_classes: tuple[str, ...]
    candidate: CurrentCandidate | None


@dataclass(frozen=True)
class StepResult:
    state: EpisodeState
    message: str
    instruction: str | None
    event: dict[str, Any]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError(f"{label} must be an array")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    return float(value)


def adapt_current_p0_output(
    observation: Mapping[str, Any], *, expected_episode_id: str, expected_goal_name: str
) -> AdaptedGrounding:
    """Validate and extract one current-frame P0 result without evaluator truth.

    Non-grounded P0 statuses are valid fail-closed observations.  A malformed,
    stale, multi-frame, or cross-frame grounded claim is rejected as a contract
    error rather than converted into a usable candidate.
    """

    required = {"schema_version", "episode_id", "observation_id", "frame_id", "captured_at_ms", "p0_output"}
    if set(observation) != required or observation.get("schema_version") != 1:
        raise ContractError("observation keys or schema_version are invalid")
    if observation["episode_id"] != expected_episode_id:
        raise ContractError("observation episode_id mismatch")
    frame_id = _nonempty(observation["frame_id"], "frame_id")
    _nonempty(observation["observation_id"], "observation_id")
    captured_at_ms = observation["captured_at_ms"]
    if not isinstance(captured_at_ms, int) or captured_at_ms < 0:
        raise ContractError("captured_at_ms must be a non-negative integer")

    output = _mapping(observation["p0_output"], "p0_output")
    if output.get("schema_version") != 1:
        raise ContractError("P0 output schema_version must be 1")
    decision = _mapping(output.get("decision"), "p0_output.decision")
    p0_status = _nonempty(decision.get("status"), "decision.status")
    allowed_statuses = {"GROUNDED", "AMBIGUOUS", "ABSTAIN_NO_RELIABLE_EVIDENCE", "INVALID_OBSERVATION"}
    if p0_status not in allowed_statuses:
        raise ContractError("unknown P0 decision status")
    decision_timestamp_ms = decision.get("decision_timestamp_ms")
    if not isinstance(decision_timestamp_ms, int) or decision_timestamp_ms < captured_at_ms:
        raise ContractError("P0 decision predates the current frame")

    provider_runs = _sequence(output.get("provider_runs"), "provider_runs")
    provider_failures: list[str] = []
    providers: dict[str, Mapping[str, Any]] = {}
    for raw in provider_runs:
        run = _mapping(raw, "provider_run")
        provider_id = _nonempty(run.get("provider_id"), "provider_id")
        if provider_id in providers:
            raise ContractError("duplicate provider_id")
        providers[provider_id] = run
        status = _nonempty(run.get("status"), "provider status")
        if status != "RUN_SUCCESS":
            provider_failures.append(status)

    if p0_status != "GROUNDED":
        if decision.get("selected_candidate_id") is not None or decision.get("source_frame_id") is not None:
            raise ContractError("non-GROUNDED P0 decision carries a candidate or source frame")
        return AdaptedGrounding(
            reliable=False,
            p0_status=p0_status,
            abstention_reason=decision.get("abstention_reason"),
            provider_failure_classes=tuple(sorted(set(provider_failures))),
            candidate=None,
        )

    selected_id = _nonempty(decision.get("selected_candidate_id"), "selected_candidate_id")
    if decision.get("source_frame_id") != frame_id:
        raise ContractError("grounded decision does not bind the current frame")
    if decision.get("goal_identity_support") != "SUPPORTED" or decision.get("spatial_support") != "SUPPORTED":
        raise ContractError("GROUNDED requires supported identity and spatial evidence")
    candidates = [_mapping(item, "candidate") for item in _sequence(output.get("candidates"), "candidates")]
    selected = [item for item in candidates if item.get("candidate_id") == selected_id]
    if len(selected) != 1:
        raise ContractError("selected candidate is missing or duplicated")
    selected_value = selected[0]
    region = _mapping(selected_value.get("region"), "selected candidate region")
    if region.get("frame_id") != frame_id or region.get("coordinate_space") != "NORMALIZED_XYXY":
        raise ContractError("selected candidate region is not current-frame normalized XYXY")
    x_min = _number(region.get("x_min"), "x_min")
    y_min = _number(region.get("y_min"), "y_min")
    x_max = _number(region.get("x_max"), "x_max")
    y_max = _number(region.get("y_max"), "y_max")
    if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
        raise ContractError("selected candidate region bounds are invalid")

    evidence = {
        _nonempty(item.get("evidence_id"), "evidence_id"): item
        for item in (_mapping(raw, "evidence") for raw in _sequence(output.get("evidence"), "evidence"))
    }
    supporting_ids = [_nonempty(item, "supporting evidence id") for item in _sequence(decision.get("supporting_evidence_ids"), "supporting_evidence_ids")]
    if not supporting_ids:
        raise ContractError("grounded decision requires current supporting evidence")
    selected_evidence_ids = {
        _nonempty(item, "candidate evidence id")
        for item in _sequence(selected_value.get("evidence_ids"), "candidate.evidence_ids")
    }
    if not set(supporting_ids) <= selected_evidence_ids:
        raise ContractError("supporting evidence is not bound to the selected candidate")
    goal_identity_match = False
    for evidence_id in supporting_ids:
        if evidence_id not in evidence:
            raise ContractError("grounded decision references unknown evidence")
        item = evidence[evidence_id]
        if list(_sequence(item.get("source_frame_ids"), "evidence.source_frame_ids")) != [frame_id]:
            raise ContractError("supporting evidence is not current-frame-only")
        valid_until_ms = item.get("valid_until_ms")
        if not isinstance(valid_until_ms, int) or valid_until_ms < decision_timestamp_ms:
            raise ContractError("supporting evidence is stale")
        identity_claim = item.get("identity_claim")
        if isinstance(identity_claim, Mapping):
            goal_identity_match = goal_identity_match or (
                identity_claim.get("target_name") == expected_goal_name
                and identity_claim.get("relation") == "entrance_of"
            )
    if not goal_identity_match:
        raise ContractError("supporting evidence does not bind the requested entrance goal")

    provider_ids = [_nonempty(item, "candidate provider_id") for item in _sequence(selected_value.get("provider_ids"), "candidate.provider_ids")]
    for provider_id in provider_ids:
        run = providers.get(provider_id)
        if run is None or run.get("status") != "RUN_SUCCESS":
            raise ContractError("selected candidate references an unavailable provider")
        if list(_sequence(run.get("source_frame_ids"), "provider.source_frame_ids")) != [frame_id]:
            raise ContractError("selected provider run is not current-frame-only")

    handoff = _mapping(decision.get("persistence_handoff_token"), "persistence_handoff_token")
    if handoff.get("candidate_id") != selected_id or handoff.get("source_frame_id") != frame_id:
        raise ContractError("P0 handoff binding drift")
    if list(_sequence(handoff.get("evidence_ids"), "handoff.evidence_ids")) != supporting_ids:
        raise ContractError("P0 handoff evidence binding drift")
    handoff_region = _mapping(handoff.get("spatial_region"), "handoff.spatial_region")
    region_keys = ("frame_id", "coordinate_space", "x_min", "y_min", "x_max", "y_max")
    if any(handoff_region.get(key) != region.get(key) for key in region_keys):
        raise ContractError("P0 handoff region binding drift")
    # The handoff is checked only for current-frame consistency and is discarded.
    confidence = _number(decision.get("confidence"), "decision.confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ContractError("decision confidence must be in [0, 1]")
    return AdaptedGrounding(
        reliable=True,
        p0_status=p0_status,
        abstention_reason=None,
        provider_failure_classes=tuple(sorted(set(provider_failures))),
        candidate=CurrentCandidate(selected_id, frame_id, x_min, y_min, x_max, y_max, confidence),
    )


def apply_observation(state: EpisodeState, observation: Mapping[str, Any], policy: Policy = Policy()) -> StepResult:
    policy.validate()
    if state.state in {State.COMPLETE.value, State.ABSTAIN.value}:
        raise ContractError("terminal episode cannot accept another observation")
    frame_id = _nonempty(observation.get("frame_id"), "frame_id")
    observation_id = _nonempty(observation.get("observation_id"), "observation_id")
    if frame_id == state.last_frame_id or observation_id == state.last_observation_id:
        raise ContractError("turn/advance/rescan requires a fresh observation and frame")
    captured_at_ms = observation.get("captured_at_ms")
    if isinstance(captured_at_ms, bool) or not isinstance(captured_at_ms, int):
        raise ContractError("captured_at_ms must be an integer")
    if captured_at_ms < state.started_at_ms:
        raise ContractError("observation predates episode start")
    contract_error: str | None = None
    try:
        grounded = adapt_current_p0_output(
            observation,
            expected_episode_id=state.episode_id,
            expected_goal_name=state.goal_name,
        )
    except ContractError as error:
        contract_error = str(error)
        grounded = AdaptedGrounding(
            reliable=False,
            p0_status="INVALID_OUTPUT",
            abstention_reason="INVALID_INPUT",
            provider_failure_classes=("INVALID_OUTPUT",),
            candidate=None,
        )

    prior_state = state.state
    state.last_frame_id = frame_id
    state.last_observation_id = observation_id
    state.observation_count += 1
    transitions: list[str] = []
    instruction: str | None = None
    candidate_audit: dict[str, Any] | None = None

    if not grounded.reliable:
        state.consecutive_unreliable += 1
        state.rescan_count += 1
        transitions.append(State.RESCAN.value)
        if state.consecutive_unreliable >= policy.max_consecutive_unreliable:
            state.state = State.ABSTAIN.value
            transitions.append(State.ABSTAIN.value)
            message = f"{FAIL_CLOSED_MESSAGE} {HUMAN_ASSISTANCE_MESSAGE}"
        else:
            state.state = State.RESCAN.value
            message = FAIL_CLOSED_MESSAGE
    else:
        candidate = grounded.candidate
        assert candidate is not None
        state.reliable_observation_count += 1
        state.consecutive_unreliable = 0
        if state.first_discovery_at_ms is None:
            state.first_discovery_at_ms = captured_at_ms
        candidate_audit = candidate.audit_dict()
        transitions.extend((State.CURRENT_CANDIDATE.value, State.ALIGN.value))
        centered = policy.center_left <= candidate.center_x <= policy.center_right
        arrival_near = centered and candidate.height >= policy.arrival_min_height

        if prior_state == State.ARRIVAL_CONFIRM.value and arrival_near:
            state.state = State.COMPLETE.value
            state.completed_at_ms = captured_at_ms
            transitions.extend((State.ARRIVAL_CONFIRM.value, State.COMPLETE.value))
            message = "已依据新的当前帧重新确认入口。任务完成。"
        elif state.instruction_count >= policy.max_instructions:
            state.state = State.ABSTAIN.value
            transitions.append(State.ABSTAIN.value)
            message = HUMAN_ASSISTANCE_MESSAGE
        elif arrival_near:
            state.state = State.ARRIVAL_CONFIRM.value
            transitions.append(State.ARRIVAL_CONFIRM.value)
            instruction = "可能已接近入口，请停下并重新扫描以确认到达。"
            message = instruction
            state.instruction_count += 1
        else:
            state.state = State.ADVANCE_AND_REOBSERVE.value
            if candidate.center_x < policy.center_left:
                instruction = "入口在当前画面左侧，请小幅向左转，然后停下重新扫描。"
            elif candidate.center_x > policy.center_right:
                instruction = "入口在当前画面右侧，请小幅向右转，然后停下重新扫描。"
            else:
                instruction = "入口位于当前画面中央，请向前移动一小步，然后停下重新扫描。"
            if "前方安全" in instruction:
                raise AssertionError("forbidden safety wording")
            transitions.append(State.ADVANCE_AND_REOBSERVE.value)
            message = instruction
            state.instruction_count += 1

    event = {
        "schema_version": 1,
        "event_type": "OBSERVATION_PROCESSED",
        "episode_id": state.episode_id,
        "location_id": state.location_id,
        "observation_id": observation_id,
        "frame_id": frame_id,
        "captured_at_ms": captured_at_ms,
        "from_state": prior_state,
        "transitions": transitions,
        "to_state": state.state,
        "p0_status": grounded.p0_status,
        "provider_failure_classes": list(grounded.provider_failure_classes),
        "contract_error": contract_error,
        "candidate": candidate_audit,
        "instruction": instruction,
        "rescan": not grounded.reliable,
        "abstention": state.state == State.ABSTAIN.value,
        "completion": state.state == State.COMPLETE.value,
        "message": message,
    }
    return StepResult(state, message, instruction, event)


def stop_episode(state: EpisodeState, *, stopped_at_ms: int, attribution: Attribution, reason: str) -> dict[str, Any]:
    if state.state in {State.COMPLETE.value, State.ABSTAIN.value}:
        raise ContractError("episode is already terminal")
    if attribution == Attribution.REGROUNDING_LOOP_MECHANICALLY_USEFUL:
        raise ContractError("a stopped episode cannot be attributed as mechanically useful")
    if not isinstance(stopped_at_ms, int) or stopped_at_ms < state.started_at_ms:
        raise ContractError("invalid stopped_at_ms")
    _nonempty(reason, "reason")
    prior = state.state
    state.state = State.ABSTAIN.value
    return {
        "schema_version": 1,
        "event_type": "EPISODE_STOPPED",
        "episode_id": state.episode_id,
        "from_state": prior,
        "to_state": State.ABSTAIN.value,
        "stopped_at_ms": stopped_at_ms,
        "failure_attribution": attribution.value,
        "reason": reason,
        "message": HUMAN_ASSISTANCE_MESSAGE,
    }


def adjudicate_episode(
    state: EpisodeState,
    *,
    adjudicated_at_ms: int,
    false_entrance_confirmation: bool,
    failure_attribution: Attribution | None = None,
) -> dict[str, Any]:
    if state.state not in {State.COMPLETE.value, State.ABSTAIN.value}:
        raise ContractError("only a terminal episode can be adjudicated")
    if not isinstance(false_entrance_confirmation, bool):
        raise ContractError("false_entrance_confirmation must be boolean")
    if false_entrance_confirmation:
        attribution = Attribution.CURRENT_FRAME_GROUNDING_BOTTLENECK
    elif state.state == State.COMPLETE.value:
        if failure_attribution not in {None, Attribution.REGROUNDING_LOOP_MECHANICALLY_USEFUL}:
            raise ContractError("a correctly completed episode has fixed mechanical attribution")
        attribution = Attribution.REGROUNDING_LOOP_MECHANICALLY_USEFUL
    else:
        if failure_attribution not in {
            Attribution.CURRENT_FRAME_GROUNDING_BOTTLENECK,
            Attribution.INTERACTION_OR_CONTROL_BOTTLENECK,
        }:
            raise ContractError("an abstained episode requires one allowed bottleneck attribution")
        attribution = failure_attribution
    completion_time_ms = (
        state.completed_at_ms - state.started_at_ms if state.completed_at_ms is not None else None
    )
    first_discovery_time_ms = (
        state.first_discovery_at_ms - state.started_at_ms if state.first_discovery_at_ms is not None else None
    )
    return {
        "schema_version": 1,
        "episode_id": state.episode_id,
        "location_id": state.location_id,
        "goal_name": state.goal_name,
        "terminal_state": state.state,
        "adjudicated_at_ms": adjudicated_at_ms,
        "false_entrance_confirmation": false_entrance_confirmation,
        "failure_attribution": attribution.value,
        "completion_time_ms": completion_time_ms,
        "first_discovery_time_ms": first_discovery_time_ms,
        "instruction_count": state.instruction_count,
        "rescan_count": state.rescan_count,
        "observation_count": state.observation_count,
        "reliable_observation_count": state.reliable_observation_count,
        "claim_ceiling": CLAIM_CEILING,
    }


def _metric(values: Iterable[int | None]) -> dict[str, Any]:
    present = [int(value) for value in values if value is not None]
    return {
        "count": len(present),
        "median": median(present) if present else None,
        "min": min(present) if present else None,
        "max": max(present) if present else None,
    }


def summarize_field_run(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    episode_ids = [_nonempty(item.get("episode_id"), "episode_id") for item in summaries]
    if len(set(episode_ids)) != len(episode_ids):
        raise ContractError("duplicate episode summaries")
    locations: dict[str, int] = {}
    for item in summaries:
        location = _nonempty(item.get("location_id"), "location_id")
        locations[location] = locations.get(location, 0) + 1
        if item.get("failure_attribution") not in {value.value for value in Attribution}:
            raise ContractError("unknown failure attribution")
    roster_complete = len(summaries) == 15 and len(locations) == 3 and set(locations.values()) == {5}
    completed = sum(item.get("terminal_state") == State.COMPLETE.value for item in summaries)
    false_confirmations = sum(item.get("false_entrance_confirmation") is True for item in summaries)
    attribution_counts = {
        value.value: sum(item.get("failure_attribution") == value.value for item in summaries)
        for value in Attribution
    }
    return {
        "schema_version": 1,
        "milestone_id": "BLINDASSIST_LAST_10M_REGROUNDING_V0",
        "status": "FIELD_EXECUTION_COMPLETE" if roster_complete else "FIELD_EXECUTION_INCOMPLETE",
        "roster": {
            "episode_count": len(summaries),
            "location_counts": dict(sorted(locations.items())),
            "required": "3_REAL_LOCATIONS_X_5_EPISODES",
        },
        "primary_safety_metric": {
            "name": "false_entrance_confirmation_count",
            "value": false_confirmations,
        },
        "task_completion_rate": {
            "numerator": completed,
            "denominator": len(summaries),
            "value": completed / len(summaries) if summaries else None,
        },
        "completion_time_ms": _metric(item.get("completion_time_ms") for item in summaries),
        "first_discovery_time_ms": _metric(item.get("first_discovery_time_ms") for item in summaries),
        "instruction_count": {
            "total": sum(int(item.get("instruction_count", 0)) for item in summaries),
            "per_episode": _metric(item.get("instruction_count") for item in summaries),
        },
        "rescan_count": {
            "total": sum(int(item.get("rescan_count", 0)) for item in summaries),
            "per_episode": _metric(item.get("rescan_count") for item in summaries),
        },
        "failure_attribution_counts": attribution_counts,
        "episodes": [dict(item) for item in summaries],
        "claim_ceiling": CLAIM_CEILING,
    }
