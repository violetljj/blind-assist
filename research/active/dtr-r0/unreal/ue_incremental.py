"""Incremental execution of the retained, unmodified X73 frame programs.

The research implementation keeps trackers and credentials in episode-function
locals.  Replaying every prefix throws those locals away.  This adapter compiles
the *explicitly named* functions' initialization and frame loop separately, then
keeps each scope alive for an episode.  One update executes one iteration of each
loop.  No episode predictor is called, and no history prefix is copied/replayed.

Source is the trusted, installed research code, not a model/data input.  Keeping
the original statements (including nested continue, precedence, and thresholds)
avoids a second handwritten implementation of the frozen algorithm.  Unsupported
function layouts fail at construction.  The batch functions remain the oracle.
Each stage retains its own unrenamed previous frame: batch renaming happens only
after that stage's episode loop and must not affect temporal lineage decisions.
Reviewed whole-function body hashes also bind the statements intentionally left
outside the loop. A changed batch body needs a reviewed contract update and
differential validation; new pre/post-loop semantics can never be silently lost.
"""
from __future__ import annotations

import ast
import copy
from contextlib import contextmanager
import importlib
import inspect
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace, ModuleType


_SOURCE_STAGES = {
    24: "plan_adherent_predictor", 25: "rigid_footprint_predictor",
    30: "adaptive_surface_interval_predictor", 31: "ambiguity_preserving_transport_predictor",
    32: "observation_conditioned_core_predictor", 33: "dormant_transport_reactivation_predictor",
    34: "bounded_dormant_occupancy_flow_predictor", 35: "dormant_flow_consensus_predictor",
    37: "motion_evidence_credit_predictor", 38: "metric_closing_bootstrap_predictor",
    39: "cross_representation_handoff_predictor", 40: "cross_representation_lateral_adjudicator",
    41: "metric_credentialed_parent_continuation", 42: "instant_closing_consensus_predictor",
    43: "authority_preserving_credential_belief", 44: "causal_velocity_cycle_credential",
    45: "causal_state_cycle_credential", 51: "provisional_motion_belief_update",
    52: "cross_parent_provisional_reidentification", 53: "anchor_redundant_parent_continuation",
    54: "metric_bootstrap_dropout_continuation", 57: "retained_core_metric_handback",
    59: "modality_evidence_reliability_router", 64: "unanchored_crossing_release",
    65: "ancestry_synchronized_conflict_handback", 67: "measurement_horizon_receding_release",
    68: "object_local_lateral_dequantization", 69: "mature_cross_route_rigid_contradiction",
    70: "triple_credential_surface_dropout_handback", 71: "entry_cotransport_occupancy_birth",
    72: "credentialed_surface_boundary_completion", 73: "credentialed_parent_hull_reconstruction",
}
_SOURCE_NAMES = {f"dtr_carla_x{number}_{suffix}" for number, suffix in _SOURCE_STAGES.items()}
SOURCE_CONTRACT = Path(__file__).with_name("ue_incremental_source_contract.json")
_contract = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
if _contract.get("schema") != "ue-incremental-x73-source-contract-v1" or _contract.get("normalization") != "INSPECT_SOURCE_UNIVERSAL_NEWLINES_STRIP_PLUS_NEWLINE":
    raise ValueError("Unsupported incremental X73 source contract")
_ADMITTED_BODIES = _contract["functions"]


def _source(function):
    return inspect.getsource(function).replace("\r\n", "\n").strip() + "\n"


def _module(number, suffix):
    return importlib.import_module(f"dtr_carla_x{number}_{suffix}")


def source_paths():
    """Local Python dependency closure for a caller's immutable source receipt."""
    import ue_dtr_replay as replay
    import ue_replay_cache as cache
    root = replay.REPO.resolve()
    pending = [replay, cache, importlib.import_module(__name__),
               *[_module(number, suffix) for number, suffix in _SOURCE_STAGES.items()]]
    seen, paths = set(), set()
    while pending:
        module = pending.pop()
        if module.__name__ in seen:
            continue
        seen.add(module.__name__)
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py" or not path.is_relative_to(root):
            continue
        paths.add(path)
        pending.extend(value for value in vars(module).values() if isinstance(value, ModuleType))
    paths.add(SOURCE_CONTRACT.resolve())
    return sorted(paths)


def _tree(function):
    expected = Path(__file__).resolve().parent.parent / "carla" / (function.__module__ + ".py")
    if function.__module__ not in _SOURCE_NAMES or Path(inspect.getsourcefile(function)).resolve() != expected:
        raise ValueError("Frame programs must belong to the explicit local X73 source allowlist")
    source = _source(function)
    identity = function.__module__ + "." + function.__name__
    if hashlib.sha256(source.encode()).hexdigest() != _ADMITTED_BODIES.get(identity):
        raise ValueError(f"Unreviewed incremental frame program body: {identity}; review and update the source contract")
    module = ast.parse(source)
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise ValueError(f"Unsupported frame program: {function.__qualname__}")
    return module.body[0]


def _code(statements, function):
    tree = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(statements), type_ignores=[]))
    return compile(tree, f"{inspect.getsourcefile(function)}::{function.__name__}:incremental", "exec")


def _frame_loop(node):
    return isinstance(node, ast.For) and any(
        isinstance(part, ast.Name) and part.id.endswith("frame")
        for part in ast.walk(node.target)
    )


class _Recent:
    """Absolute-indexed two-item history; accessing discarded/future data fails."""
    def __init__(self):
        self.count = 0
        self.items = {}
        self.first_next = None

    def append(self, value):
        self.items[self.count] = value
        self.count += 1
        self.items.pop(self.count - 3, None)

    def __len__(self):
        return self.count

    def __iter__(self):
        raise TypeError("Incremental history is not a replayable prefix")

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("Incremental history does not expose prefixes")
        if index < 0:
            index += self.count
        if index in self.items:
            return self.items[index]
        # X69 uses only the second timestamp to initialize the first period.
        # This is supplied only after the second observation has actually arrived.
        if index == 1 and self.count == 1 and self.first_next is not None:
            return {"time_s": self.first_next}
        raise IndexError(f"Frame {index} is outside the retained causal state")


class _Program:
    """A persistent scope for one original, explicitly selected frame loop."""
    INPUT_NAMES = {"value", "core", "metric", "rigid", "baseline", "observations"}

    def __init__(self, function, *, base=False, overrides=None):
        definition = _tree(function)
        loops = [node for node in definition.body if isinstance(node, ast.For)]
        if not loops:
            raise ValueError(f"No frame loop: {function.__name__}")
        loop = loops[0]
        self.function = function
        self.name = function.__module__.split("dtr_carla_")[-1]
        self.source_sha256 = hashlib.sha256(_source(function).encode()).hexdigest()
        self.base = base
        self.scope = dict(function.__globals__)
        self.scope.update(overrides or {})
        initialization = []
        for node in definition.body[:definition.body.index(loop)]:
            if isinstance(node, (ast.Expr, ast.Delete)):
                # Episode-count guards are enforced by update's source join.
                # These are the only non-assignment pre-loop statements in the
                # admitted programs (docstrings, require(...), del arguments).
                if isinstance(node, ast.Expr) and not (
                    isinstance(node.value, ast.Constant)
                    or isinstance(node.value, ast.Call)
                    and (isinstance(node.value.func, ast.Name) and node.value.func.id == "require"
                         or isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "require")
                ):
                    raise ValueError(f"Unsupported initialization: {function.__name__}")
                continue
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                raise ValueError(f"Unsupported initialization: {function.__name__}")
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if len(names) != len(targets):
                raise ValueError(f"Nonlocal initialization: {function.__name__}")
            if names & self.INPUT_NAMES:
                continue  # Upstream stages are explicit edges in IncrementalX73.
            if any(isinstance(part, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
                   for part in ast.walk(node)):
                raise ValueError(f"History-dependent initialization: {function.__name__}")
            initialization.append(node)
        # A local helper/recursive episode call in a new implementation would
        # require a reviewed adapter, never silently restore prefix recomputation.
        for node in [*initialization, *loop.body]:
            for part in ast.walk(node):
                if isinstance(part, ast.Call):
                    name = getattr(part.func, "attr", getattr(part.func, "id", ""))
                    if name == "predict_episode" or name.startswith("apply_") and name.endswith("_episode"):
                        raise ValueError(f"Episode call inside frame program: {function.__name__}")
        if loop.orelse:
            raise ValueError(f"Unsupported loop else: {function.__name__}")
        self.target = loop.target
        once = copy.deepcopy(loop)
        once.iter = ast.Tuple(elts=[ast.Name(id="_payload", ctx=ast.Load())], ctx=ast.Load())
        self.initialize_code = _code(initialization, function)
        self.update_code = _code([once], function)
        self.finish_code = None
        if not base:
            if len(loops) != 2 or not _frame_loop(loops[-1]):
                raise ValueError(f"Unsupported final frame mapping: {function.__name__}")
            self.finish_code = _code(loops[-1].body, function)
        elif len(loops) != 1:
            raise ValueError(f"Unexpected base loops: {function.__name__}")
        self.frames = _Recent()
        self.initialized = False
        self.processed_count = 0

    def _payload(self, target, inputs):
        if isinstance(target, ast.Name):
            return inputs[target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(self._payload(item, inputs) for item in target.elts)
        raise ValueError("Unsupported frame loop target")

    def update(self, *, episode, observation, candidate, calibration, ordinal,
               frame=None, metric=None, rigid=None, first_next=None):
        value = None if self.base else copy.deepcopy(frame)
        self.frames.first_next = first_next
        if not self.base:
            self.frames.append(value)
        self.scope.update(episode=episode, observations=episode.observations,
                          calibration=calibration, candidate_values=(candidate,),
                          value={"frames": self.frames}, core={"frames": self.frames},
                          metric={"frames": [metric]}, rigid={"frames": [rigid]})
        if not self.initialized:
            exec(self.initialize_code, self.scope)
            if self.base:
                self.scope["frames"] = self.frames
            self.initialized = True
        inputs = {name: value for name in ("frame", "surface_frame", "fused_frame", "base_frame",
                                           "core_frame", "routed_frame")}
        inputs.update(ordinal=ordinal, observation=observation, candidate_value=candidate,
                      metric_frame=metric, rigid_frame=rigid)
        self.scope["_payload"] = self._payload(self.target, inputs)
        exec(self.update_code, self.scope)
        self.processed_count += 1
        # Only the current frame is copied. Older stage-owned state is not
        # exposed to later stages, callers, or hypothetical-route queries.
        output = copy.deepcopy(self.frames[-1])
        if self.finish_code is not None:
            self.scope["frame"] = output
            exec(self.finish_code, self.scope)
        return output


def _mapping(function):
    """Compile only a wrapper's final per-frame arm/field mapping."""
    loops = [node for node in _tree(function).body if _frame_loop(node)]
    if not loops:
        raise ValueError(f"No per-frame mapping: {function.__name__}")
    code = _code(loops[-1].body, function)

    def apply(frame):
        scope = dict(function.__globals__)
        scope["frame"] = frame
        exec(code, scope)
        return frame
    return apply


@contextmanager
def _binding(module, name, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


class IncrementalX73:
    """One episode's causal X73 state. Call update once for each observed frame.

    update returns a compact row. last_frame, last_metric_frame and
    last_rigid_frame are detached current-frame outputs (None during WARMUP).
    The first frame is deferred until its observed successor supplies the
    original predictor's initial sample period. No future pixel is inspected.
    The native codec and inherited X32/X52 helper bindings remain scoped to the
    serial replay call, as in the original UE runner. Engines/batch calls must
    not run concurrently in one interpreter; use separate processes for that.
    """
    def __init__(self, episode_id, route_frame, calibration):
        import ue_dtr_replay as replay
        self.replay = replay
        self.episode_id = episode_id
        self.route_frame = route_frame
        self.calibration = calibration
        self.observations = _Recent()
        self.pending = None
        self.update_count = 0
        self.processed_count = 0
        self.previous_risk = False
        self.last_frame = self.last_metric_frame = self.last_rigid_frame = None
        self.last_compact = None
        self.failed = False
        self.modules = {number: _module(number, suffix) for number, suffix in _SOURCE_STAGES.items()}
        m = self.modules
        self.metric = _Program(replay.x24.predict_episode, base=True)
        self.rigid = _Program(replay.x25.predict_episode, base=True)
        self.surface = _Program(m[30].predict_episode, base=True, overrides={
            "AdaptiveSurfaceLineageTracker": m[35].DormantFlowConsensusTracker,
            "arm_frame": m[37].motion_evidence_credit_arm_frame})
        self.surface_maps = [_mapping(m[n].predict_episode) for n in (31, 32, 33, 34, 35, 37)]
        self.early = [_Program(m[n].predict_episode) for n in range(38, 46)]
        self.provisional = _Program(m[51].apply_provisional_motion_episode,
                                    overrides={"continued_row": m[52].cross_parent_continued_row})
        self.provisional_map = _mapping(m[52].apply_cross_parent_episode)
        self.middle = [
            _Program(m[53].apply_anchor_redundancy_episode),
            _Program(m[54].apply_metric_bootstrap_dropout_episode),
            _Program(m[57].apply_retained_core_metric_handback_episode),
            _Program(m[59].apply_modality_evidence_reliability_router_episode),
            _Program(replay.x65.apply_ancestry_synchronized_conflict_handback_episode),
            _Program(m[64].apply_unanchored_crossing_release_episode),
        ]
        self.ancestry_map = _mapping(replay.x65.apply_ancestry_handback_episode)
        self.late = [
            _Program(replay.x67.apply_measurement_horizon_receding_release_episode),
            _Program(replay.x68.apply_object_local_lateral_dequantization_episode),
            _Program(replay.x69.apply_mature_cross_route_rigid_contradiction_episode),
            _Program(replay.x70.apply_triple_credential_surface_dropout_handback_episode),
            _Program(replay.x71.apply_entry_cotransport_occupancy_birth_episode),
            _Program(replay.x72.apply_credentialed_surface_boundary_completion_episode),
            _Program(replay.x73.apply_credentialed_parent_hull_reconstruction_episode),
        ]

    @property
    def stages(self):
        return [self.metric, self.rigid, self.surface, *self.early, self.provisional,
                *self.middle, *self.late]

    @property
    def stats(self):
        return {"update_count": self.update_count, "processed_count": self.processed_count,
                "stage_processed_counts": {stage.name: stage.processed_count for stage in self.stages},
                "retained_frame_counts": {stage.name: len(stage.frames.items) for stage in self.stages},
                "full_prefix_replays": 0}

    def update(self, observation, candidate_value):
        if self.failed:
            raise RuntimeError("Incremental engine failed; reconstruct from verified sensor input")
        if observation.episode_id != self.episode_id or observation.sample_index != self.update_count:
            raise ValueError("Incremental input must be this episode's next sample from zero")
        timestamp = float(observation.time_s)
        if not math.isfinite(timestamp) or self.update_count and timestamp <= self.observations[-1].time_s:
            raise ValueError("Incremental source timestamps must strictly increase")
        if not isinstance(candidate_value, dict) or not isinstance(candidate_value.get("candidates"), list):
            raise ValueError("Candidate input must contain the current candidates list")
        source = candidate_value.get("source")
        if source is not None:
            for key, expected in (("episode_id", self.episode_id), ("sample_index", observation.sample_index),
                                  ("time_s", observation.time_s), ("world_frame", observation.world_frame)):
                if source.get(key) != expected:
                    raise ValueError(f"Current detector/source join mismatch: {key}")
            if str(source.get("image_sha256", "")).lower() != observation.rgb.sha256.lower():
                raise ValueError("Current detector/source image identity mismatch")
        self.observations.append(observation)
        self.update_count += 1
        if self.update_count == 1:
            self.pending = (observation, copy.deepcopy(candidate_value))
            self.last_compact = {"episode_id": self.episode_id, "sample_index": observation.sample_index,
                                 "time_s": observation.time_s, "event": "WARMUP", "route_risk": False,
                                 "risk_state": "UNKNOWN_INSUFFICIENT_HISTORY",
                                 "support_state": "ONE_FRAME_ONLY",
                                 "global_observability": "UNKNOWN_NOT_ESTIMATED_BY_X73"}
            return copy.deepcopy(self.last_compact)
        try:
            if self.pending is not None:
                self._advance(*self.pending, first_next=timestamp)
                self.pending = None
            self._advance(observation, candidate_value)
        except Exception:
            self.failed = True
            raise
        return copy.deepcopy(self.last_compact)

    def _advance(self, observation, candidate, first_next=None):
        episode = SimpleNamespace(episode_id=self.episode_id, route_frame=self.route_frame,
                                  observations=self.observations)
        inputs = dict(episode=episode, observation=observation, candidate=candidate,
                      calibration=self.calibration, ordinal=self.processed_count, first_next=first_next)
        m = self.modules
        # Existing per-invocation input reuse is bounded to this one observation.
        # In particular, it never supplies a tracker or previous prediction.
        from ue_replay_cache import cached_replay_inputs
        with cached_replay_inputs(), self.replay.native_depth_loader():
            metric = self.metric.update(**inputs)
            rigid = self.rigid.update(**inputs)
            with _binding(m[31], "_coalesce_current_shift_envelopes", m[32].representative_shift_cores):
                frame = self.surface.update(**inputs)
            for mapping in self.surface_maps:
                frame = mapping(frame)
            for stage in self.early:
                frame = stage.update(**inputs, frame=frame, metric=metric, rigid=rigid)
            with _binding(m[45], "closes_state_cycle", m[52].cross_parent_cycle), \
                 _binding(m[51], "continued_row", m[52].cross_parent_continued_row):
                frame = self.provisional.update(**inputs, frame=frame, metric=metric, rigid=rigid)
            frame = self.provisional_map(frame)
            for stage in self.middle:
                frame = stage.update(**inputs, frame=frame, metric=metric, rigid=rigid)
            frame = self.ancestry_map(frame)
            for stage in self.late:
                frame = stage.update(**inputs, frame=frame, metric=metric, rigid=rigid)
        self.last_metric_frame, self.last_rigid_frame, self.last_frame = metric, rigid, frame
        self.last_compact = self.replay.compact_frame(self.episode_id, frame, self.previous_risk)
        self.previous_risk = self.last_compact["route_risk"]
        self.processed_count += 1
