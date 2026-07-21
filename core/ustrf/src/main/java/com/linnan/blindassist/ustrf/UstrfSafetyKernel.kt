package com.linnan.blindassist.ustrf

data class UstrfCorridorCandidate(
    val offsetCells: Int,
    val hardSafe: Boolean,
    val risk: Float,
    val hasUnknown: Boolean,
    val routeAlignment: Float = 0f
)

data class UstrfCorridorPlan(val candidates: List<UstrfCorridorCandidate>, val selected: UstrfCorridorCandidate?)

sealed interface UstrfRouteFieldCorridorResolution {
    data class Available(
        val plan: UstrfCorridorPlan,
        val intrusionEvidence: UstrfRouteIntrusionEvidence
    ) : UstrfRouteFieldCorridorResolution

    data class Unavailable(val failure: UstrfRouteConditionedRiskFailure) : UstrfRouteFieldCorridorResolution
}

class UstrfCorridorPlanner(
    val gridSpec: UstrfGridSpec = UstrfGridSpec.LEGACY_KERNEL,
    private val hardOccupancy: Float = 0.60f,
    private val hardDrop: Float = 0.20f,
    private val hardHead: Float = 0.35f,
    private val hardTtcMs: Long = 1_200L,
    private val goalDeviationPenalty: Float = .05f
) {
    init {
        require(goalDeviationPenalty >= 0f)
    }

    fun plan(field: UstrfRiskField, route: UstrfRouteIntent): UstrfCorridorPlan {
        require(field.gridSpec == gridSpec) { "risk-field grid spec does not match corridor planner" }
        val candidates = gridSpec.candidateOffsets.map { evaluate(field, it) }
        val selected = candidates.filter { it.hardSafe }.minWithOrNull(
            compareBy<UstrfCorridorCandidate> { it.risk + goalDeviationPenalty * kotlin.math.abs(it.offsetCells - route.desiredOffsetCells) }
                .thenBy { kotlin.math.abs(it.offsetCells - route.desiredOffsetCells) }
                .thenBy { it.offsetCells }
        )
        return UstrfCorridorPlan(candidates, selected)
    }

    fun plan(
        field: UstrfRiskField,
        route: UstrfRouteFieldReceipt,
        decisionAtNs: Long,
        runtime: Boolean,
        interactor: UstrfRouteConditionedRiskInteractor = UstrfRouteConditionedRiskInteractor()
    ): UstrfRouteFieldCorridorResolution {
        require(field.gridSpec == gridSpec) { "risk-field grid spec does not match corridor planner" }
        val interaction = interactor.interact(field, route, decisionAtNs, runtime)
        if (interaction is UstrfRouteConditionedRiskResolution.Unavailable) {
            return UstrfRouteFieldCorridorResolution.Unavailable(interaction.failure)
        }
        val evidence = (interaction as UstrfRouteConditionedRiskResolution.Available).evidence
        val candidates = gridSpec.candidateOffsets.map { offset ->
            evaluate(field, offset).copy(routeAlignment = routeAlignment(route, offset))
        }
        val selected = candidates.filter { it.hardSafe }.minWithOrNull(
            compareBy<UstrfCorridorCandidate> { it.risk + goalDeviationPenalty * (1f - it.routeAlignment) }
                .thenByDescending { it.routeAlignment }
                .thenBy { kotlin.math.abs(it.offsetCells) }
                .thenBy { it.offsetCells }
        )
        return UstrfRouteFieldCorridorResolution.Available(UstrfCorridorPlan(candidates, selected), evidence)
    }

    private fun evaluate(field: UstrfRiskField, offset: Int): UstrfCorridorCandidate {
        var risk = 0f
        var hardSafe = true
        var unknown = false
        for (forward in 1..gridSpec.horizonCells) {
            for (lateral in (offset - gridSpec.bodyCapsuleHalfWidthCells)..(offset + gridSpec.bodyCapsuleHalfWidthCells)) {
                val cell = field.cellAt(UstrfGridCoordinate(lateral, forward))
                val dynamicRisk = if ((cell.dynamicTtcMs ?: Long.MAX_VALUE) <= hardTtcMs) 1f else 0f
                risk = maxOf(risk, cell.occupancy, cell.dropRisk, cell.headRisk, cell.uncertainty, dynamicRisk)
                val hard = cell.occupancy >= hardOccupancy || cell.dropRisk >= hardDrop || cell.headRisk >= hardHead || dynamicRisk > 0f || cell.isUnknown
                hardSafe = hardSafe && !hard
                unknown = unknown || cell.isUnknown
            }
        }
        return UstrfCorridorCandidate(offset, hardSafe, risk, unknown)
    }

    private fun routeAlignment(route: UstrfRouteFieldReceipt, offset: Int): Float {
        val total = route.weights.values.sum()
        var swept = 0f
        for (forward in 1..gridSpec.horizonCells) {
            for (lateral in (offset - gridSpec.bodyCapsuleHalfWidthCells)..(offset + gridSpec.bodyCapsuleHalfWidthCells)) {
                swept += route.weights[UstrfGridCoordinate(lateral, forward)] ?: 0f
            }
        }
        return (swept / total).coerceIn(0f, 1f)
    }
}

/** The only authority allowed to emit a USTRF experimental safety action. */
class UstrfSafetySupervisor(
    private val routeMinimumConfidence: Float = 0.70f,
    private val decisionTtlNs: Long = 250_000_000L,
    private val centralHorizonCells: Int = 4
) {
    init { require(centralHorizonCells >= 1) }

    fun decide(
        tick: UstrfTick,
        field: UstrfRiskField,
        plan: UstrfCorridorPlan?,
        externalReasons: Set<UstrfSafetyReason> = emptySet()
    ): UstrfSafetyDecision {
        val now = tick.decisionAtNs
        val reasons = linkedSetOf(UstrfSafetyReason.SHADOW_ONLY)
        reasons += externalReasons
        if (tick.health.pose != UstrfPoseState.TRACKING) reasons += UstrfSafetyReason.POSE_NOT_TRACKING
        if (tick.health.capture != UstrfEvidenceState.VALID) reasons += UstrfSafetyReason.CAPTURE_UNAVAILABLE
        if (tick.health.geometry != UstrfEvidenceState.VALID) reasons += UstrfSafetyReason.GEOMETRY_UNAVAILABLE
        if (tick.health.motion != UstrfEvidenceState.VALID) reasons += UstrfSafetyReason.MOTION_UNAVAILABLE
        if (tick.health.thermal != UstrfEvidenceState.VALID) reasons += UstrfSafetyReason.THERMAL_DEGRADED
        if (tick.perception.sourceFrame != tick.frame) reasons += UstrfSafetyReason.SOURCE_FRAME_MISMATCH
        if (!tick.perception.isFreshAt(now)) reasons += UstrfSafetyReason.PERCEPTION_STALE
        val route = tick.route
        if (route == null || !route.isValidAt(now, tick.frame.coordinateFrame, routeMinimumConfidence)) reasons += UstrfSafetyReason.ROUTE_INVALID
        if ((1..centralHorizonCells).any { field.cellAt(UstrfGridCoordinate(0, it)).isUnknown }) reasons += UstrfSafetyReason.CENTRAL_CORRIDOR_UNKNOWN
        val selected = plan?.selected
        if (route != null && plan != null && selected == null) reasons += UstrfSafetyReason.NO_SAFE_CORRIDOR
        if (selected != null && selected.risk >= 0.60f) reasons += UstrfSafetyReason.HARD_RISK
        val action = when {
            UstrfSafetyReason.POSE_NOT_TRACKING in reasons || UstrfSafetyReason.SOURCE_FRAME_MISMATCH in reasons ||
                UstrfSafetyReason.CAPTURE_UNAVAILABLE in reasons || UstrfSafetyReason.GEOMETRY_UNAVAILABLE in reasons ||
                UstrfSafetyReason.MOTION_UNAVAILABLE in reasons || UstrfSafetyReason.PERCEPTION_STALE in reasons ||
                UstrfSafetyReason.GRID_SPEC_MISMATCH in reasons ||
                UstrfSafetyReason.CENTRAL_CORRIDOR_UNKNOWN in reasons || UstrfSafetyReason.NO_SAFE_CORRIDOR in reasons ||
                UstrfSafetyReason.HARD_RISK in reasons -> UstrfSafetyAction.STOP_AND_REASSESS
            UstrfSafetyReason.ROUTE_INVALID in reasons -> UstrfSafetyAction.SCAN
            else -> UstrfSafetyAction.SLOW_DOWN
        }
        return UstrfSafetyDecision(action, selected?.risk ?: 1f, if (action == UstrfSafetyAction.SLOW_DOWN) .5f else 0f, now + decisionTtlNs, reasons, selected?.offsetCells)
    }
}

data class UstrfReplayRecord(val frameId: Long, val capturedAtNs: Long, val decision: UstrfSafetyDecision)

/** One assembled fast-loop input. The production adapter remains outside this pure Kotlin module. */
data class UstrfSessionInput(
    val frame: UstrfFrameStamp,
    val health: UstrfHealth,
    val perception: UstrfPerceptionAssembly,
    val route: UstrfRouteIntent?,
    val semanticHints: List<UstrfSemanticHint> = emptyList(),
    val decisionAtNs: Long = frame.capturedAtNs,
    /** Optional, independently verified delta for offline replay only; invalid deltas fail closed. */
    val poseDelta: UstrfVerifiedPoseDelta? = null,
    /** Continuous external route authority; mutually exclusive with the legacy offset intent. */
    val routeField: UstrfRouteFieldReceipt? = null
) {
    init {
        require(decisionAtNs >= frame.capturedAtNs)
        require(route == null || routeField == null) { "legacy route intent and continuous route field are mutually exclusive" }
    }
}

data class UstrfSessionRecord(
    val frameId: Long,
    val capturedAtNs: Long,
    val field: UstrfRiskField?,
    val decision: UstrfSafetyDecision,
    val assemblyFailures: Set<UstrfPerceptionAssemblyFailure>,
    /** Shadow-only downstream feedback contract, derived from [decision] and never authoritative. */
    val structuredOutput: UstrfStructuredSafetyOutput,
    val routeIntrusionEvidence: UstrfRouteIntrusionEvidence? = null
)

/**
 * Composition root for an Adapter-independent USTRF fast loop.
 *
 * No Adapter failure is allowed to leak a partial packet into the field. It is instead made
 * explicit to the supervisor, whose only public actions are still shadow-only safety actions.
 */
class UstrfSafetySession(
    private val fieldBuilder: UstrfRiskFieldBuilder = UstrfRiskFieldBuilder(),
    private val planner: UstrfCorridorPlanner = UstrfCorridorPlanner(),
    private val supervisor: UstrfSafetySupervisor = UstrfSafetySupervisor(),
    private val structuredOutputMapper: UstrfStructuredSafetyOutputMapper = UstrfStructuredSafetyOutputMapper()
) {
    private var priorFrame: UstrfFrameStamp? = null

    init {
        require(fieldBuilder.gridSpec == planner.gridSpec) { "field and planner grid specs must match" }
        require(fieldBuilder.gridSpec == structuredOutputMapper.gridSpec) { "field and output grid specs must match" }
    }

    fun evaluate(input: UstrfSessionInput): UstrfSessionRecord {
        priorFrame?.let { prior ->
            require(input.frame.frameId > prior.frameId) { "session frame IDs must be strictly increasing" }
            require(input.frame.capturedAtNs > prior.capturedAtNs) { "session timestamps must be strictly increasing" }
            require(input.frame.coordinateFrame == prior.coordinateFrame) { "session coordinate frame changed" }
        }
        val normalised = input.perception.normalise(input.frame, input.decisionAtNs)
        val healthFailures = input.health.fastLoopAdmissionFailures()
        val failures = normalised.failures + healthFailures
        priorFrame = input.frame
        if (normalised.availablePacket == null || failures.isNotEmpty()) return unavailableRecord(input, failures)

        val packet = normalised.availablePacket
        val field = try {
            fieldBuilder.update(packet, input.poseDelta)
        } catch (error: IllegalArgumentException) {
            return unavailableRecord(input, setOf(UstrfPerceptionAssemblyFailure.POSE_DELTA_INVALID))
        }
        var routeEvidence: UstrfRouteIntrusionEvidence? = null
        val continuousResolution = input.routeField?.let {
            planner.plan(field, it, input.decisionAtNs, runtime = true)
        }
        val route = when (continuousResolution) {
            is UstrfRouteFieldCorridorResolution.Available -> {
                routeEvidence = continuousResolution.intrusionEvidence
                UstrfRouteIntent(
                    input.frame.coordinateFrame,
                    continuousResolution.plan.selected?.offsetCells ?: 0,
                    input.routeField!!.confidence,
                    input.routeField.validUntilNs
                )
            }
            is UstrfRouteFieldCorridorResolution.Unavailable -> null
            null -> input.route?.takeIf { it.isValidAt(input.decisionAtNs, input.frame.coordinateFrame, .70f) }
        }
        val plan = when (continuousResolution) {
            is UstrfRouteFieldCorridorResolution.Available -> continuousResolution.plan
            is UstrfRouteFieldCorridorResolution.Unavailable -> null
            null -> route?.let { planner.plan(field, it) }
        }
        val tick = UstrfTick(input.frame, input.health, packet, route, input.semanticHints, input.decisionAtNs)
        val decision = supervisor.decide(tick, field, plan)
        return UstrfSessionRecord(
            input.frame.frameId,
            input.frame.capturedAtNs,
            field,
            decision,
            emptySet(),
            structuredOutputMapper.map(decision, plan?.selected),
            routeEvidence
        )
    }

    fun reset() {
        priorFrame = null
        fieldBuilder.reset()
    }

    private fun unavailableRecord(input: UstrfSessionInput, failures: Set<UstrfPerceptionAssemblyFailure>): UstrfSessionRecord {
        // A rejected atomic frame breaks the evidence chain. Do not let a later recovery frame
        // decay/reuse a field that may belong to an untracked or otherwise unverifiable interval.
        fieldBuilder.reset()
        val health = input.health.withAssemblyFailures(failures)
        val packet = UstrfPerceptionPacket(
            input.frame,
            input.frame.capturedAtNs,
            input.frame.capturedAtNs,
            emptyList(),
            fieldBuilder.gridSpec
        )
        val tick = UstrfTick(input.frame, health, packet, input.route, input.semanticHints, input.decisionAtNs)
        val reasons = failures.toSafetyReasons() + UstrfSafetyReason.PERCEPTION_ASSEMBLY_UNAVAILABLE
        val emptyField = UstrfRiskField(input.frame, emptyMap(), fieldBuilder.gridSpec)
        val decision = supervisor.decide(tick, emptyField, null, reasons)
        return UstrfSessionRecord(
            input.frame.frameId,
            input.frame.capturedAtNs,
            null,
            decision,
            failures,
            structuredOutputMapper.map(decision, null)
        )
    }

    private data class NormalisedAssembly(
        val availablePacket: UstrfPerceptionPacket?,
        val failures: Set<UstrfPerceptionAssemblyFailure>
    ) {
    }

    private fun UstrfPerceptionAssembly.normalise(frame: UstrfFrameStamp, decisionAtNs: Long): NormalisedAssembly = when (this) {
        is UstrfPerceptionAssembly.Available -> {
            when {
                packet.sourceFrame != frame -> NormalisedAssembly(null, setOf(UstrfPerceptionAssemblyFailure.GEOMETRY_SOURCE_FRAME_MISMATCH))
                packet.gridSpec != fieldBuilder.gridSpec -> NormalisedAssembly(null, setOf(UstrfPerceptionAssemblyFailure.GRID_SPEC_MISMATCH))
                packet.producedAtNs > decisionAtNs || !packet.isFreshAt(decisionAtNs) -> NormalisedAssembly(null, setOf(UstrfPerceptionAssemblyFailure.PERCEPTION_TIMING_INVALID))
                else -> NormalisedAssembly(packet, emptySet())
            }
        }
        is UstrfPerceptionAssembly.Unavailable -> NormalisedAssembly(null, failures)
    }

    private fun UstrfHealth.withAssemblyFailures(failures: Set<UstrfPerceptionAssemblyFailure>): UstrfHealth = copy(
        pose = if (UstrfPerceptionAssemblyFailure.POSE_UNAVAILABLE in failures) UstrfPoseState.MISSING else pose,
        capture = if (UstrfPerceptionAssemblyFailure.CAPTURE_UNAVAILABLE in failures) UstrfEvidenceState.MISSING else capture,
        geometry = if (failures.any { it == UstrfPerceptionAssemblyFailure.GEOMETRY_NOT_METRIC || it == UstrfPerceptionAssemblyFailure.GEOMETRY_STALE || it == UstrfPerceptionAssemblyFailure.GEOMETRY_SOURCE_FRAME_MISMATCH || it == UstrfPerceptionAssemblyFailure.GEOMETRY_UNAVAILABLE }) UstrfEvidenceState.MISSING else geometry,
        motion = if (failures.any { it == UstrfPerceptionAssemblyFailure.MOTION_UNAVAILABLE || it == UstrfPerceptionAssemblyFailure.MOTION_SOURCE_FRAME_MISMATCH || it == UstrfPerceptionAssemblyFailure.MOTION_EVIDENCE_UNAVAILABLE }) UstrfEvidenceState.MISSING else motion
    )

    private fun UstrfHealth.fastLoopAdmissionFailures(): Set<UstrfPerceptionAssemblyFailure> = buildSet {
        if (pose != UstrfPoseState.TRACKING) add(UstrfPerceptionAssemblyFailure.POSE_UNAVAILABLE)
        if (capture != UstrfEvidenceState.VALID) add(UstrfPerceptionAssemblyFailure.CAPTURE_UNAVAILABLE)
        if (geometry != UstrfEvidenceState.VALID) add(UstrfPerceptionAssemblyFailure.GEOMETRY_UNAVAILABLE)
        if (motion != UstrfEvidenceState.VALID) add(UstrfPerceptionAssemblyFailure.MOTION_EVIDENCE_UNAVAILABLE)
    }

    private fun Set<UstrfPerceptionAssemblyFailure>.toSafetyReasons(): Set<UstrfSafetyReason> = map {
        when (it) {
            UstrfPerceptionAssemblyFailure.POSE_UNAVAILABLE -> UstrfSafetyReason.POSE_NOT_TRACKING
            UstrfPerceptionAssemblyFailure.POSE_DELTA_INVALID -> UstrfSafetyReason.POSE_NOT_TRACKING
            UstrfPerceptionAssemblyFailure.CAPTURE_UNAVAILABLE -> UstrfSafetyReason.CAPTURE_UNAVAILABLE
            UstrfPerceptionAssemblyFailure.GEOMETRY_SOURCE_FRAME_MISMATCH,
            UstrfPerceptionAssemblyFailure.MOTION_SOURCE_FRAME_MISMATCH -> UstrfSafetyReason.SOURCE_FRAME_MISMATCH
            UstrfPerceptionAssemblyFailure.GEOMETRY_NOT_METRIC,
            UstrfPerceptionAssemblyFailure.GEOMETRY_STALE,
            UstrfPerceptionAssemblyFailure.GEOMETRY_UNAVAILABLE -> UstrfSafetyReason.GEOMETRY_UNAVAILABLE
            UstrfPerceptionAssemblyFailure.MOTION_UNAVAILABLE,
            UstrfPerceptionAssemblyFailure.MOTION_EVIDENCE_UNAVAILABLE -> UstrfSafetyReason.MOTION_UNAVAILABLE
            UstrfPerceptionAssemblyFailure.PERCEPTION_TIMING_INVALID -> UstrfSafetyReason.PERCEPTION_STALE
            UstrfPerceptionAssemblyFailure.GRID_SPEC_MISMATCH -> UstrfSafetyReason.GRID_SPEC_MISMATCH
        }
    }.toSet()
}

class UstrfReplayRunner(
    private val fieldBuilder: UstrfRiskFieldBuilder = UstrfRiskFieldBuilder(),
    private val planner: UstrfCorridorPlanner = UstrfCorridorPlanner(),
    private val supervisor: UstrfSafetySupervisor = UstrfSafetySupervisor()
) {
    init {
        require(fieldBuilder.gridSpec == planner.gridSpec) { "field and planner grid specs must match" }
    }

    fun run(ticks: List<UstrfTick>): List<UstrfReplayRecord> {
        var prior: UstrfFrameStamp? = null
        return ticks.map { tick ->
            prior?.let {
                require(tick.frame.frameId > it.frameId) { "replay frame IDs must be strictly increasing" }
                require(tick.frame.capturedAtNs > it.capturedAtNs) { "replay timestamps must be strictly increasing" }
                require(tick.frame.coordinateFrame == it.coordinateFrame) { "replay coordinate frame changed" }
            }
            require(tick.perception.sourceFrame == tick.frame) { "perception must bind to the current frame" }
            val field = fieldBuilder.update(tick.perception)
            val route = tick.route?.takeIf { it.isValidAt(tick.frame.capturedAtNs, tick.frame.coordinateFrame, .70f) }
            val plan = route?.let { planner.plan(field, it) }
            prior = tick.frame
            UstrfReplayRecord(tick.frame.frameId, tick.frame.capturedAtNs, supervisor.decide(tick, field, plan))
        }
    }
}
