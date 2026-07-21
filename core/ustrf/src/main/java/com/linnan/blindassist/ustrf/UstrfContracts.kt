package com.linnan.blindassist.ustrf

/** Production-isolated contracts. All timestamps use one monotonic nanosecond clock. */
data class UstrfFrameStamp(
    val frameId: Long,
    val capturedAtNs: Long,
    val coordinateFrame: String
) {
    init {
        require(frameId >= 0L) { "frameId must be non-negative" }
        require(capturedAtNs >= 0L) { "capturedAtNs must be non-negative" }
        require(coordinateFrame.isNotBlank()) { "coordinateFrame must not be blank" }
    }
}

enum class UstrfPoseState { TRACKING, DEGRADED, LOST, MISSING }
enum class UstrfEvidenceState { VALID, DEGRADED, MISSING }

data class UstrfHealth(
    val pose: UstrfPoseState,
    val capture: UstrfEvidenceState,
    val geometry: UstrfEvidenceState,
    val motion: UstrfEvidenceState,
    val thermal: UstrfEvidenceState = UstrfEvidenceState.VALID
)

data class UstrfRouteIntent(
    val coordinateFrame: String,
    val desiredOffsetCells: Int,
    val confidence: Float,
    val validUntilNs: Long
) {
    init {
        require(coordinateFrame.isNotBlank())
        require(confidence in 0f..1f)
        require(validUntilNs >= 0L)
    }

    fun isValidAt(nowNs: Long, expectedFrame: String, minimumConfidence: Float): Boolean =
        nowNs <= validUntilNs && coordinateFrame == expectedFrame && confidence >= minimumConfidence
}

data class UstrfGridCoordinate(val lateral: Int, val forward: Int) {
    init { require(forward >= 0) { "forward must be non-negative" } }
}

data class UstrfRiskObservation(
    val coordinate: UstrfGridCoordinate,
    val occupancy: Float,
    val traversability: Float,
    val dropRisk: Float,
    val headRisk: Float,
    val dynamicTtcMs: Long?,
    val uncertainty: Float,
    val source: String,
    val validUntilNs: Long
) {
    init {
        listOf(occupancy, traversability, dropRisk, headRisk, uncertainty).forEach {
            require(it in 0f..1f) { "probabilities must be in [0, 1]" }
        }
        require(dynamicTtcMs == null || dynamicTtcMs >= 0L)
        require(source.isNotBlank())
        require(validUntilNs >= 0L)
    }
}

data class UstrfPerceptionPacket(
    val sourceFrame: UstrfFrameStamp,
    val producedAtNs: Long,
    val validUntilNs: Long,
    val observations: List<UstrfRiskObservation>,
    val gridSpec: UstrfGridSpec = UstrfGridSpec.LEGACY_KERNEL
) {
    init {
        require(producedAtNs >= sourceFrame.capturedAtNs) { "perception cannot predate source" }
        require(validUntilNs >= producedAtNs) { "perception TTL expires before production" }
    }

    fun isFreshAt(nowNs: Long): Boolean = nowNs <= validUntilNs
}

/** Slow-loop information may explain a state or set a future task, but never signs a safety action. */
data class UstrfSemanticHint(
    val sourceFrame: UstrfFrameStamp,
    val producedAtNs: Long,
    val validUntilNs: Long,
    val confidence: Float,
    val label: String
) {
    init {
        require(producedAtNs >= sourceFrame.capturedAtNs)
        require(validUntilNs >= producedAtNs)
        require(confidence in 0f..1f)
        require(label.isNotBlank())
    }
}

enum class UstrfSafetyAction { SLOW_DOWN, STOP_AND_REASSESS, SCAN }

enum class UstrfSafetyReason {
    POSE_NOT_TRACKING,
    CAPTURE_UNAVAILABLE,
    GEOMETRY_UNAVAILABLE,
    MOTION_UNAVAILABLE,
    THERMAL_DEGRADED,
    SOURCE_FRAME_MISMATCH,
    PERCEPTION_ASSEMBLY_UNAVAILABLE,
    PERCEPTION_STALE,
    ROUTE_INVALID,
    CENTRAL_CORRIDOR_UNKNOWN,
    NO_SAFE_CORRIDOR,
    HARD_RISK,
    GRID_SPEC_MISMATCH,
    SHADOW_ONLY
}

data class UstrfSafetyDecision(
    val action: UstrfSafetyAction,
    val risk: Float,
    val confidence: Float,
    val validUntilNs: Long,
    val reasons: Set<UstrfSafetyReason>,
    /** Offline trace metadata only. It is never a user-facing direction command. */
    val experimentalCorridorOffsetCells: Int?
) {
    init {
        require(risk in 0f..1f)
        require(confidence in 0f..1f)
        require(validUntilNs >= 0L)
        require(reasons.isNotEmpty())
    }
}

data class UstrfTick(
    val frame: UstrfFrameStamp,
    val health: UstrfHealth,
    val perception: UstrfPerceptionPacket,
    val route: UstrfRouteIntent?,
    val semanticHints: List<UstrfSemanticHint> = emptyList(),
    /** Monotonic time when this decision is signed; it may be later than image capture. */
    val decisionAtNs: Long = frame.capturedAtNs
) {
    init { require(decisionAtNs >= frame.capturedAtNs) }
}
