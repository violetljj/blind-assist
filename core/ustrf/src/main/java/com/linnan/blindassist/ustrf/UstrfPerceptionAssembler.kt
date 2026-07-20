package com.linnan.blindassist.ustrf

/** Dynamic evidence is explicitly bound to both the originating frame and a local risk-grid cell. */
data class UstrfMotionGridEvidence(
    val sourceFrame: UstrfFrameStamp,
    val coordinate: UstrfGridCoordinate,
    val motion: UstrfRelativeMotionEvidence
)

enum class UstrfPerceptionAssemblyFailure {
    POSE_UNAVAILABLE,
    CAPTURE_UNAVAILABLE,
    GEOMETRY_SOURCE_FRAME_MISMATCH,
    GEOMETRY_UNAVAILABLE,
    MOTION_SOURCE_FRAME_MISMATCH,
    MOTION_EVIDENCE_UNAVAILABLE,
    GEOMETRY_NOT_METRIC,
    GEOMETRY_STALE,
    MOTION_UNAVAILABLE,
    PERCEPTION_TIMING_INVALID,
    POSE_DELTA_INVALID
}

sealed interface UstrfPerceptionAssembly {
    data class Available(val packet: UstrfPerceptionPacket) : UstrfPerceptionAssembly
    data class Unavailable(val failures: Set<UstrfPerceptionAssemblyFailure>) : UstrfPerceptionAssembly
}

/**
 * Pure-Kotlin seam between sensor-specific Adapters and the safety kernel.
 *
 * This intentionally assembles only co-timed, metric receipts. An empty dynamic list means that
 * the upstream motion Adapter reported no tracked dynamic evidence; it does not certify safety.
 */
class UstrfPerceptionAssembler(
    private val geometryProjector: UstrfGeometryProjector = UstrfGeometryProjector(),
    private val ttcEstimator: UstrfTtcEstimator = UstrfTtcEstimator(),
    private val dynamicCollisionRadiusMeters: Float = .75f
) {
    init {
        require(dynamicCollisionRadiusMeters > 0f)
    }

    fun assemble(
        frame: UstrfFrameStamp,
        geometry: UstrfGeometryPacket,
        dynamicMotion: List<UstrfMotionGridEvidence>,
        nowNs: Long
    ): UstrfPerceptionAssembly {
        require(nowNs >= frame.capturedAtNs) { "assembly cannot predate the source frame" }
        if (geometry.sourceFrame != frame) {
            return UstrfPerceptionAssembly.Unavailable(setOf(UstrfPerceptionAssemblyFailure.GEOMETRY_SOURCE_FRAME_MISMATCH))
        }
        val projected = geometryProjector.project(geometry, nowNs)
        if (projected is UstrfGeometryProjection.Unavailable) {
            return UstrfPerceptionAssembly.Unavailable(setOf(projected.failure.toAssemblyFailure()))
        }
        val geometryObservations = (projected as UstrfGeometryProjection.Available).observations
        val failures = dynamicMotion.mapNotNull { evidence ->
            when {
                evidence.sourceFrame != frame -> UstrfPerceptionAssemblyFailure.MOTION_SOURCE_FRAME_MISMATCH
                ttcEstimator.estimate(evidence.motion, nowNs) == null -> UstrfPerceptionAssemblyFailure.MOTION_UNAVAILABLE
                else -> null
            }
        }.toSet()
        if (failures.isNotEmpty()) return UstrfPerceptionAssembly.Unavailable(failures)

        val dynamicObservations = dynamicMotion.mapNotNull { evidence ->
            val estimate = requireNotNull(ttcEstimator.estimate(evidence.motion, nowNs))
            estimate.takeIf { it.closestDistanceMeters <= dynamicCollisionRadiusMeters }?.let {
                UstrfRiskObservation(
                    coordinate = evidence.coordinate,
                    occupancy = 0f,
                    traversability = 0f,
                    dropRisk = 0f,
                    headRisk = 0f,
                    dynamicTtcMs = it.timeToClosestApproachMs,
                    uncertainty = 0f,
                    source = evidence.motion.source,
                    validUntilNs = evidence.motion.validUntilNs
                )
            }
        }
        val validUntilNs = (listOf(geometry.validUntilNs) + dynamicMotion.map { it.motion.validUntilNs }).minOrNull()
            ?: geometry.validUntilNs
        return UstrfPerceptionAssembly.Available(
            UstrfPerceptionPacket(frame, nowNs, validUntilNs, geometryObservations + dynamicObservations)
        )
    }

    private fun UstrfGeometryProjectionFailure.toAssemblyFailure(): UstrfPerceptionAssemblyFailure = when (this) {
        UstrfGeometryProjectionFailure.SCALE_NOT_METRIC -> UstrfPerceptionAssemblyFailure.GEOMETRY_NOT_METRIC
        UstrfGeometryProjectionFailure.STALE -> UstrfPerceptionAssemblyFailure.GEOMETRY_STALE
    }
}
