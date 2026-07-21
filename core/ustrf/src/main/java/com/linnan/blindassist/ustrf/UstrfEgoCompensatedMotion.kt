package com.linnan.blindassist.ustrf

import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Two observations of the same tracked target in their respective user-local body frames.
 *
 * This contract deliberately contains no detector semantics: its only purpose is to make an
 * object's relative velocity explicit after the user's motion has been removed. It cannot be
 * constructed from frame-to-frame pixels alone without an independently verified pose delta.
 */
data class UstrfDynamicTrackPair(
    val trackId: String,
    val previousFrame: UstrfFrameStamp,
    val currentFrame: UstrfFrameStamp,
    val previousPositionMeters: UstrfVector2,
    val currentPositionMeters: UstrfVector2,
    val confidence: Float,
    val source: String,
    val validUntilNs: Long
) {
    init {
        require(trackId.isNotBlank())
        require(previousFrame.coordinateFrame == currentFrame.coordinateFrame)
        require(currentFrame.frameId > previousFrame.frameId)
        require(currentFrame.capturedAtNs > previousFrame.capturedAtNs)
        require(confidence in 0f..1f)
        require(source.isNotBlank())
        require(validUntilNs >= currentFrame.capturedAtNs)
    }
}

enum class UstrfEgoCompensatedMotionFailure {
    FRAME_BINDING_MISMATCH,
    POSE_DELTA_UNVERIFIED,
    POSE_DELTA_BINDING_MISMATCH,
    TRACK_LOW_CONFIDENCE,
    TRACK_STALE,
    TARGET_BEHIND_ORIGIN,
    TARGET_OUTSIDE_LOCAL_GRID
}

sealed interface UstrfEgoCompensatedMotionResolution {
    data class Available(
        val evidence: UstrfMotionGridEvidence,
        val egoCompensatedPreviousPositionMeters: UstrfVector2
    ) : UstrfEgoCompensatedMotionResolution

    data class Unavailable(val failure: UstrfEgoCompensatedMotionFailure) : UstrfEgoCompensatedMotionResolution
}

/**
 * Promotes a same-target pair to TTC-ready relative motion only after an explicit user-motion
 * transform. The result remains a shadow/offline input until an actual tracker and event set pass
 * their separate gates.
 */
class UstrfEgoCompensatedMotionPromoter(
    val gridSpec: UstrfGridSpec = UstrfGridSpec.LEGACY_KERNEL,
    private val minimumTrackConfidence: Float = .70f
) {
    init {
        require(minimumTrackConfidence in 0f..1f)
    }

    fun promote(
        pair: UstrfDynamicTrackPair,
        poseDelta: UstrfVerifiedPoseDelta,
        nowNs: Long
    ): UstrfEgoCompensatedMotionResolution {
        if (pair.previousFrame.coordinateFrame != pair.currentFrame.coordinateFrame) {
            return unavailable(UstrfEgoCompensatedMotionFailure.FRAME_BINDING_MISMATCH)
        }
        if (!poseDelta.verifiedForOfflineReplay) return unavailable(UstrfEgoCompensatedMotionFailure.POSE_DELTA_UNVERIFIED)
        if (poseDelta.fromFrame != pair.previousFrame || poseDelta.toFrame != pair.currentFrame) {
            return unavailable(UstrfEgoCompensatedMotionFailure.POSE_DELTA_BINDING_MISMATCH)
        }
        if (pair.confidence < minimumTrackConfidence) return unavailable(UstrfEgoCompensatedMotionFailure.TRACK_LOW_CONFIDENCE)
        if (nowNs > pair.validUntilNs) return unavailable(UstrfEgoCompensatedMotionFailure.TRACK_STALE)

        val previousInCurrent = transformPriorPosition(pair.previousPositionMeters, poseDelta)
        val current = pair.currentPositionMeters
        if (current.forward < 0f) return unavailable(UstrfEgoCompensatedMotionFailure.TARGET_BEHIND_ORIGIN)
        val coordinate = UstrfGridCoordinate(
            lateral = (current.lateral / gridSpec.cellMeters).roundToInt(),
            forward = (current.forward / gridSpec.cellMeters).roundToInt()
        )
        if (!gridSpec.contains(coordinate)) {
            return unavailable(UstrfEgoCompensatedMotionFailure.TARGET_OUTSIDE_LOCAL_GRID)
        }
        val seconds = (pair.currentFrame.capturedAtNs - pair.previousFrame.capturedAtNs) / 1_000_000_000f
        val relativeVelocity = UstrfVector2(
            forward = (current.forward - previousInCurrent.forward) / seconds,
            lateral = (current.lateral - previousInCurrent.lateral) / seconds
        )
        return UstrfEgoCompensatedMotionResolution.Available(
            evidence = UstrfMotionGridEvidence(
                sourceFrame = pair.currentFrame,
                coordinate = coordinate,
                motion = UstrfRelativeMotionEvidence(
                    relativePositionMeters = current,
                    relativeVelocityMetersPerSecond = relativeVelocity,
                    observedAtNs = pair.currentFrame.capturedAtNs,
                    validUntilNs = pair.validUntilNs,
                    source = "${pair.source}:ego-compensated:${pair.trackId}"
                ),
                gridSpec = gridSpec
            ),
            egoCompensatedPreviousPositionMeters = previousInCurrent
        )
    }

    private fun transformPriorPosition(point: UstrfVector2, delta: UstrfVerifiedPoseDelta): UstrfVector2 {
        val forward = point.forward - delta.forwardMeters
        val lateral = point.lateral - delta.lateralMeters
        val cosYaw = cos(delta.yawRadians)
        val sinYaw = sin(delta.yawRadians)
        return UstrfVector2(
            forward = cosYaw * forward + sinYaw * lateral,
            lateral = -sinYaw * forward + cosYaw * lateral
        )
    }

    private fun unavailable(failure: UstrfEgoCompensatedMotionFailure) = UstrfEgoCompensatedMotionResolution.Unavailable(failure)
}
