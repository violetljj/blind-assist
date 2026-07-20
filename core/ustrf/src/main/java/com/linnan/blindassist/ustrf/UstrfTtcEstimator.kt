package com.linnan.blindassist.ustrf

import kotlin.math.max
import kotlin.math.sqrt

data class UstrfVector2(val forward: Float, val lateral: Float) {
    fun dot(other: UstrfVector2): Float = forward * other.forward + lateral * other.lateral
    operator fun plus(other: UstrfVector2): UstrfVector2 = UstrfVector2(forward + other.forward, lateral + other.lateral)
    operator fun times(scale: Float): UstrfVector2 = UstrfVector2(forward * scale, lateral * scale)
    fun norm(): Float = sqrt(dot(this))
}

data class UstrfRelativeMotionEvidence(
    val relativePositionMeters: UstrfVector2,
    val relativeVelocityMetersPerSecond: UstrfVector2,
    val observedAtNs: Long,
    val validUntilNs: Long,
    val source: String
) {
    init {
        require(observedAtNs >= 0L && validUntilNs >= observedAtNs)
        require(source.isNotBlank())
    }
}

data class UstrfTtcEstimate(val timeToClosestApproachMs: Long, val closestDistanceMeters: Float)

/** Category-agnostic relative-motion geometry. Ego-motion compensation belongs to the upstream Adapter. */
class UstrfTtcEstimator(
    private val horizonMs: Long = 3_000L,
    private val velocityEpsilon: Float = 0.0001f
) {
    init {
        require(horizonMs > 0L && velocityEpsilon > 0f)
    }

    fun estimate(evidence: UstrfRelativeMotionEvidence, nowNs: Long): UstrfTtcEstimate? {
        if (nowNs < evidence.observedAtNs || evidence.validUntilNs < nowNs) return null
        val velocityNormSquared = evidence.relativeVelocityMetersPerSecond.dot(evidence.relativeVelocityMetersPerSecond)
        if (velocityNormSquared < velocityEpsilon) return null
        val elapsedSeconds = (nowNs - evidence.observedAtNs).toFloat() / 1_000_000_000f
        val positionNow = evidence.relativePositionMeters + evidence.relativeVelocityMetersPerSecond * elapsedSeconds
        val unconstrainedSeconds = -positionNow.dot(evidence.relativeVelocityMetersPerSecond) / velocityNormSquared
        val seconds = unconstrainedSeconds.coerceIn(0f, horizonMs / 1_000f)
        val closest = positionNow + evidence.relativeVelocityMetersPerSecond * seconds
        return UstrfTtcEstimate(max(0L, (seconds * 1_000f).toLong()), closest.norm())
    }
}
