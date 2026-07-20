package com.linnan.blindassist.ustrf

import kotlin.math.PI

data class UstrfVector3(val x: Float, val y: Float, val z: Float) {
    operator fun plus(other: UstrfVector3): UstrfVector3 = UstrfVector3(x + other.x, y + other.y, z + other.z)
    operator fun times(scale: Float): UstrfVector3 = UstrfVector3(x * scale, y * scale, z * scale)
}

/** Pose receipt in the form required before a real ARCore/VIO Adapter may feed the risk field. */
data class UstrfPoseSample(
    val timestampNs: Long,
    val worldFrame: String,
    val cameraFrame: String,
    val worldCameraTranslationM: UstrfVector3,
    val yawRad: Float,
    val gravityWorld: UstrfVector3,
    val tracking: UstrfPoseState,
    val confidence: Float,
    val validUntilNs: Long
) {
    init {
        require(timestampNs >= 0L)
        require(worldFrame.isNotBlank() && cameraFrame.isNotBlank())
        require(confidence in 0f..1f)
        require(validUntilNs >= timestampNs)
    }
}

enum class UstrfPoseLookupFailure {
    NOT_BRACKETED,
    CAMERA_FRAME_MISMATCH,
    GAP_TOO_LARGE,
    STALE,
    NOT_TRACKING,
    LOW_CONFIDENCE
}

sealed interface UstrfPoseLookup {
    data class Available(val pose: UstrfPoseSample) : UstrfPoseLookup
    data class Unavailable(val failure: UstrfPoseLookupFailure) : UstrfPoseLookup
}

class UstrfPoseBuffer(
    private val maxBracketGapNs: Long = 100_000_000L,
    private val minimumConfidence: Float = 0.70f
) {
    private val samples = mutableListOf<UstrfPoseSample>()

    init {
        require(maxBracketGapNs > 0L)
        require(minimumConfidence in 0f..1f)
    }

    fun append(sample: UstrfPoseSample) {
        samples.lastOrNull()?.let { previous ->
            require(sample.timestampNs > previous.timestampNs) { "pose timestamps must be strictly increasing" }
            require(sample.worldFrame == previous.worldFrame && sample.cameraFrame == previous.cameraFrame) {
                "one pose buffer may not silently mix coordinate frames"
            }
        }
        samples += sample
    }

    fun interpolateAt(frame: UstrfFrameStamp): UstrfPoseLookup {
        val before = samples.lastOrNull { it.timestampNs <= frame.capturedAtNs }
            ?: return UstrfPoseLookup.Unavailable(UstrfPoseLookupFailure.NOT_BRACKETED)
        val after = samples.firstOrNull { it.timestampNs >= frame.capturedAtNs }
            ?: return UstrfPoseLookup.Unavailable(UstrfPoseLookupFailure.NOT_BRACKETED)
        if (before.cameraFrame != frame.coordinateFrame || after.cameraFrame != frame.coordinateFrame) {
            return UstrfPoseLookup.Unavailable(UstrfPoseLookupFailure.CAMERA_FRAME_MISMATCH)
        }
        if (after.timestampNs - before.timestampNs > maxBracketGapNs) {
            return UstrfPoseLookup.Unavailable(UstrfPoseLookupFailure.GAP_TOO_LARGE)
        }
        if (before.validUntilNs < frame.capturedAtNs || after.validUntilNs < frame.capturedAtNs) {
            return UstrfPoseLookup.Unavailable(UstrfPoseLookupFailure.STALE)
        }
        if (before.tracking != UstrfPoseState.TRACKING || after.tracking != UstrfPoseState.TRACKING) {
            return UstrfPoseLookup.Unavailable(UstrfPoseLookupFailure.NOT_TRACKING)
        }
        if (before.confidence < minimumConfidence || after.confidence < minimumConfidence) {
            return UstrfPoseLookup.Unavailable(UstrfPoseLookupFailure.LOW_CONFIDENCE)
        }
        val fraction = if (after.timestampNs == before.timestampNs) 0f else
            (frame.capturedAtNs - before.timestampNs).toFloat() / (after.timestampNs - before.timestampNs).toFloat()
        return UstrfPoseLookup.Available(
            before.copy(
                timestampNs = frame.capturedAtNs,
                worldCameraTranslationM = before.worldCameraTranslationM * (1f - fraction) + after.worldCameraTranslationM * fraction,
                yawRad = interpolateAngle(before.yawRad, after.yawRad, fraction),
                gravityWorld = before.gravityWorld * (1f - fraction) + after.gravityWorld * fraction,
                confidence = minOf(before.confidence, after.confidence),
                validUntilNs = minOf(before.validUntilNs, after.validUntilNs)
            )
        )
    }

    private fun interpolateAngle(first: Float, second: Float, fraction: Float): Float {
        val halfTurn = PI.toFloat()
        val fullTurn = 2f * halfTurn
        var delta = (second - first) % fullTurn
        if (delta > halfTurn) delta -= fullTurn
        if (delta < -halfTurn) delta += fullTurn
        return first + delta * fraction
    }
}
