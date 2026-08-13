package com.linnan.blindassist.ustrf

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sqrt

/** A pose-bound buffered frame. Pixel/depth payloads are deliberately absent from this contract. */
data class TaroBufferedPoseFrame(
    val frame: UstrfFrameStamp,
    val pose: UstrfPoseSample
)

enum class TaroPoseDiverseSelectionFailure {
    DISABLED,
    REFERENCE_POSE_INVALID,
    NO_ELIGIBLE_BUFFERED_FRAME
}

sealed interface TaroPoseDiverseSelection {
    data class Available(
        val referenceFrame: UstrfFrameStamp,
        val selectedFrame: UstrfFrameStamp,
        val gapNs: Long,
        val translationM: Float,
        val yawDeltaRad: Float,
        val policyId: String = TaroPoseDiverseFrameSelector.POLICY_ID
    ) : TaroPoseDiverseSelection

    data class Unavailable(val failure: TaroPoseDiverseSelectionFailure) : TaroPoseDiverseSelection
}

/**
 * Experimental, default-off TARO frame selector.
 *
 * It mirrors the frozen source-time generic arm used by the research replay: among admissible
 * historical frames in the 150 ms..1 s window, maximize camera translation, then yaw diversity,
 * then prefer the smaller time gap and deterministic frame id. It cannot enable itself, decode a
 * payload, fuse risk, emit guidance, or change the default app.
 */
class TaroPoseDiverseFrameSelector(
    private val enabled: Boolean = false,
    private val minimumGapNs: Long = 150_000_000L,
    private val maximumGapNs: Long = 1_000_000_000L,
    private val minimumConfidence: Float = .70f
) {
    init {
        require(minimumGapNs > 0L && maximumGapNs >= minimumGapNs)
        require(minimumConfidence in 0f..1f)
    }

    fun select(
        referenceFrame: UstrfFrameStamp,
        referencePose: UstrfPoseSample,
        bufferedFrames: List<TaroBufferedPoseFrame>
    ): TaroPoseDiverseSelection {
        if (!enabled) return unavailable(TaroPoseDiverseSelectionFailure.DISABLED)
        if (!poseMatches(referenceFrame, referencePose) || referencePose.validUntilNs < referenceFrame.capturedAtNs) {
            return unavailable(TaroPoseDiverseSelectionFailure.REFERENCE_POSE_INVALID)
        }
        val candidates = bufferedFrames.mapNotNull { candidate ->
            val gapNs = referenceFrame.capturedAtNs - candidate.frame.capturedAtNs
            if (gapNs !in minimumGapNs..maximumGapNs) return@mapNotNull null
            if (!poseMatches(candidate.frame, candidate.pose)) return@mapNotNull null
            if (candidate.pose.worldFrame != referencePose.worldFrame || candidate.pose.cameraFrame != referencePose.cameraFrame) {
                return@mapNotNull null
            }
            val translationM = distance(referencePose.worldCameraTranslationM, candidate.pose.worldCameraTranslationM)
                ?: return@mapNotNull null
            val yawDeltaRad = angleDistance(referencePose.yawRad, candidate.pose.yawRad)
                ?: return@mapNotNull null
            Candidate(candidate.frame, gapNs, translationM, yawDeltaRad)
        }
        val selected = candidates.maxWithOrNull(
            compareBy<Candidate> { it.translationM }
                .thenBy { it.yawDeltaRad }
                .thenByDescending { it.gapNs }
                .thenBy { it.frame.frameId }
        ) ?: return unavailable(TaroPoseDiverseSelectionFailure.NO_ELIGIBLE_BUFFERED_FRAME)
        return TaroPoseDiverseSelection.Available(
            referenceFrame = referenceFrame,
            selectedFrame = selected.frame,
            gapNs = selected.gapNs,
            translationM = selected.translationM,
            yawDeltaRad = selected.yawDeltaRad
        )
    }

    private fun poseMatches(frame: UstrfFrameStamp, pose: UstrfPoseSample): Boolean =
        pose.timestampNs == frame.capturedAtNs &&
            pose.cameraFrame == frame.coordinateFrame &&
            pose.tracking == UstrfPoseState.TRACKING &&
            pose.confidence >= minimumConfidence &&
            finite(pose.worldCameraTranslationM) &&
            pose.yawRad.isFinite()

    private fun distance(first: UstrfVector3, second: UstrfVector3): Float? {
        val dx = first.x - second.x
        val dy = first.y - second.y
        val dz = first.z - second.z
        val value = sqrt((dx * dx + dy * dy + dz * dz).toDouble()).toFloat()
        return value.takeIf { it.isFinite() }
    }

    private fun angleDistance(first: Float, second: Float): Float? {
        if (!first.isFinite() || !second.isFinite()) return null
        val fullTurn = (2.0 * PI).toFloat()
        var delta = abs(first - second) % fullTurn
        if (delta > PI.toFloat()) delta = fullTurn - delta
        return delta
    }

    private fun finite(value: UstrfVector3): Boolean = value.x.isFinite() && value.y.isFinite() && value.z.isFinite()

    private fun unavailable(failure: TaroPoseDiverseSelectionFailure) = TaroPoseDiverseSelection.Unavailable(failure)

    private data class Candidate(
        val frame: UstrfFrameStamp,
        val gapNs: Long,
        val translationM: Float,
        val yawDeltaRad: Float
    )

    companion object {
        const val POLICY_ID = "TARO_POSE_DIVERSE_GENERIC_R0"
    }
}
