package com.linnan.blindassist.session

import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp

/**
 * Frozen input to the diagnostic-only JRDB LiDAR replay source.
 *
 * Target identity and the 2D region are source annotations. The signed rate is
 * derived from real LiDAR points associated through those annotations, so this
 * is not an independent perception source and cannot authorize a live mode.
 */
data class JrdbAnnotationConditionedLidarReplayInput(
    val previousFrame: FrameStamp,
    val currentFrame: FrameStamp,
    val availableAtNs: Long,
    val validUntilNs: Long,
    val trackEpoch: String,
    val target: Detection,
    val signedApproachRatePerS: Float,
    val quality: Float
)

sealed interface DualLoopJrdbReplayProposal {
    data class Available(val evidence: DualLoopGeometryEvidence) : DualLoopJrdbReplayProposal
    data class Abstained(val reason: String) : DualLoopJrdbReplayProposal
}

/**
 * Stable adapter from the frozen JRDB replay row into the production shadow
 * contract. It fails closed on any live/provenance/time ambiguity.
 */
object DualLoopJrdbReplayAdapter {
    const val SOURCE_ID = "JRDB_ANNOTATION_CONDITIONED_LIDAR_CENTROID_REPLAY_V1"
    const val COORDINATE_FRAME = "jrdb-logical-rgb360"

    fun adapt(input: JrdbAnnotationConditionedLidarReplayInput): DualLoopJrdbReplayProposal {
        val previous = input.previousFrame
        val current = input.currentFrame
        val reason = when {
            previous.clockDomain != FrameClockDomain.REPLAY_TIMELINE ||
                current.clockDomain != FrameClockDomain.REPLAY_TIMELINE ->
                "REPLAY_TIMELINE_REQUIRED"
            previous.sourceId != current.sourceId ||
                previous.coordinateFrame != COORDINATE_FRAME ||
                current.coordinateFrame != COORDINATE_FRAME ->
                "FRAME_IDENTITY_MISMATCH"
            previous.frameId >= current.frameId ||
                previous.capturedAtNs >= current.capturedAtNs ->
                "NON_MONOTONIC_FRAME_PAIR"
            input.availableAtNs < current.capturedAtNs ||
                input.validUntilNs < input.availableAtNs ->
                "INVALID_AVAILABILITY_WINDOW"
            input.trackEpoch.isBlank() -> "TRACK_EPOCH_MISSING"
            input.target.source != DetectionSource.OBJECT_DETECTOR ->
                "OBJECT_DETECTOR_BEHAVIOR_REQUIRED"
            !input.signedApproachRatePerS.isFinite() -> "NONFINITE_RATE"
            !input.quality.isFinite() || input.quality !in 0f..1f -> "INVALID_QUALITY"
            else -> null
        }
        if (reason != null) return DualLoopJrdbReplayProposal.Abstained(reason)
        return DualLoopJrdbReplayProposal.Available(
            DualLoopGeometryEvidence(
                sourceContractId = DualLoopShadowAdmitter.CONTRACT_ID,
                sourceId = SOURCE_ID,
                previousFrame = previous,
                currentFrame = current,
                availableAtNs = input.availableAtNs,
                validUntilNs = input.validUntilNs,
                availabilityClockDomain = FrameClockDomain.REPLAY_TIMELINE,
                trackEpoch = input.trackEpoch,
                targetClassId = input.target.classId,
                targetLabel = input.target.label,
                targetBoundingBox = input.target.boundingBox,
                targetFrameSize = input.target.frameSize,
                targetSource = input.target.source,
                signedApproachRatePerS = input.signedApproachRatePerS,
                quality = input.quality,
                targetProvenance = DualLoopTargetProvenance.REPLAY_ANNOTATION
            )
        )
    }
}
