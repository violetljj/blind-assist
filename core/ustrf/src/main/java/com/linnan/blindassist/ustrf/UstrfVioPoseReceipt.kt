package com.linnan.blindassist.ustrf

/**
 * A VIO candidate is intentionally distinct from a [UstrfPoseSample]. It only becomes admissible
 * after the image frame, freshness, tracking quality, inter-frame world stability,
 * gravity-planar assumption, and a separately verified camera-to-body calibration receipt are all
 * present.
 */
enum class UstrfWorldFrameStability { EPHEMERAL_PER_FRAME, INTER_FRAME_STABLE }

data class UstrfVioPoseCandidate(
    val sourceFrame: UstrfFrameStamp,
    val worldFrame: String,
    val worldFrameStability: UstrfWorldFrameStability,
    val worldCameraTranslationM: UstrfVector3,
    val yawRad: Float,
    val rollRad: Float,
    val pitchRad: Float,
    val tracking: UstrfPoseState,
    val confidence: Float,
    val validUntilNs: Long,
    val source: String
) {
    init {
        require(worldFrame.isNotBlank() && source.isNotBlank())
        require(confidence in 0f..1f)
        require(validUntilNs >= sourceFrame.capturedAtNs)
    }
}

/**
 * An auditable calibration receipt. The current planar promoter verifies but deliberately does not
 * apply this SE(3) transform: risk-field world warping requires a later full-3D contract.
 */
data class UstrfCameraBodyExtrinsicsReceipt(
    val cameraFrame: String,
    val bodyFrame: String,
    val cameraToBodyTranslationM: UstrfVector3,
    val cameraToBodyYawRad: Float,
    val calibrationId: String,
    val verifiedAtNs: Long,
    val validUntilNs: Long,
    val confidence: Float,
    val independentlyVerified: Boolean
) {
    init {
        require(cameraFrame.isNotBlank() && bodyFrame.isNotBlank() && calibrationId.isNotBlank())
        require(verifiedAtNs >= 0L && validUntilNs >= verifiedAtNs)
        require(confidence in 0f..1f)
    }
}

enum class UstrfVioPoseAdmissionFailure {
    SOURCE_FRAME_MISMATCH,
    CANDIDATE_NOT_TRACKING,
    WORLD_FRAME_NOT_INTERFRAME_STABLE,
    CANDIDATE_STALE,
    CANDIDATE_LOW_CONFIDENCE,
    TILT_OUTSIDE_PLANAR_ASSUMPTION,
    EXTRINSICS_CAMERA_FRAME_MISMATCH,
    EXTRINSICS_NOT_INDEPENDENTLY_VERIFIED,
    EXTRINSICS_STALE,
    EXTRINSICS_LOW_CONFIDENCE
}

sealed interface UstrfVioPoseAdmission {
    data class Available(
        val cameraPose: UstrfPoseSample,
        /** Calibration was checked but not applied; no body-frame/world warp is authorized here. */
        val verifiedBodyFrame: String
    ) : UstrfVioPoseAdmission

    data class Unavailable(val failure: UstrfVioPoseAdmissionFailure) : UstrfVioPoseAdmission
}

/**
 * A fail-closed bridge for a future ARCore/VIO Adapter. Its only output is a camera-frame pose
 * receipt suitable for [UstrfPoseBuffer]; it cannot authorize a camera-to-body transform or a
 * risk-field warp. For example, an ARCore frame pose is EPHEMERAL_PER_FRAME until a separately
 * measured, inter-frame-stable transform/anchor policy is admitted.
 */
class UstrfVioPoseReceiptPromoter(
    private val minimumCandidateConfidence: Float = .85f,
    private val minimumExtrinsicsConfidence: Float = .95f,
    private val maximumPlanarTiltRad: Float = .174533f
) {
    init {
        require(minimumCandidateConfidence in 0f..1f)
        require(minimumExtrinsicsConfidence in 0f..1f)
        require(maximumPlanarTiltRad in 0f..1.5707964f)
    }

    fun admit(
        candidate: UstrfVioPoseCandidate,
        captureFrame: UstrfFrameStamp,
        decisionAtNs: Long,
        extrinsics: UstrfCameraBodyExtrinsicsReceipt
    ): UstrfVioPoseAdmission {
        if (candidate.sourceFrame != captureFrame) return unavailable(UstrfVioPoseAdmissionFailure.SOURCE_FRAME_MISMATCH)
        if (candidate.tracking != UstrfPoseState.TRACKING) return unavailable(UstrfVioPoseAdmissionFailure.CANDIDATE_NOT_TRACKING)
        if (candidate.worldFrameStability != UstrfWorldFrameStability.INTER_FRAME_STABLE) {
            return unavailable(UstrfVioPoseAdmissionFailure.WORLD_FRAME_NOT_INTERFRAME_STABLE)
        }
        if (decisionAtNs > candidate.validUntilNs) return unavailable(UstrfVioPoseAdmissionFailure.CANDIDATE_STALE)
        if (candidate.confidence < minimumCandidateConfidence) return unavailable(UstrfVioPoseAdmissionFailure.CANDIDATE_LOW_CONFIDENCE)
        if (kotlin.math.abs(candidate.rollRad) > maximumPlanarTiltRad || kotlin.math.abs(candidate.pitchRad) > maximumPlanarTiltRad) {
            return unavailable(UstrfVioPoseAdmissionFailure.TILT_OUTSIDE_PLANAR_ASSUMPTION)
        }
        if (extrinsics.cameraFrame != captureFrame.coordinateFrame) return unavailable(UstrfVioPoseAdmissionFailure.EXTRINSICS_CAMERA_FRAME_MISMATCH)
        if (!extrinsics.independentlyVerified) return unavailable(UstrfVioPoseAdmissionFailure.EXTRINSICS_NOT_INDEPENDENTLY_VERIFIED)
        if (decisionAtNs > extrinsics.validUntilNs) return unavailable(UstrfVioPoseAdmissionFailure.EXTRINSICS_STALE)
        if (extrinsics.confidence < minimumExtrinsicsConfidence) return unavailable(UstrfVioPoseAdmissionFailure.EXTRINSICS_LOW_CONFIDENCE)
        return UstrfVioPoseAdmission.Available(
            cameraPose = UstrfPoseSample(
                timestampNs = captureFrame.capturedAtNs,
                worldFrame = candidate.worldFrame,
                cameraFrame = captureFrame.coordinateFrame,
                worldCameraTranslationM = candidate.worldCameraTranslationM,
                yawRad = candidate.yawRad,
                gravityWorld = UstrfVector3(0f, -9.80665f, 0f),
                tracking = UstrfPoseState.TRACKING,
                confidence = minOf(candidate.confidence, extrinsics.confidence),
                validUntilNs = minOf(candidate.validUntilNs, extrinsics.validUntilNs)
            ),
            verifiedBodyFrame = extrinsics.bodyFrame
        )
    }

    private fun unavailable(failure: UstrfVioPoseAdmissionFailure) = UstrfVioPoseAdmission.Unavailable(failure)
}
