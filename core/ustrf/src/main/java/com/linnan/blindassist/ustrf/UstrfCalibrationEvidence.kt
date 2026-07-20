package com.linnan.blindassist.ustrf

/**
 * Summary of an off-device calibration trial. The referenced source artifact stays outside the
 * repository; only its immutable SHA-256 and aggregate error metrics belong in an experiment
 * receipt. This model never treats a self-authored manifest as independent verification.
 */
data class UstrfCalibrationTrialEvidence(
    val calibrationId: String,
    val cameraFrame: String,
    val bodyFrame: String,
    val cameraCalibrationVersion: String,
    val sourceArtifactSha256: String,
    val collectorId: String,
    val reviewerId: String,
    val independentReviewApproved: Boolean,
    val sampleCount: Int,
    val poseCoverageBins: Int,
    val intrinsicsP95ReprojectionPx: Float,
    val depthRegistrationP95ErrorM: Float,
    val mountTranslationRepeatabilityM: Float,
    val mountRotationRepeatabilityDeg: Float,
    val collectedAtNs: Long,
    val validUntilNs: Long
) {
    init {
        require(calibrationId.isNotBlank() && cameraFrame.isNotBlank() && bodyFrame.isNotBlank())
        require(cameraCalibrationVersion.isNotBlank())
        require(sourceArtifactSha256.matches(Regex("[0-9a-fA-F]{64}")))
        require(collectorId.isNotBlank() && reviewerId.isNotBlank())
        require(sampleCount >= 0 && poseCoverageBins >= 0)
        require(intrinsicsP95ReprojectionPx.isFinite() && intrinsicsP95ReprojectionPx >= 0f)
        require(depthRegistrationP95ErrorM.isFinite() && depthRegistrationP95ErrorM >= 0f)
        require(mountTranslationRepeatabilityM.isFinite() && mountTranslationRepeatabilityM >= 0f)
        require(mountRotationRepeatabilityDeg.isFinite() && mountRotationRepeatabilityDeg >= 0f)
        require(collectedAtNs >= 0L && validUntilNs >= collectedAtNs)
    }
}

enum class UstrfCalibrationEvidenceFailure {
    EVIDENCE_FROM_FUTURE,
    EVIDENCE_STALE,
    INDEPENDENT_REVIEW_NOT_APPROVED,
    REVIEWER_NOT_INDEPENDENT,
    SAMPLE_COUNT_INSUFFICIENT,
    POSE_COVERAGE_INSUFFICIENT,
    INTRINSICS_REPROJECTION_TOO_LARGE,
    DEPTH_REGISTRATION_ERROR_TOO_LARGE,
    MOUNT_TRANSLATION_REPEATABILITY_TOO_LARGE,
    MOUNT_ROTATION_REPEATABILITY_TOO_LARGE
}

sealed interface UstrfCalibrationEvidenceAdmission {
    /**
     * This approves a particular evidence manifest for experimental calibration use. It does not
     * authorize safety geometry, production feedback, or any new user-facing motion command.
     */
    @ConsistentCopyVisibility
    data class Available internal constructor(
        val calibrationId: String,
        val cameraFrame: String,
        val bodyFrame: String,
        val cameraCalibrationVersion: String,
        val sourceArtifactSha256: String,
        val validUntilNs: Long
    ) : UstrfCalibrationEvidenceAdmission

    data class Unavailable(val failure: UstrfCalibrationEvidenceFailure) : UstrfCalibrationEvidenceAdmission
}

/**
 * Deterministic admission for independently reviewed calibration evidence. The thresholds are
 * experiment starting gates, not a claim of clinical or production safety adequacy.
 */
class UstrfIndependentCalibrationEvidenceVerifier(
    private val minimumSamples: Int = 30,
    private val minimumPoseCoverageBins: Int = 5,
    private val maximumIntrinsicsP95ReprojectionPx: Float = 1.5f,
    private val maximumDepthRegistrationP95ErrorM: Float = .03f,
    private val maximumMountTranslationRepeatabilityM: Float = .01f,
    private val maximumMountRotationRepeatabilityDeg: Float = 1f
) {
    init {
        require(minimumSamples > 0 && minimumPoseCoverageBins > 0)
        require(maximumIntrinsicsP95ReprojectionPx > 0f && maximumDepthRegistrationP95ErrorM > 0f)
        require(maximumMountTranslationRepeatabilityM > 0f && maximumMountRotationRepeatabilityDeg > 0f)
    }

    fun admit(
        evidence: UstrfCalibrationTrialEvidence,
        decisionAtNs: Long
    ): UstrfCalibrationEvidenceAdmission {
        if (decisionAtNs < evidence.collectedAtNs) return unavailable(UstrfCalibrationEvidenceFailure.EVIDENCE_FROM_FUTURE)
        if (decisionAtNs > evidence.validUntilNs) return unavailable(UstrfCalibrationEvidenceFailure.EVIDENCE_STALE)
        if (!evidence.independentReviewApproved) return unavailable(UstrfCalibrationEvidenceFailure.INDEPENDENT_REVIEW_NOT_APPROVED)
        if (evidence.collectorId == evidence.reviewerId) return unavailable(UstrfCalibrationEvidenceFailure.REVIEWER_NOT_INDEPENDENT)
        if (evidence.sampleCount < minimumSamples) return unavailable(UstrfCalibrationEvidenceFailure.SAMPLE_COUNT_INSUFFICIENT)
        if (evidence.poseCoverageBins < minimumPoseCoverageBins) return unavailable(UstrfCalibrationEvidenceFailure.POSE_COVERAGE_INSUFFICIENT)
        if (evidence.intrinsicsP95ReprojectionPx > maximumIntrinsicsP95ReprojectionPx) return unavailable(UstrfCalibrationEvidenceFailure.INTRINSICS_REPROJECTION_TOO_LARGE)
        if (evidence.depthRegistrationP95ErrorM > maximumDepthRegistrationP95ErrorM) return unavailable(UstrfCalibrationEvidenceFailure.DEPTH_REGISTRATION_ERROR_TOO_LARGE)
        if (evidence.mountTranslationRepeatabilityM > maximumMountTranslationRepeatabilityM) return unavailable(UstrfCalibrationEvidenceFailure.MOUNT_TRANSLATION_REPEATABILITY_TOO_LARGE)
        if (evidence.mountRotationRepeatabilityDeg > maximumMountRotationRepeatabilityDeg) return unavailable(UstrfCalibrationEvidenceFailure.MOUNT_ROTATION_REPEATABILITY_TOO_LARGE)
        return UstrfCalibrationEvidenceAdmission.Available(
            calibrationId = evidence.calibrationId,
            cameraFrame = evidence.cameraFrame,
            bodyFrame = evidence.bodyFrame,
            cameraCalibrationVersion = evidence.cameraCalibrationVersion,
            sourceArtifactSha256 = evidence.sourceArtifactSha256.lowercase(),
            validUntilNs = evidence.validUntilNs
        )
    }

    private fun unavailable(failure: UstrfCalibrationEvidenceFailure) = UstrfCalibrationEvidenceAdmission.Unavailable(failure)
}
