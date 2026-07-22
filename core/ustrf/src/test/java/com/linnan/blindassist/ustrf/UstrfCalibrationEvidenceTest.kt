package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Test

class UstrfCalibrationEvidenceTest {
    private val verifier = UstrfAiReviewedCalibrationEvidenceVerifier()

    @Test
    fun independentlyReviewedEvidenceWithinAllStartingGatesIsAdmitted() {
        val admission = verifier.admit(evidence(), 1_100L) as UstrfCalibrationEvidenceAdmission.Available
        assertEquals("mount-cal-r1", admission.calibrationId)
        assertEquals("camera-v1", admission.cameraFrame)
        assertEquals("body-v1", admission.bodyFrame)
        assertEquals(1_500L, admission.validUntilNs)
    }

    @Test
    fun missingAiConsensusNeverCountsAsReviewed() {
        val missingApproval = verifier.admit(evidence(aiConsensusApproved = false), 1_100L)
        assertEquals(UstrfCalibrationEvidenceAdmission.Unavailable(UstrfCalibrationEvidenceFailure.AI_CONSENSUS_NOT_APPROVED), missingApproval)
    }

    @Test
    fun insufficientCoverageAndEachPhysicalErrorGateFailClosed() {
        assertFailure(evidence(sampleCount = 29), UstrfCalibrationEvidenceFailure.SAMPLE_COUNT_INSUFFICIENT)
        assertFailure(evidence(poseCoverageBins = 4), UstrfCalibrationEvidenceFailure.POSE_COVERAGE_INSUFFICIENT)
        assertFailure(evidence(intrinsicsP95ReprojectionPx = 1.51f), UstrfCalibrationEvidenceFailure.INTRINSICS_REPROJECTION_TOO_LARGE)
        assertFailure(evidence(depthRegistrationP95ErrorM = .031f), UstrfCalibrationEvidenceFailure.DEPTH_REGISTRATION_ERROR_TOO_LARGE)
        assertFailure(evidence(mountTranslationRepeatabilityM = .011f), UstrfCalibrationEvidenceFailure.MOUNT_TRANSLATION_REPEATABILITY_TOO_LARGE)
        assertFailure(evidence(mountRotationRepeatabilityDeg = 1.01f), UstrfCalibrationEvidenceFailure.MOUNT_ROTATION_REPEATABILITY_TOO_LARGE)
    }

    @Test
    fun futureAndExpiredEvidenceCannotBeReused() {
        assertFailure(evidence(collectedAtNs = 1_200L), UstrfCalibrationEvidenceFailure.EVIDENCE_FROM_FUTURE, decisionAtNs = 1_100L)
        assertFailure(evidence(validUntilNs = 1_050L), UstrfCalibrationEvidenceFailure.EVIDENCE_STALE, decisionAtNs = 1_100L)
    }

    private fun assertFailure(
        evidence: UstrfCalibrationTrialEvidence,
        expected: UstrfCalibrationEvidenceFailure,
        decisionAtNs: Long = 1_100L
    ) = assertEquals(UstrfCalibrationEvidenceAdmission.Unavailable(expected), verifier.admit(evidence, decisionAtNs))

    private fun evidence(
        aiConsensusApproved: Boolean = true,
        sampleCount: Int = 30,
        poseCoverageBins: Int = 5,
        intrinsicsP95ReprojectionPx: Float = 1.5f,
        depthRegistrationP95ErrorM: Float = .03f,
        mountTranslationRepeatabilityM: Float = .01f,
        mountRotationRepeatabilityDeg: Float = 1f,
        collectedAtNs: Long = 1_000L,
        validUntilNs: Long = 1_500L
    ) = UstrfCalibrationTrialEvidence(
        calibrationId = "mount-cal-r1",
        cameraFrame = "camera-v1",
        bodyFrame = "body-v1",
        cameraCalibrationVersion = "camera-cal-v1",
        sourceArtifactSha256 = "a".repeat(64),
        collectorId = "collector-a",
        aiReviewReceiptSha256 = "b".repeat(64),
        aiConsensusApproved = aiConsensusApproved,
        sampleCount = sampleCount,
        poseCoverageBins = poseCoverageBins,
        intrinsicsP95ReprojectionPx = intrinsicsP95ReprojectionPx,
        depthRegistrationP95ErrorM = depthRegistrationP95ErrorM,
        mountTranslationRepeatabilityM = mountTranslationRepeatabilityM,
        mountRotationRepeatabilityDeg = mountRotationRepeatabilityDeg,
        collectedAtNs = collectedAtNs,
        validUntilNs = validUntilNs
    )
}
