package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfMetricGeometryReceiptTest {
    private val frame = UstrfFrameStamp(11L, 1_000L, "camera-v2")
    private val promoter = UstrfMetricGeometryReceiptPromoter()

    @Test
    fun fullyVerifiedFreshInputsPrepareProjectionButDoNotCreateGeometry() {
        val admission = promoter.admit(rawDepth(), capture(), intrinsics(), registration(), extrinsics(), calibration(), 1_100L)
        val available = admission as UstrfMetricGeometryProjectionAdmission.Available
        assertEquals(frame, available.sourceFrame)
        assertEquals("body-v2", available.bodyFrame)
        assertEquals(1_500L, available.validUntilNs)
        assertEquals("mount-cal-v1", available.calibrationId)
        assertEquals("a".repeat(64), available.calibrationSourceArtifactSha256)
    }

    @Test
    fun reprojectedDepthAndMismatchedConfidenceAreRejectedIndependently() {
        val reprojected = promoter.admit(rawDepth(depthTimestampNs = 999L), capture(), intrinsics(), registration(), extrinsics(), calibration(), 1_100L)
        assertEquals(UstrfMetricGeometryProjectionAdmission.Unavailable(UstrfMetricGeometryAdmissionFailure.RAW_DEPTH_TIMESTAMP_MISMATCH), reprojected)

        val confidenceMismatch = promoter.admit(rawDepth(confidenceTimestampNs = 999L), capture(), intrinsics(), registration(), extrinsics(), calibration(), 1_100L)
        assertEquals(UstrfMetricGeometryProjectionAdmission.Unavailable(UstrfMetricGeometryAdmissionFailure.CONFIDENCE_TIMESTAMP_MISMATCH), confidenceMismatch)
    }

    @Test
    fun unverifiedCalibrationOrRegistrationNeverAuthorizesMetricProjection() {
        val intrinsicsUnverified = promoter.admit(rawDepth(), capture(), intrinsics(independentlyVerified = false), registration(), extrinsics(), calibration(), 1_100L)
        assertEquals(UstrfMetricGeometryProjectionAdmission.Unavailable(UstrfMetricGeometryAdmissionFailure.INTRINSICS_NOT_INDEPENDENTLY_VERIFIED), intrinsicsUnverified)

        val registrationUnverified = promoter.admit(rawDepth(), capture(), intrinsics(), registration(independentlyVerified = false), extrinsics(), calibration(), 1_100L)
        assertEquals(UstrfMetricGeometryProjectionAdmission.Unavailable(UstrfMetricGeometryAdmissionFailure.REGISTRATION_NOT_INDEPENDENTLY_VERIFIED), registrationUnverified)
    }

    @Test
    fun versionFrameAndFullExtrinsicsFailuresFailClosed() {
        val wrongVersion = promoter.admit(rawDepth(), capture(), intrinsics(calibrationVersion = "other"), registration(), extrinsics(), calibration(), 1_100L)
        assertEquals(UstrfMetricGeometryProjectionAdmission.Unavailable(UstrfMetricGeometryAdmissionFailure.CALIBRATION_VERSION_MISMATCH), wrongVersion)

        val wrongDepthFrame = promoter.admit(rawDepth(), capture(), intrinsics(), registration(depthCoordinateFrame = "rotated-depth"), extrinsics(), calibration(), 1_100L)
        assertEquals(UstrfMetricGeometryProjectionAdmission.Unavailable(UstrfMetricGeometryAdmissionFailure.REGISTRATION_DEPTH_FRAME_MISMATCH), wrongDepthFrame)

        val unverifiedExtrinsics = promoter.admit(rawDepth(), capture(), intrinsics(), registration(), extrinsics(independentlyVerified = false), calibration(), 1_100L)
        assertEquals(UstrfMetricGeometryProjectionAdmission.Unavailable(UstrfMetricGeometryAdmissionFailure.EXTRINSICS_NOT_INDEPENDENTLY_VERIFIED), unverifiedExtrinsics)
    }

    @Test
    fun unavailableOrMismatchedIndependentCalibrationFailsBeforeProjection() {
        val unavailable = promoter.admit(
            rawDepth(), capture(), intrinsics(), registration(), extrinsics(),
            calibration(aiConsensusApproved = false), 1_100L
        )
        assertEquals(
            UstrfMetricGeometryProjectionAdmission.Unavailable(UstrfMetricGeometryAdmissionFailure.CALIBRATION_EVIDENCE_UNAVAILABLE),
            unavailable
        )

        val wrongBody = promoter.admit(
            rawDepth(), capture(), intrinsics(), registration(), extrinsics(),
            calibration(bodyFrame = "other-body"), 1_100L
        )
        assertEquals(
            UstrfMetricGeometryProjectionAdmission.Unavailable(UstrfMetricGeometryAdmissionFailure.CALIBRATION_BODY_FRAME_MISMATCH),
            wrongBody
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun fullExtrinsicsRejectsNonNormalizedQuaternion() {
        extrinsics(quaternion = floatArrayOf(0f, 0f, 0f, .5f))
    }

    private fun capture() = UstrfCaptureReceipt(
        frame = frame,
        hardwareTimestampNs = 900L,
        receivedAtNs = 1_050L,
        cameraClockDomain = "camera-clock-v1",
        calibrationVersion = "camera-cal-v2"
    )

    private fun rawDepth(
        depthTimestampNs: Long = frame.capturedAtNs,
        confidenceTimestampNs: Long = frame.capturedAtNs
    ) = UstrfRawDepthCandidateReceipt(
        sourceFrame = frame,
        depthTimestampNs = depthTimestampNs,
        confidenceTimestampNs = confidenceTimestampNs,
        depthCoordinateFrame = "arcore-raw-depth-native-v1",
        unit = UstrfRawDepthUnit.MILLIMETERS,
        validUntilNs = 1_500L
    )

    private fun intrinsics(
        calibrationVersion: String = "camera-cal-v2",
        independentlyVerified: Boolean = true
    ) = UstrfCameraIntrinsicsReceipt(
        cameraFrame = "camera-v2",
        calibrationVersion = calibrationVersion,
        imageWidthPx = 640,
        imageHeightPx = 480,
        focalXpx = 500f,
        focalYpx = 500f,
        principalXpx = 320f,
        principalYpx = 240f,
        verifiedAtNs = 900L,
        validUntilNs = 2_000L,
        confidence = .99f,
        independentlyVerified = independentlyVerified
    )

    private fun registration(
        depthCoordinateFrame: String = "arcore-raw-depth-native-v1",
        independentlyVerified: Boolean = true
    ) = UstrfDepthCameraRegistrationReceipt(
        depthCoordinateFrame = depthCoordinateFrame,
        cameraFrame = "camera-v2",
        calibrationVersion = "camera-cal-v2",
        transformId = "arcore-depth-to-image-v1",
        verifiedAtNs = 900L,
        validUntilNs = 2_000L,
        confidence = .99f,
        independentlyVerified = independentlyVerified
    )

    private fun extrinsics(
        independentlyVerified: Boolean = true,
        quaternion: FloatArray = floatArrayOf(0f, 0f, 0f, 1f)
    ) = UstrfCameraBodyFullExtrinsicsReceipt(
        cameraFrame = "camera-v2",
        bodyFrame = "body-v2",
        cameraToBodyTranslationM = UstrfVector3(0f, .02f, .04f),
        cameraToBodyQuaternionXyzw = quaternion,
        calibrationId = "mount-cal-v1",
        verifiedAtNs = 900L,
        validUntilNs = 1_500L,
        confidence = .99f,
        independentlyVerified = independentlyVerified
    )

    private fun calibration(
        bodyFrame: String = "body-v2",
        aiConsensusApproved: Boolean = true
    ): UstrfCalibrationEvidenceAdmission = UstrfAiReviewedCalibrationEvidenceVerifier().admit(
        UstrfCalibrationTrialEvidence(
            calibrationId = "mount-cal-v1",
            cameraFrame = "camera-v2",
            bodyFrame = bodyFrame,
            cameraCalibrationVersion = "camera-cal-v2",
            sourceArtifactSha256 = "a".repeat(64),
            collectorId = "collector",
            aiReviewReceiptSha256 = "b".repeat(64),
            aiConsensusApproved = aiConsensusApproved,
            sampleCount = 30,
            poseCoverageBins = 5,
            intrinsicsP95ReprojectionPx = 1f,
            depthRegistrationP95ErrorM = .02f,
            mountTranslationRepeatabilityM = .005f,
            mountRotationRepeatabilityDeg = .5f,
            collectedAtNs = 800L,
            validUntilNs = 1_800L
        ),
        decisionAtNs = 1_100L
    )
}
