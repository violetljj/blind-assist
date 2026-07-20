package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfVioPoseReceiptTest {
    private val frame = UstrfFrameStamp(7L, 1_000L, "camera-v1")
    private val promoter = UstrfVioPoseReceiptPromoter(maximumPlanarTiltRad = .2f)

    @Test
    fun verifiedFreshTrackingCandidateProducesCameraPoseOnly() {
        val admission = promoter.admit(candidate(), frame, 1_100L, extrinsics()) as UstrfVioPoseAdmission.Available
        assertEquals("camera-v1", admission.cameraPose.cameraFrame)
        assertEquals("body-v1", admission.verifiedBodyFrame)
        assertEquals(UstrfPoseState.TRACKING, admission.cameraPose.tracking)
        assertEquals(.9f, admission.cameraPose.confidence)
    }

    @Test
    fun unverifiedExtrinsicsFailClosedEvenForTrackingCandidate() {
        val admission = promoter.admit(candidate(), frame, 1_100L, extrinsics(independentlyVerified = false))
        assertEquals(
            UstrfVioPoseAdmission.Unavailable(UstrfVioPoseAdmissionFailure.EXTRINSICS_NOT_INDEPENDENTLY_VERIFIED),
            admission
        )
    }

    @Test
    fun wrongImageFrameAndStaleCalibrationAreRejected() {
        val wrongFrame = promoter.admit(candidate(), frame.copy(frameId = 8L), 1_100L, extrinsics())
        assertEquals(UstrfVioPoseAdmission.Unavailable(UstrfVioPoseAdmissionFailure.SOURCE_FRAME_MISMATCH), wrongFrame)

        val stale = promoter.admit(candidate(), frame, 1_500L, extrinsics(validUntilNs = 1_200L))
        assertEquals(UstrfVioPoseAdmission.Unavailable(UstrfVioPoseAdmissionFailure.EXTRINSICS_STALE), stale)
    }

    @Test
    fun tiltAndCameraMismatchAreNeverSilentlyPromoted() {
        val tilted = promoter.admit(candidate(rollRad = .21f), frame, 1_100L, extrinsics())
        assertEquals(UstrfVioPoseAdmission.Unavailable(UstrfVioPoseAdmissionFailure.TILT_OUTSIDE_PLANAR_ASSUMPTION), tilted)

        val mismatch = promoter.admit(candidate(), frame, 1_100L, extrinsics(cameraFrame = "other-camera"))
        assertEquals(UstrfVioPoseAdmission.Unavailable(UstrfVioPoseAdmissionFailure.EXTRINSICS_CAMERA_FRAME_MISMATCH), mismatch)
    }

    @Test
    fun perFrameWorldCoordinatesCannotBecomeAnInterpolableSafetyPose() {
        val admission = promoter.admit(
            candidate(worldFrameStability = UstrfWorldFrameStability.EPHEMERAL_PER_FRAME),
            frame,
            1_100L,
            extrinsics()
        )
        assertEquals(UstrfVioPoseAdmission.Unavailable(UstrfVioPoseAdmissionFailure.WORLD_FRAME_NOT_INTERFRAME_STABLE), admission)
    }

    private fun candidate(
        rollRad: Float = 0f,
        worldFrameStability: UstrfWorldFrameStability = UstrfWorldFrameStability.INTER_FRAME_STABLE
    ) = UstrfVioPoseCandidate(
        sourceFrame = frame,
        worldFrame = "arcore-world-v1",
        worldFrameStability = worldFrameStability,
        worldCameraTranslationM = UstrfVector3(1f, 2f, 3f),
        yawRad = .1f,
        rollRad = rollRad,
        pitchRad = 0f,
        tracking = UstrfPoseState.TRACKING,
        confidence = .9f,
        validUntilNs = 2_000L,
        source = "arcore"
    )

    private fun extrinsics(
        cameraFrame: String = "camera-v1",
        validUntilNs: Long = 2_000L,
        independentlyVerified: Boolean = true
    ) = UstrfCameraBodyExtrinsicsReceipt(
        cameraFrame = cameraFrame,
        bodyFrame = "body-v1",
        cameraToBodyTranslationM = UstrfVector3(0f, 0f, 0f),
        cameraToBodyYawRad = 0f,
        calibrationId = "fixture",
        verifiedAtNs = 900L,
        validUntilNs = validUntilNs,
        confidence = .95f,
        independentlyVerified = independentlyVerified
    )
}
