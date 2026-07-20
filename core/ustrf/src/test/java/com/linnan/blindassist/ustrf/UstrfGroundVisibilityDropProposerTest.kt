package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfGroundVisibilityDropProposerTest {
    @Test fun fartherObservedGroundProposesDropButMissingDepthDoesNot() {
        val proposer = UstrfGroundVisibilityDropProposer(UstrfGroundVisibilityDropConfig(sampleStridePx = 1, minimumExpectedForwardMeters = .1f))
        val farther = proposer.propose(admission(), image(2_200), intrinsics(), extrinsics(), ground(), 1_100L) as UstrfGroundVisibilityDropProposal.Available
        val missing = proposer.propose(admission(), image(0), intrinsics(), extrinsics(), ground(), 1_100L) as UstrfGroundVisibilityDropProposal.Available
        assertTrue(farther.evidence.any { it.kind == UstrfGeometryKind.DROP })
        assertEquals(0, missing.evidence.size)
    }

    @Test fun registrationMismatchFailsClosed() {
        assertEquals(UstrfGroundVisibilityDropProposal.Unavailable(UstrfGroundVisibilityDropFailure.REGISTRATION_MISMATCH),
            UstrfGroundVisibilityDropProposer().propose(admission(), image(2_200).copy(registrationTransformId = "wrong"), intrinsics(), extrinsics(), ground(), 1_100L))
    }

    private fun frame() = UstrfFrameStamp(1L, 1_000L, "camera")
    private fun admission() = UstrfMetricGeometryProjectionAdmission.Available(frame(), "depth", "camera", "body", "reg", 2_000L)
    private fun intrinsics() = UstrfCameraIntrinsicsReceipt("camera", "v1", 3, 3, 1f, 1f, 1f, 1f, 0L, 2_000L, 1f, true)
    private fun extrinsics() = UstrfCameraBodyFullExtrinsicsReceipt("camera", "body", UstrfVector3(0f, 1.5f, 0f), floatArrayOf(0f,0f,0f,1f), "m", 0L, 2_000L, 1f, true)
    private fun ground() = UstrfVerifiedGroundPlaneReceipt(frame(), "body", UstrfVector3(0f,1f,0f), 0f, 1f, true, 2_000L)
    private fun image(centerMm: Int): UstrfRegisteredMetricDepthImage {
        val depths = IntArray(9); val confidence = FloatArray(9)
        depths[7] = centerMm; confidence[7] = 1f
        return UstrfRegisteredMetricDepthImage(frame(), 3, 3, "depth", "reg", depths, confidence, 2_000L)
    }
}
