package com.linnan.blindassist.ustrfbenchmark

import com.linnan.blindassist.ustrf.UstrfFrameStamp
import com.linnan.blindassist.ustrf.UstrfPoseSample
import com.linnan.blindassist.ustrf.UstrfPoseState
import com.linnan.blindassist.ustrf.UstrfVector3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class TaroSharedCameraExactSourcePosePairerTest {
    @Test
    fun imageThenPose_emitsOnlyTheExactTimestampPayload() {
        val pairer = TaroSharedCameraExactSourcePosePairer()

        assertNull(pairer.observeImage(image(1_000L)).matchedPayload)
        val matched = pairer.observePose(frame(1_000L), admission(1_000L)).matchedPayload

        assertNotNull(matched)
        assertEquals(1_000L, matched?.sourceFrame?.capturedAtNs)
        assertEquals(1_000L, matched?.anchorPose?.timestampNs)
        assertEquals(0, pairer.snapshot().pendingImageCount)
        assertEquals(1, pairer.snapshot().exactMatchCount)
    }

    @Test
    fun poseThenImage_emitsOnlyTheExactTimestampPayload() {
        val pairer = TaroSharedCameraExactSourcePosePairer()

        assertNull(pairer.observePose(frame(2_000L), admission(2_000L)).matchedPayload)
        val matched = pairer.observeImage(image(2_000L)).matchedPayload

        assertEquals(frame(2_000L), matched?.sourceFrame)
        assertEquals(1, pairer.snapshot().exactMatchCount)
    }

    @Test
    fun adjacentTimestamps_doNotUseNearestFallback() {
        val pairer = TaroSharedCameraExactSourcePosePairer()

        assertNull(pairer.observePose(frame(3_000L), admission(3_000L)).matchedPayload)
        assertNull(pairer.observeImage(image(3_001L)).matchedPayload)

        val snapshot = pairer.snapshot()
        assertEquals(0, snapshot.exactMatchCount)
        assertEquals(1, snapshot.pendingImageCount)
        assertEquals(1, snapshot.pendingPoseCount)
    }

    @Test
    fun pendingInputs_areBoundedByAgeAndOwnedBytes() {
        val pairer = TaroSharedCameraExactSourcePosePairer(
            maximumPendingAgeNs = 100L,
            maximumPendingImageBytes = 8L
        )

        pairer.observeImage(image(1_000L, byteCount = 6))
        val byteEviction = pairer.observeImage(image(1_050L, byteCount = 6))
        assertEquals(1, byteEviction.byteCapEvictedImageCount)
        pairer.observePose(frame(1_050L), admission(1_050L))
        pairer.observePose(frame(1_060L), admission(1_060L))
        val ageEviction = pairer.observeImage(image(1_200L, byteCount = 6))

        assertTrue(ageEviction.ageEvictedPoseCount >= 1)
        assertTrue(pairer.snapshot().pendingImageBytes <= 8L)
    }

    @Test
    fun reset_removesBothSidesOfAnAnchorEpoch() {
        val pairer = TaroSharedCameraExactSourcePosePairer()
        pairer.observeImage(image(4_000L))
        pairer.observePose(frame(4_001L), admission(4_001L))

        val receipt = pairer.reset()

        assertEquals(1, receipt.evictedImageCount)
        assertEquals(1, receipt.evictedPoseCount)
        assertEquals(0, pairer.snapshot().pendingImageCount)
        assertEquals(0, pairer.snapshot().pendingPoseCount)
    }

    private fun frame(timestampNs: Long) = UstrfFrameStamp(
        frameId = timestampNs,
        capturedAtNs = timestampNs,
        coordinateFrame = CAMERA_FRAME
    )

    private fun admission(timestampNs: Long) = TaroArCoreAnchorPoseAdmission.Available(
        cameraPose = UstrfPoseSample(
            timestampNs = timestampNs,
            worldFrame = "arcore-local-anchor-v1:$SESSION_TOKEN",
            cameraFrame = CAMERA_FRAME,
            worldCameraTranslationM = UstrfVector3(0f, 0f, 0f),
            yawRad = 0f,
            gravityWorld = UstrfVector3(0f, -9.80665f, 0f),
            tracking = UstrfPoseState.TRACKING,
            confidence = 1f,
            validUntilNs = timestampNs + 1_000L
        ),
        sessionToken = SESSION_TOKEN,
        continuousTrackingFrames = 15
    )

    private fun image(timestampNs: Long, byteCount: Int = 4) = TaroSharedCameraOwnedYuvFrame(
        timestampNs = timestampNs,
        imageWidthPx = 2,
        imageHeightPx = 2,
        imageFormat = 35,
        planes = listOf(TaroOwnedYuvPlane(2, 1, ByteArray(byteCount) { it.toByte() })),
        contentSha256 = "a".repeat(64)
    )

    private companion object {
        const val CAMERA_FRAME = "arcore-camera-v1"
        const val SESSION_TOKEN = "shared-camera-pairer-test"
    }
}
