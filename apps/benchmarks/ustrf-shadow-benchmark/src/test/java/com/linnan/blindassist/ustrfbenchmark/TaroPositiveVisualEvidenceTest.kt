package com.linnan.blindassist.ustrfbenchmark

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import com.linnan.blindassist.ustrf.UstrfPoseSample
import com.linnan.blindassist.ustrf.UstrfPoseState
import com.linnan.blindassist.ustrf.UstrfVector3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class TaroPositiveVisualEvidenceTest {
    @Test
    fun passiveSelection_targets500msThenUsesFrozenTieBreaks() {
        val reference = frame(9L, 1_000_000_000L)
        val earlier = payload(3L, 480_000_000L)
        val closer = payload(4L, 510_000_000L)
        val sameGapLargerId = payload(5L, 510_000_000L)

        val selected = TaroPositiveVisualEvidence.selectPassive(
            reference,
            listOf(earlier, sameGapLargerId, closer)
        )

        assertSame(closer, selected)
    }

    @Test
    fun tokenization_isPositiveOnlyBinnedAndFocusBounded() {
        val frame = FrameSize(100, 100)
        val tokenization = TaroPositiveVisualEvidence.tokens(
            listOf(
                Detection(0, "person", .9f, BoundingBox(30f, 55f, 60f, 95f), frame),
                Detection(0, "person", .8f, BoundingBox(31f, 56f, 61f, 96f), frame),
                Detection(56, "chair", .7f, BoundingBox(0f, 0f, 10f, 10f), frame)
            )
        )

        assertEquals(2, tokenization.tokens.size)
        assertTrue(tokenization.focusedTokens.single().label == "person")
        assertFalse(tokenization.focusedTokens.any { it.label == "chair" })
    }

    @Test
    fun tokenization_doesNotSplitIdentityWhenFocusIntersectionDiffers() {
        val frame = FrameSize(100, 100)
        val tokenization = TaroPositiveVisualEvidence.tokens(
            listOf(
                Detection(0, "person", .9f, BoundingBox(35f, 34f, 45f, 49f), frame),
                Detection(0, "person", .8f, BoundingBox(35f, 45f, 45f, 55f), frame)
            )
        )

        assertEquals(1, tokenization.tokens.size)
        assertEquals(tokenization.tokens, tokenization.focusedTokens)
    }

    @Test
    fun comparison_countsOnlyTokensNewBeyondCurrent() {
        val currentToken = TaroPositiveVisualToken("chair", 1, 2)
        val passiveToken = TaroPositiveVisualToken("person", 1, 2)
        val poseToken = TaroPositiveVisualToken("bottle", 2, 1)
        val frame = frame(1L, 1L)

        val comparison = TaroPositiveVisualEvidence.compare(
            receipt(frame, setOf(currentToken), setOf(currentToken)),
            receipt(frame, setOf(currentToken, passiveToken), setOf(currentToken, passiveToken)),
            receipt(frame, setOf(currentToken, passiveToken, poseToken), setOf(currentToken, passiveToken))
        )

        assertEquals(1, comparison.passiveNewFocusedTokenCount)
        assertEquals(1, comparison.poseDiverseNewFocusedTokenCount)
        assertEquals(1, comparison.passiveNewAllTokenCount)
        assertEquals(2, comparison.poseDiverseNewAllTokenCount)
    }

    private fun receipt(
        frame: UstrfFrameStamp,
        tokens: Set<TaroPositiveVisualToken>,
        focusedTokens: Set<TaroPositiveVisualToken>
    ) = TaroPositiveVisualReceipt(frame, tokens, focusedTokens, 0.0, 0L, 0L, 0L, 0L)

    private fun payload(frameId: Long, timestampNs: Long): TaroOwnedRgbPayload {
        val frame = frame(frameId, timestampNs)
        return TaroOwnedRgbPayload(
            sourceFrame = frame,
            anchorPose = UstrfPoseSample(
                timestampNs = timestampNs,
                worldFrame = "arcore-local-anchor-v1:test",
                cameraFrame = CAMERA_FRAME,
                worldCameraTranslationM = UstrfVector3(0f, 0f, 0f),
                yawRad = 0f,
                gravityWorld = UstrfVector3(0f, -9.80665f, 0f),
                tracking = UstrfPoseState.TRACKING,
                confidence = 1f,
                validUntilNs = timestampNs + 1_000_000_000L
            ),
            imageWidthPx = 2,
            imageHeightPx = 2,
            imageFormat = 35,
            planes = listOf(TaroOwnedYuvPlane(2, 1, byteArrayOf(0, 0, 0, 0))),
            contentSha256 = "c".repeat(64)
        )
    }

    private fun frame(frameId: Long, timestampNs: Long) =
        UstrfFrameStamp(frameId, timestampNs, CAMERA_FRAME)

    private companion object {
        const val CAMERA_FRAME = "arcore-camera-v1"
    }
}
