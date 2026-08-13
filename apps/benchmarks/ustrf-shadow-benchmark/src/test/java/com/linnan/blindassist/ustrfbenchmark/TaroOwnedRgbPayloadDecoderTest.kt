package com.linnan.blindassist.ustrfbenchmark

import com.linnan.blindassist.ustrf.UstrfFrameStamp
import com.linnan.blindassist.ustrf.UstrfPoseSample
import com.linnan.blindassist.ustrf.UstrfPoseState
import com.linnan.blindassist.ustrf.UstrfVector3
import org.junit.Assert.assertEquals
import org.junit.Test

class TaroOwnedRgbPayloadDecoderTest {
    @Test
    fun delayedOwnedDecode_isDeterministicAndPreservesSourceIdentity() {
        val payload = payload()
        val decoder = TaroOwnedRgbPayloadDecoder()

        val first = decoder.decode(payload)
        val second = decoder.decode(payload)

        assertEquals(payload.sourceFrame, first.sourceFrame)
        assertEquals(payload.contentSha256, first.sourceYuvSha256)
        assertEquals(first, second)
        assertEquals(2, first.widthPx)
        assertEquals(2, first.heightPx)
        assertEquals(16, first.rgbaByteCount)
    }

    private fun payload(): TaroOwnedRgbPayload {
        val sourceFrame = UstrfFrameStamp(7L, 100L, CAMERA_FRAME)
        return TaroOwnedRgbPayload(
            sourceFrame = sourceFrame,
            anchorPose = UstrfPoseSample(
                timestampNs = sourceFrame.capturedAtNs,
                worldFrame = ANCHOR_FRAME,
                cameraFrame = CAMERA_FRAME,
                worldCameraTranslationM = UstrfVector3(0f, 0f, 0f),
                yawRad = 0f,
                gravityWorld = UstrfVector3(0f, -9.80665f, 0f),
                tracking = UstrfPoseState.TRACKING,
                confidence = 1f,
                validUntilNs = 1_000L
            ),
            imageWidthPx = 2,
            imageHeightPx = 2,
            imageFormat = 35,
            planes = listOf(
                TaroOwnedYuvPlane(2, 1, byteArrayOf(16, 235.toByte(), 235.toByte(), 16)),
                TaroOwnedYuvPlane(1, 1, byteArrayOf(128.toByte())),
                TaroOwnedYuvPlane(1, 1, byteArrayOf(128.toByte()))
            ),
            contentSha256 = "b".repeat(64)
        )
    }

    private companion object {
        const val CAMERA_FRAME = "arcore-camera-v1"
        const val ANCHOR_FRAME = "arcore-local-anchor-v1:test"
    }
}
