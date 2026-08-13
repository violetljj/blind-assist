package com.linnan.blindassist.ustrfbenchmark

import com.linnan.blindassist.ustrf.TaroPoseDiverseFrameSelector
import com.linnan.blindassist.ustrf.TaroPoseDiverseSelection
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import com.linnan.blindassist.ustrf.UstrfPoseSample
import com.linnan.blindassist.ustrf.UstrfPoseState
import com.linnan.blindassist.ustrf.UstrfVector3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class TaroOwnedRgbPayloadHistoryTest {
    @Test
    fun selectorIdentity_resolvesOnlyTheExactOwnedPayload() {
        val history = TaroOwnedRgbPayloadHistory(maximumRetainedAgeNs = 1_000L, maximumRetainedBytes = 64L)
        val historical = payload(frameId = 7L, timestampNs = 100L, byteCount = 4, xM = 0f)
        history.append(historical)
        val reference = payload(frameId = 8L, timestampNs = 300L, byteCount = 4, xM = .03f)

        val selection = TaroPoseDiverseFrameSelector(
            enabled = true,
            minimumGapNs = 150L,
            maximumGapNs = 1_000L
        ).select(reference.sourceFrame, reference.anchorPose, history.bufferedPoseFrames())

        assertTrue(selection is TaroPoseDiverseSelection.Available)
        selection as TaroPoseDiverseSelection.Available
        assertEquals(historical.sourceFrame, selection.selectedFrame)
        assertSame(historical, history.lookupExact(selection.selectedFrame))
        assertNull(history.lookupExact(selection.selectedFrame.copy(coordinateFrame = "other-camera")))
    }

    @Test
    fun history_evictsBySourceAgeWithoutNearestFallback() {
        val history = TaroOwnedRgbPayloadHistory(maximumRetainedAgeNs = 1_000L, maximumRetainedBytes = 64L)
        val oldest = payload(frameId = 1L, timestampNs = 100L, byteCount = 4)
        val newest = payload(frameId = 2L, timestampNs = 900L, byteCount = 4)
        history.append(oldest)
        history.append(newest)

        val receipt = history.advanceTo(1_101L)

        assertEquals(1, receipt.ageEvictionCount)
        assertNull(history.lookupExact(oldest.sourceFrame))
        assertSame(newest, history.lookupExact(newest.sourceFrame))
    }

    @Test
    fun history_evictsOldestPayloadBeforeExceedingByteCap() {
        val history = TaroOwnedRgbPayloadHistory(maximumRetainedAgeNs = 1_000L, maximumRetainedBytes = 6L)
        val oldest = payload(frameId = 1L, timestampNs = 100L, byteCount = 4)
        val newest = payload(frameId = 2L, timestampNs = 200L, byteCount = 4)
        history.append(oldest)

        val receipt = history.append(newest)

        assertEquals(1, receipt.byteCapEvictionCount)
        assertEquals(4L, receipt.retainedBytes)
        assertNull(history.lookupExact(oldest.sourceFrame))
        assertSame(newest, history.lookupExact(newest.sourceFrame))
    }

    private fun payload(
        frameId: Long,
        timestampNs: Long,
        byteCount: Int,
        xM: Float = 0f
    ): TaroOwnedRgbPayload {
        val frame = UstrfFrameStamp(frameId, timestampNs, CAMERA_FRAME)
        return TaroOwnedRgbPayload(
            sourceFrame = frame,
            anchorPose = UstrfPoseSample(
                timestampNs = timestampNs,
                worldFrame = ANCHOR_FRAME,
                cameraFrame = CAMERA_FRAME,
                worldCameraTranslationM = UstrfVector3(xM, 0f, 0f),
                yawRad = 0f,
                gravityWorld = UstrfVector3(0f, -9.80665f, 0f),
                tracking = UstrfPoseState.TRACKING,
                confidence = 1f,
                validUntilNs = timestampNs + 1_000L
            ),
            imageWidthPx = 2,
            imageHeightPx = 2,
            imageFormat = 35,
            planes = listOf(TaroOwnedYuvPlane(2, 1, ByteArray(byteCount))),
            contentSha256 = "a".repeat(64)
        )
    }

    private companion object {
        const val CAMERA_FRAME = "arcore-camera-v1"
        const val ANCHOR_FRAME = "arcore-local-anchor-v1:test"
    }
}
