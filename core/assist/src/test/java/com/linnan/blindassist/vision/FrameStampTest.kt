package com.linnan.blindassist.vision

import org.junit.Assert.assertEquals
import org.junit.Test

class FrameStampTest {
    @Test
    fun elapsedRealtimeStampRequiresComparableReceiptTime() {
        val stamp = FrameStamp(
            frameId = 7L,
            capturedAtNs = 1_000L,
            receivedAtNs = 1_200L,
            sourceId = "camera2:0",
            coordinateFrame = "camera2:0:analysis-buffer",
            clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
        )

        assertEquals(7L, stamp.frameId)
        assertEquals(FrameClockDomain.ANDROID_ELAPSED_REALTIME, stamp.clockDomain)
    }

    @Test(expected = IllegalArgumentException::class)
    fun elapsedRealtimeStampRejectsReceiptBeforeCapture() {
        FrameStamp(
            frameId = 0L,
            capturedAtNs = 2_000L,
            receivedAtNs = 1_000L,
            sourceId = "camera2:0",
            coordinateFrame = "camera2:0:analysis-buffer",
            clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
        )
    }

    @Test
    fun unmappedCameraClockDoesNotPretendToBeCrossClockComparable() {
        val stamp = FrameStamp(
            frameId = 0L,
            capturedAtNs = 2_000L,
            receivedAtNs = 1_000L,
            sourceId = "camera2:external",
            coordinateFrame = "camera2:external:analysis-buffer",
            clockDomain = FrameClockDomain.CAMERA_HARDWARE_UNMAPPED
        )

        assertEquals(FrameClockDomain.CAMERA_HARDWARE_UNMAPPED, stamp.clockDomain)
    }
}
