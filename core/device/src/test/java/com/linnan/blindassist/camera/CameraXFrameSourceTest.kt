package com.linnan.blindassist.camera

import com.linnan.blindassist.vision.FrameClockDomain
import org.junit.Assert.assertEquals
import org.junit.Test

class CameraXFrameSourceTest {
    @Test
    fun elapsedRealtimeReceiptDoesNotPrecedeCameraCapture() {
        assertEquals(
            2_000L,
            normalizedFrameReceiptTime(
                capturedAtNs = 2_000L,
                observedReceivedAtNs = 1_999L,
                clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
            )
        )
    }

    @Test
    fun unmappedCameraClockKeepsObservedReceiptTime() {
        assertEquals(
            1_999L,
            normalizedFrameReceiptTime(
                capturedAtNs = 2_000L,
                observedReceivedAtNs = 1_999L,
                clockDomain = FrameClockDomain.CAMERA_HARDWARE_UNMAPPED
            )
        )
    }
}
