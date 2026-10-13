package com.linnan.blindassist

import com.linnan.blindassist.capture.AtomCameraCaptureFrame
import com.linnan.blindassist.capture.ImuCaptureFrame
import com.linnan.blindassist.capture.ImuCaptureRecord
import org.junit.Assert.*
import org.junit.Test

class MobileFusionInputBufferTest {
    @Test fun futureCameraReceiptDoesNotHideTheCausalFrame() {
        val inputs = MobileFusionInputBuffer()
        val causal = image(950)
        inputs.addCamera(causal)
        inputs.addCamera(image(1_010))
        inputs.addImu(imu(980))
        inputs.addImu(imu(1_020))

        val match = inputs.match(ms(1_000))

        assertSame(causal, match.image)
        assertEquals(ms(980), match.imu?.hostReceivedMonotonicNs)
        assertEquals(0L, match.epoch)
    }

    @Test fun pairingIncludesTheExactAgeBoundaryAndRejectsOldOrFutureInputs() {
        val inputs = MobileFusionInputBuffer()
        inputs.addCamera(image(900))
        inputs.addImu(imu(900))
        assertNotNull(inputs.match(ms(1_000)).image)
        assertNotNull(inputs.match(ms(1_000)).imu)
        assertNull(inputs.match(ms(1_001)).image)
        assertNull(inputs.match(ms(1_001)).imu)
        assertNull(inputs.match(ms(899)).image)
        assertNull(inputs.match(ms(899)).imu)
    }

    @Test fun inputHistoriesRemainBoundedWhileMatchingDelayedTof() {
        val inputs = MobileFusionInputBuffer(cameraCapacity = 2, imuCapacity = 2)
        for (time in 900L..902L) {
            inputs.addCamera(image(time))
            inputs.addImu(imu(time))
        }
        assertNull(inputs.match(ms(900)).image)
        assertNull(inputs.match(ms(900)).imu)
        assertEquals(ms(901), inputs.match(ms(901)).image?.receivedElapsedNs)
        assertEquals(ms(901), inputs.match(ms(901)).imu?.hostReceivedMonotonicNs)
    }

    @Test fun streamResetClearsBothSourcesAndDoesNotRelabelAnOldSnapshot() {
        val inputs = MobileFusionInputBuffer()
        inputs.addCamera(image(990))
        inputs.addImu(imu(990))
        val before = inputs.match(ms(1_000))

        assertEquals(1L, inputs.invalidate(clearInputs = true))
        val after = inputs.match(ms(1_000))
        assertEquals(0L, before.epoch)
        assertEquals(1L, after.epoch)
        assertNull(after.image)
        assertNull(after.imu)
        assertNull(inputs.invalidate(expectedEpoch = before.epoch))
        assertEquals(1L, inputs.epoch)
    }

    @Test fun imuRestartInvalidatesCameraAndPoseFromThePreviousBoot() {
        val inputs = MobileFusionInputBuffer()
        inputs.addCamera(image(990))
        inputs.addImu(imu(990, "first"))
        val before = inputs.match(ms(1_000))

        inputs.addImu(imu(1_005, "second"))

        val after = inputs.match(ms(1_010))
        assertEquals(before.epoch + 1, after.epoch)
        assertNull(after.image)
        assertEquals("second", after.imu?.frame?.bootId)
        assertNull(inputs.match(ms(1_000)).imu)
    }

    @Test fun expiredWorkCanResetInferenceWithoutDiscardingFreshSensorReceipts() {
        val inputs = MobileFusionInputBuffer()
        inputs.addCamera(image(990))
        inputs.addImu(imu(990))
        val before = inputs.match(ms(1_000))
        assertEquals(1L, inputs.invalidate(expectedEpoch = before.epoch))
        val after = inputs.match(ms(1_010))
        assertEquals(1L, after.epoch)
        assertSame(before.image, after.image)
        assertSame(before.imu, after.imu)
    }

    @Test fun shutdownFreezesTheFinalEpochAgainstInFlightCallbacks() {
        val inputs = MobileFusionInputBuffer()
        inputs.addCamera(image(990))
        inputs.addImu(imu(990))
        inputs.retire()
        val finalEpoch = inputs.epoch

        inputs.addCamera(image(1_000))
        inputs.addImu(imu(1_000, "reboot"))
        assertNull(inputs.invalidate(clearInputs = true))
        inputs.retire()

        val after = inputs.match(ms(1_010))
        assertEquals(finalEpoch, after.epoch)
        assertNull(after.image)
        assertNull(after.imu)
    }

    private fun ms(value: Long) = value * 1_000_000L
    private fun image(timeMs: Long) = AtomCameraCaptureFrame(
        byteArrayOf(), ms(timeMs), 640, 480, timeMs, "boot", null, null, emptyMap(), "test",
    )
    private fun imu(timeMs: Long, boot: String = "boot") = ImuCaptureRecord(
        timeMs, byteArrayOf(), ms(timeMs), "imu_frame",
        ImuCaptureFrame(timeMs, timeMs * 1_000, "device", boot, List(3) { 0 }, List(3) { 0 },
            listOf(0.0, -1.0, 0.0), List(3) { 0.0 }, 0.0, 0.0, true, 10_000, 1_000, false,
            listOf(0.0, -1.0, 0.0)), emptyList(),
    )
}
