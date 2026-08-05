package com.linnan.blindassist.device.glasses

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.linnan.blindassist.camera.AtomS3rMjpegFrameSource
import com.linnan.blindassist.vision.FrameClockDomain
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class GlassesConnectionRepositoryDeviceTest {
    @Test
    fun phoneConnectsToAtomS3rStatusRangeAndStream() {
        val status = GlassesConnectionRepository()
            .connect("http://192.168.5.11")
            .getOrThrow()

        assertTrue(status.firmwareVersion.startsWith("atoms3r_m12_tof4m_"))
        assertTrue(status.tofValid)
        assertTrue(status.tofRangeMm != null)
        assertTrue(status.streamReachable)
    }

    @Test
    fun liveMjpegDecodesMonotonicFramesWithBoundTofMetadata() {
        val source = AtomS3rMjpegFrameSource("http://192.168.5.11")
        val stamps = Collections.synchronizedList(mutableListOf<Long>())
        val ranges = Collections.synchronizedList(mutableListOf<Int>())
        val clockDomains = Collections.synchronizedList(mutableListOf<FrameClockDomain>())
        val frames = CountDownLatch(5)

        try {
            source.start(
                previewView = null,
                onFrame = { frame ->
                    frame.frameStamp?.let {
                        stamps += it.capturedAtNs
                        clockDomains += it.clockDomain
                    }
                    frame.rangingSample?.rangeMm?.let(ranges::add)
                    frame.close()
                    frames.countDown()
                },
                onStarted = {},
                onError = { throw AssertionError(it) }
            )

            assertTrue("Expected five decoded AtomS3R frames", frames.await(15L, TimeUnit.SECONDS))
        } finally {
            source.shutdown()
        }

        val firstFiveStamps = stamps.take(5)
        assertEquals(5, firstFiveStamps.size)
        assertTrue(firstFiveStamps.zipWithNext().all { (left, right) -> right > left })
        assertTrue(clockDomains.all { it == FrameClockDomain.EXTERNAL_DEVICE_MONOTONIC_UNMAPPED })
        assertTrue("Expected at least one valid frame-bound ToF range", ranges.isNotEmpty())
    }
}
