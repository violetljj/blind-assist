package com.linnan.blindassist.camera

import com.linnan.blindassist.model.ReplayScenario
import com.linnan.blindassist.vision.RgbaVisionFrame
import com.linnan.blindassist.vision.VisionFrame
import java.nio.ByteBuffer
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertTrue
import org.junit.Test

class ReplayFrameSourceTest {
    @Test
    fun defaultPeriod_isTwoFramesPerSecond() {
        assertEquals(500L, ReplayFrameSource.DEFAULT_FRAME_PERIOD_MILLIS)
    }

    @Test
    fun start_isIdempotent_andEmitsIndependentFrames() {
        val frames = Collections.synchronizedList(mutableListOf<VisionFrame>())
        val startedCount = AtomicInteger()
        val frameLatch = CountDownLatch(2)
        val source = replaySource(periodMillis = 20L) { FakeRgbaFrame() }

        source.start(null, { frame ->
            frames += frame
            frameLatch.countDown()
        }, startedCount::incrementAndGet, ::failOnError)
        source.start(null, { error("Second start must be ignored") }, { error("Second start callback") }, ::failOnError)

        assertTrue(frameLatch.await(2L, TimeUnit.SECONDS))
        source.shutdown()

        assertEquals(1, startedCount.get())
        assertTrue(frames.size >= 2)
        assertNotSame(frames[0], frames[1])
    }

    @Test
    fun stop_invalidatesInFlightDecode_andClosesItsFrame() {
        val decodeStarted = CountDownLatch(1)
        val allowDecode = CountDownLatch(1)
        val decodedFrame = FakeRgbaFrame()
        val callbackCount = AtomicInteger()
        val source = replaySource {
            decodeStarted.countDown()
            assertTrue(allowDecode.await(2L, TimeUnit.SECONDS))
            decodedFrame
        }

        source.start(null, { callbackCount.incrementAndGet() }, { callbackCount.incrementAndGet() }, ::failOnError)
        assertTrue(decodeStarted.await(2L, TimeUnit.SECONDS))
        source.stop()
        allowDecode.countDown()

        assertTrue(decodedFrame.closed.await(2L, TimeUnit.SECONDS))
        Thread.sleep(50L)
        source.shutdown()

        assertEquals(0, callbackCount.get())
    }

    @Test
    fun restart_usesNewGeneration_andDropsOldGeneration() {
        val firstDecodeStarted = CountDownLatch(1)
        val allowFirstDecode = CountDownLatch(1)
        val newFrameDelivered = CountDownLatch(1)
        val firstFrame = FakeRgbaFrame()
        val decodeCount = AtomicInteger()
        val source = replaySource(periodMillis = 20L) {
            if (decodeCount.incrementAndGet() == 1) {
                firstDecodeStarted.countDown()
                assertTrue(allowFirstDecode.await(2L, TimeUnit.SECONDS))
                firstFrame
            } else {
                FakeRgbaFrame()
            }
        }

        source.start(null, { error("Old generation delivered a frame") }, {}, ::failOnError)
        assertTrue(firstDecodeStarted.await(2L, TimeUnit.SECONDS))
        source.stop()
        source.start(null, { newFrameDelivered.countDown() }, {}, ::failOnError)
        allowFirstDecode.countDown()

        assertTrue(firstFrame.closed.await(2L, TimeUnit.SECONDS))
        assertTrue(newFrameDelivered.await(2L, TimeUnit.SECONDS))
        source.shutdown()
    }

    @Test
    fun decodeFailure_isReportedOnlyOncePerSession() {
        val attempts = AtomicInteger()
        val errors = AtomicInteger()
        val enoughAttempts = CountDownLatch(3)
        val source = replaySource(periodMillis = 20L) {
            attempts.incrementAndGet()
            enoughAttempts.countDown()
            throw IllegalStateException("decode failed")
        }

        source.start(null, { error("No frame expected") }, { error("No start expected") }, { errors.incrementAndGet() })
        assertTrue(enoughAttempts.await(2L, TimeUnit.SECONDS))
        source.shutdown()

        assertTrue(attempts.get() >= 3)
        assertEquals(1, errors.get())
    }

    @Test
    fun callbackFailure_closesFrames_andReportsOnlyOnce() {
        val decodedFrames = Collections.synchronizedList(mutableListOf<FakeRgbaFrame>())
        val errors = AtomicInteger()
        val enoughFrames = CountDownLatch(3)
        val source = replaySource(periodMillis = 20L) {
            FakeRgbaFrame().also { decodedFrames += it }
        }

        source.start(null, {
            enoughFrames.countDown()
            throw IllegalStateException("consumer failed")
        }, {}, { errors.incrementAndGet() })
        assertTrue(enoughFrames.await(2L, TimeUnit.SECONDS))
        source.shutdown()

        assertEquals(1, errors.get())
        assertTrue(decodedFrames.take(3).all { it.isClosed })
    }

    @Test
    fun stop_doesNotWaitForInFlightConsumerCallback() {
        val callbackStarted = CountDownLatch(1)
        val releaseCallback = CountDownLatch(1)
        val source = replaySource { FakeRgbaFrame() }

        source.start(null, {
            callbackStarted.countDown()
            assertTrue(releaseCallback.await(2L, TimeUnit.SECONDS))
            it.close()
        }, {}, ::failOnError)
        assertTrue(callbackStarted.await(2L, TimeUnit.SECONDS))

        val startedAt = System.nanoTime()
        source.stop()
        val elapsedMillis = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startedAt)
        releaseCallback.countDown()
        source.shutdown()

        assertTrue("stop blocked for ${elapsedMillis}ms", elapsedMillis < 100L)
    }

    @Test
    fun shutdown_isTerminal() {
        val scheduler = Executors.newSingleThreadScheduledExecutor()
        val callbackCount = AtomicInteger()
        val source = ReplayFrameSource(
            scenario = ReplayScenario.NONE,
            frameDecoder = { FakeRgbaFrame() },
            scheduler = scheduler,
            framePeriodMillis = 20L
        )

        source.shutdown()
        source.start(null, { callbackCount.incrementAndGet() }, { callbackCount.incrementAndGet() }, ::failOnError)
        Thread.sleep(50L)

        assertTrue(scheduler.isShutdown)
        assertTrue(scheduler.awaitTermination(1L, TimeUnit.SECONDS))
        assertEquals(0, callbackCount.get())
    }

    private fun replaySource(
        periodMillis: Long = 20L,
        decoder: () -> RgbaVisionFrame
    ): ReplayFrameSource {
        return ReplayFrameSource(
            scenario = ReplayScenario.NONE,
            frameDecoder = decoder,
            scheduler = Executors.newSingleThreadScheduledExecutor(),
            framePeriodMillis = periodMillis
        )
    }

    private class FakeRgbaFrame : RgbaVisionFrame {
        val closed = CountDownLatch(1)
        val isClosed: Boolean get() = closed.count == 0L
        override val width: Int = 1
        override val height: Int = 1
        override val rotationDegrees: Int = 0
        override val buffer: ByteBuffer = ByteBuffer.allocateDirect(4)
        override val rowStride: Int = 4
        override val pixelStride: Int = 4

        override fun close() {
            closed.countDown()
        }
    }

    companion object {
        private fun failOnError(error: Throwable): Nothing = throw AssertionError(error)
    }
}
