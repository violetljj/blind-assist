package com.linnan.blindassist.vision

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.util.ArrayDeque

class RgbaLumaSidecarTest {
    @Test
    fun copiesLumaBeforeCallerClosesRgbaFrame() {
        val executor = QueuedExecutor()
        var delivered: ByteArray? = null
        val source = ByteBuffer.allocate(16).apply {
            put(0, 255.toByte()); put(1, 0.toByte()); put(2, 0.toByte()); put(3, 255.toByte())
            put(4, 0.toByte()); put(5, 255.toByte()); put(6, 0.toByte()); put(7, 255.toByte())
            put(8, 0.toByte()); put(9, 0.toByte()); put(10, 255.toByte()); put(11, 255.toByte())
            put(12, 255.toByte()); put(13, 255.toByte()); put(14, 255.toByte()); put(15, 255.toByte())
        }
        val frame = ClearingFrame(source)
        val sidecar = RgbaLumaSidecar(
            executor = executor,
            maxResultAgeNanos = Long.MAX_VALUE,
            outputSize = 2,
            process = { owned -> owned.pixels.copyOf() },
            onFreshResult = { result -> delivered = result.value }
        )
        try {
            assertTrue(sidecar.submit(frame, capturedAtNanos = 1L))
            frame.close()
            executor.runAll()
            assertArrayEquals(byteArrayOf(76, 149.toByte(), 28, 255.toByte()), delivered)
            assertArrayEquals(ByteArray(16), source.array())
        } finally {
            sidecar.close()
        }
    }

    @Test
    fun rejectsNewFrameAfterClose() {
        val frame = ClearingFrame(ByteBuffer.allocate(4))
        val sidecar = RgbaLumaSidecar<Int>(
            executor = QueuedExecutor(),
            maxResultAgeNanos = 1L,
            outputSize = 1,
            process = { 1 },
            onFreshResult = {}
        )
        sidecar.close()
        assertFalse(sidecar.submit(frame, capturedAtNanos = 1L))
    }

    private class ClearingFrame(private val bytes: ByteBuffer) : RgbaVisionFrame {
        override val width = 2
        override val height = 2
        override val rotationDegrees = 0
        override val rowStride = 8
        override val pixelStride = 4
        override val buffer: ByteBuffer = bytes
        override fun close() { bytes.array().fill(0) }
    }

    private class QueuedExecutor : java.util.concurrent.Executor {
        private val commands = ArrayDeque<Runnable>()
        override fun execute(command: Runnable) { commands.addLast(command) }
        fun runAll() { while (commands.isNotEmpty()) commands.removeFirst().run() }
    }
}
