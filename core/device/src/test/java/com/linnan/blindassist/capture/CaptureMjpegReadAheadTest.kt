package com.linnan.blindassist.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.EOFException
import java.util.concurrent.atomic.AtomicInteger

class CaptureMjpegReadAheadTest {
    @Test(timeout = 5000)
    fun fullQueueAppliesMeasuredBackpressureAndDoesNotReplaceFrames() {
        val reads = AtomicInteger()
        val performance = CaptureCameraPerformance()
        val reader = CaptureMjpegReadAhead(readFrame = {
            val sequence = reads.incrementAndGet()
            if (sequence > 2) throw EOFException("end fixture")
            AtomCameraCaptureFrame(byteArrayOf(sequence.toByte()), sequence.toLong(), 1, 1,
                sequence.toLong(), "fixture", null, null, emptyMap(), "fixture")
        }, shouldStop = { false }, preview = {}, clock = System::nanoTime,
            performance = performance, capacity = 1)
        try {
            val deadline = System.nanoTime() + 1_000_000_000L
            while (performance.receipt().getLong("pending_queue_backpressure_count") == 0L &&
                System.nanoTime() < deadline) Thread.sleep(5)
            assertEquals(2, reads.get())
            assertEquals(1, performance.receipt().getInt("pending_queue_backpressure_count"))
            for (sequence in 1L..2L) {
                val frame = requireNotNull(reader.next())
                assertEquals(sequence, frame.sequence)
                performance.delivering(frame, System.nanoTime())
            }
            try { reader.next(); throw AssertionError("EOF must follow both frames") }
            catch (_: EOFException) { }
            assertNull(reader.next())
        } finally { reader.close() }
        assertTrue(performance.receipt().getJSONObject("queue_enqueue_wait_ms").getDouble("all_max") > 0)
        assertEquals(0, performance.receipt().getInt("queued_frames_abandoned_on_callback_failure"))
    }
}
