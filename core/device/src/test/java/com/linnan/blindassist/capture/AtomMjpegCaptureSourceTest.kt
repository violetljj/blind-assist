package com.linnan.blindassist.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import java.util.Collections

class AtomMjpegCaptureSourceTest {
    @Test(timeout = 10000)
    fun closeWaitsForAcceptedCallbackAndTerminalStatusBeyondOldDeadline() {
        val http = Connection()
        val source = source(http)
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val frameCompleted = CountDownLatch(1)
        val stopped = CountDownLatch(1)
        val closed = CountDownLatch(1)
        val failure = AtomicReference<Throwable?>()
        val closer = Thread {
            try {
                source.close()
                assertEquals("Terminal callback must finish before close returns", 0L, stopped.count)
            } catch (error: Throwable) { failure.set(error) }
            finally { closed.countDown() }
        }
        try {
            source.start({ frame ->
                assertArrayEquals(jpeg, frame.jpeg)
                entered.countDown()
                check(release.await(8, TimeUnit.SECONDS))
                frameCompleted.countDown()
            }, { status ->
                if (status.state == CnhConnectionState.STOPPED) {
                    assertEquals("Frame callback must finish before STOPPED", 0L, frameCompleted.count)
                    stopped.countDown()
                }
            })
            assertTrue(entered.await(2, TimeUnit.SECONDS))
            closer.start()
            assertTrue(http.disconnected.await(2, TimeUnit.SECONDS))
            assertFalse("Close must not abandon an accepted callback at 1.5 seconds",
                closed.await(1800, TimeUnit.MILLISECONDS))
            release.countDown()
            assertTrue(closed.await(2, TimeUnit.SECONDS))
            failure.get()?.let { throw it }
        } finally {
            release.countDown()
            closer.join(2000)
            source.close()
        }
    }

    @Test(timeout = 5000)
    fun callbackCanCloseItsOwnSourceWithoutWaitingForItself() {
        val source = source(Connection())
        val stopped = CountDownLatch(1)
        try {
            source.start({ source.close() }, { status ->
                if (status.state == CnhConnectionState.STOPPED) stopped.countDown()
            })
            assertTrue("A worker callback must not wait for its own termination",
                stopped.await(2, TimeUnit.SECONDS))
        } finally { source.close() }
    }

    @Test(timeout = 5000)
    fun previewReadsAheadOfSlowCallbackAndCloseDrainsEveryCompleteJpeg() {
        val http = Connection(3)
        val source = source(http)
        val allPreviewed = CountDownLatch(1)
        val callbackEntered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val closed = CountDownLatch(1)
        val delivered = Collections.synchronizedList(mutableListOf<Long?>())
        val closer = Thread { source.close(); closed.countDown() }
        source.setPreviewFrameListener { if (it.sequence == 3L) allPreviewed.countDown() }
        try {
            source.start({ frame ->
                delivered.add(frame.sequence)
                if (frame.sequence == 1L) {
                    callbackEntered.countDown()
                    check(release.await(3, TimeUnit.SECONDS))
                }
            }, {})
            assertTrue(callbackEntered.await(1, TimeUnit.SECONDS))
            assertTrue("Reader/preview must advance while the durable callback waits",
                allPreviewed.await(1, TimeUnit.SECONDS))
            assertEquals(listOf(1L), delivered.toList())
            closer.start()
            assertTrue(http.disconnected.await(1, TimeUnit.SECONDS))
            assertFalse(closed.await(50, TimeUnit.MILLISECONDS))
            release.countDown()
            assertTrue(closed.await(1, TimeUnit.SECONDS))
            assertEquals(listOf(1L, 2L, 3L), delivered.toList())
            val receipt = source.capturePerformanceReceipt()
            assertEquals(3, receipt.getInt("frames_received"))
            assertEquals(3, receipt.getInt("callbacks_completed"))
            assertEquals(0, receipt.getInt("queued_frames_abandoned_on_callback_failure"))
        } finally { release.countDown(); closer.join(1000); source.close() }
    }

    @Test(timeout = 5000)
    fun closeInsideCallbackDrainsQueuedFramesBeforeCallerFinalizesStorage() {
        val source = source(Connection(3))
        val allPreviewed = CountDownLatch(1)
        val stopped = CountDownLatch(1)
        val delivered = mutableListOf<Long?>()
        val failure = AtomicReference<Throwable?>()
        source.setPreviewFrameListener { if (it.sequence == 3L) allPreviewed.countDown() }
        try {
            source.start({ frame ->
                delivered.add(frame.sequence)
                if (frame.sequence == 1L) {
                    try {
                        check(allPreviewed.await(1, TimeUnit.SECONDS))
                        source.close()
                        assertEquals(listOf(1L, 2L, 3L), delivered)
                    } catch (error: Throwable) { failure.set(error) }
                }
            }, { if (it.state == CnhConnectionState.STOPPED) stopped.countDown() })
            assertTrue(stopped.await(2, TimeUnit.SECONDS))
            failure.get()?.let { throw it }
            assertEquals(0, source.capturePerformanceReceipt().getInt("queued_frames_abandoned_on_callback_failure"))
        } finally { source.close() }
    }

    @Test(timeout = 5000)
    fun callbackFailureReportsErrorAndCountsUndeliveredCompleteFrames() {
        val source = source(Connection(3))
        val allPreviewed = CountDownLatch(1)
        val failed = CountDownLatch(1)
        source.setPreviewFrameListener { if (it.sequence == 3L) allPreviewed.countDown() }
        try {
            source.start({
                check(allPreviewed.await(1, TimeUnit.SECONDS))
                throw IllegalStateException("fixture storage failure")
            }, { if (it.state == CnhConnectionState.ERROR) failed.countDown() })
            assertTrue(failed.await(2, TimeUnit.SECONDS))
            assertEquals(2, source.capturePerformanceReceipt().getInt("queued_frames_abandoned_on_callback_failure"))
        } finally { source.close() }
    }

    private fun source(http: Connection) = AtomMjpegCaptureSource("192.168.5.9", 81,
        openConnection = { http },
        readFrame = { input, endpoint -> AtomCameraCaptureReader.read(input, endpoint, 1L) },
        monotonicClock = System::nanoTime)

    private class Connection(private val frames: Int = 1) : HttpURLConnection(URL("http://192.168.5.9:81/stream")) {
        val disconnected = CountDownLatch(1)
        override fun connect() = Unit
        override fun usingProxy() = false
        override fun disconnect() { disconnected.countDown() }
        override fun getResponseCode() = HTTP_OK
        override fun getContentType() = "multipart/x-mixed-replace; boundary=atom"
        override fun getInputStream() = (1..frames).fold(ByteArray(0)) { bytes, index ->
            bytes + ("--atom\r\nContent-Type: image/jpeg\r\nContent-Length: ${jpeg.size}\r\n" +
                "X-Frame-Sequence: $index\r\n\r\n").toByteArray(Charsets.US_ASCII) + jpeg + "\r\n".toByteArray()
        }.inputStream()
    }

    private companion object {
        val jpeg = byteArrayOf(
            0xff.toByte(), 0xd8.toByte(), 0xff.toByte(), 0xc0.toByte(),
            0, 17, 8, 1, 0xe0.toByte(), 2, 0x80.toByte(), 3,
            1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0,
            0xff.toByte(), 0xd9.toByte(),
        )
    }
}
