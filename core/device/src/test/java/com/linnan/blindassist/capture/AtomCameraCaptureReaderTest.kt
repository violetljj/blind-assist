package com.linnan.blindassist.capture

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.BufferedInputStream
import java.io.EOFException

class AtomCameraCaptureReaderTest {
    @Test
    fun preservesJpegAndDeviceHeadersWithIndependentReceipt() {
        val jpeg = jpeg()
        val input = multipart(jpeg, "X-Sequence-Id: boot-123\r\n" +
            "X-Frame-Sequence: 42\r\nX-Capture-Timestamp-Us: 8000\r\n" +
            "X-Jpeg-Ready-Timestamp-Us: 8300\r\n")
        val frame = AtomCameraCaptureReader.read(input, "http://192.168.5.9:81/stream", 900_000_000)
        assertArrayEquals(jpeg, frame.jpeg)
        assertEquals(640, frame.width)
        assertEquals(480, frame.height)
        assertEquals(42L, frame.sequence)
        assertEquals("boot-123", frame.bootId)
        assertEquals(8000L, frame.deviceCaptureTimestampUs)
        assertEquals(8300L, frame.deviceJpegReadyTimestampUs)
        assertEquals(900_000_000L, frame.receivedElapsedNs)
        assertEquals("8000", frame.headers["x-capture-timestamp-us"])
        assertTrue(frame.sourceId.startsWith("atom-mjpeg:http://192.168.5.9:81/stream"))
        assertTrue(frame.sourceId.endsWith("boot:boot-123"))
    }

    @Test
    fun missingDeviceTimestampsStayAbsentInsteadOfUsingReceipt() {
        val frame = AtomCameraCaptureReader.read(multipart(jpeg()), "test-atom", 777)
        assertEquals(777L, frame.receivedElapsedNs)
        assertNull(frame.sequence)
        assertNull(frame.bootId)
        assertNull(frame.deviceCaptureTimestampUs)
        assertNull(frame.deviceJpegReadyTimestampUs)
    }

    @Test
    fun incompleteContentLengthDoesNotPublishPartialJpeg() {
        try {
            AtomCameraCaptureReader.read(multipart(jpeg(), declaredLength = jpeg().size + 10), "test-atom", 1)
            throw AssertionError("Partial JPEG must not be published")
        } catch (_: EOFException) {
            // The reader retains Content-Length framing instead of searching inside the body.
        }
    }

    @Test
    fun missingEndMarkerDoesNotPublishAnApparentlyCompleteFrame() {
        val jpeg = jpeg().copyOfRange(0, jpeg().size - 2)
        try {
            AtomCameraCaptureReader.read(multipart(jpeg), "test-atom", 1)
            throw AssertionError("Missing EOI must not be published")
        } catch (_: IllegalArgumentException) {
            // Original body is incomplete even when the advertised byte length matches.
        }
    }

    @Test
    fun readsSuccessiveFramesWithoutChangingBytesOrMixingHeaders() {
        val jpeg = jpeg()
        val first = multipartBytes(jpeg, "X-Frame-Sequence: 1\r\n")
        val second = multipartBytes(jpeg, "X-Frame-Sequence: 2\r\n")
        val input = BufferedInputStream((first + second).inputStream())
        val frame1 = AtomCameraCaptureReader.read(input, "test-atom", 10)
        val frame2 = AtomCameraCaptureReader.read(input, "test-atom", 20)
        assertArrayEquals(jpeg, frame1.jpeg)
        assertArrayEquals(jpeg, frame2.jpeg)
        assertEquals(1L, frame1.sequence)
        assertEquals(2L, frame2.sequence)
        assertEquals(20L, frame2.receivedElapsedNs)
    }

    private fun multipart(jpeg: ByteArray, headers: String = "", declaredLength: Int = jpeg.size) =
        BufferedInputStream(multipartBytes(jpeg, headers, declaredLength).inputStream())

    private fun multipartBytes(jpeg: ByteArray, headers: String = "", declaredLength: Int = jpeg.size): ByteArray =
        ("--atom-boundary\r\nContent-Type: image/jpeg\r\nContent-Length: $declaredLength\r\n" +
            headers + "\r\n").toByteArray(Charsets.US_ASCII) + jpeg + "\r\n".toByteArray()

    /** SOF carries native dimensions; the capture reader does not decode or recompress pixels. */
    private fun jpeg() = byteArrayOf(
        0xff.toByte(), 0xd8.toByte(), 0xff.toByte(), 0xc0.toByte(),
        0, 17, 8, 1, 0xe0.toByte(), 2, 0x80.toByte(), 3,
        1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0,
        0xff.toByte(), 0xd9.toByte(),
    )
}
