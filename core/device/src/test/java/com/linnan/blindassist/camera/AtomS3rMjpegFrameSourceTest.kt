package com.linnan.blindassist.camera

import com.linnan.blindassist.vision.FrameClockDomain
import java.io.BufferedInputStream
import java.io.ByteArrayInputStream
import java.nio.charset.StandardCharsets
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class AtomS3rMjpegFrameSourceTest {
    @Test
    fun parsesMultipartHeadersAndBindsFrameToTofSample() {
        val jpeg = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 1, 2, 0xFF.toByte(), 0xD9.toByte())
        val packet = AtomS3rMjpegFrameSource.MjpegPartReader.readPacket(
            BufferedInputStream(ByteArrayInputStream(part(jpeg, tofValid = true))),
            receivedAtNs = 9_000_000L
        )

        val metadata = packet.metadata()

        assertEquals(jpeg.toList(), packet.jpeg.toList())
        assertEquals(42L, metadata.frameStamp.frameId)
        assertEquals(1_234_000L, metadata.frameStamp.capturedAtNs)
        assertEquals(FrameClockDomain.EXTERNAL_DEVICE_MONOTONIC_UNMAPPED, metadata.frameStamp.clockDomain)
        assertEquals(1_240_000L, metadata.rangingSample?.sampledAtNs)
        assertEquals(875, metadata.rangingSample?.rangeMm)
        assertEquals(3_000L, metadata.rangingSample?.ageAtFrameReadyNs)
        assertTrue(metadata.rangingSample?.valid == true)
    }

    @Test
    fun invalidTofRetainsTimestampButDoesNotExposeRange() {
        val packet = AtomS3rMjpegFrameSource.MjpegPartReader.readPacket(
            BufferedInputStream(ByteArrayInputStream(part(byteArrayOf(1), tofValid = false))),
            receivedAtNs = 9_000_000L
        )

        val ranging = packet.metadata().rangingSample

        assertFalse(ranging?.valid ?: true)
        assertNull(ranging?.rangeMm)
    }

    @Test(expected = IllegalStateException::class)
    fun missingFrameSequenceFailsClosed() {
        val text = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 1\r\n\r\nX"
        AtomS3rMjpegFrameSource.MjpegPartReader.readPacket(
            BufferedInputStream(ByteArrayInputStream(text.toByteArray(StandardCharsets.US_ASCII))),
            receivedAtNs = 9_000_000L
        ).metadata()
    }

    private fun part(jpeg: ByteArray, tofValid: Boolean): ByteArray {
        val headers = buildString {
            append("--frame\r\n")
            append("Content-Type: image/jpeg\r\n")
            append("Content-Length: ${jpeg.size}\r\n")
            append("X-Sequence-Id: boot-1\r\n")
            append("X-Frame-Sequence: 42\r\n")
            append("X-Capture-Timestamp-Us: 1234\r\n")
            append("X-ToF-Timestamp-Us: 1240\r\n")
            append("X-ToF-Age-At-Jpeg-Ready-Us: 3\r\n")
            append("X-ToF-Valid: $tofValid\r\n")
            append("X-ToF-Range-Mm: 875\r\n\r\n")
        }.toByteArray(StandardCharsets.US_ASCII)
        return headers + jpeg
    }
}
