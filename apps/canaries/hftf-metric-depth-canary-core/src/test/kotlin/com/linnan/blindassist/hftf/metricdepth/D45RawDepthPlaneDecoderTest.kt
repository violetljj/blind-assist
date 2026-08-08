package com.linnan.blindassist.hftf.metricdepth

import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class D45RawDepthPlaneDecoderTest {
    @Test
    fun paddedStridedPlanesDecodeUnsignedDepthAndConfidence() {
        val depthBuffer = ByteBuffer.allocate(18).order(ByteOrder.LITTLE_ENDIAN).apply {
            position(1)
            putShort(1, 1_000.toShort())
            putShort(3, 65_535.toShort())
            putShort(5, 0)
            putShort(9, 250.toShort())
            putShort(11, 2_000.toShort())
            putShort(13, 4_000.toShort())
            position(1)
        }
        val confidenceBuffer = ByteBuffer.allocate(12).apply {
            position(2)
            put(2, 255.toByte())
            put(3, 128.toByte())
            put(4, 0)
            put(6, 64.toByte())
            put(7, 192.toByte())
            put(8, 255.toByte())
            position(2)
        }

        val decoded = D45RawDepthPlaneDecoder.decode(
            depth = D45StridedBytePlane(3, 2, 8, 2, depthBuffer),
            confidence = D45StridedBytePlane(3, 2, 4, 1, confidenceBuffer)
        )

        assertArrayEquals(intArrayOf(1_000, 65_535, 0, 250, 2_000, 4_000), decoded.depthMillimeters)
        assertArrayEquals(
            floatArrayOf(1f, 128f / 255f, 0f, 64f / 255f, 192f / 255f, 1f),
            decoded.confidence,
            0.000001f
        )
    }

    @Test
    fun truncatedPlaneFailsBeforeAnyPartialRasterCanEscape() {
        val truncatedDepth = ByteBuffer.allocate(10)
        val confidence = ByteBuffer.allocate(6)

        assertThrows(IllegalArgumentException::class.java) {
            D45RawDepthPlaneDecoder.decode(
                depth = D45StridedBytePlane(3, 2, 6, 2, truncatedDepth),
                confidence = D45StridedBytePlane(3, 2, 3, 1, confidence)
            )
        }
    }

    @Test
    fun decodedFrameRemainsExplicitlyUnregistered() {
        val observation = D45UnregisteredRawMetricDepthFrame(
            sourceFrame = FrameStamp(
                frameId = 3L,
                capturedAtNs = 1_000L,
                receivedAtNs = 1_100L,
                sourceId = "arcore:camera0",
                coordinateFrame = "arcore:camera-image",
                clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
            ),
            sourceImageIntrinsics = MetricDepthCameraIntrinsics(
                imageWidthPx = 640,
                imageHeightPx = 480,
                focalXpx = 500f,
                focalYpx = 500f,
                principalXpx = 320f,
                principalYpx = 240f
            ),
            raster = D45DecodedRawDepthRaster(
                widthPx = 1,
                heightPx = 1,
                depthMillimeters = intArrayOf(2_000),
                confidence = floatArrayOf(0.9f)
            ),
            depthTimestampNs = 1_000L,
            confidenceTimestampNs = 1_000L,
            producedAtNs = 1_200L
        )

        assertEquals(
            D45RawDepthRegistrationState.SOURCE_REGISTRATION_UNVERIFIED,
            observation.registrationState
        )
    }
}
