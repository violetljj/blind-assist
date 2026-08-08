package com.linnan.blindassist.hftf.metricdepth

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.fail
import org.junit.Test
import java.nio.ByteBuffer

class D45Yuv420ToRgbaDecoderTest {
    @Test
    fun paddedPlanesAndNonzeroBufferPositionsDecodeBlackAndWhitePixels() {
        val yBuffer = ByteBuffer.wrap(
            bytes(
                99,
                16, 0, 235, 0, 0,
                235, 0, 16, 0
            )
        ).apply { position(1) }
        val uBuffer = ByteBuffer.wrap(bytes(77, 128, 0)).apply { position(1) }
        val vBuffer = ByteBuffer.wrap(bytes(66, 128, 0)).apply { position(1) }

        val rgba = D45Yuv420ToRgbaDecoder.decode(
            D45Yuv420Image(
                widthPx = 2,
                heightPx = 2,
                y = D45YuvBytePlane(5, 2, yBuffer),
                u = D45YuvBytePlane(2, 1, uBuffer),
                v = D45YuvBytePlane(2, 1, vBuffer)
            )
        )

        assertArrayEquals(
            byteArrayOf(
                0, 0, 0, -1,
                -1, -1, -1, -1,
                -1, -1, -1, -1,
                0, 0, 0, -1
            ),
            rgba.bytes
        )
    }

    @Test
    fun interleavedChromaPixelStrideIsRespected() {
        val rgba = D45Yuv420ToRgbaDecoder.decode(
            D45Yuv420Image(
                widthPx = 3,
                heightPx = 1,
                y = plane(bytes(81, 81, 81), rowStride = 3),
                u = plane(bytes(90, 0, 240), rowStride = 3, pixelStride = 2),
                v = plane(bytes(240, 0, 110), rowStride = 3, pixelStride = 2)
            )
        )

        assertEquals(12, rgba.bytes.size)
        assertEquals(255, rgba.bytes[3].toInt() and 0xFF)
        assertEquals(255, rgba.bytes[11].toInt() and 0xFF)
        assertEquals(false, rgba.bytes.copyOfRange(0, 3).contentEquals(
            rgba.bytes.copyOfRange(8, 11)
        ))
    }

    @Test
    fun truncatedPlaneFailsBeforePartialRgbaCanEscape() {
        try {
            D45Yuv420ToRgbaDecoder.decode(
                D45Yuv420Image(
                    widthPx = 2,
                    heightPx = 2,
                    y = plane(bytes(16, 16, 16), rowStride = 2),
                    u = plane(bytes(128), rowStride = 1),
                    v = plane(bytes(128), rowStride = 1)
                )
            )
            fail("truncated Y plane must fail")
        } catch (error: IllegalArgumentException) {
            assertEquals(
                "Y plane buffer is shorter than its declared dimensions and strides",
                error.message
            )
        }
    }

    private fun plane(
        bytes: ByteArray,
        rowStride: Int,
        pixelStride: Int = 1
    ) = D45YuvBytePlane(
        rowStrideBytes = rowStride,
        pixelStrideBytes = pixelStride,
        buffer = ByteBuffer.wrap(bytes)
    )

    private fun bytes(vararg values: Int) = ByteArray(values.size) { index ->
        values[index].toByte()
    }
}
