package com.linnan.blindassist.vision

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.ByteBuffer

class ImagePreprocessorRgbaFrameTest {
    @Test
    fun preparesRgbaFrameWithStrideWithoutConsumingSourceBuffer() {
        val source = rgbaBuffer(
            rowStride = 12,
            rows = arrayOf(
                intArrayOf(255, 0, 0, 255, 0, 255, 0, 255),
                intArrayOf(0, 0, 255, 255, 255, 255, 255, 255)
            )
        )
        val frame = FakeRgbaFrame(
            width = 2,
            height = 2,
            rotationDegrees = 0,
            rowStride = 12,
            pixelStride = 4,
            buffer = source
        )

        val input = ImagePreprocessor(inputSize = 2).prepare(frame)
        val floats = FloatArray(12)
        input.buffer.asFloatBuffer().get(floats)

        assertArrayEquals(
            floatArrayOf(
                1f, 0f, 0f,
                0f, 1f, 0f,
                0f, 0f, 1f,
                1f, 1f, 1f
            ),
            floats,
            0.001f
        )
        assertEquals(0, input.buffer.position())
        assertEquals(0, source.position())
    }

    @Test
    fun appliesRotationWhileWritingRgbaFrameToModelBuffer() {
        val frame = FakeRgbaFrame(
            width = 2,
            height = 2,
            rotationDegrees = 90,
            rowStride = 8,
            pixelStride = 4,
            buffer = rgbaBuffer(
                rowStride = 8,
                rows = arrayOf(
                    intArrayOf(255, 0, 0, 255, 0, 255, 0, 255),
                    intArrayOf(0, 0, 255, 255, 255, 255, 255, 255)
                )
            )
        )

        val input = ImagePreprocessor(inputSize = 2).prepare(frame)
        val floats = FloatArray(12)
        input.buffer.asFloatBuffer().get(floats)

        assertArrayEquals(
            floatArrayOf(
                0f, 0f, 1f,
                1f, 0f, 0f,
                1f, 1f, 1f,
                0f, 1f, 0f
            ),
            floats,
            0.001f
        )
    }

    private fun rgbaBuffer(rowStride: Int, rows: Array<IntArray>): ByteBuffer {
        val buffer = ByteBuffer.allocate(rowStride * rows.size)
        rows.forEachIndexed { y, row ->
            for (i in row.indices) {
                buffer.put(y * rowStride + i, row[i].toByte())
            }
        }
        return buffer
    }

    private class FakeRgbaFrame(
        override val width: Int,
        override val height: Int,
        override val rotationDegrees: Int,
        override val rowStride: Int,
        override val pixelStride: Int,
        override val buffer: ByteBuffer
    ) : RgbaVisionFrame {
        override fun close() = Unit
    }
}
