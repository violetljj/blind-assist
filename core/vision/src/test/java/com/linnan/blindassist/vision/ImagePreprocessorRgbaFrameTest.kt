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

    @Test
    fun applies180DegreeRotationWithoutChangingPixelValues() {
        val input = ImagePreprocessor(inputSize = 2).prepare(colourGridFrame(rotationDegrees = 180))
        val floats = FloatArray(12)
        input.buffer.asFloatBuffer().get(floats)

        assertArrayEquals(
            floatArrayOf(
                1f, 1f, 1f,
                0f, 0f, 1f,
                0f, 1f, 0f,
                1f, 0f, 0f
            ),
            floats,
            0.001f
        )
    }

    @Test
    fun applies270DegreeRotationWithoutChangingPixelValues() {
        val input = ImagePreprocessor(inputSize = 2).prepare(colourGridFrame(rotationDegrees = 270))
        val floats = FloatArray(12)
        input.buffer.asFloatBuffer().get(floats)

        assertArrayEquals(
            floatArrayOf(
                0f, 1f, 0f,
                1f, 1f, 1f,
                1f, 0f, 0f,
                0f, 0f, 1f
            ),
            floats,
            0.001f
        )
    }

    @Test
    fun optimizedSamplerMatchesReferenceAcrossRotationsAndPaddedRows() {
        val width = 7
        val height = 5
        val rowStride = width * 4 + 12
        val source = ByteBuffer.allocate(rowStride * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val offset = y * rowStride + x * 4
                source.put(offset, (x * 31 + y * 7).toByte())
                source.put(offset + 1, (x * 13 + y * 29).toByte())
                source.put(offset + 2, (x * 19 + y * 17).toByte())
                source.put(offset + 3, 255.toByte())
            }
        }

        for (rotation in listOf(0, 90, 180, 270)) {
            val frame = FakeRgbaFrame(
                width = width,
                height = height,
                rotationDegrees = rotation,
                rowStride = rowStride,
                pixelStride = 4,
                buffer = source
            )
            val input = ImagePreprocessor(inputSize = 6).prepare(frame)
            val actual = FloatArray(6 * 6 * 3)
            input.buffer.asFloatBuffer().get(actual)

            assertArrayEquals(
                "rotation=$rotation",
                referencePrepare(frame, inputSize = 6),
                actual,
                0.0001f
            )
        }
    }

    @Test
    fun truncatedBufferLeavesUnavailablePixelsBlackWithoutThrowing() {
        val source = rgbaBuffer(
            rowStride = 8,
            rows = arrayOf(
                intArrayOf(255, 0, 0, 255, 0, 255, 0, 255)
            )
        ).also { it.limit(5) }
        val frame = FakeRgbaFrame(
            width = 2,
            height = 1,
            rotationDegrees = 0,
            rowStride = 8,
            pixelStride = 4,
            buffer = source
        )

        val input = ImagePreprocessor(inputSize = 2).prepare(frame)
        val actual = FloatArray(12)
        input.buffer.asFloatBuffer().get(actual)

        assertArrayEquals(
            floatArrayOf(
                1f, 0f, 0f,
                0f, 0f, 0f,
                0f, 0f, 0f,
                0f, 0f, 0f
            ),
            actual,
            0.0001f
        )
    }

    @Test
    fun cachedPlansDoNotLeakPreviousLetterboxPixels() {
        val preprocessor = ImagePreprocessor(inputSize = 4)
        preprocessor.prepare(solidFrame(width = 4, height = 2, red = 255, green = 0))
        val secondFrame = solidFrame(width = 2, height = 4, red = 0, green = 255)

        val input = preprocessor.prepare(secondFrame)
        val actual = FloatArray(4 * 4 * 3)
        input.buffer.asFloatBuffer().get(actual)

        assertArrayEquals(referencePrepare(secondFrame, inputSize = 4), actual, 0.0001f)
    }

    private fun referencePrepare(frame: FakeRgbaFrame, inputSize: Int): FloatArray {
        val rotation = ((frame.rotationDegrees % 360) + 360) % 360
        val displayWidth = if (rotation % 180 == 0) frame.width else frame.height
        val displayHeight = if (rotation % 180 == 0) frame.height else frame.width
        val letterbox = ImagePreprocessor.calculateLetterbox(displayWidth, displayHeight, inputSize)
        val resizedWidth = (displayWidth * letterbox.scale).toInt().coerceAtLeast(1)
        val resizedHeight = (displayHeight * letterbox.scale).toInt().coerceAtLeast(1)
        val left = letterbox.dx.toInt()
        val top = letterbox.dy.toInt()
        val output = FloatArray(inputSize * inputSize * 3)
        val source = frame.buffer.duplicate()

        for (y in 0 until resizedHeight) {
            val displayY = (y / letterbox.scale).toInt().coerceIn(0, displayHeight - 1)
            val targetY = top + y
            if (targetY !in 0 until inputSize) continue
            for (x in 0 until resizedWidth) {
                val displayX = (x / letterbox.scale).toInt().coerceIn(0, displayWidth - 1)
                val targetX = left + x
                if (targetX !in 0 until inputSize) continue
                val sourceX: Int
                val sourceY: Int
                when (rotation) {
                    90 -> {
                        sourceX = displayY.coerceIn(0, frame.width - 1)
                        sourceY = (frame.height - 1 - displayX).coerceIn(0, frame.height - 1)
                    }
                    180 -> {
                        sourceX = (frame.width - 1 - displayX).coerceIn(0, frame.width - 1)
                        sourceY = (frame.height - 1 - displayY).coerceIn(0, frame.height - 1)
                    }
                    270 -> {
                        sourceX = (frame.width - 1 - displayY).coerceIn(0, frame.width - 1)
                        sourceY = displayX.coerceIn(0, frame.height - 1)
                    }
                    else -> {
                        sourceX = displayX.coerceIn(0, frame.width - 1)
                        sourceY = displayY.coerceIn(0, frame.height - 1)
                    }
                }
                val sourceOffset = sourceY * frame.rowStride + sourceX * frame.pixelStride
                val targetOffset = (targetY * inputSize + targetX) * 3
                output[targetOffset] = (source.get(sourceOffset).toInt() and 0xFF) / 255f
                output[targetOffset + 1] = (source.get(sourceOffset + 1).toInt() and 0xFF) / 255f
                output[targetOffset + 2] = (source.get(sourceOffset + 2).toInt() and 0xFF) / 255f
            }
        }
        return output
    }

    private fun colourGridFrame(rotationDegrees: Int): FakeRgbaFrame {
        return FakeRgbaFrame(
            width = 2,
            height = 2,
            rotationDegrees = rotationDegrees,
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
    }

    private fun solidFrame(
        width: Int,
        height: Int,
        red: Int,
        green: Int
    ): FakeRgbaFrame {
        val rowStride = width * 4
        val rows = Array(height) {
            IntArray(rowStride).also { row ->
                for (x in 0 until width) {
                    val offset = x * 4
                    row[offset] = red
                    row[offset + 1] = green
                    row[offset + 2] = 0
                    row[offset + 3] = 255
                }
            }
        }
        return FakeRgbaFrame(
            width = width,
            height = height,
            rotationDegrees = 0,
            rowStride = rowStride,
            pixelStride = 4,
            buffer = rgbaBuffer(rowStride, rows)
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
