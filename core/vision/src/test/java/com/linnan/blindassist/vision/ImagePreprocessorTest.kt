package com.linnan.blindassist.vision

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ImagePreprocessorTest {
    @Test
    fun landscapeLetterboxKeepsSourceSizeAndVerticalPadding() {
        val letterbox = ImagePreprocessor.calculateLetterbox(
            sourceWidth = 640,
            sourceHeight = 480,
            inputSize = 320
        )

        assertEquals(0.5f, letterbox.scale, 0.0001f)
        assertEquals(0f, letterbox.dx, 0.0001f)
        assertEquals(40f, letterbox.dy, 0.0001f)
        assertEquals(640, letterbox.sourceWidth)
        assertEquals(480, letterbox.sourceHeight)
        assertEquals(320, letterbox.inputSize)
    }

    @Test
    fun portraitLetterboxKeepsSourceSizeAndHorizontalPadding() {
        val letterbox = ImagePreprocessor.calculateLetterbox(
            sourceWidth = 480,
            sourceHeight = 640,
            inputSize = 320
        )

        assertEquals(0.5f, letterbox.scale, 0.0001f)
        assertEquals(40f, letterbox.dx, 0.0001f)
        assertEquals(0f, letterbox.dy, 0.0001f)
        assertEquals(480, letterbox.sourceWidth)
        assertEquals(640, letterbox.sourceHeight)
        assertEquals(320, letterbox.inputSize)
    }

    @Test
    fun writePixelsToBufferRewindsAndOverwritesPreviousContents() {
        val buffer = ByteBuffer
            .allocateDirect(2 * 3 * FLOAT_BYTES)
            .order(ByteOrder.nativeOrder())

        ImagePreprocessor.writePixelsToBuffer(
            intArrayOf(ARGB_RED, ARGB_GREEN),
            buffer
        )
        assertEquals(0, buffer.position())
        assertArrayEquals(
            floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f),
            buffer.asFloatBuffer().toFloatArray(),
            0.0001f
        )

        ImagePreprocessor.writePixelsToBuffer(
            intArrayOf(ARGB_BLUE, ARGB_BLACK),
            buffer
        )
        assertEquals(0, buffer.position())
        assertArrayEquals(
            floatArrayOf(0f, 0f, 1f, 0f, 0f, 0f),
            buffer.asFloatBuffer().toFloatArray(),
            0.0001f
        )
    }

    private fun java.nio.FloatBuffer.toFloatArray(): FloatArray {
        val values = FloatArray(remaining())
        get(values)
        return values
    }

    companion object {
        private const val FLOAT_BYTES = 4
        private const val ARGB_RED = -0x10000
        private const val ARGB_GREEN = -0xff0100
        private const val ARGB_BLUE = -0xffff01
        private const val ARGB_BLACK = -0x1000000
    }
}
