package com.linnan.blindassist.vision

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.ByteBuffer

class RgbaLumaResamplerTest {
    @Test
    fun samplesRgbaWithStrideWithoutChangingSourcePosition() {
        val source = ByteBuffer.allocate(12).apply {
            put(0, 255.toByte()); put(1, 0.toByte()); put(2, 0.toByte()); put(3, 255.toByte())
            put(4, 0.toByte()); put(5, 255.toByte()); put(6, 0.toByte()); put(7, 255.toByte())
        }
        val result = RgbaLumaResampler(outputSize = 2).sample(FakeFrame(2, 1, 0, 12, 4, source))
        assertArrayEquals(byteArrayOf(76.toByte(), 149.toByte(), 76.toByte(), 149.toByte()), result)
        assertEquals(0, source.position())
    }

    @Test
    fun rotatesIntoCanonicalUprightCoordinates() {
        val source = rgbaBuffer(
            intArrayOf(255, 0, 0, 255, 0, 255, 0, 255),
            intArrayOf(0, 0, 255, 255, 255, 255, 255, 255)
        )
        val result = RgbaLumaResampler(outputSize = 2).sample(FakeFrame(2, 2, 90, 8, 4, source))
        assertArrayEquals(byteArrayOf(28.toByte(), 76.toByte(), 255.toByte(), 149.toByte()), result)
    }

    @Test
    fun preservesAllQuarterTurnMappings() {
        val source = rgbaBuffer(
            intArrayOf(255, 0, 0, 255, 0, 255, 0, 255),
            intArrayOf(0, 0, 255, 255, 255, 255, 255, 255)
        )
        assertArrayEquals(
            byteArrayOf(255.toByte(), 28.toByte(), 149.toByte(), 76.toByte()),
            RgbaLumaResampler(outputSize = 2).sample(FakeFrame(2, 2, 180, 8, 4, source))
        )
        assertArrayEquals(
            byteArrayOf(149.toByte(), 255.toByte(), 76.toByte(), 28.toByte()),
            RgbaLumaResampler(outputSize = 2).sample(FakeFrame(2, 2, 270, 8, 4, source))
        )
    }

    @Test
    fun greenChannelModeUsesOnlyGreenValue() {
        val source = ByteBuffer.allocate(4).apply {
            put(0, 255.toByte()); put(1, 13); put(2, 0); put(3, 255.toByte())
        }
        val result = RgbaLumaResampler(
            outputSize = 1,
            mode = RgbaLumaResampler.Mode.GREEN_CHANNEL
        ).sample(FakeFrame(1, 1, 0, 4, 4, source))
        assertArrayEquals(byteArrayOf(13), result)
    }

    private fun rgbaBuffer(vararg rows: IntArray): ByteBuffer = ByteBuffer.allocate(rows.size * 8).also { buffer ->
        rows.forEachIndexed { rowIndex, row -> row.forEachIndexed { index, value -> buffer.put(rowIndex * 8 + index, value.toByte()) } }
    }

    private class FakeFrame(
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
