package com.linnan.blindassist.hftf.metricdepth

import java.nio.ByteBuffer

data class D45YuvBytePlane(
    val rowStrideBytes: Int,
    val pixelStrideBytes: Int,
    val buffer: ByteBuffer
) {
    init {
        require(rowStrideBytes > 0 && pixelStrideBytes > 0)
    }
}

data class D45Yuv420Image(
    val widthPx: Int,
    val heightPx: Int,
    val y: D45YuvBytePlane,
    val u: D45YuvBytePlane,
    val v: D45YuvBytePlane
) {
    init {
        require(widthPx > 0 && heightPx > 0)
    }
}

data class D45RgbaRaster(
    val widthPx: Int,
    val heightPx: Int,
    /** RGBA bytes, four tightly packed bytes per pixel. */
    val bytes: ByteArray
) {
    init {
        require(bytes.size == widthPx * heightPx * RGBA_CHANNELS)
    }

    private companion object {
        const val RGBA_CHANNELS = 4
    }
}

/**
 * Stride-safe YUV_420_888 -> RGBA conversion for the isolated ARCore detector path.
 *
 * Uses the integer BT.601 limited-range conversion used by common Android camera pipelines.
 */
object D45Yuv420ToRgbaDecoder {
    fun decode(image: D45Yuv420Image): D45RgbaRaster {
        requireCapacity(image.y, image.widthPx, image.heightPx, "Y")
        val chromaWidth = (image.widthPx + 1) / 2
        val chromaHeight = (image.heightPx + 1) / 2
        requireCapacity(image.u, chromaWidth, chromaHeight, "U")
        requireCapacity(image.v, chromaWidth, chromaHeight, "V")

        val yBuffer = image.y.buffer.duplicate()
        val uBuffer = image.u.buffer.duplicate()
        val vBuffer = image.v.buffer.duplicate()
        val yBase = yBuffer.position()
        val uBase = uBuffer.position()
        val vBase = vBuffer.position()
        val output = ByteArray(image.widthPx * image.heightPx * RGBA_CHANNELS)
        for (row in 0 until image.heightPx) {
            for (column in 0 until image.widthPx) {
                val yValue = unsigned(
                    yBuffer.get(
                        yBase +
                            row * image.y.rowStrideBytes +
                            column * image.y.pixelStrideBytes
                    )
                )
                val chromaRow = row / 2
                val chromaColumn = column / 2
                val uValue = unsigned(
                    uBuffer.get(
                        uBase +
                            chromaRow * image.u.rowStrideBytes +
                            chromaColumn * image.u.pixelStrideBytes
                    )
                )
                val vValue = unsigned(
                    vBuffer.get(
                        vBase +
                            chromaRow * image.v.rowStrideBytes +
                            chromaColumn * image.v.pixelStrideBytes
                    )
                )
                val c = (yValue - Y_OFFSET).coerceAtLeast(0)
                val d = uValue - CHROMA_OFFSET
                val e = vValue - CHROMA_OFFSET
                val red = clamp8((Y_SCALE * c + RED_V_SCALE * e + ROUNDING) shr SHIFT)
                val green = clamp8(
                    (
                        Y_SCALE * c -
                            GREEN_U_SCALE * d -
                            GREEN_V_SCALE * e +
                            ROUNDING
                        ) shr SHIFT
                )
                val blue = clamp8((Y_SCALE * c + BLUE_U_SCALE * d + ROUNDING) shr SHIFT)
                val outputOffset = (row * image.widthPx + column) * RGBA_CHANNELS
                output[outputOffset] = red.toByte()
                output[outputOffset + 1] = green.toByte()
                output[outputOffset + 2] = blue.toByte()
                output[outputOffset + 3] = 0xFF.toByte()
            }
        }
        return D45RgbaRaster(image.widthPx, image.heightPx, output)
    }

    private fun requireCapacity(
        plane: D45YuvBytePlane,
        widthPx: Int,
        heightPx: Int,
        label: String
    ) {
        val lastExclusive =
            plane.buffer.position().toLong() +
                (heightPx - 1L) * plane.rowStrideBytes +
                (widthPx - 1L) * plane.pixelStrideBytes +
                1L
        require(lastExclusive <= plane.buffer.limit()) {
            "$label plane buffer is shorter than its declared dimensions and strides"
        }
        require(plane.rowStrideBytes >= (widthPx - 1) * plane.pixelStrideBytes + 1) {
            "$label row stride truncates its last logical sample"
        }
    }

    private fun unsigned(value: Byte): Int = value.toInt() and 0xFF

    private fun clamp8(value: Int): Int = value.coerceIn(0, 255)

    private const val RGBA_CHANNELS = 4
    private const val Y_OFFSET = 16
    private const val CHROMA_OFFSET = 128
    private const val Y_SCALE = 298
    private const val RED_V_SCALE = 409
    private const val GREEN_U_SCALE = 100
    private const val GREEN_V_SCALE = 208
    private const val BLUE_U_SCALE = 516
    private const val ROUNDING = 128
    private const val SHIFT = 8
}
