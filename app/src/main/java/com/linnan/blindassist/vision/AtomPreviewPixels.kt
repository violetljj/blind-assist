package com.linnan.blindassist.vision

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.pow
import kotlin.math.roundToInt

/** Display-only processing. Input pixels, captured JPEGs and model inputs are never mutated. */
internal class AtomPreviewPixels(private val remap: ShortArray? = null) {
    init { require(remap == null || remap.size == 480 * 640 * 3) }

    val supportsUndistort: Boolean get() = remap != null
    private val tone = IntArray(256) { value ->
        val x = (value / 255.0).pow(0.96)
        // A fixed, monotonic S curve with preserved black/white endpoints; no frame-to-frame adaptation.
        ((x + 0.60 * x * (1 - x) * (2 * x - 1)) * 255).roundToInt().coerceIn(0, 255)
    }

    data class Result(val pixels: IntArray, val width: Int, val height: Int, val isUpright: Boolean)

    fun apply(source: IntArray, width: Int, height: Int, undistort: Boolean, enhance: Boolean): Result {
        require(width > 0 && height > 0 && source.size.toLong() == width.toLong() * height)
        val map = remap?.takeIf { undistort && width == 640 && height == 480 }
        if (map == null) {
            val pixels = if (enhance) IntArray(source.size) { enhancePixel(source[it]) } else source
            return Result(pixels, width, height, false)
        }
        val output = IntArray(480 * 640)
        fun pixel(x: Int, y: Int) = if (x in 0..639 && y in 0..479) source[y * 640 + x] else 0
        for (i in output.indices) {
            val x = map[i * 3].toInt(); val y = map[i * 3 + 1].toInt()
            val fraction = map[i * 3 + 2].toInt() and 1023
            val fx = fraction and 31; val fy = fraction shr 5
            val p0 = pixel(x, y); val p1 = pixel(x + 1, y)
            val p2 = pixel(x, y + 1); val p3 = pixel(x + 1, y + 1)
            val w0 = (32 - fx) * (32 - fy); val w1 = fx * (32 - fy)
            val w2 = (32 - fx) * fy; val w3 = fx * fy
            var color = -0x1000000
            for (shift in 16 downTo 0 step 8) {
                val value = (512 + ((p0 ushr shift) and 255) * w0 + ((p1 ushr shift) and 255) * w1 +
                    ((p2 ushr shift) and 255) * w2 + ((p3 ushr shift) and 255) * w3) shr 10
                color = color or ((if (enhance) tone[value] else value) shl shift)
            }
            output[i] = color
        }
        // Existing calibration packs the raw-to-upright transpose into its remap already.
        return Result(output, 480, 640, true)
    }

    private fun enhancePixel(pixel: Int) = (pixel and -0x1000000) or
        (tone[(pixel ushr 16) and 255] shl 16) or (tone[(pixel ushr 8) and 255] shl 8) or tone[pixel and 255]

    companion object {
        // Screen-checkerboard Atom calibration from 2026-10-12; same map used by MobileFusionCamera.
        const val REMAP_SHA256 = "97d14a4ac5e8be48ceb16cbd1101eb7387e8cc8c10a4fef558d3139c27f07c63"
        fun fromRemap(bytes: ByteArray): AtomPreviewPixels {
            require(bytes.size == 480 * 640 * 6) { "Unexpected Atom preview remap size" }
            val hash = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
            require(hash == REMAP_SHA256) { "Atom preview calibration identity mismatch" }
            val map = ShortArray(480 * 640 * 3)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(map)
            return AtomPreviewPixels(map)
        }
    }
}
