package com.linnan.blindassist.camera

import android.graphics.Bitmap
import com.linnan.blindassist.vision.RgbaVisionFrame
import com.linnan.blindassist.vision.FrameStamp
import com.linnan.blindassist.vision.RangingSample
import com.linnan.blindassist.vision.ExternalFrameTiming
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

/** An independently owned, tightly packed RGBA frame copied from a [Bitmap]. */
class BitmapRgbaVisionFrame private constructor(
    override val width: Int,
    override val height: Int,
    override val rotationDegrees: Int,
    override val frameStamp: FrameStamp?,
    override val rangingSample: RangingSample?,
    override val externalTiming: ExternalFrameTiming?,
    private val pixels: ByteBuffer
) : RgbaVisionFrame {
    private val closed = AtomicBoolean(false)

    override val buffer: ByteBuffer
        get() = pixels.asReadOnlyBuffer().also { it.rewind() }

    override val rowStride: Int = width * RGBA_BYTES_PER_PIXEL
    override val pixelStride: Int = RGBA_BYTES_PER_PIXEL

    override fun close() {
        closed.compareAndSet(false, true)
    }

    companion object {
        /** Copies [bitmap] so the returned frame remains valid after the bitmap is recycled. */
        fun from(
            bitmap: Bitmap,
            rotationDegrees: Int = 0,
            frameStamp: FrameStamp? = null,
            rangingSample: RangingSample? = null,
            externalTiming: ExternalFrameTiming? = null,
            externalTimingFactory: (() -> ExternalFrameTiming?)? = null
        ): BitmapRgbaVisionFrame {
            require(bitmap.width > 0 && bitmap.height > 0) { "Bitmap must have positive dimensions" }
            val argbPixels = IntArray(bitmap.width * bitmap.height)
            bitmap.getPixels(
                argbPixels,
                0,
                bitmap.width,
                0,
                0,
                bitmap.width,
                bitmap.height
            )
            val rgba = ByteBuffer.allocateDirect(argbPixels.size * RGBA_BYTES_PER_PIXEL)
            argbPixels.forEach { pixel ->
                rgba.put(((pixel shr 16) and 0xFF).toByte())
                rgba.put(((pixel shr 8) and 0xFF).toByte())
                rgba.put((pixel and 0xFF).toByte())
                rgba.put(((pixel ushr 24) and 0xFF).toByte())
            }
            rgba.rewind()
            return BitmapRgbaVisionFrame(
                width = bitmap.width,
                height = bitmap.height,
                rotationDegrees = rotationDegrees,
                frameStamp = frameStamp,
                rangingSample = rangingSample,
                externalTiming = externalTiming ?: externalTimingFactory?.invoke(),
                pixels = rgba
            )
        }

        private const val RGBA_BYTES_PER_PIXEL = 4
    }
}
