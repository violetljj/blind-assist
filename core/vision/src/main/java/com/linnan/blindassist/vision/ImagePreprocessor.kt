package com.linnan.blindassist.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

data class LetterboxInfo(
    val scale: Float,
    val dx: Float,
    val dy: Float,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val inputSize: Int
)

data class ModelInput(
    val buffer: ByteBuffer,
    val letterbox: LetterboxInfo
)

class ImagePreprocessor(private val inputSize: Int) {
    private val letterboxed = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(letterboxed)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val targetRect = RectF()
    private val pixels = IntArray(inputSize * inputSize)
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(1 * inputSize * inputSize * 3 * FLOAT_BYTES)
        .order(ByteOrder.nativeOrder())

    fun prepare(bitmap: Bitmap): ModelInput {
        val letterbox = calculateLetterbox(bitmap.width, bitmap.height, inputSize)
        val resizedWidth = (bitmap.width * letterbox.scale).toInt().coerceAtLeast(1)
        val resizedHeight = (bitmap.height * letterbox.scale).toInt().coerceAtLeast(1)

        targetRect.set(
            letterbox.dx,
            letterbox.dy,
            letterbox.dx + resizedWidth,
            letterbox.dy + resizedHeight
        )
        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(bitmap, null, targetRect, paint)
        letterboxed.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        writePixelsToBuffer(pixels, inputBuffer)

        return ModelInput(
            buffer = inputBuffer,
            letterbox = letterbox
        )
    }

    companion object {
        internal fun calculateLetterbox(
            sourceWidth: Int,
            sourceHeight: Int,
            inputSize: Int
        ): LetterboxInfo {
            val scale = min(
                inputSize.toFloat() / sourceWidth.toFloat(),
                inputSize.toFloat() / sourceHeight.toFloat()
            )
            val resizedWidth = (sourceWidth * scale).toInt().coerceAtLeast(1)
            val resizedHeight = (sourceHeight * scale).toInt().coerceAtLeast(1)
            val dx = (inputSize - resizedWidth) / 2f
            val dy = (inputSize - resizedHeight) / 2f

            return LetterboxInfo(
                scale = scale,
                dx = dx,
                dy = dy,
                sourceWidth = sourceWidth,
                sourceHeight = sourceHeight,
                inputSize = inputSize
            )
        }

        internal fun writePixelsToBuffer(pixels: IntArray, buffer: ByteBuffer) {
            buffer.rewind()
            for (pixel in pixels) {
                buffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
                buffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
                buffer.putFloat((pixel and 0xFF) / 255f)
            }
            buffer.rewind()
        }

        private const val FLOAT_BYTES = 4
    }
}
