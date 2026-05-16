package com.linnan.blindassist.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
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

object ImagePreprocessor {
    fun prepare(bitmap: Bitmap, inputSize: Int): ModelInput {
        val scale = min(
            inputSize.toFloat() / bitmap.width.toFloat(),
            inputSize.toFloat() / bitmap.height.toFloat()
        )
        val resizedWidth = (bitmap.width * scale).toInt().coerceAtLeast(1)
        val resizedHeight = (bitmap.height * scale).toInt().coerceAtLeast(1)
        val dx = (inputSize - resizedWidth) / 2f
        val dy = (inputSize - resizedHeight) / 2f

        val resized = Bitmap.createScaledBitmap(bitmap, resizedWidth, resizedHeight, true)
        val letterboxed = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(letterboxed)
        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(resized, dx, dy, Paint(Paint.FILTER_BITMAP_FLAG))
        if (resized !== bitmap) resized.recycle()

        val pixels = IntArray(inputSize * inputSize)
        letterboxed.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        letterboxed.recycle()

        val buffer = ByteBuffer
            .allocateDirect(1 * inputSize * inputSize * 3 * FLOAT_BYTES)
            .order(ByteOrder.nativeOrder())
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
            buffer.putFloat((pixel and 0xFF) / 255f)
        }
        buffer.rewind()

        return ModelInput(
            buffer = buffer,
            letterbox = LetterboxInfo(
                scale = scale,
                dx = dx,
                dy = dy,
                sourceWidth = bitmap.width,
                sourceHeight = bitmap.height,
                inputSize = inputSize
            )
        )
    }

    private const val FLOAT_BYTES = 4
}
