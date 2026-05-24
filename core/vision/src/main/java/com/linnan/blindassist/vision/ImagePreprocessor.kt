package com.linnan.blindassist.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Arrays
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
    private val letterboxed: Bitmap by lazy { Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888) }
    private val canvas: Canvas by lazy { Canvas(letterboxed) }
    private val paint: Paint by lazy { Paint(Paint.FILTER_BITMAP_FLAG) }
    private val targetRect: RectF by lazy { RectF() }
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
        canvas.drawColor(BLACK_ARGB)
        canvas.drawBitmap(bitmap, null, targetRect, paint)
        letterboxed.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        writePixelsToBuffer(pixels, inputBuffer)

        return ModelInput(
            buffer = inputBuffer,
            letterbox = letterbox
        )
    }

    fun prepare(frame: RgbaVisionFrame): ModelInput {
        val displayWidth = frame.displayWidth()
        val displayHeight = frame.displayHeight()
        val letterbox = calculateLetterbox(displayWidth, displayHeight, inputSize)
        val resizedWidth = (displayWidth * letterbox.scale).toInt().coerceAtLeast(1)
        val resizedHeight = (displayHeight * letterbox.scale).toInt().coerceAtLeast(1)
        val left = letterbox.dx.toInt()
        val top = letterbox.dy.toInt()
        val source = frame.buffer.duplicate()

        Arrays.fill(pixels, BLACK_ARGB)
        for (y in 0 until resizedHeight) {
            val displayY = (y / letterbox.scale).toInt().coerceIn(0, displayHeight - 1)
            val targetY = top + y
            if (targetY !in 0 until inputSize) continue
            for (x in 0 until resizedWidth) {
                val displayX = (x / letterbox.scale).toInt().coerceIn(0, displayWidth - 1)
                val targetX = left + x
                if (targetX !in 0 until inputSize) continue
                val mapped = mapDisplayToSource(frame, displayX, displayY)
                pixels[targetY * inputSize + targetX] = readRgbaPixel(source, frame, mapped.x, mapped.y)
            }
        }
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

        private fun RgbaVisionFrame.displayWidth(): Int {
            return if (rotationDegrees.normalizedRotation() % 180 == 0) width else height
        }

        private fun RgbaVisionFrame.displayHeight(): Int {
            return if (rotationDegrees.normalizedRotation() % 180 == 0) height else width
        }

        private fun mapDisplayToSource(frame: RgbaVisionFrame, displayX: Int, displayY: Int): SourcePoint {
            return when (frame.rotationDegrees.normalizedRotation()) {
                90 -> SourcePoint(
                    x = displayY.coerceIn(0, frame.width - 1),
                    y = (frame.height - 1 - displayX).coerceIn(0, frame.height - 1)
                )
                180 -> SourcePoint(
                    x = (frame.width - 1 - displayX).coerceIn(0, frame.width - 1),
                    y = (frame.height - 1 - displayY).coerceIn(0, frame.height - 1)
                )
                270 -> SourcePoint(
                    x = (frame.width - 1 - displayY).coerceIn(0, frame.width - 1),
                    y = displayX.coerceIn(0, frame.height - 1)
                )
                else -> SourcePoint(
                    x = displayX.coerceIn(0, frame.width - 1),
                    y = displayY.coerceIn(0, frame.height - 1)
                )
            }
        }

        private fun readRgbaPixel(buffer: ByteBuffer, frame: RgbaVisionFrame, x: Int, y: Int): Int {
            val offset = y * frame.rowStride + x * frame.pixelStride
            if (offset + 2 >= buffer.capacity()) return BLACK_ARGB
            val r = buffer.get(offset).toInt() and 0xFF
            val g = buffer.get(offset + 1).toInt() and 0xFF
            val b = buffer.get(offset + 2).toInt() and 0xFF
            val a = if (offset + 3 < buffer.capacity()) buffer.get(offset + 3).toInt() and 0xFF else 0xFF
            return (a shl 24) or (r shl 16) or (g shl 8) or b
        }

        private fun Int.normalizedRotation(): Int {
            val normalized = this % 360
            return if (normalized < 0) normalized + 360 else normalized
        }

        private const val FLOAT_BYTES = 4
        private const val BLACK_ARGB = -0x1000000
    }
}

private data class SourcePoint(val x: Int, val y: Int)
