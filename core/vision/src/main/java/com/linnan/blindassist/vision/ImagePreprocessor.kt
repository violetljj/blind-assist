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
    private val normalizedPixels = FloatArray(inputSize * inputSize * MODEL_CHANNELS)
    private val xByteOffsets = IntArray(inputSize)
    private val yByteOffsets = IntArray(inputSize)
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
        val source = frame.buffer.duplicate().order(ByteOrder.nativeOrder())
        val rotation = frame.rotationDegrees.normalizedRotation()
        val baseOffset = sourceBaseOffset(frame, rotation)

        Arrays.fill(normalizedPixels, 0f)
        for (x in 0 until resizedWidth) {
            val displayX = (x / letterbox.scale).toInt().coerceIn(0, displayWidth - 1)
            xByteOffsets[x] = when (rotation) {
                90 -> -displayX * frame.rowStride
                180 -> -displayX * frame.pixelStride
                270 -> displayX * frame.rowStride
                else -> displayX * frame.pixelStride
            }
        }
        for (y in 0 until resizedHeight) {
            val displayY = (y / letterbox.scale).toInt().coerceIn(0, displayHeight - 1)
            yByteOffsets[y] = when (rotation) {
                90 -> displayY * frame.pixelStride
                180 -> -displayY * frame.rowStride
                270 -> -displayY * frame.pixelStride
                else -> displayY * frame.rowStride
            }
        }
        for (y in 0 until resizedHeight) {
            val targetY = top + y
            if (targetY !in 0 until inputSize) continue
            val rowOffset = baseOffset + yByteOffsets[y]
            val targetRowOffset = (targetY * inputSize + left) * MODEL_CHANNELS
            for (x in 0 until resizedWidth) {
                val targetX = left + x
                if (targetX !in 0 until inputSize) continue
                writeNormalizedRgbaPixel(
                    source,
                    rowOffset + xByteOffsets[x],
                    targetRowOffset + x * MODEL_CHANNELS
                )
            }
        }
        inputBuffer.asFloatBuffer().apply {
            rewind()
            put(normalizedPixels)
            rewind()
        }
        inputBuffer.rewind()

        return ModelInput(
            buffer = inputBuffer,
            letterbox = letterbox
        )
    }

    private fun writeNormalizedRgbaPixel(
        buffer: ByteBuffer,
        offset: Int,
        targetOffset: Int
    ) {
        if (offset < 0 || offset + 2 >= buffer.limit()) return
        if (offset + 3 < buffer.limit()) {
            val rgba = buffer.getInt(offset)
            if (ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN) {
                normalizedPixels[targetOffset] = (rgba and 0xFF) / 255f
                normalizedPixels[targetOffset + 1] = ((rgba ushr 8) and 0xFF) / 255f
                normalizedPixels[targetOffset + 2] = ((rgba ushr 16) and 0xFF) / 255f
            } else {
                normalizedPixels[targetOffset] = ((rgba ushr 24) and 0xFF) / 255f
                normalizedPixels[targetOffset + 1] = ((rgba ushr 16) and 0xFF) / 255f
                normalizedPixels[targetOffset + 2] = ((rgba ushr 8) and 0xFF) / 255f
            }
            return
        }
        normalizedPixels[targetOffset] = (buffer.get(offset).toInt() and 0xFF) / 255f
        normalizedPixels[targetOffset + 1] = (buffer.get(offset + 1).toInt() and 0xFF) / 255f
        normalizedPixels[targetOffset + 2] = (buffer.get(offset + 2).toInt() and 0xFF) / 255f
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

        private fun sourceBaseOffset(frame: RgbaVisionFrame, rotation: Int): Int {
            return when (rotation) {
                90 -> (frame.height - 1) * frame.rowStride
                180 -> {
                    (frame.height - 1) * frame.rowStride +
                        (frame.width - 1) * frame.pixelStride
                }
                270 -> (frame.width - 1) * frame.pixelStride
                else -> 0
            }
        }

        private fun Int.normalizedRotation(): Int {
            val normalized = this % 360
            return if (normalized < 0) normalized + 360 else normalized
        }

        private const val FLOAT_BYTES = 4
        private const val MODEL_CHANNELS = 3
        private const val BLACK_ARGB = -0x1000000
    }
}
