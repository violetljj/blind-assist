package com.linnan.blindassist.vision

import java.nio.ByteBuffer

/**
 * Produces a canonical-upright, nearest-neighbour Luma plane from the same
 * RGBA_8888 CameraX frame used by the production YOLO preprocessor.
 *
 * This is only a transport primitive for the isolated corridor-motion
 * candidate; it has no alert semantics and is not wired into production.
 */
class RgbaLumaResampler(
    private val outputSize: Int = DEFAULT_OUTPUT_SIZE,
    private val mode: Mode = Mode.WEIGHTED_RGB
) {
    init {
        require(outputSize > 0) { "outputSize must be positive" }
    }

    fun sample(frame: RgbaVisionFrame): ByteArray {
        require(frame.width > 0 && frame.height > 0) { "frame dimensions must be positive" }
        require(frame.pixelStride >= 3) { "RGBA frame requires at least three bytes per pixel" }
        val displayWidth = if (frame.rotationDegrees.normalizedRotation() % 180 == 0) frame.width else frame.height
        val displayHeight = if (frame.rotationDegrees.normalizedRotation() % 180 == 0) frame.height else frame.width
        val source = frame.buffer.duplicate().also { it.rewind() }
        return ByteArray(outputSize * outputSize).also { target ->
            when (frame.rotationDegrees.normalizedRotation()) {
                90 -> for (targetY in 0 until outputSize) {
                    val sourceX = (targetY * displayHeight / outputSize).coerceIn(0, frame.width - 1)
                    for (targetX in 0 until outputSize) {
                        val displayX = (targetX * displayWidth / outputSize).coerceIn(0, frame.height - 1)
                        target[targetY * outputSize + targetX] = lumaAt(source, frame, sourceX, frame.height - 1 - displayX).toByte()
                    }
                }
                180 -> for (targetY in 0 until outputSize) {
                    val displayY = (targetY * displayHeight / outputSize).coerceIn(0, frame.height - 1)
                    for (targetX in 0 until outputSize) {
                        val displayX = (targetX * displayWidth / outputSize).coerceIn(0, frame.width - 1)
                        target[targetY * outputSize + targetX] = lumaAt(source, frame, frame.width - 1 - displayX, frame.height - 1 - displayY).toByte()
                    }
                }
                270 -> for (targetY in 0 until outputSize) {
                    val sourceX = (frame.width - 1 - targetY * displayHeight / outputSize).coerceIn(0, frame.width - 1)
                    for (targetX in 0 until outputSize) {
                        val sourceY = (targetX * displayWidth / outputSize).coerceIn(0, frame.height - 1)
                        target[targetY * outputSize + targetX] = lumaAt(source, frame, sourceX, sourceY).toByte()
                    }
                }
                else -> for (targetY in 0 until outputSize) {
                    val sourceY = (targetY * displayHeight / outputSize).coerceIn(0, frame.height - 1)
                    for (targetX in 0 until outputSize) {
                        val sourceX = (targetX * displayWidth / outputSize).coerceIn(0, frame.width - 1)
                        target[targetY * outputSize + targetX] = lumaAt(source, frame, sourceX, sourceY).toByte()
                    }
                }
            }
        }
    }

    private fun lumaAt(buffer: ByteBuffer, frame: RgbaVisionFrame, x: Int, y: Int): Int {
        val offset = y * frame.rowStride + x * frame.pixelStride
        if (offset + 2 >= buffer.capacity()) return 0
        if (mode == Mode.GREEN_CHANNEL) return buffer.get(offset + 1).toInt() and 0xFF
        val red = buffer.get(offset).toInt() and 0xFF
        val green = buffer.get(offset + 1).toInt() and 0xFF
        val blue = buffer.get(offset + 2).toInt() and 0xFF
        return (77 * red + 150 * green + 29 * blue) ushr 8
    }

    private fun Int.normalizedRotation(): Int {
        val normalized = this % 360
        return if (normalized < 0) normalized + 360 else normalized
    }

    private companion object {
        const val DEFAULT_OUTPUT_SIZE = 320
    }

    /** Experimental transport choice; production callers retain [WEIGHTED_RGB]. */
    enum class Mode {
        WEIGHTED_RGB,
        GREEN_CHANNEL
    }
}
