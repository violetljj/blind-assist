package com.linnan.blindassist.vision

import java.nio.ByteBuffer

interface VisionFrame : AutoCloseable {
    val width: Int
    val height: Int
    val rotationDegrees: Int

    override fun close()
}

interface RgbaVisionFrame : VisionFrame {
    val buffer: ByteBuffer
    val rowStride: Int
    val pixelStride: Int
}
