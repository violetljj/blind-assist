package com.linnan.blindassist.camera

import androidx.camera.core.ImageProxy
import com.linnan.blindassist.vision.FrameStamp
import com.linnan.blindassist.vision.RgbaVisionFrame
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

internal class ImageProxyVisionFrame(
    private val imageProxy: ImageProxy,
    override val frameStamp: FrameStamp
) : RgbaVisionFrame {
    private val plane = imageProxy.planes.first()
    private val closed = AtomicBoolean(false)

    override val width: Int = imageProxy.width
    override val height: Int = imageProxy.height
    override val rotationDegrees: Int = imageProxy.imageInfo.rotationDegrees
    override val buffer: ByteBuffer = plane.buffer.duplicate().also { it.rewind() }
    override val rowStride: Int = plane.rowStride
    override val pixelStride: Int = plane.pixelStride

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            imageProxy.close()
        }
    }
}
