package com.linnan.blindassist.camera

import android.graphics.Bitmap
import com.linnan.blindassist.vision.ExternalFrameTiming
import com.linnan.blindassist.vision.ExternalFrameTransportDiagnostics
import com.linnan.blindassist.vision.FrameStamp
import com.linnan.blindassist.vision.NativeImageVisionFrame
import com.linnan.blindassist.vision.RangingSample
import java.util.concurrent.atomic.AtomicBoolean

/** Owns a decoded bitmap directly, avoiding an XGA-sized Bitmap-to-RGBA copy. */
class OwnedBitmapVisionFrame(
    private val bitmap: Bitmap,
    private val releaseBitmap: (Bitmap) -> Unit = Bitmap::recycle,
    override val rotationDegrees: Int = 0,
    override val frameStamp: FrameStamp? = null,
    override val rangingSample: RangingSample? = null,
    override val externalTiming: ExternalFrameTiming? = null,
    override val externalTransportDiagnostics: ExternalFrameTransportDiagnostics? = null
) : NativeImageVisionFrame {
    private val closed = AtomicBoolean(false)

    init {
        require(bitmap.width > 0 && bitmap.height > 0) { "Bitmap must have positive dimensions" }
    }

    override val width: Int = bitmap.width
    override val height: Int = bitmap.height
    override val nativeImage: Any get() = bitmap

    override fun close() {
        if (closed.compareAndSet(false, true)) releaseBitmap(bitmap)
    }
}
