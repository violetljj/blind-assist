package com.linnan.blindassist.vision

import android.graphics.Bitmap

interface ObjectDetector {
    val isReady: Boolean
    val statusMessage: String

    fun detect(frame: VisionFrame): DetectorFrameResult
    fun detect(bitmap: Bitmap): DetectorFrameResult
    fun close()
}
