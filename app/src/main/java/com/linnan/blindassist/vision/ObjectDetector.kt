package com.linnan.blindassist.vision

import android.graphics.Bitmap
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.session.DetectorMetrics

interface ObjectDetector {
    val isReady: Boolean
    val statusMessage: String

    fun detect(bitmap: Bitmap): DetectorFrameResult
    fun close()
}

data class DetectorFrameResult(
    val detections: List<Detection>,
    val frameSize: FrameSize,
    val metrics: DetectorMetrics
)
