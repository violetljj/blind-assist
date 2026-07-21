package com.linnan.blindassist.vision

import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.session.DetectorMetrics

data class DetectorFrameResult(
    val detections: List<Detection>,
    val frameSize: FrameSize,
    val metrics: DetectorMetrics,
    val sourceFrame: FrameStamp? = null
)
