package com.linnan.blindassist.model

import com.linnan.blindassist.risk.DistanceEvidence

data class Detection(
    val classId: Int,
    val label: String,
    val confidence: Float,
    val boundingBox: BoundingBox,
    val frameSize: FrameSize,
    val distanceEvidence: DistanceEvidence? = null,
    val source: DetectionSource = DetectionSource.OBJECT_DETECTOR
) {
    constructor(
        classId: Int,
        label: String,
        confidence: Float,
        boundingBox: BoundingBox,
        frameSize: FrameSize
    ) : this(classId, label, confidence, boundingBox, frameSize, null, DetectionSource.OBJECT_DETECTOR)

    val areaRatio: Float get() = boundingBox.areaRatio(frameSize)
    val centerX: Float get() = boundingBox.centerX
    val centerY: Float get() = boundingBox.centerY
}

enum class DetectionSource {
    OBJECT_DETECTOR,
    SEGMENTATION
}
