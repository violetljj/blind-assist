package com.linnan.blindassist.model

import com.linnan.blindassist.risk.DistanceEvidence

data class Detection(
    val classId: Int,
    val label: String,
    val confidence: Float,
    val boundingBox: BoundingBox,
    val frameSize: FrameSize,
    val distanceEvidence: DistanceEvidence? = null
) {
    constructor(
        classId: Int,
        label: String,
        confidence: Float,
        boundingBox: BoundingBox,
        frameSize: FrameSize
    ) : this(classId, label, confidence, boundingBox, frameSize, null)

    val areaRatio: Float get() = boundingBox.areaRatio(frameSize)
    val centerX: Float get() = boundingBox.centerX
    val centerY: Float get() = boundingBox.centerY
}
