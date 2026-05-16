package com.linnan.blindassist.model

data class Detection(
    val classId: Int,
    val label: String,
    val confidence: Float,
    val boundingBox: BoundingBox,
    val frameSize: FrameSize
) {
    val areaRatio: Float get() = boundingBox.areaRatio(frameSize)
    val centerX: Float get() = boundingBox.centerX
    val centerY: Float get() = boundingBox.centerY
}
