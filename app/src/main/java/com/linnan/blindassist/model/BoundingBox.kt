package com.linnan.blindassist.model

import kotlin.math.max
import kotlin.math.min

data class BoundingBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
) {
    val width: Float get() = max(0f, right - left)
    val height: Float get() = max(0f, bottom - top)
    val centerX: Float get() = (left + right) / 2f
    val centerY: Float get() = (top + bottom) / 2f

    fun areaRatio(frame: FrameSize): Float {
        val frameArea = max(1, frame.width * frame.height).toFloat()
        return (width * height) / frameArea
    }

    fun clamped(frame: FrameSize): BoundingBox = BoundingBox(
        left = min(max(left, 0f), frame.width.toFloat()),
        top = min(max(top, 0f), frame.height.toFloat()),
        right = min(max(right, 0f), frame.width.toFloat()),
        bottom = min(max(bottom, 0f), frame.height.toFloat())
    )
}
