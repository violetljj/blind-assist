package com.linnan.blindassist.ui

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import kotlin.math.abs

internal class OverlayBoxSmoother(
    private val alpha: Float = DEFAULT_ALPHA
) {
    private val previousBoxes = mutableMapOf<String, BoundingBox>()

    fun smooth(detections: List<Detection>): List<SmoothedDetection> {
        val activeKeys = mutableSetOf<String>()
        val smoothed = detections.mapIndexed { index, detection ->
            val key = detectionKey(index, detection)
            activeKeys += key
            val previous = previousBoxes[key]
            val displayBox = if (previous == null || isLargeJump(previous, detection.boundingBox)) {
                detection.boundingBox
            } else {
                previous.lerp(detection.boundingBox, alpha)
            }
            previousBoxes[key] = displayBox
            SmoothedDetection(
                raw = detection,
                display = detection.copy(boundingBox = displayBox)
            )
        }
        previousBoxes.keys.retainAll(activeKeys)
        return smoothed
    }

    fun reset() {
        previousBoxes.clear()
    }

    private fun detectionKey(index: Int, detection: Detection): String {
        return "$index:${detection.classId}:${detection.label}"
    }

    private fun isLargeJump(previous: BoundingBox, current: BoundingBox): Boolean {
        val previousWidth = previous.width.coerceAtLeast(1f)
        val previousHeight = previous.height.coerceAtLeast(1f)
        val dx = abs(previous.centerX - current.centerX) / previousWidth
        val dy = abs(previous.centerY - current.centerY) / previousHeight
        return dx > LARGE_JUMP_RATIO || dy > LARGE_JUMP_RATIO
    }

    companion object {
        const val DEFAULT_ALPHA = 0.35f
        const val LARGE_JUMP_RATIO = 0.8f
    }
}

internal data class SmoothedDetection(
    val raw: Detection,
    val display: Detection
)

private fun BoundingBox.lerp(target: BoundingBox, alpha: Float): BoundingBox {
    return BoundingBox(
        left = left + (target.left - left) * alpha,
        top = top + (target.top - top) * alpha,
        right = right + (target.right - right) * alpha,
        bottom = bottom + (target.bottom - bottom) * alpha
    )
}
