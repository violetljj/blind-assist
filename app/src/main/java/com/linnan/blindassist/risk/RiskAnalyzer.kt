package com.linnan.blindassist.risk

import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize

class RiskAnalyzer {
    private val alertLabels = setOf(
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "traffic light",
        "stop sign",
        "bench",
        "chair",
        "potted plant"
    )

    fun analyze(detections: List<Detection>, frameSize: FrameSize): RiskResult {
        val candidates = detections
            .filter { it.label in alertLabels }
            .filter { it.confidence >= CONFIDENCE_THRESHOLD }
            .sortedWith(
                compareByDescending<Detection> { score(it, frameSize) }
                    .thenByDescending { it.confidence }
            )

        val best = candidates.firstOrNull()
            ?: return RiskResult(RiskLevel.NONE, RiskDirection.NONE, "未发现风险")

        val direction = directionFor(best, frameSize)
        val near = isNear(best, frameSize)
        val level = when {
            direction == RiskDirection.CENTER && near -> RiskLevel.HIGH
            near -> RiskLevel.MEDIUM
            else -> RiskLevel.LOW
        }

        val message = messageFor(level, direction, best.label)
        return RiskResult(level, direction, message, best)
    }

    private fun score(detection: Detection, frameSize: FrameSize): Float {
        val bottomWeight = detection.boundingBox.bottom / frameSize.height.toFloat()
        val centerBias = 1f - kotlin.math.abs(
            detection.boundingBox.centerX / frameSize.width.toFloat() - 0.5f
        )
        return detection.areaRatio * 2f + bottomWeight + centerBias * 0.5f
    }

    private fun directionFor(detection: Detection, frameSize: FrameSize): RiskDirection {
        val x = detection.boundingBox.centerX / frameSize.width.toFloat()
        return when {
            x < LEFT_BOUNDARY -> RiskDirection.LEFT
            x > RIGHT_BOUNDARY -> RiskDirection.RIGHT
            else -> RiskDirection.CENTER
        }
    }

    private fun isNear(detection: Detection, frameSize: FrameSize): Boolean {
        val bottomRatio = detection.boundingBox.bottom / frameSize.height.toFloat()
        return bottomRatio >= NEAR_BOTTOM_RATIO || detection.areaRatio >= NEAR_AREA_RATIO
    }

    private fun messageFor(level: RiskLevel, direction: RiskDirection, label: String): String {
        if (level == RiskLevel.NONE || level == RiskLevel.LOW) return "注意前方"

        val objectName = when (label) {
            "person" -> "人"
            "car", "bus", "truck", "motorcycle", "bicycle" -> "车辆"
            "bench", "chair", "potted plant" -> "障碍"
            else -> "障碍"
        }

        return when (direction) {
            RiskDirection.CENTER -> "前方有$objectName"
            RiskDirection.LEFT -> "左前方有$objectName"
            RiskDirection.RIGHT -> "右前方有$objectName"
            RiskDirection.NONE -> "注意前方"
        }
    }

    companion object {
        const val CONFIDENCE_THRESHOLD = 0.35f
        const val LEFT_BOUNDARY = 0.35f
        const val RIGHT_BOUNDARY = 0.65f
        const val NEAR_BOTTOM_RATIO = 0.55f
        const val NEAR_AREA_RATIO = 0.12f
    }
}
