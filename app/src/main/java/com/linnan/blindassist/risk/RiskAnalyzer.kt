package com.linnan.blindassist.risk

import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

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
            .map { candidateFor(it, frameSize) }
            .sortedWith(
                compareByDescending<RiskCandidate> { it.urgencyScore }
                    .thenByDescending { it.detection.confidence }
            )

        val best = candidates.firstOrNull()
            ?: return RiskResult(RiskLevel.NONE, RiskDirection.NONE, "未发现风险")

        val message = messageFor(best.level, best.direction, best.proximity, best.detection.label)
        return RiskResult(
            level = best.level,
            direction = best.direction,
            message = message,
            sourceDetection = best.detection,
            proximity = best.proximity,
            urgencyScore = best.urgencyScore
        )
    }

    private fun candidateFor(detection: Detection, frameSize: FrameSize): RiskCandidate {
        val direction = directionFor(detection, frameSize)
        val bottomRatio = detection.boundingBox.bottom / frameSize.height.toFloat()
        val areaRatio = detection.areaRatio
        val centerBias = 1f - abs(
            detection.boundingBox.centerX / frameSize.width.toFloat() - 0.5f
        ) * 2f
        val proximity = proximityFor(bottomRatio, areaRatio, direction)
        val level = levelFor(proximity, direction)
        val urgencyScore = urgencyScore(
            detection = detection,
            bottomRatio = bottomRatio,
            areaRatio = areaRatio,
            centerBias = centerBias,
            proximity = proximity
        )
        return RiskCandidate(detection, direction, proximity, level, urgencyScore)
    }

    private fun urgencyScore(
        detection: Detection,
        bottomRatio: Float,
        areaRatio: Float,
        centerBias: Float,
        proximity: ProximityBand
    ): Float {
        val proximityWeight = when (proximity) {
            ProximityBand.CRITICAL -> 4f
            ProximityBand.NEAR -> 3f
            ProximityBand.MID -> 2f
            ProximityBand.FAR -> 1f
        }
        val classWeight = classWeightFor(detection.label)
        val base = bottomRatio * 1.6f + areaRatio * 5f + centerBias.coerceIn(0f, 1f) * 0.7f
        return (base + proximityWeight + detection.confidence * 0.4f) * classWeight
    }

    private fun directionFor(detection: Detection, frameSize: FrameSize): RiskDirection {
        val x = detection.boundingBox.centerX / frameSize.width.toFloat()
        return when {
            x < LEFT_BOUNDARY -> RiskDirection.LEFT
            x > RIGHT_BOUNDARY -> RiskDirection.RIGHT
            else -> RiskDirection.CENTER
        }
    }

    private fun proximityFor(
        bottomRatio: Float,
        areaRatio: Float,
        direction: RiskDirection
    ): ProximityBand {
        val isCentered = direction == RiskDirection.CENTER
        return when {
            isCentered && (bottomRatio >= CRITICAL_BOTTOM_RATIO || areaRatio >= CRITICAL_AREA_RATIO) -> {
                ProximityBand.CRITICAL
            }
            bottomRatio >= NEAR_BOTTOM_RATIO || areaRatio >= NEAR_AREA_RATIO -> ProximityBand.NEAR
            bottomRatio >= MID_BOTTOM_RATIO || areaRatio >= MID_AREA_RATIO -> ProximityBand.MID
            else -> ProximityBand.FAR
        }
    }

    private fun levelFor(proximity: ProximityBand, direction: RiskDirection): RiskLevel {
        return when (proximity) {
            ProximityBand.CRITICAL -> RiskLevel.HIGH
            ProximityBand.NEAR -> if (direction == RiskDirection.CENTER) RiskLevel.HIGH else RiskLevel.MEDIUM
            ProximityBand.MID -> RiskLevel.LOW
            ProximityBand.FAR -> RiskLevel.NONE
        }
    }

    private fun messageFor(
        level: RiskLevel,
        direction: RiskDirection,
        proximity: ProximityBand,
        label: String
    ): String {
        if (level == RiskLevel.NONE) return "未发现风险"

        val objectName = when (label) {
            "person" -> "人"
            "car", "bus", "truck", "motorcycle", "bicycle" -> "车辆"
            "bench", "chair", "potted plant" -> "障碍"
            else -> "障碍"
        }
        val proximityText = when (proximity) {
            ProximityBand.CRITICAL -> "迫近"
            ProximityBand.NEAR -> "近处"
            ProximityBand.MID -> "中距"
            ProximityBand.FAR -> "远处"
        }

        return when (direction) {
            RiskDirection.CENTER -> "前方$proximityText 有$objectName"
            RiskDirection.LEFT -> "左前方$proximityText 有$objectName"
            RiskDirection.RIGHT -> "右前方$proximityText 有$objectName"
            RiskDirection.NONE -> "注意前方"
        }
    }

    private fun classWeightFor(label: String): Float {
        return when (label) {
            "person" -> 1.12f
            "car", "bus", "truck", "motorcycle", "bicycle" -> 1.08f
            "traffic light", "stop sign" -> 0.92f
            else -> 1f
        }
    }

    private data class RiskCandidate(
        val detection: Detection,
        val direction: RiskDirection,
        val proximity: ProximityBand,
        val level: RiskLevel,
        val urgencyScore: Float
    )

    companion object {
        const val CONFIDENCE_THRESHOLD = 0.35f
        const val LEFT_BOUNDARY = 0.35f
        const val RIGHT_BOUNDARY = 0.65f
        const val MID_BOTTOM_RATIO = 0.45f
        const val MID_AREA_RATIO = 0.06f
        const val NEAR_BOTTOM_RATIO = 0.60f
        const val NEAR_AREA_RATIO = 0.12f
        const val CRITICAL_BOTTOM_RATIO = 0.72f
        const val CRITICAL_AREA_RATIO = 0.20f
    }
}
