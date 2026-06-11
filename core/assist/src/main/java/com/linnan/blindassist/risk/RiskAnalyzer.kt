package com.linnan.blindassist.risk

import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

data class RiskAnalyzerConfig(
    val confidenceThreshold: Float = RiskAnalyzer.CONFIDENCE_THRESHOLD,
    val leftBoundary: Float = RiskAnalyzer.LEFT_BOUNDARY,
    val rightBoundary: Float = RiskAnalyzer.RIGHT_BOUNDARY,
    val midBottomRatio: Float = RiskAnalyzer.MID_BOTTOM_RATIO,
    val midAreaRatio: Float = RiskAnalyzer.MID_AREA_RATIO,
    val centerNearBottomRatio: Float = RiskAnalyzer.CENTER_NEAR_BOTTOM_RATIO,
    val centerNearAreaRatio: Float = RiskAnalyzer.CENTER_NEAR_AREA_RATIO,
    val nearBottomRatio: Float = RiskAnalyzer.NEAR_BOTTOM_RATIO,
    val nearAreaRatio: Float = RiskAnalyzer.NEAR_AREA_RATIO,
    val criticalBottomRatio: Float = RiskAnalyzer.CRITICAL_BOTTOM_RATIO,
    val criticalAreaRatio: Float = RiskAnalyzer.CRITICAL_AREA_RATIO,
    val distanceEvidenceMinConfidence: Float = RiskAnalyzer.DISTANCE_EVIDENCE_MIN_CONFIDENCE,
    val distanceEvidenceMaxPromotionSteps: Int = Int.MAX_VALUE,
    val rejectLargeDistanceEvidencePromotion: Boolean = false
) {
    init {
        require(distanceEvidenceMaxPromotionSteps >= 0) {
            "distanceEvidenceMaxPromotionSteps must be non-negative"
        }
    }

    companion object {
        val Default = RiskAnalyzerConfig()
        val CenterNearSensitive = Default.copy(centerNearBottomRatio = 0.58f, centerNearAreaRatio = 0.11f)
        val CenterNearStrict = Default.copy(centerNearBottomRatio = 0.62f, centerNearAreaRatio = 0.13f)
        val CriticalSensitive = Default.copy(criticalBottomRatio = 0.70f, criticalAreaRatio = 0.18f)
        val SideNearSensitive = Default.copy(nearBottomRatio = 0.60f, nearAreaRatio = 0.13f)
    }
}

class RiskAnalyzer(
    private val config: RiskAnalyzerConfig = RiskAnalyzerConfig.Default
) {
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
            .filter { it.confidence >= config.confidenceThreshold }
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
            urgencyScore = best.urgencyScore,
            distanceEvidence = best.distanceEvidence
        )
    }

    private fun candidateFor(detection: Detection, frameSize: FrameSize): RiskCandidate {
        val direction = directionFor(detection, frameSize)
        val bottomRatio = detection.boundingBox.bottom / frameSize.height.toFloat()
        val areaRatio = detection.areaRatio
        val centerBias = 1f - abs(
            detection.boundingBox.centerX / frameSize.width.toFloat() - 0.5f
        ) * 2f
        val geometryProximity = proximityFor(bottomRatio, areaRatio, direction)
        val distanceEvidence = activeDistanceEvidence(detection.distanceEvidence)
        val proximity = fusedProximityFor(
            geometryProximity = geometryProximity,
            distanceEvidence = distanceEvidence,
            direction = direction,
            centerBias = centerBias
        )
        val level = levelFor(proximity, direction)
        val urgencyScore = urgencyScore(
            detection = detection,
            bottomRatio = bottomRatio,
            areaRatio = areaRatio,
            centerBias = centerBias,
            proximity = proximity,
            distanceEvidence = distanceEvidence
        )
        return RiskCandidate(detection, direction, proximity, level, urgencyScore, distanceEvidence)
    }

    private fun urgencyScore(
        detection: Detection,
        bottomRatio: Float,
        areaRatio: Float,
        centerBias: Float,
        proximity: ProximityBand,
        distanceEvidence: DistanceEvidence?
    ): Float {
        val proximityWeight = when (proximity) {
            ProximityBand.CRITICAL -> 4f
            ProximityBand.NEAR -> 3f
            ProximityBand.MID -> 2f
            ProximityBand.FAR -> 1f
        }
        val classWeight = classWeightFor(detection.label)
        val depthWeight = if (distanceEvidence?.source == DistanceEvidenceSource.MONOCULAR_DEPTH) {
            distanceEvidence.confidence * distanceEvidence.relativeDepthScore * 0.6f
        } else {
            0f
        }
        val base = bottomRatio * 1.6f + areaRatio * 5f + centerBias.coerceIn(0f, 1f) * 0.7f
        return (base + proximityWeight + depthWeight + detection.confidence * 0.4f) * classWeight
    }

    private fun directionFor(detection: Detection, frameSize: FrameSize): RiskDirection {
        val x = detection.boundingBox.centerX / frameSize.width.toFloat()
        return when {
            x < config.leftBoundary -> RiskDirection.LEFT
            x > config.rightBoundary -> RiskDirection.RIGHT
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
            isCentered && (bottomRatio >= config.criticalBottomRatio || areaRatio >= config.criticalAreaRatio) -> {
                ProximityBand.CRITICAL
            }
            bottomRatio >= nearBottomRatioFor(direction) || areaRatio >= nearAreaRatioFor(direction) -> ProximityBand.NEAR
            bottomRatio >= config.midBottomRatio || areaRatio >= config.midAreaRatio -> ProximityBand.MID
            else -> ProximityBand.FAR
        }
    }

    private fun activeDistanceEvidence(evidence: DistanceEvidence?): DistanceEvidence? {
        return evidence?.takeIf {
            it.source != DistanceEvidenceSource.GEOMETRY &&
                it.confidence >= config.distanceEvidenceMinConfidence
        }
    }

    private fun fusedProximityFor(
        geometryProximity: ProximityBand,
        distanceEvidence: DistanceEvidence?,
        direction: RiskDirection,
        centerBias: Float
    ): ProximityBand {
        val depthBand = distanceEvidence?.band ?: return geometryProximity
        if (depthBand.ordinal <= geometryProximity.ordinal) {
            return geometryProximity
        }

        val isActionableLane = direction == RiskDirection.CENTER || centerBias >= 0.35f
        if (!isActionableLane) {
            return geometryProximity
        }

        val promotionSteps = depthBand.ordinal - geometryProximity.ordinal
        if (config.rejectLargeDistanceEvidencePromotion &&
            promotionSteps > config.distanceEvidenceMaxPromotionSteps
        ) {
            return geometryProximity
        }
        val cappedDepthBand = geometryProximity.moreUrgentBy(
            min(promotionSteps, config.distanceEvidenceMaxPromotionSteps)
        )

        return when {
            distanceEvidence.confidence >= 0.75f -> cappedDepthBand
            config.distanceEvidenceMaxPromotionSteps != Int.MAX_VALUE -> cappedDepthBand
            promotionSteps >= 2 -> geometryProximity.nextMoreUrgent()
            else -> cappedDepthBand
        }
    }

    private fun ProximityBand.nextMoreUrgent(): ProximityBand {
        return when (this) {
            ProximityBand.FAR -> ProximityBand.MID
            ProximityBand.MID -> ProximityBand.NEAR
            ProximityBand.NEAR -> ProximityBand.CRITICAL
            ProximityBand.CRITICAL -> ProximityBand.CRITICAL
        }
    }

    private fun ProximityBand.moreUrgentBy(steps: Int): ProximityBand {
        var current = this
        repeat(steps.coerceAtLeast(0)) {
            current = current.nextMoreUrgent()
        }
        return current
    }

    private fun nearBottomRatioFor(direction: RiskDirection): Float {
        return if (direction == RiskDirection.CENTER) config.centerNearBottomRatio else config.nearBottomRatio
    }

    private fun nearAreaRatioFor(direction: RiskDirection): Float {
        return if (direction == RiskDirection.CENTER) config.centerNearAreaRatio else config.nearAreaRatio
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

        if (proximity == ProximityBand.CRITICAL && direction == RiskDirection.CENTER) {
            return "前方很近，放慢"
        }
        if (proximity == ProximityBand.NEAR) {
            return when (direction) {
                RiskDirection.CENTER -> "前方近处，减速"
                RiskDirection.LEFT -> "左前方近处，注意避让"
                RiskDirection.RIGHT -> "右前方近处，注意避让"
                RiskDirection.NONE -> "注意前方"
            }
        }

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
        val urgencyScore: Float,
        val distanceEvidence: DistanceEvidence?
    )

    companion object {
        const val CONFIDENCE_THRESHOLD = 0.35f
        const val LEFT_BOUNDARY = 0.35f
        const val RIGHT_BOUNDARY = 0.65f
        const val MID_BOTTOM_RATIO = 0.45f
        const val MID_AREA_RATIO = 0.06f
        const val CENTER_NEAR_BOTTOM_RATIO = 0.58f
        const val CENTER_NEAR_AREA_RATIO = 0.11f
        const val NEAR_BOTTOM_RATIO = 0.62f
        const val NEAR_AREA_RATIO = 0.14f
        const val CRITICAL_BOTTOM_RATIO = 0.72f
        const val CRITICAL_AREA_RATIO = 0.20f
        const val DISTANCE_EVIDENCE_MIN_CONFIDENCE = 0.55f
    }
}
