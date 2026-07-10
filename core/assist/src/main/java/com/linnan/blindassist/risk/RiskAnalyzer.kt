package com.linnan.blindassist.risk

import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import kotlin.math.abs
import kotlin.math.max

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
    val distanceEvidenceMaxPromotionSteps: Int = ConservativeRiskFusionPolicy.DEFAULT_DEPTH_MAX_PROMOTION_STEPS,
    val rejectLargeDistanceEvidencePromotion: Boolean = ConservativeRiskFusionPolicy.DEFAULT_REJECT_LARGE_DEPTH_PROMOTION,
    val lowRiskScoreThreshold: Float = RiskAnalyzer.LOW_RISK_SCORE_THRESHOLD,
    val mediumRiskScoreThreshold: Float = RiskAnalyzer.MEDIUM_RISK_SCORE_THRESHOLD,
    val highRiskScoreThreshold: Float = RiskAnalyzer.HIGH_RISK_SCORE_THRESHOLD
) {
    init {
        require(distanceEvidenceMaxPromotionSteps >= 0) {
            "distanceEvidenceMaxPromotionSteps must be non-negative"
        }
        require(lowRiskScoreThreshold <= mediumRiskScoreThreshold) {
            "lowRiskScoreThreshold must not exceed mediumRiskScoreThreshold"
        }
        require(mediumRiskScoreThreshold <= highRiskScoreThreshold) {
            "mediumRiskScoreThreshold must not exceed highRiskScoreThreshold"
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
    private val fusionPolicy = ConservativeRiskFusionPolicy(
        ConservativeRiskFusionConfig(
            depthMaxPromotionSteps = config.distanceEvidenceMaxPromotionSteps,
            rejectLargeDepthPromotion = config.rejectLargeDistanceEvidencePromotion
        )
    )

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
            ?: return RiskResult(
                level = RiskLevel.NONE,
                direction = RiskDirection.NONE,
                message = "当前未检测到达到提醒条件的支持目标，请继续确认周围环境。",
                evidenceState = RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE
            )

        val message = messageFor(best.level, best.direction, best.proximity, best.detection.label)
        return RiskResult(
            level = best.level,
            direction = best.direction,
            message = message,
            sourceDetection = best.detection,
            proximity = best.proximity,
            urgencyScore = best.urgencyScore,
            distanceEvidence = best.distanceEvidence,
            riskScore = best.urgencyScore,
            scoreBreakdown = best.scoreBreakdown,
            evidenceState = RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE
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
        val depthFusion = fusionPolicy.fuseDepth(
            geometryProximity = geometryProximity,
            distanceEvidence = detection.distanceEvidence,
            direction = direction,
            centerBias = centerBias,
            minConfidence = config.distanceEvidenceMinConfidence
        )
        val scoreBreakdown = scoreBreakdown(
            detection = detection,
            bottomRatio = bottomRatio,
            areaRatio = areaRatio,
            centerBias = centerBias,
            direction = direction,
            proximity = depthFusion.proximity,
            distanceEvidence = depthFusion.evidence,
            fusionReason = depthFusion.reason
        )
        val level = levelFor(scoreBreakdown.total, depthFusion.proximity, direction)
        return RiskCandidate(
            detection = detection,
            direction = direction,
            proximity = depthFusion.proximity,
            level = level,
            urgencyScore = scoreBreakdown.total,
            distanceEvidence = depthFusion.evidence,
            scoreBreakdown = scoreBreakdown
        )
    }

    private fun scoreBreakdown(
        detection: Detection,
        bottomRatio: Float,
        areaRatio: Float,
        centerBias: Float,
        direction: RiskDirection,
        proximity: ProximityBand,
        distanceEvidence: DistanceEvidence?,
        fusionReason: RiskFusionReason
    ): RiskScoreBreakdown {
        val proximityWeight = when (proximity) {
            ProximityBand.CRITICAL -> 4f
            ProximityBand.NEAR -> 3f
            ProximityBand.MID -> 2f
            ProximityBand.FAR -> 1f
        }
        val classWeight = classWeightFor(detection.label)
        val confidenceScore = detection.confidence * 0.4f
        val directionWeight = when (direction) {
            RiskDirection.CENTER -> 0.35f
            RiskDirection.LEFT,
            RiskDirection.RIGHT -> 0.12f
            RiskDirection.NONE -> 0f
        }
        val depthWeight = if (distanceEvidence?.source == DistanceEvidenceSource.MONOCULAR_DEPTH) {
            distanceEvidence.confidence * distanceEvidence.relativeDepthScore * 0.6f
        } else {
            0f
        }
        val bottomScore = bottomRatio * 1.6f
        val areaScore = areaRatio * 5f
        val centerLaneScore = centerBias.coerceIn(0f, 1f) * 0.7f
        val total = (
            bottomScore +
                areaScore +
                centerLaneScore +
                directionWeight +
                proximityWeight +
                depthWeight +
                confidenceScore
            ) * classWeight
        return RiskScoreBreakdown(
            confidence = confidenceScore,
            classWeight = classWeight,
            directionWeight = directionWeight,
            proximityWeight = proximityWeight,
            bottomPosition = bottomScore,
            area = areaScore,
            centerLane = centerLaneScore,
            distanceEvidence = depthWeight,
            total = total,
            fusionSummary = fusionReason.name
        )
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

    private fun nearBottomRatioFor(direction: RiskDirection): Float {
        return if (direction == RiskDirection.CENTER) config.centerNearBottomRatio else config.nearBottomRatio
    }

    private fun nearAreaRatioFor(direction: RiskDirection): Float {
        return if (direction == RiskDirection.CENTER) config.centerNearAreaRatio else config.nearAreaRatio
    }

    private fun levelFor(score: Float, proximity: ProximityBand, direction: RiskDirection): RiskLevel {
        val scoreLevel = when {
            score >= config.highRiskScoreThreshold -> RiskLevel.HIGH
            score >= config.mediumRiskScoreThreshold -> RiskLevel.MEDIUM
            score >= config.lowRiskScoreThreshold -> RiskLevel.LOW
            else -> RiskLevel.NONE
        }
        val protectedScoreLevel = when {
            proximity == ProximityBand.FAR -> RiskLevel.NONE
            proximity == ProximityBand.MID -> RiskLevel.LOW
            direction != RiskDirection.CENTER && scoreLevel == RiskLevel.HIGH -> RiskLevel.MEDIUM
            else -> scoreLevel
        }
        val proximityLevel = when (proximity) {
            ProximityBand.CRITICAL -> RiskLevel.HIGH
            ProximityBand.NEAR -> if (direction == RiskDirection.CENTER) RiskLevel.HIGH else RiskLevel.MEDIUM
            ProximityBand.MID -> RiskLevel.LOW
            ProximityBand.FAR -> RiskLevel.NONE
        }
        return maxOf(protectedScoreLevel, proximityLevel)
    }

    private fun messageFor(
        level: RiskLevel,
        direction: RiskDirection,
        proximity: ProximityBand,
        label: String
    ): String {
        if (level == RiskLevel.NONE) return "检测到模型支持的目标，当前未达到提醒条件。"

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
        val distanceEvidence: DistanceEvidence?,
        val scoreBreakdown: RiskScoreBreakdown
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
        const val LOW_RISK_SCORE_THRESHOLD = 3.0f
        const val MEDIUM_RISK_SCORE_THRESHOLD = 4.7f
        const val HIGH_RISK_SCORE_THRESHOLD = 5.5f
    }
}
