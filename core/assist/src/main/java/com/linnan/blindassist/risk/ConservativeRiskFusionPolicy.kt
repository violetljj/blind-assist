package com.linnan.blindassist.risk

import kotlin.math.min

data class ConservativeRiskFusionConfig(
    val depthMaxPromotionSteps: Int = ConservativeRiskFusionPolicy.DEFAULT_DEPTH_MAX_PROMOTION_STEPS,
    val rejectLargeDepthPromotion: Boolean = ConservativeRiskFusionPolicy.DEFAULT_REJECT_LARGE_DEPTH_PROMOTION,
    val depthMinActionableCenterBias: Float = ConservativeRiskFusionPolicy.DEFAULT_DEPTH_MIN_ACTIONABLE_CENTER_BIAS,
    val motionMaxPromotionSteps: Int = ConservativeRiskFusionPolicy.DEFAULT_MOTION_MAX_PROMOTION_STEPS
) {
    init {
        require(depthMaxPromotionSteps >= 0) { "depthMaxPromotionSteps must be non-negative" }
        require(depthMinActionableCenterBias in 0f..1f) {
            "depthMinActionableCenterBias must be in [0, 1]"
        }
        require(motionMaxPromotionSteps >= 0) { "motionMaxPromotionSteps must be non-negative" }
    }
}

enum class RiskFusionReason {
    GEOMETRY_ONLY,
    DEPTH_PROMOTED,
    DEPTH_REJECTED_LOW_CONFIDENCE,
    DEPTH_REJECTED_SOURCE,
    DEPTH_REJECTED_NOT_CLOSER,
    DEPTH_REJECTED_LANE,
    DEPTH_REJECTED_LARGE_PROMOTION,
    MOTION_PROMOTED,
    STABILITY_PROMOTED
}

data class DepthFusionResult(
    val proximity: ProximityBand,
    val evidence: DistanceEvidence?,
    val reason: RiskFusionReason
)

data class MotionFusionResult(
    val level: RiskLevel,
    val score: Float,
    val scoreBreakdown: RiskScoreBreakdown,
    val reason: RiskFusionReason?
)

class ConservativeRiskFusionPolicy(
    private val config: ConservativeRiskFusionConfig = ConservativeRiskFusionConfig()
) {
    fun fuseDepth(
        geometryProximity: ProximityBand,
        distanceEvidence: DistanceEvidence?,
        direction: RiskDirection,
        centerBias: Float,
        minConfidence: Float
    ): DepthFusionResult {
        val evidence = distanceEvidence ?: return DepthFusionResult(
            proximity = geometryProximity,
            evidence = null,
            reason = RiskFusionReason.GEOMETRY_ONLY
        )
        if (evidence.source == DistanceEvidenceSource.GEOMETRY) {
            return DepthFusionResult(
                proximity = geometryProximity,
                evidence = null,
                reason = RiskFusionReason.DEPTH_REJECTED_SOURCE
            )
        }
        if (evidence.confidence < minConfidence) {
            return DepthFusionResult(
                proximity = geometryProximity,
                evidence = null,
                reason = RiskFusionReason.DEPTH_REJECTED_LOW_CONFIDENCE
            )
        }
        if (evidence.band.ordinal <= geometryProximity.ordinal) {
            return DepthFusionResult(
                proximity = geometryProximity,
                evidence = evidence,
                reason = RiskFusionReason.DEPTH_REJECTED_NOT_CLOSER
            )
        }

        val isActionableLane = direction == RiskDirection.CENTER ||
            centerBias >= config.depthMinActionableCenterBias
        if (!isActionableLane) {
            return DepthFusionResult(
                proximity = geometryProximity,
                evidence = evidence,
                reason = RiskFusionReason.DEPTH_REJECTED_LANE
            )
        }

        val promotionSteps = evidence.band.ordinal - geometryProximity.ordinal
        if (config.rejectLargeDepthPromotion && promotionSteps > config.depthMaxPromotionSteps) {
            return DepthFusionResult(
                proximity = geometryProximity,
                evidence = evidence,
                reason = RiskFusionReason.DEPTH_REJECTED_LARGE_PROMOTION
            )
        }

        val cappedPromotionSteps = min(promotionSteps, config.depthMaxPromotionSteps)
        val fused = geometryProximity.moreUrgentBy(cappedPromotionSteps)
        return DepthFusionResult(
            proximity = fused,
            evidence = evidence,
            reason = if (fused.ordinal > geometryProximity.ordinal) {
                RiskFusionReason.DEPTH_PROMOTED
            } else {
                RiskFusionReason.GEOMETRY_ONLY
            }
        )
    }

    fun fuseMotion(
        raw: RiskResult,
        trend: ApproachTrend,
        scoreBoost: Float
    ): MotionFusionResult {
        if (trend != ApproachTrend.APPROACHING || config.motionMaxPromotionSteps <= 0) {
            return MotionFusionResult(
                level = raw.level,
                score = raw.riskScore,
                scoreBreakdown = raw.scoreBreakdown,
                reason = null
            )
        }

        val boostedScore = raw.riskScore + scoreBoost
        val boostedBreakdown = raw.scoreBreakdown.copy(
            approachTrend = raw.scoreBreakdown.approachTrend + scoreBoost,
            total = boostedScore,
            fusionSummary = appendFusionReason(
                raw.scoreBreakdown.fusionSummary,
                RiskFusionReason.MOTION_PROMOTED
            )
        )
        return MotionFusionResult(
            level = boostedLevelFor(raw),
            score = boostedScore,
            scoreBreakdown = boostedBreakdown,
            reason = RiskFusionReason.MOTION_PROMOTED
        )
    }

    /**
     * A segmentation region is only promoted after temporal confirmation. This intentionally
     * cannot make a single-frame mask actionable and is limited to one LOW -> MEDIUM step.
     */
    fun fuseStableSegmentation(raw: RiskResult, confirmed: Boolean): MotionFusionResult {
        if (!confirmed || raw.level != RiskLevel.LOW || raw.direction != RiskDirection.CENTER ||
            raw.proximity != ProximityBand.MID
        ) {
            return MotionFusionResult(raw.level, raw.riskScore, raw.scoreBreakdown, null)
        }
        val scoreBoost = 0.7f
        val boostedScore = raw.riskScore + scoreBoost
        return MotionFusionResult(
            level = RiskLevel.MEDIUM,
            score = boostedScore,
            scoreBreakdown = raw.scoreBreakdown.copy(
                approachTrend = raw.scoreBreakdown.approachTrend + scoreBoost,
                total = boostedScore,
                fusionSummary = appendFusionReason(raw.scoreBreakdown.fusionSummary, RiskFusionReason.STABILITY_PROMOTED)
            ),
            reason = RiskFusionReason.STABILITY_PROMOTED
        )
    }

    private fun boostedLevelFor(raw: RiskResult): RiskLevel {
        if (raw.direction == RiskDirection.CENTER && raw.proximity == ProximityBand.NEAR) {
            return RiskLevel.HIGH
        }
        var promoted = raw.level
        repeat(config.motionMaxPromotionSteps) {
            promoted = promoteOne(promoted)
        }
        if (raw.direction != RiskDirection.CENTER && promoted == RiskLevel.HIGH) {
            return RiskLevel.MEDIUM
        }
        return promoted
    }

    private fun promoteOne(level: RiskLevel): RiskLevel {
        return when (level) {
            RiskLevel.NONE -> RiskLevel.LOW
            RiskLevel.LOW -> RiskLevel.MEDIUM
            RiskLevel.MEDIUM -> RiskLevel.HIGH
            RiskLevel.HIGH -> RiskLevel.HIGH
        }
    }

    companion object {
        const val DEFAULT_DEPTH_MAX_PROMOTION_STEPS = 1
        const val DEFAULT_REJECT_LARGE_DEPTH_PROMOTION = true
        const val DEFAULT_DEPTH_MIN_ACTIONABLE_CENTER_BIAS = 0.35f
        const val DEFAULT_MOTION_MAX_PROMOTION_STEPS = 1

        fun appendFusionReason(current: String, reason: RiskFusionReason): String {
            val reasonName = reason.name
            if (reason == RiskFusionReason.GEOMETRY_ONLY) {
                return current.ifBlank { reasonName }
            }
            val parts = current
                .split("+")
                .map { it.trim() }
                .filter { it.isNotEmpty() && it != RiskFusionReason.GEOMETRY_ONLY.name }
                .toMutableList()
            if (reasonName !in parts) {
                parts += reasonName
            }
            return if (parts.isEmpty()) RiskFusionReason.GEOMETRY_ONLY.name else parts.joinToString("+")
        }
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
