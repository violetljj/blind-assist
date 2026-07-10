package com.linnan.blindassist.risk

import com.linnan.blindassist.model.Detection

enum class RiskLevel {
    NONE,
    LOW,
    MEDIUM,
    HIGH
}

enum class RiskEvidenceState {
    NO_SUPPORTED_TARGET_EVIDENCE,
    SUPPORTED_TARGET_EVIDENCE
}

enum class RiskDirection {
    NONE,
    LEFT,
    CENTER,
    RIGHT
}

enum class ProximityBand {
    FAR,
    MID,
    NEAR,
    CRITICAL
}

enum class ApproachTrend {
    UNKNOWN,
    STABLE,
    APPROACHING,
    RECEDING
}

data class RiskScoreBreakdown(
    val confidence: Float = 0f,
    val classWeight: Float = 1f,
    val directionWeight: Float = 0f,
    val proximityWeight: Float = 0f,
    val bottomPosition: Float = 0f,
    val area: Float = 0f,
    val centerLane: Float = 0f,
    val distanceEvidence: Float = 0f,
    val approachTrend: Float = 0f,
    val total: Float = 0f,
    val fusionSummary: String = RiskFusionReason.GEOMETRY_ONLY.name
)

data class RiskResult(
    val level: RiskLevel,
    val direction: RiskDirection,
    val message: String,
    val sourceDetection: Detection? = null,
    val proximity: ProximityBand = ProximityBand.FAR,
    val urgencyScore: Float = 0f,
    val distanceEvidence: DistanceEvidence? = null,
    val riskScore: Float = urgencyScore,
    val scoreBreakdown: RiskScoreBreakdown = RiskScoreBreakdown(total = riskScore),
    val approachTrend: ApproachTrend = ApproachTrend.UNKNOWN,
    val evidenceState: RiskEvidenceState = if (sourceDetection != null || level != RiskLevel.NONE) {
        RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE
    } else {
        RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE
    }
) {
    constructor(
        level: RiskLevel,
        direction: RiskDirection,
        message: String,
        sourceDetection: Detection? = null,
        proximity: ProximityBand = ProximityBand.FAR,
        urgencyScore: Float = 0f
    ) : this(
        level = level,
        direction = direction,
        message = message,
        sourceDetection = sourceDetection,
        proximity = proximity,
        urgencyScore = urgencyScore,
        distanceEvidence = null,
        riskScore = urgencyScore,
        scoreBreakdown = RiskScoreBreakdown(total = urgencyScore),
        approachTrend = ApproachTrend.UNKNOWN,
        evidenceState = if (sourceDetection != null || level != RiskLevel.NONE) {
            RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE
        } else {
            RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE
        }
    )
}
