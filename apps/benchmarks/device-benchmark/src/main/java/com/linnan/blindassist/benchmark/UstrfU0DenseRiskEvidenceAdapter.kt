package com.linnan.blindassist.benchmark

import com.linnan.blindassist.risk.ApproachTrend
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskEvidenceState
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.risk.RiskScoreBreakdown
import com.linnan.blindassist.session.AssistDecisionKernel
import com.linnan.blindassist.session.AssistRiskEvidenceFrame
import com.linnan.blindassist.ustrf.UstrfRouteConditionedRiskResolution
import com.linnan.blindassist.ustrf.UstrfRouteIntrusionEvidence
import kotlin.math.max

internal data class UstrfU0DenseRiskEvidenceConfig(
    val lowThreshold: Float = 0.35f,
    val mediumThreshold: Float = 0.55f,
    val highThreshold: Float = 0.75f,
    val localPeakWeight: Float = 0.75f,
    val maximumRouteUnknownFraction: Float = 0.50f
) {
    init {
        require(lowThreshold in 0f..1f)
        require(mediumThreshold in lowThreshold..1f)
        require(highThreshold in mediumThreshold..1f)
        require(localPeakWeight in 0f..1f)
        require(maximumRouteUnknownFraction in 0f..1f)
    }
}

internal enum class UstrfU0DenseRiskEvidenceFailure {
    ROUTE_CONDITIONED_EVIDENCE_UNAVAILABLE,
    ROUTE_UNKNOWN_FRACTION_TOO_HIGH,
    EMPTY_CONTRIBUTING_FIELD,
    EMPTY_RISK_SOURCE_INVENTORY,
    INVALID_TIME_BINDING
}

internal sealed interface UstrfU0DenseRiskEvidenceResolution {
    data class Available(
        val kernelEvidence: AssistRiskEvidenceFrame,
        val normalizedRiskScore: Float,
        val sourceEvidence: UstrfRouteIntrusionEvidence
    ) : UstrfU0DenseRiskEvidenceResolution

    data class Unavailable(
        val failure: UstrfU0DenseRiskEvidenceFailure,
        val sourceFailure: String? = null
    ) : UstrfU0DenseRiskEvidenceResolution
}

/**
 * Frozen U0-only normalization from continuous route-intrusion evidence into the
 * shared kernel's object-agnostic evidence input. No box, class label, event state,
 * temporal trend, or feedback decision is manufactured here.
 */
internal object UstrfU0DenseRiskEvidenceAdapter {
    const val CONTRACT_ID = "ustrf_u0_dense_route_intrusion_to_kernel_risk_v1"

    fun normalize(
        resolution: UstrfRouteConditionedRiskResolution,
        episodeId: String,
        decisionFrameId: String,
        decisionAtMs: Long,
        config: UstrfU0DenseRiskEvidenceConfig = UstrfU0DenseRiskEvidenceConfig()
    ): UstrfU0DenseRiskEvidenceResolution {
        require(episodeId.isNotBlank() && decisionFrameId.isNotBlank())
        if (resolution is UstrfRouteConditionedRiskResolution.Unavailable) {
            return UstrfU0DenseRiskEvidenceResolution.Unavailable(
                UstrfU0DenseRiskEvidenceFailure.ROUTE_CONDITIONED_EVIDENCE_UNAVAILABLE,
                resolution.failure.name
            )
        }
        val evidence = (resolution as UstrfRouteConditionedRiskResolution.Available).evidence
        if (evidence.routeUnknownFraction > config.maximumRouteUnknownFraction) {
            return UstrfU0DenseRiskEvidenceResolution.Unavailable(
                UstrfU0DenseRiskEvidenceFailure.ROUTE_UNKNOWN_FRACTION_TOO_HIGH
            )
        }
        if (evidence.contributingCellCount <= 0) {
            return UstrfU0DenseRiskEvidenceResolution.Unavailable(
                UstrfU0DenseRiskEvidenceFailure.EMPTY_CONTRIBUTING_FIELD
            )
        }
        if (evidence.riskSources.isEmpty() || evidence.riskSources.any(String::isBlank)) {
            return UstrfU0DenseRiskEvidenceResolution.Unavailable(
                UstrfU0DenseRiskEvidenceFailure.EMPTY_RISK_SOURCE_INVENTORY
            )
        }
        val observedAtMs = evidence.sourceFrame.capturedAtNs / 1_000_000L
        val validUntilMs = evidence.validUntilNs / 1_000_000L
        if (observedAtMs != decisionAtMs || validUntilMs < decisionAtMs) {
            return UstrfU0DenseRiskEvidenceResolution.Unavailable(
                UstrfU0DenseRiskEvidenceFailure.INVALID_TIME_BINDING
            )
        }
        val score = max(
            evidence.routeIntrusionScore,
            evidence.maximumRouteCellRisk * config.localPeakWeight
        ).coerceIn(0f, 1f)
        val level = when {
            score >= config.highThreshold -> RiskLevel.HIGH
            score >= config.mediumThreshold -> RiskLevel.MEDIUM
            score >= config.lowThreshold -> RiskLevel.LOW
            else -> RiskLevel.NONE
        }
        val proximity = when (level) {
            RiskLevel.HIGH -> ProximityBand.CRITICAL
            RiskLevel.MEDIUM -> ProximityBand.NEAR
            RiskLevel.LOW -> ProximityBand.MID
            RiskLevel.NONE -> ProximityBand.FAR
        }
        val rawRisk = RiskResult(
            level = level,
            direction = if (level == RiskLevel.NONE) RiskDirection.NONE else RiskDirection.CENTER,
            message = if (level == RiskLevel.NONE) {
                "显式路线上的稠密风险证据未达到提醒阈值。"
            } else {
                "显式路线上存在稠密风险证据，请减速确认。"
            },
            sourceDetection = null,
            proximity = proximity,
            urgencyScore = score,
            riskScore = score,
            scoreBreakdown = RiskScoreBreakdown(
                total = score,
                fusionSummary = "${CONTRACT_ID}:${evidence.riskSources.sorted().joinToString("+")}"
            ),
            approachTrend = ApproachTrend.UNKNOWN,
            evidenceState = if (level == RiskLevel.NONE) {
                RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE
            } else {
                RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE
            }
        )
        return UstrfU0DenseRiskEvidenceResolution.Available(
            kernelEvidence = AssistRiskEvidenceFrame(
                sourceContractId = AssistDecisionKernel.RISK_EVIDENCE_INPUT_CONTRACT_ID,
                frameId = decisionFrameId,
                eventKey = "$episodeId:${evidence.routeIntentId}",
                observedAtMs = observedAtMs,
                validUntilMs = validUntilMs,
                rawRisk = rawRisk,
                evidenceCount = evidence.contributingCellCount
            ),
            normalizedRiskScore = score,
            sourceEvidence = evidence
        )
    }
}
