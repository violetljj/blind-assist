package com.linnan.blindassist.runtime

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.risk.ApproachTrend
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskEvidenceState
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.risk.RiskScoreBreakdown
import com.linnan.blindassist.session.AssistDecisionKernel
import com.linnan.blindassist.session.AssistFrameResult
import com.linnan.blindassist.session.AssistRiskEvidenceFrame
import com.linnan.blindassist.session.AssistSessionCoordinator
import com.linnan.blindassist.ustrf.UstrfImageObstacle
import com.linnan.blindassist.ustrf.UstrfImageRouteProxy
import com.linnan.blindassist.vision.DetectorFrameResult
import com.linnan.blindassist.vision.FrameClockDomain

/**
 * Authoritative only inside the separately packaged USTRF experiment build.
 *
 * The input is an image-plane detector/route proxy, not metric USTRF geometry. The legacy
 * RiskAnalyzer is bypassed, while the shared temporal/event/feedback kernel remains in use.
 */
internal class AssistUstrfExperimentalAdapter(
    private val coordinator: AssistSessionCoordinator,
    private val proxy: UstrfImageRouteProxy = UstrfImageRouteProxy()
) {
    fun process(
        frame: DetectorFrameResult,
        profile: AlertProfile,
        scenario: AssistScenario,
        nowMs: Long,
        decisionAtNs: Long
    ): AssistFrameResult {
        val evidence = buildEvidence(frame, nowMs, decisionAtNs)
        return coordinator.processRiskEvidence(
            evidence = evidence,
            frameSize = frame.frameSize,
            profile = profile,
            scenario = scenario,
            metrics = frame.metrics,
            nowMs = nowMs,
            sourceFrame = frame.sourceFrame,
            decisionAtNs = decisionAtNs
        )
    }

    private fun buildEvidence(
        frame: DetectorFrameResult,
        nowMs: Long,
        decisionAtNs: Long
    ): AssistRiskEvidenceFrame {
        val stamp = frame.sourceFrame
        val failure = when {
            frame.frameSize.width <= 0 || frame.frameSize.height <= 0 -> "invalid_frame_size"
            stamp == null -> "source_frame_missing"
            stamp.clockDomain != FrameClockDomain.ANDROID_ELAPSED_REALTIME -> "camera_clock_unmapped"
            decisionAtNs < stamp.capturedAtNs -> "decision_precedes_capture"
            else -> null
        }
        if (failure != null) return abstentionEvidence(frame, nowMs, decisionAtNs, failure)

        val width = frame.frameSize.width.toFloat()
        val height = frame.frameSize.height.toFloat()
        val obstacles = frame.detections.mapNotNull { detection ->
            val box = detection.boundingBox.clamped(frame.frameSize)
            if (box.width <= 0f || box.height <= 0f) return@mapNotNull null
            UstrfImageObstacle(
                left = (box.left / width).coerceIn(0f, 1f),
                top = (box.top / height).coerceIn(0f, 1f),
                right = (box.right / width).coerceIn(0f, 1f),
                bottom = (box.bottom / height).coerceIn(0f, 1f),
                confidence = detection.confidence.coerceIn(0f, 1f)
            )
        }
        val result = proxy.evaluate(obstacles)
        val score = result.routeRisk
        val level = levelFor(score)
        val risk = RiskResult(
            level = level,
            direction = if (level == RiskLevel.NONE) RiskDirection.NONE else RiskDirection.CENTER,
            message = messageFor(level),
            sourceDetection = null,
            proximity = proximityFor(level),
            urgencyScore = score,
            riskScore = score,
            scoreBreakdown = RiskScoreBreakdown(
                total = score,
                fusionSummary = "$CONTRACT_ID:route=${result.routeIntrusionCount}:best=${result.lowestProxyRiskCenterX}"
            ),
            approachTrend = ApproachTrend.UNKNOWN,
            evidenceState = if (level == RiskLevel.NONE) {
                RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE
            } else {
                RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE
            }
        )
        return AssistRiskEvidenceFrame(
            sourceContractId = AssistDecisionKernel.RISK_EVIDENCE_INPUT_CONTRACT_ID,
            frameId = stamp!!.frameId.toString(),
            eventKey = ROUTE_EVENT_KEY,
            observedAtMs = nowMs,
            validUntilMs = nowMs + EVIDENCE_TTL_MS,
            rawRisk = risk,
            evidenceCount = result.evidenceCount
        )
    }

    private fun abstentionEvidence(
        frame: DetectorFrameResult,
        nowMs: Long,
        decisionAtNs: Long,
        reason: String
    ): AssistRiskEvidenceFrame {
        val score = 1f
        return AssistRiskEvidenceFrame(
            sourceContractId = AssistDecisionKernel.RISK_EVIDENCE_INPUT_CONTRACT_ID,
            frameId = frame.sourceFrame?.frameId?.toString() ?: "missing-$decisionAtNs",
            eventKey = ABSTENTION_EVENT_KEY,
            observedAtMs = nowMs,
            validUntilMs = nowMs + EVIDENCE_TTL_MS,
            rawRisk = RiskResult(
                level = RiskLevel.HIGH,
                direction = RiskDirection.CENTER,
                message = "USTRF实验输入不完整，请停下重新扫描。",
                sourceDetection = null,
                proximity = ProximityBand.CRITICAL,
                urgencyScore = score,
                riskScore = score,
                scoreBreakdown = RiskScoreBreakdown(
                    total = score,
                    fusionSummary = "$CONTRACT_ID:abstain:$reason"
                ),
                approachTrend = ApproachTrend.UNKNOWN,
                evidenceState = RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE
            ),
            evidenceCount = 1
        )
    }

    private fun levelFor(score: Float): RiskLevel = when {
        score >= .75f -> RiskLevel.HIGH
        score >= .55f -> RiskLevel.MEDIUM
        score >= .30f -> RiskLevel.LOW
        else -> RiskLevel.NONE
    }

    private fun proximityFor(level: RiskLevel): ProximityBand = when (level) {
        RiskLevel.HIGH -> ProximityBand.CRITICAL
        RiskLevel.MEDIUM -> ProximityBand.NEAR
        RiskLevel.LOW -> ProximityBand.MID
        RiskLevel.NONE -> ProximityBand.FAR
    }

    private fun messageFor(level: RiskLevel): String = when (level) {
        RiskLevel.HIGH -> "USTRF实验：中心假设路线存在高代理风险，请停下确认。"
        RiskLevel.MEDIUM -> "USTRF实验：中心假设路线存在代理风险，请减速确认。"
        RiskLevel.LOW -> "USTRF实验：中心假设路线存在低代理风险，请减速观察。"
        RiskLevel.NONE -> "USTRF实验：未见路线代理风险；这不代表安全。"
    }

    private companion object {
        const val CONTRACT_ID = "ustrf_experiment_image_route_proxy_v1"
        const val ROUTE_EVENT_KEY = "ustrf-experiment:center-route"
        const val ABSTENTION_EVENT_KEY = "ustrf-experiment:abstain"
        const val EVIDENCE_TTL_MS = 500L
    }
}
