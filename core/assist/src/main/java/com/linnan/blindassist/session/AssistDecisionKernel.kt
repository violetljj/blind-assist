package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.RiskEventTracker
import com.linnan.blindassist.risk.ApproachTrend
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskEvidenceState
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.vision.FrameStamp

data class AssistRiskEvidenceFrame(
    val sourceContractId: String,
    val frameId: String,
    val eventKey: String,
    val observedAtMs: Long,
    val validUntilMs: Long,
    val rawRisk: RiskResult,
    val evidenceCount: Int
) {
    init {
        require(sourceContractId.isNotBlank() && frameId.isNotBlank() && eventKey.isNotBlank())
        require(observedAtMs >= 0L && validUntilMs >= observedAtMs)
        require(evidenceCount > 0)
        require(rawRisk.message.isNotBlank())
        require(rawRisk.sourceDetection == null) { "risk-evidence input must remain object agnostic" }
        require(rawRisk.distanceEvidence == null) {
            "risk-evidence input cannot inject detection-relative distance evidence"
        }
        require(rawRisk.riskScore.isFinite() && rawRisk.riskScore in 0f..1f)
        require(rawRisk.urgencyScore.isFinite() && rawRisk.urgencyScore in 0f..1f)
        require(rawRisk.scoreBreakdown.total.isFinite() && rawRisk.scoreBreakdown.total in 0f..1f)
        require(kotlin.math.abs(rawRisk.scoreBreakdown.total - rawRisk.riskScore) <= SCORE_EPSILON) {
            "risk-evidence score breakdown must bind to the normalized risk score"
        }
        require(rawRisk.approachTrend == ApproachTrend.UNKNOWN) {
            "risk-evidence input cannot inject a lifecycle trend"
        }
        if (rawRisk.level == RiskLevel.NONE) {
            require(rawRisk.direction == RiskDirection.NONE)
            require(rawRisk.proximity == com.linnan.blindassist.risk.ProximityBand.FAR)
            require(rawRisk.evidenceState == RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE)
        } else {
            require(rawRisk.direction == RiskDirection.CENTER) {
                "route-relative risk evidence must use CENTER direction"
            }
            require(rawRisk.evidenceState == RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE)
        }
    }

    private companion object {
        const val SCORE_EPSILON = 1e-6f
    }
}

/**
 * Shared production/benchmark decision path from normalized detections to one audited frame result.
 *
 * The gateway remains an effect boundary: production may apply device availability and cooldown,
 * while an isolated benchmark may return a deterministic planner decision. Both callers still use
 * the same temporal risk, stabilization, event, confirmation, and feedback-gating order here.
 */
class AssistDecisionKernel(
    private val assistEngine: AssistEngine = AssistEngine(),
    private val riskEventTracker: RiskEventTracker = RiskEventTracker(),
    private val lowConfidenceSidePersonConfirmation: LowConfidenceSidePersonConfirmation =
        LowConfidenceSidePersonConfirmation()
) {
    fun startSession(nowMs: Long = monotonicNowMs()) {
        assistEngine.startSession(nowMs)
        riskEventTracker.reset()
        lowConfidenceSidePersonConfirmation.reset()
    }

    fun reset() {
        assistEngine.reset()
        riskEventTracker.reset()
        lowConfidenceSidePersonConfirmation.reset()
    }

    fun sessionSummary(): SessionSummary = assistEngine.sessionSummary()

    fun processFrame(
        detections: List<Detection>,
        frameSize: FrameSize,
        profile: AlertProfile,
        scenario: AssistScenario,
        metrics: DetectorMetrics,
        feedbackGateway: FeedbackGateway,
        nowMs: Long = monotonicNowMs(),
        sourceFrame: FrameStamp? = null,
        decisionAtNs: Long = nowMs * NANOS_PER_MILLISECOND
    ): AssistFrameResult {
        val evaluation = assistEngine.evaluate(
            detections = detections,
            frameSize = frameSize,
            profile = profile,
            scenario = scenario,
            metrics = metrics,
            nowMs = nowMs,
            sourceFrame = sourceFrame,
            decisionAtNs = decisionAtNs
        )
        val event = riskEventTracker.update(evaluation.stableRisk, nowMs)
        val eventEvaluation = evaluation.copy(riskEvent = event)
        val feedbackDecision = decideFeedback(eventEvaluation, feedbackGateway)
        val completedEvent = riskEventTracker.recordFeedback(event, feedbackDecision)
        return assistEngine.completeFeedback(
            eventEvaluation.copy(riskEvent = completedEvent),
            feedbackDecision
        )
    }

    /**
     * Shared post-analyzer path for frame-bound, object-agnostic risk evidence.
     *
     * The caller may normalize a dense field into [RiskResult], but cannot inject a
     * detection, temporal trend, event ID, stabilization result, or feedback decision.
     */
    fun processRiskEvidence(
        evidence: AssistRiskEvidenceFrame,
        frameSize: FrameSize,
        profile: AlertProfile,
        scenario: AssistScenario,
        metrics: DetectorMetrics,
        feedbackGateway: FeedbackGateway,
        nowMs: Long,
        sourceFrame: FrameStamp? = null,
        decisionAtNs: Long = nowMs * NANOS_PER_MILLISECOND
    ): AssistFrameResult {
        require(evidence.sourceContractId == RISK_EVIDENCE_INPUT_CONTRACT_ID) {
            "unsupported risk-evidence input contract"
        }
        require(evidence.observedAtMs == nowMs) { "risk evidence must bind to the current decision frame" }
        require(nowMs <= evidence.validUntilMs) { "risk evidence is stale" }
        val evaluation = assistEngine.evaluateRiskEvidence(
            rawRisk = evidence.rawRisk,
            eventKey = evidence.eventKey,
            evidenceCount = evidence.evidenceCount,
            frameSize = frameSize,
            profile = profile,
            scenario = scenario,
            metrics = metrics,
            nowMs = nowMs,
            sourceFrame = sourceFrame,
            decisionAtNs = decisionAtNs
        )
        val event = riskEventTracker.updateExternalEvidence(evaluation.stableRisk, evidence.eventKey, nowMs)
        val eventEvaluation = evaluation.copy(riskEvent = event)
        val feedbackDecision = decideFeedback(eventEvaluation, feedbackGateway)
        val completedEvent = riskEventTracker.recordFeedback(event, feedbackDecision)
        return assistEngine.completeFeedback(
            eventEvaluation.copy(riskEvent = completedEvent),
            feedbackDecision
        )
    }

    private fun decideFeedback(
        evaluation: AssistFrameEvaluation,
        feedbackGateway: FeedbackGateway
    ): FeedbackDecision {
        val event = evaluation.riskEvent
        return when {
            event.suppressesFeedback -> FeedbackDecision(
                plan = null,
                triggered = false,
                reason = FeedbackReason.EVENT_ALREADY_ALERTED
            )
            !lowConfidenceSidePersonConfirmation.isConfirmed(evaluation.stableRisk) -> FeedbackDecision(
                plan = null,
                triggered = false,
                reason = FeedbackReason.UNSTABLE_RISK
            )
            else -> feedbackGateway.notify(
                evaluation.stableRisk,
                evaluation.profile,
                evaluation.scenario
            )
        }
    }

    companion object {
        private const val NANOS_PER_MILLISECOND = 1_000_000L
        private fun monotonicNowMs(): Long = System.nanoTime() / NANOS_PER_MILLISECOND
        const val CONTRACT_ID = "blindassist_shared_decision_kernel_v1"
        const val RISK_EVIDENCE_INPUT_CONTRACT_ID = "blindassist_shared_decision_kernel_risk_evidence_input_v1"
    }
}
