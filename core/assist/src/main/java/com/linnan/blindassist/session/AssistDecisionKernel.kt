package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackPlanner
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.RiskEventTracker
import com.linnan.blindassist.risk.DtrSignal
import com.linnan.blindassist.risk.ApproachTrend
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskEvidenceState
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.vision.FrameClockDomain
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

data class AssistDtrEvidenceFrame(
    val sourceContractId: String,
    val frameId: String,
    val eventKey: String,
    val signal: DtrSignal,
    val observedAtMs: Long,
    val validUntilMs: Long,
    val risk: RiskResult,
    val evidenceCount: Int
) {
    init {
        require(sourceContractId.isNotBlank() && frameId.isNotBlank() && eventKey.isNotBlank())
        require(observedAtMs >= 0L && validUntilMs >= observedAtMs)
        require(evidenceCount > 0)
        require(risk.sourceDetection == null && risk.distanceEvidence == null)
        require(risk.approachTrend == ApproachTrend.UNKNOWN)
        require(risk.riskScore.isFinite() && risk.riskScore in 0f..1f)
        require(risk.urgencyScore.isFinite() && risk.urgencyScore in 0f..1f)
        require(kotlin.math.abs(risk.scoreBreakdown.total - risk.riskScore) <= SCORE_EPSILON)
        when (signal) {
            DtrSignal.ONSET,
            DtrSignal.HOLD -> {
                require(risk.level != RiskLevel.NONE)
                require(risk.direction == RiskDirection.CENTER)
                require(risk.evidenceState == RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE)
            }
            DtrSignal.CLEAR -> {
                require(risk.level == RiskLevel.NONE)
                require(risk.direction == RiskDirection.NONE)
                require(risk.evidenceState == RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE)
            }
            DtrSignal.UNKNOWN -> Unit
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
        LowConfidenceSidePersonConfirmation(),
    private val dualLoopShadowAdmitter: DualLoopShadowAdmitter =
        DualLoopShadowAdmitter.liveTristateSource(),
    private val dualLoopGeometryProducer: CausalTrackTristateGeometryProducer =
        CausalTrackTristateGeometryProducer(),
    private val dualLoopSceneScaleProducer: CausalSceneScaleTristateGeometryProducer =
        CausalSceneScaleTristateGeometryProducer(),
    private val dualLoopSceneScaleBidirectionalProducer:
        CausalSceneScaleTristateGeometryProducer =
        CausalSceneScaleTristateGeometryProducer.bidirectional()
) {
    private var dualLoopContradictUntilNs = Long.MIN_VALUE

    fun startSession(nowMs: Long = monotonicNowMs()) {
        assistEngine.startSession(nowMs)
        riskEventTracker.reset()
        lowConfidenceSidePersonConfirmation.reset()
        dualLoopGeometryProducer.reset()
        dualLoopSceneScaleProducer.reset()
        dualLoopSceneScaleBidirectionalProducer.reset()
        dualLoopContradictUntilNs = Long.MIN_VALUE
    }

    fun reset() {
        assistEngine.reset()
        riskEventTracker.reset()
        lowConfidenceSidePersonConfirmation.reset()
        dualLoopGeometryProducer.reset()
        dualLoopSceneScaleProducer.reset()
        dualLoopSceneScaleBidirectionalProducer.reset()
        dualLoopContradictUntilNs = Long.MIN_VALUE
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
        decisionAtNs: Long = nowMs * NANOS_PER_MILLISECOND,
        dualLoopMode: DualLoopRuntimeMode = DualLoopRuntimeMode.OFF,
        dualLoopGeometryEvidence: DualLoopGeometryEvidence? = null,
        dualLoopDecisionClockDomain: FrameClockDomain? = sourceFrame?.clockDomain
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
        val shadowEvidence = dualLoopGeometryEvidence ?: when (dualLoopMode) {
            DualLoopRuntimeMode.OFF -> null
            DualLoopRuntimeMode.SHADOW_ABSTAIN_ONLY ->
                dualLoopGeometryProducer.produce(
                    sourceFrame = sourceFrame,
                    selectedTarget = evaluation.rawRisk.sourceDetection,
                    decisionAtNs = decisionAtNs
                )
            DualLoopRuntimeMode.ACTIVE_CONTRADICT_ONLY,
            DualLoopRuntimeMode.ACTIVE_CONTRADICT_TTL ->
                dualLoopSceneScaleProducer.produce(
                    sourceFrame = sourceFrame,
                    detections = detections,
                    selectedTarget = evaluation.rawRisk.sourceDetection,
                    decisionAtNs = decisionAtNs
                )
            DualLoopRuntimeMode.ACTIVE_CONTRADICT_TTL_CONFIRM_RELEASE ->
                dualLoopSceneScaleBidirectionalProducer.produce(
                    sourceFrame = sourceFrame,
                    detections = detections,
                    selectedTarget = evaluation.rawRisk.sourceDetection,
                    decisionAtNs = decisionAtNs
                )
        }
        val admittedShadow = dualLoopShadowAdmitter.evaluate(
            mode = dualLoopMode,
            evidence = shadowEvidence,
            sourceFrame = sourceFrame,
            detections = detections,
            baselineRisk = evaluation.rawRisk,
            decisionAtNs = decisionAtNs,
            decisionClockDomain = dualLoopDecisionClockDomain
        )
        val dualLoopShadow = if (
            admittedShadow.disposition == DualLoopShadowDisposition.EVIDENCE_ABSENT &&
            sourceFrame != null
        ) {
            admittedShadow.copy(currentFrameId = sourceFrame.frameId)
        } else {
            admittedShadow
        }
        val shadowEvaluation = evaluation.copy(dualLoopShadow = dualLoopShadow)
        // Event identity and exit must observe the current frame. Feedback still uses
        // stableRisk below so the alert hold cannot conceal raw target disappearance.
        val event = riskEventTracker.update(shadowEvaluation.rawRisk, nowMs)
        val eventEvaluation = shadowEvaluation.copy(riskEvent = event)
        val directContradiction =
            dualLoopMode in setOf(
                DualLoopRuntimeMode.ACTIVE_CONTRADICT_ONLY,
                DualLoopRuntimeMode.ACTIVE_CONTRADICT_TTL,
                DualLoopRuntimeMode.ACTIVE_CONTRADICT_TTL_CONFIRM_RELEASE
            ) &&
                dualLoopShadow.admitted &&
                dualLoopShadow.correctionDecision ==
                DualLoopCorrectionDecision.CONTRADICT_APPROACH
        val dualLoopContradicts = when (dualLoopMode) {
            DualLoopRuntimeMode.ACTIVE_CONTRADICT_ONLY -> {
                dualLoopContradictUntilNs = Long.MIN_VALUE
                directContradiction
            }
            DualLoopRuntimeMode.ACTIVE_CONTRADICT_TTL -> {
                if (directContradiction) {
                    dualLoopContradictUntilNs = saturatingAdd(
                        decisionAtNs,
                        DUAL_LOOP_CONTRADICT_HOLD_NS
                    )
                }
                decisionAtNs <= dualLoopContradictUntilNs
            }
            DualLoopRuntimeMode.ACTIVE_CONTRADICT_TTL_CONFIRM_RELEASE -> {
                when {
                    directContradiction -> {
                        dualLoopContradictUntilNs = saturatingAdd(
                            decisionAtNs,
                            DUAL_LOOP_CONTRADICT_HOLD_NS
                        )
                    }
                    dualLoopShadow.admitted &&
                        dualLoopShadow.correctionDecision ==
                        DualLoopCorrectionDecision.CONFIRM_APPROACH -> {
                        dualLoopContradictUntilNs = Long.MIN_VALUE
                    }
                }
                decisionAtNs <= dualLoopContradictUntilNs
            }
            else -> {
                dualLoopContradictUntilNs = Long.MIN_VALUE
                false
            }
        }
        val feedbackDecision = decideFeedback(
            eventEvaluation,
            feedbackGateway,
            requiresActiveEvent =
                eventEvaluation.stableRisk.sourceDetection?.source == DetectionSource.SEGMENTATION,
            dualLoopContradicts = dualLoopContradicts
        )
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
        val event = riskEventTracker.updateExternalEvidence(
            evaluation.rawRisk,
            evidence.eventKey,
            nowMs
        )
        val eventEvaluation = evaluation.copy(riskEvent = event)
        val feedbackDecision = decideFeedback(
            eventEvaluation,
            feedbackGateway,
            requiresActiveEvent = true
        )
        val completedEvent = riskEventTracker.recordFeedback(event, feedbackDecision)
        return assistEngine.completeFeedback(
            eventEvaluation.copy(riskEvent = completedEvent),
            feedbackDecision
        )
    }

    /** Shared feedback/effect path for a DTR-owned ONSET/HOLD/CLEAR/UNKNOWN lifecycle. */
    fun processDtrEvidence(
        evidence: AssistDtrEvidenceFrame,
        frameSize: FrameSize,
        profile: AlertProfile,
        scenario: AssistScenario,
        metrics: DetectorMetrics,
        feedbackGateway: FeedbackGateway,
        nowMs: Long,
        sourceFrame: FrameStamp? = null,
        decisionAtNs: Long = nowMs * NANOS_PER_MILLISECOND
    ): AssistFrameResult {
        require(evidence.sourceContractId == DTR_EVIDENCE_INPUT_CONTRACT_ID) {
            "unsupported DTR evidence input contract"
        }
        require(evidence.observedAtMs == nowMs) { "DTR evidence must bind to the current frame" }
        require(nowMs <= evidence.validUntilMs) { "DTR evidence is stale" }
        val evaluation = assistEngine.evaluateDtrEvidence(
            risk = evidence.risk,
            evidenceCount = evidence.evidenceCount,
            frameSize = frameSize,
            profile = profile,
            scenario = scenario,
            metrics = metrics,
            nowMs = nowMs,
            sourceFrame = sourceFrame,
            decisionAtNs = decisionAtNs
        )
        val event = riskEventTracker.updateExternalSignal(
            eventKey = evidence.eventKey,
            signal = evidence.signal,
            nowMs = nowMs
        )
        val eventEvaluation = evaluation.copy(riskEvent = event)
        val feedbackDecision = when (evidence.signal) {
            DtrSignal.ONSET,
            DtrSignal.HOLD -> decideFeedback(
                eventEvaluation,
                feedbackGateway,
                requiresActiveEvent = true
            )
            DtrSignal.UNKNOWN -> FeedbackDecision(
                plan = null,
                triggered = false,
                reason = FeedbackReason.UNSTABLE_RISK
            )
            DtrSignal.CLEAR -> FeedbackDecision(
                plan = null,
                triggered = false,
                reason = FeedbackReason.NO_FEEDBACK_RISK
            )
        }
        val completedEvent = if (evidence.signal == DtrSignal.CLEAR) {
            event
        } else {
            riskEventTracker.recordFeedback(event, feedbackDecision)
        }
        return assistEngine.completeFeedback(
            eventEvaluation.copy(riskEvent = completedEvent),
            feedbackDecision
        )
    }

    private fun decideFeedback(
        evaluation: AssistFrameEvaluation,
        feedbackGateway: FeedbackGateway,
        requiresActiveEvent: Boolean,
        dualLoopContradicts: Boolean = false
    ): FeedbackDecision {
        val event = evaluation.riskEvent
        return when {
            event.suppressesFeedback -> FeedbackDecision(
                plan = null,
                triggered = false,
                reason = FeedbackReason.EVENT_ALREADY_ALERTED
            )
            requiresActiveEvent && !event.active -> FeedbackDecision(
                plan = null,
                triggered = false,
                reason = FeedbackReason.UNSTABLE_RISK
            )
            !lowConfidenceSidePersonConfirmation.isConfirmed(evaluation.stableRisk) -> FeedbackDecision(
                plan = null,
                triggered = false,
                reason = FeedbackReason.UNSTABLE_RISK
            )
            dualLoopContradicts &&
                FeedbackPlanner.planFor(
                    risk = evaluation.stableRisk,
                    profile = evaluation.profile,
                    scenario = evaluation.scenario
                ) != null -> FeedbackDecision(
                plan = null,
                triggered = false,
                reason = FeedbackReason.DUAL_LOOP_CONTRADICTED
            )
            else -> feedbackGateway.notify(
                evaluation.stableRisk,
                evaluation.profile,
                evaluation.scenario
            )
        }
    }

    private fun saturatingAdd(value: Long, increment: Long): Long =
        if (value > Long.MAX_VALUE - increment) Long.MAX_VALUE else value + increment

    companion object {
        private const val NANOS_PER_MILLISECOND = 1_000_000L
        private const val DUAL_LOOP_CONTRADICT_HOLD_NS = 250_000_000L
        private fun monotonicNowMs(): Long = System.nanoTime() / NANOS_PER_MILLISECOND
        const val CONTRACT_ID = "blindassist_shared_decision_kernel_v1"
        const val RISK_EVIDENCE_INPUT_CONTRACT_ID = "blindassist_shared_decision_kernel_risk_evidence_input_v1"
        const val DTR_EVIDENCE_INPUT_CONTRACT_ID = "blindassist_dtr_event_evidence_input_v1"
    }
}
