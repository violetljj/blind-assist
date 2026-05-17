package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskAnalyzer
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.risk.RiskStabilizer

class AssistEngine(
    private val riskAnalyzer: RiskAnalyzer = RiskAnalyzer(),
    private val riskStabilizer: RiskStabilizer = RiskStabilizer(),
    private val trace: SessionTrace = SessionTrace()
) {
    fun evaluate(
        detections: List<Detection>,
        frameSize: FrameSize,
        profile: AlertProfile,
        metrics: DetectorMetrics,
        nowMs: Long = System.currentTimeMillis()
    ): AssistFrameEvaluation {
        val rawRisk = riskAnalyzer.analyze(detections, frameSize)
        val stableRisk = riskStabilizer.update(rawRisk, profile, nowMs)
        return AssistFrameEvaluation(
            rawRisk = rawRisk,
            stableRisk = stableRisk,
            detectionCount = detections.size,
            frameSize = frameSize,
            profile = profile,
            metrics = metrics,
            preliminaryReason = displayReasonFor(rawRisk, stableRisk, null)
        )
    }

    fun completeFeedback(
        evaluation: AssistFrameEvaluation,
        feedbackDecision: FeedbackDecision
    ): AssistFrameResult {
        val displayDecision = feedbackDecision.withDisplayReason(
            displayReasonFor(evaluation.rawRisk, evaluation.stableRisk, feedbackDecision)
        )
        val summary = trace.record(evaluation, displayDecision)
        return AssistFrameResult(
            evaluation = evaluation,
            feedbackDecision = displayDecision,
            sessionSummary = summary
        )
    }

    fun reset() {
        riskStabilizer.reset()
        trace.clear()
    }

    private fun displayReasonFor(
        rawRisk: RiskResult,
        stableRisk: RiskResult,
        feedbackDecision: FeedbackDecision?
    ): FeedbackReason {
        if (rawRisk.level == RiskLevel.MEDIUM && stableRisk.level == RiskLevel.NONE) {
            return FeedbackReason.UNSTABLE_RISK
        }
        if (rawRisk.level == RiskLevel.NONE && stableRisk.level != RiskLevel.NONE) {
            return FeedbackReason.HELD_ALERT
        }
        if (rawRisk.proximity == ProximityBand.FAR || rawRisk.proximity == ProximityBand.MID) {
            return FeedbackReason.DISTANCE_TOO_FAR
        }
        return feedbackDecision?.reason ?: FeedbackReason.NO_FEEDBACK_RISK
    }
}

data class DetectorMetrics(
    val totalMs: Long,
    val preprocessMs: Long,
    val inferenceMs: Long,
    val postprocessMs: Long,
    val fps: Float,
    val modelStatus: String
)

data class AssistFrameEvaluation(
    val rawRisk: RiskResult,
    val stableRisk: RiskResult,
    val detectionCount: Int,
    val frameSize: FrameSize,
    val profile: AlertProfile,
    val metrics: DetectorMetrics,
    val preliminaryReason: FeedbackReason
)

data class AssistFrameResult(
    val evaluation: AssistFrameEvaluation,
    val feedbackDecision: FeedbackDecision,
    val sessionSummary: SessionSummary
)
