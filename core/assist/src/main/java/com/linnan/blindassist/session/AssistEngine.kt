package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskAnalyzer
import com.linnan.blindassist.risk.RiskEvidenceState
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.risk.RiskStabilizer
import com.linnan.blindassist.risk.RiskEventSnapshot
import com.linnan.blindassist.risk.TemporalRiskTracker
import com.linnan.blindassist.vision.FrameStamp

class AssistEngine(
    private val riskAnalyzer: RiskAnalyzer = RiskAnalyzer(),
    private val riskStabilizer: RiskStabilizer = RiskStabilizer(),
    private val trace: SessionTrace = SessionTrace(),
    private val temporalRiskTracker: TemporalRiskTracker = TemporalRiskTracker()
) {
    fun evaluate(
        detections: List<Detection>,
        frameSize: FrameSize,
        profile: AlertProfile,
        scenario: AssistScenario = AssistScenario.GENERAL,
        metrics: DetectorMetrics,
        nowMs: Long = monotonicNowMs(),
        sourceFrame: FrameStamp? = null,
        decisionAtNs: Long = nowMs * NANOS_PER_MILLISECOND
    ): AssistFrameEvaluation {
        val preTemporalRisk = riskAnalyzer.analyze(detections, frameSize)
        val rawRisk = temporalRiskTracker.update(preTemporalRisk, nowMs)
        val stableRisk = riskStabilizer.update(rawRisk, profile, scenario, nowMs)
        return AssistFrameEvaluation(
            rawRisk = rawRisk,
            stableRisk = stableRisk,
            detectionCount = detections.size,
            frameSize = frameSize,
            profile = profile,
            scenario = scenario,
            metrics = metrics,
            preliminaryReason = displayReasonFor(rawRisk, stableRisk, null),
            evaluatedAtMs = nowMs,
            sourceFrame = sourceFrame,
            decisionAtNs = decisionAtNs,
            preTemporalRisk = preTemporalRisk
        )
    }

    /** Evaluates already-normalized, object-agnostic evidence through the shared post-analyzer chain. */
    fun evaluateRiskEvidence(
        rawRisk: RiskResult,
        eventKey: String,
        evidenceCount: Int,
        frameSize: FrameSize,
        profile: AlertProfile,
        scenario: AssistScenario = AssistScenario.GENERAL,
        metrics: DetectorMetrics,
        nowMs: Long = monotonicNowMs(),
        sourceFrame: FrameStamp? = null,
        decisionAtNs: Long = nowMs * NANOS_PER_MILLISECOND
    ): AssistFrameEvaluation {
        require(rawRisk.sourceDetection == null) { "object-agnostic risk evidence must not contain a detection" }
        require(evidenceCount > 0) { "object-agnostic risk evidence count must be positive" }
        val temporalRisk = temporalRiskTracker.updateExternalEvidence(rawRisk, eventKey, nowMs)
        val stableRisk = riskStabilizer.update(temporalRisk, profile, scenario, nowMs)
        return AssistFrameEvaluation(
            rawRisk = temporalRisk,
            stableRisk = stableRisk,
            detectionCount = 0,
            frameSize = frameSize,
            profile = profile,
            scenario = scenario,
            metrics = metrics,
            preliminaryReason = displayReasonFor(temporalRisk, stableRisk, null),
            evaluatedAtMs = nowMs,
            riskEvidenceCount = evidenceCount,
            sourceFrame = sourceFrame,
            decisionAtNs = decisionAtNs,
            preTemporalRisk = rawRisk
        )
    }

    fun completeFeedback(
        evaluation: AssistFrameEvaluation,
        feedbackDecision: FeedbackDecision
    ): AssistFrameResult {
        val displayDecision = if (feedbackDecision.reason == FeedbackReason.EVENT_ALREADY_ALERTED) {
            feedbackDecision
        } else {
            feedbackDecision.withDisplayReason(
                displayReasonFor(evaluation.rawRisk, evaluation.stableRisk, feedbackDecision)
            )
        }
        val explanation = explain(evaluation, displayDecision)
        val summary = trace.record(evaluation, displayDecision, explanation)
        return AssistFrameResult(
            evaluation = evaluation,
            feedbackDecision = displayDecision,
            explanation = explanation,
            sessionSummary = summary
        )
    }

    fun reset() {
        temporalRiskTracker.reset()
        riskStabilizer.reset()
        trace.clear()
    }

    fun startSession(nowMs: Long = monotonicNowMs()) {
        temporalRiskTracker.reset()
        riskStabilizer.reset()
        trace.start(nowMs)
    }

    fun sessionSummary(): SessionSummary {
        return trace.summary()
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
        if (rawRisk.evidenceState == RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE) {
            return FeedbackReason.NO_FEEDBACK_RISK
        }
        if (rawRisk.proximity == ProximityBand.FAR || rawRisk.proximity == ProximityBand.MID) {
            return FeedbackReason.DISTANCE_TOO_FAR
        }
        return feedbackDecision?.reason ?: FeedbackReason.NO_FEEDBACK_RISK
    }

    private fun explain(
        evaluation: AssistFrameEvaluation,
        feedbackDecision: FeedbackDecision
    ): RiskExplanation {
        val stableRisk = evaluation.stableRisk
        val rawRisk = evaluation.rawRisk
        val scenarioName = evaluation.scenario.displayName
        return when (feedbackDecision.reason) {
            FeedbackReason.TRIGGERED -> RiskExplanation(
                headline = "已按${scenarioName}策略触发提醒",
                detail = "稳定风险为${riskText(stableRisk)}，${scenarioName}场景会使用对应冷却和震动计划。",
                accessibilityText = "已触发提醒，稳定风险为${riskText(stableRisk)}，当前场景为$scenarioName。"
            )
            FeedbackReason.UNSTABLE_RISK -> RiskExplanation(
                headline = "暂不提醒：风险未稳定",
                detail = "${scenarioName}场景要求中风险连续确认；当前原始风险为${riskText(rawRisk)}。",
                accessibilityText = "暂不提醒，风险未稳定，当前场景为$scenarioName。"
            )
            FeedbackReason.DISTANCE_TOO_FAR -> RiskExplanation(
                headline = "暂不提醒：目标距离较远",
                detail = if (evaluation.riskEvidenceCount > 0) {
                    "融合${evaluation.riskEvidenceCount}个风险证据单元，但稳定风险处于${riskText(stableRisk)}。"
                } else {
                    "检测到${evaluation.detectionCount}个目标，但稳定风险处于${riskText(stableRisk)}。"
                },
                accessibilityText = "暂不提醒，目标距离较远，当前场景为$scenarioName。"
            )
            FeedbackReason.COOLDOWN -> RiskExplanation(
                headline = "暂不重复：提醒冷却中",
                detail = "${scenarioName}场景正在控制重复提醒频率，避免连续播报造成干扰。",
                accessibilityText = "暂不重复提醒，提醒冷却中，当前场景为$scenarioName。"
            )
            FeedbackReason.EVENT_ALREADY_ALERTED -> RiskExplanation(
                headline = "暂不重复：同一风险已提醒",
                detail = "该中心路径风险已完成一次反馈，等待通过、远离或清除后才会再次提醒。",
                accessibilityText = "同一风险已提醒，等待通过或清除后再判断是否需要提醒。"
            )
            FeedbackReason.DUAL_LOOP_CONTRADICTED -> RiskExplanation(
                headline = "本帧暂缓：多个检测框共同缩小",
                detail = "第二环只否决当前提醒机会；不解释运动责任，下一帧会重新判断。",
                accessibilityText = "本帧多个检测框共同缩小，与接近提醒矛盾，暂不提醒，下一帧重新判断。"
            )
            FeedbackReason.HELD_ALERT -> RiskExplanation(
                headline = "短暂保持上一条提醒",
                detail = "当前帧未重新确认目标，但稳定层仍在短时间内保留${riskText(stableRisk)}。",
                accessibilityText = "短暂保持上一条提醒，当前场景为$scenarioName。"
            )
            FeedbackReason.SPEECH_DISABLED,
            FeedbackReason.VIBRATION_DISABLED -> RiskExplanation(
                headline = "风险存在，但反馈开关关闭",
                detail = "稳定风险为${riskText(stableRisk)}，请确认语音或震动提醒是否开启。",
                accessibilityText = "风险存在，但反馈开关关闭，当前场景为$scenarioName。"
            )
            FeedbackReason.FEEDBACK_UNAVAILABLE -> RiskExplanation(
                headline = "风险存在，但反馈暂不可用",
                detail = "稳定风险为${riskText(stableRisk)}，系统未能确认语音或震动提醒已被设备接受。",
                accessibilityText = "风险存在，但语音或震动反馈暂不可用，当前场景为$scenarioName。"
            )
            FeedbackReason.NO_FEEDBACK_RISK -> RiskExplanation(
                headline = "持续检测中",
                detail = if (stableRisk.evidenceState == RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE) {
                    "检测到模型支持的目标，当前未达到提醒条件。"
                } else {
                    "当前未检测到达到提醒条件的支持目标，请继续确认周围环境。"
                },
                accessibilityText = "持续检测中，请继续确认周围环境，当前场景为$scenarioName。"
            )
        }
    }

    private fun riskText(risk: RiskResult): String {
        return "${levelText(risk.level)} ${directionText(risk.direction)} ${proximityText(risk.proximity)}"
    }

    private fun levelText(level: RiskLevel): String {
        return when (level) {
            RiskLevel.HIGH -> "高风险"
            RiskLevel.MEDIUM -> "中风险"
            RiskLevel.LOW -> "低风险"
            RiskLevel.NONE -> "未达提醒等级"
        }
    }

    private fun directionText(direction: com.linnan.blindassist.risk.RiskDirection): String {
        return when (direction) {
            com.linnan.blindassist.risk.RiskDirection.LEFT -> "左前"
            com.linnan.blindassist.risk.RiskDirection.CENTER -> "正前"
            com.linnan.blindassist.risk.RiskDirection.RIGHT -> "右前"
            com.linnan.blindassist.risk.RiskDirection.NONE -> "无方向"
        }
    }

    private fun proximityText(proximity: ProximityBand): String {
        return when (proximity) {
            ProximityBand.CRITICAL -> "迫近"
            ProximityBand.NEAR -> "近处"
            ProximityBand.MID -> "中距"
            ProximityBand.FAR -> "远处"
        }
    }

    private companion object {
        const val NANOS_PER_MILLISECOND = 1_000_000L
        fun monotonicNowMs(): Long = System.nanoTime() / NANOS_PER_MILLISECOND
    }
}

data class DetectorMetrics(
    val totalMs: Long,
    val preprocessMs: Long,
    val inferenceMs: Long,
    val postprocessMs: Long,
    val fps: Float,
    val modelStatus: String,
    val droppedFrameRate: Float = 0f,
    val inferenceP50Ms: Long = 0L,
    val inferenceP95Ms: Long = 0L
) {
    constructor(
        totalMs: Long,
        preprocessMs: Long,
        inferenceMs: Long,
        postprocessMs: Long,
        fps: Float,
        modelStatus: String
    ) : this(
        totalMs = totalMs,
        preprocessMs = preprocessMs,
        inferenceMs = inferenceMs,
        postprocessMs = postprocessMs,
        fps = fps,
        modelStatus = modelStatus,
        droppedFrameRate = 0f,
        inferenceP50Ms = 0L,
        inferenceP95Ms = 0L
    )
}

data class AssistFrameEvaluation(
    val rawRisk: RiskResult,
    val stableRisk: RiskResult,
    val detectionCount: Int,
    val frameSize: FrameSize,
    val profile: AlertProfile,
    val scenario: AssistScenario,
    val metrics: DetectorMetrics,
    val preliminaryReason: FeedbackReason,
    val evaluatedAtMs: Long,
    val riskEvent: RiskEventSnapshot = RiskEventSnapshot.none(),
    val riskEvidenceCount: Int = 0,
    val sourceFrame: FrameStamp? = null,
    val decisionAtNs: Long = evaluatedAtMs * 1_000_000L,
    /** Exact analyzer output before temporal tracking, exposed for paired replay audit. */
    val preTemporalRisk: RiskResult = rawRisk,
    /** Non-actuating second-loop admission result; never changes the delivered risk path. */
    val dualLoopShadow: DualLoopShadowObservation = DualLoopShadowObservation.off()
)

data class AssistFrameResult(
    val evaluation: AssistFrameEvaluation,
    val feedbackDecision: FeedbackDecision,
    val explanation: RiskExplanation,
    val sessionSummary: SessionSummary
)

data class RiskExplanation(
    val headline: String,
    val detail: String,
    val accessibilityText: String
)
