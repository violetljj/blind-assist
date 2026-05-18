package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
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
        scenario: AssistScenario = AssistScenario.GENERAL,
        metrics: DetectorMetrics,
        nowMs: Long = System.currentTimeMillis()
    ): AssistFrameEvaluation {
        val rawRisk = riskAnalyzer.analyze(detections, frameSize)
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
            evaluatedAtMs = nowMs
        )
    }

    fun completeFeedback(
        evaluation: AssistFrameEvaluation,
        feedbackDecision: FeedbackDecision
    ): AssistFrameResult {
        val displayDecision = feedbackDecision.withDisplayReason(
            displayReasonFor(evaluation.rawRisk, evaluation.stableRisk, feedbackDecision)
        )
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
        riskStabilizer.reset()
        trace.clear()
    }

    fun startSession(nowMs: Long = System.currentTimeMillis()) {
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
                detail = "检测到${evaluation.detectionCount}个目标，但稳定风险处于${riskText(stableRisk)}。",
                accessibilityText = "暂不提醒，目标距离较远，当前场景为$scenarioName。"
            )
            FeedbackReason.COOLDOWN -> RiskExplanation(
                headline = "暂不重复：提醒冷却中",
                detail = "${scenarioName}场景正在控制重复提醒频率，避免连续播报造成干扰。",
                accessibilityText = "暂不重复提醒，提醒冷却中，当前场景为$scenarioName。"
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
            FeedbackReason.NO_FEEDBACK_RISK -> RiskExplanation(
                headline = "继续观察：暂无可反馈风险",
                detail = if (evaluation.detectionCount > 0) {
                    "检测到${evaluation.detectionCount}个目标，但未达到近处或迫近提醒条件。"
                } else {
                    "当前帧未锁定主要目标。"
                },
                accessibilityText = "继续观察，暂无可反馈风险，当前场景为$scenarioName。"
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
            RiskLevel.NONE -> "无风险"
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
    val scenario: AssistScenario,
    val metrics: DetectorMetrics,
    val preliminaryReason: FeedbackReason,
    val evaluatedAtMs: Long
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
