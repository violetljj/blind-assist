package com.linnan.blindassist.session

import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.localization.LocalizedText
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.vision.FrameStamp
import java.util.Locale

class SessionTrace(private val capacity: Int = DEFAULT_CAPACITY) {
    private val frames = ArrayDeque<TraceFrame>()
    private var startedAtMs: Long? = null
    private var latestAtMs: Long? = null

    fun start(nowMs: Long) {
        frames.clear()
        startedAtMs = nowMs
        latestAtMs = nowMs
    }

    fun record(
        evaluation: AssistFrameEvaluation,
        feedbackDecision: FeedbackDecision,
        explanation: RiskExplanation
    ): SessionSummary {
        if (startedAtMs == null) {
            startedAtMs = evaluation.decisionAtNs / NANOS_PER_MILLISECOND
        }
        latestAtMs = evaluation.decisionAtNs / NANOS_PER_MILLISECOND
        if (frames.size == capacity) {
            frames.removeFirst()
        }
        frames.addLast(
            TraceFrame(
                level = evaluation.stableRisk.level,
                proximity = evaluation.stableRisk.proximity,
                fps = evaluation.metrics.fps,
                inferenceMs = evaluation.metrics.inferenceMs,
                feedbackReason = feedbackDecision.reason,
                explanationHeadline = explanation.headline,
                speechTriggered = feedbackDecision.speechTriggered,
                vibrationTriggered = feedbackDecision.vibrationTriggered,
                sourceFrame = evaluation.sourceFrame,
                decisionAtNs = evaluation.decisionAtNs
            )
        )
        return summary()
    }

    fun clear() {
        frames.clear()
        startedAtMs = null
        latestAtMs = null
    }

    fun summary(): SessionSummary {
        val started = startedAtMs
        if (frames.isEmpty()) {
            return SessionSummary.empty(started)
        }

        var high = 0
        var medium = 0
        var low = 0
        var none = 0
        var critical = 0
        var risky = 0
        var speech = 0
        var vibration = 0
        var fpsTotal = 0f
        var inferenceTotal = 0L
        frames.forEach { frame ->
            when (frame.level) {
                RiskLevel.HIGH -> high += 1
                RiskLevel.MEDIUM -> medium += 1
                RiskLevel.LOW -> low += 1
                RiskLevel.NONE -> none += 1
            }
            if (frame.level != RiskLevel.NONE) {
                risky += 1
            }
            if (frame.proximity == ProximityBand.CRITICAL) {
                critical += 1
            }
            if (frame.speechTriggered) {
                speech += 1
            }
            if (frame.vibrationTriggered) {
                vibration += 1
            }
            fpsTotal += frame.fps
            inferenceTotal += frame.inferenceMs
        }

        val count = frames.size
        val durationMs = if (started == null) 0L else ((latestAtMs ?: started) - started).coerceAtLeast(0L)
        return SessionSummary(
            frameCount = count,
            highCount = high,
            mediumCount = medium,
            lowCount = low,
            noneCount = none,
            riskyFrameCount = risky,
            criticalCount = critical,
            speechTriggerCount = speech,
            vibrationTriggerCount = vibration,
            durationMs = durationMs,
            hasStarted = started != null,
            averageFps = fpsTotal / count.toFloat(),
            averageInferenceMs = inferenceTotal / count,
            latestFeedbackReason = frames.last().feedbackReason,
            latestExplanation = frames.last().explanationHeadline
        )
    }

    private data class TraceFrame(
        val level: RiskLevel,
        val proximity: ProximityBand,
        val fps: Float,
        val inferenceMs: Long,
        val feedbackReason: FeedbackReason,
        val explanationHeadline: String,
        val speechTriggered: Boolean,
        val vibrationTriggered: Boolean,
        val sourceFrame: FrameStamp?,
        val decisionAtNs: Long
    )

    companion object {
        private const val NANOS_PER_MILLISECOND = 1_000_000L
        const val DEFAULT_CAPACITY = 30
    }
}

data class SessionSummary(
    val frameCount: Int,
    val highCount: Int,
    val mediumCount: Int,
    val lowCount: Int,
    val noneCount: Int,
    val riskyFrameCount: Int,
    val criticalCount: Int,
    val speechTriggerCount: Int,
    val vibrationTriggerCount: Int,
    val durationMs: Long,
    val hasStarted: Boolean,
    val averageFps: Float,
    val averageInferenceMs: Long,
    val latestFeedbackReason: FeedbackReason,
    val latestExplanation: String
) {
    val feedbackTriggerCount: Int
        get() = speechTriggerCount + vibrationTriggerCount

    fun displayText(language: AppLanguage = AppLanguage.ZH): String {
        return if (language == AppLanguage.EN) {
            "Session: ${durationText(language)} · Last $frameCount frames high/medium/low/none $highCount/$mediumCount/$lowCount/$noneCount · " +
                "Feedback ${latestFeedbackReason.displayText(language)} · " +
                "Speech/vibration $speechTriggerCount/$vibrationTriggerCount · " +
                "Avg FPS ${String.format(Locale.US, "%.1f", averageFps)} · " +
                "Avg inference ${averageInferenceMs}ms"
        } else {
            "会话：${durationText()} · 最近${frameCount}帧 高/中/低/无 $highCount/$mediumCount/$lowCount/$noneCount · " +
                "反馈 ${latestFeedbackReason.displayText} · " +
                "语音/震动 $speechTriggerCount/$vibrationTriggerCount · " +
                "平均FPS ${String.format(Locale.US, "%.1f", averageFps)} · " +
                "平均推理 ${averageInferenceMs}ms"
        }
    }

    fun fieldTestText(profileName: String, scenarioName: String, language: AppLanguage = AppLanguage.ZH): String {
        return if (language == AppLanguage.EN) {
            "Runtime: ${durationText(language)}\n" +
                "Last $frameCount frames: $riskyFrameCount risky, $criticalCount critical, high/medium/low/none $highCount/$mediumCount/$lowCount/$noneCount\n" +
                "Reminders: speech $speechTriggerCount, vibration $vibrationTriggerCount\n" +
                "Average performance: FPS ${String.format(Locale.US, "%.1f", averageFps)}, inference ${averageInferenceMs}ms\n" +
                "Current profile: $profileName\n" +
                "Current scenario: $scenarioName\n" +
                "Latest explanation: ${latestFeedbackReason.displayText(language)}"
        } else {
            "运行时长：${durationText()}\n" +
                "最近${frameCount}帧：风险${riskyFrameCount}次，迫近${criticalCount}次，高/中/低/无 $highCount/$mediumCount/$lowCount/$noneCount\n" +
                "提醒触发：语音${speechTriggerCount}次，震动${vibrationTriggerCount}次\n" +
                "平均性能：FPS ${String.format(Locale.US, "%.1f", averageFps)}，推理 ${averageInferenceMs}ms\n" +
                "当前档位：$profileName\n" +
                "当前场景：$scenarioName\n" +
                "最近解释：$latestExplanation"
        }
    }

    fun durationText(language: AppLanguage = AppLanguage.ZH): String {
        return LocalizedText.durationText(hasStarted, durationMs, language)
    }

    companion object {
        fun empty(startedAtMs: Long? = null): SessionSummary {
            return SessionSummary(
                frameCount = 0,
                highCount = 0,
                mediumCount = 0,
                lowCount = 0,
                noneCount = 0,
                riskyFrameCount = 0,
                criticalCount = 0,
                speechTriggerCount = 0,
                vibrationTriggerCount = 0,
                durationMs = 0L,
                hasStarted = startedAtMs != null,
                averageFps = 0f,
                averageInferenceMs = 0L,
                latestFeedbackReason = FeedbackReason.NO_FEEDBACK_RISK,
                latestExplanation = "暂无风险解释"
            )
        }
    }
}
