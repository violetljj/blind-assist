package com.linnan.blindassist.session

import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.risk.RiskLevel
import java.util.Locale

class SessionTrace(private val capacity: Int = DEFAULT_CAPACITY) {
    private val frames = ArrayDeque<TraceFrame>()

    fun record(
        evaluation: AssistFrameEvaluation,
        feedbackDecision: FeedbackDecision
    ): SessionSummary {
        if (frames.size == capacity) {
            frames.removeFirst()
        }
        frames.addLast(
            TraceFrame(
                level = evaluation.stableRisk.level,
                fps = evaluation.metrics.fps,
                inferenceMs = evaluation.metrics.inferenceMs,
                feedbackReason = feedbackDecision.reason
            )
        )
        return summary()
    }

    fun clear() {
        frames.clear()
    }

    fun summary(): SessionSummary {
        if (frames.isEmpty()) {
            return SessionSummary.empty()
        }

        var high = 0
        var medium = 0
        var low = 0
        var none = 0
        var fpsTotal = 0f
        var inferenceTotal = 0L
        frames.forEach { frame ->
            when (frame.level) {
                RiskLevel.HIGH -> high += 1
                RiskLevel.MEDIUM -> medium += 1
                RiskLevel.LOW -> low += 1
                RiskLevel.NONE -> none += 1
            }
            fpsTotal += frame.fps
            inferenceTotal += frame.inferenceMs
        }

        val count = frames.size
        return SessionSummary(
            frameCount = count,
            highCount = high,
            mediumCount = medium,
            lowCount = low,
            noneCount = none,
            averageFps = fpsTotal / count.toFloat(),
            averageInferenceMs = inferenceTotal / count,
            latestFeedbackReason = frames.last().feedbackReason
        )
    }

    private data class TraceFrame(
        val level: RiskLevel,
        val fps: Float,
        val inferenceMs: Long,
        val feedbackReason: FeedbackReason
    )

    companion object {
        const val DEFAULT_CAPACITY = 30
    }
}

data class SessionSummary(
    val frameCount: Int,
    val highCount: Int,
    val mediumCount: Int,
    val lowCount: Int,
    val noneCount: Int,
    val averageFps: Float,
    val averageInferenceMs: Long,
    val latestFeedbackReason: FeedbackReason
) {
    fun displayText(): String {
        return "会话：最近${frameCount}帧 高/中/低/无 $highCount/$mediumCount/$lowCount/$noneCount · " +
            "反馈 ${latestFeedbackReason.displayText} · " +
            "平均FPS ${String.format(Locale.US, "%.1f", averageFps)} · " +
            "平均推理 ${averageInferenceMs}ms"
    }

    companion object {
        fun empty(): SessionSummary {
            return SessionSummary(
                frameCount = 0,
                highCount = 0,
                mediumCount = 0,
                lowCount = 0,
                noneCount = 0,
                averageFps = 0f,
                averageInferenceMs = 0L,
                latestFeedbackReason = FeedbackReason.NO_FEEDBACK_RISK
            )
        }
    }
}

