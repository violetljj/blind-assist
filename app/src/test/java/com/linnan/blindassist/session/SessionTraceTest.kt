package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import org.junit.Assert.assertEquals
import org.junit.Test

class SessionTraceTest {
    @Test
    fun keepsOnlyLatestThirtyFrames() {
        val trace = SessionTrace()

        repeat(35) { index ->
            trace.record(
                evaluation(level = RiskLevel.HIGH, fps = index.toFloat(), inferenceMs = index.toLong()),
                FeedbackDecision(plan = null, triggered = true, reason = FeedbackReason.TRIGGERED)
            )
        }

        val summary = trace.summary()
        assertEquals(30, summary.frameCount)
        assertEquals(30, summary.highCount)
        assertEquals(0, summary.mediumCount)
        assertEquals(19.5f, summary.averageFps, 0.001f)
        assertEquals(19L, summary.averageInferenceMs)
    }

    @Test
    fun summaryCountsRiskLevelsAndAveragesMetrics() {
        val trace = SessionTrace()
        trace.record(
            evaluation(level = RiskLevel.HIGH, fps = 10f, inferenceMs = 20L),
            FeedbackDecision(plan = null, triggered = true, reason = FeedbackReason.TRIGGERED)
        )
        trace.record(
            evaluation(level = RiskLevel.MEDIUM, fps = 12f, inferenceMs = 22L),
            FeedbackDecision(plan = null, triggered = false, reason = FeedbackReason.UNSTABLE_RISK)
        )
        trace.record(
            evaluation(level = RiskLevel.LOW, fps = 14f, inferenceMs = 24L),
            FeedbackDecision(plan = null, triggered = false, reason = FeedbackReason.DISTANCE_TOO_FAR)
        )
        trace.record(
            evaluation(level = RiskLevel.NONE, fps = 16f, inferenceMs = 26L),
            FeedbackDecision(plan = null, triggered = false, reason = FeedbackReason.COOLDOWN)
        )

        val summary = trace.summary()

        assertEquals(4, summary.frameCount)
        assertEquals(1, summary.highCount)
        assertEquals(1, summary.mediumCount)
        assertEquals(1, summary.lowCount)
        assertEquals(1, summary.noneCount)
        assertEquals(13f, summary.averageFps, 0.001f)
        assertEquals(23L, summary.averageInferenceMs)
        assertEquals(FeedbackReason.COOLDOWN, summary.latestFeedbackReason)
        assertEquals("会话：最近4帧 高/中/低/无 1/1/1/1 · 反馈 冷却中 · 平均FPS 13.0 · 平均推理 23ms", summary.displayText())
    }

    private fun evaluation(
        level: RiskLevel,
        fps: Float,
        inferenceMs: Long
    ): AssistFrameEvaluation {
        val risk = RiskResult(
            level = level,
            direction = if (level == RiskLevel.NONE) RiskDirection.NONE else RiskDirection.CENTER,
            message = "测试",
            urgencyScore = level.ordinal.toFloat()
        )
        return AssistFrameEvaluation(
            rawRisk = risk,
            stableRisk = risk,
            detectionCount = 1,
            frameSize = FrameSize(1000, 1000),
            profile = AlertProfile.STANDARD,
            metrics = DetectorMetrics(
                totalMs = inferenceMs + 10L,
                preprocessMs = 4L,
                inferenceMs = inferenceMs,
                postprocessMs = 6L,
                fps = fps,
                modelStatus = "ready"
            ),
            preliminaryReason = FeedbackReason.NO_FEEDBACK_RISK
        )
    }
}

