package com.linnan.blindassist.ui

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.session.AssistEngine
import com.linnan.blindassist.session.DetectorMetrics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FieldTestSummaryMapperTest {
    @Test
    fun emptySummaryShowsWaitingState() {
        val summary = FieldTestSummaryMapper.fromSummary(
            summary = com.linnan.blindassist.session.SessionSummary.empty(),
            active = false,
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL
        )

        assertEquals("等待相机会话", summary.statusText)
        assertTrue(summary.accessibilityText.contains("现场测试摘要"))
    }

    @Test
    fun activeSummaryIncludesScenarioProfileAndCounts() {
        val engine = AssistEngine()
        val frame = FrameSize(1000, 1000)
        engine.startSession(1000L)
        val evaluation = engine.evaluate(
            detections = listOf(Detection(0, "person", 0.9f, BoundingBox(390f, 140f, 610f, 780f), frame)),
            frameSize = frame,
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.CORRIDOR,
            metrics = DetectorMetrics(35L, 5L, 22L, 8L, 12.5f, "ready"),
            nowMs = 2000L
        )
        val result = engine.completeFeedback(
            evaluation,
            FeedbackDecision(null, triggered = true, reason = FeedbackReason.TRIGGERED, speechTriggered = true)
        )

        val summary = FieldTestSummaryMapper.fromSummary(
            summary = result.sessionSummary,
            active = true,
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.CORRIDOR
        )

        assertEquals("本次相机会话进行中", summary.statusText)
        assertTrue(summary.detailText.contains("当前场景：走廊通行"))
        assertTrue(summary.accessibilityText.contains("语音提醒1次"))
    }
}
