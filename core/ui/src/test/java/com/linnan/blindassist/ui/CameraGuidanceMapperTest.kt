package com.linnan.blindassist.ui

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.session.AssistEngine
import com.linnan.blindassist.session.DetectorMetrics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraGuidanceMapperTest {
    private val frame = FrameSize(1000, 1000)

    @Test
    fun permissionDeniedMapsToReadableCameraState() {
        val guidance = CameraGuidanceMapper.permissionDenied()

        assertEquals("需要相机权限", guidance.title)
        assertEquals("permission", guidance.accessibilityKey)
        assertTrue(guidance.accessibilitySummary.contains("相机权限"))
    }

    @Test
    fun waitingIncludesScenarioAndModelStatus() {
        val guidance = CameraGuidanceMapper.waiting("模型已加载", AssistScenario.CORRIDOR)

        assertEquals("检测已开启", guidance.title)
        assertEquals("走廊通行", guidance.scenarioName)
        assertTrue(guidance.debugText.contains("模型已加载"))
    }

    @Test
    fun waitingCanUseEnglishCoreText() {
        val guidance = CameraGuidanceMapper.waiting("model ready", AssistScenario.CORRIDOR, AppLanguage.EN)

        assertEquals("Detection on", guidance.title)
        assertEquals("Corridor", guidance.scenarioName)
        assertTrue(guidance.accessibilitySummary.contains("Detection on"))
    }

    @Test
    fun frameResultMapsRiskExplanationAndDebugMetrics() {
        val engine = AssistEngine()
        val evaluation = engine.evaluate(
            detections = listOf(Detection(0, "person", 0.9f, BoundingBox(390f, 140f, 610f, 780f), frame)),
            frameSize = frame,
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.CORRIDOR,
            metrics = DetectorMetrics(35L, 5L, 22L, 8L, 12.5f, "ready"),
            nowMs = 1000L
        )
        val result = engine.completeFeedback(
            evaluation,
            FeedbackDecision(null, triggered = true, reason = FeedbackReason.TRIGGERED)
        )

        val guidance = CameraGuidanceMapper.fromFrameResult(result)

        assertEquals("走廊通行", guidance.scenarioName)
        assertEquals("已按走廊通行策略触发提醒", guidance.explanationHeadline)
        assertTrue(guidance.debugText.contains("infer 22ms"))
        assertTrue(guidance.accessibilitySummary.contains("已触发提醒"))
    }

    @Test
    fun frameResultCanUseEnglishRiskExplanationAndSummary() {
        val engine = AssistEngine()
        val evaluation = engine.evaluate(
            detections = listOf(Detection(0, "person", 0.9f, BoundingBox(390f, 140f, 610f, 780f), frame)),
            frameSize = frame,
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.CORRIDOR,
            metrics = DetectorMetrics(35L, 5L, 22L, 8L, 12.5f, "ready"),
            nowMs = 1000L
        )
        val result = engine.completeFeedback(
            evaluation,
            FeedbackDecision(null, triggered = true, reason = FeedbackReason.TRIGGERED)
        )

        val guidance = CameraGuidanceMapper.fromFrameResult(result, AppLanguage.EN)

        assertEquals("Corridor", guidance.scenarioName)
        assertEquals("Reminder triggered with Corridor strategy", guidance.explanationHeadline)
        assertTrue(guidance.debugText.contains("Profile: Standard"))
        assertTrue(guidance.accessibilitySummary.contains("Reminder triggered"))
    }
}
