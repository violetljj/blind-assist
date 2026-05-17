package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import org.junit.Assert.assertEquals
import org.junit.Test

class AssistEngineTest {
    private val frame = FrameSize(1000, 1000)

    @Test
    fun criticalHighRiskConfirmsImmediatelyAndKeepsTriggeredFeedbackReason() {
        val engine = AssistEngine()
        val evaluation = engine.evaluate(
            detections = listOf(detection("person", BoundingBox(390f, 140f, 610f, 780f))),
            frameSize = frame,
            profile = AlertProfile.STANDARD,
            metrics = metrics(),
            nowMs = 1000L
        )

        val result = engine.completeFeedback(
            evaluation,
            FeedbackDecision(plan = null, triggered = true, reason = FeedbackReason.TRIGGERED)
        )

        assertEquals(RiskLevel.HIGH, result.evaluation.stableRisk.level)
        assertEquals(ProximityBand.CRITICAL, result.evaluation.stableRisk.proximity)
        assertEquals(FeedbackReason.TRIGGERED, result.feedbackDecision.reason)
        assertEquals("会话：最近1帧 高/中/低/无 1/0/0/0 · 反馈 已触发反馈 · 平均FPS 12.5 · 平均推理 22ms", result.sessionSummary.displayText())
    }

    @Test
    fun firstMediumRiskFrameReportsUnstableFeedbackReason() {
        val engine = AssistEngine()
        val evaluation = engine.evaluate(
            detections = listOf(detection("car", BoundingBox(20f, 330f, 260f, 720f))),
            frameSize = frame,
            profile = AlertProfile.STANDARD,
            metrics = metrics(),
            nowMs = 1000L
        )

        val result = engine.completeFeedback(
            evaluation,
            FeedbackDecision(plan = null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)
        )

        assertEquals(RiskLevel.MEDIUM, result.evaluation.rawRisk.level)
        assertEquals(RiskLevel.NONE, result.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.UNSTABLE_RISK, result.feedbackDecision.reason)
    }

    @Test
    fun midAndFarRisksReportDistanceTooFarFeedbackReason() {
        val engine = AssistEngine()
        val mid = engine.evaluate(
            detections = listOf(detection("chair", BoundingBox(450f, 240f, 560f, 500f))),
            frameSize = frame,
            profile = AlertProfile.STANDARD,
            metrics = metrics(),
            nowMs = 1000L
        )
        val far = engine.evaluate(
            detections = listOf(detection("person", BoundingBox(450f, 120f, 520f, 280f))),
            frameSize = frame,
            profile = AlertProfile.STANDARD,
            metrics = metrics(),
            nowMs = 1100L
        )

        val midResult = engine.completeFeedback(
            mid,
            FeedbackDecision(plan = null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)
        )
        val farResult = engine.completeFeedback(
            far,
            FeedbackDecision(plan = null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)
        )

        assertEquals(RiskLevel.LOW, midResult.evaluation.stableRisk.level)
        assertEquals(RiskLevel.NONE, farResult.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.DISTANCE_TOO_FAR, midResult.feedbackDecision.reason)
        assertEquals(FeedbackReason.DISTANCE_TOO_FAR, farResult.feedbackDecision.reason)
    }

    @Test
    fun currentFrameWithoutDetectionsCanReportHeldStableRisk() {
        val engine = AssistEngine()
        engine.evaluate(
            detections = listOf(detection("car", BoundingBox(20f, 330f, 260f, 720f))),
            frameSize = frame,
            profile = AlertProfile.STANDARD,
            metrics = metrics(),
            nowMs = 1000L
        )
        val confirmed = engine.evaluate(
            detections = listOf(detection("car", BoundingBox(20f, 330f, 260f, 720f))),
            frameSize = frame,
            profile = AlertProfile.STANDARD,
            metrics = metrics(),
            nowMs = 1100L
        )
        assertEquals(RiskLevel.MEDIUM, confirmed.stableRisk.level)

        val held = engine.evaluate(
            detections = emptyList(),
            frameSize = frame,
            profile = AlertProfile.STANDARD,
            metrics = metrics(),
            nowMs = 1200L
        )
        val result = engine.completeFeedback(
            held,
            FeedbackDecision(plan = null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)
        )

        assertEquals(0, result.evaluation.detectionCount)
        assertEquals(RiskLevel.NONE, result.evaluation.rawRisk.level)
        assertEquals(RiskLevel.MEDIUM, result.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.HELD_ALERT, result.feedbackDecision.reason)
    }

    private fun detection(
        label: String,
        box: BoundingBox,
        confidence: Float = 0.9f
    ): Detection {
        return Detection(
            classId = 0,
            label = label,
            confidence = confidence,
            boundingBox = box,
            frameSize = frame
        )
    }

    private fun metrics(): DetectorMetrics {
        return DetectorMetrics(
            totalMs = 35L,
            preprocessMs = 5L,
            inferenceMs = 22L,
            postprocessMs = 8L,
            fps = 12.5f,
            modelStatus = "ready"
        )
    }
}
