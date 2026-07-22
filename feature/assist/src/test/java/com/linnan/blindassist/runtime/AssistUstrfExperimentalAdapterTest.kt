package com.linnan.blindassist.runtime

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.session.AssistSessionCoordinator
import com.linnan.blindassist.session.DetectorMetrics
import com.linnan.blindassist.vision.DetectorFrameResult
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AssistUstrfExperimentalAdapterTest {
    private val frameSize = FrameSize(1000, 1000)

    @Test
    fun centralObstacleBecomesObjectAgnosticUstrfRiskEvidence() {
        val gateway = RecordingGateway()
        val adapter = adapter(gateway)

        val result = adapter.process(
            frame = frame(
                clock = FrameClockDomain.ANDROID_ELAPSED_REALTIME,
                detections = listOf(detection(BoundingBox(420f, 350f, 620f, 980f)))
            ),
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            nowMs = 1_000L,
            decisionAtNs = 1_000_000_000L
        )

        assertEquals(RiskLevel.HIGH, result.evaluation.rawRisk.level)
        assertNull(result.evaluation.rawRisk.sourceDetection)
        assertTrue(result.evaluation.rawRisk.message.startsWith("USTRF实验"))
        assertEquals(RiskLevel.HIGH, gateway.lastRisk?.level)
        assertEquals(1L, result.evaluation.sourceFrame?.frameId)
    }

    @Test
    fun sideObstacleDoesNotBecomeCenterRouteRiskOrSafetyClaim() {
        val adapter = adapter(RecordingGateway())

        val result = adapter.process(
            frame = frame(
                clock = FrameClockDomain.ANDROID_ELAPSED_REALTIME,
                detections = listOf(detection(BoundingBox(10f, 350f, 170f, 980f)))
            ),
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            nowMs = 1_000L,
            decisionAtNs = 1_000_000_000L
        )

        assertEquals(RiskLevel.NONE, result.evaluation.rawRisk.level)
        assertTrue(result.evaluation.rawRisk.message.contains("不代表安全"))
    }

    @Test
    fun unmappedClockFailsClosedAsHighRiskAbstention() {
        val adapter = adapter(RecordingGateway())

        val result = adapter.process(
            frame = frame(FrameClockDomain.CAMERA_HARDWARE_UNMAPPED, emptyList()),
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            nowMs = 1_000L,
            decisionAtNs = 1_000_000_000L
        )

        assertEquals(RiskLevel.HIGH, result.evaluation.rawRisk.level)
        assertTrue(result.evaluation.rawRisk.message.contains("停下重新扫描"))
    }

    private fun adapter(gateway: RecordingGateway): AssistUstrfExperimentalAdapter =
        AssistUstrfExperimentalAdapter(AssistSessionCoordinator(feedbackGateway = gateway))

    private fun frame(clock: FrameClockDomain, detections: List<Detection>) = DetectorFrameResult(
        detections = detections,
        frameSize = frameSize,
        metrics = DetectorMetrics(1L, 0L, 1L, 0L, 30f, "ready"),
        sourceFrame = FrameStamp(
            frameId = 1L,
            capturedAtNs = 1_000_000_000L,
            receivedAtNs = 1_000_000_100L,
            sourceId = "camera2:0",
            coordinateFrame = "camera2:0:analysis-buffer",
            clockDomain = clock
        )
    )

    private fun detection(box: BoundingBox) = Detection(
        classId = 0,
        label = "person",
        confidence = .95f,
        boundingBox = box,
        frameSize = frameSize
    )

    private class RecordingGateway : FeedbackGateway {
        var lastRisk: RiskResult? = null

        override fun resetSession() = Unit

        override fun notify(
            risk: RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
            lastRisk = risk
            return FeedbackDecision(null, triggered = true, reason = FeedbackReason.TRIGGERED)
        }
    }
}
