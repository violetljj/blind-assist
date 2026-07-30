package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackPlanner
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DualLoopActiveIntegrationTest {
    private val frameSize = FrameSize(1000, 1000)

    @Test
    fun collectiveShrinkSuppressesOnlyCurrentFeedbackOpportunity() {
        val baseline = AssistDecisionKernel()
        val active = AssistDecisionKernel()
        val baselineGateway = PlannerGateway()
        val activeGateway = PlannerGateway()
        baseline.startSession(1_000L)
        active.startSession(1_000L)

        val first = listOf(
            detection(0, "person", 700f, 500f),
            detection(2, "car", 550f, 780f)
        )
        val second = listOf(
            detection(0, "person", 630f, 500f),
            detection(2, "car", 495f, 780f)
        )
        process(baseline, baselineGateway, first, 0, DualLoopRuntimeMode.OFF)
        process(active, activeGateway, first, 0, DualLoopRuntimeMode.ACTIVE_CONTRADICT_ONLY)
        val baselineResult =
            process(baseline, baselineGateway, second, 1, DualLoopRuntimeMode.OFF)
        val activeResult =
            process(active, activeGateway, second, 1, DualLoopRuntimeMode.ACTIVE_CONTRADICT_ONLY)

        assertTrue(activeResult.evaluation.dualLoopShadow.admitted)
        assertTrue(activeResult.evaluation.dualLoopShadow.feedbackMutationAllowed)
        assertFalse(activeResult.evaluation.dualLoopShadow.eventMutationAllowed)
        assertEquals(
            DualLoopCorrectionDecision.CONTRADICT_APPROACH,
            activeResult.evaluation.dualLoopShadow.correctionDecision
        )
        assertEquals(FeedbackReason.TRIGGERED, baselineResult.feedbackDecision.reason)
        assertEquals(
            FeedbackReason.DUAL_LOOP_CONTRADICTED,
            activeResult.feedbackDecision.reason
        )
        assertFalse(activeResult.feedbackDecision.triggered)
        assertEquals(baselineResult.evaluation.rawRisk, activeResult.evaluation.rawRisk)
        assertEquals(baselineResult.evaluation.stableRisk, activeResult.evaluation.stableRisk)
        assertEquals(baselineResult.evaluation.riskEvent, activeResult.evaluation.riskEvent)
        assertEquals(2, baselineGateway.notifyCalls)
        assertEquals(1, activeGateway.notifyCalls)
    }

    private fun process(
        kernel: AssistDecisionKernel,
        gateway: PlannerGateway,
        detections: List<Detection>,
        index: Int,
        mode: DualLoopRuntimeMode
    ): AssistFrameResult {
        val capturedAtNs = 1_000_000_000L + index * 100_000_000L
        return kernel.processFrame(
            detections = detections,
            frameSize = frameSize,
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            metrics = DetectorMetrics(35L, 5L, 22L, 8L, 0f, "ready"),
            feedbackGateway = gateway,
            nowMs = 1_000L + index * 100L,
            sourceFrame = FrameStamp(
                frameId = index.toLong(),
                capturedAtNs = capturedAtNs,
                receivedAtNs = capturedAtNs + 1_000_000L,
                sourceId = "camera2:0",
                coordinateFrame = "camera2:0:analysis-buffer",
                clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
            ),
            decisionAtNs = capturedAtNs + 10_000_000L,
            dualLoopMode = mode
        )
    }

    private fun detection(classId: Int, label: String, height: Float, centerX: Float) =
        Detection(
            classId = classId,
            label = label,
            confidence = 0.95f,
            boundingBox = BoundingBox(centerX - 150f, 100f, centerX + 150f, 100f + height),
            frameSize = frameSize
        )

    private class PlannerGateway : FeedbackGateway {
        var notifyCalls = 0
            private set

        override fun resetSession() = Unit

        override fun notify(
            risk: RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
            notifyCalls += 1
            val plan = FeedbackPlanner.planFor(risk, profile = profile, scenario = scenario)
            return if (plan == null) {
                FeedbackDecision(null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)
            } else {
                FeedbackDecision(plan, triggered = true, reason = FeedbackReason.TRIGGERED)
            }
        }
    }
}
