package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackPlanner
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.exp

class DualLoopShadowIntegrationTest {
    private val frameSize = FrameSize(1000, 1000)
    private val detection = Detection(
        classId = 0,
        label = "person",
        confidence = 0.95f,
        boundingBox = BoundingBox(350f, 200f, 650f, 950f),
        frameSize = frameSize
    )
    private val previousFrame = stamp(20L, 2_000_000_000L)
    private val currentFrame = stamp(21L, 2_100_000_000L)

    @Test
    fun admittedShadowEvidenceLeavesDecisionEventFeedbackAndTraceFrameExact() {
        val baselineGateway = PlannerGateway()
        val shadowGateway = PlannerGateway()
        val baseline = AssistDecisionKernel()
        val shadow = AssistDecisionKernel(
            dualLoopShadowAdmitter = DualLoopShadowAdmitter(
                admittedSourceIdentities = setOf(
                    DualLoopSourceIdentity(
                        DualLoopShadowAdmitter.CONTRACT_ID,
                        "synthetic-shadow-fixture"
                    )
                )
            )
        )
        baseline.startSession(1_900L)
        shadow.startSession(1_900L)

        val baselineResult = baseline.processFrame(
            detections = listOf(detection),
            frameSize = frameSize,
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            metrics = metrics(),
            feedbackGateway = baselineGateway,
            nowMs = 2_100L,
            sourceFrame = currentFrame,
            decisionAtNs = 2_120_000_000L
        )
        val shadowResult = shadow.processFrame(
            detections = listOf(detection),
            frameSize = frameSize,
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            metrics = metrics(),
            feedbackGateway = shadowGateway,
            nowMs = 2_100L,
            sourceFrame = currentFrame,
            decisionAtNs = 2_120_000_000L,
            dualLoopMode = DualLoopRuntimeMode.SHADOW_ABSTAIN_ONLY,
            dualLoopGeometryEvidence = evidence()
        )

        assertTrue(shadowResult.evaluation.dualLoopShadow.admitted)
        assertFalse(shadowResult.evaluation.dualLoopShadow.eventMutationAllowed)
        assertFalse(shadowResult.evaluation.dualLoopShadow.feedbackMutationAllowed)
        assertEquals(baselineResult.evaluation.rawRisk, shadowResult.evaluation.rawRisk)
        assertEquals(baselineResult.evaluation.stableRisk, shadowResult.evaluation.stableRisk)
        assertEquals(baselineResult.evaluation.riskEvent, shadowResult.evaluation.riskEvent)
        assertEquals(baselineResult.feedbackDecision, shadowResult.feedbackDecision)
        assertEquals(baselineResult.explanation, shadowResult.explanation)
        assertEquals(baselineResult.sessionSummary, shadowResult.sessionSummary)
        assertEquals(baselineGateway.notifyCalls, shadowGateway.notifyCalls)
    }

    @Test
    fun liveTristateSourceReachesShadowAdmissionWithoutMutatingBaseline() {
        val baselineGateway = PlannerGateway()
        val shadowGateway = PlannerGateway()
        val baseline = AssistDecisionKernel()
        val shadow = AssistDecisionKernel()
        baseline.startSession(900L)
        shadow.startSession(900L)
        var finalBaseline: AssistFrameResult? = null
        var finalShadow: AssistFrameResult? = null

        repeat(7) { index ->
            val height = 500f * exp(0.30f * index * 0.1f)
            val frameDetection = detection.copy(
                boundingBox = BoundingBox(350f, 200f, 650f, 200f + height)
            )
            val frame = stamp(index.toLong(), 1_000_000_000L + index * 100_000_000L)
            finalBaseline = baseline.processFrame(
                detections = listOf(frameDetection),
                frameSize = frameSize,
                profile = AlertProfile.STANDARD,
                scenario = AssistScenario.GENERAL,
                metrics = metrics(),
                feedbackGateway = baselineGateway,
                nowMs = 1_000L + index * 100L,
                sourceFrame = frame,
                decisionAtNs = frame.capturedAtNs + 10_000_000L
            )
            finalShadow = shadow.processFrame(
                detections = listOf(frameDetection),
                frameSize = frameSize,
                profile = AlertProfile.STANDARD,
                scenario = AssistScenario.GENERAL,
                metrics = metrics(),
                feedbackGateway = shadowGateway,
                nowMs = 1_000L + index * 100L,
                sourceFrame = frame,
                decisionAtNs = frame.capturedAtNs + 10_000_000L,
                dualLoopMode = DualLoopRuntimeMode.SHADOW_ABSTAIN_ONLY
            )
            assertEquals(finalBaseline!!.evaluation.rawRisk, finalShadow!!.evaluation.rawRisk)
            assertEquals(finalBaseline!!.evaluation.stableRisk, finalShadow!!.evaluation.stableRisk)
            assertEquals(finalBaseline!!.evaluation.riskEvent, finalShadow!!.evaluation.riskEvent)
            assertEquals(finalBaseline!!.feedbackDecision, finalShadow!!.feedbackDecision)
        }

        assertEquals(
            DualLoopShadowDisposition.ADMITTED_SHADOW,
            finalShadow!!.evaluation.dualLoopShadow.disposition
        )
        assertEquals(
            DualLoopCorrectionDecision.CONFIRM_APPROACH,
            finalShadow!!.evaluation.dualLoopShadow.correctionDecision
        )
        assertEquals(finalBaseline!!.sessionSummary, finalShadow!!.sessionSummary)
        assertEquals(baselineGateway.notifyCalls, shadowGateway.notifyCalls)
    }

    private fun evidence() = DualLoopGeometryEvidence(
        sourceContractId = DualLoopShadowAdmitter.CONTRACT_ID,
        sourceId = "synthetic-shadow-fixture",
        previousFrame = previousFrame,
        currentFrame = currentFrame,
        availableAtNs = 2_110_000_000L,
        validUntilNs = 2_200_000_000L,
        availabilityClockDomain = FrameClockDomain.REPLAY_TIMELINE,
        trackEpoch = "fixture-epoch-1",
        targetClassId = detection.classId,
        targetLabel = detection.label,
        targetBoundingBox = detection.boundingBox,
        targetFrameSize = detection.frameSize,
        targetSource = DetectionSource.OBJECT_DETECTOR,
        correctionDecision = DualLoopCorrectionDecision.CONFIRM_APPROACH,
        signedApproachRatePerS = 0.30f,
        quality = 0.80f
    )

    private fun metrics() = DetectorMetrics(35L, 5L, 22L, 8L, 0f, "ready")

    private fun stamp(frameId: Long, capturedAtNs: Long) = FrameStamp(
        frameId = frameId,
        capturedAtNs = capturedAtNs,
        receivedAtNs = capturedAtNs + 1_000_000L,
        sourceId = "synthetic-camera",
        coordinateFrame = "camera-optical",
        clockDomain = FrameClockDomain.REPLAY_TIMELINE
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
