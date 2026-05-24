package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.vision.DetectorFrameResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SceneReplayRegressionTest {
    private val frame = FrameSize(1000, 1000)

    @Test
    fun replayCenteredNearTargetTriggersFullRiskFeedbackAndSummaryChain() {
        val coordinator = coordinator()

        val result = coordinator.processFrame(
            detectorFrame = replayFrame(centerCriticalPerson()),
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            nowMs = 1000L
        )

        assertEquals(RiskLevel.HIGH, result.evaluation.rawRisk.level)
        assertEquals(RiskLevel.HIGH, result.evaluation.stableRisk.level)
        assertEquals(RiskDirection.CENTER, result.evaluation.stableRisk.direction)
        assertEquals(ProximityBand.CRITICAL, result.evaluation.stableRisk.proximity)
        assertEquals(FeedbackReason.TRIGGERED, result.feedbackDecision.reason)
        assertEquals(1, result.sessionSummary.highCount)
        assertEquals(1, result.sessionSummary.speechTriggerCount)
        assertEquals(1, result.sessionSummary.vibrationTriggerCount)
        assertFalse(result.explanation.headline.isBlank())
        assertTrue(result.sessionSummary.fieldTestText("Standard", "General", AppLanguage.EN).contains("Latest explanation"))
    }

    @Test
    fun replaySideTargetRequiresConfirmationBeforeFeedback() {
        val coordinator = coordinator()
        val sideFrame = replayFrame(leftNearChair())

        val first = coordinator.processFrame(sideFrame, AlertProfile.STANDARD, AssistScenario.GENERAL, nowMs = 1000L)
        val second = coordinator.processFrame(sideFrame, AlertProfile.STANDARD, AssistScenario.GENERAL, nowMs = 1100L)

        assertEquals(RiskLevel.MEDIUM, first.evaluation.rawRisk.level)
        assertEquals(RiskLevel.NONE, first.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.UNSTABLE_RISK, first.feedbackDecision.reason)
        assertEquals(RiskLevel.MEDIUM, second.evaluation.stableRisk.level)
        assertEquals(RiskDirection.LEFT, second.evaluation.stableRisk.direction)
        assertEquals(FeedbackReason.TRIGGERED, second.feedbackDecision.reason)
        assertEquals(1, second.sessionSummary.mediumCount)
        assertEquals(1, second.sessionSummary.noneCount)
    }

    @Test
    fun replayFarTargetStaysVisualOnlyAndDoesNotTriggerFeedback() {
        val result = coordinator().processFrame(
            detectorFrame = replayFrame(farVehicle()),
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            nowMs = 1000L
        )

        assertEquals(RiskLevel.NONE, result.evaluation.stableRisk.level)
        assertEquals(ProximityBand.FAR, result.evaluation.rawRisk.proximity)
        assertEquals(FeedbackReason.DISTANCE_TOO_FAR, result.feedbackDecision.reason)
        assertEquals(0, result.sessionSummary.feedbackTriggerCount)
    }

    @Test
    fun replayEmptyFrameStaysVisualOnlyWithDistanceReason() {
        val result = coordinator().processFrame(
            detectorFrame = replayFrame(),
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            nowMs = 1000L
        )

        assertEquals(0, result.evaluation.detectionCount)
        assertEquals(RiskLevel.NONE, result.evaluation.rawRisk.level)
        assertEquals(RiskLevel.NONE, result.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.DISTANCE_TOO_FAR, result.feedbackDecision.reason)
        assertEquals(1, result.sessionSummary.noneCount)
    }

    @Test
    fun replayDroppedFrameHoldsThenClearsBeforeRecoveredRiskTriggersAgain() {
        var feedbackCalls = 0
        val coordinator = coordinator { risk, _, _ ->
            feedbackCalls += 1
            when {
                risk.level == RiskLevel.NONE -> FeedbackDecision(
                    plan = null,
                    triggered = false,
                    reason = FeedbackReason.NO_FEEDBACK_RISK
                )
                feedbackCalls == 2 -> FeedbackDecision(
                    plan = null,
                    triggered = false,
                    reason = FeedbackReason.NO_FEEDBACK_RISK
                )
                else -> FeedbackDecision(
                    plan = null,
                    triggered = true,
                    reason = FeedbackReason.TRIGGERED,
                    speechTriggered = true,
                    vibrationTriggered = true
                )
            }
        }
        val riskFrame = replayFrame(centerCriticalPerson())

        coordinator.processFrame(riskFrame, AlertProfile.STANDARD, AssistScenario.GENERAL, nowMs = 1000L)
        val held = coordinator.processFrame(replayFrame(), AlertProfile.STANDARD, AssistScenario.GENERAL, nowMs = 1200L)
        val cleared = coordinator.processFrame(replayFrame(), AlertProfile.STANDARD, AssistScenario.GENERAL, nowMs = 1801L)
        val recovered = coordinator.processFrame(riskFrame, AlertProfile.STANDARD, AssistScenario.GENERAL, nowMs = 1900L)

        assertEquals(RiskLevel.HIGH, held.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.HELD_ALERT, held.feedbackDecision.reason)
        assertEquals(RiskLevel.NONE, cleared.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.DISTANCE_TOO_FAR, cleared.feedbackDecision.reason)
        assertEquals(RiskLevel.HIGH, recovered.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.TRIGGERED, recovered.feedbackDecision.reason)
    }

    @Test
    fun replayFeedbackUnavailableIsPreservedWithoutTriggerCounts() {
        val coordinator = coordinator { risk, _, _ ->
            if (risk.level == RiskLevel.NONE) {
                FeedbackDecision(plan = null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)
            } else {
                FeedbackDecision(plan = null, triggered = false, reason = FeedbackReason.FEEDBACK_UNAVAILABLE)
            }
        }

        val result = coordinator.processFrame(
            detectorFrame = replayFrame(centerCriticalPerson()),
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            nowMs = 1000L
        )

        assertEquals(RiskLevel.HIGH, result.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.FEEDBACK_UNAVAILABLE, result.feedbackDecision.reason)
        assertEquals(0, result.sessionSummary.feedbackTriggerCount)
        assertEquals(FeedbackReason.FEEDBACK_UNAVAILABLE, result.sessionSummary.latestFeedbackReason)
    }

    private fun coordinator(
        decisionFor: (RiskResult, AlertProfile, AssistScenario) -> FeedbackDecision = { risk, _, _ ->
            if (risk.level == RiskLevel.NONE) {
                FeedbackDecision(plan = null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)
            } else {
                FeedbackDecision(
                    plan = null,
                    triggered = true,
                    reason = FeedbackReason.TRIGGERED,
                    speechTriggered = true,
                    vibrationTriggered = true
                )
            }
        }
    ): AssistSessionCoordinator {
        return AssistSessionCoordinator(
            feedbackGateway = ReplayFeedbackGateway(decisionFor),
            fpsTracker = FpsTracker(clock = { 1000L })
        ).also { it.startSession(nowMs = 1000L) }
    }

    private fun replayFrame(vararg detections: Detection): DetectorFrameResult {
        return DetectorFrameResult(
            detections = detections.toList(),
            frameSize = frame,
            metrics = DetectorMetrics(
                totalMs = 34L,
                preprocessMs = 5L,
                inferenceMs = 22L,
                postprocessMs = 7L,
                fps = 0f,
                modelStatus = "replay-fixture"
            )
        )
    }

    private fun centerCriticalPerson(): Detection {
        return detection("person", BoundingBox(390f, 120f, 610f, 780f))
    }

    private fun leftNearChair(): Detection {
        return detection("chair", BoundingBox(30f, 360f, 270f, 700f))
    }

    private fun farVehicle(): Detection {
        return detection("car", BoundingBox(440f, 120f, 560f, 350f))
    }

    private fun detection(label: String, box: BoundingBox): Detection {
        return Detection(classId = 0, label = label, confidence = 0.9f, boundingBox = box, frameSize = frame)
    }

    private class ReplayFeedbackGateway(
        private val decisionFor: (RiskResult, AlertProfile, AssistScenario) -> FeedbackDecision
    ) : FeedbackGateway {
        override fun notify(
            risk: RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision = decisionFor(risk, profile, scenario)
    }
}
