package com.linnan.blindassist.session

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.vision.DetectorFrameResult
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AssistSessionCoordinatorTest {
    private val frame = FrameSize(1000, 1000)

    @Test
    fun highRiskFrameTriggersFeedbackThroughGateway() {
        val gateway = FakeFeedbackGateway(FeedbackDecision(null, triggered = true, reason = FeedbackReason.TRIGGERED))
        val coordinator = AssistSessionCoordinator(feedbackGateway = gateway, fpsTracker = fixedFpsTracker())

        val result = coordinator.processFrame(
            detectorFrame = detectorFrame(listOf(detection("person", BoundingBox(390f, 140f, 610f, 780f)))),
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            nowMs = 1000L
        )

        assertEquals(RiskLevel.HIGH, result.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.TRIGGERED, result.feedbackDecision.reason)
        assertEquals(RiskLevel.HIGH, gateway.lastRisk?.level)
    }

    @Test
    fun firstMediumFrameIsReportedAsUnstableEvenWhenGatewayHasNoFeedback() {
        val coordinator = AssistSessionCoordinator(
            feedbackGateway = FakeFeedbackGateway(FeedbackDecision(null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)),
            fpsTracker = fixedFpsTracker()
        )

        val result = coordinator.processFrame(
            detectorFrame = detectorFrame(listOf(detection("car", BoundingBox(20f, 330f, 260f, 720f)))),
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            nowMs = 1000L
        )

        assertEquals(RiskLevel.MEDIUM, result.evaluation.rawRisk.level)
        assertEquals(RiskLevel.NONE, result.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.UNSTABLE_RISK, result.feedbackDecision.reason)
    }

    @Test
    fun heldAlertSurvivesOneEmptyFrame() {
        val coordinator = AssistSessionCoordinator(
            feedbackGateway = FakeFeedbackGateway(FeedbackDecision(null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)),
            fpsTracker = fixedFpsTracker()
        )
        val mediumFrame = detectorFrame(listOf(detection("car", BoundingBox(20f, 330f, 260f, 720f))))

        coordinator.processFrame(mediumFrame, AlertProfile.STANDARD, AssistScenario.GENERAL, nowMs = 1000L)
        coordinator.processFrame(mediumFrame, AlertProfile.STANDARD, AssistScenario.GENERAL, nowMs = 1100L)
        val result = coordinator.processFrame(detectorFrame(emptyList()), AlertProfile.STANDARD, AssistScenario.GENERAL, nowMs = 1200L)

        assertEquals(RiskLevel.MEDIUM, result.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.HELD_ALERT, result.feedbackDecision.reason)
    }

    @Test
    fun cooldownAndDisabledFeedbackReasonsArePreserved() {
        val cooldown = AssistSessionCoordinator(
            feedbackGateway = FakeFeedbackGateway(FeedbackDecision(null, triggered = false, reason = FeedbackReason.COOLDOWN)),
            fpsTracker = fixedFpsTracker()
        ).processFrame(
            detectorFrame = detectorFrame(listOf(detection("person", BoundingBox(390f, 140f, 610f, 780f)))),
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            nowMs = 1000L
        )
        val disabled = AssistSessionCoordinator(
            feedbackGateway = FakeFeedbackGateway(FeedbackDecision(null, triggered = false, reason = FeedbackReason.SPEECH_DISABLED)),
            fpsTracker = fixedFpsTracker()
        ).processFrame(
            detectorFrame = detectorFrame(listOf(detection("person", BoundingBox(390f, 140f, 610f, 780f)))),
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            nowMs = 1000L
        )

        assertEquals(FeedbackReason.COOLDOWN, cooldown.feedbackDecision.reason)
        assertEquals(FeedbackReason.SPEECH_DISABLED, disabled.feedbackDecision.reason)
    }

    @Test
    fun scenarioIsPassedToGatewayAndEvaluation() {
        val gateway = FakeFeedbackGateway(FeedbackDecision(null, triggered = true, reason = FeedbackReason.TRIGGERED))
        val coordinator = AssistSessionCoordinator(feedbackGateway = gateway, fpsTracker = fixedFpsTracker())

        val result = coordinator.processFrame(
            detectorFrame = detectorFrame(listOf(detection("person", BoundingBox(390f, 140f, 610f, 780f)))),
            profile = AlertProfile.SENSITIVE,
            scenario = AssistScenario.CORRIDOR,
            nowMs = 1000L
        )

        assertEquals(AssistScenario.CORRIDOR, result.evaluation.scenario)
        assertEquals(AssistScenario.CORRIDOR, gateway.lastScenario)
        assertTrue(result.explanation.headline.contains("走廊通行"))
    }

    @Test
    fun startSessionResetsFeedbackState() {
        val gateway = FakeFeedbackGateway(FeedbackDecision(null, false, FeedbackReason.NO_FEEDBACK_RISK))
        val coordinator = AssistSessionCoordinator(feedbackGateway = gateway, fpsTracker = fixedFpsTracker())

        coordinator.startSession(nowMs = 1000L)

        assertEquals(1, gateway.resetCalls)
    }

    @Test
    fun alertedSegmentationEventDoesNotCallGatewayAgain() {
        val gateway = FakeFeedbackGateway(FeedbackDecision(null, true, FeedbackReason.TRIGGERED))
        val coordinator = AssistSessionCoordinator(feedbackGateway = gateway, fpsTracker = fixedFpsTracker())
        val segmentation = detectorFrame(listOf(detection("stairs", BoundingBox(390f, 500f, 610f, 980f), DetectionSource.SEGMENTATION)))

        coordinator.processFrame(segmentation, AlertProfile.STANDARD, AssistScenario.GENERAL, nowMs = 1000L)
        val repeated = coordinator.processFrame(segmentation, AlertProfile.STANDARD, AssistScenario.GENERAL, nowMs = 1100L)

        assertEquals(1, gateway.notifyCalls)
        assertEquals(FeedbackReason.EVENT_ALREADY_ALERTED, repeated.feedbackDecision.reason)
    }

    @Test
    fun lowConfidenceSidePersonNeedsTwoMatchingFramesBeforeFeedback() {
        val gateway = FakeFeedbackGateway(FeedbackDecision(null, true, FeedbackReason.TRIGGERED))
        val coordinator = AssistSessionCoordinator(feedbackGateway = gateway, fpsTracker = fixedFpsTracker())
        val first = detectorFrame(listOf(detection("person", BoundingBox(20f, 200f, 280f, 780f), confidence = 0.49f)))
        val second = detectorFrame(listOf(detection("person", BoundingBox(30f, 210f, 290f, 790f), confidence = 0.49f)))

        val pending = coordinator.processFrame(first, AlertProfile.SENSITIVE, AssistScenario.GENERAL, nowMs = 1000L)
        assertEquals(FeedbackReason.UNSTABLE_RISK, pending.feedbackDecision.reason)
        assertEquals(0, gateway.notifyCalls)

        val confirmed = coordinator.processFrame(second, AlertProfile.SENSITIVE, AssistScenario.GENERAL, nowMs = 1100L)
        assertEquals(FeedbackReason.TRIGGERED, confirmed.feedbackDecision.reason)
        assertEquals(1, gateway.notifyCalls)
    }

    @Test
    fun centerAndHighConfidenceSidePersonKeepImmediateFeedbackPath() {
        val centerGateway = FakeFeedbackGateway(FeedbackDecision(null, true, FeedbackReason.TRIGGERED))
        val centerCoordinator = AssistSessionCoordinator(feedbackGateway = centerGateway, fpsTracker = fixedFpsTracker())
        centerCoordinator.processFrame(
            detectorFrame(listOf(detection("person", BoundingBox(390f, 140f, 610f, 780f), confidence = 0.49f))),
            AlertProfile.STANDARD,
            AssistScenario.GENERAL,
            nowMs = 1000L
        )

        val sideGateway = FakeFeedbackGateway(FeedbackDecision(null, true, FeedbackReason.TRIGGERED))
        val sideCoordinator = AssistSessionCoordinator(feedbackGateway = sideGateway, fpsTracker = fixedFpsTracker())
        sideCoordinator.processFrame(
            detectorFrame(listOf(detection("person", BoundingBox(20f, 200f, 280f, 780f), confidence = 0.50f))),
            AlertProfile.SENSITIVE,
            AssistScenario.GENERAL,
            nowMs = 1000L
        )

        assertEquals(1, centerGateway.notifyCalls)
        assertEquals(1, sideGateway.notifyCalls)
    }

    @Test
    fun stampedFrameUsesCaptureTimeButRetainsIndependentDecisionTime() {
        val coordinator = AssistSessionCoordinator(
            feedbackGateway = FakeFeedbackGateway(FeedbackDecision(null, false, FeedbackReason.NO_FEEDBACK_RISK)),
            fpsTracker = fixedFpsTracker()
        )
        val stamp = stamp(frameId = 4L, capturedAtMs = 1_250L, receivedAtMs = 1_300L)

        val result = coordinator.processFrame(
            detectorFrame = detectorFrame(emptyList(), stamp),
            profile = AlertProfile.STANDARD,
            scenario = AssistScenario.GENERAL,
            decisionAtNs = 9_000_000_000L
        )

        assertEquals(1_250L, result.evaluation.evaluatedAtMs)
        assertEquals(9_000_000_000L, result.evaluation.decisionAtNs)
        assertEquals(stamp, result.evaluation.sourceFrame)
    }

    @Test
    fun processingDelayDoesNotChangeCaptureBoundApproachTrend() {
        val captureTimes = listOf(1_000L, 1_100L, 1_200L)
        val boxes = listOf(
            BoundingBox(400f, 200f, 600f, 600f),
            BoundingBox(390f, 180f, 610f, 680f),
            BoundingBox(370f, 140f, 630f, 780f)
        )

        fun replay(decisionOffsetMs: Long): List<com.linnan.blindassist.risk.ApproachTrend> {
            val coordinator = AssistSessionCoordinator(
                feedbackGateway = FakeFeedbackGateway(FeedbackDecision(null, false, FeedbackReason.NO_FEEDBACK_RISK)),
                fpsTracker = fixedFpsTracker()
            )
            return boxes.mapIndexed { index, box ->
                val captureMs = captureTimes[index]
                coordinator.processFrame(
                    detectorFrame = detectorFrame(
                        listOf(detection("person", box)),
                        stamp(index.toLong(), captureMs, captureMs + 10L)
                    ),
                    profile = AlertProfile.STANDARD,
                    scenario = AssistScenario.GENERAL,
                    decisionAtNs = (captureMs + decisionOffsetMs) * 1_000_000L
                ).evaluation.rawRisk.approachTrend
            }
        }

        assertEquals(replay(decisionOffsetMs = 20L), replay(decisionOffsetMs = 2_000L))
    }

    private fun fixedFpsTracker(): FpsTracker {
        return FpsTracker(clock = { 1000L }).also { it.onFrame() }
    }

    private fun detectorFrame(
        detections: List<Detection>,
        sourceFrame: FrameStamp? = null
    ): DetectorFrameResult {
        return DetectorFrameResult(detections, frame, metrics(), sourceFrame)
    }

    private fun stamp(frameId: Long, capturedAtMs: Long, receivedAtMs: Long): FrameStamp = FrameStamp(
        frameId = frameId,
        capturedAtNs = capturedAtMs * 1_000_000L,
        receivedAtNs = receivedAtMs * 1_000_000L,
        sourceId = "camera2:0",
        coordinateFrame = "camera2:0:analysis-buffer",
        clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
    )

    private fun detection(
        label: String,
        box: BoundingBox,
        source: DetectionSource = DetectionSource.OBJECT_DETECTOR,
        confidence: Float = 0.9f
    ): Detection {
        return Detection(0, label, confidence, box, frame, source = source)
    }

    private fun metrics(): DetectorMetrics {
        return DetectorMetrics(35L, 5L, 22L, 8L, 0f, "ready")
    }

    private class FakeFeedbackGateway(private val decision: FeedbackDecision) : FeedbackGateway {
        var lastRisk: RiskResult? = null
        var lastScenario: AssistScenario? = null
        var resetCalls = 0
            private set
        var notifyCalls = 0
            private set

        override fun resetSession() {
            resetCalls += 1
        }

        override fun notify(
            risk: RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
            notifyCalls += 1
            lastRisk = risk
            lastScenario = scenario
            return decision
        }
    }
}
