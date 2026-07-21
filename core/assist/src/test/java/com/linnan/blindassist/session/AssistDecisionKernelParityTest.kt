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
import com.linnan.blindassist.risk.ApproachTrend
import com.linnan.blindassist.risk.DistanceEvidence
import com.linnan.blindassist.risk.DistanceEvidenceSource
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskEventState
import com.linnan.blindassist.risk.RiskFusionReason
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.vision.DetectorFrameResult
import org.junit.Assert.assertEquals
import org.junit.Test

class AssistDecisionKernelParityTest {
    private val frameSize = FrameSize(1000, 1000)

    @Test
    fun productionCoordinatorAndDirectKernelStayFrameExactAcrossFrozenSequence() {
        val coordinatorGateway = PlannerGateway()
        val kernelGateway = PlannerGateway()
        val coordinator = AssistSessionCoordinator(
            feedbackGateway = coordinatorGateway,
            fpsTracker = FpsTracker(clock = { 1000L })
        )
        val kernel = AssistDecisionKernel()
        coordinator.startSession(900L)
        kernelGateway.resetSession()
        kernel.startSession(900L)

        val sequence = listOf(
            frozenFrame(detection("person", BoundingBox(390f, 140f, 610f, 780f))) to 1000L,
            frozenFrame(detection("person", BoundingBox(20f, 200f, 280f, 780f), confidence = 0.49f)) to 1100L,
            frozenFrame(detection("person", BoundingBox(30f, 210f, 290f, 790f), confidence = 0.49f)) to 1200L,
            frozenFrame(detection("stairs", BoundingBox(390f, 500f, 610f, 980f), DetectionSource.SEGMENTATION)) to 1300L,
            frozenFrame(detection("stairs", BoundingBox(390f, 500f, 610f, 980f), DetectionSource.SEGMENTATION)) to 1400L,
            frozenFrame() to 1500L
        )

        sequence.forEachIndexed { index, (frame, nowMs) ->
            val production = coordinator.processFrame(
                detectorFrame = frame,
                profile = AlertProfile.STANDARD,
                scenario = AssistScenario.GENERAL,
                nowMs = nowMs
            )
            val benchmark = kernel.processFrame(
                detections = frame.detections,
                frameSize = frame.frameSize,
                profile = AlertProfile.STANDARD,
                scenario = AssistScenario.GENERAL,
                metrics = frame.metrics,
                feedbackGateway = kernelGateway,
                nowMs = nowMs
            )

            assertEquals("raw risk frame $index", production.evaluation.rawRisk, benchmark.evaluation.rawRisk)
            assertEquals("stable risk frame $index", production.evaluation.stableRisk, benchmark.evaluation.stableRisk)
            assertEquals("event frame $index", production.evaluation.riskEvent, benchmark.evaluation.riskEvent)
            assertEquals("feedback frame $index", production.feedbackDecision, benchmark.feedbackDecision)
            assertEquals("explanation frame $index", production.explanation, benchmark.explanation)
            assertEquals("summary frame $index", production.sessionSummary, benchmark.sessionSummary)
        }
        assertEquals(coordinatorGateway.notifyCalls, kernelGateway.notifyCalls)
    }

    @Test
    fun frozenSegmentationSequenceMatchesPreRefactorProductionDecisionMatrix() {
        val gateway = PlannerGateway()
        val kernel = AssistDecisionKernel()
        kernel.startSession(900L)
        val stairs = detection(
            label = "stairs",
            box = BoundingBox(390f, 500f, 610f, 980f),
            source = DetectionSource.SEGMENTATION
        )

        val actual = listOf(1000L, 1100L, 1200L, 1300L).map { nowMs ->
            val result = kernel.processFrame(
                detections = listOf(stairs),
                frameSize = frameSize,
                profile = AlertProfile.STANDARD,
                scenario = AssistScenario.GENERAL,
                metrics = DetectorMetrics(35L, 5L, 22L, 8L, 0f, "ready"),
                feedbackGateway = gateway,
                nowMs = nowMs
            )
            FrozenDecision(
                rawLevel = result.evaluation.rawRisk.level,
                rawDirection = result.evaluation.rawRisk.direction,
                rawProximity = result.evaluation.rawRisk.proximity,
                rawTrend = result.evaluation.rawRisk.approachTrend,
                rawFusion = result.evaluation.rawRisk.scoreBreakdown.fusionSummary,
                stableLevel = result.evaluation.stableRisk.level,
                stableDirection = result.evaluation.stableRisk.direction,
                stableProximity = result.evaluation.stableRisk.proximity,
                feedbackReason = result.feedbackDecision.reason,
                eventState = result.evaluation.riskEvent.state,
                suppressesFeedback = result.evaluation.riskEvent.suppressesFeedback
            )
        }

        assertEquals(
            listOf(
                FrozenDecision(
                    RiskLevel.LOW,
                    RiskDirection.CENTER,
                    ProximityBand.MID,
                    ApproachTrend.UNKNOWN,
                    RiskFusionReason.GEOMETRY_ONLY.name,
                    RiskLevel.LOW,
                    RiskDirection.CENTER,
                    ProximityBand.MID,
                    FeedbackReason.DISTANCE_TOO_FAR,
                    RiskEventState.APPROACHING,
                    false
                ),
                FrozenDecision(
                    RiskLevel.MEDIUM,
                    RiskDirection.CENTER,
                    ProximityBand.MID,
                    ApproachTrend.UNKNOWN,
                    RiskFusionReason.STABILITY_PROMOTED.name,
                    RiskLevel.NONE,
                    RiskDirection.NONE,
                    ProximityBand.MID,
                    FeedbackReason.UNSTABLE_RISK,
                    RiskEventState.APPROACHING,
                    false
                ),
                FrozenDecision(
                    RiskLevel.MEDIUM,
                    RiskDirection.CENTER,
                    ProximityBand.MID,
                    ApproachTrend.STABLE,
                    RiskFusionReason.STABILITY_PROMOTED.name,
                    RiskLevel.MEDIUM,
                    RiskDirection.CENTER,
                    ProximityBand.MID,
                    FeedbackReason.TRIGGERED,
                    RiskEventState.ALERTED,
                    true
                ),
                FrozenDecision(
                    RiskLevel.MEDIUM,
                    RiskDirection.CENTER,
                    ProximityBand.MID,
                    ApproachTrend.STABLE,
                    RiskFusionReason.STABILITY_PROMOTED.name,
                    RiskLevel.MEDIUM,
                    RiskDirection.CENTER,
                    ProximityBand.MID,
                    FeedbackReason.EVENT_ALREADY_ALERTED,
                    RiskEventState.ALERTED,
                    true
                )
            ),
            actual
        )
        assertEquals(3, gateway.notifyCalls)
    }

    @Test
    fun unavailableFeedbackReceiptDoesNotConsumeSegmentationEvent() {
        val unavailable = FeedbackDecision(null, false, FeedbackReason.FEEDBACK_UNAVAILABLE)
        val acceptedPlan = requireNotNull(
            FeedbackPlanner.planFor(
                RiskResult(
                    level = RiskLevel.HIGH,
                    direction = RiskDirection.CENTER,
                    message = "fixture",
                    proximity = ProximityBand.CRITICAL
                )
            )
        )
        val gateway = QueueGateway(
            ArrayDeque(
                listOf(
                    unavailable,
                    FeedbackDecision(acceptedPlan, true, FeedbackReason.TRIGGERED)
                )
            )
        )
        val kernel = AssistDecisionKernel()
        val stairs = Detection(
            classId = 0,
            label = "stairs",
            confidence = 0.9f,
            boundingBox = BoundingBox(390f, 500f, 610f, 980f),
            frameSize = frameSize,
            distanceEvidence = DistanceEvidence(
                band = ProximityBand.CRITICAL,
                confidence = 1f,
                source = DistanceEvidenceSource.ARCORE_DEPTH,
                relativeDepthScore = 1f
            ),
            source = DetectionSource.SEGMENTATION
        )

        val first = kernel.processFrame(
            listOf(stairs),
            frameSize,
            AlertProfile.STANDARD,
            AssistScenario.GENERAL,
            DetectorMetrics(35L, 5L, 22L, 8L, 0f, "ready"),
            gateway,
            1000L
        )
        val retry = kernel.processFrame(
            listOf(stairs),
            frameSize,
            AlertProfile.STANDARD,
            AssistScenario.GENERAL,
            DetectorMetrics(35L, 5L, 22L, 8L, 0f, "ready"),
            gateway,
            1100L
        )

        assertEquals(FeedbackReason.FEEDBACK_UNAVAILABLE, first.feedbackDecision.reason)
        assertEquals(RiskEventState.APPROACHING, first.evaluation.riskEvent.state)
        assertEquals(false, first.evaluation.riskEvent.suppressesFeedback)
        assertEquals(FeedbackReason.TRIGGERED, retry.feedbackDecision.reason)
        assertEquals(RiskEventState.ALERTED, retry.evaluation.riskEvent.state)
        assertEquals(true, retry.evaluation.riskEvent.suppressesFeedback)
        assertEquals(2, gateway.notifyCalls)
    }

    private fun frozenFrame(vararg detections: Detection): DetectorFrameResult = DetectorFrameResult(
        detections = detections.toList(),
        frameSize = frameSize,
        metrics = DetectorMetrics(35L, 5L, 22L, 8L, 0f, "ready")
    )

    private fun detection(
        label: String,
        box: BoundingBox,
        source: DetectionSource = DetectionSource.OBJECT_DETECTOR,
        confidence: Float = 0.9f
    ): Detection = Detection(0, label, confidence, box, frameSize, source = source)

    private data class FrozenDecision(
        val rawLevel: RiskLevel,
        val rawDirection: RiskDirection,
        val rawProximity: ProximityBand,
        val rawTrend: ApproachTrend,
        val rawFusion: String,
        val stableLevel: RiskLevel,
        val stableDirection: RiskDirection,
        val stableProximity: ProximityBand,
        val feedbackReason: FeedbackReason,
        val eventState: RiskEventState?,
        val suppressesFeedback: Boolean
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

    private class QueueGateway(
        private val decisions: ArrayDeque<FeedbackDecision>
    ) : FeedbackGateway {
        var notifyCalls = 0
            private set

        override fun resetSession() = Unit

        override fun notify(
            risk: RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
            notifyCalls += 1
            return decisions.removeFirst()
        }
    }
}
