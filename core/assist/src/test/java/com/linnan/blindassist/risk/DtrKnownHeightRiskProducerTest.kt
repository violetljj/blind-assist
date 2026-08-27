package com.linnan.blindassist.risk

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.vision.CameraIntrinsics
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import com.linnan.blindassist.session.AssistDecisionKernel
import com.linnan.blindassist.session.AssistDtrEvidenceFrame
import com.linnan.blindassist.session.DetectorMetrics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DtrKnownHeightRiskProducerTest {
    private val frameSize = FrameSize(640, 480)
    private val intrinsics = CameraIntrinsics(500f, 500f, 320f, 240f, 640, 480)

    @Test
    fun crossingTriggersOneEventWhileLateralPassClearsAndUnknownCannotClear() {
        val crossing = DtrKnownHeightRiskProducer()
        assertEquals(
            DtrSignal.UNKNOWN,
            crossing.process(listOf(person(4.0, 2.0)), frameSize, intrinsics, stamp(0, 0.0)).signal
        )
        val onset = crossing.process(
            listOf(person(3.7, 1.85)),
            frameSize,
            intrinsics,
            stamp(1, 0.25)
        )
        assertEquals(DtrSignal.ONSET, onset.signal)
        assertTrue(onset.minimumSeparationM!! <= onset.intersectionThresholdM!!)
        assertEquals(
            DtrSignal.HOLD,
            crossing.process(
                listOf(person(3.4, 1.7)),
                frameSize,
                intrinsics,
                stamp(2, 0.50)
            ).signal
        )
        assertEquals(
            DtrSignal.UNKNOWN,
            crossing.process(
                listOf(person(3.1, 1.55)),
                frameSize,
                null,
                stamp(3, 0.80)
            ).signal
        )
        assertEquals(
            DtrSignal.HOLD,
            crossing.process(emptyList(), frameSize, intrinsics, stamp(4, 0.90)).signal
        )
        assertEquals(
            DtrSignal.CLEAR,
            crossing.process(emptyList(), frameSize, intrinsics, stamp(5, 1.40)).signal
        )

        val lateral = DtrKnownHeightRiskProducer()
        lateral.process(listOf(person(4.0, 2.0)), frameSize, intrinsics, stamp(0, 0.0))
        val safePass = lateral.process(
            listOf(person(4.0, 1.5)),
            frameSize,
            intrinsics,
            stamp(1, 0.25)
        )
        assertEquals(DtrSignal.CLEAR, safePass.signal)
        assertTrue(safePass.minimumSeparationM!! > safePass.intersectionThresholdM!!)
    }

    @Test
    fun decisionKernelNotifiesOnceAndPreservesUnknownUntilExplicitClear() {
        val gateway = CountingGateway()
        val kernel = AssistDecisionKernel()
        kernel.startSession(0L)

        val onset = kernel.processDtrEvidence(
            evidence(DtrSignal.ONSET, 0L, activeRisk()),
            frameSize,
            AlertProfile.STANDARD,
            AssistScenario.GENERAL,
            metrics(),
            gateway,
            nowMs = 0L
        )
        val hold = kernel.processDtrEvidence(
            evidence(DtrSignal.HOLD, 100L, activeRisk()),
            frameSize,
            AlertProfile.STANDARD,
            AssistScenario.GENERAL,
            metrics(),
            gateway,
            nowMs = 100L
        )
        val unknown = kernel.processDtrEvidence(
            evidence(DtrSignal.UNKNOWN, 200L, activeRisk()),
            frameSize,
            AlertProfile.STANDARD,
            AssistScenario.GENERAL,
            metrics(),
            gateway,
            nowMs = 200L
        )
        val clear = kernel.processDtrEvidence(
            evidence(DtrSignal.CLEAR, 700L, clearRisk()),
            frameSize,
            AlertProfile.STANDARD,
            AssistScenario.GENERAL,
            metrics(),
            gateway,
            nowMs = 700L
        )

        assertTrue(onset.feedbackDecision.triggered)
        assertEquals(FeedbackReason.EVENT_ALREADY_ALERTED, hold.feedbackDecision.reason)
        assertTrue(unknown.evaluation.riskEvent.active)
        assertEquals(FeedbackReason.UNSTABLE_RISK, unknown.feedbackDecision.reason)
        assertEquals(RiskEventState.CLEARED, clear.evaluation.riskEvent.state)
        assertEquals(1, gateway.notifications)
    }

    private fun person(forwardM: Double, leftM: Double): Detection {
        val height = (intrinsics.focalLengthYPx * 1.70 / forwardM).toFloat()
        val centerX = (
            intrinsics.principalPointXPx -
                leftM / forwardM * intrinsics.focalLengthXPx
            ).toFloat()
        return Detection(
            classId = 0,
            label = "person",
            confidence = 0.9f,
            boundingBox = BoundingBox(
                left = centerX - 40f,
                top = intrinsics.principalPointYPx - height / 2f,
                right = centerX + 40f,
                bottom = intrinsics.principalPointYPx + height / 2f
            ),
            frameSize = frameSize
        )
    }

    private fun stamp(frameId: Long, seconds: Double): FrameStamp {
        val timeNs = (seconds * 1_000_000_000.0).toLong()
        return FrameStamp(
            frameId = frameId,
            capturedAtNs = timeNs,
            receivedAtNs = timeNs,
            sourceId = "camera2:0",
            coordinateFrame = "camera2:0:analysis-buffer",
            clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
        )
    }

    private fun evidence(signal: DtrSignal, nowMs: Long, risk: RiskResult) =
        AssistDtrEvidenceFrame(
            sourceContractId = AssistDecisionKernel.DTR_EVIDENCE_INPUT_CONTRACT_ID,
            frameId = "frame-$nowMs",
            eventKey = "dtr-route-1",
            signal = signal,
            observedAtMs = nowMs,
            validUntilMs = nowMs + 250L,
            risk = risk,
            evidenceCount = 1
        )

    private fun activeRisk() = RiskResult(
        level = RiskLevel.HIGH,
        direction = RiskDirection.CENTER,
        message = "route intersection",
        proximity = ProximityBand.CRITICAL,
        urgencyScore = 1f,
        riskScore = 1f,
        scoreBreakdown = RiskScoreBreakdown(total = 1f),
        evidenceState = RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE
    )

    private fun clearRisk() = RiskResult(
        level = RiskLevel.NONE,
        direction = RiskDirection.NONE,
        message = "clear"
    )

    private fun metrics() = DetectorMetrics(1L, 0L, 1L, 0L, 24f, "ready")

    private class CountingGateway : FeedbackGateway {
        var notifications = 0

        override fun resetSession() = Unit

        override fun notify(
            risk: RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
            notifications += 1
            return FeedbackDecision(null, true, FeedbackReason.TRIGGERED)
        }
    }
}
