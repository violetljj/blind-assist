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
import com.linnan.blindassist.risk.ApproachTrend
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskEventState
import com.linnan.blindassist.risk.RiskEvidenceState
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.risk.RiskScoreBreakdown
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AssistDecisionKernelRiskEvidenceTest {
    private val frameSize = FrameSize(1_000, 1_000)
    private val metrics = DetectorMetrics(12, 0, 10, 2, 2f, "dense-risk-evidence-ready")

    @Test
    fun objectAgnosticEvidenceUsesSharedStabilizerEventFeedbackAndDuplicateGate() {
        val kernel = AssistDecisionKernel()
        val gateway = PlannerGateway()
        kernel.startSession(0)

        val first = process(kernel, gateway, nowMs = 0, raw = risk(RiskLevel.MEDIUM, 0.70f))
        val second = process(kernel, gateway, nowMs = 500, raw = risk(RiskLevel.MEDIUM, 0.70f))
        val third = process(kernel, gateway, nowMs = 1_000, raw = risk(RiskLevel.MEDIUM, 0.70f))

        assertEquals(RiskLevel.NONE, first.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.UNSTABLE_RISK, first.feedbackDecision.reason)
        assertEquals(RiskLevel.MEDIUM, second.evaluation.stableRisk.level)
        assertEquals(FeedbackReason.TRIGGERED, second.feedbackDecision.reason)
        assertEquals(RiskEventState.ALERTED, second.evaluation.riskEvent.state)
        assertTrue(second.evaluation.riskEvent.eventId!!.startsWith("risk-"))
        assertEquals(second.evaluation.riskEvent.eventId, third.evaluation.riskEvent.eventId)
        assertEquals(FeedbackReason.EVENT_ALREADY_ALERTED, third.feedbackDecision.reason)
        assertTrue(third.evaluation.riskEvent.suppressesFeedback)
        assertEquals(2, gateway.notifyCalls)
        listOf(first, second, third).forEach {
            assertEquals(null, it.evaluation.rawRisk.sourceDetection)
            assertEquals(0, it.evaluation.detectionCount)
            assertEquals(1, it.evaluation.riskEvidenceCount)
        }
    }

    @Test
    fun scalarHistoryAnnotatesRecedingAndKernelClearsWithoutSyntheticBox() {
        val kernel = AssistDecisionKernel()
        val gateway = PlannerGateway()
        kernel.startSession(0)
        val scores = listOf(1.0f, 0.9f, 0.7f, 0.5f, 0.3f)
        val results = scores.mapIndexed { index, score ->
            process(kernel, gateway, index * 200L, risk(RiskLevel.HIGH, score))
        }

        assertEquals(ApproachTrend.RECEDING, results[2].evaluation.rawRisk.approachTrend)
        assertEquals(RiskEventState.PASSED_OR_RECEDING, results[2].evaluation.riskEvent.state)
        assertEquals(RiskEventState.PASSED_OR_RECEDING, results[3].evaluation.riskEvent.state)
        assertEquals(null, results[4].evaluation.riskEvent.eventId)
        assertEquals(null, results[4].evaluation.rawRisk.sourceDetection)
    }

    @Test
    fun frameBindingContractAndObjectAgnosticBoundaryFailClosed() {
        val detection = Detection(0, "person", 0.9f, BoundingBox(0f, 0f, 100f, 100f), frameSize)
        val invalidRisk = risk(RiskLevel.HIGH, 1f).copy(sourceDetection = detection)
        val sourceError = runCatching {
            evidence(nowMs = 0, raw = invalidRisk)
        }.exceptionOrNull()
        assertNotNull(sourceError)

        val kernel = AssistDecisionKernel()
        kernel.startSession(0)
        val staleError = runCatching {
            kernel.processRiskEvidence(
                evidence = evidence(nowMs = 0, validUntilMs = 100, raw = risk(RiskLevel.HIGH, 1f)),
                frameSize = frameSize,
                profile = AlertProfile.STANDARD,
                scenario = AssistScenario.GENERAL,
                metrics = metrics,
                feedbackGateway = PlannerGateway(),
                nowMs = 101
            )
        }.exceptionOrNull()
        assertNotNull(staleError)
    }

    @Test
    fun normalizedEvidenceRejectsContradictoryNoneScoreAndNonMonotonicTime() {
        val contradictoryNone = risk(RiskLevel.NONE, 0f).copy(
            direction = RiskDirection.CENTER,
            proximity = ProximityBand.CRITICAL,
            evidenceState = RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE
        )
        assertNotNull(runCatching { evidence(nowMs = 0, raw = contradictoryNone) }.exceptionOrNull())
        assertNotNull(
            runCatching {
                evidence(
                    nowMs = 0,
                    raw = risk(RiskLevel.MEDIUM, 0.7f).copy(
                        scoreBreakdown = RiskScoreBreakdown(total = 0.6f)
                    )
                )
            }.exceptionOrNull()
        )

        val kernel = AssistDecisionKernel()
        val gateway = PlannerGateway()
        kernel.startSession(0)
        process(kernel, gateway, nowMs = 500, raw = risk(RiskLevel.MEDIUM, 0.7f))
        assertNotNull(
            runCatching {
                process(kernel, gateway, nowMs = 499, raw = risk(RiskLevel.MEDIUM, 0.7f))
            }.exceptionOrNull()
        )
    }

    private fun process(
        kernel: AssistDecisionKernel,
        gateway: FeedbackGateway,
        nowMs: Long,
        raw: RiskResult
    ): AssistFrameResult = kernel.processRiskEvidence(
        evidence = evidence(nowMs, raw = raw),
        frameSize = frameSize,
        profile = AlertProfile.STANDARD,
        scenario = AssistScenario.GENERAL,
        metrics = metrics,
        feedbackGateway = gateway,
        nowMs = nowMs
    )

    private fun evidence(
        nowMs: Long,
        validUntilMs: Long = nowMs + 500,
        raw: RiskResult
    ) = AssistRiskEvidenceFrame(
        sourceContractId = AssistDecisionKernel.RISK_EVIDENCE_INPUT_CONTRACT_ID,
        frameId = "frame-$nowMs",
        eventKey = "route-risk-episode-1",
        observedAtMs = nowMs,
        validUntilMs = validUntilMs,
        rawRisk = raw,
        evidenceCount = 1
    )

    private fun risk(level: RiskLevel, score: Float) = RiskResult(
        level = level,
        direction = if (level == RiskLevel.NONE) RiskDirection.NONE else RiskDirection.CENTER,
        message = if (level == RiskLevel.NONE) "route risk unavailable" else "route-relative dense risk evidence",
        sourceDetection = null,
        proximity = if (level == RiskLevel.NONE) ProximityBand.FAR else ProximityBand.NEAR,
        urgencyScore = score,
        riskScore = score,
        scoreBreakdown = RiskScoreBreakdown(total = score, fusionSummary = "EXTERNAL_DENSE_RISK_EVIDENCE"),
        approachTrend = ApproachTrend.UNKNOWN,
        evidenceState = if (level == RiskLevel.NONE) {
            RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE
        } else {
            RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE
        }
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
                FeedbackDecision(null, false, FeedbackReason.NO_FEEDBACK_RISK)
            } else {
                FeedbackDecision(plan, true, FeedbackReason.TRIGGERED)
            }
        }
    }
}
