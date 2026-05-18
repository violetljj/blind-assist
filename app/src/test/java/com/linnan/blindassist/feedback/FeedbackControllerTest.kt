package com.linnan.blindassist.feedback

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FeedbackControllerTest {
    @Test
    fun criticalHighRiskUsesUrgentFeedbackPlan() {
        val plan = FeedbackController.planFor(
            risk(RiskLevel.HIGH, ProximityBand.CRITICAL),
            AlertProfile.STANDARD
        )

        assertEquals(FeedbackController.STANDARD_CRITICAL_ALERT_COOLDOWN_MS, plan?.cooldownMs)
        assertEquals(FeedbackController.STANDARD_CRITICAL_VIBRATION_MS, plan?.vibrationMs)
    }

    @Test
    fun nearHighOrMediumRiskUsesRegularAlertPlan() {
        val high = FeedbackController.planFor(risk(RiskLevel.HIGH, ProximityBand.NEAR), AlertProfile.STANDARD)
        val medium = FeedbackController.planFor(risk(RiskLevel.MEDIUM, ProximityBand.NEAR), AlertProfile.STANDARD)

        assertEquals(FeedbackController.STANDARD_NEAR_ALERT_COOLDOWN_MS, high?.cooldownMs)
        assertEquals(FeedbackController.STANDARD_NEAR_VIBRATION_MS, high?.vibrationMs)
        assertEquals(FeedbackController.STANDARD_NEAR_ALERT_COOLDOWN_MS, medium?.cooldownMs)
        assertEquals(FeedbackController.STANDARD_NEAR_VIBRATION_MS, medium?.vibrationMs)
    }

    @Test
    fun quietProfileUsesSofterFeedbackPlan() {
        val near = FeedbackController.planFor(risk(RiskLevel.MEDIUM, ProximityBand.NEAR), AlertProfile.QUIET)
        val critical = FeedbackController.planFor(risk(RiskLevel.HIGH, ProximityBand.CRITICAL), AlertProfile.QUIET)

        assertEquals(2200L, near?.cooldownMs)
        assertEquals(100L, near?.vibrationMs)
        assertEquals(1200L, critical?.cooldownMs)
        assertEquals(260L, critical?.vibrationMs)
    }

    @Test
    fun sensitiveProfileUsesFasterFeedbackPlan() {
        val near = FeedbackController.planFor(risk(RiskLevel.MEDIUM, ProximityBand.NEAR), AlertProfile.SENSITIVE)
        val critical = FeedbackController.planFor(risk(RiskLevel.HIGH, ProximityBand.CRITICAL), AlertProfile.SENSITIVE)

        assertEquals(1000L, near?.cooldownMs)
        assertEquals(220L, near?.vibrationMs)
        assertEquals(650L, critical?.cooldownMs)
        assertEquals(520L, critical?.vibrationMs)
    }

    @Test
    fun midAndFarRisksDoNotUseSpeechOrVibrationPlan() {
        assertNull(FeedbackController.planFor(risk(RiskLevel.LOW, ProximityBand.MID)))
        assertNull(FeedbackController.planFor(risk(RiskLevel.NONE, ProximityBand.FAR)))
    }

    @Test
    fun feedbackDecisionDefaultsKeepExistingTestConstructionCompatible() {
        val decision = FeedbackDecision(
            plan = FeedbackController.planFor(risk(RiskLevel.HIGH, ProximityBand.NEAR)),
            triggered = true,
            reason = FeedbackReason.TRIGGERED
        )

        assertEquals(false, decision.speechTriggered)
        assertEquals(false, decision.vibrationTriggered)
    }

    private fun risk(level: RiskLevel, proximity: ProximityBand): RiskResult {
        return RiskResult(
            level = level,
            direction = RiskDirection.CENTER,
            message = "测试提醒",
            proximity = proximity,
            urgencyScore = proximity.ordinal.toFloat()
        )
    }
}
