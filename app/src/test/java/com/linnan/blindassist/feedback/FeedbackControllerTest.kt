package com.linnan.blindassist.feedback

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
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
    fun speechStylesUseBriefStandardAndDetailedMessages() {
        val risk = risk(RiskLevel.HIGH, ProximityBand.NEAR)

        assertEquals("正前方近处", SpeechStyle.BRIEF.messageFor(risk))
        assertEquals("测试提醒", SpeechStyle.STANDARD.messageFor(risk))
        assertEquals("正前方近处有障碍物，请注意避让", SpeechStyle.DETAILED.messageFor(risk))
    }

    @Test
    fun vibrationStrengthScalesDurationAndAmplitude() {
        val soft = FeedbackController.planFor(
            risk(RiskLevel.MEDIUM, ProximityBand.NEAR),
            AlertProfile.STANDARD,
            VibrationStrength.SOFT
        )
        val strong = FeedbackController.planFor(
            risk(RiskLevel.HIGH, ProximityBand.CRITICAL),
            AlertProfile.STANDARD,
            VibrationStrength.STRONG
        )

        assertEquals(120L, soft?.vibrationMs)
        assertEquals(96, soft?.amplitude)
        assertEquals(525L, strong?.vibrationMs)
        assertEquals(255, strong?.amplitude)
    }

    @Test
    fun fatigueLengthensRepeatedNonCriticalCooldownButNotCriticalRisk() {
        val fatigue = FeedbackFatigueController()
        val near = risk(RiskLevel.MEDIUM, ProximityBand.NEAR)
        val critical = risk(RiskLevel.HIGH, ProximityBand.CRITICAL)

        fatigue.recordTriggered(near, nowMs = 1000L)
        fatigue.recordTriggered(near, nowMs = 2400L)
        val tiredCooldown = fatigue.effectiveCooldownMs(near, baseCooldownMs = 1500L, nowMs = 3600L)
        val criticalCooldown = fatigue.effectiveCooldownMs(critical, baseCooldownMs = 850L, nowMs = 3600L)

        assertTrue(tiredCooldown > 1500L)
        assertEquals(850L, criticalCooldown)
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
