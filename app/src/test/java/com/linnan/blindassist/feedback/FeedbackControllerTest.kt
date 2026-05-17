package com.linnan.blindassist.feedback

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
            risk(RiskLevel.HIGH, ProximityBand.CRITICAL)
        )

        assertEquals(FeedbackController.CRITICAL_ALERT_COOLDOWN_MS, plan?.cooldownMs)
        assertEquals(FeedbackController.CRITICAL_VIBRATION_MS, plan?.vibrationMs)
    }

    @Test
    fun nearHighOrMediumRiskUsesRegularAlertPlan() {
        val high = FeedbackController.planFor(risk(RiskLevel.HIGH, ProximityBand.NEAR))
        val medium = FeedbackController.planFor(risk(RiskLevel.MEDIUM, ProximityBand.NEAR))

        assertEquals(FeedbackController.NEAR_ALERT_COOLDOWN_MS, high?.cooldownMs)
        assertEquals(FeedbackController.NEAR_VIBRATION_MS, high?.vibrationMs)
        assertEquals(FeedbackController.NEAR_ALERT_COOLDOWN_MS, medium?.cooldownMs)
        assertEquals(FeedbackController.NEAR_VIBRATION_MS, medium?.vibrationMs)
    }

    @Test
    fun midAndFarRisksDoNotUseSpeechOrVibrationPlan() {
        assertNull(FeedbackController.planFor(risk(RiskLevel.LOW, ProximityBand.MID)))
        assertNull(FeedbackController.planFor(risk(RiskLevel.NONE, ProximityBand.FAR)))
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
