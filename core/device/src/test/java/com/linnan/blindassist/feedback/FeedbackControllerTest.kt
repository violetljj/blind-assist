package com.linnan.blindassist.feedback

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
    fun speechStylesUseEnglishMessagesWhenLanguageIsEnglish() {
        val risk = risk(RiskLevel.HIGH, ProximityBand.NEAR)

        assertEquals("front center near", SpeechStyle.BRIEF.messageFor(risk, AppLanguage.EN))
        assertEquals("front center near risk, avoid carefully", SpeechStyle.STANDARD.messageFor(risk, AppLanguage.EN))
        assertEquals("obstacle near front center, avoid carefully", SpeechStyle.DETAILED.messageFor(risk, AppLanguage.EN))
        assertEquals("No risk detected", SpeechStyle.STANDARD.messageFor(risk(RiskLevel.NONE, ProximityBand.FAR), AppLanguage.EN))
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
    fun scenarioAdjustsCooldownAndVibrationPlanWhileGeneralKeepsExistingBehavior() {
        val general = FeedbackController.planFor(
            risk(RiskLevel.MEDIUM, ProximityBand.NEAR),
            AlertProfile.STANDARD,
            VibrationStrength.STANDARD,
            AssistScenario.GENERAL
        )
        val corridor = FeedbackController.planFor(
            risk(RiskLevel.MEDIUM, ProximityBand.NEAR),
            AlertProfile.STANDARD,
            VibrationStrength.STANDARD,
            AssistScenario.CORRIDOR
        )
        val crowded = FeedbackController.planFor(
            risk(RiskLevel.MEDIUM, ProximityBand.NEAR),
            AlertProfile.STANDARD,
            VibrationStrength.STANDARD,
            AssistScenario.CROWDED
        )

        assertEquals(FeedbackController.STANDARD_NEAR_ALERT_COOLDOWN_MS, general?.cooldownMs)
        assertEquals(FeedbackController.STANDARD_NEAR_VIBRATION_MS, general?.vibrationMs)
        assertEquals(1350L, corridor?.cooldownMs)
        assertEquals(180L, corridor?.vibrationMs)
        assertEquals(2200L, crowded?.cooldownMs)
        assertEquals(140L, crowded?.vibrationMs)
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

    @Test
    fun notifyReportsUnavailableWhenNoOutputChannelAcceptsFeedback() {
        var nowMs = 10_000L
        val speech = FakeSpeechOutput(ready = false, speakResult = true)
        val haptic = FakeHapticOutput(vibrateResult = false)
        val controller = feedbackController(speech, haptic) { nowMs }

        val unavailable = controller.notify(risk(RiskLevel.HIGH, ProximityBand.CRITICAL))

        assertFalse(unavailable.triggered)
        assertEquals(FeedbackReason.FEEDBACK_UNAVAILABLE, unavailable.reason)
        assertFalse(unavailable.speechTriggered)
        assertFalse(unavailable.vibrationTriggered)
        assertEquals(0, speech.speakCalls)
        assertEquals(1, haptic.vibrateCalls)

        speech.ready = true
        nowMs += 100L
        val delivered = controller.notify(risk(RiskLevel.HIGH, ProximityBand.CRITICAL))

        assertTrue(delivered.triggered)
        assertEquals(FeedbackReason.TRIGGERED, delivered.reason)
        assertTrue(delivered.speechTriggered)
        assertFalse(delivered.vibrationTriggered)
    }

    @Test
    fun notifyTreatsSuccessfulVibrationAsRealFeedbackWhenSpeechIsDisabled() {
        val speech = FakeSpeechOutput(ready = true, speakResult = true)
        val haptic = FakeHapticOutput(vibrateResult = true)
        val controller = feedbackController(speech, haptic)
        controller.applySettings(FeedbackRuntimeSettings(speechEnabled = false))

        val decision = controller.notify(risk(RiskLevel.HIGH, ProximityBand.CRITICAL))

        assertTrue(decision.triggered)
        assertEquals(FeedbackReason.TRIGGERED, decision.reason)
        assertFalse(decision.speechTriggered)
        assertTrue(decision.vibrationTriggered)
        assertEquals(0, speech.speakCalls)
        assertEquals(1, haptic.vibrateCalls)
    }

    @Test
    fun notifyDoesNotCallOutputsWhenBothFeedbackSwitchesAreDisabled() {
        val speech = FakeSpeechOutput(ready = true, speakResult = true)
        val haptic = FakeHapticOutput(vibrateResult = true)
        val controller = feedbackController(speech, haptic)
        controller.applySettings(
            FeedbackRuntimeSettings(
                speechEnabled = false,
                vibrationEnabled = false
            )
        )

        val decision = controller.notify(risk(RiskLevel.HIGH, ProximityBand.CRITICAL))

        assertFalse(decision.triggered)
        assertEquals(FeedbackReason.FEEDBACK_UNAVAILABLE, decision.reason)
        assertEquals(0, speech.speakCalls)
        assertEquals(0, haptic.vibrateCalls)
    }

    @Test
    fun notifyReportsUnavailableWhenSpeechSpeakFailsAndHapticIsOff() {
        val speech = FakeSpeechOutput(ready = true, speakResult = false)
        val haptic = FakeHapticOutput(vibrateResult = false)
        val controller = feedbackController(speech, haptic)
        controller.applySettings(FeedbackRuntimeSettings(vibrationEnabled = false))

        val decision = controller.notify(risk(RiskLevel.HIGH, ProximityBand.CRITICAL))

        assertFalse(decision.triggered)
        assertEquals(FeedbackReason.FEEDBACK_UNAVAILABLE, decision.reason)
        assertFalse(decision.speechTriggered)
        assertFalse(decision.vibrationTriggered)
        assertEquals(1, speech.speakCalls)
        assertEquals(0, haptic.vibrateCalls)
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

    private fun feedbackController(
        speech: FakeSpeechOutput,
        haptic: FakeHapticOutput,
        nowMs: () -> Long = { 10_000L }
    ): FeedbackController {
        return FeedbackController(
            speechOutput = speech,
            hapticOutput = haptic,
            clock = FeedbackClock { nowMs() },
            initialSettings = FeedbackRuntimeSettings()
        )
    }

    private class FakeSpeechOutput(
        override var ready: Boolean,
        private val speakResult: Boolean
    ) : SpeechOutput {
        var speakCalls: Int = 0
            private set
        var language: AppLanguage = AppLanguage.ZH
            private set

        override fun setLanguage(language: AppLanguage) {
            this.language = language
        }

        override fun speak(message: String, utteranceId: String): Boolean {
            speakCalls += 1
            return speakResult
        }

        override fun shutdown() = Unit
    }

    private class FakeHapticOutput(
        private val vibrateResult: Boolean
    ) : HapticOutput {
        var vibrateCalls: Int = 0
            private set

        override fun vibrate(plan: FeedbackPlan): Boolean {
            vibrateCalls += 1
            return vibrateResult
        }
    }
}
