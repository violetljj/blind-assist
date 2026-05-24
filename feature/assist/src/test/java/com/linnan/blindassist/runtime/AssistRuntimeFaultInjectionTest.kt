package com.linnan.blindassist.runtime

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackClock
import com.linnan.blindassist.feedback.FeedbackController
import com.linnan.blindassist.feedback.FeedbackPlan
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.feedback.FeedbackRuntimeSettings
import com.linnan.blindassist.feedback.HapticOutput
import com.linnan.blindassist.feedback.SpeechOutput
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AssistRuntimeFaultInjectionTest {
    @Test
    fun modelUnavailableAfterCameraStartedEntersErrorAndRendersModelState() {
        val transition = AssistRuntimeStateMachine(initialState = AssistRuntimeState.Starting, cameraViewsReady = true)
            .onEvent(AssistRuntimeEvent.CameraStarted(modelReady = false))

        assertEquals(AssistRuntimeState.Error("Model unavailable"), transition.state)
        assertTrue(transition.hasRenderTarget(AssistRuntimeRenderTarget.ModelUnavailable))
    }

    @Test
    fun detectorExceptionRoutedAsCameraSourceFailureStopsAndResetsRuntime() {
        val transition = AssistRuntimeStateMachine(initialState = AssistRuntimeState.Running, cameraViewsReady = true)
            .onEvent(AssistRuntimeEvent.CameraSourceFailed("detect failed"))

        assertEquals(AssistRuntimeState.Error("detect failed"), transition.state)
        assertTrue(transition.effects.contains(AssistRuntimeEffect.StopCamera))
        assertTrue(transition.effects.contains(AssistRuntimeEffect.ClearOverlay))
        assertTrue(transition.effects.contains(AssistRuntimeEffect.ResetSession))
        assertTrue(transition.hasRenderTarget(AssistRuntimeRenderTarget.CameraError))
    }

    @Test
    fun cameraSourceFailureBeforeRunningStillCleansPartialSession() {
        val transition = AssistRuntimeStateMachine(initialState = AssistRuntimeState.Starting, cameraViewsReady = true)
            .onEvent(AssistRuntimeEvent.CameraSourceFailed("camera open failed"))

        assertEquals(AssistRuntimeState.Error("camera open failed"), transition.state)
        assertTrue(transition.effects.contains(AssistRuntimeEffect.StopCamera))
        assertTrue(transition.effects.contains(AssistRuntimeEffect.ClearOverlay))
        assertTrue(transition.effects.contains(AssistRuntimeEffect.ResetSession))
        assertTrue(transition.hasRenderTarget(AssistRuntimeRenderTarget.CameraError))
    }

    @Test
    fun closeCameraAfterRuntimeErrorStillStopsClosesAndResets() {
        val machine = AssistRuntimeStateMachine(initialState = AssistRuntimeState.Running, cameraViewsReady = true)
        machine.onEvent(AssistRuntimeEvent.CameraSourceFailed("frame failure"))

        val close = machine.onEvent(AssistRuntimeEvent.CloseCamera)

        assertEquals(AssistRuntimeState.Idle, close.state)
        assertTrue(close.effects.contains(AssistRuntimeEffect.StopCamera))
        assertTrue(close.effects.contains(AssistRuntimeEffect.ClearOverlay))
        assertTrue(close.effects.contains(AssistRuntimeEffect.CloseCamera))
        assertTrue(close.effects.contains(AssistRuntimeEffect.ResetSession))
    }

    @Test
    fun detectionToggleWhileInErrorDoesNotRestartCameraSource() {
        val machine = AssistRuntimeStateMachine(initialState = AssistRuntimeState.Running, cameraViewsReady = true)
        machine.onEvent(AssistRuntimeEvent.CameraSourceFailed("frame failure"))

        val transition = machine.onEvent(AssistRuntimeEvent.DetectionChanged(enabled = true))

        assertEquals(AssistRuntimeState.Error("frame failure"), transition.state)
        assertTrue(!transition.effects.contains(AssistRuntimeEffect.StartCameraIfReady))
        assertTrue(!transition.effects.contains(AssistRuntimeEffect.StartSession))
    }

    @Test
    fun feedbackUnavailableAfterRuntimeConfigDisablesOutputsDoesNotTriggerFeedback() {
        val speech = FakeSpeechOutput(ready = true, speakResult = true)
        val haptic = FakeHapticOutput(vibrateResult = true)
        val controller = FeedbackController(
            speechOutput = speech,
            hapticOutput = haptic,
            clock = FeedbackClock { 1000L },
            initialSettings = FeedbackRuntimeSettings()
        )
        val applier = RuntimeConfigApplier(controller)

        applier.apply(
            runtimeConfig()
                .withSpeechEnabled(false)
                .withVibrationEnabled(false)
        )
        val decision = controller.notify(risk(RiskLevel.HIGH, ProximityBand.CRITICAL))

        assertFalse(decision.triggered)
        assertEquals(FeedbackReason.FEEDBACK_UNAVAILABLE, decision.reason)
        assertEquals(0, speech.speakCalls)
        assertEquals(0, haptic.vibrateCalls)
    }

    private fun AssistRuntimeTransition.hasRenderTarget(target: AssistRuntimeRenderTarget): Boolean {
        return effects.any { effect ->
            effect is AssistRuntimeEffect.Render && effect.target == target
        }
    }

    private fun runtimeConfig(): AssistRuntimeConfig {
        return AssistRuntimeConfig(
            detectionEnabled = true,
            speechEnabled = true,
            vibrationEnabled = true,
            careModeEnabled = false,
            alertProfile = AlertProfile.STANDARD,
            assistScenario = AssistScenario.GENERAL,
            speechStyle = SpeechStyle.STANDARD,
            vibrationStrength = VibrationStrength.STANDARD,
            appLanguage = AppLanguage.ZH,
            dailyUsageMode = DailyUsageMode.GENERAL_DAILY
        )
    }

    private fun risk(level: RiskLevel, proximity: ProximityBand): RiskResult {
        return RiskResult(
            level = level,
            direction = RiskDirection.CENTER,
            message = "runtime fault feedback",
            proximity = proximity,
            urgencyScore = proximity.ordinal.toFloat()
        )
    }

    private class FakeSpeechOutput(
        override val ready: Boolean,
        private val speakResult: Boolean
    ) : SpeechOutput {
        var speakCalls: Int = 0
            private set

        override fun setLanguage(language: AppLanguage) = Unit

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
