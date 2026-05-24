package com.linnan.blindassist.runtime

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.ui.BlindAssistViewModel

internal class AssistRuntimeSettingsController(
    private val appViewModel: BlindAssistViewModel,
    private val configSnapshot: AssistRuntimeConfigSnapshot,
    private val configApplier: RuntimeConfigApplier,
    private val renderer: AssistRuntimeRenderer,
    private val currentState: () -> AssistRuntimeState
) {
    fun setDetectionEnabled(enabled: Boolean) {
        appViewModel.onDetectionChange(enabled)
        syncConfigFromViewModel()
    }

    fun setSpeechEnabled(enabled: Boolean) {
        appViewModel.onSpeechChange(enabled)
        onConfigChanged(updateGuidance = false)
    }

    fun setVibrationEnabled(enabled: Boolean) {
        appViewModel.onVibrationChange(enabled)
        onConfigChanged(updateGuidance = false)
    }

    fun setSpeechStyle(style: SpeechStyle) {
        appViewModel.onSpeechStyleChange(style)
        onConfigChanged(updateGuidance = false)
    }

    fun setVibrationStrength(strength: VibrationStrength) {
        appViewModel.onVibrationStrengthChange(strength)
        onConfigChanged(updateGuidance = false)
    }

    fun setAppLanguage(language: AppLanguage) {
        appViewModel.onLanguageChange(language)
        onConfigChanged(updateGuidance = true)
    }

    fun setCareModeEnabled(enabled: Boolean) {
        appViewModel.onCareModeChange(enabled)
        onConfigChanged(updateGuidance = false)
    }

    fun setAlertProfile(profile: AlertProfile) {
        appViewModel.onProfileChange(profile)
        onConfigChanged(updateGuidance = false)
    }

    fun setAssistScenario(scenario: AssistScenario) {
        appViewModel.onScenarioChange(scenario)
        onConfigChanged(updateGuidance = true)
    }

    fun setDailyUsageMode(mode: DailyUsageMode) {
        appViewModel.onDailyUsageModeChange(mode)
        onConfigChanged(updateGuidance = true)
    }

    fun setReminderShortcut(
        profile: AlertProfile,
        speechStyle: SpeechStyle,
        vibrationStrength: VibrationStrength
    ) {
        appViewModel.onReminderShortcutChange(profile, speechStyle, vibrationStrength)
        onConfigChanged(updateGuidance = false)
    }

    fun syncConfigFromViewModel(): AssistRuntimeConfig {
        val latestConfig = configSnapshot.update(appViewModel.runtimeConfig())
        configApplier.apply(latestConfig, renderer.currentOverlay())
        return latestConfig
    }

    private fun onConfigChanged(updateGuidance: Boolean) {
        val runtimeConfig = syncConfigFromViewModel()
        renderer.updateFieldTestSummary(appViewModel.uiState.value.cameraActive, runtimeConfig)
        if (updateGuidance) {
            renderer.renderState(currentState())
        }
    }
}
