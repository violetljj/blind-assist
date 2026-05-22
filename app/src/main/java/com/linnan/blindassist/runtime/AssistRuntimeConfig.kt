package com.linnan.blindassist.runtime

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.ui.compose.AssistControlsUiState

data class AssistRuntimeConfig(
    val detectionEnabled: Boolean,
    val speechEnabled: Boolean,
    val vibrationEnabled: Boolean,
    val careModeEnabled: Boolean,
    val alertProfile: AlertProfile,
    val assistScenario: AssistScenario,
    val speechStyle: SpeechStyle,
    val vibrationStrength: VibrationStrength,
    val appLanguage: AppLanguage,
    val dailyUsageMode: DailyUsageMode
) {
    fun withDetectionEnabled(enabled: Boolean): AssistRuntimeConfig = copy(detectionEnabled = enabled)

    fun withSpeechEnabled(enabled: Boolean): AssistRuntimeConfig = copy(speechEnabled = enabled)

    fun withVibrationEnabled(enabled: Boolean): AssistRuntimeConfig = copy(vibrationEnabled = enabled)

    fun withCareModeEnabled(enabled: Boolean): AssistRuntimeConfig {
        return copy(careModeEnabled = enabled).withDerivedDailyUsageMode()
    }

    fun withAlertProfile(profile: AlertProfile): AssistRuntimeConfig {
        return copy(alertProfile = profile).withDerivedDailyUsageMode()
    }

    fun withAssistScenario(scenario: AssistScenario): AssistRuntimeConfig {
        return copy(assistScenario = scenario).withDerivedDailyUsageMode()
    }

    fun withSpeechStyle(style: SpeechStyle): AssistRuntimeConfig {
        return copy(speechStyle = style).withDerivedDailyUsageMode()
    }

    fun withVibrationStrength(strength: VibrationStrength): AssistRuntimeConfig {
        return copy(vibrationStrength = strength).withDerivedDailyUsageMode()
    }

    fun withAppLanguage(language: AppLanguage): AssistRuntimeConfig = copy(appLanguage = language)

    fun withDailyUsageMode(mode: DailyUsageMode): AssistRuntimeConfig {
        val preset = mode.config ?: return this
        return copy(
            careModeEnabled = preset.careModeEnabled,
            alertProfile = preset.profile,
            assistScenario = preset.scenario,
            speechStyle = preset.speechStyle,
            vibrationStrength = preset.vibrationStrength,
            dailyUsageMode = mode
        )
    }

    fun withReminderShortcut(
        profile: AlertProfile,
        speechStyle: SpeechStyle,
        vibrationStrength: VibrationStrength
    ): AssistRuntimeConfig {
        return copy(
            alertProfile = profile,
            speechStyle = speechStyle,
            vibrationStrength = vibrationStrength
        ).withDerivedDailyUsageMode()
    }

    private fun withDerivedDailyUsageMode(): AssistRuntimeConfig {
        return copy(
            dailyUsageMode = DailyUsageMode.fromPreferences(
                scenario = assistScenario,
                profile = alertProfile,
                speechStyle = speechStyle,
                vibrationStrength = vibrationStrength,
                careModeEnabled = careModeEnabled
            )
        )
    }

    companion object {
        fun fromControls(controls: AssistControlsUiState): AssistRuntimeConfig {
            return AssistRuntimeConfig(
                detectionEnabled = controls.detectionEnabled,
                speechEnabled = controls.speechEnabled,
                vibrationEnabled = controls.vibrationEnabled,
                careModeEnabled = controls.careModeEnabled,
                alertProfile = controls.alertProfile,
                assistScenario = controls.assistScenario,
                speechStyle = controls.speechStyle,
                vibrationStrength = controls.vibrationStrength,
                appLanguage = controls.appLanguage,
                dailyUsageMode = controls.dailyUsageMode
            )
        }
    }
}
