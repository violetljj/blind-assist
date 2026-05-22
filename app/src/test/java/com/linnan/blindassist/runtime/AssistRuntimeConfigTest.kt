package com.linnan.blindassist.runtime

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.ui.compose.AssistControlsUiState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AssistRuntimeConfigTest {
    @Test
    fun fromControlsCopiesCurrentRuntimeRelevantState() {
        val config = AssistRuntimeConfig.fromControls(
            AssistControlsUiState(
                detectionEnabled = false,
                speechEnabled = false,
                vibrationEnabled = true,
                careModeEnabled = true,
                debugVisible = true,
                alertProfile = AlertProfile.SENSITIVE,
                assistScenario = AssistScenario.CORRIDOR,
                speechStyle = SpeechStyle.DETAILED,
                vibrationStrength = VibrationStrength.STRONG,
                appLanguage = AppLanguage.EN,
                dailyUsageMode = DailyUsageMode.CUSTOM
            )
        )

        assertFalse(config.detectionEnabled)
        assertFalse(config.speechEnabled)
        assertTrue(config.vibrationEnabled)
        assertTrue(config.careModeEnabled)
        assertEquals(AlertProfile.SENSITIVE, config.alertProfile)
        assertEquals(AssistScenario.CORRIDOR, config.assistScenario)
        assertEquals(SpeechStyle.DETAILED, config.speechStyle)
        assertEquals(VibrationStrength.STRONG, config.vibrationStrength)
        assertEquals(AppLanguage.EN, config.appLanguage)
        assertEquals(DailyUsageMode.CUSTOM, config.dailyUsageMode)
    }

    @Test
    fun dailyUsageModeAppliesPresetBundle() {
        val config = baseConfig().withDailyUsageMode(DailyUsageMode.CORRIDOR)

        assertTrue(config.careModeEnabled)
        assertEquals(AlertProfile.SENSITIVE, config.alertProfile)
        assertEquals(AssistScenario.CORRIDOR, config.assistScenario)
        assertEquals(SpeechStyle.STANDARD, config.speechStyle)
        assertEquals(VibrationStrength.STANDARD, config.vibrationStrength)
        assertEquals(DailyUsageMode.CORRIDOR, config.dailyUsageMode)
    }

    @Test
    fun manualPreferenceChangeRecomputesCustomDailyMode() {
        val config = baseConfig()
            .withDailyUsageMode(DailyUsageMode.CORRIDOR)
            .withAlertProfile(AlertProfile.QUIET)

        assertEquals(AssistScenario.CORRIDOR, config.assistScenario)
        assertEquals(AlertProfile.QUIET, config.alertProfile)
        assertEquals(DailyUsageMode.CUSTOM, config.dailyUsageMode)
    }

    @Test
    fun reminderShortcutKeepsScenarioButRecomputesDailyMode() {
        val config = baseConfig()
            .withDailyUsageMode(DailyUsageMode.OUTDOOR_SLOW)
            .withReminderShortcut(
                profile = AlertProfile.QUIET,
                speechStyle = SpeechStyle.BRIEF,
                vibrationStrength = VibrationStrength.SOFT
            )

        assertEquals(AssistScenario.OUTDOOR_SLOW, config.assistScenario)
        assertEquals(AlertProfile.QUIET, config.alertProfile)
        assertEquals(SpeechStyle.BRIEF, config.speechStyle)
        assertEquals(VibrationStrength.SOFT, config.vibrationStrength)
        assertEquals(DailyUsageMode.CUSTOM, config.dailyUsageMode)
    }

    @Test
    fun languageAndFeedbackTogglesDoNotChangePresetMatch() {
        val config = baseConfig()
            .withDailyUsageMode(DailyUsageMode.INDOOR_SLOW)
            .withAppLanguage(AppLanguage.EN)
            .withSpeechEnabled(false)
            .withVibrationEnabled(false)

        assertEquals(AppLanguage.EN, config.appLanguage)
        assertFalse(config.speechEnabled)
        assertFalse(config.vibrationEnabled)
        assertEquals(DailyUsageMode.INDOOR_SLOW, config.dailyUsageMode)
    }

    private fun baseConfig(): AssistRuntimeConfig {
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
}
