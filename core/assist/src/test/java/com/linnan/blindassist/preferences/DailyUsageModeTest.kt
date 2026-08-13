package com.linnan.blindassist.preferences

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class DailyUsageModeTest {
    @Test
    fun presetModesMapToExpectedPreferenceBundles() {
        assertEquals(DailyUsageConfig(AssistScenario.GENERAL, AlertProfile.STANDARD, SpeechStyle.STANDARD, VibrationStrength.STANDARD, false), DailyUsageMode.GENERAL_DAILY.config)
        assertEquals(DailyUsageConfig(AssistScenario.INDOOR, AlertProfile.QUIET, SpeechStyle.BRIEF, VibrationStrength.SOFT, false), DailyUsageMode.INDOOR_SLOW.config)
        assertEquals(DailyUsageConfig(AssistScenario.CORRIDOR, AlertProfile.SENSITIVE, SpeechStyle.STANDARD, VibrationStrength.STANDARD, true), DailyUsageMode.CORRIDOR.config)
        assertEquals(DailyUsageConfig(AssistScenario.CROWDED, AlertProfile.QUIET, SpeechStyle.BRIEF, VibrationStrength.SOFT, false), DailyUsageMode.CROWDED.config)
        assertEquals(DailyUsageConfig(AssistScenario.OUTDOOR_SLOW, AlertProfile.STANDARD, SpeechStyle.STANDARD, VibrationStrength.STRONG, true), DailyUsageMode.OUTDOOR_SLOW.config)
    }

    @Test
    fun currentPreferencesCanBeMappedBackToPresetMode() {
        val mode = DailyUsageMode.fromPreferences(AssistScenario.CORRIDOR, AlertProfile.SENSITIVE, SpeechStyle.STANDARD, VibrationStrength.STANDARD, true)
        assertEquals(DailyUsageMode.CORRIDOR, mode)
    }

    @Test
    fun everySelectablePresetRoundTripsThroughPreferences() {
        DailyUsageMode.selectableModes.forEach { mode ->
            val config = requireNotNull(mode.config)
            assertEquals(mode, DailyUsageMode.fromPreferences(config.scenario, config.profile, config.speechStyle, config.vibrationStrength, config.careModeEnabled))
        }

        assertFalse(DailyUsageMode.selectableModes.contains(DailyUsageMode.CUSTOM))
        assertEquals(DailyUsageMode.selectableModes.size, DailyUsageMode.selectableModes.map(DailyUsageMode::storageValue).distinct().size)
        assertFalse(DailyUsageMode.selectableModes.any { it.storageValue.isBlank() })
    }

    @Test
    fun manuallyAdjustedPreferencesMapToCustomMode() {
        val mode = DailyUsageMode.fromPreferences(AssistScenario.CORRIDOR, AlertProfile.QUIET, SpeechStyle.STANDARD, VibrationStrength.STANDARD, true)
        assertEquals(DailyUsageMode.CUSTOM, mode)
    }
}