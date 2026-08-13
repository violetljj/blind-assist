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
        assertEquals(
            DailyUsageConfig(
                scenario = AssistScenario.GENERAL,
                profile = AlertProfile.STANDARD,
                speechStyle = SpeechStyle.STANDARD,
                vibrationStrength = VibrationStrength.STANDARD,
                careModeEnabled = false
            ),
            DailyUsageMode.GENERAL_DAILY.config
        )
        assertEquals(
            DailyUsageConfig(
                scenario = AssistScenario.INDOOR,
                profile = AlertProfile.QUIET,
                speechStyle = SpeechStyle.BRIEF,
                vibrationStrength = VibrationStrength.SOFT,
                careModeEnabled = false
            ),
            DailyUsageMode.INDOOR_SLOW.config
        )
        assertEquals(
            DailyUsageConfig(
                scenario = AssistScenario.CORRIDOR,
                profile = AlertProfile.SENSITIVE,
                speechStyle = SpeechStyle.STANDARD,
                vibrationStrength = VibrationStrength.STANDARD,
                careModeEnabled = true
            ),
            DailyUsageMode.CORRIDOR.config
        )
        assertEquals(
            DailyUsageConfig(
                scenario = AssistScenario.CROWDED,
                profile = AlertProfile.QUIET,
                speechStyle = SpeechStyle.BRIEF,
                vibrationStrength = VibrationStrength.SOFT,
                careModeEnabled = false
            ),
            DailyUsageMode.CROWDED.config
        )
        assertEquals(
            DailyUsageConfig(
                scenario = AssistScenario.OUTDOOR_SLOW,
                profile = AlertProfile.STANDARD,
                speechStyle = SpeechStyle.STANDARD,
                vibrationStrength = VibrationStrength.STRONG,
                careModeEnabled = true
            ),
            DailyUsageMode.OUTDOOR_SLOW.config
        )
    }

    @Test
    fun currentPreferencesCanBeMappedBackToPresetMode() {
        val mode = DailyUsageMode.fromPreferences(
            scenario = AssistScenario.CORRIDOR,
            profile = AlertProfile.SENSITIVE,
            speechStyle = SpeechStyle.STANDARD,
            vibrationStrength = VibrationStrength.STANDARD,
            careModeEnabled = true
        )

        assertEquals(DailyUsageMode.CORRIDOR, mode)
    }

    @Test
    fun everySelectablePresetRoundTripsThroughPreferences() {
        DailyUsageMode.selectableModes.forEach { mode ->
            val config = requireNotNull(mode.config)
            assertEquals(
                mode,
                DailyUsageMode.fromPreferences(
                    scenario = config.scenario,
                    profile = config.profile,
                    speechStyle = config.speechStyle,
                    vibrationStrength = config.vibrationStrength,
                    careModeEnabled = config.careModeEnabled
                )
            )
        }

        assertFalse(DailyUsageMode.selectableModes.contains(DailyUsageMode.CUSTOM))
        assertEquals(
            DailyUsageMode.selectableModes.size,
            DailyUsageMode.selectableModes.map(DailyUsageMode::storageValue).distinct().size
        )
        assertFalse(DailyUsageMode.selectableModes.any { it.storageValue.isBlank() })
    }

    @Test
    fun manuallyAdjustedPreferencesMapToCustomMode() {
        val mode = DailyUsageMode.fromPreferences(
            scenario = AssistScenario.CORRIDOR,
            profile = AlertProfile.QUIET,
            speechStyle = SpeechStyle.STANDARD,
            vibrationStrength = VibrationStrength.STANDARD,
            careModeEnabled = true
        )

        assertEquals(DailyUsageMode.CUSTOM, mode)
    }
}
