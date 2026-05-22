package com.linnan.blindassist.preferences

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.localization.LocalizedText

data class DailyUsageConfig(
    val scenario: AssistScenario,
    val profile: AlertProfile,
    val speechStyle: SpeechStyle,
    val vibrationStrength: VibrationStrength,
    val careModeEnabled: Boolean
)

enum class DailyUsageMode(
    val storageValue: String,
    val config: DailyUsageConfig?
) {
    GENERAL_DAILY(
        "general_daily",
        DailyUsageConfig(
            scenario = AssistScenario.GENERAL,
            profile = AlertProfile.STANDARD,
            speechStyle = SpeechStyle.STANDARD,
            vibrationStrength = VibrationStrength.STANDARD,
            careModeEnabled = false
        )
    ),
    INDOOR_SLOW(
        "indoor_slow",
        DailyUsageConfig(
            scenario = AssistScenario.INDOOR,
            profile = AlertProfile.QUIET,
            speechStyle = SpeechStyle.BRIEF,
            vibrationStrength = VibrationStrength.SOFT,
            careModeEnabled = false
        )
    ),
    CORRIDOR(
        "corridor",
        DailyUsageConfig(
            scenario = AssistScenario.CORRIDOR,
            profile = AlertProfile.SENSITIVE,
            speechStyle = SpeechStyle.STANDARD,
            vibrationStrength = VibrationStrength.STANDARD,
            careModeEnabled = true
        )
    ),
    CROWDED(
        "crowded",
        DailyUsageConfig(
            scenario = AssistScenario.CROWDED,
            profile = AlertProfile.QUIET,
            speechStyle = SpeechStyle.BRIEF,
            vibrationStrength = VibrationStrength.SOFT,
            careModeEnabled = false
        )
    ),
    OUTDOOR_SLOW(
        "outdoor_slow",
        DailyUsageConfig(
            scenario = AssistScenario.OUTDOOR_SLOW,
            profile = AlertProfile.STANDARD,
            speechStyle = SpeechStyle.STANDARD,
            vibrationStrength = VibrationStrength.STRONG,
            careModeEnabled = true
        )
    ),
    CUSTOM("custom", null);

    fun displayName(language: AppLanguage): String {
        return LocalizedText.dailyUsageModeName(this, language)
    }

    fun description(language: AppLanguage): String {
        return LocalizedText.dailyUsageModeDescription(this, language)
    }

    fun accessibilitySummary(language: AppLanguage): String {
        return LocalizedText.dailyUsageModeAccessibility(this, language)
    }

    companion object {
        val selectableModes: List<DailyUsageMode> = listOf(
            GENERAL_DAILY,
            INDOOR_SLOW,
            CORRIDOR,
            CROWDED,
            OUTDOOR_SLOW
        )

        fun fromPreferences(
            scenario: AssistScenario,
            profile: AlertProfile,
            speechStyle: SpeechStyle,
            vibrationStrength: VibrationStrength,
            careModeEnabled: Boolean
        ): DailyUsageMode {
            return selectableModes.firstOrNull { mode ->
                val config = mode.config ?: return@firstOrNull false
                config.scenario == scenario &&
                    config.profile == profile &&
                    config.speechStyle == speechStyle &&
                    config.vibrationStrength == vibrationStrength &&
                    config.careModeEnabled == careModeEnabled
            } ?: CUSTOM
        }
    }
}
