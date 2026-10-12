package com.linnan.blindassist.feedback

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.localization.AppLanguage
import org.junit.Assert.*
import org.junit.Test

class HardwareFeedbackSettingsTest {
    @Test fun profilesChangeOnlySteadyReminderCadence() {
        for ((profile, period) in listOf(AlertProfile.QUIET to 20_000L,
            AlertProfile.STANDARD to 12_000L, AlertProfile.SENSITIVE to 6_000L)) {
            val settings = HardwareFeedbackSettings(profile = profile)
            assertEquals(period, settings.reminderIntervalMs)
            val policy = HardwareDemoFeedbackPolicy(settings.reminderIntervalMs)
            assertEquals(HardwareDemoFeedbackEvent.OBSTACLE, policy.update(0, true, true, true, 1500))
            assertEquals(HardwareDemoFeedbackEvent.CLOSER, policy.update(1000, true, true, true, 1000))
            assertNull(policy.update(period, true, true, true, 1000))
            assertEquals(HardwareDemoFeedbackEvent.REMINDER, policy.update(period + 1000, true, true, true, 1000))
            assertEquals(HardwareDemoFeedbackEvent.CONNECTION_LOST, policy.update(period + 1001, true, false, false, null))
        }
    }

    @Test fun languageAndStyleChangeOutputWithoutInventingCategoryOrDistance() {
        val standard = HardwareFeedbackSettings(language = AppLanguage.EN)
        assertEquals("Possible obstacle ahead", standard.text(HardwareDemoFeedbackEvent.OBSTACLE))
        assertEquals("Distance decreasing", standard.copy(style = SpeechStyle.BRIEF).text(HardwareDemoFeedbackEvent.CLOSER))
        assertEquals("Possible obstacle ahead. Check your surroundings",
            standard.copy(style = SpeechStyle.DETAILED).text(HardwareDemoFeedbackEvent.OBSTACLE))
        assertEquals(standard.text(HardwareDemoFeedbackEvent.CONNECTION_LOST),
            standard.copy(style = SpeechStyle.BRIEF).text(HardwareDemoFeedbackEvent.CONNECTION_LOST))
    }

    @Test fun vibrationStrengthScalesPulsesAndLeavesPausesSilent() {
        val settings = HardwareFeedbackSettings(strength = VibrationStrength.SOFT)
        val pattern = settings.vibrationPattern(HardwareDemoFeedbackEvent.CONNECTION_LOST)
        assertArrayEquals(longArrayOf(0, 187, 120, 187), pattern)
        assertArrayEquals(intArrayOf(0, 96, 0, 96), settings.vibrationAmplitudes(pattern))
        val strong = settings.copy(strength = VibrationStrength.STRONG)
        assertArrayEquals(longArrayOf(0, 225), strong.vibrationPattern(HardwareDemoFeedbackEvent.OBSTACLE))
    }
}
