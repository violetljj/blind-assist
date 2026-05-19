package com.linnan.blindassist.preferences

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class UserPreferencesTest {
    @Test
    fun defaultPreferencesEnableSpeechAndVibrationButDisableCareMode() {
        val preferences = UserPreferences(MapPreferenceStore())

        val state = preferences.load()

        assertTrue(state.speechEnabled)
        assertTrue(state.vibrationEnabled)
        assertFalse(state.careModeEnabled)
        assertEquals(AlertProfile.STANDARD, state.alertProfile)
        assertEquals(AssistScenario.GENERAL, state.assistScenario)
        assertEquals(SpeechStyle.STANDARD, state.speechStyle)
        assertEquals(VibrationStrength.STANDARD, state.vibrationStrength)
        assertEquals(AppLanguage.ZH, state.appLanguage)
        assertFalse(state.onboardingCompleted)
    }

    @Test
    fun savedSpeechPreferenceIsLoadedAgain() {
        val store = MapPreferenceStore()
        val preferences = UserPreferences(store)

        preferences.setSpeechEnabled(false)

        assertFalse(UserPreferences(store).load().speechEnabled)
    }

    @Test
    fun savedVibrationPreferenceIsLoadedAgain() {
        val store = MapPreferenceStore()
        val preferences = UserPreferences(store)

        preferences.setVibrationEnabled(false)

        assertFalse(UserPreferences(store).load().vibrationEnabled)
    }

    @Test
    fun savedCareModePreferenceIsLoadedAgain() {
        val store = MapPreferenceStore()
        val preferences = UserPreferences(store)

        preferences.setCareModeEnabled(true)

        assertTrue(UserPreferences(store).load().careModeEnabled)
    }

    @Test
    fun savedAlertProfilePreferenceIsLoadedAgain() {
        val store = MapPreferenceStore()
        val preferences = UserPreferences(store)

        preferences.setAlertProfile(AlertProfile.SENSITIVE)

        assertEquals(AlertProfile.SENSITIVE, UserPreferences(store).load().alertProfile)
    }

    @Test
    fun savedAssistScenarioPreferenceIsLoadedAgain() {
        val store = MapPreferenceStore()
        val preferences = UserPreferences(store)

        preferences.setAssistScenario(AssistScenario.CORRIDOR)

        assertEquals(AssistScenario.CORRIDOR, UserPreferences(store).load().assistScenario)
    }

    @Test
    fun savedSpeechStylePreferenceIsLoadedAgain() {
        val store = MapPreferenceStore()
        val preferences = UserPreferences(store)

        preferences.setSpeechStyle(SpeechStyle.DETAILED)

        assertEquals(SpeechStyle.DETAILED, UserPreferences(store).load().speechStyle)
    }

    @Test
    fun savedVibrationStrengthPreferenceIsLoadedAgain() {
        val store = MapPreferenceStore()
        val preferences = UserPreferences(store)

        preferences.setVibrationStrength(VibrationStrength.STRONG)

        assertEquals(VibrationStrength.STRONG, UserPreferences(store).load().vibrationStrength)
    }

    @Test
    fun savedAppLanguagePreferenceIsLoadedAgain() {
        val store = MapPreferenceStore()
        val preferences = UserPreferences(store)

        preferences.setAppLanguage(AppLanguage.EN)

        assertEquals(AppLanguage.EN, UserPreferences(store).load().appLanguage)
    }

    @Test
    fun savedOnboardingCompletedPreferenceIsLoadedAgain() {
        val store = MapPreferenceStore()
        val preferences = UserPreferences(store)

        preferences.setOnboardingCompleted(true)

        assertTrue(UserPreferences(store).load().onboardingCompleted)
    }

    @Test
    fun unknownAlertProfileFallsBackToStandard() {
        val store = MapPreferenceStore()
        store.putString(UserPreferences.KEY_ALERT_PROFILE, "future-mode")

        assertEquals(AlertProfile.STANDARD, UserPreferences(store).load().alertProfile)
    }

    @Test
    fun unknownFeedbackPreferencesFallBackToStandard() {
        val store = MapPreferenceStore()
        store.putString(UserPreferences.KEY_ASSIST_SCENARIO, "future-scenario")
        store.putString(UserPreferences.KEY_SPEECH_STYLE, "future-style")
        store.putString(UserPreferences.KEY_VIBRATION_STRENGTH, "future-strength")
        store.putString(UserPreferences.KEY_APP_LANGUAGE, "future-language")

        val state = UserPreferences(store).load()

        assertEquals(AssistScenario.GENERAL, state.assistScenario)
        assertEquals(SpeechStyle.STANDARD, state.speechStyle)
        assertEquals(VibrationStrength.STANDARD, state.vibrationStrength)
        assertEquals(AppLanguage.ZH, state.appLanguage)
    }

    @Test
    fun detectionStateIsNotPersistedAsAUserPreference() {
        val store = MapPreferenceStore()
        val preferences = UserPreferences(store)

        preferences.setSpeechEnabled(false)
        preferences.setVibrationEnabled(false)
        preferences.setCareModeEnabled(true)
        preferences.setAlertProfile(AlertProfile.QUIET)
        preferences.setAssistScenario(AssistScenario.CROWDED)
        preferences.setSpeechStyle(SpeechStyle.BRIEF)
        preferences.setVibrationStrength(VibrationStrength.SOFT)
        preferences.setAppLanguage(AppLanguage.EN)
        preferences.setOnboardingCompleted(true)

        assertEquals(
            setOf(
                UserPreferences.KEY_SPEECH_ENABLED,
                UserPreferences.KEY_VIBRATION_ENABLED,
                UserPreferences.KEY_CARE_MODE_ENABLED,
                UserPreferences.KEY_ALERT_PROFILE,
                UserPreferences.KEY_ASSIST_SCENARIO,
                UserPreferences.KEY_SPEECH_STYLE,
                UserPreferences.KEY_VIBRATION_STRENGTH,
                UserPreferences.KEY_APP_LANGUAGE,
                UserPreferences.KEY_ONBOARDING_COMPLETED
            ),
            store.keys()
        )
    }

    private class MapPreferenceStore : PreferenceStore {
        private val values = mutableMapOf<String, Boolean>()
        private val stringValues = mutableMapOf<String, String>()

        override fun getBoolean(key: String, defaultValue: Boolean): Boolean {
            return values[key] ?: defaultValue
        }

        override fun putBoolean(key: String, value: Boolean) {
            values[key] = value
        }

        override fun getString(key: String, defaultValue: String): String {
            return stringValues[key] ?: defaultValue
        }

        override fun putString(key: String, value: String) {
            stringValues[key] = value
        }

        fun keys(): Set<String> = values.keys + stringValues.keys
    }
}
