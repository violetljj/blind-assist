package com.linnan.blindassist.preferences

import com.linnan.blindassist.alert.AlertProfile
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
    fun detectionStateIsNotPersistedAsAUserPreference() {
        val store = MapPreferenceStore()
        val preferences = UserPreferences(store)

        preferences.setSpeechEnabled(false)
        preferences.setVibrationEnabled(false)
        preferences.setCareModeEnabled(true)
        preferences.setAlertProfile(AlertProfile.QUIET)
        preferences.setOnboardingCompleted(true)

        assertEquals(
            setOf(
                UserPreferences.KEY_SPEECH_ENABLED,
                UserPreferences.KEY_VIBRATION_ENABLED,
                UserPreferences.KEY_CARE_MODE_ENABLED,
                UserPreferences.KEY_ALERT_PROFILE,
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
