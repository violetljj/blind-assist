package com.linnan.blindassist.preferences

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
    fun detectionStateIsNotPersistedAsAUserPreference() {
        val store = MapPreferenceStore()
        val preferences = UserPreferences(store)

        preferences.setSpeechEnabled(false)
        preferences.setVibrationEnabled(false)
        preferences.setCareModeEnabled(true)

        assertEquals(
            setOf(
                UserPreferences.KEY_SPEECH_ENABLED,
                UserPreferences.KEY_VIBRATION_ENABLED,
                UserPreferences.KEY_CARE_MODE_ENABLED
            ),
            store.keys()
        )
    }

    private class MapPreferenceStore : PreferenceStore {
        private val values = mutableMapOf<String, Boolean>()

        override fun getBoolean(key: String, defaultValue: Boolean): Boolean {
            return values[key] ?: defaultValue
        }

        override fun putBoolean(key: String, value: Boolean) {
            values[key] = value
        }

        fun keys(): Set<String> = values.keys
    }
}
