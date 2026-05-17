package com.linnan.blindassist.preferences

import android.content.Context
import android.content.SharedPreferences
import com.linnan.blindassist.alert.AlertProfile

data class UserPreferenceState(
    val speechEnabled: Boolean = true,
    val vibrationEnabled: Boolean = true,
    val careModeEnabled: Boolean = false,
    val alertProfile: AlertProfile = AlertProfile.STANDARD
)

class UserPreferences(private val store: PreferenceStore) {
    constructor(context: Context) : this(
        SharedPreferencesStore(
            context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        )
    )

    fun load(): UserPreferenceState {
        return UserPreferenceState(
            speechEnabled = store.getBoolean(KEY_SPEECH_ENABLED, true),
            vibrationEnabled = store.getBoolean(KEY_VIBRATION_ENABLED, true),
            careModeEnabled = store.getBoolean(KEY_CARE_MODE_ENABLED, false),
            alertProfile = AlertProfile.fromStorageValue(store.getString(KEY_ALERT_PROFILE, AlertProfile.STANDARD.storageValue))
        )
    }

    fun setSpeechEnabled(enabled: Boolean) {
        store.putBoolean(KEY_SPEECH_ENABLED, enabled)
    }

    fun setVibrationEnabled(enabled: Boolean) {
        store.putBoolean(KEY_VIBRATION_ENABLED, enabled)
    }

    fun setCareModeEnabled(enabled: Boolean) {
        store.putBoolean(KEY_CARE_MODE_ENABLED, enabled)
    }

    fun setAlertProfile(profile: AlertProfile) {
        store.putString(KEY_ALERT_PROFILE, profile.storageValue)
    }

    companion object {
        private const val PREFS_NAME = "blindassist_user_preferences"
        internal const val KEY_SPEECH_ENABLED = "speech_enabled"
        internal const val KEY_VIBRATION_ENABLED = "vibration_enabled"
        internal const val KEY_CARE_MODE_ENABLED = "care_mode_enabled"
        internal const val KEY_ALERT_PROFILE = "alert_profile"
    }
}

interface PreferenceStore {
    fun getBoolean(key: String, defaultValue: Boolean): Boolean
    fun putBoolean(key: String, value: Boolean)
    fun getString(key: String, defaultValue: String): String
    fun putString(key: String, value: String)
}

private class SharedPreferencesStore(
    private val preferences: SharedPreferences
) : PreferenceStore {
    override fun getBoolean(key: String, defaultValue: Boolean): Boolean {
        return preferences.getBoolean(key, defaultValue)
    }

    override fun putBoolean(key: String, value: Boolean) {
        preferences.edit().putBoolean(key, value).apply()
    }

    override fun getString(key: String, defaultValue: String): String {
        return preferences.getString(key, defaultValue) ?: defaultValue
    }

    override fun putString(key: String, value: String) {
        preferences.edit().putString(key, value).apply()
    }
}
