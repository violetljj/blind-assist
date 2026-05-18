package com.linnan.blindassist.preferences

import android.content.Context
import android.content.SharedPreferences
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength

data class UserPreferenceState(
    val speechEnabled: Boolean = true,
    val vibrationEnabled: Boolean = true,
    val careModeEnabled: Boolean = false,
    val alertProfile: AlertProfile = AlertProfile.STANDARD,
    val speechStyle: SpeechStyle = SpeechStyle.STANDARD,
    val vibrationStrength: VibrationStrength = VibrationStrength.STANDARD,
    val onboardingCompleted: Boolean = false
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
            alertProfile = AlertProfile.fromStorageValue(store.getString(KEY_ALERT_PROFILE, AlertProfile.STANDARD.storageValue)),
            speechStyle = SpeechStyle.fromStorageValue(store.getString(KEY_SPEECH_STYLE, SpeechStyle.STANDARD.storageValue)),
            vibrationStrength = VibrationStrength.fromStorageValue(
                store.getString(KEY_VIBRATION_STRENGTH, VibrationStrength.STANDARD.storageValue)
            ),
            onboardingCompleted = store.getBoolean(KEY_ONBOARDING_COMPLETED, false)
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

    fun setSpeechStyle(style: SpeechStyle) {
        store.putString(KEY_SPEECH_STYLE, style.storageValue)
    }

    fun setVibrationStrength(strength: VibrationStrength) {
        store.putString(KEY_VIBRATION_STRENGTH, strength.storageValue)
    }

    fun setOnboardingCompleted(completed: Boolean) {
        store.putBoolean(KEY_ONBOARDING_COMPLETED, completed)
    }

    companion object {
        private const val PREFS_NAME = "blindassist_user_preferences"
        internal const val KEY_SPEECH_ENABLED = "speech_enabled"
        internal const val KEY_VIBRATION_ENABLED = "vibration_enabled"
        internal const val KEY_CARE_MODE_ENABLED = "care_mode_enabled"
        internal const val KEY_ALERT_PROFILE = "alert_profile"
        internal const val KEY_SPEECH_STYLE = "speech_style"
        internal const val KEY_VIBRATION_STRENGTH = "vibration_strength"
        internal const val KEY_ONBOARDING_COMPLETED = "onboarding_completed"
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
