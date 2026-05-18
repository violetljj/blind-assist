package com.linnan.blindassist.ui

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.preferences.PreferenceStore
import com.linnan.blindassist.preferences.UserPreferences
import com.linnan.blindassist.ui.compose.CameraGuidanceUiState
import com.linnan.blindassist.ui.compose.FieldTestSummaryUiState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class BlindAssistViewModelTest {
    @Test
    fun defaultStateStartsWithEnabledAssistPreferencesAndOnboarding() {
        val viewModel = BlindAssistViewModel(UserPreferences(MapPreferenceStore()))

        val state = viewModel.uiState.value

        assertTrue(state.controls.detectionEnabled)
        assertTrue(state.controls.speechEnabled)
        assertTrue(state.controls.vibrationEnabled)
        assertFalse(state.controls.careModeEnabled)
        assertFalse(state.controls.debugVisible)
        assertEquals(AlertProfile.STANDARD, state.controls.alertProfile)
        assertTrue(state.showOnboarding)
        assertFalse(state.cameraActive)
        assertEquals("模型未初始化", state.modelStatus)
        assertEquals("等待相机会话", state.fieldTestSummary.statusText)
    }

    @Test
    fun savedPreferencesInitializeControlsAndOnboardingState() {
        val store = MapPreferenceStore()
        val preferences = UserPreferences(store)
        preferences.setSpeechEnabled(false)
        preferences.setVibrationEnabled(false)
        preferences.setCareModeEnabled(true)
        preferences.setAlertProfile(AlertProfile.SENSITIVE)
        preferences.setOnboardingCompleted(true)

        val state = BlindAssistViewModel(preferences).uiState.value

        assertFalse(state.controls.speechEnabled)
        assertFalse(state.controls.vibrationEnabled)
        assertTrue(state.controls.careModeEnabled)
        assertEquals(AlertProfile.SENSITIVE, state.controls.alertProfile)
        assertFalse(state.showOnboarding)
        assertTrue(state.fieldTestSummary.detailText.contains("当前档位：敏感"))
    }

    @Test
    fun settingChangesUpdateStateFlowAndPersistPreferences() {
        val store = MapPreferenceStore()
        val viewModel = BlindAssistViewModel(UserPreferences(store))

        viewModel.onSpeechChange(false)
        viewModel.onVibrationChange(false)
        viewModel.onCareModeChange(true)
        viewModel.onProfileChange(AlertProfile.QUIET)
        viewModel.onDebugVisibleChange(true)

        val state = viewModel.uiState.value
        assertFalse(state.controls.speechEnabled)
        assertFalse(state.controls.vibrationEnabled)
        assertTrue(state.controls.careModeEnabled)
        assertTrue(state.controls.debugVisible)
        assertEquals(AlertProfile.QUIET, state.controls.alertProfile)

        val reloaded = UserPreferences(store).load()
        assertFalse(reloaded.speechEnabled)
        assertFalse(reloaded.vibrationEnabled)
        assertTrue(reloaded.careModeEnabled)
        assertEquals(AlertProfile.QUIET, reloaded.alertProfile)
    }

    @Test
    fun onboardingCanBeCompletedAndShownAgain() {
        val store = MapPreferenceStore()
        val viewModel = BlindAssistViewModel(UserPreferences(store))

        viewModel.onCompleteOnboarding()

        assertFalse(viewModel.uiState.value.showOnboarding)
        assertTrue(UserPreferences(store).load().onboardingCompleted)

        viewModel.onShowOnboarding()

        assertTrue(viewModel.uiState.value.showOnboarding)
        assertTrue(UserPreferences(store).load().onboardingCompleted)
    }

    @Test
    fun dialogStateOnlyChangesDialogFlags() {
        val viewModel = BlindAssistViewModel(UserPreferences(MapPreferenceStore()))

        viewModel.onShowGlassesDialog()
        assertTrue(viewModel.uiState.value.showGlassesDialog)
        assertFalse(viewModel.uiState.value.cameraActive)
        viewModel.onDismissGlassesDialog()
        assertFalse(viewModel.uiState.value.showGlassesDialog)

        viewModel.onShowCameraPermissionDialog()
        assertTrue(viewModel.uiState.value.showCameraPermissionDialog)
        viewModel.onDismissCameraPermissionDialog()
        assertFalse(viewModel.uiState.value.showCameraPermissionDialog)

        viewModel.onCameraPermissionDenied(CameraGuidanceUiState.initial("denied"))
        assertFalse(viewModel.uiState.value.cameraActive)
        assertTrue(viewModel.uiState.value.showPermissionDeniedDialog)
        viewModel.onDismissPermissionDeniedDialog()
        assertFalse(viewModel.uiState.value.showPermissionDeniedDialog)
    }

    @Test
    fun cameraAndFieldSummaryStateCanBeUpdatedFromActivityBoundary() {
        val viewModel = BlindAssistViewModel(UserPreferences(MapPreferenceStore()))
        val summary = FieldTestSummaryUiState.empty(AlertProfile.SENSITIVE.displayName)
        val guidance = CameraGuidanceUiState.initial("ready")

        viewModel.activateCamera(summary, guidance, modelStatus = "ready")

        assertTrue(viewModel.uiState.value.cameraActive)
        assertEquals("ready", viewModel.uiState.value.modelStatus)
        assertEquals(summary, viewModel.uiState.value.fieldTestSummary)
        assertEquals(guidance, viewModel.uiState.value.cameraGuidance)

        viewModel.closeCamera(summary, guidance, modelStatus = "ready")

        assertFalse(viewModel.uiState.value.cameraActive)
    }

    @Test
    fun detectionStateIsNotPersistedAsAUserPreference() {
        val store = MapPreferenceStore()
        val viewModel = BlindAssistViewModel(UserPreferences(store))

        viewModel.onDetectionChange(false)

        assertFalse(viewModel.uiState.value.controls.detectionEnabled)
        assertFalse(store.keys().contains("detection_enabled"))
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
