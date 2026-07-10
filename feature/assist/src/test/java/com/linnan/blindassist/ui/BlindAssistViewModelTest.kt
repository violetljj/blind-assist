package com.linnan.blindassist.ui

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.model.AssistInputSource
import com.linnan.blindassist.model.ReplayScenario
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.preferences.PreferenceStore
import com.linnan.blindassist.preferences.UserPreferences
import com.linnan.blindassist.ui.compose.CameraGuidanceUiState
import com.linnan.blindassist.ui.compose.FieldTestSummaryUiState
import com.linnan.blindassist.ui.compose.GlassesConnectionState
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
        assertEquals(AssistScenario.GENERAL, state.controls.assistScenario)
        assertEquals(SpeechStyle.STANDARD, state.controls.speechStyle)
        assertEquals(VibrationStrength.STANDARD, state.controls.vibrationStrength)
        assertEquals(AppLanguage.ZH, state.controls.appLanguage)
        assertEquals(DailyUsageMode.GENERAL_DAILY, state.controls.dailyUsageMode)
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
        preferences.setAssistScenario(AssistScenario.CORRIDOR)
        preferences.setSpeechStyle(SpeechStyle.DETAILED)
        preferences.setVibrationStrength(VibrationStrength.STRONG)
        preferences.setAppLanguage(AppLanguage.EN)
        preferences.setOnboardingCompleted(true)

        val state = BlindAssistViewModel(preferences).uiState.value

        assertFalse(state.controls.speechEnabled)
        assertFalse(state.controls.vibrationEnabled)
        assertTrue(state.controls.careModeEnabled)
        assertEquals(AlertProfile.SENSITIVE, state.controls.alertProfile)
        assertEquals(AssistScenario.CORRIDOR, state.controls.assistScenario)
        assertEquals(SpeechStyle.DETAILED, state.controls.speechStyle)
        assertEquals(VibrationStrength.STRONG, state.controls.vibrationStrength)
        assertEquals(AppLanguage.EN, state.controls.appLanguage)
        assertEquals(DailyUsageMode.CUSTOM, state.controls.dailyUsageMode)
        assertFalse(state.showOnboarding)
        assertTrue(state.fieldTestSummary.detailText.contains("Current profile: Sensitive"))
        assertTrue(state.fieldTestSummary.detailText.contains("Current scenario: Corridor"))
    }

    @Test
    fun settingChangesUpdateStateFlowAndPersistPreferences() {
        val store = MapPreferenceStore()
        val viewModel = BlindAssistViewModel(UserPreferences(store))

        viewModel.onSpeechChange(false)
        viewModel.onVibrationChange(false)
        viewModel.onCareModeChange(true)
        viewModel.onProfileChange(AlertProfile.QUIET)
        viewModel.onScenarioChange(AssistScenario.OUTDOOR_SLOW)
        viewModel.onSpeechStyleChange(SpeechStyle.BRIEF)
        viewModel.onVibrationStrengthChange(VibrationStrength.SOFT)
        viewModel.onLanguageChange(AppLanguage.EN)
        viewModel.onDebugVisibleChange(true)

        val state = viewModel.uiState.value
        assertFalse(state.controls.speechEnabled)
        assertFalse(state.controls.vibrationEnabled)
        assertTrue(state.controls.careModeEnabled)
        assertTrue(state.controls.debugVisible)
        assertEquals(AlertProfile.QUIET, state.controls.alertProfile)
        assertEquals(AssistScenario.OUTDOOR_SLOW, state.controls.assistScenario)
        assertEquals(SpeechStyle.BRIEF, state.controls.speechStyle)
        assertEquals(VibrationStrength.SOFT, state.controls.vibrationStrength)
        assertEquals(AppLanguage.EN, state.controls.appLanguage)
        assertEquals(DailyUsageMode.CUSTOM, state.controls.dailyUsageMode)

        val reloaded = UserPreferences(store).load()
        assertFalse(reloaded.speechEnabled)
        assertFalse(reloaded.vibrationEnabled)
        assertTrue(reloaded.careModeEnabled)
        assertEquals(AlertProfile.QUIET, reloaded.alertProfile)
        assertEquals(AssistScenario.OUTDOOR_SLOW, reloaded.assistScenario)
        assertEquals(SpeechStyle.BRIEF, reloaded.speechStyle)
        assertEquals(VibrationStrength.SOFT, reloaded.vibrationStrength)
        assertEquals(AppLanguage.EN, reloaded.appLanguage)
    }

    @Test
    fun dailyUsageModeChangeUpdatesStateFlowAndPersistedPreferenceBundle() {
        val store = MapPreferenceStore()
        val viewModel = BlindAssistViewModel(UserPreferences(store))

        viewModel.onDailyUsageModeChange(DailyUsageMode.CORRIDOR)

        val state = viewModel.uiState.value.controls
        assertTrue(state.careModeEnabled)
        assertEquals(AlertProfile.SENSITIVE, state.alertProfile)
        assertEquals(AssistScenario.CORRIDOR, state.assistScenario)
        assertEquals(SpeechStyle.STANDARD, state.speechStyle)
        assertEquals(VibrationStrength.STANDARD, state.vibrationStrength)
        assertEquals(DailyUsageMode.CORRIDOR, state.dailyUsageMode)

        val reloaded = UserPreferences(store).load()
        assertTrue(reloaded.careModeEnabled)
        assertEquals(AlertProfile.SENSITIVE, reloaded.alertProfile)
        assertEquals(AssistScenario.CORRIDOR, reloaded.assistScenario)
        assertEquals(SpeechStyle.STANDARD, reloaded.speechStyle)
        assertEquals(VibrationStrength.STANDARD, reloaded.vibrationStrength)
    }

    @Test
    fun manuallyChangingOnePreferenceAfterDailyModeMarksStateAsCustom() {
        val viewModel = BlindAssistViewModel(UserPreferences(MapPreferenceStore()))

        viewModel.onDailyUsageModeChange(DailyUsageMode.CORRIDOR)
        viewModel.onProfileChange(AlertProfile.QUIET)

        assertEquals(DailyUsageMode.CUSTOM, viewModel.uiState.value.controls.dailyUsageMode)
    }

    @Test
    fun reminderShortcutKeepsScenarioButChangesIntensityBundle() {
        val store = MapPreferenceStore()
        val viewModel = BlindAssistViewModel(UserPreferences(store))

        viewModel.onDailyUsageModeChange(DailyUsageMode.OUTDOOR_SLOW)
        viewModel.onReminderShortcutChange(
            profile = AlertProfile.QUIET,
            speechStyle = SpeechStyle.BRIEF,
            vibrationStrength = VibrationStrength.SOFT
        )

        val state = viewModel.uiState.value.controls
        assertEquals(AssistScenario.OUTDOOR_SLOW, state.assistScenario)
        assertEquals(AlertProfile.QUIET, state.alertProfile)
        assertEquals(SpeechStyle.BRIEF, state.speechStyle)
        assertEquals(VibrationStrength.SOFT, state.vibrationStrength)
        assertEquals(DailyUsageMode.CUSTOM, state.dailyUsageMode)

        val reloaded = UserPreferences(store).load()
        assertEquals(AssistScenario.OUTDOOR_SLOW, reloaded.assistScenario)
        assertEquals(AlertProfile.QUIET, reloaded.alertProfile)
        assertEquals(SpeechStyle.BRIEF, reloaded.speechStyle)
        assertEquals(VibrationStrength.SOFT, reloaded.vibrationStrength)
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
    fun glassesCenterVisibilityAndPermissionDialogsAreIndependent() {
        val viewModel = BlindAssistViewModel(UserPreferences(MapPreferenceStore()))

        viewModel.onShowGlassesCenter()
        assertTrue(viewModel.uiState.value.showGlassesCenter)
        assertFalse(viewModel.uiState.value.cameraActive)
        viewModel.onDismissGlassesCenter()
        assertFalse(viewModel.uiState.value.showGlassesCenter)

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
    fun glassesSimulationConnectsTo82PercentAndSupportsLowBattery() {
        val viewModel = BlindAssistViewModel(UserPreferences(MapPreferenceStore()))

        viewModel.onSimulateGlassesConnection()
        assertEquals(GlassesConnectionState.CONNECTING, viewModel.uiState.value.glassesSimulator.connectionState)
        assertEquals(null, viewModel.uiState.value.glassesSimulator.batteryPercent)

        viewModel.onSimulatedGlassesConnectionCompleted()
        assertEquals(GlassesConnectionState.CONNECTED, viewModel.uiState.value.glassesSimulator.connectionState)
        assertEquals(82, viewModel.uiState.value.glassesSimulator.batteryPercent)

        viewModel.onSimulateGlassesLowBattery()
        assertEquals(15, viewModel.uiState.value.glassesSimulator.batteryPercent)
    }

    @Test
    fun staleConnectionCompletionCannotOverrideReset() {
        val viewModel = BlindAssistViewModel(UserPreferences(MapPreferenceStore()))

        viewModel.onSimulateGlassesConnection()
        viewModel.onResetGlassesSimulation()
        viewModel.onSimulatedGlassesConnectionCompleted()

        assertEquals(GlassesConnectionState.DISCONNECTED, viewModel.uiState.value.glassesSimulator.connectionState)
        assertEquals(null, viewModel.uiState.value.glassesSimulator.batteryPercent)
    }

    @Test
    fun simulatedDisconnectFallsBackToPhoneAndResetClearsTransientState() {
        val viewModel = BlindAssistViewModel(UserPreferences(MapPreferenceStore()))
        viewModel.onDebugReplayAvailabilityChanged(true)
        viewModel.onSimulateGlassesConnection()
        viewModel.onSimulatedGlassesConnectionCompleted()
        viewModel.onReplayScenarioSelected(ReplayScenario.MEDIUM_RIGHT)
        assertEquals(AssistInputSource.OFFLINE_REPLAY, viewModel.uiState.value.glassesSimulator.selectedInput)

        viewModel.onSimulateGlassesDisconnect()
        assertEquals(GlassesConnectionState.CONNECTION_LOST, viewModel.uiState.value.glassesSimulator.connectionState)
        assertEquals(AssistInputSource.PHONE_CAMERA, viewModel.uiState.value.glassesSimulator.selectedInput)
        assertEquals(null, viewModel.uiState.value.glassesSimulator.batteryPercent)

        viewModel.onResetGlassesSimulation()
        assertEquals(GlassesConnectionState.DISCONNECTED, viewModel.uiState.value.glassesSimulator.connectionState)
    }

    @Test
    fun replaySelectionRequiresDebugCapabilityAndConnectedSimulation() {
        val viewModel = BlindAssistViewModel(UserPreferences(MapPreferenceStore()))

        viewModel.onReplayScenarioSelected(ReplayScenario.LOW_CENTER)
        assertEquals(AssistInputSource.PHONE_CAMERA, viewModel.uiState.value.glassesSimulator.selectedInput)

        viewModel.onDebugReplayAvailabilityChanged(true)
        viewModel.onSimulateGlassesConnection()
        viewModel.onSimulatedGlassesConnectionCompleted()
        viewModel.onReplayScenarioSelected(ReplayScenario.LOW_CENTER)

        assertEquals(AssistInputSource.OFFLINE_REPLAY, viewModel.uiState.value.glassesSimulator.selectedInput)
        assertEquals(ReplayScenario.LOW_CENTER, viewModel.uiState.value.glassesSimulator.selectedReplayScenario)

        viewModel.onShowGlassesCenter()
        viewModel.onStartOfflineReplay()
        assertFalse(viewModel.uiState.value.showGlassesCenter)
    }

    @Test
    fun cameraAndFieldSummaryStateCanBeUpdatedFromActivityBoundary() {
        val viewModel = BlindAssistViewModel(UserPreferences(MapPreferenceStore()))
        val summary = FieldTestSummaryUiState.empty(AlertProfile.SENSITIVE.displayName, AssistScenario.CORRIDOR.displayName)
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

    @Test
    fun runtimeConfigReflectsCurrentControls() {
        val viewModel = BlindAssistViewModel(UserPreferences(MapPreferenceStore()))

        viewModel.onDailyUsageModeChange(DailyUsageMode.CORRIDOR)
        viewModel.onSpeechChange(false)
        viewModel.onLanguageChange(AppLanguage.EN)

        val config = viewModel.runtimeConfig()
        assertTrue(config.detectionEnabled)
        assertFalse(config.speechEnabled)
        assertTrue(config.careModeEnabled)
        assertEquals(AlertProfile.SENSITIVE, config.alertProfile)
        assertEquals(AssistScenario.CORRIDOR, config.assistScenario)
        assertEquals(AppLanguage.EN, config.appLanguage)
        assertEquals(DailyUsageMode.CORRIDOR, config.dailyUsageMode)
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
