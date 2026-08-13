package com.linnan.blindassist.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.linnan.blindassist.device.glasses.GlassesConnectionRepository
import com.linnan.blindassist.runtime.AssistRuntimeConfig
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.model.AssistInputSource
import com.linnan.blindassist.model.ReplayScenario
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.preferences.UserPreferences
import com.linnan.blindassist.ui.compose.AssistControlsUiState
import com.linnan.blindassist.ui.compose.CameraGuidanceUiState
import com.linnan.blindassist.ui.compose.FieldTestSummaryUiState
import com.linnan.blindassist.ui.compose.GlassesConnectionState
import com.linnan.blindassist.ui.compose.GlassesSimulatorUiState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Inject

data class BlindAssistAppUiState(
    val controls: AssistControlsUiState,
    val cameraGuidance: CameraGuidanceUiState,
    val fieldTestSummary: FieldTestSummaryUiState,
    val modelStatus: String,
    val cameraActive: Boolean,
    val activeInputSource: AssistInputSource,
    val activeReplayScenario: ReplayScenario?,
    val showOnboarding: Boolean,
    val showGlassesCenter: Boolean,
    val glassesSimulator: GlassesSimulatorUiState,
    val showCameraPermissionDialog: Boolean,
    val showPermissionDeniedDialog: Boolean
)

@HiltViewModel
class BlindAssistViewModel @Inject constructor(
    private val userPreferences: UserPreferences,
    private val glassesConnectionRepository: GlassesConnectionRepository = GlassesConnectionRepository()
) : ViewModel() {
    private val initialPreferences = userPreferences.load()

    private val _uiState = MutableStateFlow(
        BlindAssistAppUiState(
            controls = AssistControlsUiState(
                detectionEnabled = true,
                speechEnabled = initialPreferences.speechEnabled,
                vibrationEnabled = initialPreferences.vibrationEnabled,
                careModeEnabled = initialPreferences.careModeEnabled,
                debugVisible = false,
                alertProfile = initialPreferences.alertProfile,
                assistScenario = initialPreferences.assistScenario,
                speechStyle = initialPreferences.speechStyle,
                vibrationStrength = initialPreferences.vibrationStrength,
                appLanguage = initialPreferences.appLanguage,
                dailyUsageMode = DailyUsageMode.fromPreferences(
                    scenario = initialPreferences.assistScenario,
                    profile = initialPreferences.alertProfile,
                    speechStyle = initialPreferences.speechStyle,
                    vibrationStrength = initialPreferences.vibrationStrength,
                    careModeEnabled = initialPreferences.careModeEnabled
                )
            ),
            cameraGuidance = CameraGuidanceUiState.initial(
                DEFAULT_MODEL_STATUS,
                initialPreferences.assistScenario.displayName(initialPreferences.appLanguage),
                initialPreferences.appLanguage
            ),
            fieldTestSummary = FieldTestSummaryUiState.empty(
                initialPreferences.alertProfile.displayName(initialPreferences.appLanguage),
                initialPreferences.assistScenario.displayName(initialPreferences.appLanguage),
                initialPreferences.appLanguage
            ),
            modelStatus = DEFAULT_MODEL_STATUS,
            cameraActive = false,
            activeInputSource = AssistInputSource.PHONE_CAMERA,
            activeReplayScenario = null,
            showOnboarding = !initialPreferences.onboardingCompleted,
            showGlassesCenter = false,
            glassesSimulator = GlassesSimulatorUiState(),
            showCameraPermissionDialog = false,
            showPermissionDeniedDialog = false
        )
    )
    val uiState: StateFlow<BlindAssistAppUiState> = _uiState.asStateFlow()

    fun runtimeConfig(): AssistRuntimeConfig {
        return AssistRuntimeConfig.fromControls(uiState.value.controls)
    }

    fun renderCameraGuidance(guidance: CameraGuidanceUiState, modelStatus: String = uiState.value.modelStatus) {
        _uiState.update {
            it.copy(cameraGuidance = guidance, modelStatus = modelStatus)
        }
    }

    fun renderFrame(
        guidance: CameraGuidanceUiState,
        fieldTestSummary: FieldTestSummaryUiState?,
        modelStatus: String
    ) {
        _uiState.update { current ->
            val nextSummary = fieldTestSummary ?: current.fieldTestSummary
            if (
                current.cameraGuidance == guidance &&
                current.fieldTestSummary == nextSummary &&
                current.modelStatus == modelStatus
            ) {
                current
            } else {
                current.copy(
                    cameraGuidance = guidance,
                    fieldTestSummary = nextSummary,
                    modelStatus = modelStatus
                )
            }
        }
    }

    fun activateCamera(fieldTestSummary: FieldTestSummaryUiState, guidance: CameraGuidanceUiState, modelStatus: String) {
        _uiState.update {
            it.copy(
                cameraActive = true,
                fieldTestSummary = fieldTestSummary,
                cameraGuidance = guidance,
                modelStatus = modelStatus
            )
        }
    }

    fun onAssistInputSourceActivated(source: AssistInputSource, replayScenario: ReplayScenario?) {
        _uiState.update {
            it.copy(
                activeInputSource = source,
                activeReplayScenario = if (source == AssistInputSource.OFFLINE_REPLAY) replayScenario else null
            )
        }
    }

    fun closeCamera(fieldTestSummary: FieldTestSummaryUiState, guidance: CameraGuidanceUiState, modelStatus: String) {
        _uiState.update {
            it.copy(
                cameraActive = false,
                showCameraPermissionDialog = false,
                fieldTestSummary = fieldTestSummary,
                cameraGuidance = guidance,
                modelStatus = modelStatus
            )
        }
    }

    fun onCameraPermissionDenied(guidance: CameraGuidanceUiState) {
        _uiState.update {
            it.copy(
                cameraActive = false,
                showPermissionDeniedDialog = true,
                cameraGuidance = guidance
            )
        }
    }

    fun updateFieldTestSummary(fieldTestSummary: FieldTestSummaryUiState) {
        _uiState.update { it.copy(fieldTestSummary = fieldTestSummary) }
    }

    fun onDetectionChange(enabled: Boolean) {
        _uiState.update {
            it.copy(controls = it.controls.copy(detectionEnabled = enabled))
        }
    }

    fun onSpeechChange(enabled: Boolean) {
        userPreferences.setSpeechEnabled(enabled)
        _uiState.update {
            it.copy(controls = it.controls.copy(speechEnabled = enabled))
        }
    }

    fun onVibrationChange(enabled: Boolean) {
        userPreferences.setVibrationEnabled(enabled)
        _uiState.update {
            it.copy(controls = it.controls.copy(vibrationEnabled = enabled))
        }
    }

    fun onCareModeChange(enabled: Boolean) {
        userPreferences.setCareModeEnabled(enabled)
        _uiState.update {
            it.copy(controls = it.controls.copy(careModeEnabled = enabled).withDerivedDailyUsageMode())
        }
    }

    fun onDebugVisibleChange(visible: Boolean) {
        _uiState.update {
            it.copy(controls = it.controls.copy(debugVisible = visible))
        }
    }

    fun onProfileChange(profile: AlertProfile) {
        userPreferences.setAlertProfile(profile)
        _uiState.update {
            it.copy(controls = it.controls.copy(alertProfile = profile).withDerivedDailyUsageMode())
        }
    }

    fun onScenarioChange(scenario: AssistScenario) {
        userPreferences.setAssistScenario(scenario)
        _uiState.update {
            it.copy(controls = it.controls.copy(assistScenario = scenario).withDerivedDailyUsageMode())
        }
    }

    fun onSpeechStyleChange(style: SpeechStyle) {
        userPreferences.setSpeechStyle(style)
        _uiState.update {
            it.copy(controls = it.controls.copy(speechStyle = style).withDerivedDailyUsageMode())
        }
    }

    fun onVibrationStrengthChange(strength: VibrationStrength) {
        userPreferences.setVibrationStrength(strength)
        _uiState.update {
            it.copy(controls = it.controls.copy(vibrationStrength = strength).withDerivedDailyUsageMode())
        }
    }

    fun onLanguageChange(language: AppLanguage) {
        userPreferences.setAppLanguage(language)
        _uiState.update {
            it.copy(controls = it.controls.copy(appLanguage = language))
        }
    }

    fun onDailyUsageModeChange(mode: DailyUsageMode) {
        val config = mode.config ?: return
        userPreferences.setAssistScenario(config.scenario)
        userPreferences.setAlertProfile(config.profile)
        userPreferences.setSpeechStyle(config.speechStyle)
        userPreferences.setVibrationStrength(config.vibrationStrength)
        userPreferences.setCareModeEnabled(config.careModeEnabled)
        _uiState.update {
            it.copy(
                controls = it.controls.copy(
                    assistScenario = config.scenario,
                    alertProfile = config.profile,
                    speechStyle = config.speechStyle,
                    vibrationStrength = config.vibrationStrength,
                    careModeEnabled = config.careModeEnabled,
                    dailyUsageMode = mode
                )
            )
        }
    }

    fun onReminderShortcutChange(
        profile: AlertProfile,
        speechStyle: SpeechStyle,
        vibrationStrength: VibrationStrength
    ) {
        userPreferences.setAlertProfile(profile)
        userPreferences.setSpeechStyle(speechStyle)
        userPreferences.setVibrationStrength(vibrationStrength)
        _uiState.update {
            it.copy(
                controls = it.controls.copy(
                    alertProfile = profile,
                    speechStyle = speechStyle,
                    vibrationStrength = vibrationStrength
                ).withDerivedDailyUsageMode()
            )
        }
    }

    fun onCompleteOnboarding() {
        userPreferences.setOnboardingCompleted(true)
        _uiState.update { it.copy(showOnboarding = false) }
    }

    fun onShowOnboarding() {
        _uiState.update { it.copy(showOnboarding = true) }
    }

    fun onShowGlassesCenter() {
        _uiState.update { it.copy(showGlassesCenter = true) }
    }

    fun onDismissGlassesCenter() {
        _uiState.update { it.copy(showGlassesCenter = false) }
    }

    fun onDebugReplayAvailabilityChanged(available: Boolean) {
        _uiState.update { state ->
            val simulator = state.glassesSimulator
            state.copy(
                glassesSimulator = simulator.copy(
                    debugReplayAvailable = available,
                    selectedInput = if (available) simulator.selectedInput else AssistInputSource.PHONE_CAMERA
                )
            )
        }
    }

    fun onConnectGlassesDevice() {
        _uiState.update { state ->
            state.copy(
                glassesSimulator = state.glassesSimulator.copy(
                    connectionState = GlassesConnectionState.CONNECTING,
                    errorMessage = null,
                    selectedInput = AssistInputSource.PHONE_CAMERA
                )
            )
        }
        val endpoint = uiState.value.glassesSimulator.endpoint
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                glassesConnectionRepository.connect(endpoint)
            }
            _uiState.update { state ->
                if (state.glassesSimulator.connectionState != GlassesConnectionState.CONNECTING) {
                    state
                } else {
                    result.fold(
                        onSuccess = { status ->
                            state.copy(
                                glassesSimulator = state.glassesSimulator.copy(
                                    connectionState = GlassesConnectionState.CONNECTED,
                                    firmwareVersion = status.firmwareVersion,
                                    wifiRssiDbm = status.wifiRssiDbm,
                                    tofValid = status.tofValid,
                                    tofRangeMm = status.tofRangeMm,
                                    streamReachable = status.streamReachable,
                                    errorMessage = null
                                )
                            )
                        },
                        onFailure = { error ->
                            state.copy(
                                glassesSimulator = state.glassesSimulator.copy(
                                    connectionState = GlassesConnectionState.CONNECTION_LOST,
                                    firmwareVersion = null,
                                    wifiRssiDbm = null,
                                    tofValid = false,
                                    tofRangeMm = null,
                                    streamReachable = false,
                                    errorMessage = error.message ?: error.javaClass.simpleName
                                )
                            )
                        }
                    )
                }
            }
        }
    }

    fun onDisconnectGlassesDevice() {
        _uiState.update { state ->
            state.copy(
                glassesSimulator = state.glassesSimulator.copy(
                    connectionState = GlassesConnectionState.DISCONNECTED,
                    firmwareVersion = null,
                    wifiRssiDbm = null,
                    tofValid = false,
                    tofRangeMm = null,
                    streamReachable = false,
                    errorMessage = null,
                    selectedInput = AssistInputSource.PHONE_CAMERA
                )
            )
        }
    }

    @Deprecated("Use onConnectGlassesDevice for real hardware")
    fun onSimulateGlassesConnection() {
        _uiState.update { state ->
            state.copy(
                glassesSimulator = state.glassesSimulator.copy(
                    connectionState = GlassesConnectionState.CONNECTING,
                    batteryPercent = null
                )
            )
        }
    }

    @Deprecated("Compatibility helper for legacy tests")
    fun onSimulatedGlassesConnectionCompleted() {
        _uiState.update { state ->
            if (state.glassesSimulator.connectionState != GlassesConnectionState.CONNECTING) state else {
                state.copy(
                    glassesSimulator = state.glassesSimulator.copy(
                        connectionState = GlassesConnectionState.CONNECTED,
                        batteryPercent = 82
                    )
                )
            }
        }
    }

    @Deprecated("Compatibility helper for legacy tests")
    fun onSimulateGlassesLowBattery() {
        _uiState.update { state ->
            if (state.glassesSimulator.connectionState != GlassesConnectionState.CONNECTED) state else {
                state.copy(glassesSimulator = state.glassesSimulator.copy(batteryPercent = 15))
            }
        }
    }

    @Deprecated("Use onDisconnectGlassesDevice")
    fun onSimulateGlassesDisconnect() {
        _uiState.update { state ->
            state.copy(
                glassesSimulator = state.glassesSimulator.copy(
                    connectionState = GlassesConnectionState.CONNECTION_LOST,
                    batteryPercent = null,
                    selectedInput = AssistInputSource.PHONE_CAMERA
                )
            )
        }
    }

    @Deprecated("Use onDisconnectGlassesDevice")
    fun onResetGlassesSimulation() = onDisconnectGlassesDevice()

    fun onReplayScenarioSelected(scenario: ReplayScenario) {
        _uiState.update { state ->
            val simulator = state.glassesSimulator
            if (!simulator.debugReplayAvailable || simulator.connectionState != GlassesConnectionState.CONNECTED) {
                state
            } else {
                state.copy(
                    glassesSimulator = simulator.copy(
                        selectedInput = AssistInputSource.OFFLINE_REPLAY,
                        selectedReplayScenario = scenario
                    )
                )
            }
        }
    }

    fun onStartOfflineReplay() {
        _uiState.update { state ->
            if (
                state.glassesSimulator.debugReplayAvailable &&
                state.glassesSimulator.connectionState == GlassesConnectionState.CONNECTED
            ) {
                state.copy(
                    showGlassesCenter = false,
                    glassesSimulator = state.glassesSimulator.copy(
                        selectedInput = AssistInputSource.OFFLINE_REPLAY
                    )
                )
            } else {
                state
            }
        }
    }

    fun onStartGlassesHardware() {
        _uiState.update { state ->
            if (
                state.glassesSimulator.connectionState == GlassesConnectionState.CONNECTED &&
                state.glassesSimulator.streamReachable
            ) {
                state.copy(
                    showGlassesCenter = false,
                    glassesSimulator = state.glassesSimulator.copy(
                        selectedInput = AssistInputSource.GLASSES_HARDWARE
                    )
                )
            } else state
        }
    }

    fun onShowCameraPermissionDialog() {
        _uiState.update { it.copy(showCameraPermissionDialog = true) }
    }

    fun onDismissCameraPermissionDialog() {
        _uiState.update { it.copy(showCameraPermissionDialog = false) }
    }

    fun onDismissPermissionDeniedDialog() {
        _uiState.update { it.copy(showPermissionDeniedDialog = false) }
    }

    companion object {
        private const val DEFAULT_MODEL_STATUS = "模型未初始化"
    }
}

private fun AssistControlsUiState.withDerivedDailyUsageMode(): AssistControlsUiState {
    return copy(
        dailyUsageMode = DailyUsageMode.fromPreferences(
            scenario = assistScenario,
            profile = alertProfile,
            speechStyle = speechStyle,
            vibrationStrength = vibrationStrength,
            careModeEnabled = careModeEnabled
        )
    )
}
