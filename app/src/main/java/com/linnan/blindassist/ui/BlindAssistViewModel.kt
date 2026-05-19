package com.linnan.blindassist.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.CreationExtras
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.preferences.UserPreferences
import com.linnan.blindassist.ui.compose.AssistControlsUiState
import com.linnan.blindassist.ui.compose.CameraGuidanceUiState
import com.linnan.blindassist.ui.compose.FieldTestSummaryUiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class BlindAssistAppUiState(
    val controls: AssistControlsUiState,
    val cameraGuidance: CameraGuidanceUiState,
    val fieldTestSummary: FieldTestSummaryUiState,
    val modelStatus: String,
    val cameraActive: Boolean,
    val showOnboarding: Boolean,
    val showGlassesDialog: Boolean,
    val showCameraPermissionDialog: Boolean,
    val showPermissionDeniedDialog: Boolean
)

class BlindAssistViewModel(
    private val userPreferences: UserPreferences,
    initialModelStatus: String = DEFAULT_MODEL_STATUS
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
                initialModelStatus,
                initialPreferences.assistScenario.displayName(initialPreferences.appLanguage),
                initialPreferences.appLanguage
            ),
            fieldTestSummary = FieldTestSummaryUiState.empty(
                initialPreferences.alertProfile.displayName(initialPreferences.appLanguage),
                initialPreferences.assistScenario.displayName(initialPreferences.appLanguage),
                initialPreferences.appLanguage
            ),
            modelStatus = initialModelStatus,
            cameraActive = false,
            showOnboarding = !initialPreferences.onboardingCompleted,
            showGlassesDialog = false,
            showCameraPermissionDialog = false,
            showPermissionDeniedDialog = false
        )
    )
    val uiState: StateFlow<BlindAssistAppUiState> = _uiState.asStateFlow()

    fun renderCameraGuidance(guidance: CameraGuidanceUiState, modelStatus: String = uiState.value.modelStatus) {
        _uiState.update {
            it.copy(cameraGuidance = guidance, modelStatus = modelStatus)
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

    fun onShowGlassesDialog() {
        _uiState.update { it.copy(showGlassesDialog = true) }
    }

    fun onDismissGlassesDialog() {
        _uiState.update { it.copy(showGlassesDialog = false) }
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

    class Factory(
        private val userPreferences: UserPreferences
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
            if (modelClass.isAssignableFrom(BlindAssistViewModel::class.java)) {
                return BlindAssistViewModel(userPreferences) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
        }
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
