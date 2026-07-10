package com.linnan.blindassist

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.linnan.blindassist.runtime.AssistRuntimeController
import com.linnan.blindassist.runtime.AssistRuntimeControllerFactory
import com.linnan.blindassist.ui.BlindAssistViewModel
import com.linnan.blindassist.ui.compose.BlindAssistApp
import com.linnan.blindassist.ui.compose.BlindAssistTheme
import com.linnan.blindassist.ui.compose.CameraPermissionDeniedDialog
import com.linnan.blindassist.ui.compose.CameraPermissionExplanationDialog
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    private val appViewModel: BlindAssistViewModel by viewModels()

    @Inject lateinit var runtimeControllerFactory: AssistRuntimeControllerFactory
    private lateinit var runtimeController: AssistRuntimeController

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        runtimeController.onCameraPermissionResult(granted)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)

        runtimeController = runtimeControllerFactory.create(this, appViewModel).also { it.initialize() }
        appViewModel.onDebugReplayAvailabilityChanged(BuildConfig.DEBUG)

        setContent {
            val uiState by appViewModel.uiState.collectAsStateWithLifecycle()
            BlindAssistTheme {
                BlindAssistApp(
                    controls = uiState.controls,
                    cameraGuidance = uiState.cameraGuidance,
                    fieldTestSummary = uiState.fieldTestSummary,
                    modelStatus = uiState.modelStatus,
                    appVersion = BuildConfig.VERSION_NAME,
                    cameraActive = uiState.cameraActive,
                    activeInputSource = uiState.activeInputSource,
                    activeReplayScenario = uiState.activeReplayScenario,
                    showOnboarding = uiState.showOnboarding,
                    showGlassesCenter = uiState.showGlassesCenter,
                    glassesSimulator = uiState.glassesSimulator,
                    onOpenCamera = runtimeController::openCameraExperience,
                    onCloseCamera = runtimeController::closeCameraExperience,
                    onCompleteOnboarding = appViewModel::onCompleteOnboarding,
                    onShowOnboarding = appViewModel::onShowOnboarding,
                    onShowGlassesCenter = appViewModel::onShowGlassesCenter,
                    onDismissGlassesCenter = appViewModel::onDismissGlassesCenter,
                    onSimulateGlassesConnection = appViewModel::onSimulateGlassesConnection,
                    onSimulatedGlassesConnectionCompleted = appViewModel::onSimulatedGlassesConnectionCompleted,
                    onSimulateGlassesLowBattery = appViewModel::onSimulateGlassesLowBattery,
                    onSimulateGlassesDisconnect = appViewModel::onSimulateGlassesDisconnect,
                    onResetGlassesSimulation = appViewModel::onResetGlassesSimulation,
                    onReplayScenarioSelected = appViewModel::onReplayScenarioSelected,
                    onStartOfflineReplay = { scenario ->
                        appViewModel.onStartOfflineReplay()
                        runtimeController.openOfflineReplay(scenario)
                    },
                    onDetectionChange = runtimeController::setDetectionEnabled,
                    onSpeechChange = runtimeController::setSpeechEnabled,
                    onVibrationChange = runtimeController::setVibrationEnabled,
                    onCareModeChange = runtimeController::setCareModeEnabled,
                    onDebugVisibleChange = appViewModel::onDebugVisibleChange,
                    onProfileChange = runtimeController::setAlertProfile,
                    onScenarioChange = runtimeController::setAssistScenario,
                    onSpeechStyleChange = runtimeController::setSpeechStyle,
                    onVibrationStrengthChange = runtimeController::setVibrationStrength,
                    onDailyUsageModeChange = runtimeController::setDailyUsageMode,
                    onQuietShortcut = runtimeController::setQuietShortcut,
                    onSensitiveShortcut = runtimeController::setSensitiveShortcut,
                    onLanguageChange = runtimeController::setAppLanguage,
                    onCameraViewsReady = runtimeController::onCameraViewsReady
                )
                if (uiState.showCameraPermissionDialog) {
                    CameraPermissionExplanationDialog(
                        language = uiState.controls.appLanguage,
                        onContinue = {
                            runtimeController.requestCameraPermissionAfterExplanation {
                                requestCameraPermission.launch(Manifest.permission.CAMERA)
                            }
                        },
                        onDismiss = runtimeController::dismissCameraPermissionFlow
                    )
                }
                if (uiState.showPermissionDeniedDialog) {
                    CameraPermissionDeniedDialog(
                        language = uiState.controls.appLanguage,
                        onDismiss = runtimeController::dismissCameraPermissionFlow
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        runtimeController.shutdown()
    }
}
