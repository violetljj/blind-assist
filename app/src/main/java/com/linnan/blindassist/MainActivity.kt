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
import com.linnan.blindassist.runtime.AssistRuntimeControllerFactory
import com.linnan.blindassist.runtime.AssistRuntimeIntent
import com.linnan.blindassist.runtime.AssistRuntimeMode
import com.linnan.blindassist.runtime.AssistSession
import com.linnan.blindassist.ui.BlindAssistViewModel
import com.linnan.blindassist.ui.compose.BlindAssistApp
import com.linnan.blindassist.ui.compose.BlindAssistAppActions
import com.linnan.blindassist.ui.compose.BlindAssistAppState
import com.linnan.blindassist.ui.compose.BlindAssistTheme
import com.linnan.blindassist.ui.compose.AssistNavigationActions
import com.linnan.blindassist.ui.compose.AssistRuntimeUiActions
import com.linnan.blindassist.ui.compose.CameraPermissionDeniedDialog
import com.linnan.blindassist.ui.compose.CameraPermissionExplanationDialog
import com.linnan.blindassist.ui.compose.GlassesSimulatorActions
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    private val appViewModel: BlindAssistViewModel by viewModels()

    @Inject lateinit var runtimeControllerFactory: AssistRuntimeControllerFactory
    private lateinit var assistSession: AssistSession

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        assistSession.dispatch(AssistRuntimeIntent.CameraPermissionResult(granted))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)

        check(
            listOf(
                BuildConfig.USTRF_EXPERIMENT,
                BuildConfig.DUAL_LOOP_SHADOW,
                BuildConfig.DUAL_LOOP_ACTIVE
            ).count { it } <= 1
        ) {
            "isolated experimental modes cannot be enabled together"
        }
        val runtimeMode = when {
            BuildConfig.USTRF_EXPERIMENT -> AssistRuntimeMode.USTRF_EXPERIMENT
            BuildConfig.DUAL_LOOP_ACTIVE -> AssistRuntimeMode.DUAL_LOOP_ACTIVE
            BuildConfig.DUAL_LOOP_SHADOW -> AssistRuntimeMode.DUAL_LOOP_SHADOW
            else -> AssistRuntimeMode.BASELINE
        }
        assistSession = runtimeControllerFactory.create(this, appViewModel, runtimeMode).also { it.initialize() }
        appViewModel.onDebugReplayAvailabilityChanged(BuildConfig.DEBUG)

        setContent {
            val uiState by appViewModel.uiState.collectAsStateWithLifecycle()
            BlindAssistTheme {
                BlindAssistApp(
                    state = BlindAssistAppState(
                        controls = uiState.controls,
                        cameraGuidance = uiState.cameraGuidance,
                        fieldTestSummary = uiState.fieldTestSummary,
                        modelStatus = uiState.modelStatus,
                        appVersion = BuildConfig.VERSION_NAME,
                        editionLabel = when {
                            BuildConfig.NPU_CANDIDATE ->
                                "QNN HTP NPU 候选版 · 验证中 · 不可用于独立行走"
                            BuildConfig.USTRF_EXPERIMENT ->
                                "USTRF二维路线代理实验版 · 不可用于独立行走"
                            BuildConfig.DUAL_LOOP_SHADOW ->
                                "神经—几何双环影子接线版 · 不改变提醒 · 不可用于独立行走"
                            BuildConfig.DUAL_LOOP_ACTIVE ->
                                "神经—几何双环隔离纠错版 · 开发验证中 · 不可用于独立行走"
                            else -> null
                        },
                        cameraActive = uiState.cameraActive,
                        activeInputSource = uiState.activeInputSource,
                        activeReplayScenario = uiState.activeReplayScenario,
                        showOnboarding = uiState.showOnboarding,
                        showGlassesCenter = uiState.showGlassesCenter,
                        glassesSimulator = uiState.glassesSimulator
                    ),
                    actions = BlindAssistAppActions(
                        runtime = AssistRuntimeUiActions(
                            onOpenCamera = { assistSession.dispatch(AssistRuntimeIntent.OpenPhoneCamera) },
                            onCloseCamera = { assistSession.dispatch(AssistRuntimeIntent.CloseCamera) },
                            onStartOfflineReplay = { scenario ->
                                appViewModel.onStartOfflineReplay()
                                assistSession.dispatch(AssistRuntimeIntent.OpenOfflineReplay(scenario))
                            },
                            onDetectionChange = { assistSession.dispatch(AssistRuntimeIntent.DetectionEnabled(it)) },
                            onSpeechChange = { assistSession.dispatch(AssistRuntimeIntent.SpeechEnabled(it)) },
                            onVibrationChange = { assistSession.dispatch(AssistRuntimeIntent.VibrationEnabled(it)) },
                            onCareModeChange = { assistSession.dispatch(AssistRuntimeIntent.CareModeEnabled(it)) },
                            onDebugVisibleChange = appViewModel::onDebugVisibleChange,
                            onProfileChange = { assistSession.dispatch(AssistRuntimeIntent.AlertProfileChanged(it)) },
                            onScenarioChange = { assistSession.dispatch(AssistRuntimeIntent.AssistScenarioChanged(it)) },
                            onSpeechStyleChange = { assistSession.dispatch(AssistRuntimeIntent.SpeechStyleChanged(it)) },
                            onVibrationStrengthChange = { assistSession.dispatch(AssistRuntimeIntent.VibrationStrengthChanged(it)) },
                            onDailyUsageModeChange = { assistSession.dispatch(AssistRuntimeIntent.DailyUsageModeChanged(it)) },
                            onQuietShortcut = { assistSession.dispatch(AssistRuntimeIntent.QuietShortcutSelected) },
                            onSensitiveShortcut = { assistSession.dispatch(AssistRuntimeIntent.SensitiveShortcutSelected) },
                            onLanguageChange = { assistSession.dispatch(AssistRuntimeIntent.AppLanguageChanged(it)) },
                            onCameraViewsReady = { preview, externalPreview, overlay ->
                                assistSession.dispatch(
                                    AssistRuntimeIntent.CameraViewsReady(preview, externalPreview, overlay)
                                )
                            }
                        ),
                        navigation = AssistNavigationActions(
                            onCompleteOnboarding = appViewModel::onCompleteOnboarding,
                            onShowOnboarding = appViewModel::onShowOnboarding,
                            onShowGlassesCenter = appViewModel::onShowGlassesCenter,
                            onDismissGlassesCenter = appViewModel::onDismissGlassesCenter
                        ),
                        glasses = GlassesSimulatorActions(
                            onConnect = appViewModel::onConnectGlassesDevice,
                            onDisconnect = appViewModel::onDisconnectGlassesDevice,
                            onStartLiveAssist = { endpoint ->
                                appViewModel.onStartGlassesHardware()
                                assistSession.dispatch(AssistRuntimeIntent.OpenGlassesHardware(endpoint))
                            },
                            onReplayScenarioSelected = appViewModel::onReplayScenarioSelected
                        )
                    )
                )
                if (uiState.showCameraPermissionDialog) {
                    CameraPermissionExplanationDialog(
                        language = uiState.controls.appLanguage,
                        onContinue = {
                            assistSession.dispatch(
                                AssistRuntimeIntent.PermissionExplanationAccepted {
                                    requestCameraPermission.launch(Manifest.permission.CAMERA)
                                }
                            )
                        },
                        onDismiss = { assistSession.dispatch(AssistRuntimeIntent.DismissPermissionFlow) }
                    )
                }
                if (uiState.showPermissionDeniedDialog) {
                    CameraPermissionDeniedDialog(
                        language = uiState.controls.appLanguage,
                        onDismiss = { assistSession.dispatch(AssistRuntimeIntent.DismissPermissionFlow) }
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        assistSession.shutdown()
    }
}
