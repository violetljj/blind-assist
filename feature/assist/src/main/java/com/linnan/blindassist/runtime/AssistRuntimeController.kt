package com.linnan.blindassist.runtime

import android.Manifest
import android.os.SystemClock
import android.util.Log
import android.content.pm.PackageManager
import androidx.activity.ComponentActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.camera.FrameSource
import com.linnan.blindassist.camera.FrameSourceFactory
import com.linnan.blindassist.feedback.FeedbackController
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.model.AssistInputSource
import com.linnan.blindassist.model.ReplayScenario
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.session.AssistSessionCoordinator
import com.linnan.blindassist.ui.BlindAssistViewModel
import com.linnan.blindassist.ui.DetectionOverlayView
import com.linnan.blindassist.vision.ObjectDetector

internal class AssistRuntimeSession(
    private val activity: ComponentActivity,
    private val appViewModel: BlindAssistViewModel,
    private val detector: ObjectDetector,
    private val feedbackController: FeedbackController,
    private val coordinator: AssistSessionCoordinator,
    private val frameSourceFactory: FrameSourceFactory,
    private val configApplier: RuntimeConfigApplier,
    private val mode: AssistRuntimeMode = AssistRuntimeMode.BASELINE,
    private val stateMachine: AssistRuntimeStateMachine = AssistRuntimeStateMachine()
) : AssistSession {
    private var currentInputSource: AssistInputSource = AssistInputSource.PHONE_CAMERA
    private var currentReplayScenario: ReplayScenario? = null
    private val initialFrameSource: FrameSource = frameSourceFactory.create(
        source = currentInputSource,
        context = activity,
        lifecycleOwner = activity
    )
    private val configSnapshot = AssistRuntimeConfigSnapshot(appViewModel.runtimeConfig())
    private val config: AssistRuntimeConfig get() = configSnapshot.get()
    private val lifecycleGate = AssistRuntimeLifecycleGate()
    private val decisionClockNs: () -> Long = SystemClock::elapsedRealtimeNanos
    private val guidanceFactory = AssistRuntimeGuidanceFactory(detector) { config }
    private val fieldTestSummaryProvider = FieldTestSummaryProvider(coordinator)
    private val framePipelineStats = FramePipelineStats()
    private val renderer = AssistRuntimeRenderer(
        appViewModel = appViewModel,
        detector = detector,
        guidanceFactory = guidanceFactory,
        fieldTestSummaryProvider = fieldTestSummaryProvider,
        mode = mode
    )
    private val frameProcessor = AssistFrameProcessor(
        detector = detector,
        coordinator = coordinator,
        configSnapshot = configSnapshot,
        renderer = renderer,
        stats = framePipelineStats,
        lifecycleGate = lifecycleGate,
        isCameraActive = { appViewModel.uiState.value.cameraActive },
        runOnUiThread = { block -> activity.runOnUiThread(Runnable(block)) },
        onCameraFailure = ::handleCameraFailure,
        decisionClockNs = decisionClockNs,
        onDualLoopShadowObservation = { observation ->
            Log.i(
                DUAL_LOOP_LOG_TAG,
                "frame=${observation.currentFrameId} " +
                    "track=${observation.trackEpoch} " +
                    "disposition=${observation.disposition} " +
                    "decision=${observation.correctionDecision} " +
                    "rate=${observation.signedApproachRatePerS} " +
                    "quality=${observation.quality} " +
                    "reason=${observation.sourceAbstentionReason}"
            )
        },
        mode = mode
    )
    private val cameraLifecycleAdapter = AssistCameraLifecycleAdapter(
        initialFrameSource = initialFrameSource,
        renderer = renderer,
        isCameraActive = { appViewModel.uiState.value.cameraActive },
        requiresCameraPermission = { currentInputSource == AssistInputSource.PHONE_CAMERA },
        hasCameraPermission = ::hasCameraPermission,
        onCameraStarted = {
            handleTransition(stateMachine.onEvent(AssistRuntimeEvent.CameraStarted(detector.isReady)))
        },
        onCameraFailure = ::handleCameraFailure,
        runOnUiThread = { block -> activity.runOnUiThread(Runnable(block)) }
    )
    private val settingsController = AssistRuntimeSettingsController(
        appViewModel = appViewModel,
        configSnapshot = configSnapshot,
        configApplier = configApplier,
        renderer = renderer,
        currentState = { stateMachine.currentState }
    )
    private val effectExecutor = AssistRuntimeEffectExecutor(
        appViewModel = appViewModel,
        coordinator = coordinator,
        renderer = renderer,
        cameraLifecycleAdapter = cameraLifecycleAdapter,
        frameProcessor = frameProcessor,
        lifecycleGate = lifecycleGate,
        configSnapshot = configSnapshot,
        syncConfigFromViewModel = { settingsController.syncConfigFromViewModel() },
        startCameraIfReady = ::startCameraIfReady,
        decisionClockNs = decisionClockNs
    )

    override fun initialize() {
        settingsController.syncConfigFromViewModel()
        renderer.renderInitial()
    }

    override fun shutdown() {
        cameraLifecycleAdapter.stop()
        lifecycleGate.shutdown(
            resetState = {
                coordinator.reset()
                frameProcessor.resetSessionStats()
            },
            onIdle = {
                activity.runOnUiThread {
                    cameraLifecycleAdapter.shutdown()
                    detector.close()
                    feedbackController.shutdown()
                }
            }
        )
    }

    override fun dispatch(intent: AssistRuntimeIntent) {
        when (intent) {
            AssistRuntimeIntent.OpenPhoneCamera -> openCameraExperience()
            is AssistRuntimeIntent.OpenOfflineReplay -> openOfflineReplay(intent.scenario)
            AssistRuntimeIntent.CloseCamera -> closeCameraExperience()
            is AssistRuntimeIntent.PermissionExplanationAccepted -> requestCameraPermissionAfterExplanation(intent.launchPermissionRequest)
            AssistRuntimeIntent.DismissPermissionFlow -> dismissCameraPermissionFlow()
            is AssistRuntimeIntent.CameraPermissionResult -> onCameraPermissionResult(intent.granted)
            is AssistRuntimeIntent.CameraViewsReady -> onCameraViewsReady(intent.preview, intent.overlay)
            is AssistRuntimeIntent.DetectionEnabled -> setDetectionEnabled(intent.enabled)
            is AssistRuntimeIntent.SpeechEnabled -> setSpeechEnabled(intent.enabled)
            is AssistRuntimeIntent.VibrationEnabled -> setVibrationEnabled(intent.enabled)
            is AssistRuntimeIntent.SpeechStyleChanged -> setSpeechStyle(intent.style)
            is AssistRuntimeIntent.VibrationStrengthChanged -> setVibrationStrength(intent.strength)
            is AssistRuntimeIntent.AppLanguageChanged -> setAppLanguage(intent.language)
            is AssistRuntimeIntent.CareModeEnabled -> setCareModeEnabled(intent.enabled)
            is AssistRuntimeIntent.AlertProfileChanged -> setAlertProfile(intent.profile)
            is AssistRuntimeIntent.AssistScenarioChanged -> setAssistScenario(intent.scenario)
            is AssistRuntimeIntent.DailyUsageModeChanged -> setDailyUsageMode(intent.mode)
            AssistRuntimeIntent.QuietShortcutSelected -> setQuietShortcut()
            AssistRuntimeIntent.SensitiveShortcutSelected -> setSensitiveShortcut()
        }
    }

    private fun openCameraExperience() {
        openExperience(AssistInputSource.PHONE_CAMERA, null)
    }

    private fun openOfflineReplay(scenario: ReplayScenario) {
        openExperience(AssistInputSource.OFFLINE_REPLAY, scenario)
    }

    private fun openExperience(inputSource: AssistInputSource, replayScenario: ReplayScenario?) {
        if (stateMachine.currentState != AssistRuntimeState.Idle) return
        if (inputSource == AssistInputSource.OFFLINE_REPLAY) {
            requireNotNull(replayScenario) { "ReplayScenario is required for offline replay" }
        }
        if (inputSource != currentInputSource || replayScenario != currentReplayScenario) {
            val replacement = frameSourceFactory.create(
                source = inputSource,
                context = activity,
                lifecycleOwner = activity,
                replayScenario = replayScenario
            )
            cameraLifecycleAdapter.replaceFrameSource(replacement)
            currentInputSource = inputSource
            currentReplayScenario = replayScenario
        }
        appViewModel.onAssistInputSourceActivated(inputSource, replayScenario)
        handleTransition(
            stateMachine.onEvent(
                AssistRuntimeEvent.OpenCamera(
                    hasCameraPermission = hasCameraPermission(),
                    modelReady = detector.isReady,
                    inputSource = inputSource
                )
            )
        )
    }

    private fun requestCameraPermissionAfterExplanation(launchPermissionRequest: () -> Unit) {
        handleTransition(
            transition = stateMachine.onEvent(AssistRuntimeEvent.PermissionExplanationAccepted),
            launchPermissionRequest = launchPermissionRequest
        )
    }

    private fun dismissCameraPermissionFlow() {
        handleTransition(stateMachine.onEvent(AssistRuntimeEvent.PermissionFlowDismissed))
    }

    private fun onCameraPermissionResult(granted: Boolean) {
        handleTransition(
            stateMachine.onEvent(
                AssistRuntimeEvent.PermissionResult(
                    granted = granted,
                    modelReady = detector.isReady
                )
            )
        )
    }

    private fun closeCameraExperience() {
        handleTransition(stateMachine.onEvent(AssistRuntimeEvent.CloseCamera))
    }

    private fun onCameraViewsReady(preview: PreviewView?, overlay: DetectionOverlayView) {
        cameraLifecycleAdapter.onCameraViewsReady(preview, overlay)
        settingsController.syncConfigFromViewModel()
        handleTransition(stateMachine.onEvent(AssistRuntimeEvent.CameraViewsReady))
    }

    private fun setDetectionEnabled(enabled: Boolean) {
        settingsController.setDetectionEnabled(enabled)
        handleTransition(stateMachine.onEvent(AssistRuntimeEvent.DetectionChanged(enabled)))
    }

    private fun setSpeechEnabled(enabled: Boolean) {
        settingsController.setSpeechEnabled(enabled)
    }

    private fun setVibrationEnabled(enabled: Boolean) {
        settingsController.setVibrationEnabled(enabled)
    }

    private fun setSpeechStyle(style: SpeechStyle) {
        settingsController.setSpeechStyle(style)
    }

    private fun setVibrationStrength(strength: VibrationStrength) {
        settingsController.setVibrationStrength(strength)
    }

    private fun setAppLanguage(language: AppLanguage) {
        settingsController.setAppLanguage(language)
    }

    private fun setCareModeEnabled(enabled: Boolean) {
        settingsController.setCareModeEnabled(enabled)
    }

    private fun setAlertProfile(profile: AlertProfile) {
        settingsController.setAlertProfile(profile)
    }

    private fun setAssistScenario(scenario: AssistScenario) {
        settingsController.setAssistScenario(scenario)
    }

    private fun setDailyUsageMode(mode: DailyUsageMode) {
        settingsController.setDailyUsageMode(mode)
    }

    private fun setQuietShortcut() {
        settingsController.setReminderShortcut(
            profile = AlertProfile.QUIET,
            speechStyle = SpeechStyle.BRIEF,
            vibrationStrength = VibrationStrength.SOFT
        )
    }

    private fun setSensitiveShortcut() {
        settingsController.setReminderShortcut(
            profile = AlertProfile.SENSITIVE,
            speechStyle = SpeechStyle.STANDARD,
            vibrationStrength = VibrationStrength.STRONG
        )
    }

    private fun handleTransition(
        transition: AssistRuntimeTransition,
        launchPermissionRequest: (() -> Unit)? = null
    ) {
        effectExecutor.execute(transition, launchPermissionRequest)
    }

    private fun startCameraIfReady() {
        cameraLifecycleAdapter.startIfReady(frameProcessor::process)
    }

    private fun handleCameraFailure(message: String) {
        handleTransition(stateMachine.onEvent(AssistRuntimeEvent.CameraSourceFailed(message)))
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            activity,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private companion object {
        const val DUAL_LOOP_LOG_TAG = "BlindAssistDualLoop"
    }
}
