package com.linnan.blindassist.runtime

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.util.Log
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
import com.linnan.blindassist.preferences.DailyUsageMode
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.session.AssistFrameResult
import com.linnan.blindassist.session.AssistSessionCoordinator
import com.linnan.blindassist.ui.BlindAssistViewModel
import com.linnan.blindassist.ui.CameraGuidanceMapper
import com.linnan.blindassist.ui.DetectionOverlayView
import com.linnan.blindassist.ui.FieldTestSummaryMapper
import com.linnan.blindassist.ui.compose.CameraGuidanceUiState
import com.linnan.blindassist.vision.ObjectDetector
import java.util.concurrent.atomic.AtomicBoolean

class AssistRuntimeController(
    private val activity: ComponentActivity,
    private val appViewModel: BlindAssistViewModel,
    private val detector: ObjectDetector,
    private val feedbackController: FeedbackController,
    private val coordinator: AssistSessionCoordinator,
    frameSourceFactory: FrameSourceFactory,
    private val configApplier: RuntimeConfigApplier,
    private val stateMachine: AssistRuntimeStateMachine = AssistRuntimeStateMachine()
) {
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: DetectionOverlayView

    private val frameSource: FrameSource = frameSourceFactory.create(activity, activity)
    private val isProcessing = AtomicBoolean(false)
    private var lastPerfLogAtMs = 0L
    private var config = appViewModel.runtimeConfig()

    fun initialize() {
        syncConfigFromViewModel()
        renderUi(CameraGuidanceMapper.initial(detector.statusMessage, config.assistScenario, config.appLanguage))
    }

    fun shutdown() {
        stopCamera()
        frameSource.shutdown()
        detector.close()
        feedbackController.shutdown()
    }

    fun openCameraExperience() {
        handleTransition(
            stateMachine.onEvent(
                AssistRuntimeEvent.OpenCamera(
                    hasCameraPermission = hasCameraPermission(),
                    modelReady = detector.isReady
                )
            )
        )
    }

    fun requestCameraPermissionAfterExplanation(launchPermissionRequest: () -> Unit) {
        handleTransition(
            transition = stateMachine.onEvent(AssistRuntimeEvent.PermissionExplanationAccepted),
            launchPermissionRequest = launchPermissionRequest
        )
    }

    fun onCameraPermissionResult(granted: Boolean) {
        handleTransition(
            stateMachine.onEvent(
                AssistRuntimeEvent.PermissionResult(
                    granted = granted,
                    modelReady = detector.isReady
                )
            )
        )
    }

    fun closeCameraExperience() {
        handleTransition(stateMachine.onEvent(AssistRuntimeEvent.CloseCamera))
    }

    fun onCameraViewsReady(preview: PreviewView, overlay: DetectionOverlayView) {
        previewView = preview
        overlayView = overlay
        syncConfigFromViewModel()
        handleTransition(stateMachine.onEvent(AssistRuntimeEvent.CameraViewsReady))
    }

    fun setDetectionEnabled(enabled: Boolean) {
        appViewModel.onDetectionChange(enabled)
        syncConfigFromViewModel()
        handleTransition(stateMachine.onEvent(AssistRuntimeEvent.DetectionChanged(enabled)))
    }

    fun setSpeechEnabled(enabled: Boolean) {
        appViewModel.onSpeechChange(enabled)
        onConfigChanged(updateGuidance = false)
    }

    fun setVibrationEnabled(enabled: Boolean) {
        appViewModel.onVibrationChange(enabled)
        onConfigChanged(updateGuidance = false)
    }

    fun setSpeechStyle(style: SpeechStyle) {
        appViewModel.onSpeechStyleChange(style)
        onConfigChanged(updateGuidance = false)
    }

    fun setVibrationStrength(strength: VibrationStrength) {
        appViewModel.onVibrationStrengthChange(strength)
        onConfigChanged(updateGuidance = false)
    }

    fun setAppLanguage(language: AppLanguage) {
        appViewModel.onLanguageChange(language)
        onConfigChanged(updateGuidance = true)
    }

    fun setCareModeEnabled(enabled: Boolean) {
        appViewModel.onCareModeChange(enabled)
        onConfigChanged(updateGuidance = false)
    }

    fun setAlertProfile(profile: AlertProfile) {
        appViewModel.onProfileChange(profile)
        onConfigChanged(updateGuidance = false)
    }

    fun setAssistScenario(scenario: AssistScenario) {
        appViewModel.onScenarioChange(scenario)
        onConfigChanged(updateGuidance = true)
    }

    fun setDailyUsageMode(mode: DailyUsageMode) {
        appViewModel.onDailyUsageModeChange(mode)
        onConfigChanged(updateGuidance = true)
    }

    fun setQuietShortcut() {
        setReminderShortcut(
            profile = AlertProfile.QUIET,
            speechStyle = SpeechStyle.BRIEF,
            vibrationStrength = VibrationStrength.SOFT
        )
    }

    fun setSensitiveShortcut() {
        setReminderShortcut(
            profile = AlertProfile.SENSITIVE,
            speechStyle = SpeechStyle.STANDARD,
            vibrationStrength = VibrationStrength.STRONG
        )
    }

    private fun setReminderShortcut(
        profile: AlertProfile,
        speechStyle: SpeechStyle,
        vibrationStrength: VibrationStrength
    ) {
        appViewModel.onReminderShortcutChange(profile, speechStyle, vibrationStrength)
        onConfigChanged(updateGuidance = false)
    }

    private fun handleTransition(
        transition: AssistRuntimeTransition,
        launchPermissionRequest: (() -> Unit)? = null
    ) {
        transition.effects.forEach { effect ->
            when (effect) {
                AssistRuntimeEffect.ShowPermissionExplanation -> appViewModel.onShowCameraPermissionDialog()
                AssistRuntimeEffect.DismissPermissionExplanation -> appViewModel.onDismissCameraPermissionDialog()
                AssistRuntimeEffect.LaunchPermissionRequest -> launchPermissionRequest?.invoke()
                AssistRuntimeEffect.StartSession -> {
                    coordinator.startSession()
                    appViewModel.updateFieldTestSummary(currentFieldTestSummary(active = true))
                }
                AssistRuntimeEffect.ActivateCamera -> activateCameraUi()
                AssistRuntimeEffect.StartCameraIfReady -> startCameraIfReady()
                AssistRuntimeEffect.StopCamera -> stopCamera()
                AssistRuntimeEffect.ClearOverlay -> clearOverlay()
                AssistRuntimeEffect.CloseCamera -> closeCameraUi()
                AssistRuntimeEffect.ResetSession -> coordinator.reset()
                AssistRuntimeEffect.ShowPermissionDenied -> appViewModel.onCameraPermissionDenied(permissionDeniedGuidance())
                AssistRuntimeEffect.ApplyConfig -> syncConfigFromViewModel()
                is AssistRuntimeEffect.Render -> renderTarget(effect.target, effect.message)
            }
        }
    }

    private fun activateCameraUi() {
        val guidance = if (detector.isReady) {
            startingGuidance()
        } else {
            modelUnavailableGuidance()
        }
        appViewModel.activateCamera(
            fieldTestSummary = currentFieldTestSummary(active = true),
            guidance = guidance,
            modelStatus = detector.statusMessage
        )
    }

    private fun closeCameraUi() {
        appViewModel.closeCamera(
            fieldTestSummary = currentFieldTestSummary(active = false),
            guidance = CameraGuidanceMapper.initial(detector.statusMessage, config.assistScenario, config.appLanguage),
            modelStatus = detector.statusMessage
        )
    }

    private fun renderTarget(target: AssistRuntimeRenderTarget, message: String?) {
        val guidance = when (target) {
            AssistRuntimeRenderTarget.Starting -> startingGuidance()
            AssistRuntimeRenderTarget.Waiting -> {
                if (detector.isReady) {
                    CameraGuidanceMapper.waiting(detector.statusMessage, config.assistScenario, config.appLanguage)
                } else {
                    modelUnavailableGuidance()
                }
            }
            AssistRuntimeRenderTarget.Paused -> CameraGuidanceMapper.paused(config.appLanguage)
                .copy(scenarioName = config.assistScenario.displayName(config.appLanguage))
            AssistRuntimeRenderTarget.PermissionDenied -> permissionDeniedGuidance()
            AssistRuntimeRenderTarget.ModelUnavailable -> modelUnavailableGuidance()
            AssistRuntimeRenderTarget.CameraError -> CameraGuidanceMapper.cameraError(message ?: "未知错误", config.appLanguage)
        }
        renderUi(guidance)
    }

    private fun startCameraIfReady() {
        if (!appViewModel.uiState.value.cameraActive || !::previewView.isInitialized) return
        if (!hasCameraPermission()) {
            renderUi(permissionDeniedGuidance())
            return
        }
        frameSource.start(
            previewView = previewView,
            onFrame = ::processFrameBitmap,
            onStarted = {
                handleTransition(
                    stateMachine.onEvent(AssistRuntimeEvent.CameraStarted(detector.isReady))
                )
            },
            onError = { error ->
                val message = error.message ?: "未知错误"
                Log.e(PERF_TAG, "Camera start failed", error)
                handleTransition(stateMachine.onEvent(AssistRuntimeEvent.CameraStartFailed(message)))
            }
        )
    }

    private fun stopCamera() {
        frameSource.stop()
        isProcessing.set(false)
    }

    private fun clearOverlay() {
        if (::overlayView.isInitialized) {
            overlayView.update(emptyList(), null, null)
        }
    }

    private fun processFrameBitmap(bitmap: Bitmap) {
        if (!appViewModel.uiState.value.cameraActive || !config.detectionEnabled) {
            bitmap.recycle()
            return
        }
        if (!detector.isReady) {
            activity.runOnUiThread { renderUi(modelUnavailableGuidance()) }
            bitmap.recycle()
            return
        }
        if (!isProcessing.compareAndSet(false, true)) {
            bitmap.recycle()
            return
        }

        try {
            val detectorFrame = detector.detect(bitmap)
            val frameResult = coordinator.processFrame(detectorFrame, config.alertProfile, config.assistScenario)
            activity.runOnUiThread {
                if (::overlayView.isInitialized) {
                    overlayView.update(
                        detectorFrame.detections,
                        detectorFrame.frameSize,
                        frameResult.evaluation.stableRisk
                    )
                }
                renderUi(CameraGuidanceMapper.fromFrameResult(frameResult, config.appLanguage))
                appViewModel.updateFieldTestSummary(currentFieldTestSummary(active = true))
                logPerformanceIfNeeded(frameResult)
            }
        } catch (error: Throwable) {
            Log.e(PERF_TAG, "Frame processing failed", error)
            activity.runOnUiThread {
                renderUi(CameraGuidanceMapper.cameraError("检测异常：${error.message ?: "未知错误"}", config.appLanguage))
            }
        } finally {
            isProcessing.set(false)
            bitmap.recycle()
        }
    }

    private fun onConfigChanged(updateGuidance: Boolean) {
        syncConfigFromViewModel()
        appViewModel.updateFieldTestSummary(currentFieldTestSummary(appViewModel.uiState.value.cameraActive))
        if (updateGuidance) {
            renderGuidanceForCurrentState()
        }
    }

    private fun renderGuidanceForCurrentState() {
        val guidance = when (stateMachine.currentState) {
            AssistRuntimeState.Idle -> CameraGuidanceMapper.initial(
                detector.statusMessage,
                config.assistScenario,
                config.appLanguage
            )
            AssistRuntimeState.Starting -> startingGuidance()
            AssistRuntimeState.Running -> CameraGuidanceMapper.waiting(
                detector.statusMessage,
                config.assistScenario,
                config.appLanguage
            )
            AssistRuntimeState.DetectionPaused -> CameraGuidanceMapper.paused(config.appLanguage)
                .copy(scenarioName = config.assistScenario.displayName(config.appLanguage))
            AssistRuntimeState.PermissionDenied -> permissionDeniedGuidance()
            is AssistRuntimeState.Error -> {
                if (detector.isReady) {
                    CameraGuidanceMapper.cameraError((stateMachine.currentState as AssistRuntimeState.Error).message, config.appLanguage)
                } else {
                    modelUnavailableGuidance()
                }
            }
            AssistRuntimeState.PermissionExplaining,
            AssistRuntimeState.PermissionRequesting -> permissionDeniedGuidance()
        }
        renderUi(guidance)
    }

    private fun syncConfigFromViewModel() {
        config = appViewModel.runtimeConfig()
        configApplier.apply(config, if (::overlayView.isInitialized) overlayView else null)
    }

    private fun currentFieldTestSummary(active: Boolean) =
        FieldTestSummaryMapper.fromSummary(
            coordinator.sessionSummary(),
            active,
            config.alertProfile,
            config.assistScenario,
            config.appLanguage
        )

    private fun startingGuidance(): CameraGuidanceUiState {
        return CameraGuidanceMapper.starting(detector.statusMessage, config.assistScenario, config.appLanguage)
    }

    private fun modelUnavailableGuidance(): CameraGuidanceUiState {
        return CameraGuidanceMapper.modelUnavailable(detector.statusMessage, config.assistScenario, config.appLanguage)
    }

    private fun permissionDeniedGuidance(): CameraGuidanceUiState {
        return CameraGuidanceMapper.permissionDenied(config.appLanguage)
    }

    private fun renderUi(snapshot: CameraGuidanceUiState) {
        appViewModel.renderCameraGuidance(snapshot, detector.statusMessage)
    }

    private fun logPerformanceIfNeeded(frameResult: AssistFrameResult) {
        val now = System.currentTimeMillis()
        if (now - lastPerfLogAtMs < PERF_LOG_INTERVAL_MS) return
        lastPerfLogAtMs = now
        val evaluation = frameResult.evaluation
        val metrics = evaluation.metrics
        Log.i(
            PERF_TAG,
            "frame=${evaluation.frameSize.width}x${evaluation.frameSize.height}, " +
                "count=${evaluation.detectionCount}, " +
                "total=${metrics.totalMs}ms, pre=${metrics.preprocessMs}ms, " +
                "infer=${metrics.inferenceMs}ms, post=${metrics.postprocessMs}ms, " +
                "fps=${"%.1f".format(metrics.fps)}, profile=${evaluation.profile.storageValue}, " +
                "scenario=${evaluation.scenario.storageValue}, " +
                "rawRisk=${riskSummary(evaluation.rawRisk)}, stableRisk=${riskSummary(evaluation.stableRisk)}, " +
                "feedbackReason=${frameResult.feedbackDecision.reason.displayText(config.appLanguage)}, " +
                "explanation=${frameResult.explanation.headline}, " +
                "session=${frameResult.sessionSummary.displayText(config.appLanguage)}, status=${metrics.modelStatus}"
        )
    }

    private fun riskSummary(risk: RiskResult): String {
        return "${risk.level}/${risk.direction}/${risk.proximity}"
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            activity,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private companion object {
        const val PERF_LOG_INTERVAL_MS = 1000L
        const val PERF_TAG = "BlindAssistPerf"
    }
}
