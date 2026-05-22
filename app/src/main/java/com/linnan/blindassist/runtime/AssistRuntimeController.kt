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
import com.linnan.blindassist.camera.CameraXFrameSource
import com.linnan.blindassist.camera.FrameSource
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
import com.linnan.blindassist.vision.ObjectDetector
import com.linnan.blindassist.vision.TfliteYoloDetector
import java.util.concurrent.atomic.AtomicBoolean

class AssistRuntimeController(
    private val activity: ComponentActivity,
    private val appViewModel: BlindAssistViewModel,
    private val detector: ObjectDetector = TfliteYoloDetector(activity),
    private val feedbackController: FeedbackController = FeedbackController(activity),
    private val coordinator: AssistSessionCoordinator = AssistSessionCoordinator(
        feedbackGateway = feedbackController
    ),
    private val frameSource: FrameSource = CameraXFrameSource(activity, activity)
) {
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: DetectionOverlayView

    private val isProcessing = AtomicBoolean(false)
    private var lastPerfLogAtMs = 0L
    private var pendingCameraOpen = false
    private var detectionEnabled = true
    private var careModeEnabled = false
    private var alertProfile = AlertProfile.STANDARD
    private var assistScenario = AssistScenario.GENERAL
    private var appLanguage = AppLanguage.ZH

    fun initialize() {
        val initialControls = appViewModel.uiState.value.controls
        feedbackController.speechEnabled = initialControls.speechEnabled
        feedbackController.vibrationEnabled = initialControls.vibrationEnabled
        feedbackController.speechStyle = initialControls.speechStyle
        feedbackController.vibrationStrength = initialControls.vibrationStrength
        feedbackController.appLanguage = initialControls.appLanguage
        careModeEnabled = initialControls.careModeEnabled
        alertProfile = initialControls.alertProfile
        assistScenario = initialControls.assistScenario
        appLanguage = initialControls.appLanguage
        renderUi(CameraGuidanceMapper.initial(detector.statusMessage, assistScenario, appLanguage))
    }

    fun shutdown() {
        stopCamera()
        frameSource.shutdown()
        detector.close()
        feedbackController.shutdown()
    }

    fun openCameraExperience() {
        if (hasCameraPermission()) {
            activateCameraExperience()
        } else {
            appViewModel.onShowCameraPermissionDialog()
        }
    }

    fun requestCameraPermissionAfterExplanation(launchPermissionRequest: () -> Unit) {
        appViewModel.onDismissCameraPermissionDialog()
        pendingCameraOpen = true
        launchPermissionRequest()
    }

    fun onCameraPermissionResult(granted: Boolean) {
        if (granted && pendingCameraOpen) {
            pendingCameraOpen = false
            activateCameraExperience()
        } else if (!granted) {
            pendingCameraOpen = false
            appViewModel.onCameraPermissionDenied(CameraGuidanceMapper.permissionDenied(appLanguage))
        }
    }

    fun closeCameraExperience() {
        pendingCameraOpen = false
        stopCamera()
        appViewModel.closeCamera(
            fieldTestSummary = currentFieldTestSummary(active = false),
            guidance = CameraGuidanceMapper.initial(detector.statusMessage, assistScenario, appLanguage),
            modelStatus = detector.statusMessage
        )
        coordinator.reset()
    }

    fun onCameraViewsReady(preview: PreviewView, overlay: DetectionOverlayView) {
        previewView = preview
        overlayView = overlay
        overlayView.setCareMode(careModeEnabled)
        if (appViewModel.uiState.value.cameraActive) {
            startCameraIfReady()
        }
    }

    fun setDetectionEnabled(enabled: Boolean) {
        detectionEnabled = enabled
        appViewModel.onDetectionChange(enabled)
        if (enabled) {
            coordinator.startSession()
            appViewModel.updateFieldTestSummary(currentFieldTestSummary(active = true))
            renderUi(CameraGuidanceMapper.waiting(detector.statusMessage, assistScenario, appLanguage))
            startCameraIfReady()
        } else {
            appViewModel.updateFieldTestSummary(currentFieldTestSummary(active = false))
            coordinator.reset()
            if (::overlayView.isInitialized) {
                overlayView.update(emptyList(), null, null)
            }
            renderUi(CameraGuidanceMapper.paused(appLanguage))
        }
    }

    fun setSpeechEnabled(enabled: Boolean) {
        feedbackController.speechEnabled = enabled
        appViewModel.onSpeechChange(enabled)
    }

    fun setVibrationEnabled(enabled: Boolean) {
        feedbackController.vibrationEnabled = enabled
        appViewModel.onVibrationChange(enabled)
    }

    fun setSpeechStyle(style: SpeechStyle) {
        feedbackController.speechStyle = style
        appViewModel.onSpeechStyleChange(style)
    }

    fun setVibrationStrength(strength: VibrationStrength) {
        feedbackController.vibrationStrength = strength
        appViewModel.onVibrationStrengthChange(strength)
    }

    fun setAppLanguage(language: AppLanguage) {
        appLanguage = language
        feedbackController.appLanguage = language
        appViewModel.onLanguageChange(language)
        appViewModel.updateFieldTestSummary(currentFieldTestSummary(appViewModel.uiState.value.cameraActive))
        val guidance = if (appViewModel.uiState.value.cameraActive) {
            CameraGuidanceMapper.waiting(detector.statusMessage, assistScenario, appLanguage)
        } else {
            CameraGuidanceMapper.initial(detector.statusMessage, assistScenario, appLanguage)
        }
        renderUi(guidance)
    }

    fun setCareModeEnabled(enabled: Boolean) {
        careModeEnabled = enabled
        if (::overlayView.isInitialized) {
            overlayView.setCareMode(enabled)
        }
        appViewModel.onCareModeChange(enabled)
    }

    fun setAlertProfile(profile: AlertProfile) {
        alertProfile = profile
        appViewModel.onProfileChange(profile)
        appViewModel.updateFieldTestSummary(currentFieldTestSummary(appViewModel.uiState.value.cameraActive))
    }

    fun setAssistScenario(scenario: AssistScenario) {
        assistScenario = scenario
        appViewModel.onScenarioChange(scenario)
        appViewModel.updateFieldTestSummary(currentFieldTestSummary(appViewModel.uiState.value.cameraActive))
        renderUi(appViewModel.uiState.value.cameraGuidance.copy(scenarioName = scenario.displayName(appLanguage)))
    }

    fun setDailyUsageMode(mode: DailyUsageMode) {
        val config = mode.config ?: return
        assistScenario = config.scenario
        alertProfile = config.profile
        careModeEnabled = config.careModeEnabled
        feedbackController.speechStyle = config.speechStyle
        feedbackController.vibrationStrength = config.vibrationStrength
        if (::overlayView.isInitialized) {
            overlayView.setCareMode(config.careModeEnabled)
        }
        appViewModel.onDailyUsageModeChange(mode)
        appViewModel.updateFieldTestSummary(currentFieldTestSummary(appViewModel.uiState.value.cameraActive))
        renderUi(appViewModel.uiState.value.cameraGuidance.copy(scenarioName = config.scenario.displayName(appLanguage)))
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

    private fun activateCameraExperience() {
        coordinator.startSession()
        appViewModel.activateCamera(
            fieldTestSummary = currentFieldTestSummary(active = true),
            guidance = CameraGuidanceMapper.waiting(detector.statusMessage, assistScenario, appLanguage),
            modelStatus = detector.statusMessage
        )
        startCameraIfReady()
    }

    private fun startCameraIfReady() {
        if (!appViewModel.uiState.value.cameraActive || !::previewView.isInitialized) return
        if (!hasCameraPermission()) {
            renderUi(CameraGuidanceMapper.permissionDenied(appLanguage))
            return
        }
        frameSource.start(
            previewView = previewView,
            onFrame = ::processFrameBitmap,
            onStarted = { renderUi(CameraGuidanceMapper.waiting(detector.statusMessage, assistScenario, appLanguage)) },
            onError = { error ->
                Log.e(PERF_TAG, "Camera start failed", error)
                renderUi(CameraGuidanceMapper.cameraError(error.message ?: "未知错误", appLanguage))
            }
        )
    }

    private fun stopCamera() {
        frameSource.stop()
        isProcessing.set(false)
        if (::overlayView.isInitialized) {
            overlayView.update(emptyList(), null, null)
        }
    }

    private fun processFrameBitmap(bitmap: Bitmap) {
        if (!appViewModel.uiState.value.cameraActive || !detectionEnabled || !detector.isReady) {
            bitmap.recycle()
            return
        }
        if (!isProcessing.compareAndSet(false, true)) {
            bitmap.recycle()
            return
        }

        try {
            val detectorFrame = detector.detect(bitmap)
            val frameResult = coordinator.processFrame(detectorFrame, alertProfile, assistScenario)
            activity.runOnUiThread {
                overlayView.update(
                    detectorFrame.detections,
                    detectorFrame.frameSize,
                    frameResult.evaluation.stableRisk
                )
                renderUi(CameraGuidanceMapper.fromFrameResult(frameResult, appLanguage))
                appViewModel.updateFieldTestSummary(currentFieldTestSummary(active = true))
                logPerformanceIfNeeded(frameResult)
            }
        } catch (error: Throwable) {
            Log.e(PERF_TAG, "Frame processing failed", error)
            activity.runOnUiThread {
                renderUi(CameraGuidanceMapper.cameraError("检测异常：${error.message ?: "未知错误"}", appLanguage))
            }
        } finally {
            isProcessing.set(false)
            bitmap.recycle()
        }
    }

    private fun setReminderShortcut(
        profile: AlertProfile,
        speechStyle: SpeechStyle,
        vibrationStrength: VibrationStrength
    ) {
        alertProfile = profile
        feedbackController.speechStyle = speechStyle
        feedbackController.vibrationStrength = vibrationStrength
        appViewModel.onReminderShortcutChange(profile, speechStyle, vibrationStrength)
        appViewModel.updateFieldTestSummary(currentFieldTestSummary(appViewModel.uiState.value.cameraActive))
    }

    private fun currentFieldTestSummary(active: Boolean) =
        FieldTestSummaryMapper.fromSummary(coordinator.sessionSummary(), active, alertProfile, assistScenario, appLanguage)

    private fun renderUi(snapshot: com.linnan.blindassist.ui.compose.CameraGuidanceUiState) {
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
                "feedbackReason=${frameResult.feedbackDecision.reason.displayText(appLanguage)}, " +
                "explanation=${frameResult.explanation.headline}, " +
                "session=${frameResult.sessionSummary.displayText(appLanguage)}, status=${metrics.modelStatus}"
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
