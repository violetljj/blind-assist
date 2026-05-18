package com.linnan.blindassist

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.getValue
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.feedback.FeedbackController
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.SpeechStyle
import com.linnan.blindassist.feedback.VibrationStrength
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.preferences.UserPreferences
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.session.AssistDisplayFormatter
import com.linnan.blindassist.session.AssistEngine
import com.linnan.blindassist.session.DetectorMetrics
import com.linnan.blindassist.session.SessionSummary
import com.linnan.blindassist.ui.BlindAssistViewModel
import com.linnan.blindassist.ui.DetectionOverlayView
import com.linnan.blindassist.ui.compose.BlindAssistApp
import com.linnan.blindassist.ui.compose.BlindAssistTheme
import com.linnan.blindassist.ui.compose.CameraGuidanceUiState
import com.linnan.blindassist.ui.compose.FieldTestSummaryUiState
import com.linnan.blindassist.ui.compose.GlassesPlaceholderDialog
import com.linnan.blindassist.vision.TfliteYoloDetector
import com.linnan.blindassist.vision.toArgbBitmap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : ComponentActivity() {
    private val appViewModel: BlindAssistViewModel by viewModels {
        BlindAssistViewModel.Factory(userPreferences)
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: DetectionOverlayView
    private lateinit var detector: TfliteYoloDetector
    private lateinit var assistEngine: AssistEngine
    private lateinit var feedbackController: FeedbackController
    private lateinit var userPreferences: UserPreferences
    private lateinit var cameraExecutor: ExecutorService

    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraStarting = false
    private var cameraStarted = false
    private val isProcessing = AtomicBoolean(false)

    private var detectionEnabled = true
    private var frameCount = 0
    private var fpsWindowStartMs = System.currentTimeMillis()
    private var currentFps = 0f
    private var lastPerfLogAtMs = 0L
    private var careModeEnabled = false
    private var alertProfile = AlertProfile.STANDARD

    private var pendingCameraOpen = false

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted && pendingCameraOpen) {
            pendingCameraOpen = false
            activateCameraExperience()
        } else if (!granted) {
            pendingCameraOpen = false
            appViewModel.onCameraPermissionDenied(Guidance.permissionDenied())
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)

        detector = TfliteYoloDetector(this)
        assistEngine = AssistEngine()
        feedbackController = FeedbackController(this)
        userPreferences = UserPreferences(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        val initialControls = appViewModel.uiState.value.controls
        feedbackController.speechEnabled = initialControls.speechEnabled
        feedbackController.vibrationEnabled = initialControls.vibrationEnabled
        feedbackController.speechStyle = initialControls.speechStyle
        feedbackController.vibrationStrength = initialControls.vibrationStrength
        careModeEnabled = initialControls.careModeEnabled
        alertProfile = initialControls.alertProfile
        renderUi(Guidance.initial(detector.statusMessage))

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
                    showOnboarding = uiState.showOnboarding,
                    onOpenCamera = ::openCameraExperience,
                    onCloseCamera = ::closeCameraExperience,
                    onCompleteOnboarding = ::completeOnboarding,
                    onShowOnboarding = appViewModel::onShowOnboarding,
                    onGlassesPlaceholder = appViewModel::onShowGlassesDialog,
                    onDetectionChange = ::setDetectionEnabled,
                    onSpeechChange = ::setSpeechEnabled,
                    onVibrationChange = ::setVibrationEnabled,
                    onCareModeChange = ::setCareModeEnabled,
                    onDebugVisibleChange = ::setDebugVisible,
                    onProfileChange = ::setAlertProfile,
                    onSpeechStyleChange = ::setSpeechStyle,
                    onVibrationStrengthChange = ::setVibrationStrength,
                    onCameraViewsReady = ::onCameraViewsReady
                )
                if (uiState.showGlassesDialog) {
                    GlassesPlaceholderDialog(onDismiss = appViewModel::onDismissGlassesDialog)
                }
                if (uiState.showCameraPermissionDialog) {
                    com.linnan.blindassist.ui.compose.CameraPermissionExplanationDialog(
                        onContinue = ::requestCameraPermissionAfterExplanation,
                        onDismiss = appViewModel::onDismissCameraPermissionDialog
                    )
                }
                if (uiState.showPermissionDeniedDialog) {
                    com.linnan.blindassist.ui.compose.CameraPermissionDeniedDialog(
                        onDismiss = appViewModel::onDismissPermissionDeniedDialog
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopCamera()
        cameraExecutor.shutdown()
        detector.close()
        feedbackController.shutdown()
    }

    private fun openCameraExperience() {
        if (hasCameraPermission()) {
            activateCameraExperience()
        } else {
            appViewModel.onShowCameraPermissionDialog()
        }
    }

    private fun requestCameraPermissionAfterExplanation() {
        appViewModel.onDismissCameraPermissionDialog()
        pendingCameraOpen = true
        requestCameraPermission.launch(Manifest.permission.CAMERA)
    }

    private fun activateCameraExperience() {
        assistEngine.startSession()
        appViewModel.activateCamera(
            fieldTestSummary = assistEngine.sessionSummary().toFieldTestUi(active = true, profile = alertProfile),
            guidance = Guidance.waiting(detector.statusMessage),
            modelStatus = detector.statusMessage
        )
        startCameraIfReady()
    }

    private fun closeCameraExperience() {
        pendingCameraOpen = false
        stopCamera()
        appViewModel.closeCamera(
            fieldTestSummary = assistEngine.sessionSummary().toFieldTestUi(active = false, profile = alertProfile),
            guidance = Guidance.initial(detector.statusMessage),
            modelStatus = detector.statusMessage
        )
        assistEngine.reset()
    }

    private fun completeOnboarding() {
        appViewModel.onCompleteOnboarding()
    }

    private fun onCameraViewsReady(preview: PreviewView, overlay: DetectionOverlayView) {
        previewView = preview
        overlayView = overlay
        overlayView.setCareMode(careModeEnabled)
        cameraStarted = false
        cameraStarting = false
        if (appViewModel.uiState.value.cameraActive) {
            startCameraIfReady()
        }
    }

    private fun startCameraIfReady() {
        if (!appViewModel.uiState.value.cameraActive || !::previewView.isInitialized) return
        if (!hasCameraPermission()) {
            renderUi(Guidance.permissionDenied())
            return
        }
        if (cameraStarted || cameraStarting) return

        cameraStarting = true
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val provider = cameraProviderFuture.get()
                cameraProvider = provider
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val resolutionSelector = ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                        )
                    )
                    .build()
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(resolutionSelector)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .also { analyzer ->
                        analyzer.setAnalyzer(cameraExecutor) { imageProxy ->
                            try {
                                analyzeFrame(imageProxy.toArgbBitmap())
                            } finally {
                                imageProxy.close()
                            }
                        }
                    }

                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
                cameraStarted = true
                renderUi(Guidance.waiting(detector.statusMessage))
            } catch (error: Exception) {
                Log.e(PERF_TAG, "Camera start failed", error)
                renderUi(Guidance.cameraError(error.message ?: "未知错误"))
            } finally {
                cameraStarting = false
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
        cameraStarted = false
        cameraStarting = false
        isProcessing.set(false)
        if (::overlayView.isInitialized) {
            overlayView.update(emptyList(), null, null)
        }
    }

    private fun analyzeFrame(bitmap: Bitmap) {
        if (!appViewModel.uiState.value.cameraActive || !detectionEnabled || !detector.isReady) {
            bitmap.recycle()
            return
        }
        if (!isProcessing.compareAndSet(false, true)) {
            bitmap.recycle()
            return
        }

        try {
            val detections = detector.detect(bitmap)
            val frameSize = FrameSize(bitmap.width, bitmap.height)
            val fps = updateFps()
            val evaluation = assistEngine.evaluate(
                detections = detections,
                frameSize = frameSize,
                profile = alertProfile,
                metrics = DetectorMetrics(
                    totalMs = detector.lastTotalDetectMs,
                    preprocessMs = detector.lastPreprocessMs,
                    inferenceMs = detector.lastInferenceMs,
                    postprocessMs = detector.lastPostprocessMs,
                    fps = fps,
                    modelStatus = detector.statusMessage
                )
            )
            runOnUiThread {
                val feedbackDecision = feedbackController.notify(evaluation.stableRisk, alertProfile)
                val frameResult = assistEngine.completeFeedback(evaluation, feedbackDecision)
                overlayView.update(detections, frameSize, frameResult.evaluation.stableRisk)
                renderUi(
                    Guidance.fromRisk(
                        rawRisk = frameResult.evaluation.rawRisk,
                        stableRisk = frameResult.evaluation.stableRisk,
                        count = detections.size,
                        detector = detector,
                        fps = fps,
                        profile = alertProfile,
                        feedbackDecision = frameResult.feedbackDecision
                    )
                )
                appViewModel.updateFieldTestSummary(frameResult.sessionSummary.toFieldTestUi(active = true, profile = alertProfile))
                logPerformanceIfNeeded(frameResult)
            }
        } finally {
            isProcessing.set(false)
            bitmap.recycle()
        }
    }

    private fun setDetectionEnabled(enabled: Boolean) {
        detectionEnabled = enabled
        appViewModel.onDetectionChange(enabled)
        if (!enabled) {
            appViewModel.updateFieldTestSummary(assistEngine.sessionSummary().toFieldTestUi(active = false, profile = alertProfile))
            assistEngine.reset()
            if (::overlayView.isInitialized) {
                overlayView.update(emptyList(), null, null)
            }
            renderUi(Guidance.paused())
        } else {
            assistEngine.startSession()
            appViewModel.updateFieldTestSummary(assistEngine.sessionSummary().toFieldTestUi(active = true, profile = alertProfile))
            renderUi(Guidance.waiting(detector.statusMessage))
            startCameraIfReady()
        }
    }

    private fun setSpeechEnabled(enabled: Boolean) {
        feedbackController.speechEnabled = enabled
        appViewModel.onSpeechChange(enabled)
    }

    private fun setVibrationEnabled(enabled: Boolean) {
        feedbackController.vibrationEnabled = enabled
        appViewModel.onVibrationChange(enabled)
    }

    private fun setSpeechStyle(style: SpeechStyle) {
        feedbackController.speechStyle = style
        appViewModel.onSpeechStyleChange(style)
    }

    private fun setVibrationStrength(strength: VibrationStrength) {
        feedbackController.vibrationStrength = strength
        appViewModel.onVibrationStrengthChange(strength)
    }

    private fun setCareModeEnabled(enabled: Boolean) {
        careModeEnabled = enabled
        if (::overlayView.isInitialized) {
            overlayView.setCareMode(enabled)
        }
        appViewModel.onCareModeChange(enabled)
    }

    private fun setDebugVisible(visible: Boolean) {
        appViewModel.onDebugVisibleChange(visible)
    }

    private fun setAlertProfile(profile: AlertProfile) {
        alertProfile = profile
        appViewModel.onProfileChange(profile)
        appViewModel.updateFieldTestSummary(
            assistEngine.sessionSummary().toFieldTestUi(
                active = appViewModel.uiState.value.cameraActive,
                profile = alertProfile
            )
        )
    }

    private fun renderUi(snapshot: CameraGuidanceUiState) {
        val status = if (::detector.isInitialized) detector.statusMessage else appViewModel.uiState.value.modelStatus
        appViewModel.renderCameraGuidance(snapshot, status)
    }

    private fun updateFps(): Float {
        frameCount += 1
        val now = System.currentTimeMillis()
        val elapsed = now - fpsWindowStartMs
        if (elapsed >= 1000L) {
            currentFps = frameCount * 1000f / elapsed.toFloat()
            frameCount = 0
            fpsWindowStartMs = now
        }
        return currentFps
    }

    private fun logPerformanceIfNeeded(frameResult: com.linnan.blindassist.session.AssistFrameResult) {
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
                "rawRisk=${riskSummary(evaluation.rawRisk)}, stableRisk=${riskSummary(evaluation.stableRisk)}, " +
                "feedbackReason=${frameResult.feedbackDecision.reason.displayText}, " +
                "session=${frameResult.sessionSummary.displayText()}, status=${metrics.modelStatus}"
        )
    }

    private fun riskSummary(risk: com.linnan.blindassist.risk.RiskResult): String {
        return "${risk.level}/${risk.direction}/${risk.proximity}"
    }

    private fun SessionSummary.toFieldTestUi(active: Boolean, profile: AlertProfile): FieldTestSummaryUiState {
        val status = when {
            active && frameCount > 0 -> "本次相机会话进行中"
            active -> "本次相机会话已开始，等待检测帧"
            hasStarted || frameCount > 0 -> "上一场相机会话摘要"
            else -> "等待相机会话"
        }
        return FieldTestSummaryUiState(
            title = "现场测试摘要",
            detailText = fieldTestText(profile.displayName),
            statusText = status,
            accessibilityText = "现场测试摘要，$status，运行时长${durationText()}，最近${frameCount}帧风险${riskyFrameCount}次，语音提醒${speechTriggerCount}次，震动提醒${vibrationTriggerCount}次，平均FPS ${"%.1f".format(averageFps)}，平均推理${averageInferenceMs}毫秒，当前档位${profile.displayName}。"
        )
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private object Guidance {
        fun initial(modelStatus: String): CameraGuidanceUiState {
            return CameraGuidanceUiState.initial(modelStatus)
        }

        fun waiting(modelStatus: String): CameraGuidanceUiState {
            return CameraGuidanceUiState(
                title = "检测已开启",
                detail = "等待实时画面和稳定风险结果",
                targetLine = "模型状态：$modelStatus",
                careTitle = "正在观察",
                careDetail = "请自然前进，系统会在前方有风险时提醒",
                careTargetLine = "建议同时保留语音和震动提醒",
                debugText = "模型状态：$modelStatus",
                titleColor = Color.WHITE,
                statusBadge = "观察中",
                badgeColor = Color.rgb(160, 255, 215),
                badgeTextColor = Color.rgb(6, 24, 18),
                careAccessibilitySummary = "正在观察，请自然前进，系统会在前方有风险时提醒",
                accessibilitySummary = "检测已开启，等待实时画面和稳定风险结果",
                accessibilityKey = "waiting"
            )
        }

        fun paused(): CameraGuidanceUiState {
            return CameraGuidanceUiState(
                title = "检测已暂停",
                detail = "画面保留预览，目标框和风险提醒已清空",
                targetLine = "可随时重新开启检测",
                careTitle = "已暂停",
                careDetail = "当前不会识别目标，也不会发出提醒",
                careTargetLine = "打开检测后恢复观察",
                debugText = "检测关闭：不运行目标检测，不触发语音或震动提醒",
                titleColor = Color.rgb(214, 224, 235),
                statusBadge = "已暂停",
                badgeColor = Color.rgb(198, 210, 222),
                badgeTextColor = Color.rgb(12, 22, 30),
                careAccessibilitySummary = "检测已暂停，当前不会识别目标，也不会发出提醒",
                accessibilitySummary = "检测已暂停，目标框和风险提醒已清空",
                accessibilityKey = "paused"
            )
        }

        fun permissionDenied(): CameraGuidanceUiState {
            return CameraGuidanceUiState(
                title = "需要相机权限",
                detail = "请授予相机权限后再使用实时避障提醒",
                targetLine = "当前无法启动 CameraX 预览和检测",
                careTitle = "需要权限",
                careDetail = "请允许相机权限，系统才能观察前方",
                careTargetLine = "授权后会自动启动实时预览",
                debugText = "权限状态：CAMERA denied",
                titleColor = Color.rgb(255, 149, 0),
                statusBadge = "需处理",
                badgeColor = Color.rgb(255, 210, 125),
                badgeTextColor = Color.rgb(44, 25, 0),
                careAccessibilitySummary = "需要相机权限，请允许相机权限，系统才能观察前方",
                accessibilitySummary = "需要相机权限，请授予相机权限后再使用实时避障提醒",
                accessibilityKey = "permission"
            )
        }

        fun cameraError(message: String): CameraGuidanceUiState {
            return CameraGuidanceUiState(
                title = "相机启动失败",
                detail = "CameraX 暂时无法打开后置摄像头",
                targetLine = message,
                careTitle = "相机未启动",
                careDetail = "当前无法观察前方，请返回后重新进入手机摄像头",
                careTargetLine = message,
                debugText = "CameraX error：$message",
                titleColor = Color.rgb(255, 149, 0),
                statusBadge = "异常",
                badgeColor = Color.rgb(255, 210, 125),
                badgeTextColor = Color.rgb(44, 25, 0),
                careAccessibilitySummary = "相机启动失败，当前无法观察前方",
                accessibilitySummary = "相机启动失败，CameraX 暂时无法打开后置摄像头",
                accessibilityKey = "camera-error-$message"
            )
        }

        fun fromRisk(
            rawRisk: com.linnan.blindassist.risk.RiskResult,
            stableRisk: com.linnan.blindassist.risk.RiskResult,
            count: Int,
            detector: TfliteYoloDetector,
            fps: Float,
            profile: AlertProfile,
            feedbackDecision: FeedbackDecision,
        ): CameraGuidanceUiState {
            val risk = stableRisk
            val levelText = levelText(risk.level)
            val directionText = directionText(risk.direction)
            val title = when (risk.level) {
                RiskLevel.HIGH -> "$levelText：$directionText"
                RiskLevel.MEDIUM -> "$levelText：$directionText"
                RiskLevel.LOW -> levelText
                RiskLevel.NONE -> "安全观察中"
            }
            val detail = AssistDisplayFormatter.detailFor(risk)
            val targetLine = AssistDisplayFormatter.targetLine(rawRisk, risk, count)
            val careTitle = careTitle(risk.level, risk.direction, risk.proximity)
            val careDetail = AssistDisplayFormatter.careDetailFor(risk)
            val careTargetLine = AssistDisplayFormatter.careTargetLine(rawRisk, risk, count)
            val targetAccessibility = AssistDisplayFormatter.accessibilityTargetSummary(rawRisk, risk, count)
            val debug = "FPS：${"%.1f".format(fps)}\n" +
                "耗时：total ${detector.lastTotalDetectMs}ms / pre ${detector.lastPreprocessMs}ms / " +
                "infer ${detector.lastInferenceMs}ms / post ${detector.lastPostprocessMs}ms\n" +
                "模型：${detector.statusMessage}\n" +
                "最近风险判定：原始 ${riskSummaryText(rawRisk)} / 稳定 ${riskSummaryText(stableRisk)}\n" +
                AssistDisplayFormatter.urgencyLine(rawRisk, stableRisk) + "\n" +
                "提醒模式：${profile.displayName} / 反馈：${feedbackDecision.reason.displayText}"
            return CameraGuidanceUiState(
                title = title,
                detail = detail,
                targetLine = targetLine,
                careTitle = careTitle,
                careDetail = careDetail,
                careTargetLine = careTargetLine,
                debugText = debug,
                titleColor = colorForLevel(risk.level, risk.proximity),
                statusBadge = statusBadge(risk.level, risk.proximity),
                badgeColor = badgeColor(risk.level, risk.proximity),
                badgeTextColor = badgeTextColor(risk.level, risk.proximity),
                careAccessibilitySummary = "$careTitle，$careDetail，$targetAccessibility",
                accessibilitySummary = "$title，$detail，$targetAccessibility",
                accessibilityKey = "${risk.level}-${risk.direction}-${risk.proximity}-${rawRisk.level}-$count"
            )
        }

        private fun riskSummaryText(risk: com.linnan.blindassist.risk.RiskResult): String {
            return "${levelText(risk.level)} ${directionText(risk.direction)} ${proximityText(risk.proximity)}"
        }

        private fun levelText(level: RiskLevel): String {
            return when (level) {
                RiskLevel.HIGH -> "高风险"
                RiskLevel.MEDIUM -> "中风险"
                RiskLevel.LOW -> "低风险"
                RiskLevel.NONE -> "安全"
            }
        }

        private fun directionText(direction: RiskDirection): String {
            return when (direction) {
                RiskDirection.LEFT -> "左前"
                RiskDirection.CENTER -> "正前"
                RiskDirection.RIGHT -> "右前"
                RiskDirection.NONE -> "无方向"
            }
        }

        private fun proximityText(proximity: ProximityBand): String {
            return when (proximity) {
                ProximityBand.CRITICAL -> "迫近"
                ProximityBand.NEAR -> "近处"
                ProximityBand.MID -> "中距"
                ProximityBand.FAR -> "远处"
            }
        }

        private fun careTitle(
            level: RiskLevel,
            direction: RiskDirection,
            proximity: ProximityBand
        ): String {
            return when {
                proximity == ProximityBand.CRITICAL -> "立刻注意：${directionText(direction)}"
                level == RiskLevel.HIGH -> "前方有风险：${directionText(direction)}"
                level == RiskLevel.MEDIUM -> "请留意：${directionText(direction)}"
                level == RiskLevel.LOW -> "保持观察"
                else -> "前方平稳"
            }
        }

        private fun statusBadge(level: RiskLevel, proximity: ProximityBand): String {
            return when {
                proximity == ProximityBand.CRITICAL -> "迫近提醒"
                level == RiskLevel.HIGH -> "高风险"
                level == RiskLevel.MEDIUM -> "需留意"
                level == RiskLevel.LOW -> "观察"
                else -> "平稳"
            }
        }

        private fun badgeColor(level: RiskLevel, proximity: ProximityBand): Int {
            return when {
                proximity == ProximityBand.CRITICAL -> Color.rgb(255, 99, 119)
                level == RiskLevel.HIGH -> Color.rgb(255, 132, 105)
                level == RiskLevel.MEDIUM -> Color.rgb(255, 205, 112)
                level == RiskLevel.LOW -> Color.rgb(239, 226, 133)
                else -> Color.rgb(160, 255, 215)
            }
        }

        private fun badgeTextColor(level: RiskLevel, proximity: ProximityBand): Int {
            return if (proximity == ProximityBand.CRITICAL || level == RiskLevel.HIGH) {
                Color.WHITE
            } else {
                Color.rgb(15, 24, 18)
            }
        }

        private fun colorForLevel(level: RiskLevel, proximity: ProximityBand): Int {
            return when {
                proximity == ProximityBand.CRITICAL -> Color.rgb(255, 99, 119)
                level == RiskLevel.HIGH -> Color.rgb(255, 112, 97)
                level == RiskLevel.MEDIUM -> Color.rgb(255, 183, 77)
                level == RiskLevel.LOW -> Color.rgb(255, 224, 102)
                else -> Color.rgb(99, 230, 166)
            }
        }
    }

    companion object {
        private const val ANALYSIS_WIDTH = 640
        private const val ANALYSIS_HEIGHT = 480
        private const val PERF_LOG_INTERVAL_MS = 1000L
        private const val PERF_TAG = "BlindAssistPerf"
    }
}
