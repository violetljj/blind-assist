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
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.feedback.FeedbackController
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.preferences.UserPreferences
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.session.AssistDisplayFormatter
import com.linnan.blindassist.session.AssistEngine
import com.linnan.blindassist.session.DetectorMetrics
import com.linnan.blindassist.ui.DetectionOverlayView
import com.linnan.blindassist.ui.compose.AssistControlsUiState
import com.linnan.blindassist.ui.compose.BlindAssistApp
import com.linnan.blindassist.ui.compose.BlindAssistTheme
import com.linnan.blindassist.ui.compose.CameraGuidanceUiState
import com.linnan.blindassist.ui.compose.GlassesPlaceholderDialog
import com.linnan.blindassist.vision.TfliteYoloDetector
import com.linnan.blindassist.vision.toArgbBitmap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : ComponentActivity() {
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
    private var debugVisible = false
    private var careModeEnabled = false
    private var alertProfile = AlertProfile.STANDARD

    private var controlsState by mutableStateOf(
        AssistControlsUiState(
            detectionEnabled = true,
            speechEnabled = true,
            vibrationEnabled = true,
            careModeEnabled = false,
            debugVisible = false,
            alertProfile = AlertProfile.STANDARD
        )
    )
    private var guidanceState by mutableStateOf(Guidance.initial("模型未初始化"))
    private var cameraActive by mutableStateOf(false)
    private var modelStatus by mutableStateOf("模型未初始化")
    private var showGlassesDialog by mutableStateOf(false)
    private var showOnboarding by mutableStateOf(false)
    private var showCameraPermissionDialog by mutableStateOf(false)
    private var showPermissionDeniedDialog by mutableStateOf(false)
    private var pendingCameraOpen = false

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted && pendingCameraOpen) {
            pendingCameraOpen = false
            activateCameraExperience()
        } else if (!granted) {
            pendingCameraOpen = false
            cameraActive = false
            showPermissionDeniedDialog = true
            renderUi(Guidance.permissionDenied())
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

        val savedPreferences = userPreferences.load()
        feedbackController.speechEnabled = savedPreferences.speechEnabled
        feedbackController.vibrationEnabled = savedPreferences.vibrationEnabled
        careModeEnabled = savedPreferences.careModeEnabled
        alertProfile = savedPreferences.alertProfile
        showOnboarding = !savedPreferences.onboardingCompleted
        modelStatus = detector.statusMessage
        renderUi(Guidance.initial(detector.statusMessage))
        syncControlsState()

        setContent {
            BlindAssistTheme {
                BlindAssistApp(
                    controls = controlsState,
                    cameraGuidance = guidanceState,
                    modelStatus = modelStatus,
                    appVersion = BuildConfig.VERSION_NAME,
                    cameraActive = cameraActive,
                    showOnboarding = showOnboarding,
                    onOpenCamera = ::openCameraExperience,
                    onCloseCamera = ::closeCameraExperience,
                    onCompleteOnboarding = ::completeOnboarding,
                    onShowOnboarding = { showOnboarding = true },
                    onGlassesPlaceholder = { showGlassesDialog = true },
                    onDetectionChange = ::setDetectionEnabled,
                    onSpeechChange = ::setSpeechEnabled,
                    onVibrationChange = ::setVibrationEnabled,
                    onCareModeChange = ::setCareModeEnabled,
                    onDebugVisibleChange = ::setDebugVisible,
                    onProfileChange = ::setAlertProfile,
                    onCameraViewsReady = ::onCameraViewsReady
                )
                if (showGlassesDialog) {
                    GlassesPlaceholderDialog(onDismiss = { showGlassesDialog = false })
                }
                if (showCameraPermissionDialog) {
                    com.linnan.blindassist.ui.compose.CameraPermissionExplanationDialog(
                        onContinue = ::requestCameraPermissionAfterExplanation,
                        onDismiss = { showCameraPermissionDialog = false }
                    )
                }
                if (showPermissionDeniedDialog) {
                    com.linnan.blindassist.ui.compose.CameraPermissionDeniedDialog(
                        onDismiss = { showPermissionDeniedDialog = false }
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
            showCameraPermissionDialog = true
        }
    }

    private fun requestCameraPermissionAfterExplanation() {
        showCameraPermissionDialog = false
        pendingCameraOpen = true
        requestCameraPermission.launch(Manifest.permission.CAMERA)
    }

    private fun activateCameraExperience() {
        cameraActive = true
        renderUi(Guidance.waiting(detector.statusMessage))
        startCameraIfReady()
    }

    private fun closeCameraExperience() {
        cameraActive = false
        pendingCameraOpen = false
        showCameraPermissionDialog = false
        stopCamera()
        assistEngine.reset()
        renderUi(Guidance.initial(detector.statusMessage))
    }

    private fun completeOnboarding() {
        userPreferences.setOnboardingCompleted(true)
        showOnboarding = false
    }

    private fun onCameraViewsReady(preview: PreviewView, overlay: DetectionOverlayView) {
        previewView = preview
        overlayView = overlay
        overlayView.setCareMode(careModeEnabled)
        cameraStarted = false
        cameraStarting = false
        if (cameraActive) {
            startCameraIfReady()
        }
    }

    private fun startCameraIfReady() {
        if (!cameraActive || !::previewView.isInitialized) return
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
        if (!cameraActive || !detectionEnabled || !detector.isReady) {
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
                        feedbackDecision = frameResult.feedbackDecision,
                        sessionSummary = frameResult.sessionSummary.displayText()
                    )
                )
                logPerformanceIfNeeded(frameResult)
            }
        } finally {
            isProcessing.set(false)
            bitmap.recycle()
        }
    }

    private fun setDetectionEnabled(enabled: Boolean) {
        detectionEnabled = enabled
        syncControlsState()
        if (!enabled) {
            assistEngine.reset()
            if (::overlayView.isInitialized) {
                overlayView.update(emptyList(), null, null)
            }
            renderUi(Guidance.paused())
        } else {
            renderUi(Guidance.waiting(detector.statusMessage))
            startCameraIfReady()
        }
    }

    private fun setSpeechEnabled(enabled: Boolean) {
        feedbackController.speechEnabled = enabled
        userPreferences.setSpeechEnabled(enabled)
        syncControlsState()
    }

    private fun setVibrationEnabled(enabled: Boolean) {
        feedbackController.vibrationEnabled = enabled
        userPreferences.setVibrationEnabled(enabled)
        syncControlsState()
    }

    private fun setCareModeEnabled(enabled: Boolean) {
        careModeEnabled = enabled
        userPreferences.setCareModeEnabled(enabled)
        if (::overlayView.isInitialized) {
            overlayView.setCareMode(enabled)
        }
        syncControlsState()
    }

    private fun setDebugVisible(visible: Boolean) {
        debugVisible = visible
        syncControlsState()
    }

    private fun setAlertProfile(profile: AlertProfile) {
        alertProfile = profile
        userPreferences.setAlertProfile(profile)
        syncControlsState()
    }

    private fun syncControlsState() {
        controlsState = AssistControlsUiState(
            detectionEnabled = detectionEnabled,
            speechEnabled = feedbackController.speechEnabled,
            vibrationEnabled = feedbackController.vibrationEnabled,
            careModeEnabled = careModeEnabled,
            debugVisible = debugVisible,
            alertProfile = alertProfile
        )
    }

    private fun renderUi(snapshot: CameraGuidanceUiState) {
        guidanceState = snapshot
        modelStatus = if (::detector.isInitialized) detector.statusMessage else modelStatus
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

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private object Guidance {
        fun initial(modelStatus: String): CameraGuidanceUiState {
            return CameraGuidanceUiState(
                title = "初始化中",
                detail = "正在准备本地检测模型",
                targetLine = "模型状态：$modelStatus",
                careTitle = "正在准备",
                careDetail = "识别模型正在启动，请稍等",
                careTargetLine = "进入手机摄像头后会开始观察前方",
                debugText = "模型状态：$modelStatus",
                titleColor = Color.WHITE,
                statusBadge = "准备中",
                badgeColor = Color.rgb(206, 221, 235),
                badgeTextColor = Color.rgb(10, 22, 32),
                careAccessibilitySummary = "正在准备，识别模型正在启动",
                accessibilitySummary = "初始化中，正在准备本地检测模型",
                accessibilityKey = "initial"
            )
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
            sessionSummary: String
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
                "提醒模式：${profile.displayName} / 反馈：${feedbackDecision.reason.displayText}\n" +
                sessionSummary
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
