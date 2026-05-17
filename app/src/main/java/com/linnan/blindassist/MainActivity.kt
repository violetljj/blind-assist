package com.linnan.blindassist

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.animation.DecelerateInterpolator
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.feedback.FeedbackController
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.preferences.UserPreferences
import com.linnan.blindassist.session.AssistEngine
import com.linnan.blindassist.session.DetectorMetrics
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.ui.DetectionOverlayView
import com.linnan.blindassist.vision.TfliteYoloDetector
import com.linnan.blindassist.vision.toArgbBitmap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: DetectionOverlayView
    private lateinit var controlPanel: LinearLayout
    private lateinit var statusBadgeText: TextView
    private lateinit var riskTitleText: TextView
    private lateinit var riskDetailText: TextView
    private lateinit var targetText: TextView
    private lateinit var profileToggle: TextView
    private lateinit var careToggle: TextView
    private lateinit var debugToggleText: TextView
    private lateinit var debugText: TextView
    private lateinit var detector: TfliteYoloDetector
    private lateinit var assistEngine: AssistEngine
    private lateinit var feedbackController: FeedbackController
    private lateinit var userPreferences: UserPreferences
    private lateinit var cameraExecutor: ExecutorService

    private val isProcessing = AtomicBoolean(false)
    private var detectionEnabled = true
    private var frameCount = 0
    private var fpsWindowStartMs = System.currentTimeMillis()
    private var currentFps = 0f
    private var lastPerfLogAtMs = 0L
    private var debugVisible = false
    private var careModeEnabled = false
    private var alertProfile = AlertProfile.STANDARD
    private var lastRenderedTitle = ""
    private var lastAccessibilityKey = ""
    private var latestSnapshot: UiSnapshot? = null

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startCamera()
        } else {
            renderUi(UiSnapshot.permissionDenied())
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        detector = TfliteYoloDetector(this)
        assistEngine = AssistEngine()
        feedbackController = FeedbackController(this)
        userPreferences = UserPreferences(this)
        val savedPreferences = userPreferences.load()
        feedbackController.speechEnabled = savedPreferences.speechEnabled
        feedbackController.vibrationEnabled = savedPreferences.vibrationEnabled
        careModeEnabled = savedPreferences.careModeEnabled
        alertProfile = savedPreferences.alertProfile
        cameraExecutor = Executors.newSingleThreadExecutor()

        setContentView(buildContentView())
        renderUi(UiSnapshot.initial(detector.statusMessage))

        if (hasCameraPermission()) {
            startCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        detector.close()
        feedbackController.shutdown()
    }

    private fun buildContentView(): View {
        val root = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
        }

        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
        overlayView = DetectionOverlayView(this)

        root.addView(
            previewView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )
        root.addView(
            overlayView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )
        root.addView(buildControlPanel(), bottomPanelParams())
        return root
    }

    private fun buildControlPanel(): View {
        controlPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(12), dp(16), dp(14))
            background = panelBackground(careModeEnabled)
            elevation = dp(12).toFloat()
            alpha = 0f
            translationY = dp(18).toFloat()
            animate()
                .alpha(1f)
                .translationY(0f)
                .setDuration(260L)
                .setInterpolator(DecelerateInterpolator())
                .start()
        }

        val accentBar = View(this).apply {
            background = roundedBackground(Color.rgb(99, 230, 166), dp(2).toFloat())
        }
        controlPanel.addView(
            accentBar,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(3)).apply {
                bottomMargin = dp(9)
            }
        )

        controlPanel.addView(buildHeaderRow())

        statusBadgeText = TextView(this).apply {
            text = "系统准备"
            textSize = 13f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.rgb(6, 24, 18))
            gravity = Gravity.CENTER
            includeFontPadding = false
            setPadding(dp(12), dp(7), dp(12), dp(7))
            background = roundedBackground(Color.rgb(160, 255, 215), dp(14).toFloat())
            contentDescription = "当前状态：系统准备"
        }
        controlPanel.addView(
            statusBadgeText,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                topMargin = dp(2)
                bottomMargin = dp(8)
            }
        )

        riskTitleText = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 27f
            typeface = Typeface.DEFAULT_BOLD
            includeFontPadding = false
            gravity = Gravity.CENTER_VERTICAL
            text = "初始化中"
            letterSpacing = 0f
        }
        riskDetailText = TextView(this).apply {
            setTextColor(Color.rgb(230, 235, 241))
            textSize = 16f
            includeFontPadding = true
            setLineSpacing(0f, 1.08f)
        }
        targetText = TextView(this).apply {
            setTextColor(Color.rgb(183, 195, 207))
            textSize = 14f
            includeFontPadding = true
            setLineSpacing(0f, 1.06f)
        }

        val detectToggle = makeToggle("检测", true) { _, checked ->
            detectionEnabled = checked
            if (!checked) {
                assistEngine.reset()
                overlayView.update(emptyList(), null, null)
                renderUi(UiSnapshot.paused())
            } else {
                renderUi(UiSnapshot.waiting(detector.statusMessage))
            }
        }
        val speechToggle = makeToggle("语音", feedbackController.speechEnabled) { button, checked ->
            feedbackController.speechEnabled = checked
            userPreferences.setSpeechEnabled(checked)
            updateToggleDescription("语音提醒", button, checked)
        }
        val vibrationToggle = makeToggle("震动", feedbackController.vibrationEnabled) { button, checked ->
            feedbackController.vibrationEnabled = checked
            userPreferences.setVibrationEnabled(checked)
            updateToggleDescription("震动提醒", button, checked)
        }
        profileToggle = makeProfileToggle()
        careToggle = makeToggle("关怀", careModeEnabled) { button, checked ->
            careModeEnabled = checked
            lastAccessibilityKey = ""
            userPreferences.setCareModeEnabled(checked)
            updateToggleDescription("关怀模式", button, checked)
            applyCareModeUi()
            renderUi(
                latestSnapshot ?: if (detectionEnabled) {
                    UiSnapshot.waiting(detector.statusMessage)
                } else {
                    UiSnapshot.paused()
                }
            )
        }

        updateToggleDescription("目标检测", detectToggle, true)
        updateToggleDescription("语音提醒", speechToggle, feedbackController.speechEnabled)
        updateToggleDescription("震动提醒", vibrationToggle, feedbackController.vibrationEnabled)
        updateToggleDescription("关怀模式", careToggle, careModeEnabled)
        updateProfileToggle()

        debugToggleText = TextView(this).apply {
            setTextColor(Color.rgb(214, 224, 235))
            textSize = 15f
            gravity = Gravity.CENTER_VERTICAL
            minimumHeight = dp(48)
            text = "调试信息 ▸"
            contentDescription = "展开调试信息"
            setOnClickListener {
                debugVisible = !debugVisible
                updateDebugVisibility()
            }
        }
        debugText = TextView(this).apply {
            setTextColor(Color.rgb(168, 181, 194))
            textSize = 13f
            visibility = View.GONE
            setLineSpacing(0f, 1.08f)
        }

        controlPanel.addView(riskTitleText)
        controlPanel.addView(riskDetailText)
        controlPanel.addView(targetText)
        controlPanel.addView(buildControlRow(detectToggle, speechToggle, vibrationToggle))
        controlPanel.addView(buildControlRow(profileToggle, careToggle))
        controlPanel.addView(debugToggleText)
        controlPanel.addView(debugText)
        applyCareModeUi()
        return controlPanel
    }

    private fun buildHeaderRow(): View {
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val brand = TextView(this).apply {
            text = "BlindAssist"
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.rgb(235, 242, 248))
            includeFontPadding = false
            contentDescription = "BlindAssist 实时避障"
        }
        val caption = TextView(this).apply {
            text = "实时避障工作台"
            textSize = 13f
            setTextColor(Color.rgb(158, 173, 188))
            gravity = Gravity.RIGHT
            includeFontPadding = false
        }
        header.addView(brand, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        header.addView(caption, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        return header
    }

    private fun buildControlRow(vararg toggles: TextView): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(10), 0, dp(2))
            toggles.forEachIndexed { index, toggle ->
                addView(toggle, toggleParams(index == toggles.lastIndex))
            }
        }
    }

    private fun makeToggle(
        label: String,
        checked: Boolean,
        listener: (TextView, Boolean) -> Unit
    ): TextView {
        return TextView(this).apply {
            tag = checked
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            includeFontPadding = false
            minimumHeight = dp(48)
            setPadding(dp(8), 0, dp(8), 0)
            updateToggleAppearance(this, label, checked)
            setOnClickListener {
                val enabled = !(tag as Boolean)
                tag = enabled
                updateToggleAppearance(this, label, enabled)
                updateToggleDescription(label, this, enabled)
                listener(this, enabled)
            }
        }
    }

    private fun makeProfileToggle(): TextView {
        return TextView(this).apply {
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            includeFontPadding = false
            minimumHeight = dp(48)
            setPadding(dp(8), 0, dp(8), 0)
            setOnClickListener {
                alertProfile = alertProfile.next()
                userPreferences.setAlertProfile(alertProfile)
                updateProfileToggle()
                renderUi(
                    latestSnapshot ?: if (detectionEnabled) {
                        UiSnapshot.waiting(detector.statusMessage)
                    } else {
                        UiSnapshot.paused()
                    }
                )
            }
        }
    }

    private fun toggleParams(isLast: Boolean): LinearLayout.LayoutParams {
        return LinearLayout.LayoutParams(0, dp(48), 1f).apply {
            if (!isLast) rightMargin = dp(8)
        }
    }

    private fun bottomPanelParams(): FrameLayout.LayoutParams {
        return FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM
        ).apply {
            leftMargin = dp(12)
            rightMargin = dp(12)
            bottomMargin = dp(12)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
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

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(bitmap: Bitmap) {
        if (!detectionEnabled || !detector.isReady) {
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
                    UiSnapshot.fromRisk(
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

    private fun renderUi(snapshot: UiSnapshot) {
        latestSnapshot = snapshot
        val title = if (careModeEnabled) snapshot.careTitle else snapshot.title
        val detail = if (careModeEnabled) snapshot.careDetail else snapshot.detail
        val targetLine = if (careModeEnabled) snapshot.careTargetLine else snapshot.targetLine
        animateTitleIfNeeded(title)
        riskTitleText.text = title
        riskTitleText.setTextColor(snapshot.titleColor)
        riskDetailText.text = detail
        targetText.text = targetLine
        val accessibilityKey = snapshot.accessibilityKey
        if (lastAccessibilityKey != accessibilityKey) {
            riskTitleText.contentDescription = if (careModeEnabled) {
                snapshot.careAccessibilitySummary
            } else {
                snapshot.accessibilitySummary
            }
            lastAccessibilityKey = accessibilityKey
        }
        riskDetailText.contentDescription = detail
        targetText.contentDescription = targetLine
        statusBadgeText.text = snapshot.statusBadge
        statusBadgeText.setTextColor(snapshot.badgeTextColor)
        statusBadgeText.background = roundedBackground(snapshot.badgeColor, dp(14).toFloat())
        statusBadgeText.contentDescription = "当前状态：${snapshot.statusBadge}"
        debugText.text = snapshot.debugText
        updateDebugVisibility()
    }

    private fun animateTitleIfNeeded(title: String) {
        if (lastRenderedTitle == title) return
        lastRenderedTitle = title
        riskTitleText.animate().cancel()
        riskTitleText.alpha = 0.7f
        riskTitleText.translationY = dp(4).toFloat()
        riskTitleText.animate()
            .alpha(1f)
            .translationY(0f)
            .setDuration(180L)
            .setInterpolator(DecelerateInterpolator())
            .start()
    }

    private fun updateDebugVisibility() {
        debugText.visibility = if (debugVisible) View.VISIBLE else View.GONE
        debugToggleText.text = if (debugVisible) "调试信息 ▾" else "调试信息 ▸"
        debugToggleText.contentDescription = if (debugVisible) "收起调试信息" else "展开调试信息"
    }

    private fun updateToggleDescription(label: String, button: TextView, enabled: Boolean) {
        val state = if (enabled) "已开启" else "已关闭"
        button.contentDescription = "$label，$state"
    }

    private fun updateToggleAppearance(button: TextView, label: String, enabled: Boolean) {
        val state = if (enabled) "开" else "关"
        button.text = "$label $state"
        button.setTextColor(if (enabled) Color.rgb(5, 31, 25) else Color.rgb(226, 232, 238))
        button.background = roundedBackground(
            if (enabled) Color.rgb(128, 244, 198) else Color.argb(118, 52, 64, 74),
            dp(14).toFloat()
        )
    }

    private fun updateProfileToggle() {
        val next = alertProfile.next()
        profileToggle.text = "模式 ${alertProfile.displayName}"
        profileToggle.contentDescription = "提醒模式，当前${alertProfile.displayName}，点击切换为${next.displayName}"
        profileToggle.setTextColor(Color.rgb(5, 31, 25))
        profileToggle.background = roundedBackground(Color.rgb(152, 218, 255), dp(14).toFloat())
    }

    private fun applyCareModeUi() {
        controlPanel.background = panelBackground(careModeEnabled)
        overlayView.setCareMode(careModeEnabled)
        val titleSize = if (careModeEnabled) 31f else 27f
        val detailSize = if (careModeEnabled) 18f else 16f
        val targetSize = if (careModeEnabled) 15f else 14f
        riskTitleText.textSize = titleSize
        riskDetailText.textSize = detailSize
        targetText.textSize = targetSize
        debugToggleText.visibility = if (careModeEnabled) View.GONE else View.VISIBLE
        debugVisible = if (careModeEnabled) false else debugVisible
        updateDebugVisibility()
    }

    private fun panelBackground(careMode: Boolean): GradientDrawable {
        val colors = if (careMode) {
            intArrayOf(Color.argb(246, 5, 8, 10), Color.argb(246, 16, 23, 28))
        } else {
            intArrayOf(Color.argb(236, 11, 16, 21), Color.argb(232, 20, 28, 36))
        }
        return GradientDrawable(GradientDrawable.Orientation.TOP_BOTTOM, colors).apply {
            cornerRadius = dp(22).toFloat()
            setStroke(dp(1), if (careMode) Color.rgb(255, 224, 102) else Color.argb(130, 105, 128, 148))
        }
    }

    private fun roundedBackground(color: Int, radius: Float): GradientDrawable {
        return GradientDrawable().apply {
            setColor(color)
            cornerRadius = radius
        }
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

    private fun dp(value: Int): Int {
        return (value * resources.displayMetrics.density).toInt()
    }

    private data class UiSnapshot(
        val title: String,
        val detail: String,
        val targetLine: String,
        val careTitle: String,
        val careDetail: String,
        val careTargetLine: String,
        val debugText: String,
        val titleColor: Int,
        val statusBadge: String,
        val badgeColor: Int,
        val badgeTextColor: Int,
        val careAccessibilitySummary: String,
        val accessibilitySummary: String,
        val accessibilityKey: String
    ) {
        companion object {
            fun initial(modelStatus: String): UiSnapshot {
                return UiSnapshot(
                    title = "初始化中",
                    detail = "正在准备相机和本地检测模型",
                    targetLine = "模型状态：$modelStatus",
                    careTitle = "正在准备",
                    careDetail = "相机和识别模型正在启动，请稍等",
                    careTargetLine = "准备完成后会自动开始观察前方",
                    debugText = "模型状态：$modelStatus",
                    titleColor = Color.WHITE,
                    statusBadge = "准备中",
                    badgeColor = Color.rgb(206, 221, 235),
                    badgeTextColor = Color.rgb(10, 22, 32),
                    careAccessibilitySummary = "正在准备，相机和识别模型正在启动",
                    accessibilitySummary = "初始化中，正在准备相机和本地检测模型",
                    accessibilityKey = "initial"
                )
            }

            fun waiting(modelStatus: String): UiSnapshot {
                return UiSnapshot(
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

            fun paused(): UiSnapshot {
                return UiSnapshot(
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

            fun permissionDenied(): UiSnapshot {
                return UiSnapshot(
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

            fun fromRisk(
                rawRisk: com.linnan.blindassist.risk.RiskResult,
                stableRisk: com.linnan.blindassist.risk.RiskResult,
                count: Int,
                detector: TfliteYoloDetector,
                fps: Float,
                profile: AlertProfile,
                feedbackDecision: FeedbackDecision,
                sessionSummary: String
            ): UiSnapshot {
                val risk = stableRisk
                val levelText = levelText(risk.level)
                val proximityText = proximityText(risk.proximity)
                val directionText = directionText(risk.direction)
                val targetText = risk.sourceDetection?.label ?: "无主要目标"
                val title = when (risk.level) {
                    RiskLevel.HIGH -> "$levelText：$directionText"
                    RiskLevel.MEDIUM -> "$levelText：$directionText"
                    RiskLevel.LOW -> "$levelText"
                    RiskLevel.NONE -> "安全观察中"
                }
                val detail = "$proximityText · ${risk.message}"
                val targetLine = "目标：$targetText · 共 $count 个 · 紧急度 ${"%.2f".format(risk.urgencyScore)}"
                val careTitle = careTitle(risk.level, risk.direction, risk.proximity)
                val careDetail = careDetail(risk.level, risk.direction, risk.proximity, risk.message)
                val careTargetLine = if (risk.level == RiskLevel.NONE) {
                    "没有发现需要立即提醒的障碍"
                } else {
                    "主要目标：$targetText"
                }
                val debug = "FPS：${"%.1f".format(fps)}\n" +
                    "耗时：total ${detector.lastTotalDetectMs}ms / pre ${detector.lastPreprocessMs}ms / " +
                    "infer ${detector.lastInferenceMs}ms / post ${detector.lastPostprocessMs}ms\n" +
                    "模型：${detector.statusMessage}\n" +
                    "最近风险判定：原始 ${riskSummaryText(rawRisk)} / 稳定 ${riskSummaryText(stableRisk)}\n" +
                    "提醒模式：${profile.displayName} / 反馈：${feedbackDecision.reason.displayText}\n" +
                    sessionSummary
                return UiSnapshot(
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
                    careAccessibilitySummary = "$careTitle，$careDetail，$careTargetLine",
                    accessibilitySummary = "$title，$detail，$targetLine",
                    accessibilityKey = "${risk.level}-${risk.direction}-${risk.proximity}"
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

            private fun careDetail(
                level: RiskLevel,
                direction: RiskDirection,
                proximity: ProximityBand,
                message: String
            ): String {
                return when {
                    proximity == ProximityBand.CRITICAL -> "障碍可能已经很近，请放慢并确认环境"
                    level == RiskLevel.HIGH -> "建议减速，先确认${directionText(direction)}方向"
                    level == RiskLevel.MEDIUM -> "前方有目标，请继续谨慎观察"
                    level == RiskLevel.LOW -> "发现远处或中距目标，暂不触发强提醒"
                    else -> message
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
    }

    companion object {
        private const val ANALYSIS_WIDTH = 640
        private const val ANALYSIS_HEIGHT = 480
        private const val PERF_LOG_INTERVAL_MS = 1000L
        private const val PERF_TAG = "BlindAssistPerf"
    }
}
