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
import android.widget.CompoundButton
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.Switch
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.linnan.blindassist.feedback.FeedbackController
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskAnalyzer
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskStabilizer
import com.linnan.blindassist.ui.DetectionOverlayView
import com.linnan.blindassist.vision.TfliteYoloDetector
import com.linnan.blindassist.vision.toArgbBitmap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: DetectionOverlayView
    private lateinit var riskTitleText: TextView
    private lateinit var riskDetailText: TextView
    private lateinit var targetText: TextView
    private lateinit var debugToggleText: TextView
    private lateinit var debugText: TextView
    private lateinit var detector: TfliteYoloDetector
    private lateinit var riskAnalyzer: RiskAnalyzer
    private lateinit var riskStabilizer: RiskStabilizer
    private lateinit var feedbackController: FeedbackController
    private lateinit var cameraExecutor: ExecutorService

    private val isProcessing = AtomicBoolean(false)
    private var detectionEnabled = true
    private var frameCount = 0
    private var fpsWindowStartMs = System.currentTimeMillis()
    private var currentFps = 0f
    private var lastPerfLogAtMs = 0L
    private var debugVisible = false

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
        riskAnalyzer = RiskAnalyzer()
        riskStabilizer = RiskStabilizer()
        feedbackController = FeedbackController(this)
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
        val root = FrameLayout(this)

        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
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
        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(16), dp(18), dp(18))
            background = roundedBackground(Color.argb(232, 12, 17, 22), dp(18).toFloat())
        }

        riskTitleText = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 28f
            typeface = Typeface.DEFAULT_BOLD
            includeFontPadding = false
            gravity = Gravity.CENTER_VERTICAL
            text = "初始化中"
        }
        riskDetailText = TextView(this).apply {
            setTextColor(Color.rgb(230, 235, 241))
            textSize = 17f
            includeFontPadding = true
        }
        targetText = TextView(this).apply {
            setTextColor(Color.rgb(183, 195, 207))
            textSize = 14f
            includeFontPadding = true
        }

        val switches = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(10), 0, dp(4))
        }

        val detectSwitch = makeSwitch("检测", true) { _, checked ->
            detectionEnabled = checked
            if (!checked) {
                riskStabilizer.reset()
                overlayView.update(emptyList(), null, null)
                renderUi(UiSnapshot.paused())
            } else {
                renderUi(UiSnapshot.waiting(detector.statusMessage))
            }
        }
        val speechSwitch = makeSwitch("语音", true) { button, checked ->
            feedbackController.speechEnabled = checked
            updateSwitchDescription("语音提醒", button, checked)
        }
        val vibrationSwitch = makeSwitch("震动", true) { button, checked ->
            feedbackController.vibrationEnabled = checked
            updateSwitchDescription("震动提醒", button, checked)
        }

        updateSwitchDescription("目标检测", detectSwitch, true)
        updateSwitchDescription("语音提醒", speechSwitch, true)
        updateSwitchDescription("震动提醒", vibrationSwitch, true)

        switches.addView(detectSwitch, switchParams())
        switches.addView(speechSwitch, switchParams())
        switches.addView(vibrationSwitch, switchParams())

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

        panel.addView(riskTitleText)
        panel.addView(riskDetailText)
        panel.addView(targetText)
        panel.addView(switches)
        panel.addView(debugToggleText)
        panel.addView(debugText)
        return panel
    }

    private fun makeSwitch(
        label: String,
        checked: Boolean,
        listener: CompoundButton.OnCheckedChangeListener
    ): Switch {
        return Switch(this).apply {
            text = label
            textSize = 15f
            setTextColor(Color.WHITE)
            isChecked = checked
            minimumHeight = dp(48)
            setPadding(dp(4), dp(8), dp(4), dp(8))
            setOnCheckedChangeListener { button, enabled ->
                updateSwitchDescription(label, button, enabled)
                listener.onCheckedChanged(button, enabled)
            }
        }
    }

    private fun switchParams(): LinearLayout.LayoutParams {
        return LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
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
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT))
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
            val rawRisk = riskAnalyzer.analyze(detections, frameSize)
            val stableRisk = riskStabilizer.update(rawRisk)
            val fps = updateFps()
            logPerformanceIfNeeded(detections.size, frameSize, fps)
            runOnUiThread {
                overlayView.update(detections, frameSize, stableRisk)
                renderUi(UiSnapshot.fromRisk(stableRisk, detections.size, detector, fps))
                feedbackController.notify(stableRisk)
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
        riskTitleText.text = snapshot.title
        riskTitleText.setTextColor(snapshot.titleColor)
        riskDetailText.text = snapshot.detail
        targetText.text = snapshot.targetLine
        riskTitleText.contentDescription = snapshot.accessibilitySummary
        riskDetailText.contentDescription = snapshot.detail
        targetText.contentDescription = snapshot.targetLine
        debugText.text = snapshot.debugText
        updateDebugVisibility()
    }

    private fun updateDebugVisibility() {
        debugText.visibility = if (debugVisible) View.VISIBLE else View.GONE
        debugToggleText.text = if (debugVisible) "调试信息 ▾" else "调试信息 ▸"
        debugToggleText.contentDescription = if (debugVisible) "收起调试信息" else "展开调试信息"
    }

    private fun updateSwitchDescription(label: String, button: CompoundButton, enabled: Boolean) {
        val state = if (enabled) "已开启" else "已关闭"
        button.contentDescription = "$label，$state"
    }

    private fun roundedBackground(color: Int, radius: Float): GradientDrawable {
        return GradientDrawable().apply {
            setColor(color)
            cornerRadius = radius
        }
    }

    private fun logPerformanceIfNeeded(count: Int, frameSize: FrameSize, fps: Float) {
        val now = System.currentTimeMillis()
        if (now - lastPerfLogAtMs < PERF_LOG_INTERVAL_MS) return
        lastPerfLogAtMs = now
        Log.i(
            PERF_TAG,
            "frame=${frameSize.width}x${frameSize.height}, count=$count, " +
                "total=${detector.lastTotalDetectMs}ms, " +
                "pre=${detector.lastPreprocessMs}ms, " +
                "infer=${detector.lastInferenceMs}ms, " +
                "post=${detector.lastPostprocessMs}ms, " +
                "fps=${"%.1f".format(fps)}, status=${detector.statusMessage}"
        )
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
        val debugText: String,
        val titleColor: Int,
        val accessibilitySummary: String
    ) {
        companion object {
            fun initial(modelStatus: String): UiSnapshot {
                return UiSnapshot(
                    title = "初始化中",
                    detail = "正在准备相机和本地检测模型",
                    targetLine = "模型状态：$modelStatus",
                    debugText = "模型状态：$modelStatus",
                    titleColor = Color.WHITE,
                    accessibilitySummary = "初始化中，正在准备相机和本地检测模型"
                )
            }

            fun waiting(modelStatus: String): UiSnapshot {
                return UiSnapshot(
                    title = "检测已开启",
                    detail = "等待实时画面和稳定风险结果",
                    targetLine = "模型状态：$modelStatus",
                    debugText = "模型状态：$modelStatus",
                    titleColor = Color.WHITE,
                    accessibilitySummary = "检测已开启，等待实时画面和稳定风险结果"
                )
            }

            fun paused(): UiSnapshot {
                return UiSnapshot(
                    title = "检测已暂停",
                    detail = "画面保留预览，目标框和风险提醒已清空",
                    targetLine = "可随时重新开启检测",
                    debugText = "检测关闭：不运行目标检测，不触发语音或震动提醒",
                    titleColor = Color.rgb(214, 224, 235),
                    accessibilitySummary = "检测已暂停，目标框和风险提醒已清空"
                )
            }

            fun permissionDenied(): UiSnapshot {
                return UiSnapshot(
                    title = "需要相机权限",
                    detail = "请授予相机权限后再使用实时避障提醒",
                    targetLine = "当前无法启动 CameraX 预览和检测",
                    debugText = "权限状态：CAMERA denied",
                    titleColor = Color.rgb(255, 149, 0),
                    accessibilitySummary = "需要相机权限，请授予相机权限后再使用实时避障提醒"
                )
            }

            fun fromRisk(
                risk: com.linnan.blindassist.risk.RiskResult,
                count: Int,
                detector: TfliteYoloDetector,
                fps: Float
            ): UiSnapshot {
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
                val debug = "FPS：${"%.1f".format(fps)}\n" +
                    "耗时：total ${detector.lastTotalDetectMs}ms / pre ${detector.lastPreprocessMs}ms / " +
                    "infer ${detector.lastInferenceMs}ms / post ${detector.lastPostprocessMs}ms\n" +
                    "模型：${detector.statusMessage}"
                return UiSnapshot(
                    title = title,
                    detail = detail,
                    targetLine = targetLine,
                    debugText = debug,
                    titleColor = colorForLevel(risk.level, risk.proximity),
                    accessibilitySummary = "$title，$detail，$targetLine"
                )
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
