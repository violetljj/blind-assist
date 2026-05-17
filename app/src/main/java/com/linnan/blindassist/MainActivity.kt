package com.linnan.blindassist

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
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
    private lateinit var statusText: TextView
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

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startCamera()
        } else {
            statusText.text = "需要相机权限才能运行"
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
        statusText.text = detector.statusMessage

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
        statusText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 18f
            gravity = Gravity.CENTER_VERTICAL
            text = "初始化中"
        }

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
            setPadding(28, 20, 28, 24)
            setBackgroundColor(0xDD101418.toInt())
        }

        val switches = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }

        val detectSwitch = makeSwitch("检测", true) { _, checked ->
            detectionEnabled = checked
            if (!checked) {
                riskStabilizer.reset()
                overlayView.update(emptyList(), null, null)
                statusText.text = "检测已暂停"
            }
        }
        val speechSwitch = makeSwitch("语音", true) { _, checked ->
            feedbackController.speechEnabled = checked
        }
        val vibrationSwitch = makeSwitch("震动", true) { _, checked ->
            feedbackController.vibrationEnabled = checked
        }

        switches.addView(detectSwitch)
        switches.addView(speechSwitch)
        switches.addView(vibrationSwitch)

        panel.addView(statusText)
        panel.addView(switches)
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
            setTextColor(0xFFFFFFFF.toInt())
            isChecked = checked
            minimumHeight = dp(48)
            setPadding(18, 8, 18, 8)
            setOnCheckedChangeListener(listener)
        }
    }

    private fun bottomPanelParams(): FrameLayout.LayoutParams {
        return FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM
        )
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
                statusText.text = statusFor(
                    level = stableRisk.level,
                    direction = stableRisk.direction,
                    proximity = stableRisk.proximity,
                    urgencyScore = stableRisk.urgencyScore,
                    message = stableRisk.message,
                    targetLabel = stableRisk.sourceDetection?.label,
                    count = detections.size,
                    totalMs = detector.lastTotalDetectMs,
                    preprocessMs = detector.lastPreprocessMs,
                    inferenceMs = detector.lastInferenceMs,
                    postprocessMs = detector.lastPostprocessMs,
                    fps = fps,
                    modelStatus = detector.statusMessage
                )
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

    private fun statusFor(
        level: RiskLevel,
        direction: RiskDirection,
        proximity: ProximityBand,
        urgencyScore: Float,
        message: String,
        targetLabel: String?,
        count: Int,
        totalMs: Long,
        preprocessMs: Long,
        inferenceMs: Long,
        postprocessMs: Long,
        fps: Float,
        modelStatus: String
    ): String {
        val prefix = when (level) {
            RiskLevel.HIGH -> "高风险"
            RiskLevel.MEDIUM -> "中风险"
            RiskLevel.LOW -> "低风险"
            RiskLevel.NONE -> "安全"
        }
        val directionText = when (direction) {
            RiskDirection.LEFT -> "左前"
            RiskDirection.CENTER -> "正前"
            RiskDirection.RIGHT -> "右前"
            RiskDirection.NONE -> "无方向"
        }
        val proximityText = when (proximity) {
            ProximityBand.CRITICAL -> "迫近"
            ProximityBand.NEAR -> "近处"
            ProximityBand.MID -> "中距"
            ProximityBand.FAR -> "远处"
        }
        val targetText = targetLabel ?: "无目标"
        return (
            "$prefix · $proximityText · $directionText · $targetText · $message · " +
                "score ${"%.2f".format(urgencyScore)} · $count 个目标 · total ${totalMs}ms " +
                "(pre ${preprocessMs} / infer ${inferenceMs} / post ${postprocessMs}) · " +
                "%.1f FPS · $modelStatus"
            ).format(fps)
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

    companion object {
        private const val ANALYSIS_WIDTH = 640
        private const val ANALYSIS_HEIGHT = 480
        private const val PERF_LOG_INTERVAL_MS = 1000L
        private const val PERF_TAG = "BlindAssistPerf"
    }
}
