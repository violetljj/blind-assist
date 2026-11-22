package com.linnan.blindassist.egoclear.demo

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import com.linnan.blindassist.egoclear.EgoClearSnpeEncoder
import com.linnan.blindassist.egoclear.EgoClearSnpeStep
import com.linnan.blindassist.egoclear.EgoClearStreamingEngine
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Isolated handheld real-time obstacle avoidance demo powered by EgoClear-v10.6 (MoGe-2 Dual-Geom
 * Multi-Range Survival Calibration, with automatic fallback to v10.5) and EgoClear-v6.6 (DAV2-Small)
 * on Qualcomm Hexagon v75 HTP NPU, with instant 1-tap switching between v10.6 (~56 ms NPU),
 * v6.6 (~18 ms NPU), and Dual-NPU ENS(v6.6 + v10.6) (~74 ms NPU).
 */
class EgoClearDemoActivity : Activity(), LifecycleOwner {

    private enum class ModelMode(val shortLabel: String, val spokenLabel: String) {
        V105("v11", "v11 全能避障"),
        DUAL_ENS("双融合", "双主干融合"),
        V66("v6.6", "v6.6 轻量"),
    }

    private lateinit var lifecycleRegistry: LifecycleRegistry
    override val lifecycle: Lifecycle
        get() = lifecycleRegistry

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: EgoClearOverlayView
    private lateinit var telemetryBanner: TextView
    private lateinit var statusMessage: TextView

    private lateinit var btnModel: Button
    private lateinit var btnPause: Button
    private lateinit var btnRange: Button
    private lateinit var btnLens: Button
    private lateinit var btnVoice: Button
    private lateinit var btnHaptic: Button
    private lateinit var btnSens: Button
    private lateinit var btnRate: Button
    private lateinit var btnRecord: Button
    private lateinit var btnReset: Button

    private lateinit var poseTracker: EgoClearPoseTracker
    private lateinit var preprocessor: EgoClearCameraPreprocessor
    private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "EgoClearNpuWorker").apply { priority = Thread.MAX_PRIORITY - 1 }
    }

    @Volatile private var snpeV105: EgoClearSnpeEncoder? = null
    @Volatile private var snpeStepV105: EgoClearSnpeStep? = null
    @Volatile private var engineV105: EgoClearStreamingEngine? = null
    @Volatile private var v10LoadedVer: String = "v11"
    @Volatile private var snpeV66: EgoClearSnpeEncoder? = null
    @Volatile private var snpeStepV66: EgoClearSnpeStep? = null
    @Volatile private var engineV66: EgoClearStreamingEngine? = null
    @Volatile private var activeMode: ModelMode = ModelMode.V105
    @Volatile private var availableModes: List<ModelMode> = listOf(ModelMode.V105)
    @Volatile private var backendLabel: String = "加载中"
    @Volatile private var isPaused: Boolean = false
    @Volatile private var voiceEnabled: Boolean = true
    @Volatile private var hapticEnabled: Boolean = true
    @Volatile private var fullRateMode: Boolean = false
    @Volatile private var sensIndex: Int = 0
    @Volatile private var rangeIndex: Int = 0
    @Volatile private var useUltrawide: Boolean = false
    @Volatile private var minZoomRatio: Float = 1.0f
    @Volatile private var activeZoomRatio: Float = 1.0f
    @Volatile private var resetRequested: Boolean = false

    private val frameBusy = AtomicBoolean(false)
    private var lastFrameStartMs = 0L
    private var lastVibeMs = 0L
    private var lastSpeakMs = 0L
    private var lastDropSpeakMs = 0L
    private var wasAlertActive = false
    private var wasDropActive = false
    private var wasUnknownActive = false
    private var wasBlackoutActive = false
    private var lastSpokenDist = 99f
    private var wasPitchGated = false
    private var lastGravityRotation = -1
    private var frameCounter = 0L
    private var emaFps = 0.0
    private var lastDoneMs = 0L
    private var ensAlertOn = false
    private var ensPrevScore = 0.0f
    private var ensHasPrevScore = false

    // 10-second (50-frame @ 5 Hz) offline handheld replay recorder (.egoclear EGC1 format)
    @Volatile private var recordRequestedFrames: Int = 0
    private var recordStream: BufferedOutputStream? = null
    private var recordFile: File? = null
    private var recordWrittenFrames: Int = 0
    private val recordRgbU8 = ByteArray(EgoClearStreamingEngine.RGB_FLOATS)

    private var tts: TextToSpeech? = null
    @Volatile private var ttsReady = false
    private var vibrator: Vibrator? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var boundCamera: Camera? = null

    private val thresholds = floatArrayOf(0.55f, 0.42f, 0.70f)
    private val thresholdNames = arrayOf("灵敏:中", "灵敏:高", "灵敏:低")
    private val horizonBins = intArrayOf(4, 7, 10, EgoClearStreamingEngine.IDX_4M) // 1.3m, 1.9m, 2.5m, 4.1m
    private val horizonNames = arrayOf("紧距1.3m", "室内1.9m", "日常2.5m", "室外4.1m")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        lifecycleRegistry = LifecycleRegistry(this)
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_CREATE)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        poseTracker = EgoClearPoseTracker(this)
        preprocessor = EgoClearCameraPreprocessor(this)
        vibrator = resolveVibrator()
        initTts()

        setContentView(buildUi())
        loadEngineAsync()

        if (hasCameraPermission()) {
            startCamera()
        } else {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
        }
    }

    override fun onStart() {
        super.onStart()
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_START)
    }

    override fun onResume() {
        super.onResume()
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_RESUME)
        poseTracker.start()
    }

    override fun onPause() {
        poseTracker.stop()
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_PAUSE)
        super.onPause()
    }

    override fun onStop() {
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_STOP)
        super.onStop()
    }

    override fun onDestroy() {
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_DESTROY)
        cameraProvider?.unbindAll()
        tts?.stop()
        tts?.shutdown()
        tts = null
        inferenceExecutor.execute {
            finishReplayRecordingIfOpen()
            runCatching { engineV105?.close() }
            runCatching { snpeStepV105?.close() }
            runCatching { snpeV105?.close() }
            runCatching { engineV66?.close() }
            runCatching { snpeStepV66?.close() }
            runCatching { snpeV66?.close() }
            engineV105 = null
            snpeStepV105 = null
            snpeV105 = null
            engineV66 = null
            snpeStepV66 = null
            snpeV66 = null
        }
        inferenceExecutor.shutdown()
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_CAMERA && grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            showStatus("需要相机权限才能运行实时避障预览")
        }
    }

    private fun buildUi(): View {
        val dp = resources.displayMetrics.density
        val root = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
        }

        previewView = PreviewView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            )
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
        root.addView(previewView)

        overlayView = EgoClearOverlayView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            )
        }
        root.addView(overlayView)

        // Top telemetry pill
        telemetryBanner = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 12.5f
            typeface = Typeface.MONOSPACE
            setPadding((12 * dp).toInt(), (8 * dp).toInt(), (12 * dp).toInt(), (8 * dp).toInt())
            background = GradientDrawable().apply {
                setColor(Color.argb(195, 14, 18, 26))
                cornerRadius = 12f * dp
                setStroke((1 * dp).toInt(), Color.argb(110, 255, 255, 255))
            }
            text = "EgoClear-v11 · 正在初始化 Hexagon v75 NPU…"
            accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.TOP,
            ).apply {
                setMargins((12 * dp).toInt(), (36 * dp).toInt(), (12 * dp).toInt(), 0)
            }
        }
        root.addView(telemetryBanner)

        // Center diagnostic / loading message
        statusMessage = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 14f
            gravity = Gravity.CENTER
            setPadding((16 * dp).toInt(), (14 * dp).toInt(), (16 * dp).toInt(), (14 * dp).toInt())
            background = GradientDrawable().apply {
                setColor(Color.argb(215, 20, 24, 34))
                cornerRadius = 14f * dp
            }
            text = "正在加载 EgoClear-v11 / v6.6 NPU 模型…"
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.CENTER,
            ).apply {
                setMargins((24 * dp).toInt(), 0, (24 * dp).toInt(), 0)
            }
        }
        root.addView(statusMessage)

        // Bottom accessible controls bar (every button >= 48dp tall)
        val scroll = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM,
            ).apply {
                setMargins((8 * dp).toInt(), 0, (8 * dp).toInt(), (20 * dp).toInt())
            }
        }
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding((4 * dp).toInt(), (4 * dp).toInt(), (4 * dp).toInt(), (4 * dp).toInt())
        }

        fun makeButton(label: String, desc: String, onClick: () -> Unit): Button {
            return Button(this).apply {
                text = label
                contentDescription = desc
                isAllCaps = false
                setTextColor(Color.WHITE)
                textSize = 13f
                minHeight = (48 * dp).toInt()
                minimumHeight = (48 * dp).toInt()
                minWidth = (74 * dp).toInt()
                setPadding((12 * dp).toInt(), (6 * dp).toInt(), (12 * dp).toInt(), (6 * dp).toInt())
                background = GradientDrawable().apply {
                    setColor(Color.argb(215, 32, 42, 58))
                    cornerRadius = 12f * dp
                    setStroke((1 * dp).toInt(), Color.argb(130, 255, 255, 255))
                }
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    (48 * dp).toInt(),
                ).apply {
                    setMargins((4 * dp).toInt(), 0, (4 * dp).toInt(), 0)
                }
                setOnClickListener { onClick() }
            }
        }

        btnModel = makeButton("模型:加载", "切换视觉模型：全能主力、双主干融合或 v6.6 轻量档") {
            val modes = availableModes
            if (modes.size > 1) {
                val nextIdx = (modes.indexOf(activeMode) + 1) % modes.size
                activeMode = modes[nextIdx]
                val label = modeDisplayLabel(activeMode)
                backendLabel = "$label NPU"
                btnModel.text = "模型:$label"
                btnModel.contentDescription = "当前视觉模型：$label，双击切换"
                updateButtonStyle(btnModel, active = activeMode != ModelMode.V66)
                resetRequested = true
                telemetryBanner.text = "已切换至 $label"
                if (voiceEnabled && ttsReady) {
                    runCatching { tts?.speak("已切换$label", TextToSpeech.QUEUE_FLUSH, null, "egoclear_mode") }
                }
            }
        }
        btnPause = makeButton("暂停", "暂停或继续避障检测") {
            isPaused = !isPaused
            btnPause.text = if (isPaused) "继续" else "暂停"
            updateButtonStyle(btnPause, active = !isPaused)
        }
        btnRange = makeButton(horizonNames[rangeIndex], "切换预警距离：紧距1.3米、室内1.9米、日常2.5米、室外4.1米") {
            rangeIndex = (rangeIndex + 1) % horizonBins.size
            btnRange.text = horizonNames[rangeIndex]
            resetRequested = true
        }
        btnLens = makeButton("标准1.0x", "切换标准主摄或广角焦段") {
            useUltrawide = !useUltrawide
            applyZoomPreference()
        }
        btnVoice = makeButton("语音:开", "开关中文障碍语音播报") {
            voiceEnabled = !voiceEnabled
            btnVoice.text = if (voiceEnabled) "语音:开" else "语音:关"
            updateButtonStyle(btnVoice, active = voiceEnabled)
            if (!voiceEnabled) tts?.stop()
        }
        btnHaptic = makeButton("震动:开", "开关障碍近距震动提醒") {
            hapticEnabled = !hapticEnabled
            btnHaptic.text = if (hapticEnabled) "震动:开" else "震动:关"
            updateButtonStyle(btnHaptic, active = hapticEnabled)
        }
        btnSens = makeButton(thresholdNames[sensIndex], "切换预警灵敏度档位") {
            sensIndex = (sensIndex + 1) % thresholds.size
            btnSens.text = thresholdNames[sensIndex]
        }
        btnRate = makeButton("5 Hz", "切换 5 赫兹稳态或全速模式") {
            fullRateMode = !fullRateMode
            btnRate.text = if (fullRateMode) "全速" else "5 Hz"
            updateButtonStyle(btnRate, active = fullRateMode)
        }
        btnRecord = makeButton("录回放", "录制 10 秒原始画面与位姿用于离线回归评测") {
            if (recordRequestedFrames <= 0) {
                recordRequestedFrames = REPLAY_CLIP_FRAMES
                btnRecord.text = "录制中…"
                updateButtonStyle(btnRecord, active = true)
            }
        }
        btnReset = makeButton("重置", "清空时序记忆并重置步态里程") {
            resetRequested = true
            poseTracker.resetOdometry()
            telemetryBanner.text = "已重置 BEV 时序记忆"
        }

        updateButtonStyle(btnModel, active = true)
        updateButtonStyle(btnPause, active = true)
        updateButtonStyle(btnRange, active = true)
        updateButtonStyle(btnLens, active = false)
        updateButtonStyle(btnVoice, active = true)
        updateButtonStyle(btnHaptic, active = true)
        updateButtonStyle(btnRate, active = false)
        updateButtonStyle(btnRecord, active = false)

        row.addView(btnModel)
        row.addView(btnRange)
        row.addView(btnLens)
        row.addView(btnVoice)
        row.addView(btnHaptic)
        row.addView(btnSens)
        row.addView(btnRecord)
        row.addView(btnPause)
        row.addView(btnRate)
        row.addView(btnReset)
        scroll.addView(row)
        root.addView(scroll)

        return root
    }

    private fun applyZoomPreference() {
        val cam = boundCamera ?: return
        val target = if (useUltrawide && minZoomRatio < 0.95f) {
            minZoomRatio.coerceAtLeast(0.6f)
        } else {
            1.0f
        }
        activeZoomRatio = target
        cam.cameraControl.setZoomRatio(target)
        resetRequested = true
        btnLens.text = if (target < 0.95f) String.format(Locale.US, "广角%.1fx", target) else "标准1.0x"
        updateButtonStyle(btnLens, active = target < 0.95f)
    }

    private fun updateButtonStyle(btn: Button, active: Boolean) {
        val bg = btn.background as? GradientDrawable ?: return
        bg.setColor(
            if (active) Color.argb(225, 22, 118, 92) else Color.argb(215, 42, 46, 56),
        )
    }

    private fun stageToInternalIfNeeded(src: File, internalDir: File): File {
        val dst = File(internalDir, src.name)
        if (dst.isFile && dst.length() == src.length() && dst.lastModified() >= src.lastModified()) return dst
        runOnUiThread { showStatus("首次启动：正在缓存 ${src.name} 到本地高速存储…") }
        val tmp = File(internalDir, "${src.name}.tmp")
        try {
            src.inputStream().buffered(256 * 1024).use { inp ->
                tmp.outputStream().buffered(256 * 1024).use { out ->
                    inp.copyTo(out, 256 * 1024)
                }
            }
            if (dst.exists()) dst.delete()
            check(tmp.renameTo(dst)) { "Failed to stage ${src.name} to ${dst.absolutePath}" }
        } finally {
            if (tmp.exists()) tmp.delete()
        }
        return dst
    }

    private fun loadEngineAsync() {
        inferenceExecutor.execute {
            val extDir = File(getExternalFilesDir("models"), "").apply { mkdirs() }
            val intDir = File(filesDir, "models").apply { mkdirs() }

            fun resolveModel(name: String): File? {
                val ext = File(extDir, name)
                val internal = File(intDir, name)
                return when {
                    ext.isFile -> stageToInternalIfNeeded(ext, intDir)
                    internal.isFile -> internal
                    else -> null
                }
            }

            try {
                // 1. Load EgoClear-v14 Hyper-Omni (preferred) -> fallback v13 -> v12 -> v11 -> v10.6 -> v10.5;
                //    prefer full-NPU Hexagon v75 Step DLC (~6.2-8.8 ms!) over CPU ORT Step when available!
                val v14Dlc = resolveModel(V14_DLC_NAME)
                val v14StepDlc = resolveModel(V14_STEP_DLC_NAME)
                val v14StepOnnx = resolveModel(V14_STEP_NAME)
                val v13Dlc = resolveModel(V13_DLC_NAME)
                val v13StepDlc = resolveModel(V13_STEP_DLC_NAME)
                val v13StepOnnx = resolveModel(V13_STEP_NAME)
                val v12Dlc = resolveModel(V12_DLC_NAME)
                val v12StepDlc = resolveModel(V12_STEP_DLC_NAME)
                val v12StepOnnx = resolveModel(V12_STEP_NAME)
                val v11Dlc = resolveModel(V11_DLC_NAME)
                val v11StepDlc = resolveModel(V11_STEP_DLC_NAME)
                val v11StepOnnx = resolveModel(V11_STEP_NAME)
                val v106Dlc = resolveModel(V106_DLC_NAME)
                val v106StepDlc = resolveModel(V106_STEP_DLC_NAME)
                val v106StepOnnx = resolveModel(V106_STEP_NAME)
                val v10Dlc: File?
                val v10StepDlc: File?
                val v10StepOnnx: File?
                val v10Ver: String
                val v10HasOmni: Boolean
                if (v14Dlc != null && (v14StepDlc != null || v14StepOnnx != null)) {
                    v10Dlc = v14Dlc
                    v10StepDlc = v14StepDlc
                    v10StepOnnx = v14StepOnnx
                    v10Ver = "v14"
                    v10HasOmni = true
                } else if (v13Dlc != null && (v13StepDlc != null || v13StepOnnx != null)) {
                    v10Dlc = v13Dlc
                    v10StepDlc = v13StepDlc
                    v10StepOnnx = v13StepOnnx
                    v10Ver = "v13"
                    v10HasOmni = true
                } else if (v12Dlc != null && (v12StepDlc != null || v12StepOnnx != null)) {
                    v10Dlc = v12Dlc
                    v10StepDlc = v12StepDlc
                    v10StepOnnx = v12StepOnnx
                    v10Ver = "v12"
                    v10HasOmni = true
                } else if (v11Dlc != null && (v11StepDlc != null || v11StepOnnx != null)) {
                    v10Dlc = v11Dlc
                    v10StepDlc = v11StepDlc
                    v10StepOnnx = v11StepOnnx
                    v10Ver = "v11"
                    v10HasOmni = true
                } else if (v106Dlc != null && (v106StepDlc != null || v106StepOnnx != null)) {
                    v10Dlc = v106Dlc
                    v10StepDlc = v106StepDlc
                    v10StepOnnx = v106StepOnnx
                    v10Ver = "v10.6"
                    v10HasOmni = false
                } else {
                    v10Dlc = resolveModel(V105_DLC_NAME)
                    v10StepDlc = null
                    v10StepOnnx = resolveModel(V105_STEP_NAME)
                    v10Ver = "v10.5"
                    v10HasOmni = false
                }
                if (v10Dlc != null && (v10StepDlc != null || v10StepOnnx != null)) {
                    val snpe = EgoClearSnpeEncoder(
                        apkContext = applicationContext,
                        cacheContext = applicationContext,
                        dlcFile = v10Dlc,
                        auxOutputName = EgoClearSnpeEncoder.AUX_GEOM_CUES,
                        f2dSize = EgoClearSnpeEncoder.V10_F2D_SIZE,
                        auxSize = EgoClearSnpeEncoder.V10_GEOM_SIZE,
                    )
                    val snpeStep = v10StepDlc?.let { stDlc ->
                        runCatching {
                            EgoClearSnpeStep(
                                apkContext = applicationContext,
                                cacheContext = applicationContext,
                                dlcFile = stDlc,
                                auxPrefix = "geom",
                                hasOmni = v10HasOmni,
                            )
                        }.onFailure { err ->
                            Log.w(TAG, "HTP step init fallback to CPU ORT: ${err.message}")
                        }.getOrNull()
                    }
                    val eng = EgoClearStreamingEngine(
                        encoderModel = null,
                        stepModel = if (snpeStep != null) null else v10StepOnnx,
                        backend = EgoClearStreamingEngine.Backend.HTP_V75_NPU,
                        stepBackend = if (snpeStep != null) EgoClearStreamingEngine.Backend.HTP_V75_NPU else EgoClearStreamingEngine.Backend.CPU_4T,
                        snpeEncoder = snpe,
                        profile = EgoClearStreamingEngine.ModelProfile.V10_MOGE2,
                        snpeStep = snpeStep,
                    )
                    snpeV105 = snpe
                    snpeStepV105 = snpeStep
                    engineV105 = eng
                    v10LoadedVer = v10Ver
                    Log.i(
                        TAG,
                        String.format(
                            Locale.US,
                            "ENGINE_READY model=%s backend=HTP_V75_NPU stepBackend=%s omni=%b encLoadMs=%.1f stepLoadMs=%.1f totalLoadMs=%.1f",
                            v10Ver,
                            if (snpeStep != null) "HTP_V75_NPU" else "CPU_4T",
                            eng.hasOmniOutputs,
                            eng.encoderLoadMs,
                            eng.stepLoadMs,
                            eng.encoderLoadMs + eng.stepLoadMs,
                        ),
                    )
                }

                // 2. Load EgoClear-v6.6 (DAV2-Small, 42x74, 4ch disp_cues) if present; prefer HTP Step DLC
                val v66Dlc = resolveModel(V66_DLC_NAME)
                val v66StepDlc = resolveModel(V66_STEP_DLC_NAME)
                val v66Step = resolveModel(V66_STEP_NAME)
                val v66Cpu = resolveModel(V66_CPU_ENCODER_NAME)
                if ((v66StepDlc != null || v66Step != null) && (v66Dlc != null || v66Cpu != null)) {
                    if (v66Dlc != null) {
                        val snpe = EgoClearSnpeEncoder(
                            apkContext = applicationContext,
                            cacheContext = applicationContext,
                            dlcFile = v66Dlc,
                            auxOutputName = EgoClearSnpeEncoder.AUX_DISP_CUES,
                            f2dSize = EgoClearSnpeEncoder.F2D_SIZE,
                            auxSize = EgoClearSnpeEncoder.DC_SIZE,
                        )
                        val snpeStep = v66StepDlc?.let { stDlc ->
                            runCatching {
                                EgoClearSnpeStep(
                                    apkContext = applicationContext,
                                    cacheContext = applicationContext,
                                    dlcFile = stDlc,
                                    auxPrefix = "dc",
                                    hasOmni = false,
                                )
                            }.onFailure { err ->
                                Log.w(TAG, "v6.6 HTP step fallback to CPU ORT: ${err.message}")
                            }.getOrNull()
                        }
                        val eng = EgoClearStreamingEngine(
                            encoderModel = null,
                            stepModel = if (snpeStep != null) null else v66Step,
                            backend = EgoClearStreamingEngine.Backend.HTP_V75_NPU,
                            stepBackend = if (snpeStep != null) EgoClearStreamingEngine.Backend.HTP_V75_NPU else EgoClearStreamingEngine.Backend.CPU_4T,
                            snpeEncoder = snpe,
                            profile = EgoClearStreamingEngine.ModelProfile.V66_DAV2,
                            snpeStep = snpeStep,
                        )
                        snpeV66 = snpe
                        snpeStepV66 = snpeStep
                        engineV66 = eng
                        Log.i(
                            TAG,
                            String.format(
                                Locale.US,
                                "ENGINE_READY model=v6.6 backend=HTP_V75_NPU stepBackend=%s encLoadMs=%.1f stepLoadMs=%.1f totalLoadMs=%.1f",
                                if (snpeStep != null) "HTP_V75_NPU" else "CPU_4T",
                                eng.encoderLoadMs,
                                eng.stepLoadMs,
                                eng.encoderLoadMs + eng.stepLoadMs,
                            ),
                        )
                    } else if (v66Cpu != null && v66Step != null) {
                        val eng = EgoClearStreamingEngine(
                            encoderModel = v66Cpu,
                            stepModel = v66Step,
                            backend = EgoClearStreamingEngine.Backend.CPU_4T,
                            stepBackend = EgoClearStreamingEngine.Backend.CPU_4T,
                            profile = EgoClearStreamingEngine.ModelProfile.V66_DAV2,
                        )
                        engineV66 = eng
                        Log.i(TAG, "ENGINE_READY model=v6.6 backend=CPU_4T encLoadMs=${eng.encoderLoadMs} stepLoadMs=${eng.stepLoadMs}")
                    }
                }

                val modes = buildList {
                    if (engineV105 != null) add(ModelMode.V105)
                    if (engineV105 != null && engineV66 != null) add(ModelMode.DUAL_ENS)
                    if (engineV66 != null) add(ModelMode.V66)
                }
                if (modes.isEmpty()) {
                    val msg = buildString {
                        appendLine("缺少模型文件，请使用 adb push 推送到：")
                        appendLine(extDir.absolutePath)
                        appendLine("需要: $V106_DLC_NAME + $V106_STEP_NAME")
                    }
                    Log.w(TAG, msg)
                    runOnUiThread { showStatus(msg) }
                    return@execute
                }
                availableModes = modes
                activeMode = modes.first()
                val initLabel = modeDisplayLabel(activeMode)
                backendLabel = "$initLabel NPU"
                runOnUiThread {
                    statusMessage.visibility = View.GONE
                    btnModel.text = "模型:$initLabel"
                    btnModel.contentDescription = "当前视觉模型：$initLabel，双击切换"
                    updateButtonStyle(btnModel, active = activeMode != ModelMode.V66)
                    telemetryBanner.text = "EgoClear [$backendLabel] 就绪 · 等待相机帧…"
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to initialize EgoClear engine", t)
                runOnUiThread {
                    showStatus("模型初始化失败: ${t.javaClass.simpleName}: ${t.message}")
                }
            }
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener(
            {
                try {
                    val provider = future.get()
                    cameraProvider = provider
                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }
                    val resSelector = ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                Size(960, 720),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                            ),
                        )
                        .build()
                    val analysis = ImageAnalysis.Builder()
                        .setResolutionSelector(resSelector)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                    analysis.setAnalyzer(inferenceExecutor) { image ->
                        analyzeFrame(image)
                    }
                    provider.unbindAll()
                    val cam = provider.bindToLifecycle(
                        this,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analysis,
                    )
                    boundCamera = cam
                    val zState = cam.cameraInfo.zoomState.value
                    minZoomRatio = zState?.minZoomRatio ?: 1.0f
                    applyZoomPreference()
                    Log.i(TAG, String.format(Locale.US, "CAMERA_BOUND minZoom=%.2f activeZoom=%.2f", minZoomRatio, activeZoomRatio))
                } catch (t: Throwable) {
                    Log.e(TAG, "Camera bind failed", t)
                    showStatus("相机启动失败: ${t.message}")
                }
            },
            mainExecutor,
        )
    }

    private fun resetAllEngines() {
        engineV105?.resetState()
        engineV66?.resetState()
        ensAlertOn = false
        ensPrevScore = 0.0f
        ensHasPrevScore = false
    }

    private fun fuseDualOutputs(
        out105: EgoClearStreamingEngine.StepOutput,
        out66: EgoClearStreamingEngine.StepOutput,
        alertThreshold: Float,
        alertHorizonIdx: Int,
    ): EgoClearStreamingEngine.StepOutput {
        val n = out105.corrProbsMonotonic.size
        val corrFused = FloatArray(n)
        val uncFused = FloatArray(n)
        for (i in 0 until n) {
            val p1 = out105.corrProbsMonotonic[i].coerceIn(1e-5f, 1f - 1e-5f)
            val p2 = out66.corrProbsMonotonic[i].coerceIn(1e-5f, 1f - 1e-5f)
            val lAvg = 0.5 * (ln(p1 / (1f - p1)).toDouble() + ln(p2 / (1f - p2)).toDouble())
            corrFused[i] = (1.0 / (1.0 + exp(-lAvg.coerceIn(-20.0, 20.0)))).toFloat()
            uncFused[i] = 0.5f * (out105.uncLogits[i] + out66.uncLogits[i])
        }
        val stFused = FloatArray(out105.staticLogits.size) { i -> 0.5f * (out105.staticLogits[i] + out66.staticLogits[i]) }
        val dyFused = FloatArray(out105.dynamicLogits.size) { i -> 0.5f * (out105.dynamicLogits[i] + out66.dynamicLogits[i]) }
        var monoOk = true
        for (i in 1 until n) {
            if (corrFused[i] - corrFused[i - 1] < -1e-6f) {
                monoOk = false
                break
            }
        }
        val hIdx = alertHorizonIdx.coerceIn(0, n - 1)
        val pHorizon = corrFused[hIdx]
        val uncProb = (1.0 / (1.0 + exp(-uncFused[hIdx].toDouble()))).toFloat()
        val delta = if (ensHasPrevScore) pHorizon - ensPrevScore else 0.0f
        val offThr = alertThreshold * 0.72f
        val fastRise = delta >= 0.08f && pHorizon >= alertThreshold * 0.90f
        ensAlertOn = if (!ensAlertOn) {
            pHorizon >= alertThreshold || fastRise
        } else {
            pHorizon >= offThr
        }
        ensPrevScore = pHorizon
        ensHasPrevScore = true
        return EgoClearStreamingEngine.StepOutput(
            corrProbsMonotonic = corrFused,
            uncLogits = uncFused,
            staticLogits = stFused,
            dynamicLogits = dyFused,
            pAt4m = corrFused[EgoClearStreamingEngine.IDX_4M],
            pAt44m = corrFused[EgoClearStreamingEngine.IDX_44M],
            uncProbAt4m = uncProb,
            monotonicOk = monoOk,
            alertActive = ensAlertOn,
            encoderMs = out105.encoderMs + out66.encoderMs,
            stepMs = out105.stepMs + out66.stepMs,
            totalMs = out105.totalMs + out66.totalMs,
            geomCorridorMinMeters = out105.geomCorridorMinMeters,
            leftBypassProbs = out105.leftBypassProbs,
            rightBypassProbs = out105.rightBypassProbs,
            headHazardProbs = out105.headHazardProbs,
            dropOffProbs = out105.dropOffProbs,
            dynClosingProbs = out105.dynClosingProbs,
            bypassAdvice = if (ensAlertOn) out105.bypassAdvice else EgoClearStreamingEngine.BypassAdvice.CLEAR,
            dropOffHitMeters = out105.dropOffHitMeters,
            headHitMeters = out105.headHitMeters,
            dynClosingHitMeters = out105.dynClosingHitMeters,
            isUnknownUnobservable = out105.isUnknownUnobservable,
        )
    }

    private fun modeDisplayLabel(mode: ModelMode): String = when (mode) {
        ModelMode.V105 -> v10LoadedVer
        ModelMode.DUAL_ENS -> "双融合"
        ModelMode.V66 -> "v6.6"
    }

    private fun analyzeFrame(image: ImageProxy) {
        try {
            val mode = activeMode
            val eng105 = engineV105
            val eng66 = engineV66
            val ready = when (mode) {
                ModelMode.V105 -> eng105 != null
                ModelMode.V66 -> eng66 != null
                ModelMode.DUAL_ENS -> eng105 != null && eng66 != null
            }
            if (!ready) return
            if (isPaused) return

            val now = SystemClock.elapsedRealtime()
            val minIntervalMs = if (fullRateMode) 0L else 190L
            if (now - lastFrameStartMs < minIntervalMs) return
            if (!frameBusy.compareAndSet(false, true)) return
            lastFrameStartMs = now

            try {
                val pose = poseTracker.sampleForFrame()
                val hIdx = horizonBins[rangeIndex]
                val thr = thresholds[sensIndex]
                val isLandGrav = pose.gravityRotation == Surface.ROTATION_90 || pose.gravityRotation == Surface.ROTATION_270

                // Auto-reset BEV memory when switching between portrait and landscape gravity grip
                if (lastGravityRotation != -1 && lastGravityRotation != pose.gravityRotation) {
                    resetAllEngines()
                }
                lastGravityRotation = pose.gravityRotation

                // Gate extreme downward/upward tilt (e.g. phone on desk or pointed at floor/ceiling)
                // so close floor/lap pixels do not pollute recurrent BEV memory.
                if (abs(pose.pitchDeg) > 38f) {
                    if (!wasPitchGated) {
                        wasPitchGated = true
                        wasAlertActive = false
                        resetAllEngines()
                    }
                    val zeroGrid = FloatArray(EgoClearStreamingEngine.BEV_NZ * EgoClearStreamingEngine.BEV_NX)
                    val gatedState = EgoClearOverlayView.OverlayState(
                        corrProbs = FloatArray(EgoClearStreamingEngine.BEV_NZ),
                        staticProbGrid = zeroGrid,
                        dynamicProbGrid = zeroGrid,
                        headHazardProb = 0f,
                        pHorizon = 0f,
                        horizonMeters = 0.30f + (hIdx + 1) * 0.20f,
                        horizonIdx = hIdx,
                        alertActive = false,
                        alertThreshold = thr,
                        estimatedHitMeters = null,
                        pitchGated = true,
                        isLandscapeGravity = isLandGrav,
                    )
                    runOnUiThread {
                        overlayView.updateState(gatedState)
                        telemetryBanner.text = String.format(
                            Locale.US,
                            "%s · 镜头俯仰 %+.0f° 过大 · 暂停预警\n请将手机抬平至前进方向 (±38°内)",
                            backendLabel,
                            pose.pitchDeg,
                        )
                    }
                    return
                }
                if (wasPitchGated) {
                    wasPitchGated = false
                    resetAllEngines()
                }
                if (resetRequested) {
                    resetRequested = false
                    wasAlertActive = false
                    resetAllEngines()
                }

                val prep = preprocessor.process(image, activeZoomRatio, pose.gravityRotation)
                val lensBlackout = prep.meanLuma < 8f || prep.meanLuma > 248f
                if (wasBlackoutActive && !lensBlackout) {
                    resetAllEngines()
                }
                wasBlackoutActive = lensBlackout
                val dsEffective = max(pose.dsMeters, 0.30f)
                recordReplayFrameIfActive(prep, pose, dsEffective)

                val out = when (mode) {
                    ModelMode.V105 -> eng105!!.step(
                        rgbNchw = prep.rgbNchw,
                        kScaled = prep.kScaled,
                        rC2w = pose.rC2w,
                        tWorld = pose.tWorld,
                        fwdWorld = pose.fwdWorld,
                        dsMeters = dsEffective,
                        alertThreshold = thr,
                        alertHorizonIdx = hIdx,
                    )
                    ModelMode.V66 -> eng66!!.step(
                        rgbNchw = prep.rgbNchw,
                        kScaled = prep.kScaled,
                        rC2w = pose.rC2w,
                        tWorld = pose.tWorld,
                        fwdWorld = pose.fwdWorld,
                        dsMeters = dsEffective,
                        alertThreshold = thr,
                        alertHorizonIdx = hIdx,
                    )
                    ModelMode.DUAL_ENS -> {
                        val o105 = eng105!!.step(
                            rgbNchw = prep.rgbNchw,
                            kScaled = prep.kScaled,
                            rC2w = pose.rC2w,
                            tWorld = pose.tWorld,
                            fwdWorld = pose.fwdWorld,
                            dsMeters = dsEffective,
                            alertThreshold = thr,
                            alertHorizonIdx = hIdx,
                        )
                        val o66 = eng66!!.step(
                            rgbNchw = prep.rgbNchw,
                            kScaled = prep.kScaled,
                            rC2w = pose.rC2w,
                            tWorld = pose.tWorld,
                            fwdWorld = pose.fwdWorld,
                            dsMeters = dsEffective,
                            alertThreshold = thr,
                            alertHorizonIdx = hIdx,
                        )
                        fuseDualOutputs(o105, o66, thr, hIdx)
                    }
                }

                val doneMs = SystemClock.elapsedRealtime()
                if (lastDoneMs > 0L) {
                    val dt = (doneMs - lastDoneMs).coerceAtLeast(1L)
                    val instFps = 1000.0 / dt
                    emaFps = if (emaFps == 0.0) instFps else 0.75 * emaFps + 0.25 * instFps
                }
                lastDoneMs = doneMs
                frameCounter++

                val rawOvState = EgoClearOverlayView.buildState(
                    out,
                    thr,
                    hIdx,
                    pitchGated = false,
                    isLandscapeGravity = isLandGrav,
                )
                val ovState = if (lensBlackout) {
                    rawOvState.copy(
                        isUnknownUnobservable = true,
                        alertActive = false,
                        estimatedHitMeters = null,
                        dropOffHitMeters = null,
                        headHitMeters = null,
                        dynClosingHitMeters = null,
                        bypassAdvice = EgoClearStreamingEngine.BypassAdvice.CLEAR,
                    )
                } else {
                    rawOvState
                }
                triggerFeedbackIfNeeded(ovState, pose.walking, doneMs)

                val darkTag = when {
                    lensBlackout -> " · 镜头遮挡"
                    prep.meanLuma < 36f -> " · 偏暗"
                    else -> ""
                }
                val orientTag = if (isLandGrav) "横" else "竖"
                val geomTag = out.geomCorridorMinMeters?.let { String.format(Locale.US, " · 实测%.1fm", it) }
                    ?: if (activeMode != ModelMode.V66) " · 实测>5m" else ""
                val hudText = String.format(
                    Locale.US,
                    "%s · NPU %.0fms · Step %.0fms · %.1fHz · %s%.1fx\nP(%.1fm)=%.0f%% %s%s · 俯仰%+.0f°%s",
                    backendLabel,
                    out.encoderMs,
                    out.stepMs,
                    emaFps,
                    orientTag,
                    activeZoomRatio,
                    ovState.horizonMeters,
                    ovState.pHorizon * 100f,
                    when {
                        ovState.isUnknownUnobservable -> "◌看不清"
                        ovState.dropOffHitMeters != null -> "⚠落差"
                        ovState.alertActive -> "⚠预警"
                        else -> "畅通"
                    },
                    geomTag,
                    pose.pitchDeg,
                    darkTag,
                )
                if (frameCounter == 1L || frameCounter == 15L || frameCounter % 10L == 0L) {
                    if (frameCounter == 15L || frameCounter == 60L) {
                        saveDiagSnapshot(prep, pose, image, out, ovState)
                    }
                    Log.i(
                        TAG,
                        String.format(
                            Locale.US,
                            "FRAME=%d backend=%s zoom=%.2f src=%dx%d rot=%d grav=%d luma=%.0f prepMs=%.2f encMs=%.2f stepMs=%.2f fps=%.2f p0.5=%.3f p1.3=%.3f p1.9=%.3f p2.5=%.3f p4.1=%.3f geomM=%s alert=%b hitM=%s dropM=%s headM=%s dynM=%s bypass=%s unk=%b(%.2f) pitch=%.1f roll=%.1f k=[%.0f,%.0f]",
                            frameCounter,
                            backendLabel,
                            activeZoomRatio,
                            image.width,
                            image.height,
                            prep.appliedRotDeg,
                            pose.gravityRotation,
                            prep.meanLuma,
                            prep.preprocMs,
                            out.encoderMs,
                            out.stepMs,
                            emaFps,
                            out.corrProbsMonotonic[0],
                            out.corrProbsMonotonic[4],
                            out.corrProbsMonotonic[7],
                            out.corrProbsMonotonic[10],
                            out.pAt4m,
                            out.geomCorridorMinMeters?.let { String.format(Locale.US, "%.1f", it) } ?: "clear",
                            ovState.alertActive,
                            ovState.estimatedHitMeters?.let { String.format(Locale.US, "%.1f", it) } ?: "clear",
                            ovState.dropOffHitMeters?.let { String.format(Locale.US, "%.1f", it) } ?: "none",
                            ovState.headHitMeters?.let { String.format(Locale.US, "%.1f", it) } ?: "none",
                            ovState.dynClosingHitMeters?.let { String.format(Locale.US, "%.1f", it) } ?: "none",
                            ovState.bypassAdvice.name,
                            ovState.isUnknownUnobservable,
                            out.uncProbAt4m,
                            pose.pitchDeg,
                            pose.rollDeg,
                            prep.kScaled[0],
                            prep.kScaled[1],
                        ),
                    )
                }

                runOnUiThread {
                    overlayView.updateState(ovState)
                    telemetryBanner.text = hudText
                }
            } finally {
                frameBusy.set(false)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Frame inference error", t)
        } finally {
            image.close()
        }
    }

    private fun triggerFeedbackIfNeeded(state: EgoClearOverlayView.OverlayState, walking: Boolean, nowMs: Long) {
        // 0. Unobservable / occluded camera abstention (UNKNOWN gate)
        if (state.isUnknownUnobservable) {
            wasAlertActive = false
            wasDropActive = false
            if (!wasUnknownActive) {
                wasUnknownActive = true
                if (voiceEnabled && ttsReady && (nowMs - lastSpeakMs >= 2500L)) {
                    lastSpeakMs = nowMs
                    runCatching { tts?.speak("画面看不清", TextToSpeech.QUEUE_FLUSH, null, "egoclear_unk") }
                }
            }
            return
        }
        wasUnknownActive = false

        // 1. Highest safety priority: Negative Obstacle / Drop-Off (stairs down / curb / pit)
        val dropDist = state.dropOffHitMeters
        if (dropDist != null) {
            val newlyDrop = !wasDropActive
            wasDropActive = true
            if (hapticEnabled && (newlyDrop || (walking && nowMs - lastVibeMs >= 700L))) {
                lastVibeMs = nowMs
                vibrator?.let { v ->
                    runCatching {
                        v.vibrate(
                            VibrationEffect.createWaveform(
                                longArrayOf(0L, 65L, 55L, 65L, 55L, 95L),
                                intArrayOf(0, 240, 0, 240, 0, 255),
                                -1,
                            ),
                        )
                    }
                }
            }
            if (voiceEnabled && ttsReady && (newlyDrop || (walking && nowMs - lastDropSpeakMs >= 2800L))) {
                lastDropSpeakMs = nowMs
                lastSpeakMs = nowMs
                val dHalf = (dropDist * 2f).roundToInt() / 2f
                val dTxt = if (dHalf == dHalf.toInt().toFloat()) "${dHalf.toInt()}米" else "${dHalf}米"
                runCatching {
                    tts?.speak("注意前方${dTxt}下台阶落差", TextToSpeech.QUEUE_FLUSH, null, "egoclear_drop")
                }
            }
            return
        }
        wasDropActive = false

        // 2. Approaching dynamic collision (even if static corridor is clear)
        val dynDist = state.dynClosingHitMeters
        if (!state.alertActive && dynDist != null) {
            val newlyEntered = !wasAlertActive
            wasAlertActive = true
            if (hapticEnabled && (newlyEntered || (walking && nowMs - lastVibeMs >= 900L))) {
                lastVibeMs = nowMs
                vibrator?.let { v ->
                    runCatching { v.vibrate(VibrationEffect.createWaveform(longArrayOf(0L, 50L, 60L, 50L), intArrayOf(0, 180, 0, 180), -1)) }
                }
            }
            if (voiceEnabled && ttsReady && (newlyEntered || (walking && nowMs - lastSpeakMs >= 3000L))) {
                lastSpeakMs = nowMs
                runCatching { tts?.speak("前方动态靠近", TextToSpeech.QUEUE_FLUSH, null, "egoclear_dyn") }
            }
            return
        }

        if (!state.alertActive) {
            wasAlertActive = false
            lastSpokenDist = 99f
            return
        }
        val dist = state.estimatedHitMeters ?: state.horizonMeters
        val newlyEntered = !wasAlertActive
        val gotMuchCloser = dist <= lastSpokenDist - 0.55f
        wasAlertActive = true

        // 3. Proximity-scaled haptic vibration (no repeated nagging while stationary!)
        if (hapticEnabled) {
            val vibeCooldown = when {
                dist <= 1.0f -> 450L
                dist <= 1.8f -> 800L
                else -> 1200L
            }
            val shouldVibe = newlyEntered || gotMuchCloser || (walking && (nowMs - lastVibeMs >= vibeCooldown))
            if (shouldVibe) {
                lastVibeMs = nowMs
                val amp = when {
                    dist <= 1.0f -> 240
                    dist <= 1.8f -> 175
                    else -> 125
                }
                val dur = if (dist <= 1.0f) 90L else 55L
                vibrator?.let { v ->
                    runCatching {
                        v.vibrate(VibrationEffect.createOneShot(dur, amp))
                    }
                }
            }
        }

        // 4. State-change driven Chinese TTS with Omni 3-lane bypass & overhead head guidance
        if (voiceEnabled && ttsReady) {
            val shouldSpeak = newlyEntered || gotMuchCloser || (walking && (nowMs - lastSpeakMs >= 3500L))
            if (shouldSpeak) {
                lastSpeakMs = nowMs
                lastSpokenDist = dist
                val basePhrase = when {
                    (state.headHitMeters != null || state.headHazardProb >= 0.55f) && dist <= 2.5f -> "注意上方碰头"
                    dist <= 1.0f -> "近处障碍"
                    dist <= 2.2f -> {
                        val halfSteps = (dist * 2f).roundToInt() / 2f
                        if (halfSteps == halfSteps.toInt().toFloat()) {
                            "前方${halfSteps.toInt()}米有障碍"
                        } else {
                            "前方${halfSteps}米有障碍"
                        }
                    }
                    dist <= 3.5f -> "前方${dist.roundToInt()}米有障碍"
                    else -> "前方有障碍"
                }
                val phrase = if (state.bypassAdvice.spokenZh.isNotEmpty()) {
                    "$basePhrase，${state.bypassAdvice.spokenZh}"
                } else {
                    basePhrase
                }
                runCatching {
                    tts?.speak(phrase, TextToSpeech.QUEUE_FLUSH, null, "egoclear_alert")
                }
            }
        }
    }

    private fun initTts() {
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val res = tts?.setLanguage(Locale.SIMPLIFIED_CHINESE)
                tts?.setSpeechRate(1.15f)
                ttsReady = res != TextToSpeech.LANG_MISSING_DATA && res != TextToSpeech.LANG_NOT_SUPPORTED
            }
        }
    }

    @Suppress("DEPRECATION")
    private fun resolveVibrator(): Vibrator? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vm = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
            vm?.defaultVibrator
        } else {
            getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }
    }

    @Suppress("DEPRECATION")
    private fun currentDisplayRotation(): Int {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display?.rotation ?: Surface.ROTATION_0
        } else {
            windowManager.defaultDisplay?.rotation ?: Surface.ROTATION_0
        }
    }

    private fun hasCameraPermission(): Boolean {
        return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    private fun saveDiagSnapshot(
        prep: EgoClearCameraPreprocessor.PreprocResult,
        pose: EgoClearPoseTracker.PoseSnapshot,
        image: ImageProxy,
        out: EgoClearStreamingEngine.StepOutput,
        ov: EgoClearOverlayView.OverlayState,
    ) {
        runCatching {
            val w = EgoClearStreamingEngine.RGB_W
            val h = EgoClearStreamingEngine.RGB_H
            val plane = w * h
            val pixels = IntArray(plane)
            val buf = prep.rgbNchw
            for (i in 0 until plane) {
                val r = (((buf[i] * 0.229f + 0.485f) * 255f).toInt()).coerceIn(0, 255)
                val g = (((buf[plane + i] * 0.224f + 0.456f) * 255f).toInt()).coerceIn(0, 255)
                val b = (((buf[2 * plane + i] * 0.225f + 0.406f) * 255f).toInt()).coerceIn(0, 255)
                pixels[i] = Color.rgb(r, g, b)
            }
            val bmp = Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
            File(filesDir, "diag_npu_input.png").outputStream().use {
                bmp.compress(Bitmap.CompressFormat.PNG, 95, it)
            }
            bmp.recycle()
            val meta = buildString {
                appendLine("src=${image.width}x${image.height} rot=${image.imageInfo.rotationDegrees} zoom=$activeZoomRatio")
                appendLine("K=${prep.kScaled.joinToString(",")}")
                appendLine("R_c2w=${pose.rC2w.joinToString(",")}")
                appendLine("t_w=${pose.tWorld.joinToString(",")}")
                appendLine("fwd_w=${pose.fwdWorld.joinToString(",")}")
                appendLine("pitch=${pose.pitchDeg} roll=${pose.rollDeg} headProb=${ov.headHazardProb}")
                appendLine("corrProbs=${out.corrProbsMonotonic.joinToString(",") { String.format(Locale.US, "%.3f", it) }}")
            }
            File(filesDir, "diag_npu_meta.txt").writeText(meta)
        }
    }

    private fun recordReplayFrameIfActive(
        prep: EgoClearCameraPreprocessor.PreprocResult,
        pose: EgoClearPoseTracker.PoseSnapshot,
        dsEffective: Float,
    ) {
        val rem = recordRequestedFrames
        if (rem <= 0) return
        runCatching {
            if (recordStream == null) {
                val dir = File(getExternalFilesDir("replays"), "").apply { mkdirs() }
                val f = File(dir, "replay_${System.currentTimeMillis()}.egoclear")
                val bos = BufferedOutputStream(FileOutputStream(f), 256 * 1024)
                val hdr = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN)
                hdr.put("EGC1".toByteArray(Charsets.US_ASCII))
                hdr.putInt(0) // patched on close
                hdr.putInt(EgoClearStreamingEngine.RGB_H)
                hdr.putInt(EgoClearStreamingEngine.RGB_W)
                bos.write(hdr.array())
                recordFile = f
                recordStream = bos
                recordWrittenFrames = 0
            }
            val meta = ByteBuffer.allocate(22 * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (v in prep.kScaled) meta.putFloat(v)
            for (v in pose.rC2w) meta.putFloat(v)
            for (v in pose.tWorld) meta.putFloat(v)
            for (v in pose.fwdWorld) meta.putFloat(v)
            meta.putFloat(dsEffective)
            meta.putFloat(pose.pitchDeg)
            meta.putFloat(pose.rollDeg)
            recordStream!!.write(meta.array())

            val plane = EgoClearStreamingEngine.RGB_W * EgoClearStreamingEngine.RGB_H
            val src = prep.rgbNchw
            val dst = recordRgbU8
            for (i in 0 until plane) {
                dst[i] = (((src[i] * 0.229f + 0.485f) * 255f).toInt().coerceIn(0, 255)).toByte()
                dst[plane + i] = (((src[plane + i] * 0.224f + 0.456f) * 255f).toInt().coerceIn(0, 255)).toByte()
                dst[2 * plane + i] = (((src[2 * plane + i] * 0.225f + 0.406f) * 255f).toInt().coerceIn(0, 255)).toByte()
            }
            recordStream!!.write(dst)
            recordWrittenFrames++
            val nextRem = rem - 1
            recordRequestedFrames = nextRem
            if (nextRem <= 0) {
                val savedName = recordFile?.name ?: "replay.egoclear"
                finishReplayRecordingIfOpen()
                Log.i(TAG, "REPLAY_SAVED file=$savedName frames=$recordWrittenFrames")
                runOnUiThread {
                    btnRecord.text = "录回放"
                    updateButtonStyle(btnRecord, active = false)
                    telemetryBanner.text = "已保存离线回放：$savedName ($recordWrittenFrames 帧)"
                }
            }
        }.onFailure { err ->
            Log.e(TAG, "Replay recording error", err)
            recordRequestedFrames = 0
            finishReplayRecordingIfOpen()
            runOnUiThread {
                btnRecord.text = "录回放"
                updateButtonStyle(btnRecord, active = false)
            }
        }
    }

    private fun finishReplayRecordingIfOpen() {
        val bos = recordStream ?: return
        val f = recordFile
        val n = recordWrittenFrames
        recordStream = null
        recordFile = null
        runCatching {
            bos.flush()
            bos.close()
            if (f != null && f.isFile && n > 0) {
                RandomAccessFile(f, "rw").use { raf ->
                    raf.seek(4L)
                    val b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(n)
                    raf.write(b.array())
                }
            }
        }
    }

    private fun showStatus(msg: String) {
        statusMessage.text = msg
        statusMessage.visibility = View.VISIBLE
    }

    companion object {
        private const val TAG = "EgoClearDemoPerf"
        private const val REQ_CAMERA = 1001
        private const val REPLAY_CLIP_FRAMES = 50
        private const val V14_DLC_NAME = "egoclear_v14_encoder_htp_v75.dlc"
        private const val V14_STEP_DLC_NAME = "egoclear_v14_step_htp_v75.dlc"
        private const val V14_STEP_NAME = "egoclear_v14_step_fp32.onnx"
        private const val V13_DLC_NAME = "egoclear_v13_encoder_htp_v75.dlc"
        private const val V13_STEP_DLC_NAME = "egoclear_v13_step_htp_v75.dlc"
        private const val V13_STEP_NAME = "egoclear_v13_step_fp32.onnx"
        private const val V12_DLC_NAME = "egoclear_v12_encoder_htp_v75.dlc"
        private const val V12_STEP_DLC_NAME = "egoclear_v12_step_htp_v75.dlc"
        private const val V12_STEP_NAME = "egoclear_v12_step_fp32.onnx"
        private const val V11_DLC_NAME = "egoclear_v11_encoder_htp_v75.dlc"
        private const val V11_STEP_DLC_NAME = "egoclear_v11_step_htp_v75.dlc"
        private const val V11_STEP_NAME = "egoclear_v11_step_fp32.onnx"
        private const val V106_DLC_NAME = "egoclear_v106_encoder_htp_v75.dlc"
        private const val V106_STEP_DLC_NAME = "egoclear_v106_step_htp_v75.dlc"
        private const val V106_STEP_NAME = "egoclear_v106_step_fp32.onnx"
        private const val V105_DLC_NAME = "egoclear_v105_encoder_htp_v75.dlc"
        private const val V105_STEP_NAME = "egoclear_v105_step_fp32.onnx"
        private const val V66_DLC_NAME = "egoclear_v66_encoder_htp_v75_fp16.dlc"
        private const val V66_STEP_DLC_NAME = "egoclear_v66_step_htp_v75.dlc"
        private const val V66_STEP_NAME = "egoclear_v66_cleankd_step_fp16.onnx"
        private const val V66_CPU_ENCODER_NAME = "egoclear_v66_cleankd_encoder_fp16.onnx"
    }
}
