package com.linnan.blindassist.hftf

import android.Manifest
import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.ValueAnimator
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.os.Bundle
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.view.animation.LinearInterpolator
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import com.linnan.blindassist.camera.AtomS3rMjpegFrameSource
import com.linnan.blindassist.vision.VisionFrame
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.ln

private const val USE_EXTERNAL_HARDWARE = true

/** Device-only visual demo for the frozen canonical FP16 + QNN cached-context route. */
class DepthExperienceActivity : Activity(), LifecycleOwner {
    private val lifecycleRegistry = LifecycleRegistry(this)
    override val lifecycle: Lifecycle get() = lifecycleRegistry
    private lateinit var preview: PreviewView
    private lateinit var overlay: DepthHeatmapView
    private lateinit var status: TextView
    private lateinit var detail: TextView
    private var engine: DepthExperienceEngine? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        lifecycleRegistry.currentState = Lifecycle.State.CREATED
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(buildContent())
        if (USE_EXTERNAL_HARDWARE || ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startExperience()
        } else {
            status.text = "等待相机权限"
            requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST)
        }
    }

    override fun onStart() {
        super.onStart()
        lifecycleRegistry.currentState = Lifecycle.State.STARTED
        if (USE_EXTERNAL_HARDWARE || ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startExperience()
        }
    }

    override fun onResume() {
        super.onResume()
        lifecycleRegistry.currentState = Lifecycle.State.RESUMED
    }

    override fun onPause() {
        lifecycleRegistry.currentState = Lifecycle.State.STARTED
        super.onPause()
    }

    override fun onStop() {
        engine?.stopAlgorithm()
        lifecycleRegistry.currentState = Lifecycle.State.CREATED
        super.onStop()
    }

    override fun onDestroy() {
        engine?.close()
        engine = null
        lifecycleRegistry.currentState = Lifecycle.State.DESTROYED
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != CAMERA_PERMISSION_REQUEST) return
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            startExperience()
        } else {
            showFailure("需要相机权限才能体验实时深度")
        }
    }

    private fun startExperience() {
        val activeEngine = engine ?: DepthExperienceEngine(
            activity = this,
            previewView = preview,
            onStatus = { message, extra ->
                status.text = message
                status.contentDescription = message
                detail.text = extra
            },
            onDepth = overlay::show,
            onFailure = ::showFailure,
        ).also { engine = it }
        activeEngine.startCamera()
    }

    private fun showFailure(message: String) {
        status.text = "深度链路不可用"
        detail.text = message
        overlay.clearDepth()
    }

    private fun buildContent(): View {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        preview = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
            contentDescription = "实时后置相机画面"
        }
        overlay = DepthHeatmapView(this)
        root.addView(preview, FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
        root.addView(overlay, FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)

        val topPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(12), dp(18), dp(12))
            setBackgroundColor(0xB8000000.toInt())
        }
        topPanel.addView(TextView(this).apply {
            text = "BlindAssist 深度体验（设备级 Demo）"
            setTextColor(Color.WHITE)
            textSize = 20f
        })
        status = TextView(this).apply {
            text = "正在准备模型…"
            setTextColor(0xFF80DEEA.toInt())
            textSize = 18f
        }
        detail = TextView(this).apply {
            text = "首次启动需要载入约 55 MB 的本地模型"
            setTextColor(0xFFE0E0E0.toInt())
            textSize = 14f
        }
        topPanel.addView(status)
        topPanel.addView(detail)
        root.addView(topPanel, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP,
        ))

        val toggle = Button(this).apply {
            text = "全屏叠加"
            contentDescription = "切换深度图显示方式"
            setOnClickListener {
                this@DepthExperienceActivity.overlay.displayMode = when (
                    this@DepthExperienceActivity.overlay.displayMode
                ) {
                    DepthDisplayMode.SPLIT -> DepthDisplayMode.OVERLAY
                    DepthDisplayMode.OVERLAY -> DepthDisplayMode.RGB
                    DepthDisplayMode.RGB -> DepthDisplayMode.SPLIT
                }
                text = when (this@DepthExperienceActivity.overlay.displayMode) {
                    DepthDisplayMode.SPLIT -> "全屏叠加"
                    DepthDisplayMode.OVERLAY -> "仅看相机"
                    DepthDisplayMode.RGB -> "左右对比"
                }
            }
        }
        root.addView(toggle, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM or Gravity.START,
        ).apply { setMargins(dp(18), dp(18), dp(18), dp(54)) })

        root.addView(TextView(this).apply {
            text = "红色更近，蓝色更远。仅供算法体验，不能替代导盲或安全判断。"
            setTextColor(Color.WHITE)
            textSize = 14f
            gravity = Gravity.CENTER
            setPadding(dp(12), dp(8), dp(12), dp(8))
            setBackgroundColor(0xB8000000.toInt())
        }, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM,
        ))
        return root
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private companion object {
        const val CAMERA_PERMISSION_REQUEST = 701
    }
}

private class DepthExperienceEngine(
    private val activity: Activity,
    private val previewView: PreviewView,
    private val onStatus: (String, String) -> Unit,
    private val onDepth: (DepthVisual) -> Unit,
    private val onFailure: (String) -> Unit,
) : AutoCloseable {
    private val analyzerExecutor = Executors.newSingleThreadExecutor()
    private val depthExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "dav2-depth").apply {
            Process.setThreadPriority(Process.THREAD_PRIORITY_DISPLAY)
        }
    }
    private val visualExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "dav2-heatmap").apply { priority = Thread.MIN_PRIORITY }
    }
    private val active = AtomicBoolean(false)
    private val processing = AtomicBoolean(false)
    private val initializationStarted = AtomicBoolean(false)
    private val lastSubmittedAt = AtomicLong(Long.MIN_VALUE)
    private val frameSequence = AtomicLong(0L)
    private val frame = OwnedYuv420Frame(WIDTH, HEIGHT) {}
    private val externalSource = AtomS3rMjpegFrameSource(
        EXTERNAL_ENDPOINT,
        decodeSampleSize = 1,
        shouldDecode = { !processing.get() },
    )
    private val externalRgb = Dav2BitmapRgbConverter(WIDTH, HEIGHT)
    private val power = activity.getSystemService(PowerManager::class.java)
    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    private var converter: Dav2Yuv420RgbConverter? = null
    private var preprocessor: Dav2NativePreprocessor? = null
    private var runtime: Dav2QnnCachedContext? = null
    private val visualWorkspace = DepthVisual.Companion.Workspace()
    // Reused only when no previous heatmap render is pending. This removes a
    // ~710 KB direct-buffer allocation from the latency-critical frame path.
    private val externalHeatmapInput = ByteBuffer.allocateDirect(
        Dav2PreprocessContract.PLANE * 2,
    ).order(ByteOrder.nativeOrder())
    private var lastCompletedAt = 0L
    private var lastExternalHeatmapAt = 0L
    private var lastExternalHeatmap: Bitmap? = null
    private val heatmapPending = AtomicBoolean(false)
    private var riskSummaryParityVerified = false
    private var fusedCanonicalParityVerified = false
    private var bitmapCanonicalParityVerified = false
    private var lastExternalStatusAt = 0L

    fun startCamera() {
        active.set(true)
        Log.i(RESOURCE_TAG, "DA V2 start requested; external=$USE_EXTERNAL_HARDWARE endpoint=$EXTERNAL_ENDPOINT")
        if (runtime != null) {
            bindCamera()
            return
        }
        if (!initializationStarted.compareAndSet(false, true)) return
        onStatus("正在载入 QNN HTP 模型…", "模型与 FP16 合同会在启动时校验")
        depthExecutor.execute {
            try {
                Log.i(RESOURCE_TAG, "loading cached QNN context")
                val model = materializeVerifiedModel(activity)
                val nativeDir = activity.applicationInfo.nativeLibraryDir
                converter = Dav2Yuv420RgbConverter()
                preprocessor = Dav2NativePreprocessor()
                runtime = Dav2QnnCachedContext(model.absolutePath, nativeDir)
                Log.i(RESOURCE_TAG, "cached QNN context ready")
                if (!active.get()) {
                    val staleRuntime = runtime
                    val stalePreprocessor = preprocessor
                    val staleConverter = converter
                    runtime = null
                    preprocessor = null
                    converter = null
                    staleRuntime?.close()
                    stalePreprocessor?.close()
                    staleConverter?.close()
                    initializationStarted.set(false)
                    return@execute
                }
                activity.runOnUiThread {
                    if (active.get()) {
                        onStatus("模型就绪，正在打开后置相机…", "canonical FP32 → strict FP16 → QNN cached context")
                        bindCamera()
                    }
                }
            } catch (error: Throwable) {
                Log.e(RESOURCE_TAG, "DA V2 initialization failed", error)
                activity.runOnUiThread { onFailure(error.message ?: error.javaClass.simpleName) }
            }
        }
    }

    fun stopCamera() {
        active.set(false)
        externalSource.stop()
        Log.i(RESOURCE_TAG, "AtomS3R diagnostics=${externalSource.diagnostics()}")
        analysis?.clearAnalyzer()
        analysis = null
        provider?.unbindAll()
    }

    /** Fully releases QNN/HTP and native preprocessing resources when the page leaves foreground. */
    fun stopAlgorithm() {
        stopCamera()
        val oldRuntime = runtime
        val oldPreprocessor = preprocessor
        val oldConverter = converter
        runtime = null
        preprocessor = null
        converter = null
        initializationStarted.set(false)
        depthExecutor.execute {
            oldRuntime?.close()
            oldPreprocessor?.close()
            oldConverter?.close()
            Log.i(RESOURCE_TAG, "DA V2 algorithm stopped; QNN/HTP resources released")
        }
    }

    private fun bindCamera() {
        if (USE_EXTERNAL_HARDWARE) {
            bindExternalHardware()
            return
        }
        if (!active.get() || runtime == null || analysis != null) return
        val future = ProcessCameraProvider.getInstance(activity)
        future.addListener({
            try {
                if (!active.get()) return@addListener
                val cameraProvider = future.get(10, TimeUnit.SECONDS)
                val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
                val imageAnalysis = ImageAnalysis.Builder()
                    .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(
                        ResolutionStrategy(
                            Size(WIDTH, HEIGHT),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                        ),
                    ).build())
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()
                imageAnalysis.setAnalyzer(analyzerExecutor, ::analyze)
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(activity as LifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis)
                provider = cameraProvider
                analysis = imageAnalysis
                onStatus("实时深度运行中", "等待第一张深度图…")
            } catch (error: Throwable) {
                onFailure("CameraX 启动失败：${error.message ?: error.javaClass.simpleName}")
            }
        }, ContextCompat.getMainExecutor(activity))
    }

    private fun bindExternalHardware() {
        if (!active.get() || runtime == null) return
        Log.i(RESOURCE_TAG, "starting AtomS3R MJPEG source")
        externalSource.start(
            previewView = null,
            onFrame = { externalFrame ->
                if (!processing.compareAndSet(false, true)) {
                    externalFrame.close()
                    return@start
                }
                depthExecutor.execute { processExternal(externalFrame) }
            },
            onStarted = {
                Log.i(RESOURCE_TAG, "AtomS3R MJPEG source started")
                onStatus("AtomS3R 外部链路已连接", "MJPEG + ToF4M · 等待第一帧 DA V2 风险结果")
            },
            onError = { error ->
                Log.e(RESOURCE_TAG, "AtomS3R source failed", error)
                activity.runOnUiThread { onFailure("AtomS3R 链路失败：${error.message ?: error.javaClass.simpleName}") }
            },
        )
    }


    private fun processExternal(frame: VisionFrame) {
        val startedAt = SystemClock.elapsedRealtimeNanos()
        var frameClosed = false
        var processingReleased = false
        try {
            if (!active.get()) return
            val bitmap = (frame as? com.linnan.blindassist.vision.NativeImageVisionFrame)?.nativeImage as? Bitmap
                ?: error("AtomS3R frame is not a Bitmap")
            val receivedAt = frame.frameStamp?.receivedAtNs ?: startedAt
            val capturedAt = frame.frameStamp?.capturedAtNs ?: receivedAt
            val timing = frame.externalTiming
            val tofRangeMm = frame.rangingSample?.rangeMm ?: -1
            val dav2Preprocessor = requireNotNull(preprocessor)
            val input: java.nio.ByteBuffer
            val bitmapCanonicalStart = SystemClock.elapsedRealtimeNanos()
            if (!bitmapCanonicalParityVerified) {
                val rgb = externalRgb.convert(bitmap)
                val expected = dav2Preprocessor.preprocessFp16CanonicalStrictDirectFused(rgb)
                val direct = dav2Preprocessor.preprocessFp16CanonicalStrictBitmap(bitmap)
                val expectedBytes = ByteArray(expected.remaining()).also { expected.duplicate().get(it) }
                val directBytes = ByteArray(direct.remaining()).also { direct.duplicate().get(it) }
                check(expectedBytes.contentEquals(directBytes)) {
                    "Bitmap canonical FP16 parity failed"
                }
                bitmapCanonicalParityVerified = true
                fusedCanonicalParityVerified = true
                Log.i(RESOURCE_TAG, "Bitmap canonical FP16 parity passed bit_exact=true")
                input = direct
            } else {
                input = dav2Preprocessor.preprocessFp16CanonicalStrictBitmap(bitmap)
            }
            val inputReadyAt = SystemClock.elapsedRealtimeNanos()
            val rgbReadyAt = inputReadyAt
            val output = requireNotNull(runtime).execute(input, computeInputHash = false)
            val qnnReadyAt = SystemClock.elapsedRealtimeNanos()
            val riskSummary = DepthRiskSummary.Result.from(requireNotNull(preprocessor).riskSummary(output))
            val riskReadyAt = SystemClock.elapsedRealtimeNanos()
            val riskAge = nanosToMs(riskReadyAt - receivedAt)
            // The latency-critical result is complete and no remaining work
            // needs the source Bitmap. Return it to the decode pool and admit
            // one newest frame while bookkeeping/heatmap dispatch finishes on
            // this single-thread executor.
            frame.close()
            frameClosed = true
            processing.set(false)
            processingReleased = true
            if (riskReadyAt - lastExternalStatusAt >= EXTERNAL_STATUS_PERIOD_NS) {
                lastExternalStatusAt = riskReadyAt
                activity.runOnUiThread {
                    onStatus(
                        "外部中心约 ${formatMeters(riskSummary.centerMeters)} · 近处约 ${formatMeters(riskSummary.nearMeters)}",
                        "设备→风险 ${formatMs(nanosToMs(riskReadyAt - capturedAt))} ms · 手机收帧→风险 ${formatMs(riskAge)} ms · ToF $tofRangeMm mm",
                    )
                }
            }
            val renderHeatmap = !heatmapPending.get() &&
                (lastExternalHeatmap == null ||
                    qnnReadyAt - lastExternalHeatmapAt >= EXTERNAL_HEATMAP_PERIOD_NS)
            val heatmapInput = if (renderHeatmap) {
                externalHeatmapInput.clear()
                externalHeatmapInput.put(output.duplicate())
                externalHeatmapInput.flip()
                externalHeatmapInput
            } else null
            val riskResultAt = riskReadyAt
            if (renderHeatmap) {
                lastExternalHeatmapAt = riskResultAt
            }
            lastCompletedAt = riskResultAt
            val deviceMinusAndroidNs = timing?.deviceMinusAndroidNs
            val mappedJpegReadyNs = deviceMinusAndroidNs?.let { timing.deviceJpegReadyNs - it }
            val transport = frame.externalTransportDiagnostics
            Log.i(
                TIMING_TAG,
                    "{\"route\":\"atoms3r_to_dav2\",\"frame_sequence\":${frame.frameStamp?.frameId ?: -1}," +
                    "\"device_capture_to_risk_ms\":${formatMetric(riskReadyAt - capturedAt)}," +
                    "\"android_received_to_risk_ms\":${formatMetric(riskReadyAt - receivedAt)}," +
                    "\"device_capture_to_jpeg_ready_ms\":${formatMetric((timing?.deviceJpegReadyNs ?: 0L) - (timing?.deviceCaptureNs ?: 0L))}," +
                    "\"jpeg_ready_to_first_byte_ms\":${formatMetric((timing?.androidFirstByteNs ?: receivedAt) - (mappedJpegReadyNs ?: receivedAt))}," +
                    "\"first_byte_to_jpeg_complete_ms\":${formatMetric((timing?.androidJpegCompleteNs ?: receivedAt) - (timing?.androidFirstByteNs ?: receivedAt))}," +
                    "\"capture_to_android_jpeg_complete_ms\":${formatMetric((timing?.androidJpegCompleteNs ?: receivedAt) - capturedAt)}," +
                    "\"jpeg_decode_ms\":${formatMetric((timing?.androidDecodeCompleteNs ?: startedAt) - (timing?.androidDecodeStartNs ?: startedAt))}," +
                    "\"decode_complete_to_processing_ms\":${formatMetric(startedAt - (timing?.androidDecodeCompleteNs ?: startedAt))}," +
                    "\"jpeg_size_bytes\":${transport?.jpegSizeBytes ?: -1},\"wifi_rssi_dbm\":${transport?.wifiRssiDbm ?: 0}," +
                    "\"previous_frame_sequence\":${transport?.previousFrameSequence ?: -1}," +
                    "\"previous_response_write_ms\":${formatMetric(transport?.previousResponseWriteDurationNs ?: 0L)}," +
                    "\"android_body_read_calls\":${transport?.androidBodyReadCalls ?: -1}," +
                    "\"android_max_body_read_gap_ms\":${formatMetric(transport?.androidMaxBodyReadGapNs ?: 0L)}," +
                    "\"bitmap_to_rgb_ms\":0.000,\"canonical_preprocess_ms\":${formatMetric(inputReadyAt - bitmapCanonicalStart)},\"bitmap_to_canonical_ms\":${formatMetric(inputReadyAt - bitmapCanonicalStart)}," +
                    "\"qnn_execute_ms\":${formatMetric(qnnReadyAt - inputReadyAt)},\"qnn_to_risk_result_ms\":${formatMetric(riskResultAt - qnnReadyAt)}," +
                    "\"heatmap_render_ms\":0.000,\"heatmap_rendered\":$renderHeatmap,\"heatmap_render_async\":$renderHeatmap,\"depth_visual_ms\":0.000," +
                    "\"tof_range_mm\":$tofRangeMm}",
            )
            if (heatmapInput != null) {
                heatmapPending.set(true)
                visualExecutor.execute {
                    try {
                        val visual = DepthVisual.from(
                            heatmapInput,
                            0.0,
                            power.currentThermalStatus,
                            lastCompletedAt,
                            visualWorkspace,
                            renderHeatmap = true,
                            metricsOverride = riskSummary,
                        ) { center, near ->
                            if (!riskSummaryParityVerified) {
                                check(java.lang.Float.floatToRawIntBits(center) ==
                                    java.lang.Float.floatToRawIntBits(riskSummary.centerMeters) &&
                                    java.lang.Float.floatToRawIntBits(near) ==
                                    java.lang.Float.floatToRawIntBits(riskSummary.nearMeters)) {
                                    "native risk summary parity failed: native=$riskSummary kotlin=($center,$near)"
                                }
                                riskSummaryParityVerified = true
                                Log.i(RESOURCE_TAG, "native risk summary parity passed bit_exact=true")
                            }
                        }
                        lastExternalHeatmap = visual.heatmap
                        activity.runOnUiThread {
                            if (active.get()) onDepth(visual)
                        }
                    } catch (error: Throwable) {
                        Log.e(RESOURCE_TAG, "DA V2 heatmap render failed", error)
                    } finally {
                        heatmapPending.set(false)
                    }
                }
            }
        } catch (error: Throwable) {
            Log.e(RESOURCE_TAG, "DA V2 external frame failed", error)
            activity.runOnUiThread { onFailure("DA V2 外部帧失败：${error.message ?: error.javaClass.simpleName}") }
        } finally {
            if (!frameClosed) frame.close()
            if (!processingReleased) processing.set(false)
        }
    }

    private fun analyze(image: ImageProxy) {
        try {
            if (!active.get() || image.width != WIDTH || image.height != HEIGHT || image.planes.size != 3) return
            val now = SystemClock.elapsedRealtimeNanos()
            val previous = lastSubmittedAt.get()
            if (previous != Long.MIN_VALUE && now - previous < DEPTH_PERIOD_NANOS) return
            if (!processing.compareAndSet(false, true)) return
            if (!lastSubmittedAt.compareAndSet(previous, now)) {
                processing.set(false)
                return
            }
            val owned = frame.lease()
            try {
                copyPlane(image.planes[0], owned.y, WIDTH, HEIGHT)
                copyPlane(image.planes[1], owned.u, WIDTH / 2, HEIGHT / 2)
                copyPlane(image.planes[2], owned.v, WIDTH / 2, HEIGHT / 2)
                owned.width = WIDTH
                owned.height = HEIGHT
                owned.rotationDegrees = image.imageInfo.rotationDegrees
                owned.sensorTimestampNanos = image.imageInfo.timestamp
                owned.receivedAtNanos = now
                owned.copyCompletedAtNanos = SystemClock.elapsedRealtimeNanos()
                owned.sequence = frameSequence.incrementAndGet()
                depthExecutor.execute { process(owned) }
            } catch (error: Throwable) {
                owned.close()
                processing.set(false)
                throw error
            }
        } catch (error: Throwable) {
            activity.runOnUiThread { onFailure("相机帧处理失败：${error.message ?: error.javaClass.simpleName}") }
        } finally {
            image.close()
        }
    }

    private fun process(owned: OwnedYuv420Frame) {
        val startedAt = SystemClock.elapsedRealtimeNanos()
        try {
            if (!active.get()) return
            val thermal = power.currentThermalStatus
            if (thermal >= PowerManager.THERMAL_STATUS_SEVERE) {
                activity.runOnUiThread {
                    onFailure("设备温度过高，已停止深度推理（thermal=$thermal）")
                    stopCamera()
                }
                return
            }
            // Keep the YUV->RGB and canonical preprocessing buffers native/direct
            // all the way into QNN.  The ByteArray route is retained for parity
            // tests, but would add a per-frame Java/native bridge copy here.
            val rgb = requireNotNull(converter).convertDirect(owned)
            val rgbReadyAt = SystemClock.elapsedRealtimeNanos()
            val input = requireNotNull(preprocessor).preprocessFp16CanonicalStrictDirect(rgb)
            val inputReadyAt = SystemClock.elapsedRealtimeNanos()
            // Input hashing is a parity diagnostic, not part of the live route.
            val output = requireNotNull(runtime).execute(input, computeInputHash = false)
            val qnnReadyAt = SystemClock.elapsedRealtimeNanos()
            var riskReadyAt = 0L
            val visualWithoutAge = DepthVisual.from(
                output,
                0.0,
                thermal,
                lastCompletedAt,
                visualWorkspace,
                onMetricsReady = { centerMeters, nearMeters ->
                riskReadyAt = SystemClock.elapsedRealtimeNanos()
                val riskAgeMs = nanosToMs(riskReadyAt - owned.receivedAtNanos)
                activity.runOnUiThread {
                    onStatus(
                        "中心约 ${formatMeters(centerMeters)} · 近处约 ${formatMeters(nearMeters)}",
                        "风险结果 ${formatMs(riskAgeMs)} ms · HTP · thermal $thermal",
                    )
                }
                },
            )
            val completedAt = SystemClock.elapsedRealtimeNanos()
            val visual = visualWithoutAge.copy(
                pipelineMs = nanosToMs(completedAt - owned.receivedAtNanos),
            )
            lastCompletedAt = completedAt
            Log.i(
                TIMING_TAG,
                "{\"frame_sequence\":${owned.sequence}," +
                    "\"sensor_timestamp_ns\":${owned.sensorTimestampNanos}," +
                    "\"received_at_ns\":${owned.receivedAtNanos}," +
                    "\"yuv_copy_ms\":${formatMetric(owned.copyCompletedAtNanos - owned.receivedAtNanos)}," +
                    "\"executor_wait_ms\":${formatMetric(startedAt - owned.copyCompletedAtNanos)}," +
                    "\"yuv_to_rgb_ms\":${formatMetric(rgbReadyAt - startedAt)}," +
                    "\"canonical_preprocess_ms\":${formatMetric(inputReadyAt - rgbReadyAt)}," +
                    "\"qnn_execute_ms\":${formatMetric(qnnReadyAt - inputReadyAt)}," +
                    "\"depth_result_ms\":${formatMetric(riskReadyAt - qnnReadyAt)}," +
                    "\"depth_visual_ms\":${formatMetric(completedAt - qnnReadyAt)}," +
                    "\"received_to_risk_result_ms\":${formatMetric(riskReadyAt - owned.receivedAtNanos)}," +
                    "\"sensor_to_risk_result_ms\":${formatMetric(riskReadyAt - owned.sensorTimestampNanos)}," +
                    "\"received_to_result_ms\":${formatMetric(completedAt - owned.receivedAtNanos)}," +
                    "\"sensor_to_result_ms\":${formatMetric(completedAt - owned.sensorTimestampNanos)}," +
                    "\"thermal_status\":$thermal}",
            )
            activity.runOnUiThread {
                onDepth(visual)
                onStatus(
                    "中心约 ${formatMeters(visual.centerMeters)} · 近处约 ${formatMeters(visual.nearMeters)}",
                    "全链路 ${formatMs(visual.pipelineMs)} ms · 深度刷新 ${formatHz(visual.updateHz)} Hz · HTP · thermal ${visual.thermalStatus}",
                )
            }
        } catch (error: Throwable) {
            activity.runOnUiThread { onFailure("深度推理失败：${error.message ?: error.javaClass.simpleName}") }
        } finally {
            owned.close()
            processing.set(false)
        }
    }

    override fun close() {
        stopAlgorithm()
        externalSource.shutdown()
        analyzerExecutor.shutdownNow()
        visualExecutor.shutdownNow()
        depthExecutor.execute {
            runtime?.close()
            preprocessor?.close()
            converter?.close()
            runtime = null
            preprocessor = null
            converter = null
        }
        depthExecutor.shutdown()
        depthExecutor.awaitTermination(5, TimeUnit.SECONDS)
    }

    private fun copyPlane(plane: ImageProxy.PlaneProxy, target: ByteArray, width: Int, height: Int) {
        val source = plane.buffer.duplicate()
        for (row in 0 until height) {
            val sourceStart = row * plane.rowStride
            val targetStart = row * width
            if (plane.pixelStride == 1) {
                source.position(sourceStart)
                source.get(target, targetStart, width)
            } else {
                for (column in 0 until width) {
                    target[targetStart + column] = source.get(sourceStart + column * plane.pixelStride)
                }
            }
        }
    }

    private fun elapsedMs(startedAt: Long) =
        (SystemClock.elapsedRealtimeNanos() - startedAt) / 1_000_000.0

    private fun nanosToMs(value: Long) = value / 1_000_000.0
    private fun formatMetric(valueNanos: Long) = String.format(Locale.US, "%.3f", nanosToMs(valueNanos))

    private fun formatMeters(value: Float): String =
        if (value.isFinite()) String.format(Locale.US, "%.2f m", value) else "未知"

    private fun formatMs(value: Double) = String.format(Locale.US, "%.1f", value)
    private fun formatHz(value: Double) = if (value > 0.0) String.format(Locale.US, "%.1f", value) else "--"

    private companion object {
        const val WIDTH = 640
        const val HEIGHT = 480
        const val TIMING_TAG = "Dav2FrameTiming"
        const val RESOURCE_TAG = "Dav2Resources"
        const val EXTERNAL_ENDPOINT = "http://192.168.5.11"
        val EXTERNAL_STATUS_PERIOD_NS = TimeUnit.MILLISECONDS.toNanos(200)
        // Heatmap rendering is diagnostic/UI work; risk output remains per frame.
        val EXTERNAL_HEATMAP_PERIOD_NS = TimeUnit.SECONDS.toNanos(1)
        val DEPTH_PERIOD_NANOS = TimeUnit.MILLISECONDS.toNanos(500)

    }
}

internal data class DepthVisual(
    val heatmap: Bitmap,
    val centerMeters: Float,
    val nearMeters: Float,
    val pipelineMs: Double,
    val updateHz: Double,
    val thermalStatus: Int,
) {
    companion object {
        private const val MAP_WIDTH = 343
        private const val MAP_HEIGHT = 259
        private const val MAP_PIXELS = MAP_WIDTH * MAP_HEIGHT
        private val HALF_TO_FLOAT = FloatArray(1 shl 16) { bits ->
            android.util.Half.toFloat(bits.toShort())
        }
        private const val CENTER_CAPACITY =
            (MAP_WIDTH * 3 / 5 - MAP_WIDTH * 2 / 5) *
                (MAP_HEIGHT * 3 / 5 - MAP_HEIGHT * 2 / 5)
        private val SOURCE_ROWS = IntArray(MAP_HEIGHT) { row ->
            row * Dav2PreprocessContract.OUTPUT_HEIGHT / MAP_HEIGHT
        }
        private val SOURCE_COLUMNS = IntArray(MAP_WIDTH) { column ->
            column * Dav2PreprocessContract.OUTPUT_WIDTH / MAP_WIDTH
        }
        private val SOURCE_OFFSETS = IntArray(MAP_PIXELS) { index ->
            val row = index / MAP_WIDTH
            val column = index - row * MAP_WIDTH
            SOURCE_ROWS[row] * Dav2PreprocessContract.OUTPUT_WIDTH + SOURCE_COLUMNS[column]
        }
        private val CENTER_FLAGS = BooleanArray(MAP_PIXELS) { index ->
            val row = index / MAP_WIDTH
            val column = index - row * MAP_WIDTH
            column in MAP_WIDTH * 2 / 5 until MAP_WIDTH * 3 / 5 &&
                row in MAP_HEIGHT * 2 / 5 until MAP_HEIGHT * 3 / 5
        }

        internal class Workspace {
            val depths = FloatArray(MAP_PIXELS)
            val sampled = FloatArray(MAP_PIXELS)
            val center = FloatArray(CENTER_CAPACITY)
            val pixels = IntArray(MAP_PIXELS)
            val hsv = FloatArray(3)
        }

        fun from(
            output: java.nio.ByteBuffer,
            pipelineMs: Double,
            thermalStatus: Int,
            lastCompletedAt: Long,
            workspace: Workspace = Workspace(),
            renderHeatmap: Boolean = true,
            previousHeatmap: Bitmap? = null,
            metricsOverride: DepthRiskSummary.Result? = null,
            onMetricsReady: (centerMeters: Float, nearMeters: Float) -> Unit = { _, _ -> },
        ): DepthVisual {
            if (!renderHeatmap && metricsOverride != null) {
                onMetricsReady(metricsOverride.centerMeters, metricsOverride.nearMeters)
                val now = SystemClock.elapsedRealtimeNanos()
                return DepthVisual(
                    heatmap = requireNotNull(previousHeatmap),
                    centerMeters = metricsOverride.centerMeters,
                    nearMeters = metricsOverride.nearMeters,
                    pipelineMs = pipelineMs,
                    updateHz = if (lastCompletedAt > 0L && now > lastCompletedAt) {
                        1_000_000_000.0 / (now - lastCompletedAt)
                    } else 0.0,
                    thermalStatus = thermalStatus,
                )
            }
            val source = output.duplicate().order(java.nio.ByteOrder.nativeOrder()).apply { position(0) }.asShortBuffer()
            val center = workspace.center
            val sampled = workspace.sampled
            val depths = workspace.depths
            val pixels = workspace.pixels
            val hsv = workspace.hsv
            var centerSize = 0
            var sampledSize = 0
            for (index in 0 until MAP_PIXELS) {
                val depth = HALF_TO_FLOAT[source.get(SOURCE_OFFSETS[index]).toInt() and 0xffff]
                depths[index] = depth
                if (depth.isFinite() && depth >= 0.1f && depth <= 50f) {
                    sampled[sampledSize++] = depth
                    if (CENTER_FLAGS[index]) center[centerSize++] = depth
                }
            }
            val kotlinCenterMeters = percentile(center, centerSize, 0.5)
            val kotlinNearMeters = percentile(sampled, sampledSize, 0.1)
            onMetricsReady(kotlinCenterMeters, kotlinNearMeters)
            val centerMeters = metricsOverride?.centerMeters ?: kotlinCenterMeters
            val nearMeters = metricsOverride?.nearMeters ?: kotlinNearMeters
            if (renderHeatmap) {
                val colorNear = percentile(sampled, sampledSize, 0.05)
                val colorFar = percentile(sampled, sampledSize, 0.95)
                val logNear = if (colorNear.isFinite() && colorNear > 0f) {
                    ln(colorNear.toDouble())
                } else Double.NaN
                val logRange = if (colorFar.isFinite() && colorFar > colorNear) {
                    ln(colorFar.toDouble()) - logNear
                } else Double.NaN
                for (index in depths.indices) {
                    pixels[index] = depthColor(
                        depths[index],
                        colorNear,
                        colorFar,
                        logNear,
                        logRange,
                        hsv,
                    )
                }
            }
            val now = SystemClock.elapsedRealtimeNanos()
            val updateHz = if (lastCompletedAt > 0L && now > lastCompletedAt) {
                1_000_000_000.0 / (now - lastCompletedAt)
            } else 0.0
            return DepthVisual(
                heatmap = if (renderHeatmap) {
                    Bitmap.createBitmap(pixels, MAP_WIDTH, MAP_HEIGHT, Bitmap.Config.ARGB_8888)
                } else {
                    requireNotNull(previousHeatmap) { "previous heatmap required when rendering is skipped" }
                },
                centerMeters = centerMeters,
                nearMeters = nearMeters,
                pipelineMs = pipelineMs,
                updateHz = updateHz,
                thermalStatus = thermalStatus,
            )
        }

        private fun percentile(sorted: FloatArray, size: Int, quantile: Double): Float {
            if (size == 0) return Float.NaN
            val index = (quantile * (size - 1)).toInt().coerceIn(0, size - 1)
            return selectKth(sorted, size, index)
        }

        /** Returns the same order statistic as a full ascending sort, in-place. */
        private fun selectKth(values: FloatArray, size: Int, target: Int): Float {
            var left = 0
            var right = size - 1
            while (left < right) {
                val pivot = values[(left + right) ushr 1]
                var i = left
                var j = right
                while (i <= j) {
                    while (values[i] < pivot) i++
                    while (values[j] > pivot) j--
                    if (i <= j) {
                        val swap = values[i]
                        values[i] = values[j]
                        values[j] = swap
                        i++
                        j--
                    }
                }
                when {
                    target <= j -> right = j
                    target >= i -> left = i
                    else -> return values[target]
                }
            }
            return values[left]
        }

        private fun depthColor(
            depth: Float,
            colorNear: Float,
            colorFar: Float,
            logNear: Double,
            logRange: Double,
            hsv: FloatArray,
        ): Int {
            if (!depth.isFinite() || depth <= 0f || !colorNear.isFinite() ||
                !colorFar.isFinite() || colorFar <= colorNear || !logRange.isFinite()
            ) return Color.TRANSPARENT
            val normalized = ((ln(depth.coerceIn(colorNear, colorFar).toDouble()) - logNear) /
                logRange).toFloat().coerceIn(0f, 1f)
            val brightness = 1f - normalized * 0.48f
            hsv[0] = normalized * 240f
            hsv[1] = 0.98f
            hsv[2] = brightness
            return Color.HSVToColor(hsv)
        }
    }
}

/** Native fixed-sample risk summary; the first rendered frame is parity-checked against Kotlin. */
internal object DepthRiskSummary {
    data class Result(val centerMeters: Float, val nearMeters: Float) {
        companion object {
            fun from(values: FloatArray): Result {
                require(values.size == 2) { "invalid native risk summary result" }
                return Result(values[0], values[1])
            }
        }
    }
}

private enum class DepthDisplayMode { SPLIT, OVERLAY, RGB }

/** Temporary external-camera bridge; keeps the DA V2 input contract direct and bounded. */
private class Dav2BitmapRgbConverter(
    private val width: Int,
    private val height: Int,
) : AutoCloseable {
    private var sourcePixels = IntArray(0)
    private val output = java.nio.ByteBuffer.allocateDirect(width * height * 3)
    private val referenceOutput = java.nio.ByteBuffer.allocateDirect(width * height * 3)
    private var parityVerified = false

    fun convert(bitmap: Bitmap): java.nio.ByteBuffer {
        output.clear()
        check(nativeConvertBitmap(bitmap, output, width, height)) {
            "native Bitmap to RGB conversion failed"
        }
        output.position(0)
        output.limit(width * height * 3)
        if (!parityVerified) {
            convertReference(bitmap, referenceOutput)
            var maxAbsDiff = 0
            for (index in 0 until output.limit()) {
                val difference = kotlin.math.abs(
                    (output.get(index).toInt() and 0xff) -
                        (referenceOutput.get(index).toInt() and 0xff),
                )
                if (difference > maxAbsDiff) maxAbsDiff = difference
            }
            check(maxAbsDiff == 0) { "native Bitmap RGB parity failed: max_abs_diff=$maxAbsDiff" }
            parityVerified = true
            Log.i("Dav2Resources", "native Bitmap RGB parity passed max_abs_diff=0 bytes=${output.limit()}")
        }
        return output
    }

    private fun convertReference(bitmap: Bitmap, destination: java.nio.ByteBuffer) {
        val required = bitmap.width * bitmap.height
        if (sourcePixels.size < required) sourcePixels = IntArray(required)
        bitmap.getPixels(sourcePixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        destination.clear()
        val sourceWidth = bitmap.width
        val sourceHeight = bitmap.height
        for (y in 0 until height) {
            val sourceY = y * sourceHeight / height
            val sourceRow = sourceY * sourceWidth
            for (x in 0 until width) {
                val sourceX = x * sourceWidth / width
                val pixel = sourcePixels[sourceRow + sourceX]
                destination.put(((pixel shr 16) and 0xff).toByte())
                destination.put(((pixel shr 8) and 0xff).toByte())
                destination.put((pixel and 0xff).toByte())
            }
        }
        destination.flip()
    }

    override fun close() = Unit

    private external fun nativeConvertBitmap(
        bitmap: Bitmap,
        output: java.nio.ByteBuffer,
        outputWidth: Int,
        outputHeight: Int,
    ): Boolean

    companion object { init { System.loadLibrary("dav2_preprocess_native") } }
}

private class DepthHeatmapView(context: Context) : View(context) {
    private val bitmapPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { isFilterBitmap = true }
    private val reticlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = resources.displayMetrics.density * 2f
    }
    private var bitmap: Bitmap? = null
    private var previousBitmap: Bitmap? = null
    private var blendProgress = 1f
    private var blendAnimator: ValueAnimator? = null
    var displayMode: DepthDisplayMode = DepthDisplayMode.SPLIT
        set(value) {
            field = value
            invalidate()
        }

    init {
        isClickable = false
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
    }

    fun show(visual: DepthVisual) {
        if (bitmap == null) {
            bitmap = visual.heatmap
            blendProgress = 1f
            invalidate()
            return
        }
        blendAnimator?.cancel()
        previousBitmap = bitmap
        bitmap = visual.heatmap
        blendProgress = 0f
        blendAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = BLEND_DURATION_MS
            interpolator = LinearInterpolator()
            addUpdateListener {
                blendProgress = it.animatedValue as Float
                invalidate()
            }
            addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) {
                    previousBitmap = null
                    blendProgress = 1f
                    invalidate()
                }
            })
            start()
        }
    }

    fun clearDepth() {
        blendAnimator?.cancel()
        blendAnimator = null
        previousBitmap = null
        bitmap = null
        blendProgress = 1f
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        bitmap?.let { current ->
            val old = previousBitmap
            when (displayMode) {
                DepthDisplayMode.SPLIT -> {
                    drawBlended(canvas, old, current, Rect(width / 2, 0, width, height), 255)
                    canvas.drawLine(width / 2f, 0f, width / 2f, height.toFloat(), reticlePaint)
                }
                DepthDisplayMode.OVERLAY -> {
                    drawBlended(canvas, old, current, Rect(0, 0, width, height), 164)
                }
                DepthDisplayMode.RGB -> Unit
            }
        }
        val cx = width / 2f
        val cy = height / 2f
        val radius = resources.displayMetrics.density * 22f
        canvas.drawCircle(cx, cy, radius, reticlePaint)
        canvas.drawLine(cx - radius * 1.5f, cy, cx + radius * 1.5f, cy, reticlePaint)
        canvas.drawLine(cx, cy - radius * 1.5f, cx, cy + radius * 1.5f, reticlePaint)
    }

    override fun onDetachedFromWindow() {
        blendAnimator?.cancel()
        blendAnimator = null
        super.onDetachedFromWindow()
    }

    private fun drawBlended(
        canvas: Canvas,
        old: Bitmap?,
        current: Bitmap,
        destination: Rect,
        baseAlpha: Int,
    ) {
        if (old != null && blendProgress < 1f) {
            bitmapPaint.alpha = (baseAlpha * (1f - blendProgress)).toInt().coerceIn(0, 255)
            canvas.drawBitmap(old, null, destination, bitmapPaint)
            bitmapPaint.alpha = (baseAlpha * blendProgress).toInt().coerceIn(0, 255)
            canvas.drawBitmap(current, null, destination, bitmapPaint)
        } else {
            bitmapPaint.alpha = baseAlpha
            canvas.drawBitmap(current, null, destination, bitmapPaint)
        }
    }

    private companion object {
        const val BLEND_DURATION_MS = 110L
    }
}

private fun materializeVerifiedModel(context: Context): File {
    val output = File(context.filesDir, MODEL_ASSET)
    if (!output.isFile || output.length() != MODEL_BYTES) {
        val temporary = File(context.filesDir, "$MODEL_ASSET.partial")
        context.assets.open(MODEL_ASSET).use { input ->
            temporary.outputStream().buffered().use(input::copyTo)
        }
        check(temporary.length() == MODEL_BYTES) { "模型大小不符：${temporary.length()}" }
        if (output.exists()) check(output.delete()) { "无法替换旧模型" }
        check(temporary.renameTo(output)) { "无法保存模型" }
    }
    val digest = MessageDigest.getInstance("SHA-256")
    FileInputStream(output).buffered().use { input ->
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
    }
    val sha = digest.digest().joinToString("") { "%02X".format(it) }
    check(sha == MODEL_SHA256) { "模型哈希不符：$sha" }
    return output
}

private const val MODEL_ASSET = "model-sm8650-cached.dlc"
private const val MODEL_BYTES = 55_087_141L
private const val MODEL_SHA256 = "2BB02F37FEF177FF4B02B8EE0C416EE9FF998BCEEF9786B92959E1F682EBAA24"
