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
import android.os.SystemClock
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
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.ln

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
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startExperience()
        } else {
            status.text = "等待相机权限"
            requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST)
        }
    }

    override fun onStart() {
        super.onStart()
        lifecycleRegistry.currentState = Lifecycle.State.STARTED
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
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
        engine?.stopCamera()
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
    private val depthExecutor = Executors.newSingleThreadExecutor()
    private val active = AtomicBoolean(false)
    private val processing = AtomicBoolean(false)
    private val initializationStarted = AtomicBoolean(false)
    private val lastSubmittedAt = AtomicLong(Long.MIN_VALUE)
    private val frame = OwnedYuv420Frame(WIDTH, HEIGHT) {}
    private val power = activity.getSystemService(PowerManager::class.java)
    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    private var converter: Dav2Yuv420RgbConverter? = null
    private var preprocessor: Dav2NativePreprocessor? = null
    private var runtime: Dav2QnnCachedContext? = null
    private val visualWorkspace = DepthVisual.Workspace()
    private var lastCompletedAt = 0L

    fun startCamera() {
        active.set(true)
        if (runtime != null) {
            bindCamera()
            return
        }
        if (!initializationStarted.compareAndSet(false, true)) return
        onStatus("正在载入 QNN HTP 模型…", "模型与 FP16 合同会在启动时校验")
        depthExecutor.execute {
            try {
                val model = materializeVerifiedModel(activity)
                val nativeDir = activity.applicationInfo.nativeLibraryDir
                converter = Dav2Yuv420RgbConverter()
                preprocessor = Dav2NativePreprocessor()
                runtime = Dav2QnnCachedContext(model.absolutePath, nativeDir)
                activity.runOnUiThread {
                    onStatus("模型就绪，正在打开后置相机…", "canonical FP32 → strict FP16 → QNN cached context")
                    if (active.get()) bindCamera()
                }
            } catch (error: Throwable) {
                activity.runOnUiThread { onFailure(error.message ?: error.javaClass.simpleName) }
            }
        }
    }

    fun stopCamera() {
        active.set(false)
        analysis?.clearAnalyzer()
        analysis = null
        provider?.unbindAll()
    }

    private fun bindCamera() {
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
            val input = requireNotNull(preprocessor).preprocessFp16CanonicalStrictDirect(rgb)
            val output = requireNotNull(runtime).execute(input)
            val visual = DepthVisual.from(
                output,
                elapsedMs(startedAt),
                thermal,
                lastCompletedAt,
                visualWorkspace,
            )
            lastCompletedAt = SystemClock.elapsedRealtimeNanos()
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
        stopCamera()
        analyzerExecutor.shutdownNow()
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

    private fun formatMeters(value: Float): String =
        if (value.isFinite()) String.format(Locale.US, "%.2f m", value) else "未知"

    private fun formatMs(value: Double) = String.format(Locale.US, "%.1f", value)
    private fun formatHz(value: Double) = if (value > 0.0) String.format(Locale.US, "%.1f", value) else "--"

    private companion object {
        const val WIDTH = 640
        const val HEIGHT = 480
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
        private const val CENTER_CAPACITY =
            (MAP_WIDTH * 3 / 5 - MAP_WIDTH * 2 / 5) *
                (MAP_HEIGHT * 3 / 5 - MAP_HEIGHT * 2 / 5)
        private val SOURCE_ROWS = IntArray(MAP_HEIGHT) { row ->
            row * Dav2PreprocessContract.OUTPUT_HEIGHT / MAP_HEIGHT
        }
        private val SOURCE_COLUMNS = IntArray(MAP_WIDTH) { column ->
            column * Dav2PreprocessContract.OUTPUT_WIDTH / MAP_WIDTH
        }

        internal class Workspace {
            val depths = FloatArray(MAP_PIXELS)
            val sampled = FloatArray(MAP_PIXELS)
            val center = FloatArray(CENTER_CAPACITY)
            val pixels = IntArray(MAP_PIXELS)
        }

        fun from(
            output: java.nio.ByteBuffer,
            pipelineMs: Double,
            thermalStatus: Int,
            lastCompletedAt: Long,
            workspace: Workspace = Workspace(),
        ): DepthVisual {
            val source = output.duplicate().order(java.nio.ByteOrder.nativeOrder()).apply { position(0) }.asShortBuffer()
            val center = workspace.center
            val sampled = workspace.sampled
            val depths = workspace.depths
            val pixels = workspace.pixels
            var centerSize = 0
            var sampledSize = 0
            for (row in 0 until MAP_HEIGHT) {
                val sourceRow = SOURCE_ROWS[row]
                for (column in 0 until MAP_WIDTH) {
                    val sourceColumn = SOURCE_COLUMNS[column]
                    val depth = halfBitsToFloat(source.get(sourceRow * Dav2PreprocessContract.OUTPUT_WIDTH + sourceColumn))
                    depths[row * MAP_WIDTH + column] = depth
                    if (depth.isFinite() && depth in 0.1f..50f) {
                        sampled[sampledSize++] = depth
                        if (column in MAP_WIDTH * 2 / 5 until MAP_WIDTH * 3 / 5 &&
                            row in MAP_HEIGHT * 2 / 5 until MAP_HEIGHT * 3 / 5
                        ) center[centerSize++] = depth
                    }
                }
            }
            val colorNear = percentile(sampled, sampledSize, 0.05)
            val colorFar = percentile(sampled, sampledSize, 0.95)
            for (index in depths.indices) {
                pixels[index] = depthColor(depths[index], colorNear, colorFar)
            }
            val centerMeters = percentile(center, centerSize, 0.5)
            val nearMeters = percentile(sampled, sampledSize, 0.1)
            val now = SystemClock.elapsedRealtimeNanos()
            val updateHz = if (lastCompletedAt > 0L && now > lastCompletedAt) {
                1_000_000_000.0 / (now - lastCompletedAt)
            } else 0.0
            return DepthVisual(
                heatmap = Bitmap.createBitmap(pixels, MAP_WIDTH, MAP_HEIGHT, Bitmap.Config.ARGB_8888),
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

        private fun depthColor(depth: Float, colorNear: Float, colorFar: Float): Int {
            if (!depth.isFinite() || depth <= 0f || !colorNear.isFinite() ||
                !colorFar.isFinite() || colorFar <= colorNear
            ) return Color.TRANSPARENT
            val normalized = ((ln(depth.coerceIn(colorNear, colorFar).toDouble()) - ln(colorNear.toDouble())) /
                (ln(colorFar.toDouble()) - ln(colorNear.toDouble()))).toFloat().coerceIn(0f, 1f)
            val brightness = 1f - normalized * 0.48f
            return Color.HSVToColor(floatArrayOf(normalized * 240f, 0.98f, brightness))
        }
    }
}

private enum class DepthDisplayMode { SPLIT, OVERLAY, RGB }

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
