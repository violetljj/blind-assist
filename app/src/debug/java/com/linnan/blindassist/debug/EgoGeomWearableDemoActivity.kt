package com.linnan.blindassist.debug

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.linnan.blindassist.vision.egogeom.CameraImuFusionMode
import com.linnan.blindassist.vision.egogeom.EgoGeomContracts
import com.linnan.blindassist.vision.egogeom.EgoGeomQnnRuntimeFactory
import com.linnan.blindassist.vision.egogeom.EgoGeomRuntime
import com.linnan.blindassist.vision.egogeom.GlassAlertTier
import com.linnan.blindassist.vision.egogeom.GlassCorridorRiskEvaluator
import com.linnan.blindassist.vision.egogeom.WearableHazardChannel
import com.linnan.blindassist.vision.egogeom.WearableMountProfile
import com.linnan.blindassist.vision.egogeom.WearableRiskArbitrator
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.hypot

/**
 * Debug-only standalone live Camera + IMU + Hexagon v75 (`v4.1 + G2`) wearable obstacle demo.
 * Does NOT touch default `MainActivity` or release builds.
 * Launch via: `adb shell am start -n com.linnan.blindassist/.debug.EgoGeomWearableDemoActivity`
 */
class EgoGeomWearableDemoActivity : ComponentActivity(), SensorEventListener {

    private data class LiveHudState(
        val status: String = "正在初始化 Hexagon v75 NPU...",
        val modelName: String = "",
        val htpMs: Double = 0.0,
        val fps: Double = 0.0,
        val channel: WearableHazardChannel = WearableHazardChannel.NONE,
        val glassTier: GlassAlertTier = GlassAlertTier.NONE,
        val glassMean: Float = 0f,
        val coneScore: Float = 0f,
        val gyroScore: Float = 0f,
        val centerDepthM: Float = Float.POSITIVE_INFINITY,
        val glassMap: FloatArray? = null,
        val mountProfile: WearableMountProfile = WearableMountProfile.HEAD_MOUNTED,
        val showHeatmap: Boolean = true,
        val hapticEnabled: Boolean = true,
    )

    private var hudState by mutableStateOf(LiveHudState())
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)

    @Volatile private var runtime: EgoGeomRuntime? = null
    @Volatile private var arbitrator = WearableRiskArbitrator(
        mode = CameraImuFusionMode.ZERO_VIO_CAMERA_IMU,
        mount = WearableMountProfile.HEAD_MOUNTED,
    )
    @Volatile private var rWcLatest = floatArrayOf(1f, 0f, 0f, 0f, 0f, -1f, 0f, 1f, 0f)
    @Volatile private var gCamLatest = floatArrayOf(0f, 1f, 0f)
    @Volatile private var fwdLatest = floatArrayOf(0f, 1f, 0f)

    private var sensorManager: SensorManager? = null
    private var previewView: PreviewView? = null
    private var cameraController: LifecycleCameraController? = null
    private var lastFrameNs = 0L
    private var emaFps = 0.0
    private var lastHapticMs = 0L

    private val rgbBuf = FloatArray(EgoGeomContracts.RGB_FLOATS)
    private val rayBuf = FloatArray(EgoGeomContracts.RAYS_FLOATS)
    private val grayBuf = FloatArray(EgoGeomContracts.S4_HEIGHT * EgoGeomContracts.S4_WIDTH)
    private val s4Dep = FloatArray(EgoGeomContracts.S4_HEIGHT * EgoGeomContracts.S4_WIDTH)
    private val s4Dh = FloatArray(EgoGeomContracts.S4_HEIGHT * EgoGeomContracts.S4_WIDTH)
    private val s4Nx = FloatArray(EgoGeomContracts.S4_HEIGHT * EgoGeomContracts.S4_WIDTH)
    private val s4Nd = FloatArray(EgoGeomContracts.S4_HEIGHT * EgoGeomContracts.S4_WIDTH)
    private val s4Nz = FloatArray(EgoGeomContracts.S4_HEIGHT * EgoGeomContracts.S4_WIDTH)
    private val s4Ls = FloatArray(EgoGeomContracts.S4_HEIGHT * EgoGeomContracts.S4_WIDTH)

    private val camPermLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) startCameraIfReady() else hudState = hudState.copy(status = "需要相机权限才能启动实时演示")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

        setContent {
            MaterialTheme {
                DemoScreen(
                    state = hudState,
                    onBindPreview = { pv ->
                        previewView = pv
                        ensureCameraPermissionAndStart()
                    },
                    onToggleMount = {
                        val next = if (hudState.mountProfile.name == WearableMountProfile.HEAD_MOUNTED.name) {
                            WearableMountProfile.NECK_MOUNTED
                        } else {
                            WearableMountProfile.HEAD_MOUNTED
                        }
                        arbitrator = WearableRiskArbitrator(CameraImuFusionMode.ZERO_VIO_CAMERA_IMU, next)
                        hudState = hudState.copy(mountProfile = next)
                    },
                    onToggleHeatmap = { hudState = hudState.copy(showHeatmap = !hudState.showHeatmap) },
                    onToggleHaptic = { hudState = hudState.copy(hapticEnabled = !hudState.hapticEnabled) },
                )
            }
        }

        inferenceExecutor.execute { initRuntime() }
    }

    private fun initRuntime() {
        try {
            val baseDir = File(getExternalFilesDir(null) ?: filesDir, "egogeom-v41-g2")
            val candidates = listOf(
                File(baseDir, EgoGeomQnnRuntimeFactory.PURE_RGB_G2_BIN),
                File(baseDir, EgoGeomQnnRuntimeFactory.OMNI_G2_BIN),
                File(baseDir, EgoGeomQnnRuntimeFactory.PURE_RGB_BIN),
                File(baseDir, EgoGeomQnnRuntimeFactory.OMNI_BIN),
            )
            val model = candidates.firstOrNull { it.isFile }
            if (model == null) {
                hudState = hudState.copy(status = "未找到模型二进制: ${baseDir.absolutePath}")
                return
            }
            val rt = EgoGeomQnnRuntimeFactory.create(application, model)
            runtime = rt
            hudState = hudState.copy(
                status = if (rt.supportsGlass) "Hexagon v75 就绪 (v4.1 + G2 玻璃头)" else "Hexagon v75 就绪 (无玻璃头)",
                modelName = model.name,
            )
        } catch (e: Throwable) {
            hudState = hudState.copy(status = "NPU 初始化失败: ${e.message}")
        }
    }

    override fun onResume() {
        super.onResume()
        val sm = sensorManager ?: return
        val rot = sm.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
            ?: sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        if (rot != null) sm.registerListener(this, rot, SensorManager.SENSOR_DELAY_GAME)
    }

    override fun onPause() {
        super.onPause()
        sensorManager?.unregisterListener(this)
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceExecutor.execute {
            runCatching { runtime?.close() }
            runtime = null
        }
        inferenceExecutor.shutdown()
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_GAME_ROTATION_VECTOR &&
            event.sensor.type != Sensor.TYPE_ROTATION_VECTOR
        ) return
        val rWd = FloatArray(9)
        SensorManager.getRotationMatrixFromVector(rWd, event.values)
        // OpenCV camera frame relative to portrait Android device frame: diag(1, -1, -1)
        val rWc = floatArrayOf(
            rWd[0], -rWd[1], -rWd[2],
            rWd[3], -rWd[4], -rWd[5],
            rWd[6], -rWd[7], -rWd[8],
        )
        rWcLatest = rWc
        gCamLatest = floatArrayOf(-rWc[6], -rWc[7], -rWc[8])
        val fxW = rWc[2]; val fyW = rWc[5]
        val fn = hypot(fxW.toDouble(), fyW.toDouble()).toFloat()
        if (fn > 0.15f) {
            fwdLatest = floatArrayOf(fxW / fn, fyW / fn, 0f)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun ensureCameraPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCameraIfReady()
        } else {
            camPermLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCameraIfReady() {
        val pv = previewView ?: return
        if (cameraController != null) return
        val controller = LifecycleCameraController(this).apply {
            cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            setEnabledUseCases(CameraController.IMAGE_ANALYSIS)
            imageAnalysisBackpressureStrategy = ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
            imageAnalysisOutputImageFormat = ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
            setImageAnalysisAnalyzer(inferenceExecutor, ::analyzeFrame)
            bindToLifecycle(this@EgoGeomWearableDemoActivity)
        }
        cameraController = controller
        pv.controller = controller
    }

    private fun analyzeFrame(proxy: ImageProxy) {
        if (!busy.compareAndSet(false, true)) {
            proxy.close()
            return
        }
        try {
            val rt = runtime
            if (rt == null) return
            val rawBmp = proxy.toBitmap()
            val rotDeg = proxy.imageInfo.rotationDegrees
            val bmp = if (rotDeg != 0) {
                val m = Matrix().apply { postRotate(rotDeg.toFloat()) }
                Bitmap.createBitmap(rawBmp, 0, 0, rawBmp.width, rawBmp.height, m, false).also { rawBmp.recycle() }
            } else {
                rawBmp
            }
            val pixels = IntArray(bmp.width * bmp.height)
            bmp.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)
            val bw = bmp.width; val bh = bmp.height
            bmp.recycle()

            EgoGeomContracts.preprocessArgbToNchw(pixels, bw, bh, rgbBuf, grayBuf)
            // Approximate S24U main wide camera pinhole focal (~68 deg HFOV -> fx ~ 285 px on 384w)
            val fx = 285f; val fy = 285f; val cx = 192f; val cy = 144f
            val gCam = gCamLatest
            val rWc = rWcLatest
            val fwd = fwdLatest
            EgoGeomContracts.buildGravityAlignedRays(fx, fy, cx, cy, gCam[0], gCam[1], gCam[2], rayBuf)
            val hostFocal = EgoGeomContracts.analyticalHostFocal(fx.toDouble(), fy.toDouble(), cx.toDouble(), cy.toDouble())

            val out = rt.infer(rgbBuf, rayBuf, hostFocal, tofGridS4 = null, hasTof = false)

            // Subsample 288x384 -> 72x96 s4 + center patch median-ish depth
            val h4 = EgoGeomContracts.S4_HEIGHT; val w4 = EgoGeomContracts.S4_WIDTH
            val plane = EgoGeomContracts.DEPTH_FLOATS
            var centerSum = 0.0; var centerCnt = 0
            for (y in 0 until h4) {
                val sr = (y * 4) * EgoGeomContracts.WIDTH; val dr = y * w4
                for (x in 0 until w4) {
                    val si = sr + x * 4; val di = dr + x
                    val d = out.depthM[si]
                    s4Dep[di] = d; s4Dh[di] = out.deltaHM[si]; s4Ls[di] = out.logSigma[si]
                    s4Nx[di] = out.normalGrav[si]; s4Nd[di] = out.normalGrav[plane + si]; s4Nz[di] = out.normalGrav[2 * plane + si]
                    if (y in 28..44 && x in 38..58 && d.isFinite()) {
                        centerSum += d; centerCnt++
                    }
                }
            }
            val centerD = if (centerCnt > 0) (centerSum / centerCnt).toFloat() else Float.POSITIVE_INFINITY

            val arb = arbitrator.step(
                solidM6ParRisk = 0f,
                depthS4 = s4Dep, deltaHS4 = s4Dh,
                nrmRightS4 = s4Nx, nrmDownS4 = s4Nd, nrmFwdS4 = s4Nz,
                logSigmaS4 = s4Ls, grayS4 = grayBuf,
                fx = fx, fy = fy, cx = cx, cy = cy,
                rWc = rWc, strictFwd = fwd,
                glassProbS16 = out.glassProbS16,
            )

            val nowNs = SystemClock.elapsedRealtimeNanos()
            if (lastFrameNs > 0L) {
                val inst = 1e9 / (nowNs - lastFrameNs).coerceAtLeast(1_000_000L)
                emaFps = if (emaFps <= 0.1) inst else 0.8 * emaFps + 0.2 * inst
            }
            lastFrameNs = nowNs

            if (hudState.hapticEnabled && arb.primaryChannel != WearableHazardChannel.NONE) {
                maybeVibrate(arb.primaryChannel)
            }

            hudState = hudState.copy(
                htpMs = out.executeMs,
                fps = emaFps,
                channel = arb.primaryChannel,
                glassTier = arb.glass.tier,
                glassMean = arb.glass.corridorMeanProb,
                coneScore = arb.narrowConeScore,
                gyroScore = arb.gyroTtcScore,
                centerDepthM = centerD,
                glassMap = out.glassProbS16?.copyOf(),
            )
        } finally {
            proxy.close()
            busy.set(false)
        }
    }

    private fun maybeVibrate(channel: WearableHazardChannel) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastHapticMs < 1100L) return
        lastHapticMs = now
        val ms = when (channel) {
            WearableHazardChannel.GLASS_HAZARD -> 130L
            WearableHazardChannel.OVERHANG_LOOMING -> 45L
            WearableHazardChannel.SOLID_OBSTACLE -> 80L
            WearableHazardChannel.NONE -> return
        }
        val vib = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager)?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }
        vib?.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE))
    }

    @Composable
    private fun DemoScreen(
        state: LiveHudState,
        onBindPreview: (PreviewView) -> Unit,
        onToggleMount: () -> Unit,
        onToggleHeatmap: () -> Unit,
        onToggleHaptic: () -> Unit,
    ) {
        val bannerColor = when (state.channel) {
            WearableHazardChannel.GLASS_HAZARD -> Color(0xFFD32F2F)
            WearableHazardChannel.OVERHANG_LOOMING -> Color(0xFFF57C00)
            WearableHazardChannel.SOLID_OBSTACLE -> Color(0xFFC2185B)
            WearableHazardChannel.NONE -> Color(0xFF2E7D32)
        }
        val bannerText = when (state.channel) {
            WearableHazardChannel.GLASS_HAZARD -> "⚠ 注意前方玻璃 (${state.glassTier.name})"
            WearableHazardChannel.OVERHANG_LOOMING -> "⚠ 注意上方悬空 / 逼近"
            WearableHazardChannel.SOLID_OBSTACLE -> "⚠ 前方实心障碍"
            WearableHazardChannel.NONE -> "通道畅通 · Camera + IMU 实时运行中"
        }

        Surface(modifier = Modifier.fillMaxSize(), color = Color.Black) {
            Box(modifier = Modifier.fillMaxSize()) {
                AndroidView(
                    modifier = Modifier.fillMaxSize(),
                    factory = { ctx ->
                        PreviewView(ctx).apply {
                            scaleType = PreviewView.ScaleType.FILL_CENTER
                            onBindPreview(this)
                        }
                    },
                )

                // Translucent 18x24 G2 glass probability heatmap overlay
                if (state.showHeatmap && state.glassMap != null) {
                    val gMap = state.glassMap
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        val cw = size.width / GlassCorridorRiskEvaluator.S16_WIDTH
                        val ch = size.height / GlassCorridorRiskEvaluator.S16_HEIGHT
                        for (r in 0 until GlassCorridorRiskEvaluator.S16_HEIGHT) {
                            for (c in 0 until GlassCorridorRiskEvaluator.S16_WIDTH) {
                                val p = gMap[r * GlassCorridorRiskEvaluator.S16_WIDTH + c]
                                if (p >= 0.25f) {
                                    val alpha = ((p - 0.20f) * 0.72f).coerceIn(0.05f, 0.65f)
                                    val color = if (p >= 0.70f) Color(0xFFFF1744) else Color(0xFFFFD600)
                                    drawRect(
                                        color = color.copy(alpha = alpha),
                                        topLeft = Offset(c * cw, r * ch),
                                        size = Size(cw + 1f, ch + 1f),
                                    )
                                }
                            }
                        }
                        // Center corridor visual guide
                        drawRect(
                            color = Color.Green.copy(alpha = 0.65f),
                            topLeft = Offset(size.width * 0.28f, size.height * 0.25f),
                            size = Size(size.width * 0.44f, size.height * 0.50f),
                            style = Stroke(width = 3f),
                        )
                    }
                }

                // Top status banner
                Column(
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .fillMaxWidth()
                        .padding(top = 40.dp, start = 14.dp, end = 14.dp),
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(bannerColor.copy(alpha = 0.88f), RoundedCornerShape(12.dp))
                            .padding(horizontal = 14.dp, vertical = 10.dp),
                    ) {
                        Text(bannerText, color = Color.White, fontSize = 18.sp, fontWeight = FontWeight.Bold)
                    }
                    Spacer(Modifier.height(6.dp))
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color.Black.copy(alpha = 0.65f), RoundedCornerShape(10.dp))
                            .padding(10.dp),
                    ) {
                        Text(
                            text = "${state.status}\n" +
                                "HTP: ${"%.1f".format(state.htpMs)} ms | ${"%.1f".format(state.fps)} FPS | 中心估距: ${"%.2f".format(state.centerDepthM)} m\n" +
                                "玻璃均值: ${"%.3f".format(state.glassMean)} (${state.glassTier}) | 窄锥TTC: ${"%.2f".format(state.coneScore)} | 陀螺TTC: ${"%.2f".format(state.gyroScore)}",
                            color = Color.White,
                            fontSize = 12.sp,
                            lineHeight = 17.sp,
                        )
                    }
                }

                // Bottom controls: Mount toggle (Head vs Neck), Heatmap toggle, Haptic toggle
                Row(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .padding(bottom = 28.dp, start = 12.dp, end = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Button(
                        onClick = onToggleMount,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1565C0)),
                    ) {
                        Text(
                            if (state.mountProfile.name == WearableMountProfile.HEAD_MOUNTED.name) "头戴模式" else "颈挂模式",
                            fontSize = 13.sp,
                        )
                    }
                    Button(
                        onClick = onToggleHeatmap,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (state.showHeatmap) Color(0xFF6A1B9A) else Color(0xFF424242),
                        ),
                    ) {
                        Text(if (state.showHeatmap) "玻璃热图:开" else "玻璃热图:关", fontSize = 13.sp)
                    }
                    Button(
                        onClick = onToggleHaptic,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (state.hapticEnabled) Color(0xFF00695C) else Color(0xFF424242),
                        ),
                    ) {
                        Text(if (state.hapticEnabled) "震动:开" else "震动:关", fontSize = 13.sp)
                    }
                }
            }
        }
    }
}
