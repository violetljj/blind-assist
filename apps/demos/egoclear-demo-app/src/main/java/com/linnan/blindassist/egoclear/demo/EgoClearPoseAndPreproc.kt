package com.linnan.blindassist.egoclear.demo

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.SystemClock
import android.view.Surface
import androidx.camera.core.ImageProxy
import com.linnan.blindassist.egoclear.EgoClearStreamingEngine
import kotlin.math.PI
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Hardware IMU + step-detector causal pose tracker for handheld EgoClear-v6.6 streaming.
 *
 * Produces exact OpenCV-camera-to-Z-up-world rotation (R_c2w) matching any screen orientation,
 * smoothed causal walking heading (fwdWorld = [fx, fy, 0]), and planar step odometry (tWorld, ds).
 */
class EgoClearPoseTracker(context: Context) : SensorEventListener {

    data class PoseSnapshot(
        val rC2w: FloatArray,
        val tWorld: FloatArray,
        val fwdWorld: FloatArray,
        val dsMeters: Float,
        val pitchDeg: Float,
        val rollDeg: Float,
        val gravityRotation: Int,   // Surface.ROTATION_0 / 90 / 180 / 270 derived directly from IMU gravity
        val imuReady: Boolean,
        val walking: Boolean,
        val stepCount: Int,
    )

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rotSensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
    private val stepSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR)

    private val lock = Any()
    private val rotWorldDev = FloatArray(9).apply {
        // Default upright portrait facing +Y (North) before first IMU sample arrives
        this[0] = 1f; this[5] = -1f; this[7] = 1f
    }
    private var hasImu = false
    private var posX = 0f
    private var posY = 0f
    private var fwdX = 0f
    private var fwdY = 1f
    private var hasFwd = false

    private var pendingStepMeters = 0f
    private var lastStepElapsedMs = 0L
    private var lastSampleElapsedMs = 0L
    private var totalSteps = 0

    fun start() {
        rotSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        stepSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    fun resetOdometry() {
        synchronized(lock) {
            posX = 0f
            posY = 0f
            pendingStepMeters = 0f
            hasFwd = false
            totalSteps = 0
            lastSampleElapsedMs = 0L
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ROTATION_VECTOR, Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                val r = FloatArray(9)
                SensorManager.getRotationMatrixFromVector(r, event.values)
                synchronized(lock) {
                    System.arraycopy(r, 0, rotWorldDev, 0, 9)
                    hasImu = true
                }
            }
            Sensor.TYPE_STEP_DETECTOR -> {
                val now = SystemClock.elapsedRealtime()
                synchronized(lock) {
                    // Spread each walking step (~0.65 m) smoothly across upcoming 5 Hz frames
                    pendingStepMeters = min(pendingStepMeters + STEP_LENGTH_M, 1.3f)
                    lastStepElapsedMs = now
                    totalSteps += 1
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    fun sampleForFrame(): PoseSnapshot {
        val now = SystemClock.elapsedRealtime()
        synchronized(lock) {
            val rW = rotWorldDev
            // Determine true gravity-upright orientation directly from IMU world-UP vector in device coords
            // (row 2 of R_world_dev = [rW[6], rW[7], rW[8]]). Works even when Android Auto-Rotate is locked!
            val upX = rW[6]
            val upY = rW[7]
            val gravRot = if (kotlin.math.abs(upY) >= kotlin.math.abs(upX)) {
                if (upY >= 0f) Surface.ROTATION_0 else Surface.ROTATION_180
            } else {
                if (upX >= 0f) Surface.ROTATION_90 else Surface.ROTATION_270
            }
            val rC2w = FloatArray(9)
            // Exact right-handed (det = +1) mapping from gravity-upright OpenCV camera (x right, y down, z fwd)
            // to Android device frame (x right, y up, z out-of-screen), composed with R_world_dev.
            when (gravRot) {
                Surface.ROTATION_90 -> {
                    // Top of phone points left: x_c = -y_d, y_c = -x_d, z_c = -z_d
                    for (row in 0..2) {
                        val b = row * 3
                        rC2w[b + 0] = -rW[b + 1]
                        rC2w[b + 1] = -rW[b + 0]
                        rC2w[b + 2] = -rW[b + 2]
                    }
                }
                Surface.ROTATION_180 -> {
                    // Upside-down portrait: x_c = -x_d, y_c = +y_d, z_c = -z_d
                    for (row in 0..2) {
                        val b = row * 3
                        rC2w[b + 0] = -rW[b + 0]
                        rC2w[b + 1] = rW[b + 1]
                        rC2w[b + 2] = -rW[b + 2]
                    }
                }
                Surface.ROTATION_270 -> {
                    // Top of phone points right: x_c = +y_d, y_c = +x_d, z_c = -z_d
                    for (row in 0..2) {
                        val b = row * 3
                        rC2w[b + 0] = rW[b + 1]
                        rC2w[b + 1] = rW[b + 0]
                        rC2w[b + 2] = -rW[b + 2]
                    }
                }
                else -> {
                    // Natural portrait (ROTATION_0): x_c = +x_d, y_c = -y_d, z_c = -z_d
                    for (row in 0..2) {
                        val b = row * 3
                        rC2w[b + 0] = rW[b + 0]
                        rC2w[b + 1] = -rW[b + 1]
                        rC2w[b + 2] = -rW[b + 2]
                    }
                }
            }

            // Optical forward in world frame = column 2 of R_c2w
            val camFwdX = rC2w[2]
            val camFwdY = rC2w[5]
            val camFwdZ = rC2w[8].coerceIn(-1f, 1f)
            val horizNorm = hypot(camFwdX.toDouble(), camFwdY.toDouble()).toFloat()
            if (horizNorm > 0.15f) {
                val ux = camFwdX / horizNorm
                val uy = camFwdY / horizNorm
                if (!hasFwd) {
                    fwdX = ux
                    fwdY = uy
                    hasFwd = true
                } else {
                    // Smooth horizontal walking heading to suppress hand shake
                    val alpha = 0.35f
                    val mx = (1f - alpha) * fwdX + alpha * ux
                    val my = (1f - alpha) * fwdY + alpha * uy
                    val mn = hypot(mx.toDouble(), my.toDouble()).toFloat().coerceAtLeast(1e-5f)
                    fwdX = mx / mn
                    fwdY = my / mn
                }
            }

            val dtSec = if (lastSampleElapsedMs > 0L) {
                ((now - lastSampleElapsedMs) / 1000.0f).coerceIn(0.02f, 0.5f)
            } else {
                0f
            }
            lastSampleElapsedMs = now

            val ds = if (pendingStepMeters > 1e-4f && dtSec > 0f) {
                val maxAdvance = WALK_SPEED_MPS * dtSec
                val advance = min(pendingStepMeters, maxAdvance).coerceIn(0f, 0.45f)
                pendingStepMeters = max(0f, pendingStepMeters - advance)
                posX += fwdX * advance
                posY += fwdY * advance
                advance
            } else {
                0f
            }

            val pitchDeg = (asin(camFwdZ.toDouble()) * (180.0 / PI)).toFloat()
            val rollDeg = (atan2(rC2w[6].toDouble(), (-rC2w[7]).toDouble()) * (180.0 / PI)).toFloat()
            val walking = (now - lastStepElapsedMs) < 1_400L && lastStepElapsedMs > 0L

            return PoseSnapshot(
                rC2w = rC2w,
                tWorld = floatArrayOf(posX, posY, 0f),
                fwdWorld = floatArrayOf(fwdX, fwdY, 0f),
                dsMeters = ds,
                pitchDeg = pitchDeg,
                rollDeg = rollDeg,
                gravityRotation = gravRot,
                imuReady = hasImu,
                walking = walking,
                stepCount = totalSteps,
            )
        }
    }

    companion object {
        private const val STEP_LENGTH_M = 0.65f
        private const val WALK_SPEED_MPS = 1.25f
    }
}

/**
 * Zero-allocation YUV_420_888 -> upright 16:9 center-crop -> 518x294 ImageNet NCHW preprocessor
 * paired with Camera2 physical intrinsics scaling.
 */
class EgoClearCameraPreprocessor(context: Context) {

    // Normalized focal lengths relative to the raw sensor active array dimensions
    private val normFxSensor: Float
    private val normFySensor: Float
    private val sensorAspect: Float
    val sensorOrientationDeg: Int

    private val rgbBuffer = FloatArray(EgoClearStreamingEngine.RGB_FLOATS)

    init {
        var nFx = 0.72f
        var nFy = 0.96f
        var sAspect = 4f / 3f
        var sOrient = 90
        try {
            val cm = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            for (id in cm.cameraIdList) {
                val chars = cm.getCameraCharacteristics(id)
                val facing = chars.get(CameraCharacteristics.LENS_FACING)
                if (facing != CameraCharacteristics.LENS_FACING_BACK) continue
                sOrient = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
                val active = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                val intr = chars.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
                val focals = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                val phys = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                if (active != null && active.width() > 0 && active.height() > 0) {
                    sAspect = active.width().toFloat() / active.height().toFloat()
                    if (intr != null && intr.size >= 4 && intr[0] > 10f && intr[1] > 10f) {
                        nFx = intr[0] / active.width().toFloat()
                        nFy = intr[1] / active.height().toFloat()
                    } else if (focals != null && focals.isNotEmpty() && phys != null && phys.width > 0f && phys.height > 0f) {
                        nFx = focals[0] / phys.width
                        nFy = focals[0] / phys.height
                    }
                }
                break
            }
        } catch (_: Throwable) {
            // Keep sensible ~69 deg HFOV fallback
        }
        normFxSensor = nFx.coerceIn(0.35f, 2.0f)
        normFySensor = nFy.coerceIn(0.35f, 2.0f)
        sensorAspect = sAspect.coerceIn(1.0f, 2.5f)
        sensorOrientationDeg = ((sOrient % 360) + 360) % 360
    }

    data class PreprocResult(
        val rgbNchw: FloatArray,
        val kScaled: FloatArray,
        val preprocMs: Double,
        val meanLuma: Float,
        val appliedRotDeg: Int,
    )

    /**
     * Converts CameraX YUV_420_888 ImageProxy into gravity-upright ImageNet-normalized NCHW [1,3,294,518]
     * plus exact K_scaled [fx, fy, cx, cy]. Uses IMU gravityRotation rather than display lock.
     */
    fun process(image: ImageProxy, zoomRatio: Float = 1.0f, gravityRotation: Int = Surface.ROTATION_0): PreprocResult {
        val t0 = SystemClock.elapsedRealtimeNanos()
        val srcW = image.width
        val srcH = image.height
        val gravDeg = when (gravityRotation) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        val rot = ((sensorOrientationDeg - gravDeg) % 360 + 360) % 360
        val swap = (rot == 90 || rot == 270)
        val upW = if (swap) srcH else srcW
        val upH = if (swap) srcW else srcH

        val targetAspect = EgoClearStreamingEngine.RGB_W.toFloat() / EgoClearStreamingEngine.RGB_H.toFloat()
        val upAspect = upW.toFloat() / upH.toFloat()
        val cropW: Float
        val cropH: Float
        val cropX0: Float
        val cropY0: Float
        if (upAspect > targetAspect) {
            cropH = upH.toFloat()
            cropW = cropH * targetAspect
            cropX0 = (upW - cropW) * 0.5f
            cropY0 = 0f
        } else {
            cropW = upW.toFloat()
            cropH = cropW / targetAspect
            cropX0 = 0f
            cropY0 = (upH - cropH) * 0.5f
        }

        // Exact K_scaled: active array -> raw YUV buffer (uniform crop-to-fit) -> upright -> 16:9 crop -> 518x294
        val rawAspect = srcW.toFloat() / srcH.toFloat()
        val fxRaw: Float
        val fyRaw: Float
        if (rawAspect >= sensorAspect) {
            // Raw YUV keeps full sensor width, crops height vertically
            fxRaw = normFxSensor * srcW
            fyRaw = normFySensor * (srcW / sensorAspect)
        } else {
            // Raw YUV keeps full sensor height, crops width horizontally
            fxRaw = normFxSensor * (srcH * sensorAspect)
            fyRaw = normFySensor * srcH
        }
        val safeZoom = zoomRatio.coerceIn(0.45f, 4.0f)
        val fxUp = (if (swap) fyRaw else fxRaw) * safeZoom
        val fyUp = (if (swap) fxRaw else fyRaw) * safeZoom
        val fxScaled = fxUp * (EgoClearStreamingEngine.RGB_W.toFloat() / cropW)
        val fyScaled = fyUp * (EgoClearStreamingEngine.RGB_H.toFloat() / cropH)
        val kScaled = floatArrayOf(
            fxScaled,
            fyScaled,
            EgoClearStreamingEngine.RGB_W * 0.5f,
            EgoClearStreamingEngine.RGB_H * 0.5f,
        )

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        val yRowStride = yPlane.rowStride
        val yPixStride = yPlane.pixelStride
        val uRowStride = uPlane.rowStride
        val uPixStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixStride = vPlane.pixelStride

        val outW = EgoClearStreamingEngine.RGB_W
        val outH = EgoClearStreamingEngine.RGB_H
        val planeSize = outW * outH
        val out = rgbBuffer

        // Fast C++ DirectBuffer JNI path (~1.0 ms on Snapdragon 8 Gen 3 vs ~15 ms Kotlin loop)
        val nativeLuma = com.linnan.blindassist.egoclear.EgoClearNativePreproc.preprocessYuv420ToNchw(
            yBuf, yRowStride, yPixStride,
            uBuf, uRowStride, uPixStride,
            vBuf, vRowStride, vPixStride,
            srcW, srcH, rot,
            cropX0, cropY0, cropW, cropH,
            outW, outH, out,
        )
        if (nativeLuma >= 0f) {
            val ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
            return PreprocResult(
                rgbNchw = out,
                kScaled = kScaled,
                preprocMs = ms,
                meanLuma = nativeLuma,
                appliedRotDeg = rot,
            )
        }

        val scaleX = cropW / outW.toFloat()
        val scaleY = cropH / outH.toFloat()
        val maxSx = srcW - 1
        val maxSy = srcH - 1

        var outIdx = 0
        var lumaSum = 0L
        for (oy in 0 until outH) {
            val uy = cropY0 + (oy + 0.5f) * scaleY
            for (ox in 0 until outW) {
                val ux = cropX0 + (ox + 0.5f) * scaleX
                val sx: Int
                val sy: Int
                when (rot) {
                    90 -> {
                        sx = uy.toInt().coerceIn(0, maxSx)
                        sy = (maxSy - ux.toInt()).coerceIn(0, maxSy)
                    }
                    180 -> {
                        sx = (maxSx - ux.toInt()).coerceIn(0, maxSx)
                        sy = (maxSy - uy.toInt()).coerceIn(0, maxSy)
                    }
                    270 -> {
                        sx = (maxSx - uy.toInt()).coerceIn(0, maxSx)
                        sy = ux.toInt().coerceIn(0, maxSy)
                    }
                    else -> {
                        sx = ux.toInt().coerceIn(0, maxSx)
                        sy = uy.toInt().coerceIn(0, maxSy)
                    }
                }

                val yVal = (yBuf.get(sy * yRowStride + sx * yPixStride).toInt() and 0xFF)
                lumaSum += yVal
                val uvX = sx shr 1
                val uvY = sy shr 1
                val uVal = (uBuf.get(uvY * uRowStride + uvX * uPixStride).toInt() and 0xFF) - 128
                val vVal = (vBuf.get(uvY * vRowStride + uvX * vPixStride).toInt() and 0xFF) - 128

                // Fast integer BT.601 YUV -> RGB [0, 255]
                var r = yVal + ((1436 * vVal) shr 10)
                var g = yVal - ((352 * uVal + 731 * vVal) shr 10)
                var b = yVal + ((1815 * uVal) shr 10)
                if (r < 0) r = 0 else if (r > 255) r = 255
                if (g < 0) g = 0 else if (g > 255) g = 255
                if (b < 0) b = 0 else if (b > 255) b = 255

                out[outIdx] = r * INV_255_STD_R - MEAN_OVER_STD_R
                out[planeSize + outIdx] = g * INV_255_STD_G - MEAN_OVER_STD_G
                out[2 * planeSize + outIdx] = b * INV_255_STD_B - MEAN_OVER_STD_B
                outIdx++
            }
        }

        val ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
        val meanLuma = lumaSum.toFloat() / planeSize.toFloat()
        return PreprocResult(
            rgbNchw = out,
            kScaled = kScaled,
            preprocMs = ms,
            meanLuma = meanLuma,
            appliedRotDeg = rot,
        )
    }

    companion object {
        private const val INV_255_STD_R = 1.0f / (255.0f * 0.229f)
        private const val INV_255_STD_G = 1.0f / (255.0f * 0.224f)
        private const val INV_255_STD_B = 1.0f / (255.0f * 0.225f)
        private const val MEAN_OVER_STD_R = 0.485f / 0.229f
        private const val MEAN_OVER_STD_G = 0.456f / 0.224f
        private const val MEAN_OVER_STD_B = 0.406f / 0.225f
    }
}
