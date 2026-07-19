package com.linnan.blindassist.benchmark

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot

@RunWith(AndroidJUnit4::class)
class GravityReprojectionAuditTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun captureAndAutomaticallyComparePredictedGravityWithSceneLines() {
        assertTrue("OpenCV local runtime failed to load", OpenCVLoader.initLocal())
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        assertNotNull("rotation-vector sensor unavailable", rotationSensor)
        val sensorSamples = mutableListOf<RotationSample>()
        val sensorThread = HandlerThread("r836-gravity-reprojection").apply { start() }
        val sensorListener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                val quaternionWxyz = FloatArray(4).also { SensorManager.getQuaternionFromVector(it, event.values) }
                synchronized(sensorSamples) { sensorSamples += RotationSample(event.timestamp, quaternionWxyz.map(Float::toDouble).toDoubleArray()) }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
        }
        assertTrue(
            "rotation-vector listener registration failed",
            sensorManager.registerListener(sensorListener, rotationSensor, SensorManager.SENSOR_DELAY_GAME, Handler(sensorThread.looper))
        )
        Thread.sleep(SENSOR_LEAD_IN_MS)

        val provider = ProcessCameraProvider.getInstance(context).get(PROVIDER_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        val owner = TestLifecycleOwner()
        val executor = Executors.newSingleThreadExecutor()
        val latch = CountDownLatch(REQUIRED_CAMERA_FRAMES)
        var selectedFrame: CapturedFrame? = null
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(productionResolutionSelector())
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        var cameraId = "unknown"
        var frameIndex = 0
        analysis.setAnalyzer(executor) { image ->
            try {
                frameIndex += 1
                if (frameIndex == SELECTED_FRAME_INDEX) selectedFrame = CapturedFrame.from(image)
                latch.countDown()
            } finally {
                image.close()
            }
        }

        try {
            instrumentation.runOnMainSync {
                provider.unbindAll()
                owner.resume()
                val camera = provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
                cameraId = Camera2CameraInfo.from(camera.cameraInfo).cameraId
            }
            assertTrue("insufficient camera frames", latch.await(CAPTURE_TIMEOUT_SECONDS, TimeUnit.SECONDS))
        } finally {
            instrumentation.runOnMainSync {
                provider.unbindAll()
                owner.destroy()
            }
            analysis.clearAnalyzer()
            executor.shutdown()
            executor.awaitTermination(2, TimeUnit.SECONDS)
            sensorManager.unregisterListener(sensorListener)
            sensorThread.quitSafely()
            sensorThread.join(2_000)
        }

        val frame = requireNotNull(selectedFrame)
        val samples = synchronized(sensorSamples) { sensorSamples.sortedBy { it.timestampNs } }
        val previous = requireNotNull(samples.lastOrNull { it.timestampNs <= frame.timestampNs })
        val next = requireNotNull(samples.firstOrNull { it.timestampNs >= frame.timestampNs })
        val interpolation = if (next.timestampNs == previous.timestampNs) 0.0 else
            (frame.timestampNs - previous.timestampNs).toDouble() / (next.timestampNs - previous.timestampNs)
        val deviceToWorldQuaternion = slerp(previous.quaternionWxyz, next.quaternionWxyz, interpolation)
        val deviceToWorld = quaternionWxyzToMatrix(deviceToWorldQuaternion)

        val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)
        val lensPoseXyzw = requireNotNull(characteristics[CameraCharacteristics.LENS_POSE_ROTATION]).map(Float::toDouble).toDoubleArray()
        val sensorIntrinsics = requireNotNull(characteristics[CameraCharacteristics.LENS_INTRINSIC_CALIBRATION]).map(Float::toDouble).toDoubleArray()
        val rawPose = AndroidCameraPoseComposer.compose(deviceToWorld, lensPoseXyzw)
        assertTrue(rawPose.failureReason, rawPose.valid)
        val geometry = CameraAnalysisGeometryMapper.map(
            frame.sensorToBuffer,
            sensorIntrinsics,
            frame.width,
            frame.height,
            frame.rotationDegrees,
            rawPose.worldEnuToCameraSensor
        )
        assertTrue(geometry.failureReason, geometry.valid)

        val displayBitmap = frame.toDisplayBitmap()
        val lineAudit = auditSceneLines(displayBitmap, geometry)
        val outputDir = File(requireNotNull(context.getExternalFilesDir(null)), "r836-gravity-reprojection").apply { mkdirs() }
        val imageFile = File(outputDir, "r836_display_frame.png")
        FileOutputStream(imageFile).use { displayBitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        val metadataFile = File(outputDir, "r836_gravity_reprojection.json")
        val report = JSONObject()
            .put("schema", "blindassist_gravity_reprojection_audit_v1")
            .put("camera_id", cameraId)
            .put("frame_timestamp_ns", frame.timestampNs)
            .put("rotation_previous_timestamp_ns", previous.timestampNs)
            .put("rotation_next_timestamp_ns", next.timestampNs)
            .put("rotation_interpolation_factor", interpolation)
            .put("device_to_world_quaternion_wxyz", JSONArray(deviceToWorldQuaternion.toList()))
            .put("display_size", JSONArray(listOf(geometry.displayWidthPx, geometry.displayHeightPx)))
            .put("display_intrinsics_fx_fy_cx_cy", JSONArray(listOf(geometry.fxPx, geometry.fyPx, geometry.cxPx, geometry.cyPx)))
            .put("world_to_display_camera", JSONArray(geometry.worldEnuToDisplayCamera.toList()))
            .put("predicted_world_up_camera", JSONArray(lineAudit.worldUpCamera.toList()))
            .put("predicted_vertical_vanishing_point", JSONArray(lineAudit.vanishingPoint.toList()))
            .put("scene_line_metrics", lineAudit.toJson())
            .put("display_frame_path", imageFile.absolutePath)
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("automatic_scene_evidence_only", true)
                .put("real_reprojection_validated", lineAudit.informative && lineAudit.pass)
                .put("app_runtime_authorized", false)
                .put("production_authorized", false))
        metadataFile.writeText(report.toString(2), Charsets.UTF_8)
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("display frame was not persisted", imageFile.isFile && imageFile.length() > 0)
        assertTrue("metadata was not persisted", metadataFile.isFile && metadataFile.length() > 0)
        displayBitmap.recycle()
    }

    private fun auditSceneLines(bitmap: Bitmap, geometry: AnalysisProjectionGeometry): LineAudit {
        val rgba = Mat()
        val gray = Mat()
        val edges = Mat()
        val lines = Mat()
        try {
            Utils.bitmapToMat(bitmap, rgba)
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.GaussianBlur(gray, gray, org.opencv.core.Size(5.0, 5.0), 1.0)
            Imgproc.Canny(gray, edges, 60.0, 160.0)
            Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 35, 35.0, 12.0)
            val up = doubleArrayOf(
                geometry.worldEnuToDisplayCamera[2],
                geometry.worldEnuToDisplayCamera[5],
                geometry.worldEnuToDisplayCamera[8]
            )
            val finiteVp = abs(up[2]) > 1e-4
            val vp = if (finiteVp) doubleArrayOf(
                geometry.fxPx * up[0] / up[2] + geometry.cxPx,
                geometry.fyPx * up[1] / up[2] + geometry.cyPx
            ) else doubleArrayOf(Double.NaN, Double.NaN)
            val candidates = ArrayList<LineMetric>()
            for (row in 0 until lines.rows()) {
                val line = lines.get(row, 0) ?: continue
                if (line.size < 4) continue
                val dx = line[2] - line[0]
                val dy = line[3] - line[1]
                val length = hypot(dx, dy)
                if (length < 35.0) continue
                val midpointX = (line[0] + line[2]) / 2.0
                val midpointY = (line[1] + line[3]) / 2.0
                val expectedX = if (finiteVp) vp[0] - midpointX else geometry.fxPx * up[0]
                val expectedY = if (finiteVp) vp[1] - midpointY else geometry.fyPx * up[1]
                val denominator = length * hypot(expectedX, expectedY)
                if (denominator <= 1e-9) continue
                val cosine = abs((dx * expectedX + dy * expectedY) / denominator).coerceIn(0.0, 1.0)
                candidates += LineMetric(length, Math.toDegrees(acos(cosine)))
            }
            val totalLength = candidates.sumOf { it.lengthPx }
            val aligned = candidates.filter { it.angleErrorDegrees <= ALIGNMENT_THRESHOLD_DEGREES }
            val alignedLength = aligned.sumOf { it.lengthPx }
            val support = if (totalLength > 0.0) alignedLength / totalLength else 0.0
            val informative = candidates.size >= 5 && aligned.size >= 2 && alignedLength >= 100.0
            val pass = informative && support >= MINIMUM_ALIGNED_LENGTH_FRACTION
            return LineAudit(
                up, vp, finiteVp, candidates.size, aligned.size, totalLength, alignedLength, support,
                candidates.map { it.angleErrorDegrees }.sorted().take(5), informative, pass
            )
        } finally {
            rgba.release(); gray.release(); edges.release(); lines.release()
        }
    }

    private data class LineMetric(val lengthPx: Double, val angleErrorDegrees: Double)

    private data class LineAudit(
        val worldUpCamera: DoubleArray,
        val vanishingPoint: DoubleArray,
        val finiteVanishingPoint: Boolean,
        val candidateCount: Int,
        val alignedCount: Int,
        val totalLengthPx: Double,
        val alignedLengthPx: Double,
        val alignedLengthFraction: Double,
        val fiveSmallestAngleErrorsDegrees: List<Double>,
        val informative: Boolean,
        val pass: Boolean
    ) {
        fun toJson() = JSONObject()
            .put("finite_vanishing_point", finiteVanishingPoint)
            .put("candidate_line_count", candidateCount)
            .put("aligned_line_count", alignedCount)
            .put("total_line_length_px", totalLengthPx)
            .put("aligned_line_length_px", alignedLengthPx)
            .put("aligned_line_length_fraction", alignedLengthFraction)
            .put("alignment_threshold_degrees", ALIGNMENT_THRESHOLD_DEGREES)
            .put("five_smallest_angle_errors_degrees", JSONArray(fiveSmallestAngleErrorsDegrees))
            .put("informative", informative)
            .put("pass", pass)
    }

    private data class RotationSample(val timestampNs: Long, val quaternionWxyz: DoubleArray)

    private data class CapturedFrame(
        val width: Int,
        val height: Int,
        val rotationDegrees: Int,
        val timestampNs: Long,
        val sensorToBuffer: DoubleArray,
        val argbPixels: IntArray
    ) {
        fun toDisplayBitmap(): Bitmap {
            val rotation = ((rotationDegrees % 360) + 360) % 360
            val displayWidth = if (rotation % 180 == 0) width else height
            val displayHeight = if (rotation % 180 == 0) height else width
            val output = IntArray(displayWidth * displayHeight)
            for (y in 0 until height) for (x in 0 until width) {
                val (displayX, displayY) = when (rotation) {
                    90 -> (height - 1 - y) to x
                    180 -> (width - 1 - x) to (height - 1 - y)
                    270 -> y to (width - 1 - x)
                    else -> x to y
                }
                output[displayY * displayWidth + displayX] = argbPixels[y * width + x]
            }
            return Bitmap.createBitmap(displayWidth, displayHeight, Bitmap.Config.ARGB_8888).also {
                it.setPixels(output, 0, displayWidth, 0, 0, displayWidth, displayHeight)
            }
        }

        companion object {
            fun from(image: ImageProxy): CapturedFrame {
                val plane = image.planes.first()
                val buffer = plane.buffer.duplicate()
                val pixels = IntArray(image.width * image.height)
                for (y in 0 until image.height) for (x in 0 until image.width) {
                    val offset = y * plane.rowStride + x * plane.pixelStride
                    val r = buffer.get(offset).toInt() and 0xff
                    val g = buffer.get(offset + 1).toInt() and 0xff
                    val b = buffer.get(offset + 2).toInt() and 0xff
                    val a = buffer.get(offset + 3).toInt() and 0xff
                    pixels[y * image.width + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
                }
                val matrix = FloatArray(9).also { image.imageInfo.sensorToBufferTransformMatrix.getValues(it) }
                return CapturedFrame(
                    image.width, image.height, image.imageInfo.rotationDegrees, image.imageInfo.timestamp,
                    matrix.map(Float::toDouble).toDoubleArray(), pixels
                )
            }
        }
    }

    private fun slerp(a: DoubleArray, bInput: DoubleArray, t: Double): DoubleArray {
        val b = bInput.copyOf()
        var dot = a.indices.sumOf { a[it] * b[it] }
        if (dot < 0.0) {
            for (i in b.indices) b[i] = -b[i]
            dot = -dot
        }
        if (dot > 0.9995) return normalize(DoubleArray(4) { a[it] + t * (b[it] - a[it]) })
        val theta = acos(dot.coerceIn(-1.0, 1.0))
        val sinTheta = kotlin.math.sin(theta)
        val wa = kotlin.math.sin((1.0 - t) * theta) / sinTheta
        val wb = kotlin.math.sin(t * theta) / sinTheta
        return normalize(DoubleArray(4) { wa * a[it] + wb * b[it] })
    }

    private fun normalize(q: DoubleArray): DoubleArray {
        val norm = kotlin.math.sqrt(q.sumOf { it * it })
        return DoubleArray(q.size) { q[it] / norm }
    }

    private fun quaternionWxyzToMatrix(q: DoubleArray): DoubleArray {
        val w = q[0]; val x = q[1]; val y = q[2]; val z = q[3]
        return doubleArrayOf(
            1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
            2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w,
            2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y
        )
    }

    private class TestLifecycleOwner : LifecycleOwner {
        private val registry = LifecycleRegistry(this)
        override val lifecycle: Lifecycle get() = registry
        fun resume() { registry.currentState = Lifecycle.State.RESUMED }
        fun destroy() { registry.currentState = Lifecycle.State.DESTROYED }
    }

    private fun productionResolutionSelector(): ResolutionSelector = ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
        .setResolutionStrategy(ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
        .build()

    private companion object {
        const val REQUIRED_CAMERA_FRAMES = 20
        const val SELECTED_FRAME_INDEX = 10
        const val SENSOR_LEAD_IN_MS = 300L
        const val PROVIDER_TIMEOUT_SECONDS = 10L
        const val CAPTURE_TIMEOUT_SECONDS = 15L
        const val ALIGNMENT_THRESHOLD_DEGREES = 10.0
        const val MINIMUM_ALIGNED_LENGTH_FRACTION = 0.12
        const val REPORT_KEY = "r836_report"
    }
}
