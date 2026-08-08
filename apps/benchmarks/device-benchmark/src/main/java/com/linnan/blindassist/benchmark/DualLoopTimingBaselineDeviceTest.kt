package com.linnan.blindassist.benchmark

import android.Manifest
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.os.SystemClock
import android.util.Range
import android.util.Size
import android.view.Surface
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
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
import com.linnan.blindassist.vision.RgbaLumaResampler
import com.linnan.blindassist.vision.RgbaVisionFrame
import com.linnan.blindassist.vision.RuntimeObjectDetectorFactory
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.File
import java.nio.ByteBuffer
import java.security.MessageDigest
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * F-1B0 timing-only baseline. It records when complete semantic and geometry
 * results become legally readable; it never evaluates risk, alert, or A/B effect.
 */
@RunWith(AndroidJUnit4::class)
class DualLoopTimingBaselineDeviceTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule =
        GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun dualLoopTimingBaseline_recordsCausalAvailability() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val cameraTimestampSource = backCameraTimestampSource(
            instrumentation.targetContext.getSystemService(CameraManager::class.java)
        )
        assertEquals(
            "Camera sensor timestamps are not proven to share the Android realtime domain",
            CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME,
            cameraTimestampSource
        )

        val semantic = runSemanticBaseline(cameraTimestampSource)
        val geometry = runGeometryBaseline()
        assertTrue(semantic.length() >= TARGET_RESULTS)
        assertTrue(geometry.length() >= TARGET_RESULTS)

        val receipt = JSONObject()
            .put("schema", "blindassist_dual_loop_f1b0_timing_baseline_v1")
            .put("device_model", android.os.Build.MODEL)
            .put("soc_model", android.os.Build.SOC_MODEL)
            .put("android_version", android.os.Build.VERSION.RELEASE)
            .put("android_api", android.os.Build.VERSION.SDK_INT)
            .put("clock_domain", CLOCK_DOMAIN)
            .put("camera_timestamp_source", "REALTIME")
            .put("camera_clock_mapping_status", "VERIFIED_ANDROID_REALTIME")
            .put("semantic_queue_policy", "CameraX_KEEP_ONLY_LATEST_then_inline_detector")
            .put("geometry_queue_policy", "isolated_inline_sequential_no_backlog")
            .put("effect_outputs_accessed", false)
            .put("alerts_invoked", false)
            .put("risk_outputs_serialized", false)
            .put("production_routing_changed", false)
            .put("semantic_results", semantic)
            .put("geometry_results", geometry)
            .put(
                "important_limit",
                "Timing-only development baseline. Geometry is isolated from YOLO and uses hash-bound RGB assets; no event effect, risk, alert, or production behavior is evaluated."
            )
        val output = File(
            checkNotNull(instrumentation.targetContext.getExternalFilesDir(null)),
            RESULT_RELATIVE_PATH
        )
        output.parentFile?.mkdirs()
        output.writeText(receipt.toString(2), Charsets.UTF_8)
    }

    private fun runSemanticBaseline(cameraTimestampSource: Int?): JSONArray {
        check(cameraTimestampSource == CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME)
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val provider = ProcessCameraProvider.getInstance(instrumentation.targetContext)
            .get(PROVIDER_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        val detector = RuntimeObjectDetectorFactory.create(instrumentation.targetContext)
        check(detector.isReady) { detector.statusMessage }
        val typedDetector = detector as? TfliteYoloDetector
            ?: error("Production route did not create TfliteYoloDetector")
        val backend = typedDetector.executionBackend.wireName
        assertEquals("qualcomm_qnn_htp", backend)
        val routeReason = detector.statusMessage
        val owner = TestLifecycleOwner()
        val executor = Executors.newSingleThreadExecutor()
        val completed = CountDownLatch(1)
        val rows = JSONArray()
        val previewTexture = SurfaceTexture(false)
        val previewSurface = Surface(previewTexture)
        val preview = Preview.Builder()
            .setTargetFrameRate(Range(TARGET_FPS, TARGET_FPS))
            .build()
        instrumentation.runOnMainSync {
            preview.setSurfaceProvider { request ->
                previewTexture.setDefaultBufferSize(
                    request.resolution.width,
                    request.resolution.height
                )
                request.provideSurface(previewSurface, executor) {}
            }
        }
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(
                        AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
                    )
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            Size(CAMERA_WIDTH, CAMERA_HEIGHT),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                        )
                    )
                    .build()
            )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(executor) { image ->
            if (rows.length() >= TARGET_RESULTS) {
                image.close()
                completed.countDown()
            } else {
                val receivedAt = SystemClock.elapsedRealtimeNanos()
                val capturedAt = image.imageInfo.timestamp
                val queuedAt = receivedAt
                val startedAt = SystemClock.elapsedRealtimeNanos()
                val frame = ImageProxyRgbaFrame(image)
                try {
                    detector.detect(frame)
                    val completedAt = SystemClock.elapsedRealtimeNanos()
                    val publishedAt = SystemClock.elapsedRealtimeNanos()
                    val availableAt = publishedAt
                    val consumedAt = SystemClock.elapsedRealtimeNanos()
                    assertCausal(
                        capturedAt,
                        receivedAt,
                        queuedAt,
                        startedAt,
                        completedAt,
                        publishedAt,
                        availableAt,
                        consumedAt
                    )
                    rows.put(
                        JSONObject()
                            .put("capturedAt", capturedAt)
                            .put("receivedAt", receivedAt)
                            .put("queuedAt", queuedAt)
                            .put("startedAt", startedAt)
                            .put("completedAt", completedAt)
                            .put("publishedAt", publishedAt)
                            .put("availableAt", availableAt)
                            .put("consumedAt", consumedAt)
                            .put("clockDomain", CLOCK_DOMAIN)
                            .put("dropReason", "NONE")
                            .put("detectorBackend", backend)
                            .put("backendRouteReason", routeReason)
                    )
                    if (rows.length() >= TARGET_RESULTS) completed.countDown()
                } finally {
                    frame.close()
                }
            }
        }
        try {
            instrumentation.runOnMainSync {
                owner.resume()
                provider.bindToLifecycle(
                    owner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
            }
            assertTrue(
                "CameraX/QNN timing trace did not complete",
                completed.await(SEMANTIC_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            )
        } finally {
            instrumentation.runOnMainSync {
                provider.unbind(preview, analysis)
                preview.setSurfaceProvider(null)
                owner.destroy()
            }
            analysis.clearAnalyzer()
            previewSurface.release()
            previewTexture.release()
            detector.close()
            executor.shutdown()
            executor.awaitTermination(2, TimeUnit.SECONDS)
        }
        return rows
    }

    private fun runGeometryBaseline(): JSONArray {
        assertTrue("OpenCV local runtime failed to load", OpenCVLoader.initLocal())
        val context = InstrumentationRegistry.getInstrumentation().context
        val receiptBytes = context.assets.open("$ASSET_ROOT/machine_redaction_receipt.json")
            .use { it.readBytes() }
        assertEquals(RECEIPT_SHA256, sha256(receiptBytes))
        val sourceReceipt = JSONObject(receiptBytes.toString(Charsets.UTF_8))
        val frames = sourceReceipt.getJSONArray("frames")
        val rgba = List(minOf(frames.length(), TARGET_RESULTS + 1)) { index ->
            val row = frames.getJSONObject(index)
            val bytes = context.assets.open(
                "$ASSET_ROOT/images/${row.getString("file_name")}"
            ).use { it.readBytes() }
            assertEquals(row.getString("sha256"), sha256(bytes))
            val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                ?: error("Cannot decode bound geometry frame")
            try {
                rgba640x480(bitmap)
            } finally {
                bitmap.recycle()
            }
        }
        val resampler = RgbaLumaResampler(GEOMETRY_SIZE)
        val probe = SparseLkGeometryProbe(maxCorners = 300)
        val rows = JSONArray()
        var previous: Mat? = null
        var previousObservationAt = 0L
        try {
            rgba.forEach { bytes ->
                val currentObservationAt = SystemClock.elapsedRealtimeNanos()
                val current = lumaMat(resampler.sample(RgbaByteFrame(bytes)))
                val prior = previous
                if (prior == null) {
                    previous = current
                    previousObservationAt = currentObservationAt
                    return@forEach
                }
                val queuedAt = currentObservationAt
                val startedAt = SystemClock.elapsedRealtimeNanos()
                val vector = try {
                    probe.measure(prior, current)
                } finally {
                    prior.release()
                }
                val completedAt = SystemClock.elapsedRealtimeNanos()
                previous = current
                val publishedAt = SystemClock.elapsedRealtimeNanos()
                val availableAt = publishedAt
                val consumedAt = SystemClock.elapsedRealtimeNanos()
                assertCausal(
                    previousObservationAt,
                    currentObservationAt,
                    queuedAt,
                    startedAt,
                    completedAt,
                    publishedAt,
                    availableAt,
                    consumedAt
                )
                rows.put(
                    JSONObject()
                        .put("previousObservationAt", previousObservationAt)
                        .put("currentObservationAt", currentObservationAt)
                        .put("geometryQueuedAt", queuedAt)
                        .put("geometryStartedAt", startedAt)
                        .put("geometryCompletedAt", completedAt)
                        .put("geometryPublishedAt", publishedAt)
                        .put("geometryAvailableAt", availableAt)
                        .put("geometryConsumedAt", consumedAt)
                        .put("clockDomain", CLOCK_DOMAIN)
                        .put("dropReason", "NONE")
                        .put(
                            "abstainReason",
                            if (vector.success) "NONE" else "SPARSE_LK_QUALITY_ABSTAIN"
                        )
                )
                previousObservationAt = currentObservationAt
            }
        } finally {
            previous?.release()
            probe.close()
        }
        return rows
    }

    private fun backCameraTimestampSource(manager: CameraManager): Int? {
        val id = manager.cameraIdList.first { cameraId ->
            manager.getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.LENS_FACING) ==
                CameraCharacteristics.LENS_FACING_BACK
        }
        return manager.getCameraCharacteristics(id)
            .get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
    }

    private fun assertCausal(vararg values: Long) {
        assertTrue(
            "Non-causal timing trace: ${values.joinToString()}",
            values.asList().zipWithNext().all { (earlier, later) -> earlier <= later }
        )
    }

    private fun rgba640x480(bitmap: Bitmap): ByteArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, CAMERA_WIDTH, CAMERA_HEIGHT, true)
        return try {
            val pixels = IntArray(CAMERA_WIDTH * CAMERA_HEIGHT)
            scaled.getPixels(pixels, 0, CAMERA_WIDTH, 0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)
            ByteArray(CAMERA_WIDTH * CAMERA_HEIGHT * 4).also { bytes ->
                pixels.forEachIndexed { index, pixel ->
                    val offset = index * 4
                    bytes[offset] = ((pixel shr 16) and 0xFF).toByte()
                    bytes[offset + 1] = ((pixel shr 8) and 0xFF).toByte()
                    bytes[offset + 2] = (pixel and 0xFF).toByte()
                    bytes[offset + 3] = ((pixel ushr 24) and 0xFF).toByte()
                }
            }
        } finally {
            if (scaled !== bitmap) scaled.recycle()
        }
    }

    private fun lumaMat(bytes: ByteArray): Mat =
        Mat(GEOMETRY_SIZE, GEOMETRY_SIZE, CvType.CV_8UC1).also {
            it.put(0, 0, bytes)
        }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { "%02x".format(it) }

    private class ImageProxyRgbaFrame(private val image: ImageProxy) : RgbaVisionFrame {
        private val closed = AtomicBoolean(false)
        private val plane = image.planes.first()
        override val width: Int = image.width
        override val height: Int = image.height
        override val rotationDegrees: Int = image.imageInfo.rotationDegrees
        override val buffer: ByteBuffer = plane.buffer.duplicate().also { it.rewind() }
        override val rowStride: Int = plane.rowStride
        override val pixelStride: Int = plane.pixelStride
        override fun close() {
            if (closed.compareAndSet(false, true)) image.close()
        }
    }

    private class RgbaByteFrame(private val bytes: ByteArray) : RgbaVisionFrame {
        override val width = CAMERA_WIDTH
        override val height = CAMERA_HEIGHT
        override val rotationDegrees = 0
        override val rowStride = CAMERA_WIDTH * 4
        override val pixelStride = 4
        override val buffer: ByteBuffer = ByteBuffer.wrap(bytes)
        override fun close() = Unit
    }

    private class TestLifecycleOwner : LifecycleOwner {
        private val registry = LifecycleRegistry(this)
        override val lifecycle: Lifecycle get() = registry
        fun resume() {
            registry.currentState = Lifecycle.State.RESUMED
        }
        fun destroy() {
            registry.currentState = Lifecycle.State.DESTROYED
        }
    }

    private companion object {
        const val ASSET_ROOT = "sparse_lk_sanpo"
        const val RECEIPT_SHA256 =
            "d99b794c235e0f24a0656fe8da3f1719b283a795571e6de323e03997a9501129"
        const val CAMERA_WIDTH = 640
        const val CAMERA_HEIGHT = 480
        const val GEOMETRY_SIZE = 320
        const val TARGET_RESULTS = 24
        const val TARGET_FPS = 24
        const val PROVIDER_TIMEOUT_SECONDS = 10L
        const val SEMANTIC_TIMEOUT_SECONDS = 30L
        const val CLOCK_DOMAIN = "ANDROID_ELAPSED_REALTIME_NANOS"
        const val RESULT_RELATIVE_PATH =
            "benchmark-results/dual-loop-f1b0-timing-baseline.json"
    }
}
