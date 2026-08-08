package com.linnan.blindassist.benchmark.ustrf

import android.Manifest
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Size
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
import com.linnan.blindassist.ustrf.UstrfCaptureReceipt
import com.linnan.blindassist.ustrf.UstrfCaptureReceiptValidation
import com.linnan.blindassist.ustrf.UstrfCaptureReceiptValidator
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import com.linnan.blindassist.vision.RgbaLumaSidecar
import com.linnan.blindassist.vision.RgbaVisionFrame
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Test-APK-only USTRF timestamp/shadow spike. It does not run YOLO, pose/VIO, USTRF planning,
 * feedback, or persistence. `imageInfo.timestamp` is retained as the only source capture time.
 */
@RunWith(AndroidJUnit4::class)
class UstrfCameraTimestampShadowDeviceTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun cameraTimestamp_receiptRotationBracketAndShadowFreshness_areAuditable() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val sensorManager = context.getSystemService(android.content.Context.SENSOR_SERVICE) as SensorManager
        val rotationSensor = requireNotNull(sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR))
        val rotations = mutableListOf<Long>()
        val rotationLock = Any()
        val sensorThread = HandlerThread("ustrf-camera-timestamp-rotation").apply { start() }
        val sensorListener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                synchronized(rotationLock) { rotations += event.timestamp }
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
        val analyserExecutor = Executors.newSingleThreadExecutor()
        val workerExecutor = Executors.newSingleThreadExecutor()
        val frameIds = AtomicLong()
        val rows = mutableListOf<Row>()
        val rowsLock = Any()
        val captured = CountDownLatch(REQUIRED_FRAMES)
        val sidecarAccepted = AtomicInteger()
        val sidecarFresh = AtomicInteger()
        val sidecarFailures = AtomicInteger()
        val sidecarResultAgesNs = mutableListOf<Long>()
        val sidecarAgesLock = Any()
        val receiptValidator = UstrfCaptureReceiptValidator(MAX_CAPTURE_AGE_NS)
        val sidecar = RgbaLumaSidecar(
            executor = workerExecutor,
            maxResultAgeNanos = MAX_RESULT_AGE_NS,
            process = { owned -> owned.pixels.firstOrNull()?.toInt() ?: 0 },
            onFreshResult = { result ->
                sidecarFresh.incrementAndGet()
                synchronized(sidecarAgesLock) { sidecarResultAgesNs += result.ageNanos }
            },
            onFailure = { sidecarFailures.incrementAndGet() },
            nowNanos = SystemClock::elapsedRealtimeNanos
        )
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(productionResolutionSelector())
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(analyserExecutor) { image ->
            val capturedAtNs = image.imageInfo.timestamp
            val receivedAtNs = SystemClock.elapsedRealtimeNanos()
            val frame = UstrfFrameStamp(frameIds.getAndIncrement(), capturedAtNs, CAMERA_FRAME)
            val validation = receiptValidator.validate(
                UstrfCaptureReceipt(
                    frame = frame,
                    hardwareTimestampNs = capturedAtNs,
                    receivedAtNs = receivedAtNs,
                    cameraClockDomain = CAMERA_CLOCK_DOMAIN,
                    calibrationVersion = CALIBRATION_RECEIPT_VERSION
                ),
                decisionAtNs = receivedAtNs
            )
            val proxy = ImageProxyRgbaFrame(image)
            try {
                if (sidecar.submit(proxy, capturedAtNs)) sidecarAccepted.incrementAndGet()
                synchronized(rowsLock) {
                    if (rows.size < REQUIRED_FRAMES) rows += Row(capturedAtNs, receivedAtNs, validation)
                }
                captured.countDown()
            } finally {
                proxy.close()
            }
        }

        try {
            instrumentation.runOnMainSync {
                provider.unbindAll()
                owner.resume()
                provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            }
            assertTrue("insufficient camera frames", captured.await(CAPTURE_TIMEOUT_SECONDS, TimeUnit.SECONDS))
            Thread.sleep(WORKER_DRAIN_SETTLE_MS)
        } finally {
            instrumentation.runOnMainSync {
                provider.unbindAll()
                owner.destroy()
            }
            analysis.clearAnalyzer()
            sidecar.close()
            analyserExecutor.shutdown()
            workerExecutor.shutdown()
            analyserExecutor.awaitTermination(2, TimeUnit.SECONDS)
            workerExecutor.awaitTermination(2, TimeUnit.SECONDS)
            sensorManager.unregisterListener(sensorListener)
            sensorThread.quitSafely()
            sensorThread.join(2_000)
        }

        val frozenRows = synchronized(rowsLock) { rows.toList() }
        val sensorTimes = synchronized(rotationLock) { rotations.sorted() }
        val bracketSpans = frozenRows.mapNotNull { row ->
            val before = sensorTimes.lastOrNull { it <= row.capturedAtNs }
            val after = sensorTimes.firstOrNull { it >= row.capturedAtNs }
            if (before == null || after == null) null else after - before
        }
        val callbackLagNs = frozenRows.map { it.receivedAtNs - it.capturedAtNs }
        val resultAges = synchronized(sidecarAgesLock) { sidecarResultAgesNs.toList() }
        val receiptFailures = frozenRows.mapNotNull { (it.validation as? UstrfCaptureReceiptValidation.Unavailable)?.failure }
            .groupingBy { it.name }
            .eachCount()
        val report = JSONObject()
            .put("schema", "blindassist_ustrf_camera_timestamp_shadow_v1")
            .put("frame_count", frozenRows.size)
            .put("capture_timestamp_source", "ImageProxy.imageInfo.timestamp")
            .put("camera_clock_domain", CAMERA_CLOCK_DOMAIN)
            .put("capture_receipt_valid_count", frozenRows.count { it.validation == UstrfCaptureReceiptValidation.Valid })
            .put("capture_receipt_failure_counts", JSONObject(receiptFailures))
            .put("callback_minus_capture_max_ms", callbackLagNs.maxOrNull()!! / 1_000_000.0)
            .put("rotation_bracket_coverage", bracketSpans.size.toDouble() / frozenRows.size)
            .put("rotation_bracket_max_ms", bracketSpans.maxOrNull()!! / 1_000_000.0)
            .put("shadow_accepted_count", sidecarAccepted.get())
            .put("shadow_fresh_result_count", sidecarFresh.get())
            .put("shadow_result_age_max_ms", resultAges.maxOrNull()?.div(1_000_000.0) ?: JSONObject.NULL)
            .put("shadow_failure_count", sidecarFailures.get())
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("pose_or_vio_validated", false)
                .put("camera_to_body_extrinsics_validated", false)
                .put("app_runtime_authorized", false)
                .put("production_authorized", false))
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("no rows captured", frozenRows.size == REQUIRED_FRAMES)
        assertTrue("capture timestamps must be strictly monotonic", frozenRows.zipWithNext().all { (a, b) -> b.capturedAtNs > a.capturedAtNs })
        assertTrue("capture receipt was rejected: ${frozenRows.firstOrNull { it.validation != UstrfCaptureReceiptValidation.Valid }}", frozenRows.all { it.validation == UstrfCaptureReceiptValidation.Valid })
        assertTrue("rotation brackets missing", bracketSpans.size == frozenRows.size)
        assertTrue("rotation bracket too wide: ${bracketSpans.maxOrNull()}", bracketSpans.maxOrNull()!! <= MAX_ROTATION_BRACKET_NS)
        assertTrue("shadow worker failed", sidecarFailures.get() == 0)
        assertTrue("no shadow frames accepted", sidecarAccepted.get() > 0)
        assertTrue("no fresh shadow result delivered", sidecarFresh.get() > 0)
    }

    private data class Row(
        val capturedAtNs: Long,
        val receivedAtNs: Long,
        val validation: UstrfCaptureReceiptValidation
    )

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
        const val REQUIRED_FRAMES = 30
        const val SENSOR_LEAD_IN_MS = 300L
        const val PROVIDER_TIMEOUT_SECONDS = 10L
        const val CAPTURE_TIMEOUT_SECONDS = 15L
        const val WORKER_DRAIN_SETTLE_MS = 500L
        const val MAX_CAPTURE_AGE_NS = 250_000_000L
        const val MAX_RESULT_AGE_NS = 150_000_000L
        const val MAX_ROTATION_BRACKET_NS = 50_000_000L
        const val CAMERA_FRAME = "camera-v1"
        const val CAMERA_CLOCK_DOMAIN = "android-camera-timestamp-v1"
        const val CALIBRATION_RECEIPT_VERSION = "benchmark-unknown-intrinsics-v1"
        const val REPORT_KEY = "ustrf_timestamp_shadow_report"
    }
}
