package com.linnan.blindassist.ustrfbenchmark

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
import android.util.Log
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
import com.linnan.blindassist.ustrf.UstrfPoseBuffer
import com.linnan.blindassist.ustrf.UstrfPoseLookup
import com.linnan.blindassist.ustrf.UstrfPoseLookupFailure
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

/** Isolated package test APK for USTRF CameraX receipt and shadow freshness evidence. */
@RunWith(AndroidJUnit4::class)
class UstrfCameraTimestampShadowBenchmarkTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun cameraTimestampReceipt_rotationBracket_andLatestOnlyShadow_areAuditable() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val sensorManager = context.getSystemService(android.content.Context.SENSOR_SERVICE) as SensorManager
        val rotationSensor = requireNotNull(sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR))
        val rotations = mutableListOf<RotationSample>()
        val rotationsLock = Any()
        val sensorThread = HandlerThread("ustrf-isolated-rotation").apply { start() }
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                synchronized(rotationsLock) { rotations += RotationSample(event.timestamp, event.values.copyOf()) }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
        }
        assertTrue("rotation listener registration failed", sensorManager.registerListener(listener, rotationSensor, SensorManager.SENSOR_DELAY_GAME, Handler(sensorThread.looper)))
        Thread.sleep(SENSOR_LEAD_IN_MS)

        val provider = ProcessCameraProvider.getInstance(context).get(PROVIDER_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        val owner = TestLifecycleOwner()
        val analyserExecutor = Executors.newSingleThreadExecutor()
        val workerExecutor = Executors.newSingleThreadExecutor()
        val rows = mutableListOf<Row>()
        val rowsLock = Any()
        val frameIds = AtomicLong()
        val receiptValidator = UstrfCaptureReceiptValidator(MAX_CAPTURE_AGE_NS)
        val accepted = AtomicInteger()
        val fresh = AtomicInteger()
        val failures = AtomicInteger()
        val resultAgesNs = mutableListOf<Long>()
        val resultAgesLock = Any()
        val captured = CountDownLatch(REQUIRED_FRAMES)
        val sidecar = RgbaLumaSidecar(
            executor = workerExecutor,
            maxResultAgeNanos = MAX_RESULT_AGE_NS,
            process = { owned -> owned.pixels.firstOrNull()?.toInt() ?: 0 },
            onFreshResult = { result ->
                fresh.incrementAndGet()
                synchronized(resultAgesLock) { resultAgesNs += result.ageNanos }
            },
            onFailure = { failures.incrementAndGet() },
            nowNanos = SystemClock::elapsedRealtimeNanos
        )
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(productionResolutionSelector())
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(analyserExecutor) { image ->
            val captureNs = image.imageInfo.timestamp
            val receivedNs = SystemClock.elapsedRealtimeNanos()
            val frame = UstrfFrameStamp(frameIds.getAndIncrement(), captureNs, CAMERA_FRAME)
            val validation = receiptValidator.validate(
                UstrfCaptureReceipt(frame, captureNs, receivedNs, CAMERA_CLOCK_DOMAIN, CALIBRATION_RECEIPT_VERSION),
                receivedNs
            )
            val proxy = ImageProxyRgbaFrame(image)
            try {
                if (sidecar.submit(proxy, captureNs)) accepted.incrementAndGet()
                synchronized(rowsLock) { if (rows.size < REQUIRED_FRAMES) rows += Row(captureNs, receivedNs, validation) }
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
            assertTrue("insufficient CameraX frames", captured.await(CAPTURE_TIMEOUT_SECONDS, TimeUnit.SECONDS))
            Thread.sleep(WORKER_DRAIN_SETTLE_MS)
        } finally {
            instrumentation.runOnMainSync { provider.unbindAll(); owner.destroy() }
            analysis.clearAnalyzer()
            sidecar.close()
            analyserExecutor.shutdown()
            workerExecutor.shutdown()
            analyserExecutor.awaitTermination(2, TimeUnit.SECONDS)
            workerExecutor.awaitTermination(2, TimeUnit.SECONDS)
            sensorManager.unregisterListener(listener)
            sensorThread.quitSafely()
            sensorThread.join(2_000)
        }

        val frozenRows = synchronized(rowsLock) { rows.toList() }
        val sensorTimes = synchronized(rotationsLock) { rotations.sortedBy { it.timestampNs } }
        val brackets = frozenRows.mapNotNull { row ->
            val before = sensorTimes.lastOrNull { it.timestampNs <= row.captureNs }
            val after = sensorTimes.firstOrNull { it.timestampNs >= row.captureNs }
            if (before == null || after == null) null else after.timestampNs - before.timestampNs
        }
        val orientationOnlyAdapter = UstrfOrientationOnlyPoseAdapter()
        val failClosedOrientationReceipts = frozenRows.count { row ->
            val before = sensorTimes.lastOrNull { it.timestampNs <= row.captureNs }
            val after = sensorTimes.firstOrNull { it.timestampNs >= row.captureNs }
            if (before == null || after == null) false else {
                val buffer = UstrfPoseBuffer(MAX_ROTATION_BRACKET_NS)
                buffer.append(orientationOnlyAdapter.receipt(before.timestampNs, before.values, CAMERA_FRAME, before.timestampNs + MAX_CAPTURE_AGE_NS))
                if (after.timestampNs > before.timestampNs) {
                    buffer.append(orientationOnlyAdapter.receipt(after.timestampNs, after.values, CAMERA_FRAME, after.timestampNs + MAX_CAPTURE_AGE_NS))
                }
                buffer.interpolateAt(UstrfFrameStamp(0L, row.captureNs, CAMERA_FRAME)) == UstrfPoseLookup.Unavailable(UstrfPoseLookupFailure.NOT_TRACKING)
            }
        }
        val ages = synchronized(resultAgesLock) { resultAgesNs.toList() }
        val receiptFailures = frozenRows.mapNotNull { (it.validation as? UstrfCaptureReceiptValidation.Unavailable)?.failure }
            .groupingBy { it.name }.eachCount()
        val report = JSONObject()
            .put("schema", "blindassist_ustrf_isolated_camera_timestamp_shadow_v1")
            .put("package", context.packageName)
            .put("capture_timestamp_source", "ImageProxy.imageInfo.timestamp")
            .put("frame_count", frozenRows.size)
            .put("receipt_valid_count", frozenRows.count { it.validation == UstrfCaptureReceiptValidation.Valid })
            .put("receipt_failure_counts", JSONObject(receiptFailures))
            .put("rotation_bracket_coverage", brackets.size.toDouble() / frozenRows.size)
            .put("rotation_bracket_max_ms", brackets.maxOrNull()!! / 1_000_000.0)
            .put("orientation_only_pose_fail_closed_count", failClosedOrientationReceipts)
            .put("shadow_accepted_count", accepted.get())
            .put("shadow_fresh_count", fresh.get())
            .put("shadow_result_age_max_ms", ages.maxOrNull()?.div(1_000_000.0) ?: JSONObject.NULL)
            .put("shadow_failure_count", failures.get())
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("orientation_only_pose_tracking_authorized", false)
                .put("pose_or_vio_validated", false)
                .put("production_authorized", false))
        Log.i(TAG, "USTRF_ISOLATED_TIMESTAMP_SHADOW_JSON $report")
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("incorrect frame count", frozenRows.size == REQUIRED_FRAMES)
        assertTrue("capture timestamps not monotonic", frozenRows.zipWithNext().all { (a, b) -> b.captureNs > a.captureNs })
        assertTrue("invalid capture receipt", frozenRows.all { it.validation == UstrfCaptureReceiptValidation.Valid })
        assertTrue("missing rotation brackets", brackets.size == frozenRows.size)
        assertTrue("rotation bracket too wide", brackets.maxOrNull()!! <= MAX_ROTATION_BRACKET_NS)
        assertTrue("orientation-only receipts must fail closed", failClosedOrientationReceipts == frozenRows.size)
        assertTrue("shadow worker failed", failures.get() == 0)
        assertTrue("no shadow input accepted", accepted.get() > 0)
        assertTrue("no fresh shadow output", fresh.get() > 0)
    }

    private data class Row(val captureNs: Long, val receivedNs: Long, val validation: UstrfCaptureReceiptValidation)
    private data class RotationSample(val timestampNs: Long, val values: FloatArray)

    private class ImageProxyRgbaFrame(private val image: ImageProxy) : RgbaVisionFrame {
        private val closed = AtomicBoolean(false)
        private val plane = image.planes.first()
        override val width = image.width
        override val height = image.height
        override val rotationDegrees = image.imageInfo.rotationDegrees
        override val buffer: ByteBuffer = plane.buffer.duplicate().also { it.rewind() }
        override val rowStride = plane.rowStride
        override val pixelStride = plane.pixelStride
        override fun close() { if (closed.compareAndSet(false, true)) image.close() }
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
        const val REPORT_KEY = "ustrf_isolated_timestamp_shadow_report"
        const val TAG = "UstrfShadowBenchmark"
    }
}
