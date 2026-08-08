package com.linnan.blindassist.benchmark

import android.Manifest
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
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
import org.json.JSONObject
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
class CameraRotationTimestampPairingAuditTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun auditCaptureTimestampBracketingByRotationVectorSamples() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val sensorManager = context.getSystemService(android.content.Context.SENSOR_SERVICE) as SensorManager
        val rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        assertNotNull("rotation-vector sensor unavailable", rotationSensor)
        val sensorSamples = mutableListOf<Long>()
        val sensorThread = HandlerThread("r835-rotation-pairing").apply { start() }
        val sensorListener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                synchronized(sensorSamples) { sensorSamples += event.timestamp }
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
        val cameraTimestamps = mutableListOf<Long>()
        val latch = CountDownLatch(REQUIRED_CAMERA_FRAMES)
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(productionResolutionSelector())
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(executor) { image ->
            try {
                synchronized(cameraTimestamps) {
                    if (cameraTimestamps.size < REQUIRED_CAMERA_FRAMES) cameraTimestamps += image.imageInfo.timestamp
                }
                latch.countDown()
            } finally {
                image.close()
            }
        }

        try {
            instrumentation.runOnMainSync {
                provider.unbindAll()
                owner.resume()
                provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
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

        val cameras = synchronized(cameraTimestamps) { cameraTimestamps.sorted() }
        val sensors = synchronized(sensorSamples) { sensorSamples.sorted() }
        val pairs = cameras.map { cameraTs ->
            val previous = sensors.lastOrNull { it <= cameraTs }
            val next = sensors.firstOrNull { it >= cameraTs }
            Pairing(cameraTs, previous, next)
        }
        val bracketed = pairs.filter { it.previousNs != null && it.nextNs != null }
        val nearestDelta = bracketed.map { minOf(abs(it.cameraNs - it.previousNs!!), abs(it.nextNs!! - it.cameraNs)) }
        val bracketSpan = bracketed.map { it.nextNs!! - it.previousNs!! }
        val interpolationFactors = bracketed.map {
            val span = it.nextNs!! - it.previousNs!!
            if (span == 0L) 0.0 else (it.cameraNs - it.previousNs!!).toDouble() / span
        }
        val report = JSONObject()
            .put("schema", "blindassist_camera_rotation_timestamp_pairing_audit_v1")
            .put("camera_frame_count", cameras.size)
            .put("rotation_sample_count", sensors.size)
            .put("camera_frames_bracketed", bracketed.size)
            .put("bracketing_coverage", bracketed.size.toDouble() / cameras.size)
            .put("nearest_rotation_sample_delta_ms", JSONObject()
                .put("median", nearestDelta.medianNanos() / 1_000_000.0)
                .put("maximum", nearestDelta.maxOrNull()!! / 1_000_000.0))
            .put("rotation_bracket_span_ms", JSONObject()
                .put("median", bracketSpan.medianNanos() / 1_000_000.0)
                .put("maximum", bracketSpan.maxOrNull()!! / 1_000_000.0))
            .put("interpolation_factor_range", JSONObject()
                .put("minimum", interpolationFactors.minOrNull())
                .put("maximum", interpolationFactors.maxOrNull()))
            .put("pairing_rule", "SLERP between nearest rotation-vector samples bracketing ImageInfo.timestamp")
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("timestamp_pairing_capability_validated", bracketed.size == cameras.size)
                .put("interpolated_pose_accuracy_validated", false)
                .put("real_reprojection_validated", false)
                .put("app_runtime_authorized", false)
                .put("production_authorized", false))
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("camera timestamps not monotonic", cameras.zipWithNext().all { (a, b) -> b > a })
        assertTrue("rotation timestamps not monotonic", sensors.zipWithNext().all { (a, b) -> b > a })
        assertTrue("not all camera frames were bracketed", bracketed.size == cameras.size)
        assertTrue("rotation bracket too wide: ${bracketSpan.maxOrNull()}", bracketSpan.maxOrNull()!! <= MAX_BRACKET_NS)
    }

    private data class Pairing(val cameraNs: Long, val previousNs: Long?, val nextNs: Long?)

    private fun List<Long>.medianNanos(): Double {
        val sorted = sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) (sorted[middle - 1] + sorted[middle]) / 2.0 else sorted[middle].toDouble()
    }

    private class TestLifecycleOwner : LifecycleOwner {
        private val registry = LifecycleRegistry(this)
        override val lifecycle: Lifecycle get() = registry
        fun resume() { registry.currentState = Lifecycle.State.RESUMED }
        fun destroy() { registry.currentState = Lifecycle.State.DESTROYED }
    }

    private fun productionResolutionSelector(): ResolutionSelector = ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
        .setResolutionStrategy(
            ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
        )
        .build()

    private companion object {
        const val REQUIRED_CAMERA_FRAMES = 30
        const val SENSOR_LEAD_IN_MS = 300L
        const val MAX_BRACKET_NS = 50_000_000L
        const val PROVIDER_TIMEOUT_SECONDS = 10L
        const val CAPTURE_TIMEOUT_SECONDS = 15L
        const val REPORT_KEY = "r835_report"
    }
}
