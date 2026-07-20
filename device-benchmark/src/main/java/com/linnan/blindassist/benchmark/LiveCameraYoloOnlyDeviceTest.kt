package com.linnan.blindassist.benchmark

import android.Manifest
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.linnan.blindassist.vision.RgbaVisionFrame
import com.linnan.blindassist.vision.TfliteYoloDetector
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

/** Matched live-CameraX control for [LiveCameraYoloShadowSidecarDeviceTest]. */
@RunWith(AndroidJUnit4::class)
class LiveCameraYoloOnlyDeviceTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun liveCamera_yoloOnly_reportsAggregateTimingOnly() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val provider = ProcessCameraProvider.getInstance(instrumentation.targetContext).get(PROVIDER_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        val detector = TfliteYoloDetector(instrumentation.context)
        check(detector.isReady) { detector.statusMessage }
        val owner = TestLifecycleOwner()
        val analyserExecutor = Executors.newSingleThreadExecutor()
        val metrics = Metrics()
        val enoughFrames = CountDownLatch(1)
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(analyserExecutor) { image ->
            if (detectAndClose(image, detector, metrics) && metrics.analysed.get() >= MIN_ANALYSED_FRAMES) enoughFrames.countDown()
        }

        try {
            instrumentation.runOnMainSync {
                provider.unbind(analysis)
                owner.resume()
                provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            }
            assertTrue(
                "Camera did not analyse $MIN_ANALYSED_FRAMES frames within ${CAPTURE_TIMEOUT_SECONDS}s",
                enoughFrames.await(CAPTURE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            )
        } finally {
            instrumentation.runOnMainSync {
                provider.unbind(analysis)
                owner.destroy()
            }
            analysis.clearAnalyzer()
            detector.close()
            analyserExecutor.shutdown()
            analyserExecutor.awaitTermination(2, TimeUnit.SECONDS)
        }

        val result = metrics.toJson()
            .put("schema", "blindassist_live_camera_yolo_only_v1")
            .put("device_under_test", android.os.Build.MODEL)
            .put("camera_selector", "back")
            .put("input_format", "CameraX RGBA_8888, KEEP_ONLY_LATEST")
            .put("detector", "existing YOLO test-APK asset")
            .put("raw_frames_persisted", false)
            .put("alerts_invoked", false)
            .put("production_routing_changed", false)
            .put("important_limit", "A real-camera YOLO timing control only; it does not establish alert quality or production readiness.")
        Log.i(TAG, "LIVE_CAMERA_YOLO_ONLY_JSON $result")
        assertTrue("camera frames were not analysed", metrics.analysed.get() >= MIN_ANALYSED_FRAMES)
        assertTrue("detector failure: ${metrics.failureMessage}", metrics.failures.get() == 0)
    }

    private fun detectAndClose(image: ImageProxy, detector: TfliteYoloDetector, metrics: Metrics): Boolean {
        val started = SystemClock.elapsedRealtimeNanos()
        val frame = ImageProxyRgbaFrame(image)
        return try {
            detector.detect(frame)
            metrics.record(SystemClock.elapsedRealtimeNanos() - started)
            true
        } catch (failure: Throwable) {
            metrics.recordFailure(failure)
            false
        } finally {
            frame.close()
        }
    }

    private class Metrics {
        val analysed = AtomicInteger()
        val failures = AtomicInteger()
        private val samplesMicros = ArrayList<Long>()
        private val lock = Any()
        @Volatile var failureMessage: String? = null

        fun record(elapsedNanos: Long) {
            analysed.incrementAndGet()
            synchronized(lock) { samplesMicros += elapsedNanos / 1_000L }
        }

        fun recordFailure(failure: Throwable) {
            failures.incrementAndGet()
            failureMessage = failure.javaClass.simpleName + ": " + (failure.message ?: "no message")
        }

        fun toJson(): JSONObject = synchronized(lock) {
            JSONObject()
                .put("analysed_frame_count", analysed.get())
                .put("analyser_p50_ms", percentileMillis(50.0))
                .put("analyser_p95_ms", percentileMillis(95.0))
                .put("detector_failure_count", failures.get())
        }

        private fun percentileMillis(percent: Double): Any {
            if (samplesMicros.isEmpty()) return JSONObject.NULL
            val ordered = samplesMicros.sorted()
            return ordered[kotlin.math.ceil(ordered.size * percent / 100.0).toInt() - 1].toDouble() / 1_000.0
        }
    }

    private class ImageProxyRgbaFrame(private val image: ImageProxy) : RgbaVisionFrame {
        private val closed = AtomicBoolean(false)
        private val plane = image.planes.first()
        override val width: Int = image.width
        override val height: Int = image.height
        override val rotationDegrees: Int = image.imageInfo.rotationDegrees
        override val buffer: ByteBuffer = plane.buffer.duplicate().also { it.rewind() }
        override val rowStride: Int = plane.rowStride
        override val pixelStride: Int = plane.pixelStride
        override fun close() { if (closed.compareAndSet(false, true)) image.close() }
    }

    private class TestLifecycleOwner : LifecycleOwner {
        private val registry = LifecycleRegistry(this)
        override val lifecycle: Lifecycle get() = registry
        fun resume() { registry.currentState = Lifecycle.State.RESUMED }
        fun destroy() { registry.currentState = Lifecycle.State.DESTROYED }
    }

    private companion object {
        const val MIN_ANALYSED_FRAMES = 200
        const val PROVIDER_TIMEOUT_SECONDS = 10L
        const val CAPTURE_TIMEOUT_SECONDS = 25L
        const val TAG = "LiveCameraYoloOnly"
    }
}
