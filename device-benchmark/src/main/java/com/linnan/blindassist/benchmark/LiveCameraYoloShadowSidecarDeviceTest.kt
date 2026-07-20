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
import com.linnan.blindassist.vision.RgbaLumaSidecar
import com.linnan.blindassist.vision.RgbaLumaResampler
import com.linnan.blindassist.vision.RgbaVisionFrame
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.nio.ByteBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Test-APK-only end-to-end timing check: a live CameraX frame is copied into a
 * shadow sidecar before the existing YOLO detector consumes it. It has no
 * RiskEventTracker or feedback path and persists no camera frames.
 */
@RunWith(AndroidJUnit4::class)
class LiveCameraYoloShadowSidecarDeviceTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun liveCamera_yoloPlusGreenChannelShadowSidecar_reportsAggregateTimingOnly() {
        assertTrue("OpenCV local runtime failed to load", OpenCVLoader.initLocal())
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val provider = ProcessCameraProvider.getInstance(instrumentation.targetContext).get(PROVIDER_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        val detector = TfliteYoloDetector(instrumentation.context)
        check(detector.isReady) { detector.statusMessage }
        val owner = TestLifecycleOwner()
        val analyserExecutor = Executors.newSingleThreadExecutor()
        val workerExecutor = Executors.newSingleThreadExecutor()
        val metrics = Metrics()
        val enoughFrames = CountDownLatch(1)
        val worker = GeometryWorker()
        val sidecar = RgbaLumaSidecar(
            executor = workerExecutor,
            maxResultAgeNanos = MAX_RESULT_AGE_NANOS,
            mode = RgbaLumaResampler.Mode.GREEN_CHANNEL,
            process = worker::process,
            onFreshResult = { result -> metrics.recordDelivery(result.ageNanos, result.value) },
            onFailure = metrics::recordFailure,
            nowNanos = SystemClock::elapsedRealtimeNanos
        )
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(analyserExecutor) { image ->
            val accepted = submitThenDetect(image, sidecar, detector, metrics)
            if (accepted && metrics.analysed.get() >= MIN_ANALYSED_FRAMES) enoughFrames.countDown()
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
            Thread.sleep(WORKER_DRAIN_SETTLE_MS)
        } finally {
            instrumentation.runOnMainSync {
                provider.unbind(analysis)
                owner.destroy()
            }
            analysis.clearAnalyzer()
            sidecar.close()
            detector.close()
            analyserExecutor.shutdown()
            workerExecutor.shutdown()
            analyserExecutor.awaitTermination(2, TimeUnit.SECONDS)
            workerExecutor.awaitTermination(2, TimeUnit.SECONDS)
            worker.close()
        }

        val result = metrics.toJson()
            .put("schema", "blindassist_live_camera_yolo_shadow_sidecar_v1")
            .put("device_under_test", android.os.Build.MODEL)
            .put("camera_selector", "back")
            .put("input_format", "CameraX RGBA_8888, KEEP_ONLY_LATEST")
            .put("detector", "existing YOLO test-APK asset")
            .put("sidecar", "320x320 green-channel luma plus test-only Sparse-LK five-channel probe")
            .put("luma_mode", "GREEN_CHANNEL")
            .put("max_result_age_ms", MAX_RESULT_AGE_NANOS / 1_000_000.0)
            .put("raw_frames_persisted", false)
            .put("alerts_invoked", false)
            .put("production_routing_changed", false)
            .put("important_limit", "A real-camera combined timing probe only. It does not establish event recall, false-alert rate, clearance, or production readiness.")
        Log.i(TAG, "LIVE_CAMERA_YOLO_SHADOW_SIDECAR_JSON $result")
        assertTrue("camera frames were not analysed", metrics.analysed.get() >= MIN_ANALYSED_FRAMES)
        assertTrue("sidecar reported a worker failure: ${metrics.failureMessage}", metrics.failures.get() == 0)
    }

    private fun submitThenDetect(
        image: ImageProxy,
        sidecar: RgbaLumaSidecar<SparseLkGeometryVector?>,
        detector: TfliteYoloDetector,
        metrics: Metrics
    ): Boolean {
        val frameStart = SystemClock.elapsedRealtimeNanos()
        val frame = ImageProxyRgbaFrame(image)
        return try {
            val accepted = sidecar.submit(frame, frameStart)
            val detectorStart = SystemClock.elapsedRealtimeNanos()
            detector.detect(frame)
            metrics.recordAnalysis(
                totalNanos = SystemClock.elapsedRealtimeNanos() - frameStart,
                detectorNanos = SystemClock.elapsedRealtimeNanos() - detectorStart,
                accepted = accepted
            )
            accepted
        } catch (failure: Throwable) {
            metrics.recordFailure(failure)
            false
        } finally {
            frame.close()
        }
    }

    private class GeometryWorker : AutoCloseable {
        private val probe = SparseLkGeometryProbe(maxCorners = 300)
        private var previous: Mat? = null

        fun process(input: RgbaLumaSidecar.OwnedLumaFrame): SparseLkGeometryVector? {
            val current = Mat(SIZE, SIZE, CvType.CV_8UC1).also { it.put(0, 0, input.pixels) }
            val prior = previous
            if (prior == null) {
                previous = current
                return null
            }
            return try { probe.measure(prior, current) } finally { prior.release(); previous = current }
        }

        override fun close() {
            previous?.release()
            previous = null
            probe.close()
        }
    }

    private class Metrics {
        val analysed = AtomicInteger()
        private val rejected = AtomicInteger()
        val failures = AtomicInteger()
        private val delivered = AtomicInteger()
        private val vectors = AtomicInteger()
        private val successfulVectors = AtomicInteger()
        private val totalMicros = ArrayList<Long>()
        private val detectorMicros = ArrayList<Long>()
        private val resultMicros = ArrayList<Long>()
        private val lock = Any()
        @Volatile var failureMessage: String? = null

        fun recordAnalysis(totalNanos: Long, detectorNanos: Long, accepted: Boolean) {
            if (accepted) analysed.incrementAndGet() else rejected.incrementAndGet()
            synchronized(lock) {
                totalMicros += totalNanos / 1_000L
                detectorMicros += detectorNanos / 1_000L
            }
        }

        fun recordDelivery(ageNanos: Long, vector: SparseLkGeometryVector?) {
            delivered.incrementAndGet()
            vector?.let {
                vectors.incrementAndGet()
                if (it.success) successfulVectors.incrementAndGet()
            }
            synchronized(lock) { resultMicros += ageNanos / 1_000L }
        }

        fun recordFailure(failure: Throwable) {
            failures.incrementAndGet()
            failureMessage = failure.javaClass.simpleName + ": " + (failure.message ?: "no message")
        }

        fun toJson(): JSONObject = synchronized(lock) {
            JSONObject()
                .put("analysed_frame_count", analysed.get())
                .put("rejected_frame_count", rejected.get())
                .put("fresh_result_count", delivered.get())
                .put("geometry_vector_count", vectors.get())
                .put("geometry_success_rate", if (vectors.get() == 0) JSONObject.NULL else successfulVectors.get().toDouble() / vectors.get())
                .put("combined_analyser_p50_ms", percentileMillis(totalMicros, 50.0))
                .put("combined_analyser_p95_ms", percentileMillis(totalMicros, 95.0))
                .put("detector_only_p50_ms", percentileMillis(detectorMicros, 50.0))
                .put("detector_only_p95_ms", percentileMillis(detectorMicros, 95.0))
                .put("capture_receive_to_result_p50_ms", percentileMillis(resultMicros, 50.0))
                .put("capture_receive_to_result_p95_ms", percentileMillis(resultMicros, 95.0))
                .put("worker_failure_count", failures.get())
        }

        private fun percentileMillis(samples: List<Long>, percent: Double): Any {
            if (samples.isEmpty()) return JSONObject.NULL
            val ordered = samples.sorted()
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
        const val SIZE = 320
        const val MIN_ANALYSED_FRAMES = 100
        const val PROVIDER_TIMEOUT_SECONDS = 10L
        const val CAPTURE_TIMEOUT_SECONDS = 20L
        const val WORKER_DRAIN_SETTLE_MS = 500L
        const val MAX_RESULT_AGE_NANOS = 150_000_000L
        const val TAG = "LiveCameraYoloSidecar"
    }
}
