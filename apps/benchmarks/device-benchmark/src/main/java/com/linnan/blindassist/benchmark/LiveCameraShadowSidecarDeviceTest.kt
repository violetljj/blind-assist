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
import com.linnan.blindassist.vision.RgbaVisionFrame
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
 * A test-APK-only, real-CameraX shadow check for the optional geometry sidecar.
 *
 * It does not invoke YOLO, RiskEventTracker, speech, vibration, or any alert
 * policy. Camera frames are not persisted: the analyser hands a copied luma
 * buffer to [RgbaLumaSidecar], whose owned buffer is cleared after processing.
 */
@RunWith(AndroidJUnit4::class)
class LiveCameraShadowSidecarDeviceTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun liveCamera_shadowSidecar_reportsOnlyAggregateFreshnessAndLatency() {
        assertTrue("OpenCV local runtime failed to load", OpenCVLoader.initLocal())
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val provider = ProcessCameraProvider.getInstance(context).get(PROVIDER_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        val owner = TestLifecycleOwner()
        val analyserExecutor = Executors.newSingleThreadExecutor()
        val workerExecutor = Executors.newSingleThreadExecutor()
        val metrics = Metrics()
        val enoughFrames = CountDownLatch(1)
        val worker = GeometryWorker()
        val sidecar = RgbaLumaSidecar(
            executor = workerExecutor,
            maxResultAgeNanos = MAX_RESULT_AGE_NANOS,
            process = worker::process,
            onFreshResult = { result ->
                metrics.recordDelivery(result.ageNanos, result.value)
            },
            onFailure = { failure -> metrics.recordFailure(failure) },
            nowNanos = SystemClock::elapsedRealtimeNanos
        )
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(analyserExecutor) { image ->
            val accepted = submitAndClose(image, sidecar, metrics)
            if (accepted && metrics.accepted.get() >= MIN_ACCEPTED_FRAMES) enoughFrames.countDown()
        }

        try {
            instrumentation.runOnMainSync {
                provider.unbind(analysis)
                owner.resume()
                provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            }
            assertTrue(
                "Camera did not supply $MIN_ACCEPTED_FRAMES frames within ${CAPTURE_TIMEOUT_SECONDS}s",
                enoughFrames.await(CAPTURE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            )
            // Let the single queued worker item finish so delivery freshness is observable.
            Thread.sleep(WORKER_DRAIN_SETTLE_MS)
        } finally {
            instrumentation.runOnMainSync {
                provider.unbind(analysis)
                owner.destroy()
            }
            analysis.clearAnalyzer()
            sidecar.close()
            analyserExecutor.shutdown()
            workerExecutor.shutdown()
            analyserExecutor.awaitTermination(2, TimeUnit.SECONDS)
            workerExecutor.awaitTermination(2, TimeUnit.SECONDS)
            worker.close()
        }

        val result = metrics.toJson()
            .put("schema", "blindassist_live_camera_shadow_sidecar_v1")
            .put("device_under_test", android.os.Build.MODEL)
            .put("camera_selector", "back")
            .put("input_format", "CameraX RGBA_8888, KEEP_ONLY_LATEST")
            .put("sidecar", "320x320 luma plus test-only Sparse-LK five-channel probe")
            .put("max_result_age_ms", MAX_RESULT_AGE_NANOS / 1_000_000.0)
            .put("raw_frames_persisted", false)
            .put("yolo_invoked", false)
            .put("alerts_invoked", false)
            .put("production_routing_changed", false)
            .put("important_limit", "A shadow-only real-camera lifecycle/freshness probe, not an alert-quality or production-performance result.")
        Log.i(TAG, "LIVE_CAMERA_SHADOW_SIDECAR_JSON $result")
        assertTrue("camera frames were not accepted", metrics.accepted.get() >= MIN_ACCEPTED_FRAMES)
        assertTrue("sidecar reported a worker failure: ${metrics.failureMessage}", metrics.failures.get() == 0)
    }

    private fun submitAndClose(image: ImageProxy, sidecar: RgbaLumaSidecar<SparseLkGeometryVector?>, metrics: Metrics): Boolean {
        val began = SystemClock.elapsedRealtimeNanos()
        val frame = ImageProxyRgbaFrame(image)
        return try {
            val accepted = sidecar.submit(frame, began)
            metrics.recordSubmission(SystemClock.elapsedRealtimeNanos() - began, accepted)
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
            return try {
                probe.measure(prior, current)
            } finally {
                prior.release()
                previous = current
            }
        }

        override fun close() {
            previous?.release()
            previous = null
            probe.close()
        }
    }

    private class Metrics {
        val accepted = AtomicInteger()
        private val rejected = AtomicInteger()
        val failures = AtomicInteger()
        private val delivered = AtomicInteger()
        private val geometryVectors = AtomicInteger()
        private val geometrySuccesses = AtomicInteger()
        private val submitMicros = ArrayList<Long>()
        private val resultMicros = ArrayList<Long>()
        private val lock = Any()
        @Volatile var failureMessage: String? = null

        fun recordSubmission(elapsedNanos: Long, wasAccepted: Boolean) {
            if (wasAccepted) accepted.incrementAndGet() else rejected.incrementAndGet()
            synchronized(lock) { submitMicros += elapsedNanos / 1_000L }
        }

        fun recordDelivery(ageNanos: Long, vector: SparseLkGeometryVector?) {
            delivered.incrementAndGet()
            vector?.let {
                geometryVectors.incrementAndGet()
                if (it.success) geometrySuccesses.incrementAndGet()
            }
            synchronized(lock) { resultMicros += ageNanos / 1_000L }
        }

        fun recordFailure(failure: Throwable) {
            failures.incrementAndGet()
            failureMessage = failure.javaClass.simpleName + ": " + (failure.message ?: "no message")
        }

        fun toJson(): JSONObject = synchronized(lock) {
            JSONObject()
                .put("accepted_frame_count", accepted.get())
                .put("rejected_frame_count", rejected.get())
                .put("fresh_result_count", delivered.get())
                .put("geometry_vector_count", geometryVectors.get())
                .put("geometry_success_rate", if (geometryVectors.get() == 0) JSONObject.NULL else geometrySuccesses.get().toDouble() / geometryVectors.get())
                .put("analyser_submit_p50_ms", percentileMillis(submitMicros, 50.0))
                .put("analyser_submit_p95_ms", percentileMillis(submitMicros, 95.0))
                .put("capture_receive_to_result_p50_ms", percentileMillis(resultMicros, 50.0))
                .put("capture_receive_to_result_p95_ms", percentileMillis(resultMicros, 95.0))
                .put("worker_failure_count", failures.get())
        }

        private fun percentileMillis(samples: List<Long>, percent: Double): Any {
            if (samples.isEmpty()) return JSONObject.NULL
            val ordered = samples.sorted()
            val index = kotlin.math.ceil(ordered.size * percent / 100.0).toInt() - 1
            return ordered[index].toDouble() / 1_000.0
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

    private companion object {
        const val SIZE = 320
        const val MIN_ACCEPTED_FRAMES = 100
        const val PROVIDER_TIMEOUT_SECONDS = 10L
        const val CAPTURE_TIMEOUT_SECONDS = 15L
        const val WORKER_DRAIN_SETTLE_MS = 500L
        const val MAX_RESULT_AGE_NANOS = 150_000_000L
        const val TAG = "LiveCameraShadow"
    }
}
