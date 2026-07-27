package com.linnan.blindassist.benchmark

import android.Manifest
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Surface
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
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
import com.linnan.blindassist.session.DetectorMetrics
import com.linnan.blindassist.vision.ObjectDetector
import com.linnan.blindassist.vision.RgbaVisionFrame
import com.linnan.blindassist.vision.RuntimeObjectDetectorFactory
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/** Production-routed live CameraX throughput probe with delivery and processing rates kept separate. */
@RunWith(AndroidJUnit4::class)
class LiveCameraYoloOnlyDeviceTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    @androidx.annotation.OptIn(markerClass = [ExperimentalCamera2Interop::class])
    fun liveCamera_yoloOnly_reportsAggregateTimingOnly() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val provider = ProcessCameraProvider.getInstance(instrumentation.targetContext).get(PROVIDER_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        val detector = RuntimeObjectDetectorFactory.create(instrumentation.targetContext)
        check(detector.isReady) { detector.statusMessage }
        val owner = TestLifecycleOwner()
        val analyserExecutor = Executors.newSingleThreadExecutor()
        val metrics = Metrics()
        val durationReached = CountDownLatch(1)
        val typedDetector = detector as? TfliteYoloDetector
        val actualBackend = typedDetector?.executionBackend?.wireName ?: detector.javaClass.name
        val readyStatus = detector.statusMessage
        val previewTexture = SurfaceTexture(false)
        val previewSurface = Surface(previewTexture)
        val previewBuilder = Preview.Builder()
            .setTargetFrameRate(Range(TARGET_FPS, TARGET_FPS))
        Camera2Interop.Extender(previewBuilder).setSessionCaptureCallback(
            object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(
                    session: CameraCaptureSession,
                    request: CaptureRequest,
                    result: TotalCaptureResult
                ) {
                    result.get(android.hardware.camera2.CaptureResult.SENSOR_TIMESTAMP)
                        ?.let(metrics::recordProducedCapture)
                }
            }
        )
        val preview = previewBuilder.build()
        instrumentation.runOnMainSync {
            preview.setSurfaceProvider { request ->
                previewTexture.setDefaultBufferSize(request.resolution.width, request.resolution.height)
                request.provideSurface(previewSurface, analyserExecutor) {}
            }
        }
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                        )
                    )
                    .build()
            )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(analyserExecutor) { image ->
            if (detectAndClose(image, detector, metrics) && metrics.elapsedNanos() >= MEASURE_NANOS) {
                durationReached.countDown()
            }
        }

        try {
            instrumentation.runOnMainSync {
                provider.unbind(analysis)
                owner.resume()
                provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            }
            assertTrue(
                "CameraX production route did not sustain ${MEASURE_SECONDS}s within ${CAPTURE_TIMEOUT_SECONDS}s",
                durationReached.await(CAPTURE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            )
        } finally {
            instrumentation.runOnMainSync {
                provider.unbind(preview, analysis)
                preview.setSurfaceProvider(null)
                owner.destroy()
            }
            previewSurface.release()
            previewTexture.release()
            analysis.clearAnalyzer()
            detector.close()
            analyserExecutor.shutdown()
            analyserExecutor.awaitTermination(2, TimeUnit.SECONDS)
        }

        val result = metrics.toJson()
            .put("schema", "blindassist_live_camera_production_route_v4")
            .put("device_under_test", android.os.Build.MODEL)
            .put("camera_selector", "back")
            .put("requested_analysis_resolution", "${ANALYSIS_WIDTH}x${ANALYSIS_HEIGHT}")
            .put(
                "input_format",
                "CameraX Preview plus RGBA_8888 ImageAnalysis, KEEP_ONLY_LATEST, production resolution selector"
            )
            .put("detector", "formal RuntimeObjectDetectorFactory route")
            .put("actual_backend", actualBackend)
            .put("detector_ready_status_before_measurement", readyStatus)
            .put("measurement_seconds_target", MEASURE_SECONDS)
            .put("requested_camera_fps_range", "${TARGET_FPS}-${TARGET_FPS}")
            .put("analysis_resolution_target_accepted", metrics.matchesResolution(ANALYSIS_WIDTH, ANALYSIS_HEIGHT))
            .put("raw_frames_persisted", false)
            .put("alerts_invoked", false)
            .put("production_routing_changed", false)
            .put(
                "important_limit",
                "CameraX Preview plus analyzer and production-routed detector. It does not measure Compose rendering FPS. " +
                    "Analyzer loss is matched against Camera2 capture-result sensor timestamps inside the analyzer window."
            )
        val output = File(
            checkNotNull(instrumentation.targetContext.getExternalFilesDir(null)),
            RESULT_RELATIVE_PATH
        )
        output.parentFile?.mkdirs()
        output.writeText(result.toString(2), Charsets.UTF_8)
        Log.i(TAG, "LIVE_CAMERA_YOLO_ONLY_JSON $result")
        assertTrue("camera frames were not analysed", metrics.analysed.get() > 0)
        assertTrue(
            "CameraX did not select ${ANALYSIS_WIDTH}x${ANALYSIS_HEIGHT}",
            metrics.matchesResolution(ANALYSIS_WIDTH, ANALYSIS_HEIGHT)
        )
        assertTrue(
            "Camera did not accept fixed $TARGET_FPS FPS",
            result.getBoolean("camera_target_fps_request_accepted")
        )
        assertTrue("detector failure: ${metrics.failureMessage}", metrics.failures.get() == 0)
        assertTrue(
            "formal device route did not select Qualcomm QNN HTP: ${typedDetector?.executionBackend}",
            typedDetector?.executionBackend?.wireName == "qualcomm_qnn_htp"
        )
    }

    private fun detectAndClose(image: ImageProxy, detector: ObjectDetector, metrics: Metrics): Boolean {
        val receivedAt = SystemClock.elapsedRealtimeNanos()
        val capturedAt = image.imageInfo.timestamp
        val frame = ImageProxyRgbaFrame(image)
        return try {
            val detectorResult = detector.detect(frame)
            metrics.record(
                capturedAtNanos = capturedAt,
                receivedAtNanos = receivedAt,
                completedAtNanos = SystemClock.elapsedRealtimeNanos(),
                width = image.width,
                height = image.height,
                rotationDegrees = image.imageInfo.rotationDegrees,
                rowStride = image.planes.first().rowStride,
                pixelStride = image.planes.first().pixelStride,
                detectorMetrics = detectorResult.metrics
            )
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
        private val captureGapsMicros = ArrayList<Long>()
        private val preprocessMillis = ArrayList<Long>()
        private val inferenceMillis = ArrayList<Long>()
        private val postprocessMillis = ArrayList<Long>()
        private val detectorTotalMillis = ArrayList<Long>()
        private val producedCaptureTimestamps = ArrayList<Long>()
        private val analysedCaptureTimestamps = ArrayList<Long>()
        private val lock = Any()
        private var firstReceivedNanos = 0L
        private var lastCompletedNanos = 0L
        private var firstCapturedNanos = 0L
        private var lastCapturedNanos = 0L
        private var lastSeenCapturedNanos = 0L
        private var observedWidth = 0
        private var observedHeight = 0
        private var observedRotationDegrees = 0
        private var observedRowStride = 0
        private var observedPixelStride = 0
        private var frameLayoutChangeCount = 0
        @Volatile var failureMessage: String? = null

        fun recordProducedCapture(capturedAtNanos: Long) {
            if (capturedAtNanos > 0L) {
                synchronized(lock) { producedCaptureTimestamps += capturedAtNanos }
            }
        }

        fun record(
            capturedAtNanos: Long,
            receivedAtNanos: Long,
            completedAtNanos: Long,
            width: Int,
            height: Int,
            rotationDegrees: Int,
            rowStride: Int,
            pixelStride: Int,
            detectorMetrics: DetectorMetrics
        ) {
            analysed.incrementAndGet()
            synchronized(lock) {
                if (firstReceivedNanos == 0L) {
                    firstReceivedNanos = receivedAtNanos
                    firstCapturedNanos = capturedAtNanos
                }
                if (lastSeenCapturedNanos != 0L && capturedAtNanos > lastSeenCapturedNanos) {
                    captureGapsMicros += (capturedAtNanos - lastSeenCapturedNanos) / 1_000L
                }
                lastSeenCapturedNanos = capturedAtNanos
                lastCapturedNanos = capturedAtNanos
                analysedCaptureTimestamps += capturedAtNanos
                lastCompletedNanos = completedAtNanos
                if (observedWidth != 0 && (
                        observedWidth != width ||
                            observedHeight != height ||
                            observedRotationDegrees != rotationDegrees ||
                            observedRowStride != rowStride ||
                            observedPixelStride != pixelStride
                        )
                ) {
                    frameLayoutChangeCount += 1
                }
                observedWidth = width
                observedHeight = height
                observedRotationDegrees = rotationDegrees
                observedRowStride = rowStride
                observedPixelStride = pixelStride
                samplesMicros += (completedAtNanos - receivedAtNanos) / 1_000L
                preprocessMillis += detectorMetrics.preprocessMs
                inferenceMillis += detectorMetrics.inferenceMs
                postprocessMillis += detectorMetrics.postprocessMs
                detectorTotalMillis += detectorMetrics.totalMs
            }
        }

        fun elapsedNanos(): Long = synchronized(lock) {
            if (firstReceivedNanos == 0L) 0L else lastCompletedNanos - firstReceivedNanos
        }

        fun matchesResolution(width: Int, height: Int): Boolean = synchronized(lock) {
            observedWidth == width && observedHeight == height
        }

        fun recordFailure(failure: Throwable) {
            failures.incrementAndGet()
            failureMessage = failure.javaClass.simpleName + ": " + (failure.message ?: "no message")
        }

        fun toJson(): JSONObject = synchronized(lock) {
            val processingSeconds = (lastCompletedNanos - firstReceivedNanos).coerceAtLeast(1L) / 1_000_000_000.0
            val captureSeconds = (lastCapturedNanos - firstCapturedNanos).coerceAtLeast(1L) / 1_000_000_000.0
            val medianGapMicros = percentileMicros(captureGapsMicros, 50.0)
            val largeGapThresholdMicros = medianGapMicros?.times(1.5)
            val largeGapCount = largeGapThresholdMicros?.let { threshold ->
                captureGapsMicros.count { it > threshold }
            } ?: 0
            val producedInWindow = producedCaptureTimestamps
                .asSequence()
                .filter { it in firstCapturedNanos..lastCapturedNanos }
                .distinct()
                .sorted()
                .toList()
            val analysedTimestamps = analysedCaptureTimestamps.toHashSet()
            val producerMatchedAnalyserCount = producedInWindow.count(analysedTimestamps::contains)
            val producerMissingFromAnalyserCount =
                (producedInWindow.size - producerMatchedAnalyserCount).coerceAtLeast(0)
            val producerSeconds = if (producedInWindow.size > 1) {
                (producedInWindow.last() - producedInWindow.first()) / 1_000_000_000.0
            } else {
                0.0
            }
            val producerFps = rate(producedInWindow.size - 1, producerSeconds)
            val producerGapsMicros = producedInWindow.zipWithNext { earlier, later ->
                (later - earlier) / 1_000L
            }
            val cameraAcceptedTargetFps = producerFps >= MIN_ACCEPTED_CAMERA_FPS
            JSONObject()
                .put("analysed_frame_count", analysed.get())
                .put("observed_resolution", "${observedWidth}x${observedHeight}")
                .put("observed_rotation_degrees", observedRotationDegrees)
                .put("observed_row_stride", observedRowStride)
                .put("observed_pixel_stride", observedPixelStride)
                .put("frame_layout_change_count", frameLayoutChangeCount)
                .put("measurement_wall_seconds", processingSeconds)
                .put("camera_delivery_fps_from_capture_timestamps", rate(analysed.get() - 1, captureSeconds))
                .put("completed_processing_fps_wall_clock", rate(analysed.get(), processingSeconds))
                .put("capture_gap_p50_ms", nullableMillis(medianGapMicros))
                .put("capture_gap_p95_ms", percentileMillis(captureGapsMicros, 95.0))
                .put("capture_gap_max_ms", nullableMillis(captureGapsMicros.maxOrNull()))
                .put("capture_gap_gt_1_5x_median_count", largeGapCount)
                .put("capture_gap_sample_count", captureGapsMicros.size)
                .put("camera_capture_result_count_in_analyser_window", producedInWindow.size)
                .put("camera_capture_result_fps", producerFps)
                .put("camera_capture_result_gap_p50_ms", percentileMillis(producerGapsMicros, 50.0))
                .put("camera_capture_result_gap_p95_ms", percentileMillis(producerGapsMicros, 95.0))
                .put("camera_target_fps_request_accepted", cameraAcceptedTargetFps)
                .put("camera_target_fps_acceptance_min_fps", MIN_ACCEPTED_CAMERA_FPS)
                .put("capture_results_matched_to_analyser", producerMatchedAnalyserCount)
                .put("capture_results_missing_from_analyser", producerMissingFromAnalyserCount)
                .put(
                    "analyser_drop_rate_from_capture_results",
                    if (producedInWindow.isEmpty()) {
                        JSONObject.NULL
                    } else {
                        producerMissingFromAnalyserCount.toDouble() / producedInWindow.size
                    }
                )
                .put("analyser_detect_p50_ms", percentileMillis(samplesMicros, 50.0))
                .put("analyser_detect_p95_ms", percentileMillis(samplesMicros, 95.0))
                .put("analyser_detect_max_ms", nullableMillis(samplesMicros.maxOrNull()))
                .put("detector_preprocess_p50_ms", percentileValue(preprocessMillis, 50.0))
                .put("detector_preprocess_p95_ms", percentileValue(preprocessMillis, 95.0))
                .put("detector_inference_p50_ms", percentileValue(inferenceMillis, 50.0))
                .put("detector_inference_p95_ms", percentileValue(inferenceMillis, 95.0))
                .put("detector_postprocess_p50_ms", percentileValue(postprocessMillis, 50.0))
                .put("detector_postprocess_p95_ms", percentileValue(postprocessMillis, 95.0))
                .put("detector_total_p50_ms", percentileValue(detectorTotalMillis, 50.0))
                .put("detector_total_p95_ms", percentileValue(detectorTotalMillis, 95.0))
                .put("detector_failure_count", failures.get())
        }

        private fun percentileMicros(samples: List<Long>, percent: Double): Long? {
            if (samples.isEmpty()) return null
            val ordered = samples.sorted()
            return ordered[kotlin.math.ceil(ordered.size * percent / 100.0).toInt() - 1]
        }

        private fun percentileMillis(samples: List<Long>, percent: Double): Any =
            nullableMillis(percentileMicros(samples, percent))

        private fun percentileValue(samples: List<Long>, percent: Double): Any =
            percentileMicros(samples, percent) ?: JSONObject.NULL

        private fun nullableMillis(micros: Number?): Any =
            micros?.toDouble()?.div(1_000.0) ?: JSONObject.NULL

        private fun rate(frames: Int, seconds: Double): Double =
            if (frames <= 0 || seconds <= 0.0) 0.0 else frames / seconds
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
        const val ANALYSIS_WIDTH = 640
        const val ANALYSIS_HEIGHT = 480
        const val TARGET_FPS = 24
        const val MIN_ACCEPTED_CAMERA_FPS = 22.0
        const val MEASURE_SECONDS = 60L
        const val MEASURE_NANOS = MEASURE_SECONDS * 1_000_000_000L
        const val PROVIDER_TIMEOUT_SECONDS = 10L
        const val CAPTURE_TIMEOUT_SECONDS = 75L
        const val RESULT_RELATIVE_PATH = "benchmark-results/live-camera-production-route.json"
        const val TAG = "LiveCameraYoloOnly"
    }
}
