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
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.ceil
import kotlin.math.roundToInt

/**
 * Test-APK-only live CameraX timing probe. It runs the default YOLO detector and then a
 * feature-only, untrained INT8 event-head fixture. It never routes any output to risk, feedback,
 * speech or vibration, and it does not persist camera frames or detections.
 */
@RunWith(AndroidJUnit4::class)
class LiveCameraYoloEventHeadDeviceTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun liveCamera_yoloPlusUntrainedEventHead_reportsAggregateTimingOnly() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val provider = ProcessCameraProvider.getInstance(instrumentation.targetContext).get(PROVIDER_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        val detector = TfliteYoloDetector(instrumentation.context)
        check(detector.isReady) { detector.statusMessage }
        val eventHead = EventHeadFixture(instrumentation.context)
        val owner = TestLifecycleOwner()
        val executor = Executors.newSingleThreadExecutor()
        val metrics = Metrics()
        val enoughFrames = CountDownLatch(1)
        val window = YoloImuCausalWindow()
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(executor) { image ->
            if (detectFeatureHeadAndClose(image, detector, eventHead, window, metrics) && metrics.analysed.get() >= MIN_ANALYSED_FRAMES) {
                enoughFrames.countDown()
            }
        }
        try {
            instrumentation.runOnMainSync {
                provider.unbind(analysis)
                owner.resume()
                provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            }
            assertTrue("Camera did not analyse $MIN_ANALYSED_FRAMES frames within ${CAPTURE_TIMEOUT_SECONDS}s", enoughFrames.await(CAPTURE_TIMEOUT_SECONDS, TimeUnit.SECONDS))
        } finally {
            instrumentation.runOnMainSync { provider.unbind(analysis); owner.destroy() }
            analysis.clearAnalyzer()
            executor.shutdown()
            executor.awaitTermination(2, TimeUnit.SECONDS)
            eventHead.close()
            detector.close()
        }
        val result = metrics.toJson()
            .put("schema", "blindassist_live_camera_yolo_event_head_fixture_v0")
            .put("device_under_test", android.os.Build.MODEL)
            .put("camera_selector", "back")
            .put("input_format", "CameraX RGBA_8888, KEEP_ONLY_LATEST")
            .put("detector", "existing YOLO test-APK asset")
            .put("event_head", "untrained INT8 fixture; no output semantics")
            .put("raw_frames_persisted", false)
            .put("detections_persisted", false)
            .put("alerts_invoked", false)
            .put("production_routing_changed", false)
            .put("important_limit", "A real-camera timing fixture only; it does not establish event quality, calibration, shadow authorization or production readiness.")
        Log.i(TAG, "LIVE_CAMERA_YOLO_EVENT_HEAD_JSON $result")
        assertTrue("camera frames were not analysed", metrics.analysed.get() >= MIN_ANALYSED_FRAMES)
        assertTrue("fixture failure: ${metrics.failureMessage}", metrics.failures.get() == 0)
        assertTrue("total P95 exceeded $TOTAL_P95_BUDGET_MS ms: ${metrics.p95Millis()}", metrics.p95Millis() <= TOTAL_P95_BUDGET_MS)
    }

    private fun detectFeatureHeadAndClose(
        image: ImageProxy,
        detector: TfliteYoloDetector,
        head: EventHeadFixture,
        window: YoloImuCausalWindow,
        metrics: Metrics
    ): Boolean {
        val started = SystemClock.elapsedRealtimeNanos()
        val frame = ImageProxyRgbaFrame(image)
        return try {
            val detection = detector.detect(frame)
            val sequence = window.append(YoloImuFeatureFrame(SystemClock.elapsedRealtimeNanos(), detection.detections))
            if (sequence != null) {
                head.run(sequence)
                metrics.recordHeadRun()
            }
            metrics.record(SystemClock.elapsedRealtimeNanos() - started)
            true
        } catch (failure: Throwable) {
            metrics.recordFailure(failure)
            false
        } finally {
            frame.close()
        }
    }

    private class EventHeadFixture(context: android.content.Context) : AutoCloseable {
        private val interpreter = Interpreter(mapAsset(context), Interpreter.Options().setNumThreads(1).setUseXNNPACK(true)).also { it.allocateTensors() }
        private val motionInput = ByteBuffer.allocateDirect(YoloImuCausalWindow.MOTION_VALUES_PER_FRAME * 8).order(ByteOrder.nativeOrder())
        private val spatialInput = ByteBuffer.allocateDirect(YoloImuCausalWindow.GRID_VALUES_PER_FRAME * 8).order(ByteOrder.nativeOrder())
        private val outputs = hashMapOf<Int, Any>(0 to output(8), 1 to output(4), 2 to output(1), 3 to output(3), 4 to output(1))

        fun run(sequence: YoloImuCausalSequence) {
            quantizeInto(sequence.motionSequence, MOTION_SCALE, MOTION_ZERO, motionInput)
            quantizeInto(sequence.spatialSequence, SPATIAL_SCALE, SPATIAL_ZERO, spatialInput)
            outputs.values.forEach { (it as ByteBuffer).rewind() }
            interpreter.runForMultipleInputsOutputs(
                arrayOf(motionInput, spatialInput),
                outputs
            )
        }

        override fun close() = interpreter.close()

        private fun mapAsset(context: android.content.Context): MappedByteBuffer = context.assets.openFd(MODEL_ASSET).use { descriptor ->
            FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
                channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
            }
        }

        private fun quantizeInto(values: FloatArray, scale: Float, zero: Int, target: ByteBuffer) {
            target.rewind()
            values.forEach { value -> target.put((value / scale).roundToInt().plus(zero).coerceIn(-128, 127).toByte()) }
            target.rewind()
        }

        private fun output(count: Int): ByteBuffer = ByteBuffer.allocateDirect(count).order(ByteOrder.nativeOrder())
    }

    private class Metrics {
        val analysed = AtomicInteger()
        val failures = AtomicInteger()
        private val headRuns = AtomicInteger()
        private val samplesMicros = ArrayList<Long>()
        private val lock = Any()
        @Volatile var failureMessage: String? = null

        fun record(elapsedNanos: Long) { analysed.incrementAndGet(); synchronized(lock) { samplesMicros += elapsedNanos / 1_000L } }
        fun recordHeadRun() { headRuns.incrementAndGet() }
        fun recordFailure(failure: Throwable) { failures.incrementAndGet(); failureMessage = failure.javaClass.simpleName + ": " + (failure.message ?: "no message") }
        fun p95Millis(): Double = synchronized(lock) { percentileMillis(95.0) as? Double ?: Double.POSITIVE_INFINITY }
        fun toJson(): JSONObject = synchronized(lock) {
            JSONObject().put("analysed_frame_count", analysed.get()).put("event_head_frame_count", headRuns.get()).put("combined_analyser_p50_ms", percentileMillis(50.0)).put("combined_analyser_p95_ms", percentileMillis(95.0)).put("failure_count", failures.get())
        }
        private fun percentileMillis(percent: Double): Any {
            if (samplesMicros.isEmpty()) return JSONObject.NULL
            val ordered = samplesMicros.sorted()
            return ordered[ceil(ordered.size * percent / 100.0).toInt() - 1].toDouble() / 1_000.0
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
        const val MODEL_ASSET = "corridor_causal_tcn_int8_v0.tflite"
        const val MOTION_SCALE = 0.015686275f
        const val MOTION_ZERO = -1
        const val SPATIAL_SCALE = 0.0039215595f
        const val SPATIAL_ZERO = -128
        const val MIN_ANALYSED_FRAMES = 200
        const val PROVIDER_TIMEOUT_SECONDS = 10L
        const val CAPTURE_TIMEOUT_SECONDS = 30L
        const val TOTAL_P95_BUDGET_MS = 70.0
        const val TAG = "LiveCameraYoloEventHead"
    }
}
