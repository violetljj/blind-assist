package com.linnan.blindassist.benchmark

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.vision.LatestOnlySidecar
import com.linnan.blindassist.vision.RgbaLumaResampler
import com.linnan.blindassist.vision.RgbaVisionFrame
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.nio.ByteBuffer
import java.security.MessageDigest
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Exercises the production-shaped hand-off: the detector thread copies luma,
 * submits an owned frame to a one-slot sidecar, then invokes YOLO. No Android
 * camera, alert policy, or production routing is changed by this test APK.
 */
@RunWith(AndroidJUnit4::class)
class LatestOnlySparseLkSidecarDeviceBenchmarkTest {
    private val context = InstrumentationRegistry.getInstrumentation().context
    private val lumaResampler = RgbaLumaResampler(SIZE)

    @Test
    fun benchmarkLatestOnlySparseLkSidecarUnderDetectorPacing() {
        assertTrue("OpenCV local runtime failed to load", OpenCVLoader.initLocal())
        val receiptBytes = context.assets.open("$ASSET_ROOT/machine_redaction_receipt.json").use { it.readBytes() }
        assertEquals(RECEIPT_SHA256, sha256(receiptBytes))
        val receipt = JSONObject(receiptBytes.toString(Charsets.UTF_8))
        assertFalse(receipt.getBoolean("risk_or_event_truth_present"))
        assertFalse(receipt.getBoolean("training_execution_authorized"))
        val rgba = receipt.getJSONArray("frames").let { rows ->
            List(rows.length()) { index ->
                val row = rows.getJSONObject(index)
                val bytes = context.assets.open("$ASSET_ROOT/images/${row.getString("file_name")}").use { it.readBytes() }
                assertEquals(row.getString("sha256"), sha256(bytes))
                val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: error("cannot decode frame")
                try { rgba640x480(bitmap) } finally { bitmap.recycle() }
            }
        }
        assertEquals(FRAME_COUNT, rgba.size)
        val lumaCopyBaseline = runDetectorWithLumaCopy(rgba)
        val latestOnlySidecar = runLatestOnlySidecar(rgba)
        val interleaved = runInterleavedLumaCopyAndSidecar(rgba)
        val output = JSONObject()
            .put("schema", "blindassist_latest_only_sparse_lk_sidecar_device_v1")
            .put("device_under_test", android.os.Build.MODEL)
            .put("source", "50 hash-verified SANPO redacted RGB frames")
            .put("receipt_sha256", RECEIPT_SHA256)
            .put("max_result_age_ms", MAX_RESULT_AGE_NANOS / 1_000_000.0)
            .put("branches", JSONObject().put("yolo_with_luma_copy_only", lumaCopyBaseline).put("latest_only_sidecar_300", latestOnlySidecar).put("interleaved_luma_copy_and_sidecar_300", interleaved))
            .put("training_authorized", false)
            .put("production_model_replacement_authorized", false)
            .put("important_limit", "Timing and bounded-queue test only. No camera lifecycle, event labels, alert metrics, or production routing are evaluated.")
        Log.i(TAG, "LATEST_ONLY_SPARSE_LK_SIDECAR_JSON $output")
    }

    private fun runDetectorWithLumaCopy(rgba: List<ByteArray>): JSONArray {
        val detector = TfliteYoloDetector(context)
        check(detector.isReady) { detector.statusMessage }
        return try {
            JSONArray().also { runs ->
                repeat(RUNS) { runIndex ->
                    repeat(WARMUP_TRANSITIONS) { index ->
                        sampledLuma(rgba[index + 1])
                        detector.detect(RgbaByteFrame(rgba[index + 1]))
                    }
                    val samples = LongArray(TRANSITIONS)
                    for (index in 1 until rgba.size) {
                        val began = SystemClock.elapsedRealtimeNanos()
                        sampledLuma(rgba[index])
                        detector.detect(RgbaByteFrame(rgba[index]))
                        samples[index - 1] = SystemClock.elapsedRealtimeNanos() - began
                    }
                    runs.put(JSONObject().put("run_index", runIndex + 1).put("transition_count", samples.size).put("detector_p50_ms", percentile(samples, 50.0)).put("detector_p95_ms", percentile(samples, 95.0)))
                }
            }
        } finally { detector.close() }
    }

    private fun runLatestOnlySidecar(rgba: List<ByteArray>): JSONArray {
        val detector = TfliteYoloDetector(context)
        check(detector.isReady) { detector.statusMessage }
        return try {
            repeat(WARMUP_TRANSITIONS) { index ->
                detector.detect(RgbaByteFrame(rgba[index + 1]))
            }
            JSONArray().also { runs ->
                repeat(RUNS) { runIndex ->
                    runs.put(runMeasuredLatestOnlySidecar(rgba, runIndex + 1, detector))
                }
            }
        } finally { detector.close() }
    }

    /** Fixed alternating order avoids treating sequential branch timing as a speed comparison. */
    private fun runInterleavedLumaCopyAndSidecar(rgba: List<ByteArray>): JSONArray {
        val detector = TfliteYoloDetector(context)
        check(detector.isReady) { detector.statusMessage }
        return try {
            repeat(WARMUP_TRANSITIONS) { index ->
                sampledLuma(rgba[index + 1])
                detector.detect(RgbaByteFrame(rgba[index + 1]))
            }
            val order = listOf("luma_copy_only", "latest_only_sidecar_300", "latest_only_sidecar_300", "luma_copy_only", "luma_copy_only", "latest_only_sidecar_300")
            JSONArray().also { rows ->
                order.forEachIndexed { index, branch ->
                    val row = if (branch == "luma_copy_only") {
                        runDetectorWithLumaCopyOnce(rgba, index + 1, detector)
                    } else {
                        runMeasuredLatestOnlySidecar(rgba, index + 1, detector)
                    }
                    rows.put(row.put("sequence_index", index + 1).put("branch", branch))
                }
            }
        } finally { detector.close() }
    }

    private fun runDetectorWithLumaCopyOnce(rgba: List<ByteArray>, runIndex: Int, detector: TfliteYoloDetector): JSONObject {
        val samples = LongArray(TRANSITIONS)
        for (index in 1 until rgba.size) {
            val began = SystemClock.elapsedRealtimeNanos()
            sampledLuma(rgba[index])
            detector.detect(RgbaByteFrame(rgba[index]))
            samples[index - 1] = SystemClock.elapsedRealtimeNanos() - began
        }
        return JSONObject()
            .put("run_index", runIndex)
            .put("transition_count", samples.size)
            .put("detector_p50_ms", percentile(samples, 50.0))
            .put("detector_p95_ms", percentile(samples, 95.0))
    }

    private fun runMeasuredLatestOnlySidecar(rgba: List<ByteArray>, runIndex: Int, detector: TfliteYoloDetector): JSONObject {
        val executor = Executors.newSingleThreadExecutor()
        val worker = GeometryWorker()
        val primed = CountDownLatch(1)
        val delivered = CountDownLatch(TRANSITIONS)
        val deliveredVectors = AtomicInteger(0)
        val successes = AtomicInteger(0)
        val closeCount = AtomicInteger(0)
        var inlierSum = 0.0
        var residualSum = 0.0
        val metricsLock = Any()
        val sidecar = LatestOnlySidecar<LumaInput, SparseLkGeometryVector?>(
            executor = executor,
            maxResultAgeNanos = MAX_RESULT_AGE_NANOS,
            process = { input -> worker.process(input).also { if (it == null) primed.countDown() } },
            onFreshResult = { result ->
                result.value?.let { vector ->
                    deliveredVectors.incrementAndGet()
                    if (vector.success) {
                        successes.incrementAndGet()
                        synchronized(metricsLock) { inlierSum += vector.inlierRatio; residualSum += vector.lowerCorridorResidual }
                    }
                    delivered.countDown()
                }
            }
        )
        try {
            assertTrue(sidecar.submit(LumaInput(sampledLuma(rgba[0]), closeCount), SystemClock.elapsedRealtimeNanos()))
            assertTrue("sidecar did not prime", primed.await(WORKER_TIMEOUT_SECONDS, TimeUnit.SECONDS))
            val samples = LongArray(TRANSITIONS)
            for (index in 1 until rgba.size) {
                val captureTime = SystemClock.elapsedRealtimeNanos()
                val luma = sampledLuma(rgba[index])
                val began = SystemClock.elapsedRealtimeNanos()
                assertTrue(sidecar.submit(LumaInput(luma, closeCount), captureTime))
                detector.detect(RgbaByteFrame(rgba[index]))
                samples[index - 1] = SystemClock.elapsedRealtimeNanos() - began
            }
            assertTrue("sidecar did not deliver every measured vector", delivered.await(WORKER_TIMEOUT_SECONDS, TimeUnit.SECONDS))
            assertEquals(TRANSITIONS, deliveredVectors.get())
            assertEquals(FRAME_COUNT, closeCount.get())
            val successCount = successes.get()
            val averages = synchronized(metricsLock) { Pair(if (successCount > 0) inlierSum / successCount else JSONObject.NULL, if (successCount > 0) residualSum / successCount else JSONObject.NULL) }
            return JSONObject()
                .put("run_index", runIndex)
                .put("transition_count", TRANSITIONS)
                .put("detector_p50_ms", percentile(samples, 50.0))
                .put("detector_p95_ms", percentile(samples, 95.0))
                .put("submitted_input_count", FRAME_COUNT)
                .put("closed_input_count", closeCount.get())
                .put("delivered_vector_count", deliveredVectors.get())
                .put("geometry_success_rate", successCount.toDouble() / TRANSITIONS)
                .put("mean_inlier_ratio", averages.first)
                .put("mean_lower_corridor_residual", averages.second)
        } finally {
            sidecar.close()
            executor.shutdownNow()
            worker.close()
        }
    }

    private fun sampledLuma(rgba: ByteArray): ByteArray = lumaResampler.sample(RgbaByteFrame(rgba))

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
        } finally { if (scaled !== bitmap) scaled.recycle() }
    }

    private fun lumaMat(bytes: ByteArray): Mat = Mat(SIZE, SIZE, CvType.CV_8UC1).also { it.put(0, 0, bytes) }
    private fun percentile(values: LongArray, percent: Double): Double = values.sortedArray()[kotlin.math.ceil(values.size * percent / 100.0).toInt() - 1].toDouble() / 1_000_000.0
    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    private class LumaInput(val bytes: ByteArray, private val closeCount: AtomicInteger) : AutoCloseable {
        private val closed = AtomicBoolean(false)
        override fun close() { if (closed.compareAndSet(false, true)) closeCount.incrementAndGet() }
    }

    private class GeometryWorker : AutoCloseable {
        private val probe = SparseLkGeometryProbe(maxCorners = 300)
        private var previous: Mat? = null

        fun process(input: LumaInput): SparseLkGeometryVector? {
            val current = lumaMatStatic(input.bytes)
            val prior = previous
            if (prior == null) {
                previous = current
                return null
            }
            return try { probe.measure(prior, current) } finally { prior.release(); previous = current }
        }

        override fun close() { previous?.release(); previous = null; probe.close() }

        private companion object {
            fun lumaMatStatic(bytes: ByteArray): Mat = Mat(SIZE, SIZE, CvType.CV_8UC1).also { it.put(0, 0, bytes) }
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

    private companion object {
        const val ASSET_ROOT = "sparse_lk_sanpo"
        const val RECEIPT_SHA256 = "d99b794c235e0f24a0656fe8da3f1719b283a795571e6de323e03997a9501129"
        const val CAMERA_WIDTH = 640
        const val CAMERA_HEIGHT = 480
        const val SIZE = 320
        const val FRAME_COUNT = 50
        const val TRANSITIONS = FRAME_COUNT - 1
        const val RUNS = 3
        const val WARMUP_TRANSITIONS = 5
        const val WORKER_TIMEOUT_SECONDS = 10L
        const val MAX_RESULT_AGE_NANOS = 150_000_000L
        const val TAG = "LatestOnlySparseLk"
    }
}
