package com.linnan.blindassist.benchmark

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
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
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import java.nio.ByteBuffer
import java.util.concurrent.Callable
import java.security.MessageDigest
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** Same-device timing A/B only: shipped YOLO versus shipped YOLO plus sparse-LK geometry. */
@RunWith(AndroidJUnit4::class)
class SparseLkYoloCombinedDeviceBenchmarkTest {
    private val context = InstrumentationRegistry.getInstrumentation().context

    @Test
    fun compareCurrentYoloWithRgbaSparseLkIncrement() {
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
        assertEquals(50, rgba.size)
        val yoloOnly = runBranch(rgba, maxCorners = null)
        val candidates = JSONObject()
        listOf(300, 200, 150).forEach { corners -> candidates.put("rgba_sparse_lk_$corners", runBranch(rgba, maxCorners = corners)) }
        val concurrentSidecar = runConcurrentSidecar(rgba, maxCorners = 300)
        val output = JSONObject()
            .put("schema", "blindassist_yolo_sparse_lk_same_device_ab_v1")
            .put("device_under_test", android.os.Build.MODEL)
            .put("source", "50 hash-verified SANPO redacted RGB frames")
            .put("receipt_sha256", RECEIPT_SHA256)
            .put("branches", JSONObject().put("current_yolo_only", yoloOnly).put("candidates", candidates).put("concurrent_sidecar_300", concurrentSidecar))
            .put("training_authorized", false)
            .put("production_model_replacement_authorized", false)
            .put("important_limit", "Timing A/B only. The concurrent sidecar is a scheduling experiment, not production routing. No event labels, alert metrics, or production routing are evaluated.")
        Log.i(TAG, "YOLO_SPARSE_LK_AB_JSON $output")
    }

    private fun runBranch(rgba: List<ByteArray>, maxCorners: Int?): JSONArray {
        val detector = TfliteYoloDetector(context)
        val geometry = maxCorners?.let { SparseLkGeometryProbe(maxCorners = it) }
        check(detector.isReady) { detector.statusMessage }
        return try {
            val runs = JSONArray()
            repeat(RUNS) { runIndex ->
                var previous = lumaMat(sampledLuma(rgba[0]))
                try {
                    repeat(WARMUP_TRANSITIONS) { index ->
                        detector.detect(RgbaByteFrame(rgba[index + 1]))
                        if (geometry != null) {
                            val current = lumaMat(sampledLuma(rgba[index + 1])); geometry.measure(previous, current); previous.release(); previous = current
                        }
                    }
                    val samples = LongArray(rgba.size - 1); var successes = 0; var inlierSum = 0.0; var residualSum = 0.0
                    for (index in 1 until rgba.size) {
                        val started = SystemClock.elapsedRealtimeNanos()
                        detector.detect(RgbaByteFrame(rgba[index]))
                        if (geometry != null) {
                            val current = lumaMat(sampledLuma(rgba[index]))
                            val vector = geometry.measure(previous, current)
                            if (vector.success) { successes++; inlierSum += vector.inlierRatio; residualSum += vector.lowerCorridorResidual }
                            previous.release(); previous = current
                        }
                        samples[index - 1] = SystemClock.elapsedRealtimeNanos() - started
                    }
                    runs.put(JSONObject().put("run_index", runIndex + 1).put("transition_count", samples.size).put("geometry_success_rate", if (geometry != null) successes.toDouble() / samples.size else JSONObject.NULL).put("mean_inlier_ratio", if (successes > 0) inlierSum / successes else JSONObject.NULL).put("mean_lower_corridor_residual", if (successes > 0) residualSum / successes else JSONObject.NULL).put("p50_ms", percentile(samples, 50.0)).put("p95_ms", percentile(samples, 95.0)))
                } finally { previous.release() }
            }
            runs
        } finally { geometry?.close(); detector.close() }
    }

    /**
     * Measures CPU contention when full geometry is kept off the detector's
     * serial path.  This is intentionally a benchmark-only scheduling probe:
     * it has neither alert semantics nor an application integration point.
     */
    private fun runConcurrentSidecar(rgba: List<ByteArray>, maxCorners: Int): JSONArray {
        val detector = TfliteYoloDetector(context)
        check(detector.isReady) { detector.statusMessage }
        val geometry = SparseLkGeometryProbe(maxCorners = maxCorners)
        val executor = Executors.newFixedThreadPool(2)
        return try {
            // Warm each worker outside the measurement window, using the same source shape.
            repeat(WARMUP_TRANSITIONS) { index -> detector.detect(RgbaByteFrame(rgba[index + 1])) }
            var warmupPrevious = lumaMat(sampledLuma(rgba[0]))
            try {
                repeat(WARMUP_TRANSITIONS) { index ->
                    val current = lumaMat(sampledLuma(rgba[index + 1]))
                    geometry.measure(warmupPrevious, current)
                    warmupPrevious.release()
                    warmupPrevious = current
                }
            } finally { warmupPrevious.release() }

            JSONArray().also { runs ->
                repeat(RUNS) { runIndex ->
                    val detectorSamples = LongArray(rgba.size - 1)
                    val geometrySamples = LongArray(rgba.size - 1)
                    val start = CountDownLatch(1)
                    val detectorDone = executor.submit {
                        check(start.await(WORKER_TIMEOUT_SECONDS, TimeUnit.SECONDS)) { "detector start timed out" }
                        for (index in 1 until rgba.size) {
                            val began = SystemClock.elapsedRealtimeNanos()
                            detector.detect(RgbaByteFrame(rgba[index]))
                            detectorSamples[index - 1] = SystemClock.elapsedRealtimeNanos() - began
                        }
                    }
                    val geometryDone = executor.submit(Callable {
                        check(start.await(WORKER_TIMEOUT_SECONDS, TimeUnit.SECONDS)) { "geometry start timed out" }
                        var previous = lumaMat(sampledLuma(rgba[0]))
                        var successes = 0
                        var inlierSum = 0.0
                        var residualSum = 0.0
                        try {
                            for (index in 1 until rgba.size) {
                                val began = SystemClock.elapsedRealtimeNanos()
                                val current = lumaMat(sampledLuma(rgba[index]))
                                val vector = geometry.measure(previous, current)
                                geometrySamples[index - 1] = SystemClock.elapsedRealtimeNanos() - began
                                if (vector.success) {
                                    successes++
                                    inlierSum += vector.inlierRatio
                                    residualSum += vector.lowerCorridorResidual
                                }
                                previous.release()
                                previous = current
                            }
                            ConcurrentGeometrySummary(successes, inlierSum, residualSum)
                        } finally { previous.release() }
                    })
                    start.countDown()
                    detectorDone.get(WORKER_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                    val geometrySummary = geometryDone.get(WORKER_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                    runs.put(
                        JSONObject()
                            .put("run_index", runIndex + 1)
                            .put("transition_count", detectorSamples.size)
                            .put("detector_p50_ms", percentile(detectorSamples, 50.0))
                            .put("detector_p95_ms", percentile(detectorSamples, 95.0))
                            .put("sidecar_p50_ms", percentile(geometrySamples, 50.0))
                            .put("sidecar_p95_ms", percentile(geometrySamples, 95.0))
                            .put("geometry_success_rate", geometrySummary.successes.toDouble() / geometrySamples.size)
                            .put("mean_inlier_ratio", if (geometrySummary.successes > 0) geometrySummary.inlierSum / geometrySummary.successes else JSONObject.NULL)
                            .put("mean_lower_corridor_residual", if (geometrySummary.successes > 0) geometrySummary.residualSum / geometrySummary.successes else JSONObject.NULL)
                    )
                }
            }
        } finally {
            executor.shutdownNow()
            geometry.close()
            detector.close()
        }
    }

    private fun rgba640x480(bitmap: Bitmap): ByteArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, CAMERA_WIDTH, CAMERA_HEIGHT, true)
        return try {
            val pixels = IntArray(CAMERA_WIDTH * CAMERA_HEIGHT); scaled.getPixels(pixels, 0, CAMERA_WIDTH, 0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)
            ByteArray(CAMERA_WIDTH * CAMERA_HEIGHT * 4).also { bytes -> pixels.forEachIndexed { index, pixel -> val offset = index * 4; bytes[offset] = ((pixel shr 16) and 0xFF).toByte(); bytes[offset + 1] = ((pixel shr 8) and 0xFF).toByte(); bytes[offset + 2] = (pixel and 0xFF).toByte(); bytes[offset + 3] = ((pixel ushr 24) and 0xFF).toByte() } }
        } finally { if (scaled !== bitmap) scaled.recycle() }
    }

    private fun sampledLuma(rgba: ByteArray): ByteArray = ByteArray(SIZE * SIZE).also { output -> for (y in 0 until SIZE) for (x in 0 until SIZE) { val offset = ((y * CAMERA_HEIGHT / SIZE) * CAMERA_WIDTH + x * CAMERA_WIDTH / SIZE) * 4; val r = rgba[offset].toInt() and 0xFF; val g = rgba[offset + 1].toInt() and 0xFF; val b = rgba[offset + 2].toInt() and 0xFF; output[y * SIZE + x] = ((77 * r + 150 * g + 29 * b) ushr 8).toByte() } }
    private fun lumaMat(bytes: ByteArray): Mat = Mat(SIZE, SIZE, CvType.CV_8UC1).also { it.put(0, 0, bytes) }
    private fun percentile(values: LongArray, percent: Double): Double = values.sortedArray()[kotlin.math.ceil(values.size * percent / 100.0).toInt() - 1].toDouble() / 1_000_000.0
    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    private class RgbaByteFrame(private val bytes: ByteArray) : RgbaVisionFrame {
        override val width = CAMERA_WIDTH; override val height = CAMERA_HEIGHT; override val rotationDegrees = 0; override val rowStride = CAMERA_WIDTH * 4; override val pixelStride = 4; override val buffer: ByteBuffer = ByteBuffer.wrap(bytes)
        override fun close() = Unit
    }

    private data class ConcurrentGeometrySummary(val successes: Int, val inlierSum: Double, val residualSum: Double)

    private companion object { const val ASSET_ROOT = "sparse_lk_sanpo"; const val RECEIPT_SHA256 = "d99b794c235e0f24a0656fe8da3f1719b283a795571e6de323e03997a9501129"; const val CAMERA_WIDTH = 640; const val CAMERA_HEIGHT = 480; const val SIZE = 320; const val MAX_CORNERS = 300; const val RUNS = 3; const val WARMUP_TRANSITIONS = 5; const val WORKER_TIMEOUT_SECONDS = 30L; const val TAG = "SparseLkYoloAb" }
}
