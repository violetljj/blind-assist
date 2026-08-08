package com.linnan.blindassist.benchmark

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Mat
import org.opencv.core.CvType
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import java.io.File
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * Test-APK-only timing check for the separately locked Sparse-LK proposal.
 * It has no connection to the production alert path and reads only redacted RGB.
 */
@RunWith(AndroidJUnit4::class)
class SparseLkDeviceBenchmarkTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private lateinit var geometry: SparseLkGeometryProbe

    @Test
    fun benchmarkSparseLk300OnRedactedSanpoFrames() {
        assertTrue("OpenCV local runtime failed to load", OpenCVLoader.initLocal())
        geometry = SparseLkGeometryProbe()
        val receiptBytes = readAssetBytes("$ASSET_ROOT/machine_redaction_receipt.json")
        assertEquals(RECEIPT_SHA256, sha256Bytes(receiptBytes))
        val receiptText = receiptBytes.toString(Charsets.UTF_8)
        val receipt = JSONObject(receiptText)
        assertFalse("source must not contain event truth", receipt.getBoolean("risk_or_event_truth_present"))
        assertFalse("source must not authorize training", receipt.getBoolean("training_execution_authorized"))
        val names = receipt.getJSONArray("frames").let { rows ->
            List(rows.length()) { index -> rows.getJSONObject(index).getString("file_name") }
        }
        assertEquals(50, names.size)
        val frames = names.map { name ->
            val bytes = testContext.assets.open("$ASSET_ROOT/images/$name").use { it.readBytes() }
            val expected = receipt.getJSONArray("frames").let { rows ->
                (0 until rows.length()).map { index -> rows.getJSONObject(index) }.first { it.getString("file_name") == name }.getString("sha256")
            }
            assertEquals("frame hash differs: $name", expected, sha256Bytes(bytes))
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: error("cannot decode $name")
        }
        val grayscaleFrames = frames.map(::toGray320)
        val grayscaleBytes = grayscaleFrames.map(::grayBytes)
        val rgbaFrames = frames.map(::rgba640x480)
        try {
            val fullRuns = benchmarkRuns(frames.size) { index -> estimate(frames[index - 1], frames[index]) }
            val grayRuns = benchmarkRuns(grayscaleFrames.size) { index -> estimateGray(grayscaleFrames[index - 1], grayscaleFrames[index]) }
            val lumaCopyRuns = benchmarkStreamingLumaCopies(grayscaleBytes)
            val rgbaSamplingRuns = benchmarkStreamingRgbaSampling(rgbaFrames)
            val output = JSONObject()
                .put("schema", "blindassist_sparse_lk_300_same_device_benchmark_v2")
                .put("created_at_utc", utcNow())
                .put("device_under_test", android.os.Build.MODEL)
                .put("implementation", "OpenCV Android: Shi-Tomasi 300 corners, pyramidal LK, RANSAC homography")
                .put("source", "50 hash-verified SANPO redacted RGB frames")
                .put("receipt_sha256", RECEIPT_SHA256)
                .put("measurement_branches", JSONObject()
                    .put("full_bitmap_to_homography", fullRuns)
                    .put("preprocessed_gray_to_homography", grayRuns)
                    .put("streaming_luma_copy_to_homography", lumaCopyRuns)
                    .put("streaming_rgba_sample_to_homography", rgbaSamplingRuns))
                .put("training_authorized", false)
                .put("production_model_replacement_authorized", false)
                .put("important_limit", "Timing-only Android test APK result; it does not measure alerts, misses, false alerts, clearance, or production runtime integration.")
            val root = File(requireNotNull(targetContext.getExternalFilesDir(null)), "sparse-lk-device-benchmark")
            val directory = File(root, utcNow().replace(":", "").replace("-", ""))
            check(directory.mkdirs()) { "cannot create output directory" }
            val report = File(directory, "report.json")
            report.writeText(output.toString(2), Charsets.UTF_8)
            Log.i(TAG, "SPARSE_LK_DEVICE_REPORT ${report.absolutePath}")
            Log.i(TAG, "SPARSE_LK_DEVICE_JSON ${output}")
        } finally {
            frames.forEach(Bitmap::recycle)
            grayscaleFrames.forEach(Mat::release)
        }
    }

    private fun estimate(previousBitmap: Bitmap, currentBitmap: Bitmap): Boolean {
        val previous = toGray320(previousBitmap)
        val current = toGray320(currentBitmap)
        return try {
            estimateGray(previous, current)
        } finally {
            previous.release(); current.release()
        }
    }

    private fun toGray320(bitmap: Bitmap): Mat {
        val rgba = Mat(); val gray = Mat()
        try {
            Utils.bitmapToMat(bitmap, rgba)
            Imgproc.resize(rgba, rgba, Size(SIZE.toDouble(), SIZE.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            return gray
        } finally {
            rgba.release()
        }
    }

    private fun estimateGray(previous: Mat, current: Mat): Boolean {
        return geometry.measure(previous, current).success
    }

    private fun grayBytes(gray: Mat): ByteArray = ByteArray(SIZE * SIZE).also { gray.get(0, 0, it) }

    private fun lumaMat(bytes: ByteArray): Mat = Mat(SIZE, SIZE, CvType.CV_8UC1).also { it.put(0, 0, bytes) }

    /** Deterministic stand-in for CameraX RGBA_8888, created outside the timer. */
    private fun rgba640x480(bitmap: Bitmap): ByteArray {
        val resized = Bitmap.createScaledBitmap(bitmap, CAMERA_WIDTH, CAMERA_HEIGHT, true)
        return try {
            val pixels = IntArray(CAMERA_WIDTH * CAMERA_HEIGHT)
            resized.getPixels(pixels, 0, CAMERA_WIDTH, 0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)
            ByteArray(CAMERA_WIDTH * CAMERA_HEIGHT * 4).also { bytes ->
                pixels.forEachIndexed { index, pixel ->
                    val offset = index * 4
                    bytes[offset] = ((pixel shr 16) and 0xFF).toByte()
                    bytes[offset + 1] = ((pixel shr 8) and 0xFF).toByte()
                    bytes[offset + 2] = (pixel and 0xFF).toByte()
                    bytes[offset + 3] = ((pixel ushr 24) and 0xFF).toByte()
                }
            }
        } finally {
            if (resized !== bitmap) resized.recycle()
        }
    }

    private fun sampledLumaFromRgba(rgba: ByteArray): ByteArray = ByteArray(SIZE * SIZE).also { gray ->
        for (y in 0 until SIZE) {
            val sourceY = y * CAMERA_HEIGHT / SIZE
            for (x in 0 until SIZE) {
                val sourceX = x * CAMERA_WIDTH / SIZE
                val offset = (sourceY * CAMERA_WIDTH + sourceX) * 4
                val red = rgba[offset].toInt() and 0xFF
                val green = rgba[offset + 1].toInt() and 0xFF
                val blue = rgba[offset + 2].toInt() and 0xFF
                gray[y * SIZE + x] = ((77 * red + 150 * green + 29 * blue) ushr 8).toByte()
            }
        }
    }

    /** One current-frame luma copy per transition, retaining the previous gray frame. */
    private fun benchmarkStreamingLumaCopies(bytes: List<ByteArray>): JSONArray {
        val runs = JSONArray()
        repeat(RUNS) { runIndex ->
            var warmupPrevious = lumaMat(bytes[0])
            try {
                repeat(WARMUP_TRANSITIONS) { warmupIndex ->
                    val current = lumaMat(bytes[warmupIndex + 1])
                    estimateGray(warmupPrevious, current)
                    warmupPrevious.release()
                    warmupPrevious = current
                }
            } finally {
                warmupPrevious.release()
            }
            var previous = lumaMat(bytes[0])
            try {
                val samplesNs = LongArray(bytes.size - 1)
                var successes = 0
                for (index in 1 until bytes.size) {
                    val started = SystemClock.elapsedRealtimeNanos()
                    val current = lumaMat(bytes[index])
                    if (estimateGray(previous, current)) successes++
                    samplesNs[index - 1] = SystemClock.elapsedRealtimeNanos() - started
                    previous.release()
                    previous = current
                }
                runs.put(JSONObject().put("run_index", runIndex + 1).put("transition_count", samplesNs.size).put("homography_success_rate", successes.toDouble() / samplesNs.size).put("p50_ms", percentileMs(samplesNs, 50.0)).put("p95_ms", percentileMs(samplesNs, 95.0)))
            } finally {
                previous.release()
            }
        }
        return runs
    }

    /** Current CameraX-compatible source shape: direct RGBA sampling, one retained previous gray frame. */
    private fun benchmarkStreamingRgbaSampling(frames: List<ByteArray>): JSONArray {
        val runs = JSONArray()
        repeat(RUNS) { runIndex ->
            var warmupPrevious = lumaMat(sampledLumaFromRgba(frames[0]))
            try {
                repeat(WARMUP_TRANSITIONS) { warmupIndex ->
                    val current = lumaMat(sampledLumaFromRgba(frames[warmupIndex + 1]))
                    estimateGray(warmupPrevious, current)
                    warmupPrevious.release()
                    warmupPrevious = current
                }
            } finally {
                warmupPrevious.release()
            }
            var previous = lumaMat(sampledLumaFromRgba(frames[0]))
            try {
                val samplesNs = LongArray(frames.size - 1)
                var successes = 0
                for (index in 1 until frames.size) {
                    val started = SystemClock.elapsedRealtimeNanos()
                    val current = lumaMat(sampledLumaFromRgba(frames[index]))
                    if (estimateGray(previous, current)) successes++
                    samplesNs[index - 1] = SystemClock.elapsedRealtimeNanos() - started
                    previous.release()
                    previous = current
                }
                runs.put(JSONObject().put("run_index", runIndex + 1).put("transition_count", samplesNs.size).put("homography_success_rate", successes.toDouble() / samplesNs.size).put("p50_ms", percentileMs(samplesNs, 50.0)).put("p95_ms", percentileMs(samplesNs, 95.0)))
            } finally {
                previous.release()
            }
        }
        return runs
    }

    private fun benchmarkRuns(frameCount: Int, estimate: (Int) -> Boolean): JSONArray {
        val runs = JSONArray()
        repeat(RUNS) { runIndex ->
            repeat(WARMUP_TRANSITIONS) { warmupIndex -> estimate(warmupIndex + 1) }
            val samplesNs = LongArray(frameCount - 1)
            var successes = 0
            for (index in 1 until frameCount) {
                val started = SystemClock.elapsedRealtimeNanos()
                if (estimate(index)) successes++
                samplesNs[index - 1] = SystemClock.elapsedRealtimeNanos() - started
            }
            runs.put(JSONObject().put("run_index", runIndex + 1).put("transition_count", samplesNs.size).put("homography_success_rate", successes.toDouble() / samplesNs.size).put("p50_ms", percentileMs(samplesNs, 50.0)).put("p95_ms", percentileMs(samplesNs, 95.0)))
        }
        return runs
    }

    private fun readAssetBytes(path: String): ByteArray = testContext.assets.open(path).use { it.readBytes() }
    private fun sha256Bytes(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    private fun percentileMs(values: LongArray, percent: Double): Double = values.sortedArray()[kotlin.math.ceil(values.size * percent / 100.0).toInt() - 1].toDouble() / 1_000_000.0
    private fun utcNow(): String = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).apply { timeZone = TimeZone.getTimeZone("UTC") }.format(Date())

    private companion object {
        const val ASSET_ROOT = "sparse_lk_sanpo"
        const val RECEIPT_SHA256 = "d99b794c235e0f24a0656fe8da3f1719b283a795571e6de323e03997a9501129"
        const val SIZE = 320
        const val MAX_CORNERS = 300
        const val CAMERA_WIDTH = 640
        const val CAMERA_HEIGHT = 480
        const val RUNS = 3
        const val WARMUP_TRANSITIONS = 5
        const val TAG = "SparseLkDeviceBenchmark"
    }
}
