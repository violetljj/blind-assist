package com.linnan.blindassist.vision

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Existing Development frame only; model files are supplied explicitly, never downloaded. */
@RunWith(AndroidJUnit4::class)
class Dav2SnpeDepthEstimatorDeviceTest {
    private val previousMean = floatArrayOf(.485f, .456f, .406f)
    private val previousStd = floatArrayOf(.229f, .224f, .225f)

    @Test
    fun cachedPreprocessingMatchesPreviousFloatOrder() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val pixels = IntArray(480 * 640) { i ->
            val x = i % 480
            val y = i / 480
            // Deterministic texture includes edges, alternating extremes and gradients.
            val r = (x * 37 + y * 13) and 255
            val g = if ((x + y) % 2 == 0) 0 else 255
            val b = (x xor (y * 19)) and 255
            (255 shl 24) or (r shl 16) or (g shl 8) or b
        }
        val bitmap = Bitmap.createBitmap(pixels, 480, 640, Bitmap.Config.ARGB_8888)
        try {
            val previous = previousPreprocessNchw(bitmap)
            val cached = Dav2SnpeDepthEstimator.preprocessNchw(bitmap)
            var different = previous.indices.count { previous[it].toRawBits() != cached[it].toRawBits() }
            var compared = previous.size
            for (color in intArrayOf(0xff000000.toInt(), 0xffffffff.toInt())) {
                val flat = Bitmap.createBitmap(IntArray(pixels.size) { color }, 480, 640, Bitmap.Config.ARGB_8888)
                try {
                    val flatPrevious = previousPreprocessNchw(flat)
                    val flatCached = Dav2SnpeDepthEstimator.preprocessNchw(flat)
                    different += flatPrevious.indices.count { flatPrevious[it].toRawBits() != flatCached[it].toRawBits() }
                    compared += flatPrevious.size
                } finally { flat.recycle() }
            }
            val report = JSONObject().put("evidence", "synthetic Android preprocessing engineering check")
                .put("implementation", "cached_taps_native_scalar_no_fast_math_no_fma")
                .put("timing_scope", "Bitmap getPixels and NCHW preprocessing only; excludes camera, HTP and display")
                .put("compared_floats", compared).put("different_raw_bits", different)
                .put("warmup_pairs", 6).put("measured_pairs", 7)
            repeat(6) {
                previousPreprocessNchw(bitmap)
                Dav2SnpeDepthEstimator.preprocessNchw(bitmap)
            }
            val previousMs = ArrayList<Double>()
            val cachedMs = ArrayList<Double>()
            var checksum = 0f
            fun measure(cachedRun: Boolean) {
                val start = SystemClock.elapsedRealtimeNanos()
                val output = if (cachedRun) Dav2SnpeDepthEstimator.preprocessNchw(bitmap)
                    else previousPreprocessNchw(bitmap)
                val ms = (SystemClock.elapsedRealtimeNanos() - start) / 1e6
                (if (cachedRun) cachedMs else previousMs).add(ms)
                checksum += output[output.size / 2]
            }
            repeat(7) { i ->
                measure(i % 2 == 0)
                measure(i % 2 != 0)
            }
            report.put("previous_ms", JSONArray(previousMs)).put("cached_ms", JSONArray(cachedMs))
                .put("previous_median_ms", previousMs.sorted()[3])
                .put("cached_median_ms", cachedMs.sorted()[3]).put("checksum", checksum)
            val directory = File(context.noBackupFilesDir, "fusion-experiment").apply { mkdirs() }
            File(directory, "preprocess-cache-${System.currentTimeMillis()}.json").writeText(report.toString(2))
            assertTrue("Cached interpolation changed $different float values", different == 0)
        } finally {
            bitmap.recycle()
        }
    }

    @Test
    fun preprocessingMatchesOfficialCubicAndAlignCorners() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(InstrumentationRegistry.getArguments().getString("dav2FixtureRoot")
            ?: File(context.filesDir, "dav2-smoke").absolutePath)
        val bitmap = requireNotNull(BitmapFactory.decodeFile(File(directory, "synthetic_upright.png").absolutePath))
        try {
            val actual = Dav2SnpeDepthEstimator.preprocessNchw(bitmap)
            val reference = readRaw(File(directory, "synthetic_nchw.raw"))
            check(actual.size == reference.size)
            val maxError = actual.indices.maxOf { abs(actual[it] - reference[it]) }
            assertTrue("OpenCV float cubic preprocessing max error=$maxError", maxError < 2e-4f)
        } finally {
            bitmap.recycle()
        }
        val native = readRaw(File(directory, "synthetic_disparity.raw"))
        val actual = Dav2SnpeDepthEstimator.resizeAlignCorners(native, 480, 640)
        val reference = readRaw(File(directory, "synthetic_resized.raw"))
        val maxError = actual.indices.maxOf { abs(actual[it] - reference[it]) }
        assertTrue("PyTorch align_corners resize max error=$maxError", maxError < 2e-4f)
    }

    @Test
    fun existingRawFrameMatchesSnpeHtpReference() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val directory = File(InstrumentationRegistry.getArguments().getString("dav2FixtureRoot")
            ?: File(context.filesDir, "dav2-smoke").absolutePath)
        val input = readRaw(File(directory, "input_686x518.raw"))
        val reference = readRaw(File(directory, "out_686x518_htp.raw"))
        val report = JSONObject().put("evidence", "existing Development single-frame Android SNPE engineering check")
            .put("timing_scope", "executeNativeInput including tensor copies; excludes bitmap preprocessing, anchoring and notifications")
        val runs = JSONArray()
        val started = SystemClock.elapsedRealtimeNanos()
        Dav2SnpeDepthEstimator(context.applicationContext as Application,
            File(directory, "dav2s_686x518_fp32.dlc")).use { estimator ->
            report.put("initialization_ms", (SystemClock.elapsedRealtimeNanos() - started) / 1e6)
                .put("input_shape", JSONArray(estimator.inputShape.toList()))
                .put("runtime", estimator.runtime.name)
            assertTrue("Expected explicit HTP/DSP runtime", estimator.runtime.name == "DSP")
            repeat(3) {
                val start = SystemClock.elapsedRealtimeNanos()
                val actual = estimator.executeNativeInput(input)
                val ms = (SystemClock.elapsedRealtimeNanos() - start) / 1e6
                check(actual.size == reference.size)
                val relative = actual.indices.map { i -> abs(actual[i] - reference[i]) / max(abs(reference[i]), 1e-3f) }.sorted()
                val correlation = correlation(actual, reference)
                val median = relative[relative.size / 2]
                runs.put(JSONObject().put("execute_ms", ms).put("correlation", correlation)
                    .put("median_relative_error", median).put("max_relative_error", relative.last()))
                report.put("runs", runs)
                File(directory, "android_snpe_check.json").writeText(report.toString(2))
                assertTrue("HTP output correlation=$correlation", correlation > .999)
                assertTrue("HTP median relative error=$median", median < .01f)
            }
        }
    }

    private fun readRaw(file: File): FloatArray {
        check(file.isFile) { "Supply existing local DAV2 fixture in App files/dav2-smoke: $file" }
        val bytes = file.readBytes()
        check(bytes.size % 4 == 0)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also(buffer::get)
    }

    /** Frozen pre-cache loop: a numerical and timing reference, never used by the app. */
    private fun previousPreprocessNchw(bitmap: Bitmap): FloatArray {
        val width = Dav2SnpeDepthEstimator.WIDTH
        val height = Dav2SnpeDepthEstimator.HEIGHT
        val src = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(src, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val result = FloatArray(width * height * 3)
        fun weights(x: Float): FloatArray {
            val a = -.75f
            val c0 = ((a * (x + 1) - 5 * a) * (x + 1) + 8 * a) * (x + 1) - 4 * a
            val c1 = ((a + 2) * x - (a + 3)) * x * x + 1
            val t = 1 - x
            val c2 = ((a + 2) * t - (a + 3)) * t * t + 1
            return floatArrayOf(c0, c1, c2, 1 - c0 - c1 - c2)
        }
        for (y in 0 until height) {
            val sy = (y + .5) * bitmap.height / height - .5
            val iy = floor(sy).toInt()
            val wy = weights((sy - iy).toFloat())
            for (x in 0 until width) {
                val sx = (x + .5) * bitmap.width / width - .5
                val ix = floor(sx).toInt()
                val wx = weights((sx - ix).toFloat())
                for (c in 0..2) {
                    var value = 0f
                    for (dy in 0..3) {
                        var row = 0f
                        val offset = (iy + dy - 1).coerceIn(0, bitmap.height - 1) * bitmap.width
                        for (dx in 0..3) {
                            val pixel = src[offset + (ix + dx - 1).coerceIn(0, bitmap.width - 1)]
                            row += ((pixel ushr (16 - 8 * c)) and 255) / 255f * wx[dx]
                        }
                        value += row * wy[dy]
                    }
                    result[c * width * height + y * width + x] = (value - previousMean[c]) / previousStd[c]
                }
            }
        }
        return result
    }

    private fun correlation(a: FloatArray, b: FloatArray): Double {
        val ma = a.average(); val mb = b.average()
        var aa = 0.0; var bb = 0.0; var ab = 0.0
        for (i in a.indices) {
            val da = a[i] - ma; val db = b[i] - mb
            aa += da * da; bb += db * db; ab += da * db
        }
        return ab / sqrt(aa * bb)
    }
}
