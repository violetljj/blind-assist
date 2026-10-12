package com.linnan.blindassist.vision

import android.app.Application
import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
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
