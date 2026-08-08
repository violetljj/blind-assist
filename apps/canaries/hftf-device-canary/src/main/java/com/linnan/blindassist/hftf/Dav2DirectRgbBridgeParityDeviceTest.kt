package com.linnan.blindassist.hftf

import android.os.Bundle
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class Dav2DirectRgbBridgeParityDeviceTest {
    @Test
    fun directRgbAndCanonicalTensorAreBitExactForEveryRotation() {
        var rgbMismatches = 0
        var tensorMismatches = 0
        var firstRgbMismatch = -1
        var firstTensorMismatch = -1
        val frame = OwnedYuv420Frame(WIDTH, HEIGHT) {}.lease().apply {
            width = WIDTH
            height = HEIGHT
            for (index in y.indices) y[index] = ((index * 17 + index / WIDTH * 13) and 0xff).toByte()
            for (index in u.indices) u[index] = ((index * 29 + 71) and 0xff).toByte()
            for (index in v.indices) v[index] = ((index * 31 + 19) and 0xff).toByte()
        }

        Dav2Yuv420RgbConverter().use { converter ->
            Dav2NativePreprocessor().use { preprocessor ->
                for (rotation in intArrayOf(0, 90, 180, 270)) {
                    frame.rotationDegrees = rotation
                    val legacyRgb = converter.convert(frame).copyOf()
                    val directRgb = converter.convertDirect(frame)
                    for (index in legacyRgb.indices) {
                        if (legacyRgb[index] != directRgb.get(index)) {
                            rgbMismatches++
                            if (firstRgbMismatch < 0) firstRgbMismatch = rotation * 1_000_000 + index
                        }
                    }

                    val legacyTensorBuffer = preprocessor.preprocessFp16CanonicalStrict(legacyRgb)
                    val legacyTensor = ShortArray(Dav2PreprocessContract.OUTPUT_ELEMENTS)
                    legacyTensorBuffer.asShortBuffer().get(legacyTensor)
                    val directTensor = preprocessor.preprocessFp16CanonicalStrictDirect(directRgb).asShortBuffer()
                    for (index in legacyTensor.indices) {
                        if (legacyTensor[index] != directTensor.get(index)) {
                            tensorMismatches++
                            if (firstTensorMismatch < 0) firstTensorMismatch = rotation * 2_000_000 + index
                        }
                    }
                }

                repeat(5) {
                    preprocessor.preprocessFp16CanonicalStrict(converter.convert(frame))
                    preprocessor.preprocessFp16CanonicalStrictDirect(converter.convertDirect(frame))
                }
                val legacyMs = ArrayList<Double>(REPETITIONS)
                val directMs = ArrayList<Double>(REPETITIONS)
                repeat(REPETITIONS) { iteration ->
                    fun legacy() {
                        val started = SystemClock.elapsedRealtimeNanos()
                        preprocessor.preprocessFp16CanonicalStrict(converter.convert(frame))
                        legacyMs += (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0
                    }
                    fun direct() {
                        val started = SystemClock.elapsedRealtimeNanos()
                        preprocessor.preprocessFp16CanonicalStrictDirect(converter.convertDirect(frame))
                        directMs += (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0
                    }
                    if (iteration and 1 == 0) { legacy(); direct() } else { direct(); legacy() }
                }
                benchmarkReport = JSONObject()
                    .put("repetitions_per_arm", REPETITIONS)
                    .put("legacy_ms", latencyJson(legacyMs))
                    .put("direct_ms", latencyJson(directMs))
            }
        }
        frame.close()
        val report = JSONObject()
            .put("schema", "blindassist_dav2_direct_rgb_bridge_parity_r0")
            .put("rotations", 4)
            .put("rgb_elements_compared", 4L * Dav2PreprocessContract.INPUT_BYTES)
            .put("tensor_elements_compared", 4L * Dav2PreprocessContract.OUTPUT_ELEMENTS)
            .put("rgb_mismatches", rgbMismatches)
            .put("tensor_mismatches", tensorMismatches)
            .put("first_rgb_mismatch", firstRgbMismatch)
            .put("first_tensor_mismatch", firstTensorMismatch)
            .put("definite_rgb_copy_bytes_eliminated_per_inference", Dav2PreprocessContract.INPUT_BYTES)
            .put("paired_benchmark", benchmarkReport)
            .put("pass", rgbMismatches == 0 && tensorMismatches == 0)
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        File(instrumentation.targetContext.filesDir, REPORT_FILE).writeText(report.toString(2))
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
        assertEquals(report.toString(), 0, rgbMismatches + tensorMismatches)
    }

    private fun latencyJson(values: List<Double>): JSONObject {
        val sorted = values.sorted()
        fun percentile(q: Double): Double {
            val position = q * (sorted.size - 1)
            val lower = position.toInt()
            val upper = minOf(lower + 1, sorted.lastIndex)
            return sorted[lower] * (1 - position + lower) + sorted[upper] * (position - lower)
        }
        return JSONObject().put("p50", percentile(.5)).put("p95", percentile(.95))
            .put("mean", values.average()).put("maximum", sorted.last())
    }

    private companion object {
        const val WIDTH = 640
        const val HEIGHT = 480
        const val REPETITIONS = 100
        const val REPORT_KEY = "dav2_direct_rgb_bridge_parity_r0_report"
        const val REPORT_FILE = "dav2-direct-rgb-bridge-parity-r0.json"
    }

    private lateinit var benchmarkReport: JSONObject
}
