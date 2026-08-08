package com.linnan.blindassist.hftf

import android.os.Bundle
import android.os.Debug
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.hftf.metricdepth.KnownHeightGroundPipeline
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class Dav2DirectDepthBridgeParityDeviceTest {
    @Test
    fun fusedDecodeResizeAndDirectGeometryMatchStagedPath() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val repetitions = arguments.getString("repetitions")?.toInt() ?: 100
        require(repetitions >= 100)
        val realBytes = File(requireNotNull(arguments.getString("depthFp16Path"))).readBytes()
        assertEquals(Dav2PreprocessContract.PLANE * 2, realBytes.size)
        val cases = listOf(
            "real_qnn_fp16" to directInput(realBytes),
            "all_half_patterns_tiled" to allHalfPatternsTiled(),
        )
        val reports = JSONArray()
        var allPass = true
        Dav2NativePreprocessor().use { preprocessor ->
            for ((name, input) in cases) {
                val report = parityCase(name, input, preprocessor)
                reports.put(report)
                allPass = allPass && report.getBoolean("pass")
            }
            assertTrue("direct depth bridge parity failed: $reports", allPass)
            val realInput = cases.first().second
            val raw = FloatArray(Dav2PreprocessContract.PLANE)
            val aligned = FloatArray(ALIGNED_ELEMENTS)
            val direct = alignedDirectBuffer()
            repeat(3) {
                staged(realInput, raw, aligned, preprocessor)
                direct(realInput, direct, preprocessor)
            }
            val stagedBench = benchmark(repetitions) { staged(realInput, raw, aligned, preprocessor) }
            val directBench = benchmark(repetitions) { direct(realInput, direct, preprocessor) }
            val report = JSONObject()
                .put("schema", "blindassist_dav2_direct_depth_bridge_parity_r0")
                .put("contract", JSONObject()
                    .put("source", "QNN FP16 518x686 direct output semantics")
                    .put("decode", "Android Half raw-bit compatible")
                    .put("resize", "bilinear align_corners 518x686 to 480x640")
                    .put("geometry", "frozen native parity-gated geometry")
                    .put("java_raw_depth_bytes_removed", Dav2PreprocessContract.PLANE * 4)
                    .put("aligned_depth", "owned native direct buffer remains materialized for pipeline handoff"))
                .put("cases", reports)
                .put("repetitions", repetitions)
                .put("staged", stagedBench)
                .put("direct", directBench)
                .put("gate_pass", allPass)
            File(instrumentation.targetContext.filesDir, REPORT_FILE).writeText(report.toString(2))
            instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
            assertTrue(report.toString(), report.getBoolean("gate_pass"))
        }
    }

    private fun parityCase(name: String, input: ByteBuffer, preprocessor: Dav2NativePreprocessor): JSONObject {
        val raw = FloatArray(Dav2PreprocessContract.PLANE)
        val expected = FloatArray(ALIGNED_ELEMENTS)
        val actual = alignedDirectBuffer()
        preprocessor.decodeFp16ToFloatStrict(input, raw)
        resizeDepthAlignCorners(raw, expected)
        preprocessor.decodeResizeFp16AlignCornersStrict(input, actual)
        val floats = actual.order(ByteOrder.nativeOrder()).asFloatBuffer()
        var rawBitMismatches = 0
        var nonFiniteClassMismatches = 0
        var maximumFiniteError = 0.0
        for (index in expected.indices) {
            val left = expected[index]
            val right = floats.get(index)
            if (left.isFinite() && right.isFinite()) {
                if (left.toRawBits() != right.toRawBits()) rawBitMismatches++
                maximumFiniteError = maxOf(maximumFiniteError, kotlin.math.abs(left.toDouble() - right))
            } else if (nonFiniteClass(left) != nonFiniteClass(right)) {
                nonFiniteClassMismatches++
            }
        }
        val geometryParity = geometryParity(
            Dav2NativeGeometry.evaluate(expected, WIDTH, HEIGHT, FX, FY, CX, CY, CAMERA_HEIGHT_M),
            Dav2NativeGeometry.evaluateDirect(actual, WIDTH, HEIGHT, FX, FY, CX, CY, CAMERA_HEIGHT_M),
        )
        val pass = rawBitMismatches == 0 && nonFiniteClassMismatches == 0 && geometryParity.getBoolean("pass")
        return JSONObject().put("case", name).put("aligned_elements", ALIGNED_ELEMENTS)
            .put("finite_raw_bit_mismatches", rawBitMismatches)
            .put("nonfinite_class_mismatches", nonFiniteClassMismatches)
            .put("maximum_finite_absolute_error", maximumFiniteError)
            .put("geometry", geometryParity).put("pass", pass)
    }

    private fun geometryParity(expected: Any, actual: Any): JSONObject {
        val report = JSONObject().put("staged_type", expected.javaClass.simpleName)
            .put("direct_type", actual.javaClass.simpleName)
        if (expected is KnownHeightGroundPipeline.Unknown && actual is KnownHeightGroundPipeline.Unknown) {
            return report.put("staged_reason", expected.reason).put("direct_reason", actual.reason)
                .put("maximum_absolute_field_error", 0.0).put("pass", expected.reason == actual.reason)
        }
        if (expected !is KnownHeightGroundPipeline.Geometry || actual !is KnownHeightGroundPipeline.Geometry) {
            return report.put("pass", false)
        }
        var maximum = 0.0
        fun observe(left: Double, right: Double) { maximum = maxOf(maximum, kotlin.math.abs(left - right)) }
        observe(expected.relativeHeight, actual.relativeHeight)
        observe(expected.normalizedMedianResidual, actual.normalizedMedianResidual)
        observe(expected.inlierFraction, actual.inlierFraction)
        expected.normal.indices.forEach { observe(expected.normal[it], actual.normal[it]) }
        expected.features.indices.forEach { observe(expected.features[it], actual.features[it]) }
        return report.put("maximum_absolute_field_error", maximum).put("pass", maximum <= 1e-10)
    }

    private fun staged(input: ByteBuffer, raw: FloatArray, aligned: FloatArray, preprocessor: Dav2NativePreprocessor): Any {
        preprocessor.decodeFp16ToFloatStrict(input, raw)
        resizeDepthAlignCorners(raw, aligned)
        return Dav2NativeGeometry.evaluate(aligned, WIDTH, HEIGHT, FX, FY, CX, CY, CAMERA_HEIGHT_M)
    }

    private fun direct(input: ByteBuffer, output: ByteBuffer, preprocessor: Dav2NativePreprocessor): Any {
        preprocessor.decodeResizeFp16AlignCornersStrict(input, output)
        return Dav2NativeGeometry.evaluateDirect(output, WIDTH, HEIGHT, FX, FY, CX, CY, CAMERA_HEIGHT_M)
    }

    private fun benchmark(repetitions: Int, block: () -> Unit): JSONObject {
        val wall = DoubleArray(repetitions)
        val cpu = DoubleArray(repetitions)
        val allocatedBefore = Debug.getRuntimeStat("art.gc.bytes-allocated")?.toLongOrNull() ?: -1
        repeat(repetitions) { index ->
            val start = SystemClock.elapsedRealtimeNanos(); val cpuStart = Debug.threadCpuTimeNanos()
            block(); wall[index] = (SystemClock.elapsedRealtimeNanos() - start) / 1e6
            cpu[index] = (Debug.threadCpuTimeNanos() - cpuStart) / 1e6
        }
        val allocatedAfter = Debug.getRuntimeStat("art.gc.bytes-allocated")?.toLongOrNull() ?: -1
        return JSONObject().put("wall_ms", summary(wall)).put("thread_cpu_ms", summary(cpu))
            .put("allocated_bytes_total", if (allocatedBefore >= 0 && allocatedAfter >= allocatedBefore) allocatedAfter - allocatedBefore else JSONObject.NULL)
    }

    private fun summary(values: DoubleArray): JSONObject {
        val sorted = values.sortedArray()
        fun percentile(q: Double): Double {
            val p = q * (sorted.size - 1); val lo = p.toInt(); val hi = minOf(lo + 1, sorted.lastIndex)
            return sorted[lo] * (1 - p + lo) + sorted[hi] * (p - lo)
        }
        return JSONObject().put("p50", percentile(.5)).put("p95", percentile(.95))
            .put("maximum", sorted.last()).put("mean", values.average())
    }

    private fun resizeDepthAlignCorners(input: FloatArray, output: FloatArray) {
        for (row in 0 until HEIGHT) {
            val sy = row.toDouble() * (Dav2PreprocessContract.OUTPUT_HEIGHT - 1) / (HEIGHT - 1)
            val y0 = sy.toInt(); val y1 = minOf(y0 + 1, Dav2PreprocessContract.OUTPUT_HEIGHT - 1); val fy = sy - y0
            for (column in 0 until WIDTH) {
                val sx = column.toDouble() * (Dav2PreprocessContract.OUTPUT_WIDTH - 1) / (WIDTH - 1)
                val x0 = sx.toInt(); val x1 = minOf(x0 + 1, Dav2PreprocessContract.OUTPUT_WIDTH - 1); val fx = sx - x0
                val top = input[y0 * Dav2PreprocessContract.OUTPUT_WIDTH + x0] * (1 - fx) + input[y0 * Dav2PreprocessContract.OUTPUT_WIDTH + x1] * fx
                val bottom = input[y1 * Dav2PreprocessContract.OUTPUT_WIDTH + x0] * (1 - fx) + input[y1 * Dav2PreprocessContract.OUTPUT_WIDTH + x1] * fx
                output[row * WIDTH + column] = (top * (1 - fy) + bottom * fy).toFloat()
            }
        }
    }

    private fun directInput(bytes: ByteArray) = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
        put(bytes); flip()
    }

    private fun allHalfPatternsTiled() = ByteBuffer.allocateDirect(Dav2PreprocessContract.PLANE * 2)
        .order(ByteOrder.nativeOrder()).apply {
            val shorts = asShortBuffer()
            for (index in 0 until Dav2PreprocessContract.PLANE) shorts.put(index, (index and 0xffff).toShort())
            position(0); limit(Dav2PreprocessContract.PLANE * 2)
        }

    private fun alignedDirectBuffer() = ByteBuffer.allocateDirect(ALIGNED_ELEMENTS * 4).order(ByteOrder.nativeOrder())
    private fun nonFiniteClass(value: Float) = when {
        value.isFinite() -> "FINITE"
        value.isNaN() -> "NAN"
        value == Float.POSITIVE_INFINITY -> "+INF"
        else -> "-INF"
    }

    private companion object {
        const val WIDTH = 640; const val HEIGHT = 480; const val ALIGNED_ELEMENTS = WIDTH * HEIGHT
        const val FX = 320.0; const val FY = 320.0; const val CX = 320.0; const val CY = 240.0
        const val CAMERA_HEIGHT_M = 1.0341161949454936
        const val REPORT_KEY = "dav2_direct_depth_bridge_parity_r0_report"
        const val REPORT_FILE = "dav2-direct-depth-bridge-parity-r0.json"
    }
}
