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
class Dav2NativeGeometryParityDeviceTest {
    @Test
    fun nativeMatchesFrozenKotlinGeometry() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val repetitions = arguments.getString("repetitions")?.toInt() ?: 100
        require(repetitions >= 100)
        val raw = readRawFloat(File(requireNotNull(arguments.getString("depthPath"))))
        assertEquals(Dav2PreprocessContract.PLANE, raw.size)
        val realDepth = resizeDepthAlignCorners(raw)
        val cases = listOf(
            "real_dav2" to realDepth,
            "scaled_0_75" to realDepth.mapValues { it * 0.75f },
            "scaled_1_25" to realDepth.mapValues { it * 1.25f },
            "periodic_nan_7" to realDepth.mapIndexedValues { index, value ->
                if (index % 7 == 0) Float.NaN else value
            },
            "roi_stripes_zero" to realDepth.mapIndexedValues { index, value ->
                if ((index / WIDTH) >= 264 && (index / WIDTH) % 12 == 0) 0.0f else value
            },
            "deterministic_small_noise" to realDepth.mapIndexedValues { index, value ->
                value + (kotlin.math.sin(index * 0.017) * 0.005).toFloat()
            },
            "all_zero" to FloatArray(WIDTH * HEIGHT),
            "too_few_valid" to FloatArray(WIDTH * HEIGHT).also { depth ->
                repeat(99) { index -> depth[(HEIGHT - 1) * WIDTH + index * 4] = 1.0f }
            },
        )
        val parityReports = JSONArray()
        var allParity = true
        for ((name, depth) in cases) {
            val report = parity(name, evaluateKotlin(depth), evaluateNative(depth))
            parityReports.put(report)
            allParity = allParity && report.getBoolean("pass")
        }
        assertTrue("native geometry parity failed: $parityReports", allParity)

        repeat(3) { evaluateKotlin(realDepth); evaluateNative(realDepth) }
        val kotlinBench = benchmark(repetitions) { evaluateKotlin(realDepth) }
        val nativeBench = benchmark(repetitions) { evaluateNative(realDepth) }
        val speedup = kotlinBench.getJSONObject("wall_ms").getDouble("p50") /
            nativeBench.getJSONObject("wall_ms").getDouble("p50")
        val report = JSONObject()
            .put("schema", "blindassist_dav2_native_geometry_parity_r0")
            .put("contract", JSONObject()
                .put("canonical", "KnownHeightGroundPipeline.evaluateGeometry")
                .put("sampling", "frozen lower_roi=0.55 stride=4 cap=5000")
                .put("ransac", "java.util.Random-compatible seed=1729 iterations=240")
                .put("thresholds", "frozen; no threshold or early-stop changes")
                .put("maximum_absolute_field_error", MAX_ABSOLUTE_ERROR))
            .put("cases", parityReports)
            .put("repetitions", repetitions)
            .put("kotlin", kotlinBench)
            .put("native", nativeBench)
            .put("p50_speedup", speedup)
            .put("gate_pass", allParity)
        File(instrumentation.targetContext.filesDir, REPORT_FILE).writeText(report.toString(2))
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
        assertTrue(report.toString(), report.getBoolean("gate_pass"))
    }

    private fun evaluateKotlin(depth: FloatArray): Any = KnownHeightGroundPipeline.evaluateGeometry(
        depth, WIDTH, HEIGHT, FX, FY, CX, CY, CAMERA_HEIGHT_M,
    )

    private fun evaluateNative(depth: FloatArray): Any = Dav2NativeGeometry.evaluate(
        depth, WIDTH, HEIGHT, FX, FY, CX, CY, CAMERA_HEIGHT_M,
    )

    private fun parity(name: String, expected: Any, actual: Any): JSONObject {
        val report = JSONObject().put("case", name)
            .put("kotlin_type", expected.javaClass.simpleName)
            .put("native_type", actual.javaClass.simpleName)
        if (expected is KnownHeightGroundPipeline.Unknown && actual is KnownHeightGroundPipeline.Unknown) {
            return report.put("kotlin_reason", expected.reason).put("native_reason", actual.reason)
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
        return report.put("maximum_absolute_field_error", maximum).put("pass", maximum <= MAX_ABSOLUTE_ERROR)
    }

    private fun benchmark(repetitions: Int, block: () -> Unit): JSONObject {
        val wall = DoubleArray(repetitions)
        val cpu = DoubleArray(repetitions)
        val allocatedBefore = stat("art.gc.bytes-allocated")
        repeat(repetitions) { index ->
            val wallStart = SystemClock.elapsedRealtimeNanos()
            val cpuStart = Debug.threadCpuTimeNanos()
            block()
            wall[index] = (SystemClock.elapsedRealtimeNanos() - wallStart) / 1_000_000.0
            cpu[index] = (Debug.threadCpuTimeNanos() - cpuStart) / 1_000_000.0
        }
        val allocatedAfter = stat("art.gc.bytes-allocated")
        return JSONObject().put("wall_ms", summary(wall)).put("thread_cpu_ms", summary(cpu))
            .put("allocated_bytes_total", if (allocatedBefore >= 0 && allocatedAfter >= allocatedBefore) allocatedAfter - allocatedBefore else JSONObject.NULL)
    }

    private fun summary(values: DoubleArray): JSONObject {
        val sorted = values.sortedArray()
        fun percentile(q: Double): Double {
            val position = q * (sorted.size - 1)
            val lower = position.toInt()
            val upper = minOf(lower + 1, sorted.lastIndex)
            return sorted[lower] * (1 - position + lower) + sorted[upper] * (position - lower)
        }
        return JSONObject().put("p50", percentile(.5)).put("p95", percentile(.95))
            .put("maximum", sorted.last()).put("mean", values.average())
    }

    private fun resizeDepthAlignCorners(input: FloatArray): FloatArray {
        val output = FloatArray(WIDTH * HEIGHT)
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
        return output
    }

    private fun readRawFloat(file: File): FloatArray {
        assertTrue(file.isFile)
        val bytes = file.readBytes()
        return FloatArray(bytes.size / 4).also { ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
    }

    private inline fun FloatArray.mapValues(transform: (Float) -> Float) =
        FloatArray(size) { index -> transform(this[index]) }

    private inline fun FloatArray.mapIndexedValues(transform: (Int, Float) -> Float) =
        FloatArray(size) { index -> transform(index, this[index]) }

    private fun stat(name: String) = Debug.getRuntimeStat(name)?.toLongOrNull() ?: -1L

    private companion object {
        const val WIDTH = 640
        const val HEIGHT = 480
        const val FX = 320.0
        const val FY = 320.0
        const val CX = 320.0
        const val CY = 240.0
        const val CAMERA_HEIGHT_M = 1.0341161949454936
        const val MAX_ABSOLUTE_ERROR = 1e-10
        const val REPORT_KEY = "dav2_native_geometry_parity_r0_report"
        const val REPORT_FILE = "dav2-native-geometry-parity-r0.json"
    }
}
