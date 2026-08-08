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
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class GeometryOptimizationDeviceTest {
    @Test
    fun frozenGeometryReferenceVsReusableWorkspace() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val repetitions = arguments.getString("repetitions")?.toInt() ?: 100
        require(repetitions >= 100)
        val raw = readRawFloat(File(requireNotNull(arguments.getString("depthPath"))))
        assertEquals(Dav2PreprocessContract.PLANE, raw.size)
        val depth = resizeDepthAlignCorners(raw)
        val args = GeometryArgs(depth)
        val reference = evaluateReference(args)
        val optimized = evaluateOptimized(args)
        val parity = parity(reference, optimized)
        assertTrue("geometry parity failed: $parity", parity.getBoolean("pass"))
        repeat(3) { evaluateReference(args); evaluateOptimized(args) }
        val referenceBench = benchmark(repetitions) { evaluateReference(args) }
        val optimizedBench = benchmark(repetitions) { evaluateOptimized(args) }
        val speedup = referenceBench.getJSONObject("wall_ms").getDouble("p50") /
            optimizedBench.getJSONObject("wall_ms").getDouble("p50")
        val report = JSONObject()
            .put("schema", "blindassist_geometry_equivalent_optimization_r0")
            .put("repetitions", repetitions)
            .put("contract", JSONObject()
                .put("sampling", "frozen stride=4 and deterministic cap=5000")
                .put("ransac", "frozen java.util.Random seed=1729 iterations=240")
                .put("thresholds_and_early_stop", "unchanged; no early stop")
                .put("optimization", "cached rays + reusable SoA points/masks/residual/depth buffers"))
            .put("parity", parity)
            .put("reference", referenceBench)
            .put("optimized", optimizedBench)
            .put("p50_speedup", speedup)
            .put("gate_pass", parity.getBoolean("pass") && speedup > 1.2)
        File(instrumentation.targetContext.filesDir, REPORT_FILE).writeText(report.toString(2))
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
        assertTrue(report.toString(), report.getBoolean("gate_pass"))
    }

    private fun evaluateReference(args: GeometryArgs): Any = KnownHeightGroundPipeline.evaluateGeometryReference(
        args.depth, 640, 480, 320.0, 320.0, 320.0, 240.0, 1.0341161949454936,
    )

    private fun evaluateOptimized(args: GeometryArgs): Any = KnownHeightGroundPipeline.evaluateGeometry(
        args.depth, 640, 480, 320.0, 320.0, 320.0, 240.0, 1.0341161949454936,
    )

    private fun parity(expected: Any, actual: Any): JSONObject {
        val report = JSONObject().put("reference_type", expected.javaClass.simpleName)
            .put("optimized_type", actual.javaClass.simpleName)
        if (expected is KnownHeightGroundPipeline.Unknown && actual is KnownHeightGroundPipeline.Unknown) {
            return report.put("reference_reason", expected.reason).put("optimized_reason", actual.reason)
                .put("pass", expected.reason == actual.reason)
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
        return report.put("maximum_absolute_field_error", maximum).put("pass", maximum <= 1e-12)
    }

    private fun benchmark(repetitions: Int, block: () -> Unit): JSONObject {
        val wall = DoubleArray(repetitions)
        val cpu = DoubleArray(repetitions)
        val before = RuntimeStats.capture()
        repeat(repetitions) { index ->
            val wallStart = SystemClock.elapsedRealtimeNanos()
            val cpuStart = Debug.threadCpuTimeNanos()
            block()
            wall[index] = (SystemClock.elapsedRealtimeNanos() - wallStart) / 1_000_000.0
            cpu[index] = (Debug.threadCpuTimeNanos() - cpuStart) / 1_000_000.0
        }
        val after = RuntimeStats.capture()
        val allocationAvailable = before.allocated >= 0 && after.allocated >= before.allocated
        val allocated = if (allocationAvailable) after.allocated - before.allocated else -1
        return JSONObject().put("wall_ms", summary(wall)).put("thread_cpu_ms", summary(cpu))
            .put("allocated_bytes_total", if (allocationAvailable) allocated else JSONObject.NULL)
            .put("allocated_bytes_per_iteration", if (allocationAvailable) allocated.toDouble() / repetitions else JSONObject.NULL)
            .put("gc_count_delta", (after.gcCount - before.gcCount).coerceAtLeast(0))
            .put("gc_time_ms_delta", (after.gcTime - before.gcTime).coerceAtLeast(0))
    }

    private fun summary(values: DoubleArray): JSONObject {
        val sorted = values.sortedArray()
        fun percentile(q: Double): Double {
            val position = q * (sorted.size - 1); val lower = position.toInt(); val upper = minOf(lower + 1, sorted.lastIndex)
            return sorted[lower] * (1 - position + lower) + sorted[upper] * (position - lower)
        }
        return JSONObject().put("p50", percentile(.5)).put("p95", percentile(.95))
            .put("maximum", sorted.last()).put("mean", values.average())
    }

    private fun resizeDepthAlignCorners(input: FloatArray): FloatArray {
        val output = FloatArray(640 * 480)
        for (row in 0 until 480) {
            val sy = row.toDouble() * (Dav2PreprocessContract.OUTPUT_HEIGHT - 1) / 479
            val y0 = sy.toInt(); val y1 = minOf(y0 + 1, Dav2PreprocessContract.OUTPUT_HEIGHT - 1); val fy = sy - y0
            for (column in 0 until 640) {
                val sx = column.toDouble() * (Dav2PreprocessContract.OUTPUT_WIDTH - 1) / 639
                val x0 = sx.toInt(); val x1 = minOf(x0 + 1, Dav2PreprocessContract.OUTPUT_WIDTH - 1); val fx = sx - x0
                val top = input[y0 * Dav2PreprocessContract.OUTPUT_WIDTH + x0] * (1 - fx) + input[y0 * Dav2PreprocessContract.OUTPUT_WIDTH + x1] * fx
                val bottom = input[y1 * Dav2PreprocessContract.OUTPUT_WIDTH + x0] * (1 - fx) + input[y1 * Dav2PreprocessContract.OUTPUT_WIDTH + x1] * fx
                output[row * 640 + column] = (top * (1 - fy) + bottom * fy).toFloat()
            }
        }
        return output
    }

    private fun readRawFloat(file: File): FloatArray {
        assertTrue(file.isFile)
        val bytes = file.readBytes()
        return FloatArray(bytes.size / 4).also { ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
    }

    private data class GeometryArgs(val depth: FloatArray)
    private data class RuntimeStats(val allocated: Long, val gcCount: Long, val gcTime: Long) {
        companion object {
            fun capture() = RuntimeStats(stat("art.gc.bytes-allocated"), stat("art.gc.gc-count"), stat("art.gc.gc-time"))
            private fun stat(name: String) = Debug.getRuntimeStat(name)?.toLongOrNull() ?: -1L
        }
    }
    private companion object {
        const val REPORT_KEY = "geometry_equivalent_optimization_r0_report"
        const val REPORT_FILE = "geometry-equivalent-optimization-r0.json"
    }
}
