package com.linnan.blindassist.hftf

import android.app.KeyguardManager
import android.os.Bundle
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
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
class Dav2PreprocessOptimizationDeviceTest {
    @Test
    fun strictFloat32ToFloat16MatchesAndroidHalf() {
        val values = ArrayList<Float>(0x7c00 * 3)
        values += listOf(
            0.0f,
            -0.0f,
            Float.POSITIVE_INFINITY,
            Float.NEGATIVE_INFINITY,
            Float.NaN,
            Float.fromBits(0x00000001),
            Float.fromBits(0x80000001.toInt()),
            65504.0f,
            65520.0f,
            -65520.0f,
        )
        // Every positive finite half value plus every exact midpoint exercises
        // normal, subnormal, carry, overflow, and ties-to-even behavior.
        for (bits in 0 until 0x7c00) {
            val lower = halfBitsToFloat(bits.toShort())
            values += lower
            if (bits + 1 < 0x7c00) {
                val upper = halfBitsToFloat((bits + 1).toShort())
                val midpoint = (lower + upper) * 0.5f
                values += midpoint
                values += -midpoint
            }
        }
        val input = ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder())
        val floats = input.asFloatBuffer()
        values.forEachIndexed { index, value -> floats.put(index, value) }
        input.position(0)
        input.limit(values.size * 4)

        Dav2NativePreprocessor().use { native ->
            val actual = native.convertFp32ToFp16Strict(input, values.size).asShortBuffer()
            values.forEachIndexed { index, value ->
                val expected = floatToHalfBits(value).toInt() and 0xffff
                val observed = actual.get(index).toInt() and 0xffff
                if (value.isNaN()) {
                    assertTrue("NaN became non-NaN: 0x${observed.toString(16)}", observed and 0x7c00 == 0x7c00 && observed and 0x03ff != 0)
                } else {
                    assertEquals("half mismatch at index=$index value=$value", expected, observed)
                }
            }
        }
    }

    @Test
    fun cpuBoundaryMicrobench() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val corpus = File(requireNotNull(arguments.getString("corpusRoot")))
        val repetitions = arguments.getString("repetitions")?.toInt() ?: 100
        require(repetitions >= 100)
        val rgb = readNpyBytes(File(corpus, "clean/rgb_640x480_uint8.npy"))
        val official = readNpyFloat(File(corpus, "clean/normalized_nchw_fp32_1x3x518x686.npy"))
        assertEquals(Dav2PreprocessContract.INPUT_BYTES, rgb.size)
        assertEquals(Dav2PreprocessContract.OUTPUT_ELEMENTS, official.size)

        val reference = Dav2ReferencePreprocessor()
        val resized = FloatArray(Dav2PreprocessContract.OUTPUT_ELEMENTS)
        val packed = FloatArray(Dav2PreprocessContract.OUTPUT_ELEMENTS)
        val normalized = FloatArray(Dav2PreprocessContract.OUTPUT_ELEMENTS)
        reference.resizeRgb(rgb, resized)
        reference.rgbToNchw(resized, packed)
        reference.normalize(packed, normalized)

        val table = Dav2KotlinTablePreprocessor()
        repeat(WARMUP_RUNS) { table.preprocess(rgb) }
        val tableValues = floatArray(table.preprocess(rgb))
        val half = ShortArray(Dav2PreprocessContract.OUTPUT_ELEMENTS)

        val report = JSONObject()
            .put("schema", "blindassist_cpu_boundary_microbench_r0")
            .put("contract", "rgb640x480_float255_opencv_inter_cubic_686x518_imagenet_nchw")
            .put("repetitions", repetitions)
            .put("screen_state", screenState())
            .put("memory_before", memoryJson())
            .put("parity", JSONObject())
            .put("stages", JSONObject())

        val referenceParity = parity(normalized, official)
        val tableParity = parity(tableValues, official)
        report.getJSONObject("parity")
            .put("reference_double_vs_official", referenceParity)
            .put("kotlin_float_table_vs_official", tableParity)
        assertTrue("reference preprocess drift: $referenceParity", referenceParity.getDouble("max_abs") <= 2e-4)
        assertTrue("Kotlin table preprocess drift: $tableParity", tableParity.getDouble("max_abs") <= 5e-4)

        val stages = report.getJSONObject("stages")
        stages.put("resize_reference_double", benchmark(repetitions) { reference.resizeRgb(rgb, resized) })
        stages.put("rgb_to_nchw", benchmark(repetitions) { reference.rgbToNchw(resized, packed) })
        stages.put("normalization", benchmark(repetitions) { reference.normalize(packed, normalized) })
        stages.put(
            "float32_to_float16",
            benchmark(repetitions) {
                for (index in normalized.indices) half[index] = floatToHalfBits(normalized[index])
            },
        )
        stages.put(
            "buffer_allocation_and_copy",
            benchmark(repetitions) {
                val copy = normalized.copyOf()
                ByteBuffer.allocateDirect(copy.size * 4).order(ByteOrder.nativeOrder())
                    .asFloatBuffer().put(copy)
            },
        )
        stages.put("kotlin_float_table_fused_reused", benchmark(repetitions) { table.preprocess(rgb) })

        Dav2NativePreprocessor().use { native ->
            repeat(WARMUP_RUNS) { native.preprocessFp32(rgb) }
            val nativeValues = floatArray(native.preprocessFp32(rgb))
            val nativeParity = parity(nativeValues, official)
            report.getJSONObject("parity").put("native_opencv_neon_fp32_vs_official", nativeParity)
            assertTrue("Native OpenCV preprocess drift: $nativeParity", nativeParity.getDouble("max_abs") <= 5e-5)
            val nativeFp16Values = floatArrayFromHalf(native.preprocessFp16Strict(rgb))
            val nativeFp16Parity = parity(nativeFp16Values, official)
            report.getJSONObject("parity").put("native_opencv_neon_fp32_then_strict_fp16_vs_official", nativeFp16Parity)
            assertTrue("Native strict FP16 preprocess drift: $nativeFp16Parity", nativeFp16Parity.getDouble("max_abs") <= 0.002)
            stages.put("native_opencv_neon_fp32_reused", benchmark(repetitions) { native.preprocessFp32(rgb) })
            stages.put("native_opencv_neon_fp32_then_strict_fp16_reused", benchmark(repetitions) {
                native.preprocessFp16Strict(rgb)
            })
            val canonical = floatArray(native.preprocessFp32Canonical(rgb))
            val canonicalParity = parity(canonical, official)
            val canonicalHalf = floatArrayFromHalf(native.preprocessFp16CanonicalStrict(rgb))
            val canonicalHalfParity = parity(canonicalHalf, official)
            report.getJSONObject("parity")
                .put("native_canonical_fp32_vs_official", canonicalParity)
                .put("native_canonical_strict_fp16_vs_official", canonicalHalfParity)
            stages.put("native_canonical_fp32_then_strict_fp16_reused", benchmark(repetitions) {
                native.preprocessFp16CanonicalStrict(rgb)
            })
            stages.put("native_opencv_neon_fp16_fused_control_reused", benchmark(repetitions) {
                native.preprocessFp16Fused(rgb)
            })
        }

        report.put("memory_after", memoryJson())
        report.put("admission", admission(stages))
        val outputName = arguments.getString("outputName") ?: "cpu-boundary-microbench-r0.json"
        File(instrumentation.targetContext.filesDir, outputName).writeText(report.toString(2))
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
    }

    private fun benchmark(repetitions: Int, block: () -> Unit): JSONObject {
        val wall = DoubleArray(repetitions)
        val cpu = DoubleArray(repetitions)
        val processCpu = DoubleArray(repetitions)
        val runtimeBefore = RuntimeStatSnapshot.capture()
        repeat(repetitions) { index ->
            val wallBefore = SystemClock.elapsedRealtimeNanos()
            val cpuBefore = Debug.threadCpuTimeNanos()
            val processCpuBefore = android.os.Process.getElapsedCpuTime()
            block()
            wall[index] = (SystemClock.elapsedRealtimeNanos() - wallBefore) / 1_000_000.0
            cpu[index] = (Debug.threadCpuTimeNanos() - cpuBefore) / 1_000_000.0
            processCpu[index] = (android.os.Process.getElapsedCpuTime() - processCpuBefore).toDouble()
        }
        val runtimeAfter = RuntimeStatSnapshot.capture()
        val allocationAvailable = runtimeBefore.allocatedBytes >= 0 &&
            runtimeAfter.allocatedBytes >= runtimeBefore.allocatedBytes
        val allocatedBytes = if (allocationAvailable) {
            runtimeAfter.allocatedBytes - runtimeBefore.allocatedBytes
        } else {
            -1L
        }
        val gcCount = (runtimeAfter.gcCount - runtimeBefore.gcCount).coerceAtLeast(0)
        val gcTimeMs = (runtimeAfter.gcTimeMs - runtimeBefore.gcTimeMs).coerceAtLeast(0)
        return JSONObject()
            .put("wall_ms", summary(wall))
            .put("thread_cpu_ms", summary(cpu))
            .put("process_cpu_ms", summary(processCpu))
            .put("allocated_bytes_total", if (allocationAvailable) allocatedBytes else JSONObject.NULL)
            .put("allocated_bytes_per_iteration", if (allocationAvailable) allocatedBytes.toDouble() / repetitions else JSONObject.NULL)
            .put("gc_count_delta", gcCount)
            .put("gc_time_ms_delta", gcTimeMs)
    }

    private fun summary(values: DoubleArray): JSONObject {
        val sorted = values.sortedArray()
        return JSONObject()
            .put("p50", percentile(sorted, 0.50))
            .put("p95", percentile(sorted, 0.95))
            .put("maximum", sorted.last())
            .put("mean", values.average())
    }

    private fun percentile(sorted: DoubleArray, quantile: Double): Double {
        val position = quantile * (sorted.size - 1)
        val lower = position.toInt()
        val upper = minOf(lower + 1, sorted.lastIndex)
        return sorted[lower] * (1.0 - (position - lower)) + sorted[upper] * (position - lower)
    }

    private fun parity(actual: FloatArray, expected: FloatArray): JSONObject {
        assertEquals(expected.size, actual.size)
        var maximum = 0.0
        var sum = 0.0
        val errors = DoubleArray(actual.size)
        for (index in actual.indices) {
            assertTrue("non-finite preprocess value at $index", actual[index].isFinite())
            val error = kotlin.math.abs(actual[index].toDouble() - expected[index])
            errors[index] = error
            sum += error
            maximum = maxOf(maximum, error)
        }
        errors.sort()
        return JSONObject()
            .put("elements", errors.size)
            .put("mean_abs", sum / errors.size)
            .put("p95_abs", errors[(0.95 * (errors.size - 1)).toInt()])
            .put("max_abs", maximum)
    }

    private fun admission(stages: JSONObject): JSONObject {
        val kotlinP50 = stages.getJSONObject("kotlin_float_table_fused_reused")
            .getJSONObject("wall_ms").getDouble("p50")
        val nativeP50 = stages.getJSONObject("native_opencv_neon_fp32_reused")
            .getJSONObject("wall_ms").getDouble("p50")
        return JSONObject()
            .put("kotlin_under_100ms", kotlinP50 < 100.0)
            .put("native_under_100ms", nativeP50 < 100.0)
            .put("native_under_60ms", nativeP50 < 60.0)
            .put("native_under_40ms", nativeP50 < 40.0)
            .put("gpu_experiment_needed_by_cpu_gate", nativeP50 >= 100.0)
    }

    private fun screenState(): JSONObject {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val power = context.getSystemService(PowerManager::class.java)
        val keyguard = context.getSystemService(KeyguardManager::class.java)
        return JSONObject()
            .put("interactive", power.isInteractive)
            .put("keyguard_locked", keyguard.isKeyguardLocked)
            .put("thermal_status", power.currentThermalStatus)
    }

    private fun memoryJson(): JSONObject {
        val runtime = Runtime.getRuntime()
        return JSONObject()
            .put("pss_kib", Debug.getPss())
            .put("java_heap_used_bytes", runtime.totalMemory() - runtime.freeMemory())
            .put("native_heap_allocated_bytes", Debug.getNativeHeapAllocatedSize())
    }

    private fun floatArray(buffer: ByteBuffer): FloatArray {
        buffer.rewind()
        return FloatArray(Dav2PreprocessContract.OUTPUT_ELEMENTS).also { buffer.asFloatBuffer().get(it) }
    }

    private fun floatArrayFromHalf(buffer: ByteBuffer): FloatArray {
        buffer.rewind()
        val shorts = buffer.order(ByteOrder.nativeOrder()).asShortBuffer()
        return FloatArray(Dav2PreprocessContract.OUTPUT_ELEMENTS) { halfBitsToFloat(shorts.get(it)) }
    }

    private fun readNpyBytes(file: File): ByteArray {
        val bytes = file.readBytes()
        val offset = npyDataOffset(bytes)
        return bytes.copyOfRange(offset, bytes.size)
    }

    private fun readNpyFloat(file: File): FloatArray {
        val bytes = file.readBytes()
        val offset = npyDataOffset(bytes)
        val buffer = ByteBuffer.wrap(bytes, offset, bytes.size - offset).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also(buffer::get)
    }

    private fun npyDataOffset(bytes: ByteArray): Int {
        assertTrue(bytes.size > 12)
        assertEquals(0x93, bytes[0].toInt() and 0xff)
        val major = bytes[6].toInt() and 0xff
        val headerLength = if (major == 1) {
            (bytes[8].toInt() and 0xff) or ((bytes[9].toInt() and 0xff) shl 8)
        } else {
            ByteBuffer.wrap(bytes, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int
        }
        return if (major == 1) 10 + headerLength else 12 + headerLength
    }

    private data class RuntimeStatSnapshot(
        val processCpuMs: Long,
        val allocatedBytes: Long,
        val gcCount: Long,
        val gcTimeMs: Long,
    ) {
        companion object {
            fun capture() = RuntimeStatSnapshot(
                processCpuMs = android.os.Process.getElapsedCpuTime(),
                allocatedBytes = runtimeStat("art.gc.bytes-allocated"),
                gcCount = runtimeStat("art.gc.gc-count"),
                gcTimeMs = runtimeStat("art.gc.gc-time"),
            )

            private fun runtimeStat(name: String): Long = Debug.getRuntimeStat(name)?.toLongOrNull() ?: -1L
        }
    }

    private companion object {
        const val REPORT_KEY = "cpu_boundary_microbench_r0_report"
        const val WARMUP_RUNS = 5
    }
}
