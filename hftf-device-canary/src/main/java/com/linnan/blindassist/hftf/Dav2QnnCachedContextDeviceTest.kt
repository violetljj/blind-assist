package com.linnan.blindassist.hftf

import android.os.Bundle
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.hftf.metricdepth.KnownHeightGroundPipeline
import com.linnan.blindassist.hftf.metricdepth.KnownHeightScaleStudent
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class Dav2QnnCachedContextDeviceTest {
    @Test
    fun cachedContextExecuteAndFp16Parity() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val cachedDlc = File(requireNotNull(arguments.getString("cachedDlcPath")))
        val corpus = File(requireNotNull(arguments.getString("corpusRoot")))
        val cliOutput = File(requireNotNull(arguments.getString("cliOutputPath")))
        val repetitions = arguments.getString("repetitions")?.toInt() ?: 10
        require(repetitions >= 10)
        assertTrue(cachedDlc.isFile)
        assertTrue(cliOutput.isFile)

        val official = readNpyFloat(File(corpus, "clean/normalized_nchw_fp32_1x3x518x686.npy"))
        val rgb = readNpyBytes(File(corpus, "clean/rgb_640x480_uint8.npy"))
        val officialFp16 = fp16Buffer(official)
        val cliReference = readRawFloat(cliOutput)
        assertEquals(Dav2PreprocessContract.OUTPUT_HEIGHT * Dav2PreprocessContract.OUTPUT_WIDTH, cliReference.size)

        val report = JSONObject()
            .put("schema", "blindassist_qnn_native_cached_context_r0")
            .put("cached_dlc_path", cachedDlc.absolutePath)
            .put("cached_dlc_size_bytes", cachedDlc.length())
            .put("repetitions", repetitions)
            .put("thermal_status_before", thermalStatus())
            .put("memory_before", memoryJson())
        val gateFailures = mutableListOf<String>()
        val nativeLibraryDir = arguments.getString("qnnRuntimeDir")
            ?: instrumentation.context.applicationInfo.nativeLibraryDir
        val initStart = SystemClock.elapsedRealtimeNanos()
        Dav2QnnCachedContext(cachedDlc.absolutePath, nativeLibraryDir).use { runtime ->
            report.put("context_init_ms", elapsedMs(initStart)).put("tensor_metadata", runtime.metadata)
            assertEquals("FLOAT_16", runtime.metadata.getString("input_type"))
            assertEquals("FLOAT_16", runtime.metadata.getString("output_type"))
            assertEquals(Dav2PreprocessContract.OUTPUT_ELEMENTS * 2, runtime.metadata.getInt("input_bytes"))
            assertEquals(Dav2PreprocessContract.PLANE * 2, runtime.metadata.getInt("output_bytes"))

            runtime.execute(officialFp16)
            val executeLatencies = DoubleArray(repetitions)
            repeat(repetitions) { index ->
                val start = SystemClock.elapsedRealtimeNanos()
                runtime.execute(officialFp16)
                executeLatencies[index] = elapsedMs(start)
            }
            val officialOutput = fp16FloatArray(runtime.output)
            val cliParity = parity(officialOutput, cliReference)
            val appCliGatePass = cliParity.getDouble("mean_abs") <= 0.002 &&
                cliParity.getDouble("p95_abs") <= 0.005 &&
                cliParity.getDouble("max_abs") <= 0.02
            report
                .put("graph_execute_ms", latencyJson(executeLatencies))
                .put("cli_parity", cliParity)
                .put("app_cli_gate_pass", appCliGatePass)
            if (!appCliGatePass) gateFailures += "App/CLI depth drift: $cliParity"

            Dav2NativePreprocessor().use { preprocessor ->
                val nativeFp32 = preprocessor.preprocessFp32(rgb)
                val fp32InputParity = parity(floatArray(nativeFp32), official)
                val fastStrictFp16 = preprocessor.preprocessFp16Strict(rgb)
                val fastFp16TensorParity = halfBitParity(fastStrictFp16, officialFp16)
                val canonicalFp32 = preprocessor.preprocessFp32Canonical(rgb)
                val canonicalFp32Parity = parity(floatArray(canonicalFp32), official)
                val canonicalFp16 = preprocessor.preprocessFp16CanonicalStrict(rgb)
                val canonicalFp16Parity = halfBitParity(canonicalFp16, officialFp16)
                val canonicalFp16Sha256 = sha256(canonicalFp16)
                report
                    .put("native_fast_fp32_input_parity", fp32InputParity)
                    .put("native_fast_strict_fp16_tensor_parity", fastFp16TensorParity)
                    .put("native_canonical_fp32_input_parity", canonicalFp32Parity)
                    .put("native_canonical_strict_fp16_tensor_parity", canonicalFp16Parity)
                    .put("official_fp16_sha256", sha256(officialFp16))
                    .put("native_canonical_strict_fp16_sha256", canonicalFp16Sha256)
                if (canonicalFp16Parity.getInt("mismatch_elements") != 0) {
                    gateFailures += "canonical strict FP16 tensor is not bit-exact: $canonicalFp16Parity"
                }
                val fullLatencies = DoubleArray(repetitions)
                repeat(repetitions) { index ->
                    val start = SystemClock.elapsedRealtimeNanos()
                    runtime.execute(preprocessor.preprocessFp16(rgb))
                    fullLatencies[index] = elapsedMs(start)
                }
                val fusedOutput = fp16FloatArray(runtime.output)
                val preprocessImpact = parity(fusedOutput, officialOutput)
                val fp16PreprocessDepthGatePass = preprocessImpact.getDouble("mean_abs") <= 0.002 &&
                    preprocessImpact.getDouble("p95_abs") <= 0.005 &&
                    preprocessImpact.getDouble("max_abs") <= 0.02
                report
                    .put("native_preprocess_fp16_plus_graph_execute_ms", latencyJson(fullLatencies))
                    .put("fp16_preprocess_depth_parity", preprocessImpact)
                    .put("fp16_preprocess_depth_gate_pass", fp16PreprocessDepthGatePass)
                if (!fp16PreprocessDepthGatePass) gateFailures +=
                    "FP16 preprocess depth drift: $preprocessImpact"
                val downstreamOfficial = downstream(officialOutput)
                val downstreamFused = downstream(fusedOutput)
                report.put("downstream_official_fp16", downstreamOfficial).put("downstream_native_fp16", downstreamFused)
                if (downstreamOfficial.getString("status") != downstreamFused.getString("status")) {
                    gateFailures += "downstream status mismatch: $downstreamOfficial vs $downstreamFused"
                }
                if (downstreamOfficial.getString("status") == "VALID" &&
                    downstreamFused.getString("status") == "VALID"
                ) {
                    val heightDrift = kotlin.math.abs(
                        downstreamOfficial.getDouble("relative_height") - downstreamFused.getDouble("relative_height"),
                    )
                    val scaleDrift = kotlin.math.abs(
                        downstreamOfficial.getDouble("student_scale") - downstreamFused.getDouble("student_scale"),
                    )
                    if (heightDrift > 0.01) gateFailures += "downstream relative-height drift: $heightDrift"
                    if (scaleDrift > 0.01) gateFailures += "downstream student-scale drift: $scaleDrift"
                }
            }
        }
        report
            .put("thermal_status_after", thermalStatus())
            .put("memory_after", memoryJson())
            .put("gate_pass", gateFailures.isEmpty())
            .put("gate_failures", JSONArray(gateFailures))
        File(instrumentation.targetContext.filesDir, REPORT_FILE).writeText(report.toString(2))
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
        assertTrue(gateFailures.joinToString(separator = "\n"), gateFailures.isEmpty())
    }

    private fun downstream(rawDepth: FloatArray): JSONObject {
        val depth = resizeDepthAlignCorners(rawDepth)
        val geometry = KnownHeightGroundPipeline.evaluateGeometry(
            depth,
            640,
            480,
            320.0,
            320.0,
            320.0,
            240.0,
            1.0341161949454936,
        )
        if (geometry !is KnownHeightGroundPipeline.Geometry) {
            return JSONObject().put("status", "UNKNOWN").put("reason", geometry.toString())
        }
        val student = KnownHeightScaleStudent.frozen().predict(geometry.features)
        if (student !is KnownHeightScaleStudent.Prediction.Valid) {
            return JSONObject().put("status", "UNKNOWN").put("reason", student.toString())
        }
        return JSONObject()
            .put("status", "VALID")
            .put("relative_height", geometry.relativeHeight)
            .put("inlier_fraction", geometry.inlierFraction)
            .put("student_scale", student.scale)
    }

    private fun resizeDepthAlignCorners(input: FloatArray): FloatArray {
        val output = FloatArray(640 * 480)
        for (row in 0 until 480) {
            val sourceY = row.toDouble() * (Dav2PreprocessContract.OUTPUT_HEIGHT - 1) / 479
            val y0 = sourceY.toInt()
            val y1 = minOf(y0 + 1, Dav2PreprocessContract.OUTPUT_HEIGHT - 1)
            val fy = sourceY - y0
            for (column in 0 until 640) {
                val sourceX = column.toDouble() * (Dav2PreprocessContract.OUTPUT_WIDTH - 1) / 639
                val x0 = sourceX.toInt()
                val x1 = minOf(x0 + 1, Dav2PreprocessContract.OUTPUT_WIDTH - 1)
                val fx = sourceX - x0
                val top = input[y0 * Dav2PreprocessContract.OUTPUT_WIDTH + x0] * (1.0 - fx) +
                    input[y0 * Dav2PreprocessContract.OUTPUT_WIDTH + x1] * fx
                val bottom = input[y1 * Dav2PreprocessContract.OUTPUT_WIDTH + x0] * (1.0 - fx) +
                    input[y1 * Dav2PreprocessContract.OUTPUT_WIDTH + x1] * fx
                output[row * 640 + column] = (top * (1.0 - fy) + bottom * fy).toFloat()
            }
        }
        return output
    }

    private fun fp16Buffer(values: FloatArray): ByteBuffer =
        ByteBuffer.allocateDirect(values.size * 2).order(ByteOrder.nativeOrder()).apply {
            val shorts = asShortBuffer()
            for (index in values.indices) shorts.put(index, floatToHalfBits(values[index]))
            rewind()
        }

    private fun fp16FloatArray(buffer: ByteBuffer): FloatArray {
        buffer.rewind()
        val shorts = buffer.asShortBuffer()
        return FloatArray(Dav2PreprocessContract.PLANE) { halfBitsToFloat(shorts.get(it)) }
    }

    private fun floatArray(buffer: ByteBuffer): FloatArray {
        val copy = buffer.duplicate().order(ByteOrder.nativeOrder())
        copy.position(0)
        return FloatArray(copy.limit() / 4).also { copy.asFloatBuffer().get(it) }
    }

    private fun halfBitParity(actual: ByteBuffer, expected: ByteBuffer): JSONObject {
        val actualShorts = actual.duplicate().order(ByteOrder.nativeOrder()).apply { position(0) }.asShortBuffer()
        val expectedShorts = expected.duplicate().order(ByteOrder.nativeOrder()).apply { position(0) }.asShortBuffer()
        assertEquals(expectedShorts.remaining(), actualShorts.remaining())
        var mismatches = 0
        var first = -1
        var firstExpected = 0
        var firstActual = 0
        var actualLower = 0
        var actualHigher = 0
        val channelMismatches = IntArray(3)
        for (index in 0 until actualShorts.remaining()) {
            val observed = actualShorts.get(index).toInt() and 0xffff
            val wanted = expectedShorts.get(index).toInt() and 0xffff
            if (observed != wanted) {
                if (first < 0) {
                    first = index
                    firstExpected = wanted
                    firstActual = observed
                }
                val observedFloat = halfBitsToFloat(observed.toShort())
                val wantedFloat = halfBitsToFloat(wanted.toShort())
                if (observedFloat < wantedFloat) actualLower++ else if (observedFloat > wantedFloat) actualHigher++
                channelMismatches[index / Dav2PreprocessContract.PLANE]++
                mismatches++
            }
        }
        return JSONObject()
            .put("elements", actualShorts.remaining())
            .put("mismatch_elements", mismatches)
            .put("first_mismatch_index", first)
            .put("first_expected_bits", if (first < 0) JSONObject.NULL else "0x${firstExpected.toString(16).padStart(4, '0')}")
            .put("first_actual_bits", if (first < 0) JSONObject.NULL else "0x${firstActual.toString(16).padStart(4, '0')}")
            .put("actual_lower_than_expected", actualLower)
            .put("actual_higher_than_expected", actualHigher)
            .put("channel_mismatch_elements", JSONArray(channelMismatches.toList()))
    }

    private fun sha256(buffer: ByteBuffer): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val copy = buffer.duplicate().apply { position(0) }
        val chunk = ByteArray(64 * 1024)
        while (copy.hasRemaining()) {
            val count = minOf(copy.remaining(), chunk.size)
            copy.get(chunk, 0, count)
            digest.update(chunk, 0, count)
        }
        return digest.digest().joinToString("") { "%02X".format(it) }
    }

    private fun parity(actual: FloatArray, expected: FloatArray): JSONObject {
        assertEquals(expected.size, actual.size)
        val errors = DoubleArray(actual.size)
        var sum = 0.0
        for (index in actual.indices) {
            assertTrue(actual[index].isFinite())
            errors[index] = kotlin.math.abs(actual[index].toDouble() - expected[index])
            sum += errors[index]
        }
        errors.sort()
        return JSONObject()
            .put("elements", errors.size)
            .put("mean_abs", sum / errors.size)
            .put("p95_abs", errors[(0.95 * (errors.size - 1)).toInt()])
            .put("max_abs", errors.last())
    }

    private fun latencyJson(values: DoubleArray): JSONObject {
        val sorted = values.sortedArray()
        return JSONObject()
            .put("all", JSONArray(values.toList()))
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

    private fun readNpyBytes(file: File): ByteArray {
        val bytes = file.readBytes()
        return bytes.copyOfRange(npyOffset(bytes), bytes.size)
    }

    private fun readNpyFloat(file: File): FloatArray {
        val bytes = file.readBytes()
        val offset = npyOffset(bytes)
        val buffer = ByteBuffer.wrap(bytes, offset, bytes.size - offset).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also(buffer::get)
    }

    private fun npyOffset(bytes: ByteArray): Int {
        val major = bytes[6].toInt() and 0xff
        val length = if (major == 1) {
            (bytes[8].toInt() and 0xff) or ((bytes[9].toInt() and 0xff) shl 8)
        } else {
            ByteBuffer.wrap(bytes, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int
        }
        return if (major == 1) 10 + length else 12 + length
    }

    private fun readRawFloat(file: File): FloatArray {
        val bytes = file.readBytes()
        return FloatArray(bytes.size / 4).also {
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it)
        }
    }

    private fun thermalStatus(): Int = InstrumentationRegistry.getInstrumentation().targetContext
        .getSystemService(PowerManager::class.java).currentThermalStatus

    private fun memoryJson(): JSONObject {
        val runtime = Runtime.getRuntime()
        return JSONObject()
            .put("pss_kib", Debug.getPss())
            .put("java_heap_used_bytes", runtime.totalMemory() - runtime.freeMemory())
            .put("native_heap_allocated_bytes", Debug.getNativeHeapAllocatedSize())
    }

    private fun elapsedMs(startNanos: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - startNanos) / 1_000_000.0

    private companion object {
        const val REPORT_KEY = "qnn_native_cached_context_r0_report"
        const val REPORT_FILE = "qnn-native-cached-context-r0.json"
    }
}
