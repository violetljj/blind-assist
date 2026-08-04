package com.linnan.blindassist.hftf

import android.os.Bundle
import android.os.Debug
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.security.MessageDigest
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter

@RunWith(AndroidJUnit4::class)
class Dav2MetricAndroidTfliteTest {
    @Test
    fun fixedGradientParityAndShortPerformance() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val model = File(requireNotNull(InstrumentationRegistry.getArguments().getString("modelPath")))
        assertTrue(model.isFile)
        assertEquals(MODEL_SHA256, sha256(model))
        val report = JSONObject()
            .put("schema", "blindassist_dav2_metric_android_tflite_short_performance_r0")
            .put("model_sha256", MODEL_SHA256)
            .put("runtime", "litert-1.4.2-cpu-4threads")
            .put("memory_before", memoryJson())
        val options = Interpreter.Options().setNumThreads(4)
        val loadStarted = SystemClock.elapsedRealtimeNanos()
        Interpreter(mapReadOnly(model), options).use { interpreter ->
            report.put("session_load_ms", elapsedMs(loadStarted))
            report.put("input_shape", JSONArray(interpreter.getInputTensor(0).shape().toList()))
            report.put("output_shape", JSONArray(interpreter.getOutputTensor(0).shape().toList()))
            val input = validationInput()
            val output = ByteBuffer.allocateDirect(OUTPUT_COUNT * 4).order(ByteOrder.nativeOrder())
            repeat(WARMUP_RUNS) { output.rewind(); interpreter.run(input, output); validate(output) }
            val latencies = ArrayList<Double>()
            repeat(MEASURED_RUNS) {
                output.rewind()
                val started = SystemClock.elapsedRealtimeNanos()
                interpreter.run(input, output)
                latencies += elapsedMs(started)
                validate(output)
            }
            val sorted = latencies.sorted()
            report
                .put("warmup_runs", WARMUP_RUNS)
                .put("measured_runs", MEASURED_RUNS)
                .put(
                    "da_latency_ms",
                    JSONObject()
                        .put("all", JSONArray(latencies))
                        .put("p50", percentile(sorted, 0.50))
                        .put("p95", percentile(sorted, 0.95))
                        .put("max", sorted.last()),
                )
                .put("processing_fps", 1000.0 / latencies.average())
                .put("memory_after_runs", memoryJson())
        }
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
    }

    private fun validationInput(): ByteBuffer {
        val output = ByteBuffer.allocateDirect(INPUT_COUNT * 4).order(ByteOrder.nativeOrder())
        val means = floatArrayOf(0.485f, 0.456f, 0.406f)
        val deviations = floatArrayOf(0.229f, 0.224f, 0.225f)
        for (channel in 0..2) for (row in 0 until HEIGHT) for (column in 0 until WIDTH) {
            val x = column.toFloat() / (WIDTH - 1)
            val y = row.toFloat() / (HEIGHT - 1)
            val value = when (channel) { 0 -> x; 1 -> y; else -> 0.5f * (x + y) }
            output.putFloat((value - means[channel]) / deviations[channel])
        }
        output.rewind()
        return output
    }

    private fun validate(output: ByteBuffer) {
        output.rewind()
        val values = output.asFloatBuffer()
        var minimum = Float.POSITIVE_INFINITY
        var maximum = Float.NEGATIVE_INFINITY
        while (values.hasRemaining()) {
            val value = values.get()
            assertTrue(value.isFinite())
            if (value < minimum) minimum = value
            if (value > maximum) maximum = value
        }
        assertEquals(EXPECTED_MIN_M, minimum.toDouble(), OUTPUT_EXTREMA_TOLERANCE_M)
        assertEquals(EXPECTED_MAX_M, maximum.toDouble(), OUTPUT_EXTREMA_TOLERANCE_M)
    }

    private fun mapReadOnly(file: File): ByteBuffer = FileInputStream(file).use { stream ->
        stream.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    private fun percentile(sorted: List<Double>, quantile: Double): Double {
        val position = quantile * (sorted.size - 1)
        val lower = position.toInt()
        val upper = minOf(lower + 1, sorted.lastIndex)
        return sorted[lower] * (1.0 - (position - lower)) + sorted[upper] * (position - lower)
    }

    private fun memoryJson(): JSONObject {
        val runtime = Runtime.getRuntime()
        return JSONObject()
            .put("pss_kib", Debug.getPss())
            .put("java_heap_used_bytes", runtime.totalMemory() - runtime.freeMemory())
            .put("native_heap_allocated_bytes", Debug.getNativeHeapAllocatedSize())
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { stream ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02X".format(Locale.US, it) }
    }

    private fun elapsedMs(start: Long) = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0

    private companion object {
        const val REPORT_KEY = "dav2_metric_android_tflite_short_performance_report"
        const val MODEL_SHA256 = "0277FBC74C73D95433B43BEE9D61DD08F1E79B67A2F64A6DA871F3A23FBED8E3"
        const val HEIGHT = 518
        const val WIDTH = 686
        const val INPUT_COUNT = 3 * HEIGHT * WIDTH
        const val OUTPUT_COUNT = HEIGHT * WIDTH
        const val WARMUP_RUNS = 2
        const val MEASURED_RUNS = 10
        const val EXPECTED_MIN_M = 1.4515197277069092
        const val EXPECTED_MAX_M = 2.0870625972747803
        const val OUTPUT_EXTREMA_TOLERANCE_M = 0.005
    }
}
