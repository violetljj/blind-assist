package com.linnan.blindassist.hftf

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.FileInputStream
import java.nio.FloatBuffer
import java.security.MessageDigest
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class Dav2MetricAndroidPerformanceTest {
    @Test
    fun fixedGradientParityAndShortPerformance() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val model = File(requireNotNull(arguments.getString("modelPath")))
        val backend = arguments.getString("backend") ?: "cpu"
        require(backend == "cpu" || backend == "nnapi") { "unsupported backend $backend" }
        assertTrue("missing model ${model.absolutePath}", model.isFile)
        assertEquals(MODEL_SHA256, sha256(model))

        val environment = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            if (backend == "nnapi") addNnapi()
        }
        val report = JSONObject()
            .put("schema", "blindassist_dav2_metric_android_short_performance_r0")
            .put("model_sha256", MODEL_SHA256)
            .put("device", deviceJson())
            .put("runtime", "onnxruntime-android-1.26.0-$backend-4threads")
            .put("memory_before", memoryJson())
        try {
            val loadStarted = SystemClock.elapsedRealtimeNanos()
            environment.createSession(model.absolutePath, options).use { session ->
                report.put("session_load_ms", elapsedMs(loadStarted))
                report.put("memory_after_load", memoryJson())
                val input = makeValidationInput()
                OnnxTensor.createTensor(
                    environment,
                    FloatBuffer.wrap(input),
                    longArrayOf(1, 3, HEIGHT.toLong(), WIDTH.toLong()),
                ).use { tensor ->
                    repeat(WARMUP_RUNS) { session.run(mapOf("image" to tensor)).use(::validateOutput) }
                    val latencies = ArrayList<Double>()
                    repeat(MEASURED_RUNS) {
                        val started = SystemClock.elapsedRealtimeNanos()
                        session.run(mapOf("image" to tensor)).use { output ->
                            latencies += elapsedMs(started)
                            validateOutput(output)
                        }
                    }
                    report
                        .put("warmup_runs", WARMUP_RUNS)
                        .put("measured_runs", MEASURED_RUNS)
                        .put("da_latency_ms", latencyJson(latencies))
                        .put("processing_fps", 1000.0 / latencies.average())
                        .put("memory_after_runs", memoryJson())
                        .put("thermal_status", thermalStatus())
                }
            }
        } finally {
            options.close()
        }
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
    }

    private fun makeValidationInput(): FloatArray {
        val plane = HEIGHT * WIDTH
        val output = FloatArray(3 * plane)
        val means = floatArrayOf(0.485f, 0.456f, 0.406f)
        val standardDeviations = floatArrayOf(0.229f, 0.224f, 0.225f)
        for (row in 0 until HEIGHT) {
            val y = row.toFloat() / (HEIGHT - 1)
            for (column in 0 until WIDTH) {
                val x = column.toFloat() / (WIDTH - 1)
                val rgb = floatArrayOf(x, y, 0.5f * (x + y))
                val pixel = row * WIDTH + column
                for (channel in 0..2) {
                    output[channel * plane + pixel] =
                        (rgb[channel] - means[channel]) / standardDeviations[channel]
                }
            }
        }
        return output
    }

    private fun validateOutput(result: OrtSession.Result) {
        val tensor = result[0] as OnnxTensor
        val values = tensor.floatBuffer
        var minimum = Float.POSITIVE_INFINITY
        var maximum = Float.NEGATIVE_INFINITY
        for (index in 0 until values.capacity()) {
            val value = values[index]
            assertTrue("non-finite output at $index", value.isFinite())
            if (value < minimum) minimum = value
            if (value > maximum) maximum = value
        }
        assertEquals(EXPECTED_MIN_M, minimum.toDouble(), OUTPUT_EXTREMA_TOLERANCE_M)
        assertEquals(EXPECTED_MAX_M, maximum.toDouble(), OUTPUT_EXTREMA_TOLERANCE_M)
    }

    private fun latencyJson(values: List<Double>): JSONObject {
        val sorted = values.sorted()
        return JSONObject()
            .put("all", JSONArray(values))
            .put("p50", percentile(sorted, 0.50))
            .put("p95", percentile(sorted, 0.95))
            .put("max", sorted.last())
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

    private fun deviceJson() = JSONObject()
        .put("manufacturer", Build.MANUFACTURER)
        .put("model", Build.MODEL)
        .put("device", Build.DEVICE)
        .put("android_release", Build.VERSION.RELEASE)
        .put("sdk", Build.VERSION.SDK_INT)

    private fun thermalStatus(): Int {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        return context.getSystemService(PowerManager::class.java).currentThermalStatus
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

    private fun elapsedMs(startNanos: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - startNanos) / 1_000_000.0

    private companion object {
        const val REPORT_KEY = "dav2_metric_android_short_performance_report"
        const val MODEL_SHA256 = "870339770E21675830F7E2020983DDA058752D237C8B86951ED1E6F9A6243D01"
        const val HEIGHT = 518
        const val WIDTH = 686
        const val WARMUP_RUNS = 2
        const val MEASURED_RUNS = 10
        const val EXPECTED_MIN_M = 1.4515197277069092
        const val EXPECTED_MAX_M = 2.0870625972747803
        const val OUTPUT_EXTREMA_TOLERANCE_M = 0.005
    }
}
