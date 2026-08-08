package com.linnan.blindassist.hftf

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.FileInputStream
import java.nio.FloatBuffer
import java.security.MessageDigest
import java.util.Locale
import kotlin.math.max
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MetricDepthAndroidDualArmCanaryTest {
    @Test
    fun frozenMetricDepthDualArmRuntime() {
        val arguments = InstrumentationRegistry.getArguments()
        val metric3d = ModelArm(
            id = "metric3d_vit_small",
            path = requiredArgument(arguments.getString("metric3dPath"), "metric3dPath"),
            sha256 = METRIC3D_SHA256,
            inputs = metric3dInputs(),
        )
        val unidepth = ModelArm(
            id = "unidepth_v2_vits14_camera",
            path = requiredArgument(arguments.getString("unidepthPath"), "unidepthPath"),
            sha256 = UNIDEPTH_SHA256,
            inputs = unidepthInputs(),
        )
        val environment = OrtEnvironment.getEnvironment()
        val result = JSONObject()
            .put("schema_version", 1)
            .put("scope", "deployment_runtime_only_no_quality_claim")
            .put("device", deviceJson())
            .put("runtime", JSONObject().put("name", "onnxruntime-android").put("version", "1.26.0"))

        val cpuResults = JSONArray()
        var cpuFailure: Throwable? = null
        for (arm in listOf(metric3d, unidepth)) {
            try {
                cpuResults.put(runArm(environment, arm, Backend.CPU))
            } catch (failure: Throwable) {
                cpuResults.put(failureJson(arm, Backend.CPU, failure))
                cpuFailure = cpuFailure ?: failure
            }
        }
        result.put("cpu", cpuResults)
        result.put(
            "cpu_terminal",
            if (cpuFailure == null) "DUALARM_ANDROID_CPU_EXECUTION_SUPPORTED"
            else "DUALARM_ANDROID_CPU_EXECUTION_NOT_SUPPORTED",
        )

        val runNnapi = arguments.getString("runNnapi")?.toBooleanStrictOrNull() ?: true
        if (runNnapi) {
            val nnapiResults = JSONArray()
            var nnapiComparable = true
            for (arm in listOf(metric3d, unidepth)) {
                try {
                    nnapiResults.put(runArm(environment, arm, Backend.NNAPI))
                } catch (failure: Throwable) {
                    nnapiResults.put(failureJson(arm, Backend.NNAPI, failure))
                    nnapiComparable = false
                }
            }
            result.put("nnapi", nnapiResults)
            result.put(
                "nnapi_terminal",
                if (nnapiComparable) "DUALARM_ANDROID_NNAPI_EXECUTION_SUPPORTED"
                else "DUALARM_ANDROID_NNAPI_NOT_COMPARABLE",
            )
        } else {
            result.put("nnapi_terminal", "NOT_RUN_BY_EXPLICIT_ARGUMENT")
        }

        Log.i(
            TAG,
            "TERMINALS cpu=${result.getString("cpu_terminal")}" +
                " nnapi=${result.getString("nnapi_terminal")}",
        )
        assertTrue("common CPU backend failed: ${cpuFailure?.stackTraceToString()}", cpuFailure == null)
    }

    private fun runArm(
        environment: OrtEnvironment,
        arm: ModelArm,
        backend: Backend,
    ): JSONObject {
        val model = File(arm.path)
        assertTrue("missing model ${model.absolutePath}", model.isFile)
        val actualSha = sha256(model)
        assertEquals("model hash mismatch for ${arm.id}", arm.sha256, actualSha)
        Log.i(TAG, "START arm=${arm.id} backend=${backend.id} size=${model.length()}")

        val before = memoryJson()
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            if (backend == Backend.NNAPI) addNnapi()
        }
        try {
            val loadStart = SystemClock.elapsedRealtimeNanos()
            environment.createSession(model.absolutePath, options).use { session ->
                val loadMs = elapsedMs(loadStart)
                val afterLoad = memoryJson()
                val inputMetadata = metadataJson(session.inputInfo)
                val outputMetadata = metadataJson(session.outputInfo)
                val tensors = LinkedHashMap<String, OnnxTensor>()
                try {
                    for ((name, input) in arm.inputs) {
                        tensors[name] = OnnxTensor.createTensor(
                            environment,
                            FloatBuffer.wrap(input.values),
                            input.shape,
                        )
                    }
                    session.run(tensors).use { warmup -> validateOutputs(warmup) }
                    val latencies = ArrayList<Double>(MEASURED_RUNS)
                    repeat(MEASURED_RUNS) {
                        val start = SystemClock.elapsedRealtimeNanos()
                        session.run(tensors).use { outputs ->
                            latencies += elapsedMs(start)
                            validateOutputs(outputs)
                        }
                    }
                    val afterRuns = memoryJson()
                    val summary = JSONObject()
                        .put("arm", arm.id)
                        .put("backend", backend.id)
                        .put("status", "PASS")
                        .put("model_path", model.absolutePath)
                        .put("model_sha256", actualSha)
                        .put("model_size_bytes", model.length())
                        .put("session_load_ms", loadMs)
                        .put("warmup_runs", 1)
                        .put("measured_runs", MEASURED_RUNS)
                        .put("latency_ms", latencyJson(latencies))
                        .put("memory_before", before)
                        .put("memory_after_load", afterLoad)
                        .put("memory_after_runs", afterRuns)
                        .put("inputs", inputMetadata)
                        .put("outputs", outputMetadata)
                    Log.i(TAG, "ARM_RESULT=${summary}")
                    return summary
                } finally {
                    tensors.values.forEach(OnnxTensor::close)
                }
            }
        } finally {
            options.close()
        }
    }

    private fun validateOutputs(result: OrtSession.Result) {
        assertTrue("model returned no outputs", result.size() > 0)
        for (index in 0 until result.size()) {
            val tensor = result[index] as? OnnxTensor
                ?: throw AssertionError("output $index is not a tensor")
            val values = tensor.floatBuffer
            val count = values.capacity()
            assertTrue("output $index is empty", count > 0)
            val stride = max(1, count / MAX_FINITE_SAMPLES)
            var position = 0
            while (position < count) {
                assertTrue("non-finite output $index at $position", values[position].isFinite())
                position += stride
            }
        }
    }

    private fun metric3dInputs(): Map<String, TensorInput> {
        val height = 616
        val width = 1064
        val plane = height * width
        val values = FloatArray(3 * plane)
        val channels = floatArrayOf(123.675f, 116.28f, 103.53f)
        for (channel in 0 until 3) {
            java.util.Arrays.fill(values, channel * plane, (channel + 1) * plane, channels[channel])
        }
        return mapOf("pixel_values" to TensorInput(values, longArrayOf(1, 3, height.toLong(), width.toLong())))
    }

    private fun unidepthInputs(): Map<String, TensorInput> {
        val height = 434
        val width = 574
        val plane = height * width
        val rgbs = FloatArray(3 * plane)
        val rays = FloatArray(3 * plane)
        val scale = 0.8838834764831844f
        val fx = 542.822841f * scale
        val fy = 542.57687f * scale
        val cx = 315.59352f * scale
        val cy = 237.756098f * scale
        for (v in 0 until height) {
            for (u in 0 until width) {
                val x = (u - cx) / fx
                val y = (v - cy) / fy
                val norm = sqrt(x * x + y * y + 1f)
                val index = v * width + u
                rays[index] = x / norm
                rays[plane + index] = y / norm
                rays[2 * plane + index] = 1f / norm
            }
        }
        val shape = longArrayOf(1, 3, height.toLong(), width.toLong())
        return linkedMapOf(
            "rgbs" to TensorInput(rgbs, shape),
            "rays" to TensorInput(rays, shape),
        )
    }

    private fun metadataJson(info: Map<String, *>): JSONObject {
        val output = JSONObject()
        for ((name, value) in info) output.put(name, value.toString())
        return output
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
        if (sorted.size == 1) return sorted[0]
        val position = quantile * (sorted.size - 1)
        val lower = position.toInt()
        val upper = minOf(lower + 1, sorted.lastIndex)
        val fraction = position - lower
        return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
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
        .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))

    private fun failureJson(arm: ModelArm, backend: Backend, failure: Throwable): JSONObject {
        Log.e(TAG, "FAIL arm=${arm.id} backend=${backend.id}", failure)
        return JSONObject()
            .put("arm", arm.id)
            .put("backend", backend.id)
            .put("status", "FAIL")
            .put("exception", failure.javaClass.name)
            .put("message", failure.message)
    }

    private fun requiredArgument(value: String?, name: String): String =
        requireNotNull(value?.takeIf(String::isNotBlank)) { "missing instrumentation argument $name" }

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

    private data class ModelArm(
        val id: String,
        val path: String,
        val sha256: String,
        val inputs: Map<String, TensorInput>,
    )

    private data class TensorInput(val values: FloatArray, val shape: LongArray)

    private enum class Backend(val id: String) { CPU("ort_cpu"), NNAPI("ort_nnapi") }

    companion object {
        private const val TAG = "MetricDepthDualArmR0"
        private const val MEASURED_RUNS = 3
        private const val MAX_FINITE_SAMPLES = 4096
        private const val METRIC3D_SHA256 = "674A665052F01BB2B64687200182F3380F0A462F4B6EF2E51FEFD06C84D8EE75"
        private const val UNIDEPTH_SHA256 = "2BA1FE9F9D8F050FBE83C164C5B5D01234119EE44273AC825F233702415AB958"
    }
}
