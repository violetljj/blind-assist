package com.linnan.blindassist.benchmark

import android.graphics.BitmapFactory
import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.vision.ImagePreprocessor
import com.linnan.blindassist.vision.TfliteYoloDetector
import com.qualcomm.qti.QnnDelegate
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Benchmark-only CPU versus Qualcomm QNN HTP comparison.
 *
 * This test never changes the production detector backend. A valid result
 * requires QNN's FP16 HTP capability, a live delegate, successful graph
 * preparation, successful inference, and finite outputs on the same inputs.
 */
@RunWith(AndroidJUnit4::class)
class QnnHtpYoloDeviceBenchmarkTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun compareCpuAndQnnHtpOnSameYoloInputs() {
        val imageLimit = arguments.getString(ARG_IMAGE_LIMIT)
            ?.toIntOrNull()
            ?.coerceIn(1, MAX_IMAGE_LIMIT)
            ?: DEFAULT_IMAGE_LIMIT
        val measuredRuns = arguments.getString(ARG_MEASURED_RUNS)
            ?.toIntOrNull()
            ?.coerceIn(1, MAX_MEASURED_RUNS)
            ?: DEFAULT_MEASURED_RUNS
        val warmupRuns = arguments.getString(ARG_WARMUP_RUNS)
            ?.toIntOrNull()
            ?.coerceIn(1, MAX_WARMUP_RUNS)
            ?: DEFAULT_WARMUP_RUNS

        val imagePaths = loadImagePaths().take(imageLimit)
        assertTrue("BlindAssist EvalSet images are unavailable", imagePaths.isNotEmpty())

        System.loadLibrary("cdsprpc")
        val fp16Capability = QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)
        assertTrue("QNN runtime does not report HTP FP16 capability", fp16Capability)
        val qnnVersion = QnnDelegate.getVersion().toList()

        val qnnOptions = QnnDelegate.Options().apply {
            setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
            setSkelLibraryDir(testContext.applicationInfo.nativeLibraryDir)
            setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
            setHtpPerformanceMode(
                QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE
            )
            setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
            setProfiling(QnnDelegate.Options.ProfilingOptions.DETAILED_PROFILING)
            setCacheDir(targetContext.codeCacheDir.absolutePath)
            setModelToken(MODEL_TOKEN)
        }

        val qnnDelegate = QnnDelegate(qnnOptions)
        assertTrue("QNN HTP delegate is not available", qnnDelegate.isAvailable)

        val mappedModel = loadMappedAsset(TfliteYoloDetector.MODEL_ASSET)
        val cpuInitStart = System.nanoTime()
        val cpu = Interpreter(mappedModel, Interpreter.Options().setNumThreads(CPU_THREADS))
        cpu.allocateTensors()
        val cpuInitMs = elapsedMs(cpuInitStart)

        val qnnInitStart = System.nanoTime()
        val htp = Interpreter(
            mappedModel,
            Interpreter.Options()
                .setNumThreads(CPU_THREADS)
                .addDelegate(qnnDelegate)
        )
        htp.allocateTensors()
        val qnnInitMs = elapsedMs(qnnInitStart)

        val inputShape = cpu.getInputTensor(0).shape()
        val outputShape = cpu.getOutputTensor(0).shape()
        assertTrue(
            "Unexpected YOLO input shape: ${inputShape.contentToString()}",
            inputShape.contentEquals(intArrayOf(1, TfliteYoloDetector.INPUT_SIZE, TfliteYoloDetector.INPUT_SIZE, 3))
        )
        assertTrue(
            "CPU/QNN output shapes differ",
            outputShape.contentEquals(htp.getOutputTensor(0).shape())
        )
        val outputBytes = cpu.getOutputTensor(0).numBytes()
        assertTrue("CPU/QNN output sizes differ", outputBytes == htp.getOutputTensor(0).numBytes())

        val preprocessor = ImagePreprocessor(TfliteYoloDetector.INPUT_SIZE)
        val inputs = imagePaths.map { path ->
            val bitmap = decodeBitmap(path)
            try {
                val prepared = preprocessor.prepare(bitmap)
                ByteBuffer.allocateDirect(prepared.buffer.capacity())
                    .order(ByteOrder.nativeOrder())
                    .apply {
                        prepared.buffer.rewind()
                        put(prepared.buffer)
                        rewind()
                    }
            } finally {
                bitmap.recycle()
            }
        }
        val cpuOutput = directOutput(outputBytes)
        val htpOutput = directOutput(outputBytes)

        repeat(warmupRuns) { index ->
            invoke(cpu, inputs[index % inputs.size], cpuOutput)
            invoke(htp, inputs[index % inputs.size], htpOutput)
        }

        val cpuTimings = ArrayList<Double>(measuredRuns)
        val htpTimings = ArrayList<Double>(measuredRuns)
        repeat(measuredRuns) { index ->
            val input = inputs[index % inputs.size]
            cpuTimings += timedInvoke(cpu, input, cpuOutput)
            htpTimings += timedInvoke(htp, input, htpOutput)
        }

        val parityRows = JSONArray()
        var globalMaxAbs = 0.0
        var weightedAbsSum = 0.0
        var comparedValueCount = 0L
        var allFinite = true
        imagePaths.forEachIndexed { index, path ->
            invoke(cpu, inputs[index], cpuOutput)
            invoke(htp, inputs[index], htpOutput)
            val cpuValues = floats(cpuOutput)
            val htpValues = floats(htpOutput)
            val parity = compareOutputs(cpuValues, htpValues)
            globalMaxAbs = maxOf(globalMaxAbs, parity.maxAbs)
            weightedAbsSum += parity.meanAbs * parity.valueCount
            comparedValueCount += parity.valueCount
            allFinite = allFinite && parity.allFinite
            parityRows.put(
                JSONObject()
                    .put("image_path", path)
                    .put("value_count", parity.valueCount)
                    .put("all_finite", parity.allFinite)
                    .put("max_abs", parity.maxAbs)
                    .put("mean_abs", parity.meanAbs)
                    .put("cosine_similarity", parity.cosineSimilarity)
                    .put("cpu_scores_at_or_above_0_35", parity.cpuScoresAtThreshold)
                    .put("htp_scores_at_or_above_0_35", parity.htpScoresAtThreshold)
            )
        }
        assertTrue("QNN HTP produced non-finite output", allFinite)

        val profilingBytes = qnnDelegate.profilingResult
        val report = JSONObject()
            .put("schema", REPORT_SCHEMA)
            .put("disposition", "BENCHMARK_ONLY_NO_PRODUCTION_BACKEND_CHANGE")
            .put("device", JSONObject()
                .put("manufacturer", Build.MANUFACTURER)
                .put("model", Build.MODEL)
                .put("soc_model", Build.SOC_MODEL)
                .put("android_release", Build.VERSION.RELEASE)
                .put("sdk_int", Build.VERSION.SDK_INT))
            .put("model", JSONObject()
                .put("asset", TfliteYoloDetector.MODEL_ASSET)
                .put("sha256", sha256Asset(TfliteYoloDetector.MODEL_ASSET))
                .put("input_shape", JSONArray(inputShape.toList()))
                .put("output_shape", JSONArray(outputShape.toList())))
            .put("qnn", JSONObject()
                .put("maven_version", QNN_MAVEN_VERSION)
                .put("runtime_version", JSONArray(qnnVersion))
                .put("backend", "HTP_BACKEND")
                .put("precision", "HTP_PRECISION_FP16")
                .put("delegate_available", true)
                .put("htp_fp16_capability", fp16Capability)
                .put("skel_library_dir", testContext.applicationInfo.nativeLibraryDir)
                .put("profiling_result_bytes", profilingBytes?.size ?: 0))
            .put("protocol", JSONObject()
                .put("image_count", imagePaths.size)
                .put("warmup_runs_per_backend", warmupRuns)
                .put("measured_runs_per_backend", measuredRuns)
                .put("cpu_threads", CPU_THREADS)
                .put("interleaved_cpu_htp", true)
                .put("invoke_only_timing", true))
            .put("initialization_ms", JSONObject()
                .put("cpu", round3(cpuInitMs))
                .put("qnn_htp", round3(qnnInitMs)))
            .put("latency_ms", JSONObject()
                .put("cpu", stats(cpuTimings))
                .put("qnn_htp", stats(htpTimings))
                .put("p50_speedup", round3(percentile(cpuTimings, 0.50) / percentile(htpTimings, 0.50)))
                .put("p95_speedup", round3(percentile(cpuTimings, 0.95) / percentile(htpTimings, 0.95))))
            .put("raw_output_parity", JSONObject()
                .put("all_finite", allFinite)
                .put("compared_value_count", comparedValueCount)
                .put("global_max_abs", globalMaxAbs)
                .put(
                    "global_mean_abs",
                    if (comparedValueCount == 0L) JSONObject.NULL else weightedAbsSum / comparedValueCount
                )
                .put("per_image", parityRows))

        val artifactDir = File(targetContext.filesDir, ARTIFACT_DIR).apply { mkdirs() }
        File(artifactDir, "benchmark.json").writeText(report.toString(2), Charsets.UTF_8)
        if (profilingBytes != null && profilingBytes.isNotEmpty()) {
            File(artifactDir, "qnn-profile.bin").writeBytes(profilingBytes)
        }
        Log.i(TAG, "artifactDir=${artifactDir.absolutePath}")
        Log.i(
            TAG,
            "cpuP50=${percentile(cpuTimings, 0.50)}ms htpP50=${percentile(htpTimings, 0.50)}ms " +
                "p50Speedup=${percentile(cpuTimings, 0.50) / percentile(htpTimings, 0.50)}"
        )

        htp.close()
        cpu.close()
        qnnDelegate.close()
    }

    private fun loadImagePaths(): List<String> {
        return testContext.assets.open(EVALSET_MANIFEST)
            .bufferedReader(Charsets.UTF_8)
            .useLines { lines ->
                lines.filter { it.isNotBlank() }
                    .map { JSONObject(it).getString("image_path") }
                    .map { "$EVALSET_ROOT/$it" }
                    .toList()
            }
    }

    private fun decodeBitmap(assetPath: String) =
        testContext.assets.open(assetPath).use { input ->
            requireNotNull(BitmapFactory.decodeStream(input)) { "Failed to decode $assetPath" }
        }

    private fun loadMappedAsset(assetName: String): MappedByteBuffer {
        val descriptor = testContext.assets.openFd(assetName)
        descriptor.use {
            FileInputStream(it.fileDescriptor).channel.use { channel ->
                return channel.map(FileChannel.MapMode.READ_ONLY, it.startOffset, it.declaredLength)
            }
        }
    }

    private fun directOutput(bytes: Int): ByteBuffer =
        ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())

    private fun invoke(interpreter: Interpreter, input: ByteBuffer, output: ByteBuffer) {
        input.rewind()
        output.rewind()
        interpreter.run(input, output)
    }

    private fun timedInvoke(
        interpreter: Interpreter,
        input: ByteBuffer,
        output: ByteBuffer
    ): Double {
        input.rewind()
        output.rewind()
        val start = System.nanoTime()
        interpreter.run(input, output)
        return elapsedMs(start)
    }

    private fun floats(output: ByteBuffer): FloatArray {
        output.rewind()
        return FloatArray(output.capacity() / Float.SIZE_BYTES).also {
            output.asFloatBuffer().get(it)
        }
    }

    private fun compareOutputs(cpu: FloatArray, htp: FloatArray): OutputParity {
        require(cpu.size == htp.size)
        var maxAbs = 0.0
        var absSum = 0.0
        var dot = 0.0
        var cpuNorm = 0.0
        var htpNorm = 0.0
        var allFinite = true
        var cpuScores = 0
        var htpScores = 0
        cpu.indices.forEach { index ->
            val a = cpu[index].toDouble()
            val b = htp[index].toDouble()
            allFinite = allFinite && a.isFinite() && b.isFinite()
            val difference = abs(a - b)
            maxAbs = maxOf(maxAbs, difference)
            absSum += difference
            dot += a * b
            cpuNorm += a * a
            htpNorm += b * b
            if (index >= BOX_CHANNELS * PREDICTION_COUNT) {
                if (a >= TfliteYoloDetector.CONFIDENCE_THRESHOLD) cpuScores++
                if (b >= TfliteYoloDetector.CONFIDENCE_THRESHOLD) htpScores++
            }
        }
        val denominator = sqrt(cpuNorm) * sqrt(htpNorm)
        return OutputParity(
            valueCount = cpu.size.toLong(),
            allFinite = allFinite,
            maxAbs = maxAbs,
            meanAbs = absSum / cpu.size,
            cosineSimilarity = if (denominator == 0.0) 1.0 else dot / denominator,
            cpuScoresAtThreshold = cpuScores,
            htpScoresAtThreshold = htpScores
        )
    }

    private fun stats(values: List<Double>) = JSONObject()
        .put("count", values.size)
        .put("min", round3(values.min()))
        .put("p50", round3(percentile(values, 0.50)))
        .put("p95", round3(percentile(values, 0.95)))
        .put("max", round3(values.max()))
        .put("mean", round3(values.average()))

    private fun percentile(values: List<Double>, quantile: Double): Double {
        val sorted = values.sorted()
        if (sorted.size == 1) return sorted.first()
        val position = quantile.coerceIn(0.0, 1.0) * (sorted.lastIndex)
        val lower = position.toInt()
        val upper = minOf(lower + 1, sorted.lastIndex)
        val weight = position - lower
        return sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }

    private fun elapsedMs(startNanos: Long): Double =
        (System.nanoTime() - startNanos) / 1_000_000.0

    private fun round3(value: Double): Double =
        kotlin.math.round(value * 1000.0) / 1000.0

    private fun sha256Asset(assetName: String): String =
        testContext.assets.open(assetName).use { input ->
            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
            digest.digest().joinToString("") { "%02x".format(it) }
        }

    private data class OutputParity(
        val valueCount: Long,
        val allFinite: Boolean,
        val maxAbs: Double,
        val meanAbs: Double,
        val cosineSimilarity: Double,
        val cpuScoresAtThreshold: Int,
        val htpScoresAtThreshold: Int
    )

    companion object {
        private const val TAG = "QnnHtpYoloBenchmark"
        private const val REPORT_SCHEMA = "blindassist_qnn_htp_yolo_benchmark_v1"
        private const val ARTIFACT_DIR = "qnn-htp-yolo-benchmark"
        private const val EVALSET_ROOT = "blindassist_evalset"
        private const val EVALSET_MANIFEST = "$EVALSET_ROOT/manifest.jsonl"
        private const val ARG_IMAGE_LIMIT = "qnnImageLimit"
        private const val ARG_MEASURED_RUNS = "qnnMeasuredRuns"
        private const val ARG_WARMUP_RUNS = "qnnWarmupRuns"
        private const val DEFAULT_IMAGE_LIMIT = 20
        private const val MAX_IMAGE_LIMIT = 100
        private const val DEFAULT_MEASURED_RUNS = 100
        private const val MAX_MEASURED_RUNS = 500
        private const val DEFAULT_WARMUP_RUNS = 10
        private const val MAX_WARMUP_RUNS = 50
        private const val CPU_THREADS = 4
        private const val BOX_CHANNELS = 4
        private const val PREDICTION_COUNT = 2100
        private const val MODEL_TOKEN = "blindassist-yolo11n-fp16-320-qnn-htp-v1"
        private const val QNN_MAVEN_VERSION = "2.34.0"
    }
}
