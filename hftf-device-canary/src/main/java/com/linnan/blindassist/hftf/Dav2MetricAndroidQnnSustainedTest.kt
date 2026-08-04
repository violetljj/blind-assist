package com.linnan.blindassist.hftf

import android.os.Bundle
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.hftf.metricdepth.KnownHeightGroundPipeline
import com.linnan.blindassist.hftf.metricdepth.KnownHeightScaleStudent
import com.qualcomm.qti.QnnDelegate
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
class Dav2MetricAndroidQnnSustainedTest {
    @Test
    fun sustainedCpuBoundaryStagesPerformance() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val corpus = File(requireNotNull(arguments.getString("corpusRoot")))
        val npuOutput = File(requireNotNull(arguments.getString("npuOutputPath")))
        val durationMs = arguments.getString("durationMs")?.toLong() ?: 600_000L
        require(durationMs >= 60_000L)
        val rgb = readNpyBytes(File(corpus, "clean/rgb_640x480_uint8.npy"))
        assertEquals(CAMERA_WIDTH * CAMERA_HEIGHT * 3, rgb.size)
        assertEquals(OUTPUT_COUNT * 4L, npuOutput.length())
        val rawDepth = readRawFloats(npuOutput)
        assertTrue(rawDepth.all { it.isFinite() })

        val stages = linkedMapOf(
            "preprocess" to ArrayList<Double>(),
            "npu_output_copy" to ArrayList(),
            "postprocess" to ArrayList(),
            "ransac_features" to ArrayList(),
            "student" to ArrayList(),
            "cpu_boundary_total" to ArrayList(),
        )
        val samples = JSONArray()
        val report = baseReport("blindassist_dav2_metric_android_qnn_cpu_boundary_sustained_r0")
            .put("runtime", "android-kotlin-cpu-boundaries-around-qairt-htp")
            .put("qairt_online_dlc_sha256", QAIRT_ONLINE_DLC_SHA256)
            .put("qairt_cached_dlc_sha256", QAIRT_CACHED_DLC_SHA256)
            .put("display_condition", arguments.getString("displayCondition") ?: "UNSPECIFIED")
            .put("duration_target_ms", durationMs)
            .put("npu_output_sha256", sha256(npuOutput))
            .put("memory_before", memoryJson())
            .put("thermal_status_before", thermalStatus())
        val startedMs = SystemClock.elapsedRealtime()
        var nextSampleMs = startedMs
        var validPipelineFrames = 0
        while (SystemClock.elapsedRealtime() - startedMs < durationMs) {
            val totalStarted = SystemClock.elapsedRealtimeNanos()
            var stageStarted = totalStarted
            preprocessRgb(rgb)
            stages.getValue("preprocess") += elapsedMs(stageStarted)

            stageStarted = SystemClock.elapsedRealtimeNanos()
            val outputCopy = rawDepth.copyOf()
            stages.getValue("npu_output_copy") += elapsedMs(stageStarted)

            stageStarted = SystemClock.elapsedRealtimeNanos()
            val depth = resizeDepthAlignCorners(outputCopy)
            stages.getValue("postprocess") += elapsedMs(stageStarted)

            stageStarted = SystemClock.elapsedRealtimeNanos()
            val pipeline = KnownHeightGroundPipeline.evaluateGeometry(
                depth, CAMERA_WIDTH, CAMERA_HEIGHT, 320.0, 320.0, 320.0, 240.0, CAMERA_HEIGHT_M,
            )
            stages.getValue("ransac_features") += elapsedMs(stageStarted)

            stageStarted = SystemClock.elapsedRealtimeNanos()
            if (pipeline is KnownHeightGroundPipeline.Geometry) {
                KnownHeightScaleStudent.frozen().predict(pipeline.features)
                validPipelineFrames++
            }
            stages.getValue("student") += elapsedMs(stageStarted)
            stages.getValue("cpu_boundary_total") += elapsedMs(totalStarted)

            val now = SystemClock.elapsedRealtime()
            if (now >= nextSampleMs) {
                samples.put(
                    JSONObject()
                        .put("elapsed_ms", now - startedMs)
                        .put("thermal_status", thermalStatus())
                        .put("memory", memoryJson()),
                )
                nextSampleMs += 30_000L
            }
        }
        val actualDurationMs = SystemClock.elapsedRealtime() - startedMs
        val stageJson = JSONObject()
        for ((name, values) in stages) stageJson.put(name, latencyJson(values))
        report
            .put("duration_actual_ms", actualDurationMs)
            .put("frames", stages.getValue("cpu_boundary_total").size)
            .put("valid_pipeline_frames", validPipelineFrames)
            .put("cpu_boundary_fps", stages.getValue("cpu_boundary_total").size * 1000.0 / actualDurationMs)
            .put("stage_latency_ms", stageJson)
            .put("samples_30s", samples)
            .put("memory_after", memoryJson())
            .put("thermal_status_after", thermalStatus())
        persistReport(CPU_BOUNDARY_REPORT_FILE, report)
        instrumentation.sendStatus(2, Bundle().apply { putString(CPU_BOUNDARY_REPORT_KEY, report.toString()) })
    }

    @Test
    fun qnnProbeAndShortPerformance() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val model = requiredModel()
        val report = baseReport("blindassist_dav2_metric_android_qnn_short_r0")
            .put("memory_before", memoryJson())
        createQnnRuntime(model).use { runtime ->
            report
                .put("qnn_runtime_version", JSONArray(QnnDelegate.getVersion().toList()))
                .put("delegate_init_ms", runtime.initMs)
                .put("input_shape", JSONArray(runtime.interpreter.getInputTensor(0).shape().toList()))
                .put("output_shape", JSONArray(runtime.interpreter.getOutputTensor(0).shape().toList()))
                .put("memory_after_load", memoryJson())
            assertShapes(runtime.interpreter)
            val input = validationInput()
            val output = outputBuffer()
            repeat(SHORT_WARMUP_RUNS) { invoke(runtime.interpreter, input, output) }
            val latencies = ArrayList<Double>()
            repeat(SHORT_MEASURED_RUNS) {
                val started = SystemClock.elapsedRealtimeNanos()
                invoke(runtime.interpreter, input, output)
                latencies += elapsedMs(started)
            }
            report
                .put("warmup_runs", SHORT_WARMUP_RUNS)
                .put("measured_runs", SHORT_MEASURED_RUNS)
                .put("da_latency_ms", latencyJson(latencies, includeAll = true))
                .put("processing_fps", 1000.0 / latencies.average())
                .put("output", outputSummary(output))
                .put("qnn_profiling_result_bytes", runtime.delegate.profilingResult?.size ?: 0)
                .put("memory_after_runs", memoryJson())
                .put("thermal_status", thermalStatus())
        }
        instrumentation.sendStatus(2, Bundle().apply { putString(SHORT_REPORT_KEY, report.toString()) })
    }

    @Test
    fun sustainedFullPipelinePerformance() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val model = requiredModel()
        val corpus = File(requireNotNull(arguments.getString("corpusRoot")))
        val durationMs = arguments.getString("durationMs")?.toLong() ?: 600_000L
        require(durationMs >= 60_000L)
        val rgb = readNpyBytes(File(corpus, "clean/rgb_640x480_uint8.npy"))
        assertEquals(CAMERA_WIDTH * CAMERA_HEIGHT * 3, rgb.size)

        val stages = linkedMapOf(
            "preprocess" to ArrayList<Double>(),
            "da" to ArrayList(),
            "postprocess" to ArrayList(),
            "ransac_features" to ArrayList(),
            "student" to ArrayList(),
            "total" to ArrayList(),
        )
        val samples = JSONArray()
        val report = baseReport("blindassist_dav2_metric_android_qnn_sustained_pipeline_r0")
            .put("duration_target_ms", durationMs)
            .put("preprocess_contract", "rgb640x480_to_nchw518x686_opencv_style_cubic_imagenet")
            .put("memory_before", memoryJson())
            .put("thermal_status_before", thermalStatus())
        createQnnRuntime(model).use { runtime ->
            report
                .put("qnn_runtime_version", JSONArray(QnnDelegate.getVersion().toList()))
                .put("delegate_init_ms", runtime.initMs)
            assertShapes(runtime.interpreter)
            repeat(2) {
                val input = inputBuffer(preprocessRgb(rgb))
                invoke(runtime.interpreter, input, outputBuffer())
            }
            val startedMs = SystemClock.elapsedRealtime()
            var nextSampleMs = startedMs
            var validPipelineFrames = 0
            while (SystemClock.elapsedRealtime() - startedMs < durationMs) {
                val totalStarted = SystemClock.elapsedRealtimeNanos()
                var stageStarted = totalStarted
                val input = inputBuffer(preprocessRgb(rgb))
                stages.getValue("preprocess") += elapsedMs(stageStarted)

                stageStarted = SystemClock.elapsedRealtimeNanos()
                val output = outputBuffer()
                invoke(runtime.interpreter, input, output)
                val rawDepth = floatArray(output)
                stages.getValue("da") += elapsedMs(stageStarted)

                stageStarted = SystemClock.elapsedRealtimeNanos()
                val depth = resizeDepthAlignCorners(rawDepth)
                stages.getValue("postprocess") += elapsedMs(stageStarted)

                stageStarted = SystemClock.elapsedRealtimeNanos()
                val pipeline = KnownHeightGroundPipeline.evaluateGeometry(
                    depth, CAMERA_WIDTH, CAMERA_HEIGHT, 320.0, 320.0, 320.0, 240.0, CAMERA_HEIGHT_M,
                )
                stages.getValue("ransac_features") += elapsedMs(stageStarted)

                stageStarted = SystemClock.elapsedRealtimeNanos()
                if (pipeline is KnownHeightGroundPipeline.Geometry) {
                    KnownHeightScaleStudent.frozen().predict(pipeline.features)
                    validPipelineFrames++
                }
                stages.getValue("student") += elapsedMs(stageStarted)
                stages.getValue("total") += elapsedMs(totalStarted)

                val now = SystemClock.elapsedRealtime()
                if (now >= nextSampleMs) {
                    samples.put(
                        JSONObject()
                            .put("elapsed_ms", now - startedMs)
                            .put("thermal_status", thermalStatus())
                            .put("memory", memoryJson()),
                    )
                    nextSampleMs += 30_000L
                }
            }
            val actualDurationMs = SystemClock.elapsedRealtime() - startedMs
            val stageJson = JSONObject()
            for ((name, values) in stages) stageJson.put(name, latencyJson(values))
            report
                .put("duration_actual_ms", actualDurationMs)
                .put("frames", stages.getValue("total").size)
                .put("valid_pipeline_frames", validPipelineFrames)
                .put("processing_fps", stages.getValue("total").size * 1000.0 / actualDurationMs)
                .put("stage_latency_ms", stageJson)
                .put("samples_30s", samples)
                .put("qnn_profiling_result_bytes", runtime.delegate.profilingResult?.size ?: 0)
                .put("memory_after", memoryJson())
                .put("thermal_status_after", thermalStatus())
        }
        instrumentation.sendStatus(2, Bundle().apply { putString(SUSTAINED_REPORT_KEY, report.toString()) })
    }

    private fun requiredModel(): File {
        val model = File(requireNotNull(InstrumentationRegistry.getArguments().getString("modelPath")))
        assertTrue(model.isFile)
        assertEquals(MODEL_SHA256, sha256(model))
        return model
    }

    private fun baseReport(schema: String): JSONObject = JSONObject()
        .put("schema", schema)
        .put("model_sha256", MODEL_SHA256)
        .put("runtime", "litert-1.4.2-qnn-2.47.0-htp-fp16")
        .put("backend", "HTP_BACKEND")
        .put("precision", "HTP_PRECISION_FP16")

    private fun persistReport(name: String, report: JSONObject) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        File(context.filesDir, name).writeText(report.toString(2))
    }

    private fun createQnnRuntime(model: File): QnnRuntime {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val targetContext = instrumentation.targetContext
        System.loadLibrary("cdsprpc")
        assertTrue(
            "QNN runtime does not report HTP FP16 capability",
            QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16),
        )
        val cacheDir = File(targetContext.codeCacheDir, "dav2-metric-qnn-r0").apply { mkdirs() }
        val options = QnnDelegate.Options().apply {
            setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
            setSkelLibraryDir(targetContext.applicationInfo.nativeLibraryDir)
            setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
            setHtpPerformanceMode(
                QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE,
            )
            setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
            setProfiling(QnnDelegate.Options.ProfilingOptions.DETAILED_PROFILING)
            setCacheDir(cacheDir.absolutePath)
            setModelToken(MODEL_TOKEN)
        }
        val delegate = QnnDelegate(options)
        assertTrue("QNN HTP delegate is unavailable", delegate.isAvailable)
        val started = SystemClock.elapsedRealtimeNanos()
        try {
            val interpreter = Interpreter(
                mapReadOnly(model),
                Interpreter.Options().setNumThreads(CPU_THREADS).addDelegate(delegate),
            )
            interpreter.allocateTensors()
            return QnnRuntime(interpreter, delegate, elapsedMs(started))
        } catch (error: Throwable) {
            delegate.close()
            throw error
        }
    }

    private fun assertShapes(interpreter: Interpreter) {
        assertTrue(interpreter.getInputTensor(0).shape().contentEquals(intArrayOf(1, 3, HEIGHT, WIDTH)))
        assertTrue(interpreter.getOutputTensor(0).shape().contentEquals(intArrayOf(1, HEIGHT, WIDTH)))
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

    private fun invoke(interpreter: Interpreter, input: ByteBuffer, output: ByteBuffer) {
        input.rewind()
        output.rewind()
        interpreter.run(input, output)
    }

    private fun outputSummary(output: ByteBuffer): JSONObject {
        val values = floatArray(output)
        assertTrue(values.all { it.isFinite() })
        return JSONObject().put("minimum_m", values.min()).put("maximum_m", values.max())
    }

    private fun inputBuffer(values: FloatArray): ByteBuffer =
        ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder()).apply {
            asFloatBuffer().put(values)
            rewind()
        }

    private fun outputBuffer(): ByteBuffer =
        ByteBuffer.allocateDirect(OUTPUT_COUNT * 4).order(ByteOrder.nativeOrder())

    private fun floatArray(buffer: ByteBuffer): FloatArray {
        buffer.rewind()
        return FloatArray(OUTPUT_COUNT).also { buffer.asFloatBuffer().get(it) }
    }

    private fun preprocessRgb(rgb: ByteArray): FloatArray {
        val output = FloatArray(INPUT_COUNT)
        val means = doubleArrayOf(0.485, 0.456, 0.406)
        val deviations = doubleArrayOf(0.229, 0.224, 0.225)
        val plane = HEIGHT * WIDTH
        val scaleX = CAMERA_WIDTH.toDouble() / WIDTH
        val scaleY = CAMERA_HEIGHT.toDouble() / HEIGHT
        for (row in 0 until HEIGHT) {
            val sourceY = (row + 0.5) * scaleY - 0.5
            val baseY = kotlin.math.floor(sourceY).toInt()
            for (column in 0 until WIDTH) {
                val sourceX = (column + 0.5) * scaleX - 0.5
                val baseX = kotlin.math.floor(sourceX).toInt()
                for (channel in 0..2) {
                    var weighted = 0.0
                    for (dy in -1..2) {
                        val y = (baseY + dy).coerceIn(0, CAMERA_HEIGHT - 1)
                        val wy = cubicWeight(sourceY - baseY - dy)
                        for (dx in -1..2) {
                            val x = (baseX + dx).coerceIn(0, CAMERA_WIDTH - 1)
                            val pixel = rgb[(y * CAMERA_WIDTH + x) * 3 + channel].toInt() and 0xff
                            weighted += pixel * wy * cubicWeight(sourceX - baseX - dx)
                        }
                    }
                    output[channel * plane + row * WIDTH + column] =
                        ((weighted / 255.0 - means[channel]) / deviations[channel]).toFloat()
                }
            }
        }
        return output
    }

    private fun cubicWeight(value: Double): Double {
        val x = kotlin.math.abs(value)
        val a = -0.75
        return when {
            x <= 1.0 -> (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
            x < 2.0 -> a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
            else -> 0.0
        }
    }

    private fun resizeDepthAlignCorners(input: FloatArray): FloatArray {
        val output = FloatArray(CAMERA_WIDTH * CAMERA_HEIGHT)
        for (row in 0 until CAMERA_HEIGHT) {
            val sourceY = row.toDouble() * (HEIGHT - 1) / (CAMERA_HEIGHT - 1)
            val y0 = sourceY.toInt()
            val y1 = minOf(y0 + 1, HEIGHT - 1)
            val fy = sourceY - y0
            for (column in 0 until CAMERA_WIDTH) {
                val sourceX = column.toDouble() * (WIDTH - 1) / (CAMERA_WIDTH - 1)
                val x0 = sourceX.toInt()
                val x1 = minOf(x0 + 1, WIDTH - 1)
                val fx = sourceX - x0
                val top = input[y0 * WIDTH + x0] * (1.0 - fx) + input[y0 * WIDTH + x1] * fx
                val bottom = input[y1 * WIDTH + x0] * (1.0 - fx) + input[y1 * WIDTH + x1] * fx
                output[row * CAMERA_WIDTH + column] = (top * (1.0 - fy) + bottom * fy).toFloat()
            }
        }
        return output
    }

    private fun readNpyBytes(file: File): ByteArray {
        val bytes = file.readBytes()
        val major = bytes[6].toInt() and 0xff
        val headerLength = if (major == 1) {
            (bytes[8].toInt() and 0xff) or ((bytes[9].toInt() and 0xff) shl 8)
        } else {
            ByteBuffer.wrap(bytes, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int
        }
        val offset = if (major == 1) 10 + headerLength else 12 + headerLength
        return bytes.copyOfRange(offset, bytes.size)
    }

    private fun readRawFloats(file: File): FloatArray {
        val bytes = file.readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / 4).also { buffer.asFloatBuffer().get(it) }
    }

    private fun latencyJson(values: List<Double>, includeAll: Boolean = false): JSONObject {
        val sorted = values.sorted()
        return JSONObject()
            .apply { if (includeAll) put("all", JSONArray(values)) }
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

    private fun thermalStatus(): Int = InstrumentationRegistry.getInstrumentation().targetContext
        .getSystemService(PowerManager::class.java).currentThermalStatus

    private fun mapReadOnly(file: File): ByteBuffer = FileInputStream(file).use { stream ->
        stream.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
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

    private data class QnnRuntime(
        val interpreter: Interpreter,
        val delegate: QnnDelegate,
        val initMs: Double,
    ) : AutoCloseable {
        override fun close() {
            interpreter.close()
            delegate.close()
        }
    }

    private companion object {
        const val SHORT_REPORT_KEY = "dav2_metric_android_qnn_short_report"
        const val SUSTAINED_REPORT_KEY = "dav2_metric_android_qnn_sustained_report"
        const val CPU_BOUNDARY_REPORT_KEY = "dav2_metric_android_qnn_cpu_boundary_report"
        const val CPU_BOUNDARY_REPORT_FILE = "dav2-metric-android-qnn-cpu-boundary-r0.json"
        const val MODEL_SHA256 = "0277FBC74C73D95433B43BEE9D61DD08F1E79B67A2F64A6DA871F3A23FBED8E3"
        const val QAIRT_ONLINE_DLC_SHA256 = "8D2439D1646A3EEF8F74F6C69762A064C9D123775928CA54E5A8441132CEFA3B"
        const val QAIRT_CACHED_DLC_SHA256 = "2BB02F37FEF177FF4B02B8EE0C416EE9FF998BCEEF9786B92959E1F682EBAA24"
        const val MODEL_TOKEN = "blindassist-dav2-metric-hypersim-vits-518x686-qnn-r0"
        const val HEIGHT = 518
        const val WIDTH = 686
        const val INPUT_COUNT = 3 * HEIGHT * WIDTH
        const val OUTPUT_COUNT = HEIGHT * WIDTH
        const val CAMERA_WIDTH = 640
        const val CAMERA_HEIGHT = 480
        const val CAMERA_HEIGHT_M = 1.0341161949454936
        const val CPU_THREADS = 4
        const val SHORT_WARMUP_RUNS = 2
        const val SHORT_MEASURED_RUNS = 10
    }
}
