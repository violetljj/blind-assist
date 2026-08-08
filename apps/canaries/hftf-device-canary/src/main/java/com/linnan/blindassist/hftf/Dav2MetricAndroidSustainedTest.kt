package com.linnan.blindassist.hftf

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
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
import java.nio.FloatBuffer
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class Dav2MetricAndroidSustainedTest {
    @Test
    fun sustainedFullPipelinePerformance() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val model = File(requireNotNull(arguments.getString("modelPath")))
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
        val environment = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val report = JSONObject()
            .put("schema", "blindassist_dav2_metric_android_sustained_pipeline_r0")
            .put("duration_target_ms", durationMs)
            .put("runtime", "onnxruntime-android-1.26.0-cpu-4threads")
            .put("preprocess_contract", "rgb640x480_to_nchw518x686_opencv_style_cubic_imagenet")
            .put("memory_before", memoryJson())
            .put("thermal_status_before", thermalStatus())
        try {
            environment.createSession(model.absolutePath, options).use { session ->
                repeat(2) {
                    val input = preprocessRgb(rgb)
                    OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), INPUT_SHAPE).use { tensor ->
                        session.run(mapOf("image" to tensor)).close()
                    }
                }
                val startedMs = SystemClock.elapsedRealtime()
                var nextSampleMs = startedMs
                var validPipelineFrames = 0
                while (SystemClock.elapsedRealtime() - startedMs < durationMs) {
                    val totalStarted = SystemClock.elapsedRealtimeNanos()
                    var stageStarted = totalStarted
                    val input = preprocessRgb(rgb)
                    stages.getValue("preprocess") += elapsedMs(stageStarted)

                    stageStarted = SystemClock.elapsedRealtimeNanos()
                    val rawDepth = OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), INPUT_SHAPE).use { tensor ->
                        session.run(mapOf("image" to tensor)).use { result ->
                            val values = (result[0] as OnnxTensor).floatBuffer
                            FloatArray(values.capacity()) { values[it] }
                        }
                    }
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
                    .put("memory_after", memoryJson())
                    .put("thermal_status_after", thermalStatus())
            }
        } finally {
            options.close()
        }
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
    }

    private fun preprocessRgb(rgb: ByteArray): FloatArray {
        val output = FloatArray(3 * HEIGHT * WIDTH)
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

    private fun latencyJson(values: List<Double>): JSONObject {
        val sorted = values.sorted()
        return JSONObject()
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

    private fun thermalStatus(): Int {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        return context.getSystemService(PowerManager::class.java).currentThermalStatus
    }

    private fun elapsedMs(start: Long) = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0

    private companion object {
        const val REPORT_KEY = "dav2_metric_android_sustained_pipeline_report"
        const val HEIGHT = 518
        const val WIDTH = 686
        const val CAMERA_WIDTH = 640
        const val CAMERA_HEIGHT = 480
        const val CAMERA_HEIGHT_M = 1.0341161949454936
        val INPUT_SHAPE = longArrayOf(1, 3, HEIGHT.toLong(), WIDTH.toLong())
    }
}
