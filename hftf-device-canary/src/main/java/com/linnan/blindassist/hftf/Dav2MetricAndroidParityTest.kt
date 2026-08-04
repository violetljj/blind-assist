package com.linnan.blindassist.hftf

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.Bundle
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.hftf.metricdepth.KnownHeightGroundPipeline
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class Dav2MetricAndroidParityTest {
    @Test
    fun frozenStressCorpusMatchesPytorchAndPipelineReferences() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val model = File(requireNotNull(arguments.getString("modelPath")))
        val corpus = File(requireNotNull(arguments.getString("corpusRoot")))
        assertTrue(model.isFile)
        assertTrue(corpus.isDirectory)
        val environment = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val report = JSONObject()
            .put("schema", "blindassist_dav2_metric_android_parity_r0")
            .put("neural", JSONArray())
            .put("downstream", JSONArray())
        try {
            environment.createSession(model.absolutePath, options).use { session ->
                Log.i(TAG, "session_created")
                for (scenario in NEURAL_SCENARIOS) {
                    Log.i(TAG, "neural_start=$scenario")
                    val root = File(corpus, scenario)
                    val input = readNpyFloat(File(root, "normalized_nchw_fp32_1x3x518x686.npy"))
                    val reference = readNpyFloat(File(root, "pytorch_fp32_raw_depth_1x518x686.npy"))
                    Log.i(TAG, "npy_loaded=$scenario")
                    assertEquals(INPUT_COUNT, input.size)
                    assertEquals(OUTPUT_COUNT, reference.size)
                    val actual = OnnxTensor.createTensor(
                        environment,
                        FloatBuffer.wrap(input),
                        longArrayOf(1, 3, HEIGHT.toLong(), WIDTH.toLong()),
                    ).use { tensor ->
                        session.run(mapOf("image" to tensor)).use { result ->
                            Log.i(TAG, "ort_complete=$scenario")
                            val output = result[0] as OnnxTensor
                            val values = output.floatBuffer
                            FloatArray(values.capacity()) { values[it] }
                        }
                    }
                    val metrics = errorMetrics(actual, reference).put("scenario", scenario)
                    Log.i(TAG, "metrics_complete=$scenario")
                    report.getJSONArray("neural").put(metrics)
                    assertTrue("$scenario mean error", metrics.getDouble("mean_abs_error_m") <= MAX_MEAN_ABS_M)
                    assertTrue("$scenario p95 error", metrics.getDouble("p95_abs_error_m") <= MAX_P95_ABS_M)
                    assertTrue("$scenario max error", metrics.getDouble("max_abs_error_m") <= MAX_ABS_M)
                    Log.i(TAG, "neural_complete=$scenario metrics=$metrics")
                }
            }
        } finally {
            options.close()
        }

        for (reference in DOWNSTREAM) {
            Log.i(TAG, "downstream_start=${reference.id}")
            val depth = readNpyFloat(File(corpus, "downstream/${reference.file}"))
            val result = KnownHeightGroundPipeline.evaluate(
                depth = depth,
                width = 640,
                height = 480,
                fx = 320.0,
                fy = 320.0,
                cx = 320.0,
                cy = 240.0,
                cameraHeightM = 1.0341161949454936,
            )
            assertTrue("${reference.id} unexpectedly rejected: $result", result is KnownHeightGroundPipeline.Valid)
            result as KnownHeightGroundPipeline.Valid
            val relativeHeightError = kotlin.math.abs(result.relativeHeight - reference.relativeHeight)
            val studentScaleError = kotlin.math.abs(result.studentScale - reference.studentScale)
            report.getJSONArray("downstream").put(
                JSONObject()
                    .put("scenario", reference.id)
                    .put("status", "VALID")
                    .put("relative_height", result.relativeHeight)
                    .put("relative_height_abs_error", relativeHeightError)
                    .put("inlier_fraction", result.inlierFraction)
                    .put("student_scale", result.studentScale)
                    .put("student_scale_abs_error", studentScaleError),
            )
            assertTrue("${reference.id} relative height drift", relativeHeightError <= MAX_RELATIVE_HEIGHT_ERROR)
            assertTrue("${reference.id} student drift", studentScaleError <= MAX_STUDENT_SCALE_ERROR)
            Log.i(TAG, "downstream_complete=${reference.id}")
        }
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
    }

    private fun errorMetrics(actual: FloatArray, expected: FloatArray): JSONObject {
        assertEquals(expected.size, actual.size)
        val errors = DoubleArray(actual.size)
        var sum = 0.0
        for (index in actual.indices) {
            assertTrue(actual[index].isFinite())
            val error = kotlin.math.abs(actual[index].toDouble() - expected[index])
            errors[index] = error
            sum += error
        }
        errors.sort()
        return JSONObject()
            .put("element_count", errors.size)
            .put("mean_abs_error_m", sum / errors.size)
            .put("p95_abs_error_m", errors[(0.95 * (errors.size - 1)).toInt()])
            .put("max_abs_error_m", errors.last())
    }

    private fun readNpyFloat(file: File): FloatArray {
        val bytes = file.readBytes()
        assertTrue("invalid npy ${file.absolutePath}", bytes.size > 12)
        assertEquals(0x93, bytes[0].toInt() and 0xff)
        val major = bytes[6].toInt() and 0xff
        val headerLength = if (major == 1) {
            (bytes[8].toInt() and 0xff) or ((bytes[9].toInt() and 0xff) shl 8)
        } else {
            ByteBuffer.wrap(bytes, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int
        }
        val dataOffset = if (major == 1) 10 + headerLength else 12 + headerLength
        val buffer = ByteBuffer.wrap(bytes, dataOffset, bytes.size - dataOffset)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer()
        return FloatArray(buffer.remaining()).also(buffer::get)
    }

    private data class DownstreamReference(
        val id: String,
        val file: String,
        val relativeHeight: Double,
        val studentScale: Double,
    )

    private companion object {
        const val REPORT_KEY = "dav2_metric_android_parity_report"
        const val TAG = "Dav2MetricParityR0"
        const val HEIGHT = 518
        const val WIDTH = 686
        const val INPUT_COUNT = 3 * HEIGHT * WIDTH
        const val OUTPUT_COUNT = HEIGHT * WIDTH
        const val MAX_MEAN_ABS_M = 0.002
        const val MAX_P95_ABS_M = 0.005
        const val MAX_ABS_M = 0.02
        const val MAX_RELATIVE_HEIGHT_ERROR = 0.10
        const val MAX_STUDENT_SCALE_ERROR = 0.10
        val NEURAL_SCENARIOS = listOf("clean", "gaussian_sigma3", "motion_horizontal_length17")
        val DOWNSTREAM = listOf(
            DownstreamReference("clean", "clean_depth_640x480_fp32.npy", 1.00917506145239, 0.5913180296662874),
            DownstreamReference(
                "lower_roi_full_width_bottom_50pct_nan",
                "lower_roi_full_width_bottom_50pct_nan_depth_640x480_fp32.npy",
                1.0234638534583842,
                0.5288203916595537,
            ),
            DownstreamReference(
                "local_horizontal_linear_amplitude20pct_polarity_p1",
                "local_horizontal_linear_amplitude20pct_polarity_p1_depth_640x480_fp32.npy",
                1.041824169031234,
                0.5422885875475691,
            ),
        )
    }
}
