package com.linnan.blindassist.benchmark

import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.vision.ImagePreprocessor
import com.linnan.blindassist.vision.RgbaVisionFrame
import com.qualcomm.qti.QnnDelegate
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.ceil

/** Candidate-only comparison of CPU preprocessing and an equivalent QNN preprocessing graph. */
@RunWith(AndroidJUnit4::class)
class QnnPreprocessFusionDeviceBenchmarkTest {
    @Test
    fun compareCpuAndQnnPreprocessingOnSameRgbaInput() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val testContext = instrumentation.context
        val targetContext = instrumentation.targetContext
        val artifactDir = File(targetContext.filesDir, ARTIFACT_DIR).apply { mkdirs() }
        val reportFile = File(artifactDir, REPORT_FILENAME)
        var qnnDelegate: QnnDelegate? = null
        var interpreter: Interpreter? = null

        try {
            System.loadLibrary("cdsprpc")
            assertTrue(
                "QNN HTP FP16 capability is unavailable",
                QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)
            )
            val options = QnnDelegate.Options().apply {
                setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                setSkelLibraryDir(targetContext.applicationInfo.nativeLibraryDir)
                setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
                setHtpPerformanceMode(
                    QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE
                )
                setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
                setProfiling(QnnDelegate.Options.ProfilingOptions.DETAILED_PROFILING)
                setCacheDir(targetContext.codeCacheDir.absolutePath)
                setModelToken(MODEL_TOKEN)
            }
            val delegate = QnnDelegate(options)
            qnnDelegate = delegate
            assertTrue("QNN preprocessing delegate is unavailable", delegate.isAvailable)

            val initializationStartedAt = System.nanoTime()
            val localInterpreter = Interpreter(
                loadMappedAsset(testContext, MODEL_ASSET),
                Interpreter.Options().setNumThreads(CPU_THREADS).addDelegate(delegate)
            )
            interpreter = localInterpreter
            localInterpreter.allocateTensors()
            val initializationMs = elapsedMillis(initializationStartedAt)
            val inputTensor = localInterpreter.getInputTensor(0)
            val outputTensor = localInterpreter.getOutputTensor(0)
            assertTrue(
                "unexpected preprocessing input: ${inputTensor.shape().contentToString()} ${inputTensor.dataType()}",
                inputTensor.dataType() == DataType.UINT8 &&
                    inputTensor.shape().contentEquals(intArrayOf(1, SOURCE_HEIGHT, SOURCE_WIDTH, RGBA_CHANNELS))
            )
            assertTrue(
                "unexpected preprocessing output: ${outputTensor.shape().contentToString()} ${outputTensor.dataType()}",
                outputTensor.dataType() == DataType.FLOAT32 &&
                    outputTensor.shape().contentEquals(intArrayOf(1, INPUT_SIZE, INPUT_SIZE, RGB_CHANNELS))
            )

            val frame = SyntheticRgbaFrame()
            val preprocessor = ImagePreprocessor(INPUT_SIZE)
            val qnnOutput = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())

            repeat(WARMUP_ITERATIONS) {
                preprocessor.prepare(frame)
                invoke(localInterpreter, frame.buffer, qnnOutput)
            }

            val cpuSamples = LongArray(MEASURED_ITERATIONS)
            val qnnSamples = LongArray(MEASURED_ITERATIONS)
            repeat(MEASURED_ITERATIONS) { index ->
                var startedAt = System.nanoTime()
                preprocessor.prepare(frame)
                cpuSamples[index] = System.nanoTime() - startedAt

                startedAt = System.nanoTime()
                invoke(localInterpreter, frame.buffer, qnnOutput)
                qnnSamples[index] = System.nanoTime() - startedAt
            }

            val cpuValues = floats(preprocessor.prepare(frame).buffer)
            invoke(localInterpreter, frame.buffer, qnnOutput)
            val qnnValues = floats(qnnOutput)
            val parity = compare(cpuValues, qnnValues)
            val profiling = delegate.profilingResult
            val report = JSONObject()
                .put("schema", "blindassist_qnn_preprocess_fusion_benchmark_v1")
                .put("disposition", "CANDIDATE_ONLY")
                .put("device", JSONObject()
                    .put("model", Build.MODEL)
                    .put("soc_model", Build.SOC_MODEL)
                    .put("sdk_int", Build.VERSION.SDK_INT))
                .put("model_asset", MODEL_ASSET)
                .put("model_token", MODEL_TOKEN)
                .put("input_shape", JSONArray(inputTensor.shape().toList()))
                .put("output_shape", JSONArray(outputTensor.shape().toList()))
                .put("warmup_iterations", WARMUP_ITERATIONS)
                .put("measured_iterations", MEASURED_ITERATIONS)
                .put("initialization_ms", initializationMs)
                .put("qnn_profiling_result_bytes", profiling?.size ?: 0)
                .put("cpu_preprocess_us", stats(cpuSamples))
                .put("qnn_preprocess_us", stats(qnnSamples))
                .put("parity", JSONObject()
                    .put("value_count", parity.valueCount)
                    .put("max_abs", parity.maxAbs)
                    .put("mean_abs", parity.meanAbs)
                    .put("acceptance_max_abs", MAX_ACCEPTED_ABS_ERROR)
                    .put("accepted", parity.maxAbs <= MAX_ACCEPTED_ABS_ERROR))
            reportFile.writeText(report.toString(2), Charsets.UTF_8)
            if (profiling != null && profiling.isNotEmpty()) {
                File(artifactDir, PROFILE_FILENAME).writeBytes(profiling)
            }
            assertTrue(
                "QNN preprocessing diverged from CPU: maxAbs=${parity.maxAbs}",
                parity.maxAbs <= MAX_ACCEPTED_ABS_ERROR
            )
        } catch (error: Throwable) {
            if (!reportFile.exists()) {
                reportFile.writeText(
                    JSONObject()
                        .put("schema", "blindassist_qnn_preprocess_fusion_benchmark_v1")
                        .put("disposition", "QNN_PREPROCESS_CANDIDATE_FAILED")
                        .put("error_class", error.javaClass.name)
                        .put("error_message", error.message ?: JSONObject.NULL)
                        .toString(2),
                    Charsets.UTF_8
                )
            }
            throw error
        } finally {
            interpreter?.close()
            qnnDelegate?.close()
        }
    }

    private fun invoke(interpreter: Interpreter, input: ByteBuffer, output: ByteBuffer) {
        input.rewind()
        output.rewind()
        interpreter.run(input, output)
        output.rewind()
    }

    private fun floats(buffer: ByteBuffer): FloatArray {
        val source = buffer.duplicate().order(ByteOrder.nativeOrder()).also { it.rewind() }
        return FloatArray(source.remaining() / Float.SIZE_BYTES).also {
            source.asFloatBuffer().get(it)
        }
    }

    private fun compare(reference: FloatArray, candidate: FloatArray): Parity {
        require(reference.size == candidate.size)
        var maxAbs = 0.0
        var sumAbs = 0.0
        reference.indices.forEach { index ->
            val difference = abs(reference[index].toDouble() - candidate[index].toDouble())
            maxAbs = maxOf(maxAbs, difference)
            sumAbs += difference
        }
        return Parity(
            valueCount = reference.size,
            maxAbs = maxAbs,
            meanAbs = if (reference.isEmpty()) 0.0 else sumAbs / reference.size
        )
    }

    private fun stats(samples: LongArray): JSONObject {
        val ordered = samples.sortedArray()
        return JSONObject()
            .put("p50", percentileMicros(ordered, 50.0))
            .put("p95", percentileMicros(ordered, 95.0))
            .put("p99", percentileMicros(ordered, 99.0))
            .put("max", ordered.last().toDouble() / NANOS_PER_MICROSECOND)
    }

    private fun percentileMicros(ordered: LongArray, percentile: Double): Double {
        val index = ceil(ordered.size * percentile / 100.0).toInt().coerceIn(1, ordered.size) - 1
        return ordered[index].toDouble() / NANOS_PER_MICROSECOND
    }

    private fun elapsedMillis(startedAt: Long): Double =
        (System.nanoTime() - startedAt) / NANOS_PER_MILLISECOND

    private fun loadMappedAsset(context: android.content.Context, assetName: String): MappedByteBuffer {
        val descriptor = context.assets.openFd(assetName)
        descriptor.use {
            FileInputStream(it.fileDescriptor).channel.use { channel ->
                return channel.map(FileChannel.MapMode.READ_ONLY, it.startOffset, it.declaredLength)
            }
        }
    }

    private data class Parity(
        val valueCount: Int,
        val maxAbs: Double,
        val meanAbs: Double
    )

    private class SyntheticRgbaFrame : RgbaVisionFrame {
        override val width: Int = SOURCE_WIDTH
        override val height: Int = SOURCE_HEIGHT
        override val rotationDegrees: Int = ROTATION_DEGREES
        override val rowStride: Int = SOURCE_WIDTH * RGBA_CHANNELS
        override val pixelStride: Int = RGBA_CHANNELS
        override val buffer: ByteBuffer = ByteBuffer
            .allocateDirect(rowStride * height)
            .order(ByteOrder.nativeOrder())
            .also { target ->
                for (y in 0 until height) {
                    for (x in 0 until width) {
                        val offset = y * rowStride + x * pixelStride
                        target.put(offset, ((x * 31 + y * 7) and 0xFF).toByte())
                        target.put(offset + 1, ((x * 13 + y * 29) and 0xFF).toByte())
                        target.put(offset + 2, ((x * 19 + y * 17) and 0xFF).toByte())
                        target.put(offset + 3, 0xFF.toByte())
                    }
                }
                target.rewind()
            }

        override fun close() = Unit
    }

    private companion object {
        const val MODEL_ASSET = "qnn_preprocess/rgba640x480_rot90_letterbox320.tflite"
        const val MODEL_TOKEN = "blindassist_qnn_preprocess_rgba640_rot90_v1"
        const val ARTIFACT_DIR = "qnn-preprocess-fusion"
        const val REPORT_FILENAME = "benchmark.json"
        const val PROFILE_FILENAME = "qnn-profile.bin"
        const val SOURCE_WIDTH = 640
        const val SOURCE_HEIGHT = 480
        const val ROTATION_DEGREES = 90
        const val INPUT_SIZE = 320
        const val RGBA_CHANNELS = 4
        const val RGB_CHANNELS = 3
        const val CPU_THREADS = 4
        const val WARMUP_ITERATIONS = 200
        const val MEASURED_ITERATIONS = 1_000
        const val NANOS_PER_MICROSECOND = 1_000.0
        const val NANOS_PER_MILLISECOND = 1_000_000.0
        const val MAX_ACCEPTED_ABS_ERROR = 0.001
    }
}
