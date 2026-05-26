package com.linnan.blindassist.benchmark

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.vision.ImagePreprocessor
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import kotlin.math.roundToInt

@RunWith(AndroidJUnit4::class)
class Yolo26nDeviceBenchmarkTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun benchmarkYolo26nWithoutChangingDefaultModel() {
        val imageLimit = arguments.getString(ARG_IMAGE_LIMIT)?.toIntOrNull() ?: DEFAULT_IMAGE_LIMIT
        val pureWarmup = arguments.getString(ARG_PURE_WARMUP)?.toIntOrNull() ?: DEFAULT_PURE_WARMUP
        val pureRuns = arguments.getString(ARG_PURE_RUNS)?.toIntOrNull() ?: DEFAULT_PURE_RUNS
        val appRunsPerImage = arguments.getString(ARG_APP_RUNS_PER_IMAGE)?.toIntOrNull() ?: DEFAULT_APP_RUNS_PER_IMAGE

        assertEquals("yolo11n_fp16_320.tflite", TfliteYoloDetector.MODEL_ASSET)
        assertTrue("yolo26n test asset is missing", assetExists(YOLO26N_ASSET))
        assertTrue("coco labels test asset is missing", assetExists(TfliteYoloDetector.LABELS_ASSET))

        val imageAssets = testContext.assets.list(COCO_IMAGES_DIR)
            ?.filter { it.endsWith(".jpg", ignoreCase = true) }
            ?.sorted()
            ?.take(imageLimit)
            .orEmpty()
        assertTrue("Expected at least one COCO100 image asset", imageAssets.isNotEmpty())

        val artifactDir = createArtifactDir(targetContext)
        val pure = runPureInterpreterBenchmark(imageAssets, pureWarmup, pureRuns)
        val app = runAppDetectorBenchmark(imageAssets, appRunsPerImage)
        assertFalse("App-link benchmark produced failures", app.failures.isNotEmpty())

        val payload = JSONObject()
            .put("created_at_utc", utcNow())
            .put("model", YOLO26N_ASSET)
            .put("default_model_asset", TfliteYoloDetector.MODEL_ASSET)
            .put("image_count", imageAssets.size)
            .put("image_limit", imageLimit)
            .put("pure_interpreter", pure.toJson())
            .put("app_detector", app.toJson())
            .put("asset_sources", JSONObject()
                .put("model", ".downloads/detector-lab/exports/yolo26n_fp16_320.tflite")
                .put("images", ".downloads/detector-lab/datasets/coco100/images")
                .put("labels", "app/src/main/assets/coco_labels.txt"))

        val jsonFile = File(artifactDir, "benchmark.json")
        jsonFile.writeText(payload.toString(2), Charsets.UTF_8)
        File(artifactDir, "benchmark.md").writeText(markdownReport(payload), Charsets.UTF_8)
        File(artifactDir, "image-assets.txt").writeText(
            imageAssets.joinToString(separator = "\n", postfix = "\n") { "$COCO_IMAGES_DIR/$it" },
            Charsets.UTF_8
        )

        Log.i(TAG, "artifactDir=${artifactDir.absolutePath}")
        Log.i(TAG, "pure_p50=${pure.invokeStats.p50Ms}ms pure_p95=${pure.invokeStats.p95Ms}ms")
        Log.i(TAG, "app_total_p50=${app.totalStats.p50Ms}ms app_infer_p50=${app.inferenceStats.p50Ms}ms")
    }

    private fun runPureInterpreterBenchmark(
        imageAssets: List<String>,
        warmup: Int,
        runs: Int
    ): PureBenchmarkResult {
        val preprocessor = ImagePreprocessor(TfliteYoloDetector.INPUT_SIZE)
        val model = loadMappedAsset(YOLO26N_ASSET)
        val options = Interpreter.Options().setNumThreads(4)
        val initStart = System.nanoTime()
        val interpreter = Interpreter(model, options)
        val initMs = elapsedMs(initStart)
        interpreter.allocateTensors()

        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)
        val inputShape = inputTensor.shape().toList()
        val outputShape = outputTensor.shape().toList()
        assertEquals(intArrayOf(1, TfliteYoloDetector.INPUT_SIZE, TfliteYoloDetector.INPUT_SIZE, 3).toList(), inputShape)
        assertEquals(3, outputShape.size)

        val outputBuffer = ByteBuffer
            .allocateDirect(outputTensor.numBytes())
            .order(ByteOrder.nativeOrder())
        val timings = mutableListOf<Double>()
        repeat(warmup) { index ->
            decodeBitmap("$COCO_IMAGES_DIR/${imageAssets[index % imageAssets.size]}").use { bitmap ->
                val input = preprocessor.prepare(bitmap)
                outputBuffer.rewind()
                interpreter.run(input.buffer, outputBuffer)
            }
        }
        repeat(runs) { index ->
            decodeBitmap("$COCO_IMAGES_DIR/${imageAssets[index % imageAssets.size]}").use { bitmap ->
                val input = preprocessor.prepare(bitmap)
                outputBuffer.rewind()
                val start = System.nanoTime()
                interpreter.run(input.buffer, outputBuffer)
                timings += elapsedMs(start)
            }
        }
        interpreter.close()
        return PureBenchmarkResult(
            backend = "tensorflow-lite-cpu-4threads",
            initMs = round3(initMs),
            warmup = warmup,
            runs = runs,
            inputShape = inputShape,
            outputShape = outputShape,
            invokeStats = Stats.from(timings)
        )
    }

    private fun runAppDetectorBenchmark(
        imageAssets: List<String>,
        runsPerImage: Int
    ): AppBenchmarkResult {
        val detector = TfliteYoloDetector(
            context = testContext,
            modelAssetName = YOLO26N_ASSET,
            labelsAssetName = TfliteYoloDetector.LABELS_ASSET
        )
        assertTrue("yolo26n detector failed to initialize: ${detector.statusMessage}", detector.isReady)
        val readyModelStatus = detector.statusMessage

        val preprocess = mutableListOf<Double>()
        val inference = mutableListOf<Double>()
        val postprocess = mutableListOf<Double>()
        val total = mutableListOf<Double>()
        val detections = mutableListOf<Int>()
        val failures = mutableListOf<String>()

        try {
            imageAssets.forEach { image ->
                repeat(runsPerImage) {
                    try {
                        decodeBitmap("$COCO_IMAGES_DIR/$image").use { bitmap ->
                            val result = detector.detect(bitmap)
                            preprocess += detector.lastPreprocessMs.toDouble()
                            inference += detector.lastInferenceMs.toDouble()
                            postprocess += detector.lastPostprocessMs.toDouble()
                            total += detector.lastTotalDetectMs.toDouble()
                            detections += result.detections.size
                        }
                    } catch (error: Throwable) {
                        failures += "$image: ${error.javaClass.simpleName}: ${error.message}"
                    }
                }
            }
        } finally {
            detector.close()
        }

        return AppBenchmarkResult(
            runs = imageAssets.size * runsPerImage,
            runsPerImage = runsPerImage,
            modelStatus = readyModelStatus,
            preprocessStats = Stats.from(preprocess),
            inferenceStats = Stats.from(inference),
            postprocessStats = Stats.from(postprocess),
            totalStats = Stats.from(total),
            detectionCountStats = Stats.from(detections.map { it.toDouble() }),
            failures = failures
        )
    }

    private fun loadMappedAsset(assetName: String): MappedByteBuffer {
        val descriptor = testContext.assets.openFd(assetName)
        FileInputStream(descriptor.fileDescriptor).use { stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength
            )
        }
    }

    private fun decodeBitmap(assetName: String): Bitmap {
        testContext.assets.open(assetName).use { stream ->
            return requireNotNull(BitmapFactory.decodeStream(stream)) {
                "Failed to decode asset bitmap: $assetName"
            }
        }
    }

    private fun assetExists(assetName: String): Boolean {
        return try {
            testContext.assets.open(assetName).close()
            true
        } catch (_: Throwable) {
            false
        }
    }

    private fun createArtifactDir(context: Context): File {
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "yolo26n-benchmark")
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        return File(root, stamp).also { check(it.mkdirs() || it.isDirectory) }
    }

    private fun markdownReport(payload: JSONObject): String {
        val pure = payload.getJSONObject("pure_interpreter")
        val app = payload.getJSONObject("app_detector")
        val pureStats = pure.getJSONObject("invoke_ms")
        val appInfer = app.getJSONObject("inference_ms")
        val appTotal = app.getJSONObject("total_ms")
        return buildString {
            appendLine("# yolo26n Device Benchmark")
            appendLine()
            appendLine("- Model: `${payload.getString("model")}`")
            appendLine("- Default app model remains: `${payload.getString("default_model_asset")}`")
            appendLine("- Image count: `${payload.getInt("image_count")}`")
            appendLine()
            appendLine("| Path | Runs | P50 ms | P95 ms | Min ms | Max ms |")
            appendLine("| --- | ---: | ---: | ---: | ---: | ---: |")
            appendLine("| Pure TFLite invoke | ${pure.getInt("runs")} | ${pureStats.getDouble("p50")} | ${pureStats.getDouble("p95")} | ${pureStats.getDouble("min")} | ${pureStats.getDouble("max")} |")
            appendLine("| App detector inference | ${app.getInt("runs")} | ${appInfer.getDouble("p50")} | ${appInfer.getDouble("p95")} | ${appInfer.getDouble("min")} | ${appInfer.getDouble("max")} |")
            appendLine("| App detector total | ${app.getInt("runs")} | ${appTotal.getDouble("p50")} | ${appTotal.getDouble("p95")} | ${appTotal.getDouble("min")} | ${appTotal.getDouble("max")} |")
            appendLine()
            appendLine("Notes:")
            appendLine("- Pure TFLite invoke uses CPU with 4 threads because no standalone Android benchmark_model binary is bundled.")
            appendLine("- App detector path uses the BlindAssist preprocessor, TfliteYoloDetector output parsing, and NMS.")
            appendLine("- yolo26n is packaged only in the instrumentation test APK assets.")
        }
    }

    private fun elapsedMs(startNanos: Long): Double {
        return (System.nanoTime() - startNanos) / 1_000_000.0
    }

    private fun utcNow(): String {
        val formatter = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
        formatter.timeZone = TimeZone.getTimeZone("UTC")
        return formatter.format(Date())
    }

    private fun round3(value: Double): Double {
        return (value * 1000.0).roundToInt() / 1000.0
    }

    private fun Bitmap.use(block: (Bitmap) -> Unit) {
        try {
            block(this)
        } finally {
            recycle()
        }
    }

    private data class PureBenchmarkResult(
        val backend: String,
        val initMs: Double,
        val warmup: Int,
        val runs: Int,
        val inputShape: List<Int>,
        val outputShape: List<Int>,
        val invokeStats: Stats
    ) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("backend", backend)
                .put("init_ms", initMs)
                .put("warmup", warmup)
                .put("runs", runs)
                .put("input_shape", JSONArray(inputShape))
                .put("output_shape", JSONArray(outputShape))
                .put("invoke_ms", invokeStats.toJson())
        }
    }

    private data class AppBenchmarkResult(
        val runs: Int,
        val runsPerImage: Int,
        val modelStatus: String,
        val preprocessStats: Stats,
        val inferenceStats: Stats,
        val postprocessStats: Stats,
        val totalStats: Stats,
        val detectionCountStats: Stats,
        val failures: List<String>
    ) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("runs", runs)
                .put("runs_per_image", runsPerImage)
                .put("model_status", modelStatus)
                .put("preprocess_ms", preprocessStats.toJson())
                .put("inference_ms", inferenceStats.toJson())
                .put("postprocess_ms", postprocessStats.toJson())
                .put("total_ms", totalStats.toJson())
                .put("detection_counts", detectionCountStats.toJson())
                .put("failures", JSONArray(failures))
        }
    }

    private data class Stats(
        val count: Int,
        val meanMs: Double,
        val p50Ms: Double,
        val p95Ms: Double,
        val minMs: Double,
        val maxMs: Double
    ) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("count", count)
                .put("mean", meanMs)
                .put("p50", p50Ms)
                .put("p95", p95Ms)
                .put("min", minMs)
                .put("max", maxMs)
        }

        companion object {
            fun from(values: List<Double>): Stats {
                if (values.isEmpty()) {
                    return Stats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
                }
                val sorted = values.sorted()
                return Stats(
                    count = values.size,
                    meanMs = round3Value(values.average()),
                    p50Ms = round3Value(percentile(sorted, 0.50)),
                    p95Ms = round3Value(percentile(sorted, 0.95)),
                    minMs = round3Value(sorted.first()),
                    maxMs = round3Value(sorted.last())
                )
            }

            private fun percentile(sorted: List<Double>, fraction: Double): Double {
                val index = ((sorted.size - 1) * fraction).roundToInt().coerceIn(0, sorted.lastIndex)
                return sorted[index]
            }

            private fun round3Value(value: Double): Double {
                return (value * 1000.0).roundToInt() / 1000.0
            }
        }
    }

    companion object {
        private const val TAG = "Yolo26nBenchmark"
        private const val YOLO26N_ASSET = "yolo26n_fp16_320.tflite"
        private const val COCO_IMAGES_DIR = "images"
        private const val ARG_IMAGE_LIMIT = "imageLimit"
        private const val ARG_PURE_WARMUP = "pureWarmup"
        private const val ARG_PURE_RUNS = "pureRuns"
        private const val ARG_APP_RUNS_PER_IMAGE = "appRunsPerImage"
        private const val DEFAULT_IMAGE_LIMIT = 100
        private const val DEFAULT_PURE_WARMUP = 10
        private const val DEFAULT_PURE_RUNS = 100
        private const val DEFAULT_APP_RUNS_PER_IMAGE = 1
    }
}
