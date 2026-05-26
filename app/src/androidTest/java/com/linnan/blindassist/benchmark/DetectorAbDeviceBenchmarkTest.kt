package com.linnan.blindassist.benchmark

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskAnalyzer
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
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
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

@RunWith(AndroidJUnit4::class)
class DetectorAbDeviceBenchmarkTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()
    private val riskAnalyzer = RiskAnalyzer()

    @Test
    fun compareYolo11nAndYolo26nOnSameDeviceImagesAndMetrics() {
        val imageLimit = arguments.getString(ARG_IMAGE_LIMIT)?.toIntOrNull() ?: DEFAULT_IMAGE_LIMIT
        val pureWarmup = arguments.getString(ARG_PURE_WARMUP)?.toIntOrNull() ?: DEFAULT_PURE_WARMUP
        val pureRuns = arguments.getString(ARG_PURE_RUNS)?.toIntOrNull() ?: DEFAULT_PURE_RUNS
        val appRunsPerImage = arguments.getString(ARG_APP_RUNS_PER_IMAGE)?.toIntOrNull() ?: DEFAULT_APP_RUNS_PER_IMAGE
        val matchIouThreshold = arguments.getString(ARG_MATCH_IOU_THRESHOLD)?.toFloatOrNull()
            ?: DEFAULT_MATCH_IOU_THRESHOLD

        assertEquals(YOLO11N_ASSET, TfliteYoloDetector.MODEL_ASSET)
        MODEL_SPECS.forEach { spec ->
            assertTrue("${spec.assetName} test asset is missing", assetExists(spec.assetName))
        }
        assertTrue("coco labels test asset is missing", assetExists(TfliteYoloDetector.LABELS_ASSET))
        assertTrue("COCO100 annotation asset is missing", assetExists(COCO_ANNOTATIONS_ASSET))

        val labels = loadLabels()
        val images = loadBenchmarkImages(labels).take(imageLimit)
        assertTrue("Expected at least one annotated COCO100 image asset", images.isNotEmpty())

        val artifactDir = createArtifactDir(targetContext)
        val modelResults = MODEL_SPECS.map { spec ->
            val pure = runPureInterpreterBenchmark(spec, images, pureWarmup, pureRuns)
            val app = runAppDetectorBenchmark(spec, images, appRunsPerImage, matchIouThreshold)
            assertFalse("${spec.id} App detector produced failures", app.failures.isNotEmpty())
            ModelBenchmarkResult(spec, pure, app)
        }

        val payload = JSONObject()
            .put("created_at_utc", utcNow())
            .put("device_under_test", "instrumentation-connected-device")
            .put("default_model_asset", TfliteYoloDetector.MODEL_ASSET)
            .put("image_count", images.size)
            .put("image_limit", imageLimit)
            .put("app_runs_per_image", appRunsPerImage)
            .put("match_iou_threshold", matchIouThreshold.toDouble())
            .put("confidence_threshold", TfliteYoloDetector.CONFIDENCE_THRESHOLD.toDouble())
            .put("detector_iou_threshold", TfliteYoloDetector.IOU_THRESHOLD.toDouble())
            .put("models", JSONArray(modelResults.map { it.toJson() }))
            .put("recommendation", recommendation(modelResults))
            .put("asset_sources", JSONObject()
                .put("yolo11n", "app/src/main/assets/yolo11n_fp16_320.tflite")
                .put("yolo26n", ".downloads/detector-lab/exports/yolo26n_fp16_320.tflite")
                .put("images", ".downloads/detector-lab/datasets/coco100/images")
                .put("annotations", ".downloads/detector-lab/datasets/coco100/coco100_annotations.json")
                .put("labels", "app/src/main/assets/coco_labels.txt"))

        File(artifactDir, "benchmark.json").writeText(payload.toString(2), Charsets.UTF_8)
        File(artifactDir, "benchmark.md").writeText(markdownReport(payload), Charsets.UTF_8)
        File(artifactDir, "per-image.csv").writeText(perImageCsv(modelResults), Charsets.UTF_8)
        File(artifactDir, "false-positives.json").writeText(
            JSONArray(modelResults.flatMap { it.app.falsePositives }).toString(2),
            Charsets.UTF_8
        )
        File(artifactDir, "false-negatives.json").writeText(
            JSONArray(modelResults.flatMap { it.app.falseNegatives }).toString(2),
            Charsets.UTF_8
        )
        File(artifactDir, "risk-mismatches.json").writeText(
            JSONArray(modelResults.flatMap { it.app.riskMismatches }).toString(2),
            Charsets.UTF_8
        )
        File(artifactDir, "image-assets.txt").writeText(
            images.joinToString(separator = "\n", postfix = "\n") { it.relativePath },
            Charsets.UTF_8
        )

        Log.i(TAG, "artifactDir=${artifactDir.absolutePath}")
        modelResults.forEach { result ->
            Log.i(
                TAG,
                "${result.spec.id} p50=${result.app.totalStats.p50Ms}ms " +
                    "ap50=${result.app.quality.ap50} recall=${result.app.quality.recall}"
            )
        }
    }

    private fun runPureInterpreterBenchmark(
        spec: ModelSpec,
        images: List<BenchmarkImage>,
        warmup: Int,
        runs: Int
    ): PureBenchmarkResult {
        val preprocessor = ImagePreprocessor(TfliteYoloDetector.INPUT_SIZE)
        val model = loadMappedAsset(spec.assetName)
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
            decodeBitmap(images[index % images.size].relativePath).use { bitmap ->
                val input = preprocessor.prepare(bitmap)
                outputBuffer.rewind()
                interpreter.run(input.buffer, outputBuffer)
            }
        }
        repeat(runs) { index ->
            decodeBitmap(images[index % images.size].relativePath).use { bitmap ->
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
        spec: ModelSpec,
        images: List<BenchmarkImage>,
        runsPerImage: Int,
        matchIouThreshold: Float
    ): AppBenchmarkResult {
        val detector = TfliteYoloDetector(
            context = testContext,
            modelAssetName = spec.assetName,
            labelsAssetName = TfliteYoloDetector.LABELS_ASSET
        )
        assertTrue("${spec.id} detector failed to initialize: ${detector.statusMessage}", detector.isReady)
        val readyModelStatus = "ready"

        val preprocess = mutableListOf<Double>()
        val inference = mutableListOf<Double>()
        val postprocess = mutableListOf<Double>()
        val total = mutableListOf<Double>()
        val failures = mutableListOf<String>()
        val perImage = mutableListOf<PerImageModelResult>()
        val firstRunDetectionsByImage = linkedMapOf<String, List<Detection>>()

        try {
            images.forEach { image ->
                val runs = mutableListOf<FrameRun>()
                repeat(runsPerImage) {
                    try {
                        decodeBitmap(image.relativePath).use { bitmap ->
                            val frameSize = FrameSize(bitmap.width, bitmap.height)
                            val result = detector.detect(bitmap)
                            val risk = riskAnalyzer.analyze(result.detections, frameSize)
                            preprocess += detector.lastPreprocessMs.toDouble()
                            inference += detector.lastInferenceMs.toDouble()
                            postprocess += detector.lastPostprocessMs.toDouble()
                            total += detector.lastTotalDetectMs.toDouble()
                            runs += FrameRun(
                                detections = result.detections,
                                risk = risk,
                                preprocessMs = detector.lastPreprocessMs.toDouble(),
                                inferenceMs = detector.lastInferenceMs.toDouble(),
                                postprocessMs = detector.lastPostprocessMs.toDouble(),
                                totalMs = detector.lastTotalDetectMs.toDouble()
                            )
                        }
                    } catch (error: Throwable) {
                        failures += "${image.fileName}: ${error.javaClass.simpleName}: ${error.message}"
                    }
                }

                val firstRun = runs.firstOrNull() ?: FrameRun(emptyList(), noneRisk(), 0.0, 0.0, 0.0, 0.0)
                val gtRisk = riskAnalyzer.analyze(image.groundTruthDetections(), image.frameSize)
                val evaluation = evaluateDetections(
                    image = image,
                    detections = firstRun.detections,
                    matchIouThreshold = matchIouThreshold
                )
                val stability = StabilitySummary.from(runs)
                perImage += PerImageModelResult(image, evaluation, gtRisk, firstRun.risk, stability)
                firstRunDetectionsByImage[image.fileName] = firstRun.detections
            }
        } finally {
            detector.close()
        }

        val quality = aggregateQuality(images, perImage, firstRunDetectionsByImage, matchIouThreshold)
        val riskQuality = aggregateRiskQuality(perImage)
        return AppBenchmarkResult(
            runs = images.size * runsPerImage,
            runsPerImage = runsPerImage,
            modelStatus = readyModelStatus,
            preprocessStats = Stats.from(preprocess),
            inferenceStats = Stats.from(inference),
            postprocessStats = Stats.from(postprocess),
            totalStats = Stats.from(total),
            quality = quality,
            riskQuality = riskQuality,
            stability = StabilityAggregate.from(perImage.map { it.stability }),
            perImage = perImage,
            falsePositives = perImage.flatMap { it.evaluation.falsePositives }.map { it.toJson(spec.id) },
            falseNegatives = perImage.flatMap { it.evaluation.falseNegatives }.map { it.toJson(spec.id) },
            riskMismatches = perImage.mapNotNull { it.riskMismatchJson(spec.id) },
            failures = failures
        )
    }

    private fun loadBenchmarkImages(labels: List<String>): List<BenchmarkImage> {
        val root = JSONObject(readAssetText(COCO_ANNOTATIONS_ASSET))
        val images = root.getJSONArray("images")
        return (0 until images.length()).map { index ->
            val imageJson = images.getJSONObject(index)
            val fileName = imageJson.getString("file_name")
            val frameSize = FrameSize(imageJson.getInt("width"), imageJson.getInt("height"))
            val annotations = imageJson.getJSONArray("annotations")
            val groundTruth = (0 until annotations.length()).mapNotNull { annotationIndex ->
                val annotation = annotations.getJSONObject(annotationIndex)
                val label = annotation.getString("category_name")
                val classId = labels.indexOf(label)
                if (classId < 0) {
                    null
                } else {
                    val xyxy = annotation.getJSONArray("bbox_xyxy")
                    GroundTruthBox(
                        id = annotation.getInt("id"),
                        classId = classId,
                        label = label,
                        box = BoundingBox(
                            left = xyxy.getDouble(0).toFloat(),
                            top = xyxy.getDouble(1).toFloat(),
                            right = xyxy.getDouble(2).toFloat(),
                            bottom = xyxy.getDouble(3).toFloat()
                        ).clamped(frameSize)
                    )
                }
            }
            BenchmarkImage(
                fileName = fileName,
                relativePath = imageJson.getString("relative_path"),
                frameSize = frameSize,
                groundTruth = groundTruth
            )
        }.filter { assetExists(it.relativePath) }
            .sortedBy { it.fileName }
    }

    private fun evaluateDetections(
        image: BenchmarkImage,
        detections: List<Detection>,
        matchIouThreshold: Float
    ): DetectionEvaluation {
        val matchedGt = BooleanArray(image.groundTruth.size)
        var truePositives = 0
        val matchedLabels = mutableListOf<String>()
        val falsePositives = mutableListOf<FalsePositive>()

        detections.sortedByDescending { it.confidence }.forEach { detection ->
            var bestIndex = -1
            var bestIou = 0f
            image.groundTruth.forEachIndexed { index, gt ->
                if (!matchedGt[index] && gt.label == detection.label) {
                    val score = iou(detection.boundingBox, gt.box)
                    if (score > bestIou) {
                        bestIou = score
                        bestIndex = index
                    }
                }
            }
            if (bestIndex >= 0 && bestIou >= matchIouThreshold) {
                matchedGt[bestIndex] = true
                truePositives += 1
                matchedLabels += detection.label
            } else {
                falsePositives += FalsePositive(
                    image = image.fileName,
                    label = detection.label,
                    confidence = detection.confidence,
                    box = detection.boundingBox,
                    bestIou = bestIou
                )
            }
        }

        val falseNegatives = image.groundTruth
            .filterIndexed { index, _ -> !matchedGt[index] }
            .map { gt ->
                FalseNegative(
                    image = image.fileName,
                    label = gt.label,
                    box = gt.box,
                    bestIou = detections
                        .filter { it.label == gt.label }
                        .maxOfOrNull { iou(it.boundingBox, gt.box) }
                        ?: 0f
                )
            }
        return DetectionEvaluation(
            image = image.fileName,
            gtCount = image.groundTruth.size,
            detectionCount = detections.size,
            truePositives = truePositives,
            matchedLabels = matchedLabels,
            falsePositives = falsePositives,
            falseNegatives = falseNegatives
        )
    }

    private fun aggregateQuality(
        images: List<BenchmarkImage>,
        perImage: List<PerImageModelResult>,
        detectionsByImage: Map<String, List<Detection>>,
        matchIouThreshold: Float
    ): QualitySummary {
        val tp = perImage.sumOf { it.evaluation.truePositives }
        val fp = perImage.sumOf { it.evaluation.falsePositiveCount }
        val fn = perImage.sumOf { it.evaluation.falseNegativeCount }
        val categorySummaries = RISK_LABELS.associateWith { label ->
            val categoryTp = perImage.sumOf { it.evaluation.truePositivesFor(label) }
            val categoryFp = perImage.sumOf { it.evaluation.falsePositives.count { fpItem -> fpItem.label == label } }
            val categoryFn = perImage.sumOf { it.evaluation.falseNegatives.count { fnItem -> fnItem.label == label } }
            QualityNumbers.from(categoryTp, categoryFp, categoryFn, images.size)
        }
        return QualitySummary(
            numbers = QualityNumbers.from(tp, fp, fn, images.size),
            ap50 = round3(ap50(images, detectionsByImage, matchIouThreshold)),
            riskCategorySummaries = categorySummaries
        )
    }

    private fun aggregateRiskQuality(perImage: List<PerImageModelResult>): RiskQualitySummary {
        var riskTp = 0
        var riskFp = 0
        var riskFn = 0
        var directionMismatch = 0
        var nearRiskMiss = 0
        perImage.forEach { item ->
            val expected = item.groundTruthRisk
            val actual = item.modelRisk
            val expectedActive = expected.level != RiskLevel.NONE
            val actualActive = actual.level != RiskLevel.NONE
            when {
                expectedActive && actualActive -> riskTp += 1
                !expectedActive && actualActive -> riskFp += 1
                expectedActive && !actualActive -> riskFn += 1
            }
            if (expectedActive && actualActive && expected.direction != actual.direction) {
                directionMismatch += 1
            }
            if (
                expected.proximity.ordinal >= ProximityBand.NEAR.ordinal &&
                actual.proximity.ordinal < expected.proximity.ordinal
            ) {
                nearRiskMiss += 1
            }
        }
        return RiskQualitySummary(
            riskTp = riskTp,
            riskFp = riskFp,
            riskFn = riskFn,
            directionMismatch = directionMismatch,
            nearRiskMiss = nearRiskMiss
        )
    }

    private fun ap50(
        images: List<BenchmarkImage>,
        detectionsByImage: Map<String, List<Detection>>,
        matchIouThreshold: Float
    ): Double {
        val totalGt = images.sumOf { it.groundTruth.size }
        if (totalGt == 0) return 0.0
        val matchedByImage = images.associate { it.fileName to BooleanArray(it.groundTruth.size) }
        val imageByName = images.associateBy { it.fileName }
        val predictions = detectionsByImage.flatMap { (imageName, detections) ->
            detections.map { DetectionPrediction(imageName, it) }
        }.sortedByDescending { it.detection.confidence }

        var tp = 0
        var fp = 0
        var previousRecall = 0.0
        var area = 0.0
        predictions.forEach { prediction ->
            val image = imageByName[prediction.imageName]
            val matched = matchedByImage[prediction.imageName]
            var bestIndex = -1
            var bestIou = 0f
            if (image != null && matched != null) {
                image.groundTruth.forEachIndexed { index, gt ->
                    if (!matched[index] && gt.label == prediction.detection.label) {
                        val score = iou(prediction.detection.boundingBox, gt.box)
                        if (score > bestIou) {
                            bestIou = score
                            bestIndex = index
                        }
                    }
                }
            }
            if (bestIndex >= 0 && bestIou >= matchIouThreshold && matched != null) {
                matched[bestIndex] = true
                tp += 1
            } else {
                fp += 1
            }
            val recall = tp.toDouble() / totalGt.toDouble()
            val precision = tp.toDouble() / max(1, tp + fp).toDouble()
            area += precision * max(0.0, recall - previousRecall)
            previousRecall = recall
        }
        return area
    }

    private fun recommendation(modelResults: List<ModelBenchmarkResult>): JSONObject {
        val baseline = modelResults.firstOrNull { it.spec.id == "yolo11n" }
        val candidate = modelResults.firstOrNull { it.spec.id == "yolo26n" }
        if (baseline == null || candidate == null) {
            return JSONObject()
                .put("decision", "not_enough_data")
                .put("reason", "Baseline or candidate result is missing.")
        }
        val baselineQuality = baseline.app.quality
        val candidateQuality = candidate.app.quality
        val candidateNotWorse =
            candidateQuality.recall >= baselineQuality.recall &&
                candidateQuality.precision >= baselineQuality.precision &&
                candidate.app.riskQuality.riskFn <= baseline.app.riskQuality.riskFn &&
                candidate.app.stability.riskLevelFlipRate <= baseline.app.stability.riskLevelFlipRate
        return if (candidateNotWorse) {
            JSONObject()
                .put("decision", "candidate_ok_for_next_stage")
                .put("replace_default_model_now", false)
                .put("reason", "yolo26n is not worse on recall, precision, risk FN, and risk stability in this COCO100 A/B run; keep it as a candidate until real walking images confirm the result.")
        } else {
            JSONObject()
                .put("decision", "do_not_replace_default_model")
                .put("replace_default_model_now", false)
                .put("reason", "yolo26n did not meet the no-regression gate for detection quality or risk stability in this COCO100 A/B run.")
        }
    }

    private fun perImageCsv(modelResults: List<ModelBenchmarkResult>): String {
        val lines = mutableListOf(
            "image,model,gt_count,detection_count,tp,fp,fn,precision,recall,f1,gt_risk_level,model_risk_level,gt_direction,model_direction,gt_proximity,model_proximity,risk_key_variants,risk_level_flip,box_jitter"
        )
        modelResults.forEach { model ->
            model.app.perImage.forEach { item ->
                lines += listOf(
                    item.image.fileName,
                    model.spec.id,
                    item.evaluation.gtCount,
                    item.evaluation.detectionCount,
                    item.evaluation.truePositives,
                    item.evaluation.falsePositiveCount,
                    item.evaluation.falseNegativeCount,
                    item.evaluation.precision,
                    item.evaluation.recall,
                    item.evaluation.f1,
                    item.groundTruthRisk.level,
                    item.modelRisk.level,
                    item.groundTruthRisk.direction,
                    item.modelRisk.direction,
                    item.groundTruthRisk.proximity,
                    item.modelRisk.proximity,
                    item.stability.riskKeyVariants,
                    item.stability.riskLevelFlip,
                    item.stability.sourceBoxJitter
                ).joinToString(",") { csvValue(it) }
            }
        }
        return lines.joinToString(separator = "\n", postfix = "\n")
    }

    private fun markdownReport(payload: JSONObject): String {
        val recommendation = payload.getJSONObject("recommendation")
        val models = payload.getJSONArray("models")
        return buildString {
            appendLine("# Detector A/B Device Benchmark")
            appendLine()
            appendLine("- Dataset: `COCO100 val2017 fixed sample`")
            appendLine("- Image count: `${payload.getInt("image_count")}`")
            appendLine("- App runs per image: `${payload.getInt("app_runs_per_image")}`")
            appendLine("- Match IoU threshold: `${payload.getDouble("match_iou_threshold")}`")
            appendLine("- Default app model remains: `${payload.getString("default_model_asset")}`")
            appendLine("- Recommendation: `${recommendation.getString("decision")}`")
            appendLine("- Replace default model now: `${recommendation.getBoolean("replace_default_model_now")}`")
            appendLine()
            appendLine("| Model | AP50 | Precision | Recall | F1 | FP/img | FN/img | Risk FN | Risk FP | Risk flips | Total P50 ms | Total P95 ms |")
            appendLine("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for (index in 0 until models.length()) {
                val model = models.getJSONObject(index)
                val app = model.getJSONObject("app_detector")
                val quality = app.getJSONObject("quality")
                val risk = app.getJSONObject("risk_quality")
                val stability = app.getJSONObject("stability")
                val total = app.getJSONObject("total_ms")
                appendLine(
                    "| `${model.getString("id")}` | ${quality.getDouble("ap50")} | ${quality.getDouble("precision")} | " +
                        "${quality.getDouble("recall")} | ${quality.getDouble("f1")} | ${quality.getDouble("fp_per_image")} | " +
                        "${quality.getDouble("fn_per_image")} | ${risk.getInt("risk_fn")} | ${risk.getInt("risk_fp")} | " +
                        "${stability.getDouble("risk_level_flip_rate")} | ${total.getDouble("p50")} | ${total.getDouble("p95")} |"
                )
            }
            appendLine()
            appendLine("Decision note: ${recommendation.getString("reason")}")
            appendLine()
            appendLine("Notes:")
            appendLine("- This A/B run compares both models on the same connected device and the same COCO100 image assets.")
            appendLine("- Detection matching uses same-class IoU at the configured threshold.")
            appendLine("- COCO100 is useful for repeatable detector quality checks, but real walking-assist images are still required before replacing the default model.")
        }
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

    private fun readAssetText(assetName: String): String {
        return testContext.assets.open(assetName)
            .bufferedReader(Charsets.UTF_8)
            .use { it.readText() }
    }

    private fun loadLabels(): List<String> {
        return readAssetText(TfliteYoloDetector.LABELS_ASSET)
            .lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .toList()
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
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "detector-ab-benchmark")
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        return File(root, stamp).also { check(it.mkdirs() || it.isDirectory) }
    }

    private fun noneRisk(): RiskResult {
        return RiskResult(RiskLevel.NONE, RiskDirection.NONE, "no risk")
    }

    private fun iou(a: BoundingBox, b: BoundingBox): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = a.width * a.height + b.width * b.height - intersection
        return if (union <= 0f) 0f else intersection / union
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

    private fun round3(value: Float): Double = round3(value.toDouble())

    private fun csvValue(value: Any): String {
        val text = value.toString()
        return if (text.any { it == ',' || it == '"' || it == '\n' }) {
            "\"" + text.replace("\"", "\"\"") + "\""
        } else {
            text
        }
    }

    private fun Bitmap.use(block: (Bitmap) -> Unit) {
        try {
            block(this)
        } finally {
            recycle()
        }
    }

    private data class ModelSpec(
        val id: String,
        val assetName: String,
        val source: String
    )

    private data class BenchmarkImage(
        val fileName: String,
        val relativePath: String,
        val frameSize: FrameSize,
        val groundTruth: List<GroundTruthBox>
    ) {
        fun groundTruthDetections(): List<Detection> {
            return groundTruth.map { gt ->
                Detection(
                    classId = gt.classId,
                    label = gt.label,
                    confidence = 1f,
                    boundingBox = gt.box,
                    frameSize = frameSize
                )
            }
        }
    }

    private data class GroundTruthBox(
        val id: Int,
        val classId: Int,
        val label: String,
        val box: BoundingBox
    )

    private data class FrameRun(
        val detections: List<Detection>,
        val risk: RiskResult,
        val preprocessMs: Double,
        val inferenceMs: Double,
        val postprocessMs: Double,
        val totalMs: Double
    )

    private data class ModelBenchmarkResult(
        val spec: ModelSpec,
        val pure: PureBenchmarkResult,
        val app: AppBenchmarkResult
    ) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("id", spec.id)
                .put("asset", spec.assetName)
                .put("source", spec.source)
                .put("pure_interpreter", pure.toJson())
                .put("app_detector", app.toJson())
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
        val quality: QualitySummary,
        val riskQuality: RiskQualitySummary,
        val stability: StabilityAggregate,
        val perImage: List<PerImageModelResult>,
        val falsePositives: List<JSONObject>,
        val falseNegatives: List<JSONObject>,
        val riskMismatches: List<JSONObject>,
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
                .put("quality", quality.toJson())
                .put("risk_quality", riskQuality.toJson())
                .put("stability", stability.toJson())
                .put("per_image", JSONArray(perImage.map { it.toJson() }))
                .put("failures", JSONArray(failures))
        }
    }

    private data class PerImageModelResult(
        val image: BenchmarkImage,
        val evaluation: DetectionEvaluation,
        val groundTruthRisk: RiskResult,
        val modelRisk: RiskResult,
        val stability: StabilitySummary
    ) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("image", image.fileName)
                .put("gt_count", evaluation.gtCount)
                .put("detection_count", evaluation.detectionCount)
                .put("quality", evaluation.toJson())
                .put("ground_truth_risk", riskToJson(groundTruthRisk))
                .put("model_risk", riskToJson(modelRisk))
                .put("stability", stability.toJson())
        }

        fun riskMismatchJson(modelId: String): JSONObject? {
            val expectedActive = groundTruthRisk.level != RiskLevel.NONE
            val actualActive = modelRisk.level != RiskLevel.NONE
            val mismatch =
                expectedActive != actualActive ||
                    (expectedActive && actualActive && groundTruthRisk.direction != modelRisk.direction) ||
                    (groundTruthRisk.proximity.ordinal >= ProximityBand.NEAR.ordinal &&
                        modelRisk.proximity.ordinal < groundTruthRisk.proximity.ordinal)
            if (!mismatch) return null
            return JSONObject()
                .put("model", modelId)
                .put("image", image.fileName)
                .put("ground_truth_risk", riskToJson(groundTruthRisk))
                .put("model_risk", riskToJson(modelRisk))
        }
    }

    private data class DetectionEvaluation(
        val image: String,
        val gtCount: Int,
        val detectionCount: Int,
        val truePositives: Int,
        val matchedLabels: List<String>,
        val falsePositives: List<FalsePositive>,
        val falseNegatives: List<FalseNegative>
    ) {
        val falsePositiveCount: Int get() = falsePositives.size
        val falseNegativeCount: Int get() = falseNegatives.size
        val precision: Double get() = round3(truePositives.toDouble() / max(1, truePositives + falsePositiveCount).toDouble())
        val recall: Double get() = round3(truePositives.toDouble() / max(1, truePositives + falseNegativeCount).toDouble())
        val f1: Double
            get() {
                val p = precision
                val r = recall
                return if (p + r <= 0.0) 0.0 else round3(2.0 * p * r / (p + r))
            }

        fun truePositivesFor(label: String): Int {
            return matchedLabels.count { it == label }
        }

        fun toJson(): JSONObject {
            return JSONObject()
                .put("tp", truePositives)
                .put("fp", falsePositiveCount)
                .put("fn", falseNegativeCount)
                .put("precision", precision)
                .put("recall", recall)
                .put("f1", f1)
        }
    }

    private data class FalsePositive(
        val image: String,
        val label: String,
        val confidence: Float,
        val box: BoundingBox,
        val bestIou: Float
    ) {
        fun toJson(modelId: String): JSONObject {
            return JSONObject()
                .put("model", modelId)
                .put("image", image)
                .put("label", label)
                .put("confidence", round3(confidence))
                .put("bbox_xyxy", boxToJson(box))
                .put("best_iou", round3(bestIou))
        }
    }

    private data class FalseNegative(
        val image: String,
        val label: String,
        val box: BoundingBox,
        val bestIou: Float
    ) {
        fun toJson(modelId: String): JSONObject {
            return JSONObject()
                .put("model", modelId)
                .put("image", image)
                .put("label", label)
                .put("bbox_xyxy", boxToJson(box))
                .put("best_iou", round3(bestIou))
        }
    }

    private data class QualitySummary(
        val numbers: QualityNumbers,
        val ap50: Double,
        val riskCategorySummaries: Map<String, QualityNumbers>
    ) {
        val precision: Double get() = numbers.precision
        val recall: Double get() = numbers.recall

        fun toJson(): JSONObject {
            val byCategory = JSONObject()
            riskCategorySummaries.forEach { (label, summary) ->
                byCategory.put(label, summary.toJson())
            }
            return numbers.toJson()
                .put("ap50", ap50)
                .put("risk_categories", byCategory)
        }
    }

    private data class QualityNumbers(
        val tp: Int,
        val fp: Int,
        val fn: Int,
        val imageCount: Int
    ) {
        val precision: Double get() = round3(tp.toDouble() / max(1, tp + fp).toDouble())
        val recall: Double get() = round3(tp.toDouble() / max(1, tp + fn).toDouble())
        val f1: Double
            get() {
                val p = precision
                val r = recall
                return if (p + r <= 0.0) 0.0 else round3(2.0 * p * r / (p + r))
            }
        val fpPerImage: Double get() = round3(fp.toDouble() / max(1, imageCount).toDouble())
        val fnPerImage: Double get() = round3(fn.toDouble() / max(1, imageCount).toDouble())

        fun toJson(): JSONObject {
            return JSONObject()
                .put("tp", tp)
                .put("fp", fp)
                .put("fn", fn)
                .put("precision", precision)
                .put("recall", recall)
                .put("f1", f1)
                .put("fp_per_image", fpPerImage)
                .put("fn_per_image", fnPerImage)
        }

        companion object {
            fun from(tp: Int, fp: Int, fn: Int, imageCount: Int): QualityNumbers {
                return QualityNumbers(tp, fp, fn, imageCount)
            }
        }
    }

    private data class RiskQualitySummary(
        val riskTp: Int,
        val riskFp: Int,
        val riskFn: Int,
        val directionMismatch: Int,
        val nearRiskMiss: Int
    ) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("risk_tp", riskTp)
                .put("risk_fp", riskFp)
                .put("risk_fn", riskFn)
                .put("direction_mismatch", directionMismatch)
                .put("near_risk_miss", nearRiskMiss)
        }
    }

    private data class StabilitySummary(
        val detectionCountMin: Int,
        val detectionCountMax: Int,
        val detectionCountRange: Int,
        val riskKeyVariants: Int,
        val riskLevelFlip: Boolean,
        val sourceBoxJitter: Double
    ) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("detection_count_min", detectionCountMin)
                .put("detection_count_max", detectionCountMax)
                .put("detection_count_range", detectionCountRange)
                .put("risk_key_variants", riskKeyVariants)
                .put("risk_level_flip", riskLevelFlip)
                .put("source_box_jitter", sourceBoxJitter)
        }

        companion object {
            fun from(runs: List<FrameRun>): StabilitySummary {
                if (runs.isEmpty()) {
                    return StabilitySummary(0, 0, 0, 0, false, 0.0)
                }
                val counts = runs.map { it.detections.size }
                val riskKeys = runs.map { riskKey(it.risk) }.toSet()
                val levels = runs.map { it.risk.level }.toSet()
                val boxes = runs.mapNotNull { it.risk.sourceDetection?.boundingBox }
                val reference = boxes.firstOrNull()
                val averageIou = if (reference == null || boxes.size <= 1) {
                    1.0
                } else {
                    boxes.drop(1).map { box ->
                        val left = max(reference.left, box.left)
                        val top = max(reference.top, box.top)
                        val right = min(reference.right, box.right)
                        val bottom = min(reference.bottom, box.bottom)
                        val intersection = max(0f, right - left) * max(0f, bottom - top)
                        val union = reference.width * reference.height + box.width * box.height - intersection
                        if (union <= 0f) 0f else intersection / union
                    }.average()
                }
                return StabilitySummary(
                    detectionCountMin = counts.minOrNull() ?: 0,
                    detectionCountMax = counts.maxOrNull() ?: 0,
                    detectionCountRange = (counts.maxOrNull() ?: 0) - (counts.minOrNull() ?: 0),
                    riskKeyVariants = riskKeys.size,
                    riskLevelFlip = levels.size > 1,
                    sourceBoxJitter = ((1.0 - averageIou).coerceIn(0.0, 1.0) * 1000.0).roundToInt() / 1000.0
                )
            }
        }
    }

    private data class StabilityAggregate(
        val riskLevelFlipRate: Double,
        val meanDetectionCountRange: Double,
        val meanSourceBoxJitter: Double
    ) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("risk_level_flip_rate", riskLevelFlipRate)
                .put("mean_detection_count_range", meanDetectionCountRange)
                .put("mean_source_box_jitter", meanSourceBoxJitter)
        }

        companion object {
            fun from(items: List<StabilitySummary>): StabilityAggregate {
                if (items.isEmpty()) return StabilityAggregate(0.0, 0.0, 0.0)
                return StabilityAggregate(
                    riskLevelFlipRate = round3(items.count { it.riskLevelFlip }.toDouble() / items.size.toDouble()),
                    meanDetectionCountRange = round3(items.map { it.detectionCountRange.toDouble() }.average()),
                    meanSourceBoxJitter = round3(items.map { it.sourceBoxJitter }.average())
                )
            }
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

    private data class DetectionPrediction(
        val imageName: String,
        val detection: Detection
    )

    companion object {
        private const val TAG = "DetectorAbBenchmark"
        private const val YOLO11N_ASSET = "yolo11n_fp16_320.tflite"
        private const val YOLO26N_ASSET = "yolo26n_fp16_320.tflite"
        private const val COCO_ANNOTATIONS_ASSET = "coco100_annotations.json"
        private const val ARG_IMAGE_LIMIT = "imageLimit"
        private const val ARG_PURE_WARMUP = "pureWarmup"
        private const val ARG_PURE_RUNS = "pureRuns"
        private const val ARG_APP_RUNS_PER_IMAGE = "appRunsPerImage"
        private const val ARG_MATCH_IOU_THRESHOLD = "matchIouThreshold"
        private const val DEFAULT_IMAGE_LIMIT = 100
        private const val DEFAULT_PURE_WARMUP = 10
        private const val DEFAULT_PURE_RUNS = 100
        private const val DEFAULT_APP_RUNS_PER_IMAGE = 3
        private const val DEFAULT_MATCH_IOU_THRESHOLD = 0.5f
        private val MODEL_SPECS = listOf(
            ModelSpec("yolo11n", YOLO11N_ASSET, "app/src/main/assets/yolo11n_fp16_320.tflite"),
            ModelSpec("yolo26n", YOLO26N_ASSET, ".downloads/detector-lab/exports/yolo26n_fp16_320.tflite")
        )
        private val RISK_LABELS = setOf(
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "truck",
            "traffic light",
            "stop sign",
            "bench",
            "chair",
            "potted plant"
        )

        private fun riskKey(risk: RiskResult): String {
            val label = risk.sourceDetection?.label ?: "none"
            return "${risk.level}|${risk.direction}|${risk.proximity}|$label"
        }

        private fun riskToJson(risk: RiskResult): JSONObject {
            return JSONObject()
                .put("level", risk.level.name)
                .put("direction", risk.direction.name)
                .put("proximity", risk.proximity.name)
                .put("source_label", risk.sourceDetection?.label ?: JSONObject.NULL)
                .put("source_confidence", risk.sourceDetection?.confidence?.let { round3(it) } ?: JSONObject.NULL)
                .put("urgency_score", round3(risk.urgencyScore))
        }

        private fun boxToJson(box: BoundingBox): JSONArray {
            return JSONArray(
                listOf(
                    round3(box.left),
                    round3(box.top),
                    round3(box.right),
                    round3(box.bottom)
                )
            )
        }

        private fun round3(value: Double): Double {
            return (value * 1000.0).roundToInt() / 1000.0
        }

        private fun round3(value: Float): Double = round3(value.toDouble())
    }
}
