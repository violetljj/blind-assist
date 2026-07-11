package com.linnan.blindassist.benchmark

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackPlanner
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.ApproachTrend
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskAnalyzer
import com.linnan.blindassist.risk.RiskAnalyzerConfig
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.risk.TemporalRiskTracker
import com.linnan.blindassist.risk.BlindAssistSanpoTaxonomy
import com.linnan.blindassist.risk.DenseSemanticMask
import com.linnan.blindassist.risk.TraversabilitySegmentationAnalyzer
import com.linnan.blindassist.vision.DepthEvidenceSamplingConfig
import com.linnan.blindassist.vision.ImagePreprocessor
import com.linnan.blindassist.vision.TfliteYoloDetector
import com.linnan.blindassist.vision.TfliteMonocularDepthEstimator
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

    @Test
    fun compareYolo11nAndYolo26nOnSameDeviceImagesAndMetrics() {
        val datasetKind = arguments.getString(ARG_DATASET_KIND) ?: DATASET_KIND_COCO100
        val riskConfigName = arguments.getString(ARG_RISK_CONFIG) ?: RISK_CONFIG_CURRENT
        val comparisonMode = arguments.getString(ARG_COMPARISON_MODE) ?: COMPARISON_MODE_DETECTOR_AB
        val depthModelAsset = arguments.getString(ARG_DEPTH_MODEL_ASSET) ?: TfliteMonocularDepthEstimator.MODEL_ASSET
        val baseRiskAnalyzerConfig = riskAnalyzerConfigFor(riskConfigName)
        val depthCloserIsLarger = arguments.getString(ARG_DEPTH_CLOSER_IS_LARGER)?.toBooleanStrictOrNull() ?: true
        val singleDepthSamplingConfig = depthSamplingConfigFromArguments()
        val imageLimit = arguments.getString(ARG_IMAGE_LIMIT)?.toIntOrNull() ?: DEFAULT_IMAGE_LIMIT
        val pureWarmup = arguments.getString(ARG_PURE_WARMUP)?.toIntOrNull() ?: DEFAULT_PURE_WARMUP
        val pureRuns = arguments.getString(ARG_PURE_RUNS)?.toIntOrNull() ?: DEFAULT_PURE_RUNS
        val appRunsPerImage = arguments.getString(ARG_APP_RUNS_PER_IMAGE)?.toIntOrNull() ?: DEFAULT_APP_RUNS_PER_IMAGE
        val matchIouThreshold = arguments.getString(ARG_MATCH_IOU_THRESHOLD)?.toFloatOrNull()
            ?: DEFAULT_MATCH_IOU_THRESHOLD

        assertEquals(YOLO11N_ASSET, TfliteYoloDetector.MODEL_ASSET)
        val modelSpecs = modelSpecsFor(
            comparisonMode = comparisonMode,
            depthModelAsset = depthModelAsset,
            depthCloserIsLarger = depthCloserIsLarger,
            baseRiskAnalyzerConfig = baseRiskAnalyzerConfig,
            singleDepthSamplingConfig = singleDepthSamplingConfig
        )
        modelSpecs.forEach { spec ->
            assertTrue("${spec.assetName} test asset is missing", assetExists(spec.assetName))
            if (spec.depthFusion) {
                assertTrue("${spec.depthModelAssetName} depth test asset is missing", assetExists(spec.depthModelAssetName))
            }
        }
        assertTrue("coco labels test asset is missing", assetExists(TfliteYoloDetector.LABELS_ASSET))

        val labels = loadLabels()
        val dataset = loadBenchmarkDataset(datasetKind, labels)
        val images = dataset.images.take(imageLimit)
        assertTrue("Expected at least one annotated ${dataset.name} image asset", images.isNotEmpty())
        if (modelSpecs.any { it.traversabilityOracle }) {
            images.forEach { image ->
                val maskPath = image.sourceMaskPath
                assertTrue("SANPO source mask is missing for ${image.fileName}", maskPath != null && assetExists(maskPath))
            }
        }

        val artifactDir = createArtifactDir(targetContext)
        val pureCache = mutableMapOf<String, PureBenchmarkResult>()
        val modelResults = modelSpecs.map { spec ->
            val pure = pureCache.getOrPut(spec.assetName) {
                runPureInterpreterBenchmark(spec, images, pureWarmup, pureRuns)
            }
            val app = runAppDetectorBenchmark(spec, images, appRunsPerImage, matchIouThreshold)
            assertFalse("${spec.id} App detector produced failures", app.failures.isNotEmpty())
            ModelBenchmarkResult(spec, pure, app)
        }

        val payload = JSONObject()
            .put("created_at_utc", utcNow())
            .put("device_under_test", "instrumentation-connected-device")
            .put("dataset_kind", datasetKind)
            .put("dataset_name", dataset.name)
            .put("comparison_mode", comparisonMode)
            .put("risk_config", riskConfigName)
            .put("depth_model_asset", depthModelAsset)
            .put("depth_closer_is_larger", depthCloserIsLarger)
            .put("depth_sampling_config", singleDepthSamplingConfig.toJson())
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
                .put("depth_model", depthModelAsset)
                .put("images", ".downloads/detector-lab/datasets/coco100/images")
                .put("annotations", ".downloads/detector-lab/datasets/coco100/coco100_annotations.json")
                .put("blindassist_evalset", "test-artifacts.local/datasets/blindassist-evalset-20260527-impl")
                .put("labels", "app/src/main/assets/coco_labels.txt"))

        File(artifactDir, "benchmark.json").writeText(payload.toString(2), Charsets.UTF_8)
        File(artifactDir, "benchmark.md").writeText(markdownReport(payload), Charsets.UTF_8)
        File(artifactDir, "per-image.csv").writeText(perImageCsv(modelResults), Charsets.UTF_8)
        if (comparisonMode == COMPARISON_MODE_DEPTH_FUSION_SWEEP) {
            File(artifactDir, "sweep-summary.json").writeText(sweepSummaryJson(modelResults).toString(2), Charsets.UTF_8)
            File(artifactDir, "sweep-summary.csv").writeText(sweepSummaryCsv(modelResults), Charsets.UTF_8)
            File(artifactDir, "sweep-summary.md").writeText(sweepSummaryMarkdown(modelResults), Charsets.UTF_8)
        }
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
        val riskAnalyzer = RiskAnalyzer(spec.riskAnalyzerConfig)
        val detector = TfliteYoloDetector(
            context = testContext,
            modelAssetName = spec.assetName,
            labelsAssetName = TfliteYoloDetector.LABELS_ASSET
        )
        assertTrue("${spec.id} detector failed to initialize: ${detector.statusMessage}", detector.isReady)
        val depthEstimator = if (spec.depthFusion) {
            TfliteMonocularDepthEstimator(
                context = testContext,
                modelAssetName = spec.depthModelAssetName,
                closerIsLarger = spec.depthCloserIsLarger
            ).also { estimator ->
                assertTrue("${spec.id} depth estimator failed to initialize: ${estimator.statusMessage}", estimator.isReady)
            }
        } else {
            null
        }
        val readyModelStatus = "ready"

        val preprocess = mutableListOf<Double>()
        val inference = mutableListOf<Double>()
        val postprocess = mutableListOf<Double>()
        val depth = mutableListOf<Double>()
        val total = mutableListOf<Double>()
        val failures = mutableListOf<String>()
        val perImage = mutableListOf<PerImageModelResult>()
        val firstRunDetectionsByImage = linkedMapOf<String, List<Detection>>()
        val temporalRiskTracker = TemporalRiskTracker()
        val traversabilityAnalyzer = TraversabilitySegmentationAnalyzer()
        var activeSequenceId: String? = null

        try {
            images.sortedForTemporalEvaluation().forEach { image ->
                val runs = mutableListOf<FrameRun>()
                repeat(runsPerImage) {
                    try {
                        decodeBitmap(image.relativePath).use { bitmap ->
                            val frameSize = FrameSize(bitmap.width, bitmap.height)
                            val depthResult = depthEstimator?.estimate(bitmap)
                            val result = detector.detect(bitmap)
                            val detections = if (depthResult != null) {
                                result.detections.map { detection ->
                                    detection.copy(
                                        distanceEvidence = depthResult.depthMap.sampleEvidence(
                                            detection.boundingBox,
                                            frameSize,
                                            spec.depthSamplingConfig
                                        )
                                    )
                                }
                            } else {
                                result.detections
                            }
                            val oracleStart = System.nanoTime()
                            val oracleDetections = if (spec.traversabilityOracle) {
                                val maskPath = requireNotNull(image.sourceMaskPath)
                                traversabilityAnalyzer.analyze(decodeSemanticMask(maskPath), frameSize).riskDetections
                            } else {
                                emptyList()
                            }
                            val oracleMs = if (spec.traversabilityOracle) elapsedMs(oracleStart) else 0.0
                            val riskInputs = if (spec.traversabilityOracle) {
                                detections + oracleDetections
                            } else {
                                detections
                            }
                            val risk = riskAnalyzer.analyze(riskInputs, frameSize)
                            preprocess += detector.lastPreprocessMs.toDouble()
                            inference += detector.lastInferenceMs.toDouble()
                            postprocess += detector.lastPostprocessMs.toDouble()
                            val depthMs = depthResult?.metrics?.totalMs?.toDouble() ?: 0.0
                            val totalMs = detector.lastTotalDetectMs.toDouble() + depthMs + oracleMs
                            depth += depthMs
                            total += totalMs
                            runs += FrameRun(
                                detections = detections,
                                risk = risk,
                                preprocessMs = detector.lastPreprocessMs.toDouble(),
                                inferenceMs = detector.lastInferenceMs.toDouble(),
                                postprocessMs = detector.lastPostprocessMs.toDouble(),
                                depthMs = depthMs,
                                totalMs = totalMs
                            )
                        }
                    } catch (error: Throwable) {
                        failures += "${image.fileName}: ${error.javaClass.simpleName}: ${error.message}"
                    }
                }

                val firstRun = runs.firstOrNull() ?: FrameRun(emptyList(), noneRisk(), 0.0, 0.0, 0.0, 0.0, 0.0)
                val sequenceId = image.expectedRisk?.sequenceId
                val frameIndex = image.expectedRisk?.frameIndex
                val modelRisk = if (sequenceId != null && frameIndex != null) {
                    if (activeSequenceId != sequenceId) {
                        temporalRiskTracker.reset()
                        activeSequenceId = sequenceId
                    }
                    temporalRiskTracker.update(firstRun.risk, frameIndex.toLong() * TEMPORAL_FRAME_STEP_MS)
                } else {
                    temporalRiskTracker.reset()
                    activeSequenceId = null
                    firstRun.risk
                }
                val gtRisk = expectedRiskResult(image, riskAnalyzer)
                val evaluation = evaluateDetections(
                    image = image,
                    detections = firstRun.detections,
                    matchIouThreshold = matchIouThreshold
                )
                val stability = StabilitySummary.from(runs)
                perImage += PerImageModelResult(
                    image = image,
                    evaluation = evaluation,
                    groundTruthRisk = gtRisk,
                    modelRisk = modelRisk,
                    actualAlert = alertFor(modelRisk, image.expectedRisk?.scenario ?: AssistScenario.GENERAL),
                    stability = stability
                )
                firstRunDetectionsByImage[image.fileName] = firstRun.detections
            }
        } finally {
            depthEstimator?.close()
            detector.close()
        }

        val quality = aggregateQuality(images, perImage, firstRunDetectionsByImage, matchIouThreshold)
        val riskQuality = aggregateRiskQuality(perImage)
        val blindAssistMetrics = aggregateBlindAssistMetrics(perImage, matchIouThreshold)
        return AppBenchmarkResult(
            runs = images.size * runsPerImage,
            runsPerImage = runsPerImage,
            modelStatus = readyModelStatus,
            preprocessStats = Stats.from(preprocess),
            inferenceStats = Stats.from(inference),
            postprocessStats = Stats.from(postprocess),
            depthStats = Stats.from(depth),
            totalStats = Stats.from(total),
            quality = quality,
            riskQuality = riskQuality,
            blindAssistMetrics = blindAssistMetrics,
            stability = StabilityAggregate.from(perImage.map { it.stability }),
            fusionSummaryCounts = perImage
                .groupingBy { it.modelRisk.scoreBreakdown.fusionSummary }
                .eachCount(),
            perImage = perImage,
            falsePositives = perImage.flatMap { it.evaluation.falsePositives }.map { it.toJson(spec.id) },
            falseNegatives = perImage.flatMap { it.evaluation.falseNegatives }.map { it.toJson(spec.id) },
            riskMismatches = perImage.mapNotNull { it.riskMismatchJson(spec.id) },
            failures = failures
        )
    }

    private fun loadBenchmarkDataset(kind: String, labels: List<String>): BenchmarkDataset {
        return when (kind) {
            DATASET_KIND_COCO100 -> {
                assertTrue("COCO100 annotation asset is missing", assetExists(COCO_ANNOTATIONS_ASSET))
                BenchmarkDataset("COCO100 val2017 fixed sample", loadCocoBenchmarkImages(labels))
            }
            DATASET_KIND_BLINDASSIST_EVALSET -> {
                assertTrue("BlindAssist evalset manifest asset is missing", assetExists(BLINDASSIST_EVALSET_MANIFEST_ASSET))
                assertTrue("BlindAssist evalset spec asset is missing", assetExists(BLINDASSIST_EVALSET_SPEC_ASSET))
                BenchmarkDataset("BlindAssist EvalSet", loadBlindAssistEvalSetImages(labels))
            }
            else -> error("Unsupported datasetKind: $kind")
        }
    }

    private fun loadCocoBenchmarkImages(labels: List<String>): List<BenchmarkImage> {
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
                        id = annotation.getInt("id").toString(),
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
                groundTruth = groundTruth,
                expectedRisk = null
            )
        }.filter { assetExists(it.relativePath) }
            .sortedBy { it.fileName }
    }

    private fun loadBlindAssistEvalSetImages(labels: List<String>): List<BenchmarkImage> {
        return readAssetText(BLINDASSIST_EVALSET_MANIFEST_ASSET)
            .lineSequence()
            .filter { it.isNotBlank() }
            .map { line ->
                val row = JSONObject(line)
                val frameSize = FrameSize(row.getInt("width"), row.getInt("height"))
                val objects = row.getJSONArray("objects")
                val groundTruth = (0 until objects.length()).mapNotNull { objectIndex ->
                    val obj = objects.getJSONObject(objectIndex)
                    val label = obj.getString("class")
                    val classId = labels.indexOf(label)
                    if (classId < 0) {
                        null
                    } else {
                        val xyxy = obj.getJSONArray("bbox_xyxy")
                        GroundTruthBox(
                            id = obj.getString("id"),
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
                val imagePath = row.getString("image_path")
                val attributes = row.optJSONObject("attributes")
                val sourceRegions = row.optJSONArray("source_regions")?.let { regions ->
                    (0 until regions.length()).mapNotNull { regionIndex ->
                        val region = regions.getJSONObject(regionIndex)
                        val classId = region.optInt("sanpo_class_id", -1)
                        val label = BlindAssistSanpoTaxonomy.riskLabelFor(classId) ?: return@mapNotNull null
                        val xyxy = region.getJSONArray("bbox_xyxy")
                        SourceRegion(
                            id = region.getString("id"),
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
                }.orEmpty()
                BenchmarkImage(
                    fileName = imagePath.substringAfterLast('/'),
                    relativePath = "$BLINDASSIST_EVALSET_ASSET_PREFIX/$imagePath",
                    frameSize = frameSize,
                    groundTruth = groundTruth,
                    sourceRegions = sourceRegions,
                    sourcePrimaryRegionId = row.optionalString("source_primary_region_id"),
                    sourceMaskPath = "$BLINDASSIST_EVALSET_ASSET_PREFIX/source_masks/test/${imagePath.substringAfterLast('/')}",
                    expectedRisk = BlindAssistExpectedRisk(
                        direction = RiskDirection.valueOf(row.getString("expected_risk_direction")),
                        distanceBand = ProximityBand.valueOf(row.getString("expected_distance_band")),
                        shouldAlert = row.getBoolean("expected_should_alert"),
                        riskLevel = RiskLevel.valueOf(row.getString("expected_risk_level")),
                        scenario = AssistScenario.valueOf(row.optString("assist_scenario", AssistScenario.GENERAL.name)),
                        primaryObjectId = row.optionalString("primary_object_id"),
                        sceneBucket = attributes?.optionalString("scene_bucket").orEmpty(),
                        sequenceId = row.optionalString("sequence_id"),
                        frameIndex = row.optionalInt("frame_index"),
                        expectedApproachState = row.optionalApproachTrend("expected_approach_state"),
                        expectedApproachAlert = row.optionalBoolean("expected_approach_alert"),
                        expectedTimeToAlertFrames = row.optionalInt("expected_time_to_alert_frames")
                    )
                )
            }
            .filter { assetExists(it.relativePath) }
            .sortedBy { it.fileName }
            .toList()
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

    private fun aggregateBlindAssistMetrics(
        perImage: List<PerImageModelResult>,
        matchIouThreshold: Float
    ): BlindAssistMetrics {
        val labeled = perImage.filter { it.image.expectedRisk != null }
        val expectedAlerts = labeled.filter { requireNotNull(it.image.expectedRisk).shouldAlert }
        val expectedNonAlerts = labeled.filter { !requireNotNull(it.image.expectedRisk).shouldAlert }
        val expectedCenterAlerts = expectedAlerts.filter { requireNotNull(it.image.expectedRisk).direction == RiskDirection.CENTER }
        val primaryRows = labeled.filter { !requireNotNull(it.image.expectedRisk).primaryObjectId.isNullOrBlank() }
        val sourcePrimaryRows = labeled.filter { !it.image.sourcePrimaryRegionId.isNullOrBlank() }
        val approachRows = labeled.filter { requireNotNull(it.image.expectedRisk).hasApproachLabel }
        val expectedApproaching = approachRows.filter {
            requireNotNull(it.image.expectedRisk).expectedApproachState == ApproachTrend.APPROACHING
        }
        val expectedApproachAlerts = approachRows.filter {
            requireNotNull(it.image.expectedRisk).expectedApproachAlert == true
        }
        val expectedNonApproachAlerts = approachRows.filter {
            requireNotNull(it.image.expectedRisk).expectedApproachAlert == false
        }
        val actualApproachingRows = approachRows.filter { it.modelRisk.approachTrend == ApproachTrend.APPROACHING }
        return BlindAssistMetrics(
            centerRiskRecall = ratio(
                expectedCenterAlerts.count {
                    it.actualAlert && it.modelRisk.direction == RiskDirection.CENTER
                },
                expectedCenterAlerts.size
            ),
            alertRecall = ratio(expectedAlerts.count { it.actualAlert }, expectedAlerts.size),
            alertFalsePositiveRate = ratio(expectedNonAlerts.count { it.actualAlert }, expectedNonAlerts.size),
            distanceBandAccuracy = ratio(
                labeled.count { item ->
                    item.modelRisk.proximity == requireNotNull(item.image.expectedRisk).distanceBand
                },
                labeled.size
            ),
            riskLevelAccuracy = ratio(
                labeled.count { item ->
                    item.modelRisk.level == requireNotNull(item.image.expectedRisk).riskLevel
                },
                labeled.size
            ),
            primaryObjectHitRate = ratio(
                primaryRows.count { primaryObjectHit(it, matchIouThreshold) },
                primaryRows.size
            ),
            sourcePrimaryRegionHitRate = ratio(
                sourcePrimaryRows.count { sourcePrimaryRegionHit(it, matchIouThreshold) },
                sourcePrimaryRows.size
            ),
            criticalMissCount = expectedAlerts.count { item ->
                val expected = requireNotNull(item.image.expectedRisk)
                expected.distanceBand == ProximityBand.CRITICAL &&
                    (!item.actualAlert || item.modelRisk.proximity.ordinal < ProximityBand.CRITICAL.ordinal)
            },
            approachRiskRecall = ratio(
                expectedApproaching.count { it.modelRisk.approachTrend == ApproachTrend.APPROACHING },
                expectedApproaching.size
            ),
            approachFalsePositiveRate = ratio(
                expectedNonApproachAlerts.count { it.modelRisk.approachTrend == ApproachTrend.APPROACHING },
                expectedNonApproachAlerts.size
            ),
            approachDirectionAccuracy = ratio(
                actualApproachingRows.count { item ->
                    item.modelRisk.direction == requireNotNull(item.image.expectedRisk).direction
                },
                actualApproachingRows.size
            ),
            approachCriticalMissCount = expectedApproachAlerts.count { item ->
                item.modelRisk.approachTrend != ApproachTrend.APPROACHING || !item.actualAlert
            },
            meanTimeToAlertFrames = meanTimeToAlertFrames(approachRows),
            approachLabeledSequenceCount = approachRows.mapNotNull { it.image.expectedRisk?.sequenceId }.toSet().size,
            labeledImageCount = labeled.size
        )
    }

    private fun meanTimeToAlertFrames(approachRows: List<PerImageModelResult>): Double {
        val deltas = approachRows
            .groupBy { it.image.expectedRisk?.sequenceId }
            .filterKeys { it != null }
            .values
            .mapNotNull { sequenceRows ->
                val sortedRows = sequenceRows.sortedBy { it.image.expectedRisk?.frameIndex ?: Int.MAX_VALUE }
                val firstExpectedAlert = sortedRows.firstOrNull {
                    it.image.expectedRisk?.expectedApproachAlert == true
                } ?: return@mapNotNull null
                val startFrame = firstExpectedAlert.image.expectedRisk?.frameIndex ?: return@mapNotNull null
                val actualAlertFrame = sortedRows.firstOrNull { item ->
                    val index = item.image.expectedRisk?.frameIndex ?: return@firstOrNull false
                    index >= startFrame &&
                        item.modelRisk.approachTrend == ApproachTrend.APPROACHING &&
                        item.actualAlert
                }?.image?.expectedRisk?.frameIndex ?: return@mapNotNull null
                (actualAlertFrame - startFrame).coerceAtLeast(0).toDouble()
            }
        return if (deltas.isEmpty()) 0.0 else round3(deltas.average())
    }

    private fun primaryObjectHit(item: PerImageModelResult, matchIouThreshold: Float): Boolean {
        val expected = item.image.expectedRisk ?: return false
        val expectedBox = item.image.groundTruth.firstOrNull { it.id == expected.primaryObjectId } ?: return false
        val actual = item.modelRisk.sourceDetection ?: return false
        return actual.label == expectedBox.label && iou(actual.boundingBox, expectedBox.box) >= matchIouThreshold
    }

    private fun sourcePrimaryRegionHit(item: PerImageModelResult, matchIouThreshold: Float): Boolean {
        val expectedId = item.image.sourcePrimaryRegionId ?: return false
        val expectedRegion = item.image.sourceRegions.firstOrNull { it.id == expectedId } ?: return false
        val actual = item.modelRisk.sourceDetection ?: return false
        return actual.source == DetectionSource.SEGMENTATION &&
            actual.label == expectedRegion.label &&
            intersectionOverActual(actual.boundingBox, expectedRegion.box) >= matchIouThreshold
    }

    private fun expectedRiskResult(image: BenchmarkImage, riskAnalyzer: RiskAnalyzer): RiskResult {
        val expected = image.expectedRisk ?: return riskAnalyzer.analyze(image.groundTruthDetections(), image.frameSize)
        return RiskResult(
            level = expected.riskLevel,
            direction = expected.direction,
            message = "expected",
            sourceDetection = image.primaryGroundTruthDetection(),
            proximity = expected.distanceBand,
            urgencyScore = if (expected.shouldAlert) 1f else 0f
        )
    }

    private fun alertFor(risk: RiskResult, scenario: AssistScenario): Boolean {
        return FeedbackPlanner.planFor(risk, scenario = scenario) != null
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
        val baseline = modelResults.firstOrNull { it.spec.id == "baseline_geometry" }
            ?: modelResults.firstOrNull { it.spec.id == "baseline_yolo_geometry" }
            ?: modelResults.firstOrNull { it.spec.id == "yolo11n" }
        val candidates = modelResults.filter {
            it !== baseline && (it.spec.depthFusion || it.spec.traversabilityOracle || it.spec.id == "yolo26n")
        }
        if (baseline == null || candidates.isEmpty()) {
            return JSONObject()
                .put("decision", "not_enough_data")
                .put("reason", "Baseline or candidate results are missing.")
        }
        val baselineQuality = baseline.app.quality
        val hasBlindAssistLabels = baseline.app.blindAssistMetrics.labeledImageCount > 0
        val passingCandidates = candidates.filter { candidate ->
            val candidateQuality = candidate.app.quality
            val baselineAssist = baseline.app.blindAssistMetrics
            val candidateAssist = candidate.app.blindAssistMetrics
            if (hasBlindAssistLabels) {
                candidateQuality.ap50 >= baselineQuality.ap50 &&
                    candidateQuality.recall >= baselineQuality.recall &&
                    candidateQuality.precision >= baselineQuality.precision &&
                    candidateAssist.alertRecall >= baselineAssist.alertRecall &&
                    candidateAssist.centerRiskRecall >= baselineAssist.centerRiskRecall &&
                    candidateAssist.criticalMissCount <= baselineAssist.criticalMissCount &&
                    candidateAssist.alertFalsePositiveRate <= baselineAssist.alertFalsePositiveRate + 0.02 &&
                    candidateAssist.distanceBandAccuracy >= baselineAssist.distanceBandAccuracy &&
                    candidateAssist.riskLevelAccuracy >= baselineAssist.riskLevelAccuracy - 0.02 &&
                    candidateAssist.primaryObjectHitRate >= baselineAssist.primaryObjectHitRate - 0.02 &&
                    (!candidate.spec.traversabilityOracle || candidateAssist.sourcePrimaryRegionHitRate >= 0.80)
            } else {
                candidateQuality.recall >= baselineQuality.recall &&
                    candidateQuality.precision >= baselineQuality.precision &&
                    candidate.app.riskQuality.riskFn <= baseline.app.riskQuality.riskFn &&
                    candidate.app.stability.riskLevelFlipRate <= baseline.app.stability.riskLevelFlipRate
            }
        }
        val bestPassing = passingCandidates.minWithOrNull(
            compareBy<ModelBenchmarkResult> { it.app.blindAssistMetrics.alertFalsePositiveRate }
                .thenBy { it.app.blindAssistMetrics.criticalMissCount }
                .thenByDescending { it.app.blindAssistMetrics.distanceBandAccuracy }
                .thenBy { it.app.totalStats.p50Ms }
        )
        val bestObserved = candidates.minWithOrNull(
            compareBy<ModelBenchmarkResult> { it.app.blindAssistMetrics.alertFalsePositiveRate }
                .thenBy { it.app.blindAssistMetrics.criticalMissCount }
                .thenByDescending { it.app.blindAssistMetrics.distanceBandAccuracy }
                .thenBy { it.app.totalStats.p50Ms }
        )
        return if (bestPassing != null) {
            JSONObject()
                .put(
                    "decision",
                    when {
                        bestPassing.spec.depthFusion -> "depth_fusion_candidate_ok_for_next_stage"
                        bestPassing.spec.traversabilityOracle -> "traversability_rules_ok_for_model_stage"
                        else -> "candidate_ok_for_next_stage"
                    }
                )
                .put("replace_default_model_now", false)
                .put("best_candidate", bestPassing.spec.id)
                .put("reason", "${bestPassing.spec.id} passed the active no-regression gate; keep it as a candidate until broader real walking validation confirms the result.")
        } else {
            JSONObject()
                .put("decision", if (candidates.any { it.spec.depthFusion }) "do_not_promote_depth_fusion" else "do_not_replace_default_model")
                .put("replace_default_model_now", false)
                .put("best_observed_candidate", bestObserved?.spec?.id ?: JSONObject.NULL)
                .put("reason", "No candidate met the no-regression gate for detection quality and BlindAssist risk metrics in this run.")
        }
    }

    private fun perImageCsv(modelResults: List<ModelBenchmarkResult>): String {
        val lines = mutableListOf(
            "image,model,gt_count,detection_count,tp,fp,fn,precision,recall,f1,expected_should_alert,actual_alert,expected_risk_level,model_risk_level,expected_direction,model_direction,expected_distance_band,model_proximity,approach_trend,fusion_summary,expected_approach_state,expected_approach_alert,sequence_id,frame_index,distance_evidence_source,distance_evidence_band,distance_evidence_confidence,relative_depth_score,primary_object_id,scene_bucket,risk_key_variants,risk_level_flip,box_jitter"
        )
        modelResults.forEach { model ->
            model.app.perImage.forEach { item ->
                val expected = item.image.expectedRisk
                val evidence = item.modelRisk.distanceEvidence
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
                    expected?.shouldAlert ?: "",
                    item.actualAlert,
                    expected?.riskLevel ?: item.groundTruthRisk.level,
                    item.modelRisk.level,
                    expected?.direction ?: item.groundTruthRisk.direction,
                    item.modelRisk.direction,
                    expected?.distanceBand ?: item.groundTruthRisk.proximity,
                    item.modelRisk.proximity,
                    item.modelRisk.approachTrend,
                    item.modelRisk.scoreBreakdown.fusionSummary,
                    expected?.expectedApproachState ?: "",
                    expected?.expectedApproachAlert ?: "",
                    expected?.sequenceId ?: "",
                    expected?.frameIndex ?: "",
                    evidence?.source ?: "",
                    evidence?.band ?: "",
                    evidence?.confidence?.let { round3(it) } ?: "",
                    evidence?.relativeDepthScore?.let { round3(it) } ?: "",
                    expected?.primaryObjectId ?: "",
                    expected?.sceneBucket ?: "",
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
            appendLine("- Dataset: `${payload.getString("dataset_name")}`")
            appendLine("- Dataset kind: `${payload.getString("dataset_kind")}`")
            appendLine("- Comparison mode: `${payload.getString("comparison_mode")}`")
            appendLine("- Risk config: `${payload.getString("risk_config")}`")
            appendLine("- Depth model asset: `${payload.getString("depth_model_asset")}`")
            appendLine("- Image count: `${payload.getInt("image_count")}`")
            appendLine("- App runs per image: `${payload.getInt("app_runs_per_image")}`")
            appendLine("- Match IoU threshold: `${payload.getDouble("match_iou_threshold")}`")
            appendLine("- Default app model remains: `${payload.getString("default_model_asset")}`")
            appendLine("- Recommendation: `${recommendation.getString("decision")}`")
            appendLine("- Replace default model now: `${recommendation.getBoolean("replace_default_model_now")}`")
            appendLine()
            appendLine("| Model | Depth fusion | AP50 | Precision | Recall | F1 | FP/img | FN/img | Center risk recall | Alert recall | Alert FP rate | Distance acc | Risk level acc | Primary hit | SANPO primary hit | Critical miss | Approach recall | Approach FP | Approach dir acc | Approach critical miss | Mean alert frames | Approach sequences | Fusion summary counts | Depth P50 ms | Total P50 ms | Total P95 ms |")
            appendLine("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
            for (index in 0 until models.length()) {
                val model = models.getJSONObject(index)
                val app = model.getJSONObject("app_detector")
                val quality = app.getJSONObject("quality")
                val assist = app.getJSONObject("blindassist_metrics")
                val depth = app.getJSONObject("depth_ms")
                val total = app.getJSONObject("total_ms")
                val fusionCounts = app.getJSONObject("fusion_summary_counts")
                appendLine(
                    "| `${model.getString("id")}` | `${model.getBoolean("depth_fusion")}` | ${quality.getDouble("ap50")} | ${quality.getDouble("precision")} | " +
                        "${quality.getDouble("recall")} | ${quality.getDouble("f1")} | ${quality.getDouble("fp_per_image")} | " +
                        "${quality.getDouble("fn_per_image")} | ${assist.getDouble("centerRiskRecall")} | " +
                        "${assist.getDouble("alertRecall")} | ${assist.getDouble("alertFalsePositiveRate")} | " +
                        "${assist.getDouble("distanceBandAccuracy")} | ${assist.getDouble("riskLevelAccuracy")} | " +
                        "${assist.getDouble("primaryObjectHitRate")} | ${assist.getDouble("sourcePrimaryRegionHitRate")} | ${assist.getInt("criticalMissCount")} | " +
                        "${assist.getDouble("approachRiskRecall")} | ${assist.getDouble("approachFalsePositiveRate")} | " +
                        "${assist.getDouble("approachDirectionAccuracy")} | ${assist.getInt("approachCriticalMissCount")} | " +
                        "${assist.getDouble("meanTimeToAlertFrames")} | ${assist.getInt("approachLabeledSequenceCount")} | " +
                        "`${fusionCounts.toCompactString()}` | ${depth.getDouble("p50")} | ${total.getDouble("p50")} | ${total.getDouble("p95")} |"
                )
            }
            appendLine()
            appendLine("Decision note: ${recommendation.getString("reason")}")
            appendLine()
            appendLine("Notes:")
            appendLine("- This A/B run compares both models on the same connected device and the same image assets.")
            appendLine("- Detection matching uses same-class IoU at the configured threshold.")
            appendLine("- BlindAssist risk labels are project-specific eval labels and are not a standalone safety guarantee.")
        }
    }

    private fun JSONObject.toCompactString(): String {
        val keys = keys().asSequence().toList().sorted()
        return keys.joinToString(separator = ";") { key -> "$key=${getInt(key)}" }
    }

    private fun sweepSummaryJson(modelResults: List<ModelBenchmarkResult>): JSONArray {
        return JSONArray(sweepRows(modelResults).map { row ->
            JSONObject()
                .put("id", row.result.spec.id)
                .put("depth_closer_is_larger", row.result.spec.depthCloserIsLarger)
                .put("depth_sampling_config", row.result.spec.depthSamplingConfig.toJson())
                .put("passed_gate", row.passedGate)
                .put("alert_false_positive_rate_delta", round3(row.alertFalsePositiveRateDelta))
                .put("critical_miss_delta", row.criticalMissDelta)
                .put("distance_band_accuracy_delta", round3(row.distanceBandAccuracyDelta))
                .put("center_risk_recall_delta", round3(row.centerRiskRecallDelta))
                .put("alert_recall_delta", round3(row.alertRecallDelta))
                .put("depth_p50_ms", row.result.app.depthStats.p50Ms)
                .put("total_p50_ms", row.result.app.totalStats.p50Ms)
                .put("total_p95_ms", row.result.app.totalStats.p95Ms)
        })
    }

    private fun sweepSummaryCsv(modelResults: List<ModelBenchmarkResult>): String {
        val lines = mutableListOf(
            "id,passed_gate,closer_is_larger,sample_percentile,inner_crop_ratio,min_confidence,critical_threshold,near_threshold,mid_threshold,alert_fp_delta,critical_miss_delta,distance_acc_delta,center_risk_recall_delta,alert_recall_delta,depth_p50_ms,total_p50_ms,total_p95_ms"
        )
        sweepRows(modelResults).forEach { row ->
            val config = row.result.spec.depthSamplingConfig
            lines += listOf(
                row.result.spec.id,
                row.passedGate,
                row.result.spec.depthCloserIsLarger,
                config.samplePercentile,
                config.innerCropRatio,
                config.minConfidence,
                config.criticalThreshold,
                config.nearThreshold,
                config.midThreshold,
                round3(row.alertFalsePositiveRateDelta),
                row.criticalMissDelta,
                round3(row.distanceBandAccuracyDelta),
                round3(row.centerRiskRecallDelta),
                round3(row.alertRecallDelta),
                row.result.app.depthStats.p50Ms,
                row.result.app.totalStats.p50Ms,
                row.result.app.totalStats.p95Ms
            ).joinToString(",") { csvValue(it) }
        }
        return lines.joinToString(separator = "\n", postfix = "\n")
    }

    private fun sweepSummaryMarkdown(modelResults: List<ModelBenchmarkResult>): String {
        return buildString {
            appendLine("# Depth Fusion Sweep Summary")
            appendLine()
            appendLine("| Candidate | Pass | FP delta | Critical miss delta | Distance acc delta | Center recall delta | Alert recall delta | Depth P50 ms | Total P50/P95 ms |")
            appendLine("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            sweepRows(modelResults).forEach { row ->
                appendLine(
                    "| `${row.result.spec.id}` | `${row.passedGate}` | ${round3(row.alertFalsePositiveRateDelta)} | " +
                        "${row.criticalMissDelta} | ${round3(row.distanceBandAccuracyDelta)} | " +
                        "${round3(row.centerRiskRecallDelta)} | ${round3(row.alertRecallDelta)} | " +
                        "${row.result.app.depthStats.p50Ms} | ${row.result.app.totalStats.p50Ms}/${row.result.app.totalStats.p95Ms} |"
                )
            }
        }
    }

    private fun sweepRows(modelResults: List<ModelBenchmarkResult>): List<SweepRow> {
        val baseline = modelResults.firstOrNull { it.spec.id == "baseline_geometry" } ?: return emptyList()
        val baselineAssist = baseline.app.blindAssistMetrics
        return modelResults
            .filter { it.spec.depthFusion }
            .map { result ->
                val assist = result.app.blindAssistMetrics
                val alertFalsePositiveRateDelta = assist.alertFalsePositiveRate - baselineAssist.alertFalsePositiveRate
                val criticalMissDelta = assist.criticalMissCount - baselineAssist.criticalMissCount
                val distanceBandAccuracyDelta = assist.distanceBandAccuracy - baselineAssist.distanceBandAccuracy
                val centerRiskRecallDelta = assist.centerRiskRecall - baselineAssist.centerRiskRecall
                val alertRecallDelta = assist.alertRecall - baselineAssist.alertRecall
                SweepRow(
                    result = result,
                    passedGate = alertFalsePositiveRateDelta <= 0.02 &&
                        criticalMissDelta <= 0 &&
                        distanceBandAccuracyDelta >= 0.0 &&
                        centerRiskRecallDelta >= 0.0 &&
                        alertRecallDelta >= 0.0,
                    alertFalsePositiveRateDelta = alertFalsePositiveRateDelta,
                    criticalMissDelta = criticalMissDelta,
                    distanceBandAccuracyDelta = distanceBandAccuracyDelta,
                    centerRiskRecallDelta = centerRiskRecallDelta,
                    alertRecallDelta = alertRecallDelta
                )
            }
            .sortedWith(
                compareByDescending<SweepRow> { if (it.passedGate) 1 else 0 }
                    .thenBy { it.alertFalsePositiveRateDelta }
                    .thenBy { it.criticalMissDelta }
                    .thenByDescending { it.distanceBandAccuracyDelta }
                    .thenBy { it.result.app.totalStats.p50Ms }
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

    private fun intersectionOverActual(actual: BoundingBox, expected: BoundingBox): Float {
        val left = max(actual.left, expected.left)
        val top = max(actual.top, expected.top)
        val right = min(actual.right, expected.right)
        val bottom = min(actual.bottom, expected.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val actualArea = actual.width * actual.height
        return if (actualArea <= 0f) 0f else intersection / actualArea
    }

    private fun decodeSemanticMask(assetName: String): DenseSemanticMask {
        val original = decodeBitmap(assetName)
        val bitmap = if (original.width == TRAVERSABILITY_MASK_SIZE && original.height == TRAVERSABILITY_MASK_SIZE) {
            original
        } else {
            Bitmap.createScaledBitmap(original, TRAVERSABILITY_MASK_SIZE, TRAVERSABILITY_MASK_SIZE, false)
        }
        try {
            val pixels = IntArray(bitmap.width * bitmap.height)
            bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            return DenseSemanticMask(
                width = bitmap.width,
                height = bitmap.height,
                classIds = IntArray(pixels.size) { index -> (pixels[index] shr 16) and 0xFF }
            )
        } finally {
            bitmap.recycle()
            if (bitmap !== original) original.recycle()
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

    private fun round3(value: Float): Double = round3(value.toDouble())

    private fun csvValue(value: Any): String {
        val text = value.toString()
        return if (text.any { it == ',' || it == '"' || it == '\n' }) {
            "\"" + text.replace("\"", "\"\"") + "\""
        } else {
            text
        }
    }

    private fun DepthEvidenceSamplingConfig.toJson(): JSONObject {
        return JSONObject()
            .put("sample_percentile", round3(samplePercentile))
            .put("inner_crop_ratio", round3(innerCropRatio))
            .put("lower_half_only", lowerHalfOnly)
            .put("min_samples", minSamples)
            .put("min_local_range", round3(minLocalRange))
            .put("min_confidence", round3(minConfidence))
            .put("critical_threshold", round3(criticalThreshold))
            .put("near_threshold", round3(nearThreshold))
            .put("mid_threshold", round3(midThreshold))
    }

    private fun depthSamplingConfigFromArguments(): DepthEvidenceSamplingConfig {
        return DepthEvidenceSamplingConfig(
            samplePercentile = arguments.getString(ARG_DEPTH_SAMPLE_PERCENTILE)?.toFloatOrNull() ?: 0.50f,
            innerCropRatio = arguments.getString(ARG_DEPTH_INNER_CROP_RATIO)?.toFloatOrNull() ?: 1.0f,
            lowerHalfOnly = arguments.getString(ARG_DEPTH_LOWER_HALF_ONLY)?.toBooleanStrictOrNull() ?: true,
            minSamples = arguments.getString(ARG_DEPTH_MIN_SAMPLES)?.toIntOrNull() ?: 4,
            minLocalRange = arguments.getString(ARG_DEPTH_MIN_LOCAL_RANGE)?.toFloatOrNull() ?: 0f,
            minConfidence = arguments.getString(ARG_DEPTH_MIN_CONFIDENCE)?.toFloatOrNull() ?: 0.55f,
            criticalThreshold = arguments.getString(ARG_DEPTH_CRITICAL_THRESHOLD)?.toFloatOrNull() ?: 0.78f,
            nearThreshold = arguments.getString(ARG_DEPTH_NEAR_THRESHOLD)?.toFloatOrNull() ?: 0.58f,
            midThreshold = arguments.getString(ARG_DEPTH_MID_THRESHOLD)?.toFloatOrNull() ?: 0.35f
        )
    }

    private fun Bitmap.use(block: (Bitmap) -> Unit) {
        try {
            block(this)
        } finally {
            recycle()
        }
    }

    private fun JSONObject.optionalBoolean(name: String): Boolean? {
        return if (has(name) && !isNull(name)) getBoolean(name) else null
    }

    private fun JSONObject.optionalInt(name: String): Int? {
        return if (has(name) && !isNull(name)) getInt(name) else null
    }

    private fun JSONObject.optionalString(name: String): String? {
        return if (has(name) && !isNull(name)) getString(name).takeIf { it.isNotBlank() } else null
    }

    private fun JSONObject.optionalApproachTrend(name: String): ApproachTrend? {
        val value = optionalString(name) ?: return null
        return runCatching { ApproachTrend.valueOf(value) }.getOrNull()
    }

    private data class ModelSpec(
        val id: String,
        val assetName: String,
        val source: String,
        val depthFusion: Boolean = false,
        val depthModelAssetName: String = "",
        val depthCloserIsLarger: Boolean = true,
        val depthSamplingConfig: DepthEvidenceSamplingConfig = DepthEvidenceSamplingConfig(),
        val traversabilityOracle: Boolean = false,
        val riskAnalyzerConfig: RiskAnalyzerConfig = RiskAnalyzerConfig.Default
    )

    private data class BenchmarkDataset(
        val name: String,
        val images: List<BenchmarkImage>
    )

    private fun List<BenchmarkImage>.sortedForTemporalEvaluation(): List<BenchmarkImage> {
        return sortedWith(
            compareBy<BenchmarkImage> { it.expectedRisk?.sequenceId ?: "\uFFFF" }
                .thenBy { it.expectedRisk?.frameIndex ?: Int.MAX_VALUE }
                .thenBy { it.fileName }
        )
    }

    private data class BenchmarkImage(
        val fileName: String,
        val relativePath: String,
        val frameSize: FrameSize,
        val groundTruth: List<GroundTruthBox>,
        val sourceRegions: List<SourceRegion> = emptyList(),
        val sourcePrimaryRegionId: String? = null,
        val sourceMaskPath: String? = null,
        val expectedRisk: BlindAssistExpectedRisk?
    ) {
        fun groundTruthDetections(): List<Detection> {
            return groundTruth.map { it.toDetection(frameSize) }
        }

        fun primaryGroundTruthDetection(): Detection? {
            val primaryId = expectedRisk?.primaryObjectId ?: return null
            return groundTruth.firstOrNull { it.id == primaryId }?.toDetection(frameSize)
        }
    }

    private data class SourceRegion(
        val id: String,
        val classId: Int,
        val label: String,
        val box: BoundingBox
    ) {
    }

    private data class GroundTruthBox(
        val id: String,
        val classId: Int,
        val label: String,
        val box: BoundingBox
    ) {
        fun toDetection(frameSize: FrameSize): Detection {
            return Detection(
                classId = classId,
                label = label,
                confidence = 1f,
                boundingBox = box,
                frameSize = frameSize
            )
        }
    }

    private data class BlindAssistExpectedRisk(
        val direction: RiskDirection,
        val distanceBand: ProximityBand,
        val shouldAlert: Boolean,
        val riskLevel: RiskLevel,
        val scenario: AssistScenario,
        val primaryObjectId: String?,
        val sceneBucket: String,
        val sequenceId: String? = null,
        val frameIndex: Int? = null,
        val expectedApproachState: ApproachTrend? = null,
        val expectedApproachAlert: Boolean? = null,
        val expectedTimeToAlertFrames: Int? = null
    ) {
        val hasApproachLabel: Boolean
            get() = sequenceId != null &&
                frameIndex != null &&
                (
                    expectedApproachState != null ||
                        expectedApproachAlert != null ||
                        expectedTimeToAlertFrames != null
                    )

        fun toJson(): JSONObject {
            return JSONObject()
                .put("expected_risk_direction", direction.name)
                .put("expected_distance_band", distanceBand.name)
                .put("expected_should_alert", shouldAlert)
                .put("expected_risk_level", riskLevel.name)
                .put("assist_scenario", scenario.name)
                .put("primary_object_id", primaryObjectId ?: JSONObject.NULL)
                .put("scene_bucket", sceneBucket)
                .put("sequence_id", sequenceId ?: JSONObject.NULL)
                .put("frame_index", frameIndex ?: JSONObject.NULL)
                .put("expected_approach_state", expectedApproachState?.name ?: JSONObject.NULL)
                .put("expected_approach_alert", expectedApproachAlert ?: JSONObject.NULL)
                .put("expected_time_to_alert_frames", expectedTimeToAlertFrames ?: JSONObject.NULL)
        }
    }

    private data class FrameRun(
        val detections: List<Detection>,
        val risk: RiskResult,
        val preprocessMs: Double,
        val inferenceMs: Double,
        val postprocessMs: Double,
        val depthMs: Double,
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
                .put("depth_fusion", spec.depthFusion)
                .put(
                    "depth_model_asset",
                    if (spec.depthModelAssetName.isBlank()) JSONObject.NULL else spec.depthModelAssetName
                )
                .put("depth_closer_is_larger", spec.depthCloserIsLarger)
                .put("traversability_oracle", spec.traversabilityOracle)
                .put(
                    "depth_sampling_config",
                    JSONObject()
                        .put("sample_percentile", spec.depthSamplingConfig.samplePercentile.toDouble())
                        .put("inner_crop_ratio", spec.depthSamplingConfig.innerCropRatio.toDouble())
                        .put("lower_half_only", spec.depthSamplingConfig.lowerHalfOnly)
                        .put("min_samples", spec.depthSamplingConfig.minSamples)
                        .put("min_local_range", spec.depthSamplingConfig.minLocalRange.toDouble())
                        .put("min_confidence", spec.depthSamplingConfig.minConfidence.toDouble())
                        .put("critical_threshold", spec.depthSamplingConfig.criticalThreshold.toDouble())
                        .put("near_threshold", spec.depthSamplingConfig.nearThreshold.toDouble())
                        .put("mid_threshold", spec.depthSamplingConfig.midThreshold.toDouble())
                )
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
        val depthStats: Stats,
        val totalStats: Stats,
        val quality: QualitySummary,
        val riskQuality: RiskQualitySummary,
        val blindAssistMetrics: BlindAssistMetrics,
        val stability: StabilityAggregate,
        val fusionSummaryCounts: Map<String, Int>,
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
                .put("depth_ms", depthStats.toJson())
                .put("total_ms", totalStats.toJson())
                .put("quality", quality.toJson())
                .put("risk_quality", riskQuality.toJson())
                .put("blindassist_metrics", blindAssistMetrics.toJson())
                .put("stability", stability.toJson())
                .put("fusion_summary_counts", JSONObject(fusionSummaryCounts))
                .put("per_image", JSONArray(perImage.map { it.toJson() }))
                .put("failures", JSONArray(failures))
        }
    }

    private data class PerImageModelResult(
        val image: BenchmarkImage,
        val evaluation: DetectionEvaluation,
        val groundTruthRisk: RiskResult,
        val modelRisk: RiskResult,
        val actualAlert: Boolean,
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
                .put("expected_blindassist", image.expectedRisk?.toJson() ?: JSONObject.NULL)
                .put("actual_alert", actualAlert)
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
                .put("expected_blindassist", image.expectedRisk?.toJson() ?: JSONObject.NULL)
                .put("ground_truth_risk", riskToJson(groundTruthRisk))
                .put("model_risk", riskToJson(modelRisk))
                .put("actual_alert", actualAlert)
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

    private data class BlindAssistMetrics(
        val centerRiskRecall: Double,
        val alertRecall: Double,
        val alertFalsePositiveRate: Double,
        val distanceBandAccuracy: Double,
        val riskLevelAccuracy: Double,
        val primaryObjectHitRate: Double,
        val sourcePrimaryRegionHitRate: Double,
        val criticalMissCount: Int,
        val approachRiskRecall: Double,
        val approachFalsePositiveRate: Double,
        val approachDirectionAccuracy: Double,
        val approachCriticalMissCount: Int,
        val meanTimeToAlertFrames: Double,
        val approachLabeledSequenceCount: Int,
        val labeledImageCount: Int
    ) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("centerRiskRecall", centerRiskRecall)
                .put("alertRecall", alertRecall)
                .put("alertFalsePositiveRate", alertFalsePositiveRate)
                .put("distanceBandAccuracy", distanceBandAccuracy)
                .put("riskLevelAccuracy", riskLevelAccuracy)
                .put("primaryObjectHitRate", primaryObjectHitRate)
                .put("sourcePrimaryRegionHitRate", sourcePrimaryRegionHitRate)
                .put("criticalMissCount", criticalMissCount)
                .put("approachRiskRecall", approachRiskRecall)
                .put("approachFalsePositiveRate", approachFalsePositiveRate)
                .put("approachDirectionAccuracy", approachDirectionAccuracy)
                .put("approachCriticalMissCount", approachCriticalMissCount)
                .put("meanTimeToAlertFrames", meanTimeToAlertFrames)
                .put("approachLabeledSequenceCount", approachLabeledSequenceCount)
                .put("labeledImageCount", labeledImageCount)
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

    private data class SweepRow(
        val result: ModelBenchmarkResult,
        val passedGate: Boolean,
        val alertFalsePositiveRateDelta: Double,
        val criticalMissDelta: Int,
        val distanceBandAccuracyDelta: Double,
        val centerRiskRecallDelta: Double,
        val alertRecallDelta: Double
    )

    companion object {
        private const val TAG = "DetectorAbBenchmark"
        private const val YOLO11N_ASSET = "yolo11n_fp16_320.tflite"
        private const val YOLO26N_ASSET = "yolo26n_fp16_320.tflite"
        private const val COCO_ANNOTATIONS_ASSET = "coco100_annotations.json"
        private const val BLINDASSIST_EVALSET_ASSET_PREFIX = "blindassist_evalset"
        private const val BLINDASSIST_EVALSET_MANIFEST_ASSET = "$BLINDASSIST_EVALSET_ASSET_PREFIX/manifest.jsonl"
        private const val BLINDASSIST_EVALSET_SPEC_ASSET = "$BLINDASSIST_EVALSET_ASSET_PREFIX/dataset_spec.json"
        private const val DATASET_KIND_COCO100 = "Coco100"
        private const val DATASET_KIND_BLINDASSIST_EVALSET = "BlindAssistEvalSet"
        private const val COMPARISON_MODE_DETECTOR_AB = "DetectorAb"
        private const val COMPARISON_MODE_DEPTH_FUSION = "DepthFusion"
        private const val COMPARISON_MODE_DEPTH_FUSION_SWEEP = "DepthFusionSweep"
        private const val COMPARISON_MODE_SANPO_TRAVERSABILITY_ORACLE = "SanpoTraversabilityOracle"
        private const val RISK_CONFIG_CURRENT = "current"
        private const val RISK_CONFIG_CENTER_NEAR_SENSITIVE = "center_near_sensitive"
        private const val RISK_CONFIG_CENTER_NEAR_STRICT = "center_near_strict"
        private const val RISK_CONFIG_CRITICAL_SENSITIVE = "critical_sensitive"
        private const val RISK_CONFIG_SIDE_NEAR_SENSITIVE = "side_near_sensitive"
        private const val TEMPORAL_FRAME_STEP_MS = 100L
        private const val TRAVERSABILITY_MASK_SIZE = 512
        private const val ARG_DATASET_KIND = "datasetKind"
        private const val ARG_RISK_CONFIG = "riskConfig"
        private const val ARG_COMPARISON_MODE = "comparisonMode"
        private const val ARG_DEPTH_MODEL_ASSET = "depthModelAsset"
        private const val ARG_DEPTH_CLOSER_IS_LARGER = "depthCloserIsLarger"
        private const val ARG_DEPTH_SAMPLE_PERCENTILE = "depthSamplePercentile"
        private const val ARG_DEPTH_INNER_CROP_RATIO = "depthInnerCropRatio"
        private const val ARG_DEPTH_LOWER_HALF_ONLY = "depthLowerHalfOnly"
        private const val ARG_DEPTH_MIN_SAMPLES = "depthMinSamples"
        private const val ARG_DEPTH_MIN_LOCAL_RANGE = "depthMinLocalRange"
        private const val ARG_DEPTH_MIN_CONFIDENCE = "depthMinConfidence"
        private const val ARG_DEPTH_CRITICAL_THRESHOLD = "depthCriticalThreshold"
        private const val ARG_DEPTH_NEAR_THRESHOLD = "depthNearThreshold"
        private const val ARG_DEPTH_MID_THRESHOLD = "depthMidThreshold"
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
        private fun modelSpecsFor(
            comparisonMode: String,
            depthModelAsset: String,
            depthCloserIsLarger: Boolean,
            baseRiskAnalyzerConfig: RiskAnalyzerConfig,
            singleDepthSamplingConfig: DepthEvidenceSamplingConfig
        ): List<ModelSpec> {
            val conservativeDepthRiskConfig = baseRiskAnalyzerConfig.copy(distanceEvidenceMaxPromotionSteps = 1)
            return when (comparisonMode) {
                COMPARISON_MODE_DETECTOR_AB -> listOf(
                    ModelSpec(
                        "yolo11n",
                        YOLO11N_ASSET,
                        "app/src/main/assets/yolo11n_fp16_320.tflite",
                        riskAnalyzerConfig = baseRiskAnalyzerConfig
                    ),
                    ModelSpec(
                        "yolo26n",
                        YOLO26N_ASSET,
                        ".downloads/detector-lab/exports/yolo26n_fp16_320.tflite",
                        riskAnalyzerConfig = baseRiskAnalyzerConfig
                    )
                )
                COMPARISON_MODE_DEPTH_FUSION -> listOf(
                    ModelSpec(
                        "baseline_geometry",
                        YOLO11N_ASSET,
                        "app/src/main/assets/yolo11n_fp16_320.tflite",
                        riskAnalyzerConfig = baseRiskAnalyzerConfig
                    ),
                    ModelSpec(
                        id = "candidate_depth_fusion",
                        assetName = YOLO11N_ASSET,
                        source = "app/src/main/assets/yolo11n_fp16_320.tflite + device-benchmark assets/$depthModelAsset",
                        depthFusion = true,
                        depthModelAssetName = depthModelAsset,
                        depthCloserIsLarger = depthCloserIsLarger,
                        depthSamplingConfig = singleDepthSamplingConfig,
                        riskAnalyzerConfig = conservativeDepthRiskConfig
                    )
                )
                COMPARISON_MODE_DEPTH_FUSION_SWEEP -> {
                    val baseline = ModelSpec(
                        "baseline_geometry",
                        YOLO11N_ASSET,
                        "app/src/main/assets/yolo11n_fp16_320.tflite",
                        riskAnalyzerConfig = baseRiskAnalyzerConfig
                    )
                    listOf(baseline) + depthFusionSweepSpecs(depthModelAsset, conservativeDepthRiskConfig)
                }
                COMPARISON_MODE_SANPO_TRAVERSABILITY_ORACLE -> listOf(
                    ModelSpec(
                        id = "baseline_yolo_geometry",
                        assetName = YOLO11N_ASSET,
                        source = "app/src/main/assets/yolo11n_fp16_320.tflite",
                        riskAnalyzerConfig = baseRiskAnalyzerConfig
                    ),
                    ModelSpec(
                        id = "candidate_sanpo_traversability_oracle",
                        assetName = YOLO11N_ASSET,
                        source = "YOLO11n + SANPO source_regions oracle (benchmark only)",
                        traversabilityOracle = true,
                        riskAnalyzerConfig = baseRiskAnalyzerConfig
                    )
                )
                else -> error("Unsupported comparisonMode: $comparisonMode")
            }
        }

        private fun depthFusionSweepSpecs(
            depthModelAsset: String,
            riskAnalyzerConfig: RiskAnalyzerConfig
        ): List<ModelSpec> {
            val thresholdSets = listOf(
                ThresholdSet("strict", 0.86f, 0.68f, 0.45f),
                ThresholdSet("balanced", 0.80f, 0.62f, 0.40f),
                ThresholdSet("current", 0.78f, 0.58f, 0.35f)
            )
            val specs = mutableListOf<ModelSpec>()
            specs += depthFusionSpec(
                id = "fastdepth_orig_p50_conf055_c100_t78_58_35",
                depthModelAsset = depthModelAsset,
                closerIsLarger = true,
                config = DepthEvidenceSamplingConfig(),
                riskAnalyzerConfig = riskAnalyzerConfig
            )
            for (samplePercentile in listOf(0.50f, 0.25f, 0.10f)) {
                for (innerCropRatio in listOf(0.50f, 0.60f)) {
                    for (minConfidence in listOf(0.65f, 0.75f)) {
                        for (threshold in thresholdSets) {
                            val config = DepthEvidenceSamplingConfig(
                                samplePercentile = samplePercentile,
                                innerCropRatio = innerCropRatio,
                                lowerHalfOnly = false,
                                minSamples = 4,
                                minLocalRange = 0.05f,
                                minConfidence = minConfidence,
                                criticalThreshold = threshold.critical,
                                nearThreshold = threshold.near,
                                midThreshold = threshold.mid
                            )
                            specs += depthFusionSpec(
                                id = "fastdepth_inv_p${percentCode(samplePercentile)}_conf${percentCode(minConfidence)}_c${percentCode(innerCropRatio)}_t${percentCode(threshold.critical)}_${percentCode(threshold.near)}_${percentCode(threshold.mid)}",
                                depthModelAsset = depthModelAsset,
                                closerIsLarger = false,
                                config = config,
                                riskAnalyzerConfig = riskAnalyzerConfig
                            )
                        }
                    }
                }
            }
            return specs
        }

        private fun depthFusionSpec(
            id: String,
            depthModelAsset: String,
            closerIsLarger: Boolean,
            config: DepthEvidenceSamplingConfig,
            riskAnalyzerConfig: RiskAnalyzerConfig
        ): ModelSpec {
            return ModelSpec(
                id = id,
                assetName = YOLO11N_ASSET,
                source = "app/src/main/assets/yolo11n_fp16_320.tflite + device-benchmark assets/$depthModelAsset",
                depthFusion = true,
                depthModelAssetName = depthModelAsset,
                depthCloserIsLarger = closerIsLarger,
                depthSamplingConfig = config,
                riskAnalyzerConfig = riskAnalyzerConfig
            )
        }

        private data class ThresholdSet(
            val name: String,
            val critical: Float,
            val near: Float,
            val mid: Float
        )

        private fun percentCode(value: Float): String {
            return (value * 100f).roundToInt().toString().padStart(2, '0')
        }
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
            val evidence = risk.distanceEvidence
            return JSONObject()
                .put("level", risk.level.name)
                .put("direction", risk.direction.name)
                .put("proximity", risk.proximity.name)
                .put("source_label", risk.sourceDetection?.label ?: JSONObject.NULL)
                .put("source_confidence", risk.sourceDetection?.confidence?.let { round3(it) } ?: JSONObject.NULL)
                .put("urgency_score", round3(risk.urgencyScore))
                .put("risk_score", round3(risk.riskScore))
                .put("approach_trend", risk.approachTrend.name)
                .put("fusion_summary", risk.scoreBreakdown.fusionSummary)
                .put(
                    "score_breakdown",
                    JSONObject()
                        .put("confidence", round3(risk.scoreBreakdown.confidence))
                        .put("class_weight", round3(risk.scoreBreakdown.classWeight))
                        .put("direction_weight", round3(risk.scoreBreakdown.directionWeight))
                        .put("proximity_weight", round3(risk.scoreBreakdown.proximityWeight))
                        .put("bottom_position", round3(risk.scoreBreakdown.bottomPosition))
                        .put("area", round3(risk.scoreBreakdown.area))
                        .put("center_lane", round3(risk.scoreBreakdown.centerLane))
                        .put("distance_evidence", round3(risk.scoreBreakdown.distanceEvidence))
                        .put("approach_trend", round3(risk.scoreBreakdown.approachTrend))
                        .put("total", round3(risk.scoreBreakdown.total))
                        .put("fusion_summary", risk.scoreBreakdown.fusionSummary)
                )
                .put(
                    "distance_evidence",
                    if (evidence == null) {
                        JSONObject.NULL
                    } else {
                        JSONObject()
                            .put("band", evidence.band.name)
                            .put("confidence", round3(evidence.confidence))
                            .put("source", evidence.source.name)
                            .put("relative_depth_score", round3(evidence.relativeDepthScore))
                    }
                )
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

        private fun ratio(numerator: Int, denominator: Int): Double {
            return if (denominator <= 0) 0.0 else round3(numerator.toDouble() / denominator.toDouble())
        }

        private fun riskAnalyzerConfigFor(name: String): RiskAnalyzerConfig {
            return when (name) {
                RISK_CONFIG_CURRENT -> RiskAnalyzerConfig.Default
                RISK_CONFIG_CENTER_NEAR_SENSITIVE -> RiskAnalyzerConfig.CenterNearSensitive
                RISK_CONFIG_CENTER_NEAR_STRICT -> RiskAnalyzerConfig.CenterNearStrict
                RISK_CONFIG_CRITICAL_SENSITIVE -> RiskAnalyzerConfig.CriticalSensitive
                RISK_CONFIG_SIDE_NEAR_SENSITIVE -> RiskAnalyzerConfig.SideNearSensitive
                else -> error("Unsupported riskConfig: $name")
            }
        }
    }
}
