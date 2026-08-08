package com.linnan.blindassist.benchmark

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.PowerManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackPlanner
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.DenseSemanticMask
import com.linnan.blindassist.risk.RiskAnalyzer
import com.linnan.blindassist.risk.TraversabilityClass
import com.linnan.blindassist.risk.TraversabilitySegmentationAnalyzer
import com.linnan.blindassist.risk.TraversabilityTaxonomy
import com.linnan.blindassist.session.AssistDecisionKernel
import com.linnan.blindassist.session.AssistEngine
import com.linnan.blindassist.session.DetectorMetrics
import com.qualcomm.qti.QnnDelegate
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
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
import java.security.MessageDigest
import java.time.Instant
import java.util.zip.ZipFile
import kotlin.math.ceil
import kotlin.math.roundToInt

/**
 * Isolated RISKSEG-R0 technical preflight. This class is packaged only in the
 * benchmark test APK and cannot change the production application's YOLO route.
 */
@RunWith(AndroidJUnit4::class)
class RisksegPidnetQnnPreflightDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun runRisksegPidnetQnnPreflight() {
        val durationMs = arguments.getString(ARG_DURATION_MS)?.toLongOrNull()
            ?: DEFAULT_SMOKE_DURATION_MS
        require(durationMs in MIN_DURATION_MS..MAX_DURATION_MS) {
            "$ARG_DURATION_MS must be in $MIN_DURATION_MS..$MAX_DURATION_MS"
        }
        val runRole = arguments.getString(ARG_RUN_ROLE) ?: RUN_ROLE_TECHNICAL_PREFLIGHT
        require(runRole == RUN_ROLE_TECHNICAL_PREFLIGHT || runRole == RUN_ROLE_TRAINED_FINAL) {
            "$ARG_RUN_ROLE must be $RUN_ROLE_TECHNICAL_PREFLIGHT or $RUN_ROLE_TRAINED_FINAL"
        }
        val formalSustainedRun = durationMs >= FORMAL_DURATION_MS
        val modelSha256 = sha256Asset(MODEL_ASSET)
        val modelToken = "${MODEL_TOKEN_PREFIX}_${modelSha256.take(16)}"
        val cacheDir = File(
            targetContext.codeCacheDir,
            "${CACHE_DIR_PREFIX}-${modelSha256.take(16)}"
        ).apply { mkdirs() }
        val cacheBefore = cacheInventory(cacheDir)
        val canarySha256 = sha256Asset(CANARY_ASSET)
        val natural = decodeBitmap(CANARY_ASSET)
        val synthetic = Bitmap.createBitmap(INPUT_WIDTH, INPUT_HEIGHT, Bitmap.Config.ARGB_8888)
        val analyzer = TraversabilitySegmentationAnalyzer(taxonomy = RisksegR0Taxonomy)
        val gateway = PlannerGateway()
        val kernel = AssistDecisionKernel(assistEngine = AssistEngine(riskAnalyzer = RiskAnalyzer()))
        val power = targetContext.getSystemService(PowerManager::class.java)

        var firstRuntime: QnnRuntime? = null
        var sustainedRuntime: QnnRuntime? = null
        try {
            firstRuntime = openRuntime(cacheDir, modelToken)
            val syntheticCanary = firstRuntime.runPipeline(
                source = synthetic,
                analyzer = analyzer,
                kernel = kernel,
                gateway = gateway,
                nowMs = 0L
            )
            kernel.reset()
            gateway.resetSession()
            val naturalCanary = firstRuntime.runPipeline(
                source = natural,
                analyzer = analyzer,
                kernel = kernel,
                gateway = gateway,
                nowMs = FRAME_STEP_MS
            )
            val firstProfiling = firstRuntime.profilingResult()
            firstRuntime.close()
            firstRuntime = null

            val cacheAfterCompile = cacheInventory(cacheDir)
            assertTrue(
                "QNN delegate did not emit a reusable cached context in ${cacheDir.absolutePath}",
                cacheAfterCompile.isNotEmpty()
            )

            sustainedRuntime = openRuntime(cacheDir, modelToken)
            repeat(WARMUP_RUNS) { index ->
                sustainedRuntime.runPipeline(
                    source = natural,
                    analyzer = analyzer,
                    kernel = kernel,
                    gateway = gateway,
                    nowMs = (index + 2L) * FRAME_STEP_MS
                )
            }
            kernel.reset()
            gateway.resetSession()

            val samples = mutableListOf<PipelineSample>()
            val thermalStatuses = mutableListOf<Int>()
            val failures = mutableListOf<String>()
            val startedNs = System.nanoTime()
            var index = 0L
            while (elapsedMs(startedNs) < durationMs.toDouble()) {
                try {
                    val sample = sustainedRuntime.runPipeline(
                        source = natural,
                        analyzer = analyzer,
                        kernel = kernel,
                        gateway = gateway,
                        nowMs = index * FRAME_STEP_MS
                    )
                    samples += sample.copy(elapsedSinceStartMs = elapsedMs(startedNs))
                    thermalStatuses += currentThermalStatus(power)
                } catch (error: Throwable) {
                    failures += "${error.javaClass.name}:${error.message}"
                    break
                }
                index += 1
            }
            val secondProfiling = sustainedRuntime.profilingResult()
            val cacheAfterReuse = cacheInventory(cacheDir)
            assertTrue("No timed pipeline samples were recorded", samples.isNotEmpty())
            assertTrue("Timed pipeline failures: $failures", failures.isEmpty())

            val totalStats = stats(samples.map { it.totalMs })
            val initialWindow = samples.filter {
                it.elapsedSinceStartMs <= minOf(INITIAL_WINDOW_MS.toDouble(), durationMs.toDouble())
            }
            val finalWindow = samples.filter {
                it.elapsedSinceStartMs >= maxOf(0.0, durationMs - FINAL_WINDOW_MS.toDouble())
            }
            val initialP95 = stats(initialWindow.map { it.totalMs }).p95
            val finalP95 = stats(finalWindow.map { it.totalMs }).p95
            val degradationRatio = if (initialP95 == 0.0) 0.0 else finalP95 / initialP95
            val maximumThermalStatus = thermalStatuses.maxOrNull()
                ?: PowerManager.THERMAL_STATUS_NONE
            val allClasses = samples.flatMap { it.uniqueClasses }.distinct().sorted()

            if (formalSustainedRun) {
                assertTrue("total P95 ${totalStats.p95}ms exceeds 100ms", totalStats.p95 <= 100.0)
                assertTrue(
                    "final/initial P95 degradation $degradationRatio exceeds 1.20x",
                    degradationRatio <= 1.20
                )
                assertTrue(
                    "severe thermal status observed: $maximumThermalStatus",
                    maximumThermalStatus < PowerManager.THERMAL_STATUS_SEVERE
                )
            }
            assertTrue("argmax class outside 0..3: $allClasses", allClasses.all { it in 0..3 })

            val report = JSONObject()
                .put("schema_version", "blindassist.riskseg_r0.pidnet_qnn_preflight.v1")
                .put("protocol_id", "RISKSEG-R0")
                .put("created_at_utc", Instant.now().toString())
                .put(
                    "status",
                    if (formalSustainedRun) {
                        if (runRole == RUN_ROLE_TRAINED_FINAL) {
                            "QNN_HTP_FORMAL_SUSTAINED_TRAINED_FINAL_PASS"
                        } else {
                            "QNN_HTP_FORMAL_SUSTAINED_PREFLIGHT_PASS"
                        }
                    } else {
                        if (runRole == RUN_ROLE_TRAINED_FINAL) {
                            "QNN_HTP_TRAINED_FINAL_SMOKE_PASS"
                        } else {
                            "QNN_HTP_SMOKE_PASS"
                        }
                    }
                )
                .put("run_role", runRole)
                .put("formal_sustained_run", formalSustainedRun)
                .put("duration_requested_ms", durationMs)
                .put("duration_observed_ms", samples.last().elapsedSinceStartMs)
                .put("sample_count", samples.size)
                .put("failure_count", failures.size)
                .put("failures", JSONArray(failures))
                .put("device", deviceJson())
                .put("delegate", JSONObject()
                    .put("backend", "QNN_HTP")
                    .put("precision", "HTP_PRECISION_QUANTIZED")
                    .put("capability", "HTP_RUNTIME_QUANTIZED")
                    .put("capability_available", true)
                    .put("delegate_available", true)
                    .put("version", JSONArray(QnnDelegate.getVersion().toList()))
                    .put("profiling_first_compile_size_bytes", firstProfiling.size)
                    .put("profiling_first_compile_sha256", sha256(firstProfiling))
                    .put("profiling_reuse_size_bytes", secondProfiling.size)
                    .put("profiling_reuse_sha256", sha256(secondProfiling)))
                .put("model", JSONObject()
                    .put("asset", MODEL_ASSET)
                    .put("sha256", modelSha256)
                    .put("class_order", JSONArray(CLASS_ORDER))
                    .put("input_shape", JSONArray(INPUT_SHAPE.toList()))
                    .put("output_shape", JSONArray(OUTPUT_SHAPE.toList()))
                    .put("input_dtype", "INT8")
                    .put("output_dtype", "INT8")
                    .put("normalization", "(rgb/255 - ImageNet mean) / ImageNet std"))
                .put("canaries", JSONObject()
                    .put("synthetic_zero_rgb", syntheticCanary.toJson())
                    .put("train_rgb_non_eval", naturalCanary.toJson()
                        .put("asset_sha256", canarySha256)))
                .put("pipeline_contract", JSONObject()
                    .put("preprocess", "bitmap_resize_rgb_imagenet_normalize_int8_quantize")
                    .put("inference", "LiteRT_QNN_HTP_quantized")
                    .put("postprocess", "int8_dequantize_finite_check_argmax")
                    .put("mask_adapter", "frozen_TraversabilitySegmentationAnalyzer")
                    .put("risk_event_chain", "frozen_AssistDecisionKernel")
                    .put("decision_kernel_contract_id", AssistDecisionKernel.CONTRACT_ID)
                    .put("class_binding", "benchmark_only_RisksegR0Taxonomy"))
                .put("timing_ms", JSONObject()
                    .put("preprocess", stats(samples.map { it.preprocessMs }).toJson())
                    .put("inference", stats(samples.map { it.inferenceMs }).toJson())
                    .put("dequantize_argmax", stats(samples.map { it.postprocessMs }).toJson())
                    .put("mask_adapter", stats(samples.map { it.adapterMs }).toJson())
                    .put("decision_event_chain", stats(samples.map { it.decisionMs }).toJson())
                    .put("total", totalStats.toJson())
                    .put("initial_window_p95", initialP95)
                    .put("final_window_p95", finalP95)
                    .put("final_over_initial_p95_ratio", degradationRatio))
                .put("thermal", JSONObject()
                    .put("maximum_status", maximumThermalStatus)
                    .put("severe_or_critical_observed",
                        maximumThermalStatus >= PowerManager.THERMAL_STATUS_SEVERE)
                    .put("samples", thermalStatuses.size))
                .put("argmax_unique_classes", JSONArray(allClasses))
                .put("qnn_cached_context", JSONObject()
                    .put("cache_dir", cacheDir.absolutePath)
                    .put("model_token", modelToken)
                    .put("before", JSONArray(cacheBefore.map { it.toJson() }))
                    .put("after_first_compile", JSONArray(cacheAfterCompile.map { it.toJson() }))
                    .put("after_reuse", JSONArray(cacheAfterReuse.map { it.toJson() })))
                .put("gates", JSONObject()
                    .put("failure_count_zero", failures.isEmpty())
                    .put("total_p95_at_most_100_ms", totalStats.p95 <= 100.0)
                    .put("degradation_at_most_1_20x", degradationRatio <= 1.20)
                    .put("no_severe_thermal",
                        maximumThermalStatus < PowerManager.THERMAL_STATUS_SEVERE)
                    .put("qnn_cached_context_created", cacheAfterCompile.isNotEmpty())
                    .put("strict_int8_tensor_contract", true)
                    .put("argmax_in_0_to_3", allClasses.all { it in 0..3 }))

            val artifactDir = File(
                requireNotNull(targetContext.getExternalFilesDir(null)),
                if (runRole == RUN_ROLE_TRAINED_FINAL) {
                    "riskseg-r0-pidnet-qnn-trained-final"
                } else {
                    "riskseg-r0-pidnet-qnn-preflight"
                }
            ).apply { mkdirs() }
            File(artifactDir, REPORT_FILE).writeText(report.toString(2), Charsets.UTF_8)
            instrumentation.sendStatus(
                0,
                android.os.Bundle().apply {
                    putString("riskseg_report_path", File(artifactDir, REPORT_FILE).absolutePath)
                    putString("riskseg_status", report.getString("status"))
                }
            )
        } finally {
            firstRuntime?.close()
            sustainedRuntime?.close()
            natural.recycle()
            synthetic.recycle()
        }
    }

    private fun openRuntime(cacheDir: File, modelToken: String): QnnRuntime {
        System.loadLibrary("cdsprpc")
        assertTrue(
            "QNN HTP quantized capability is unavailable",
            QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED)
        )
        val skelLibraryDir = prepareSkelLibraryDir()
        val options = QnnDelegate.Options().apply {
            setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
            setSkelLibraryDir(skelLibraryDir.absolutePath)
            setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_QUANTIZED)
            setHtpPerformanceMode(
                QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE
            )
            setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
            setProfiling(QnnDelegate.Options.ProfilingOptions.BASIC_PROFILING)
            setCacheDir(cacheDir.absolutePath)
            setModelToken(modelToken)
        }
        val initStart = System.nanoTime()
        val delegate = QnnDelegate(options)
        assertTrue("QNN HTP delegate is unavailable", delegate.isAvailable)
        val interpreter = Interpreter(
            loadMappedAsset(MODEL_ASSET),
            Interpreter.Options().setNumThreads(CPU_THREADS).addDelegate(delegate)
        )
        interpreter.allocateTensors()
        delegate.performanceVote()
        val input = interpreter.getInputTensor(0)
        val output = interpreter.getOutputTensor(0)
        assertEquals(INPUT_SHAPE.toList(), input.shape().toList())
        assertEquals(OUTPUT_SHAPE.toList(), output.shape().toList())
        assertEquals(DataType.INT8, input.dataType())
        assertEquals(DataType.INT8, output.dataType())
        assertTrue("input quantization scale must be positive", input.quantizationParams().scale > 0f)
        assertTrue("output quantization scale must be positive", output.quantizationParams().scale > 0f)
        return QnnRuntime(
            interpreter = interpreter,
            delegate = delegate,
            inputScale = input.quantizationParams().scale,
            inputZeroPoint = input.quantizationParams().zeroPoint,
            outputScale = output.quantizationParams().scale,
            outputZeroPoint = output.quantizationParams().zeroPoint,
            initMs = elapsedMs(initStart)
        )
    }

    private fun prepareSkelLibraryDir(): File {
        val directory = File(targetContext.codeCacheDir, SKEL_DIR_NAME).apply { mkdirs() }
        val sourceApk = File(targetContext.applicationInfo.sourceDir)
        ZipFile(sourceApk).use { archive ->
            val matchingEntries = archive.entries().asSequence()
                .filter { entry ->
                    !entry.isDirectory &&
                        entry.name.startsWith("lib/arm64-v8a/") &&
                        entry.name.endsWith("Skel.so")
                }
                .toList()
            require(matchingEntries.isNotEmpty()) {
                "No arm64 QNN skel libraries found in ${sourceApk.absolutePath}"
            }
            matchingEntries.forEach { entry ->
                val destination = File(directory, entry.name.substringAfterLast('/'))
                if (!destination.isFile || destination.length() != entry.size) {
                    archive.getInputStream(entry).use { input ->
                        destination.outputStream().use(input::copyTo)
                    }
                }
                require(destination.setReadable(true, true) || destination.canRead())
                require(destination.setExecutable(true, true) || destination.canExecute())
            }
        }
        return directory
    }

    private fun decodeBitmap(asset: String): Bitmap =
        requireNotNull(testContext.assets.open(asset).use(BitmapFactory::decodeStream)) {
            "Could not decode $asset"
        }

    private fun loadMappedAsset(asset: String): MappedByteBuffer {
        val descriptor = testContext.assets.openFd(asset)
        FileInputStream(descriptor.fileDescriptor).use { stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength
            )
        }
    }

    private fun sha256Asset(asset: String): String =
        testContext.assets.open(asset).use { stream ->
            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val read = stream.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
            digest.digest().toHex()
        }

    private fun cacheInventory(root: File): List<CacheFile> =
        root.walkTopDown()
            .filter { it.isFile }
            .map { file ->
                CacheFile(
                    relativePath = file.relativeTo(root).invariantSeparatorsPath,
                    sizeBytes = file.length(),
                    sha256 = FileInputStream(file).use { sha256(it.readBytes()) }
                )
            }
            .sortedBy { it.relativePath }
            .toList()

    private fun deviceJson(): JSONObject = JSONObject()
        .put("manufacturer", Build.MANUFACTURER)
        .put("model", Build.MODEL)
        .put("device", Build.DEVICE)
        .put("soc_model", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else "")
        .put("api_level", Build.VERSION.SDK_INT)
        .put("supported_abis", JSONArray(Build.SUPPORTED_ABIS.toList()))

    private fun currentThermalStatus(power: PowerManager): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            power.currentThermalStatus
        } else {
            PowerManager.THERMAL_STATUS_NONE
        }

    private fun stats(values: List<Double>): TimingStats {
        require(values.isNotEmpty())
        val sorted = values.sorted()
        fun percentile(p: Double): Double {
            val index = ceil(p * sorted.size).toInt().coerceIn(1, sorted.size) - 1
            return sorted[index]
        }
        return TimingStats(
            count = sorted.size,
            p50 = percentile(0.50),
            p95 = percentile(0.95),
            maximum = sorted.last()
        )
    }

    private fun elapsedMs(startNs: Long): Double = (System.nanoTime() - startNs) / 1_000_000.0

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).toHex()

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }

    private class QnnRuntime(
        private val interpreter: Interpreter,
        private val delegate: QnnDelegate,
        private val inputScale: Float,
        private val inputZeroPoint: Int,
        private val outputScale: Float,
        private val outputZeroPoint: Int,
        val initMs: Double
    ) : AutoCloseable {
        private val input = ByteBuffer.allocateDirect(INPUT_WIDTH * INPUT_HEIGHT * 3)
            .order(ByteOrder.nativeOrder())
        private val output = ByteBuffer.allocateDirect(INPUT_WIDTH * INPUT_HEIGHT * CLASS_COUNT)
            .order(ByteOrder.nativeOrder())

        fun runPipeline(
            source: Bitmap,
            analyzer: TraversabilitySegmentationAnalyzer,
            kernel: AssistDecisionKernel,
            gateway: FeedbackGateway,
            nowMs: Long
        ): PipelineSample {
            val totalStart = System.nanoTime()
            val preprocessStart = System.nanoTime()
            val scaled = if (source.width == INPUT_WIDTH && source.height == INPUT_HEIGHT) {
                source
            } else {
                Bitmap.createScaledBitmap(source, INPUT_WIDTH, INPUT_HEIGHT, true)
            }
            val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
            scaled.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)
            input.rewind()
            pixels.forEach { pixel ->
                quantizeChannel((pixel shr 16) and 0xff, 0)
                quantizeChannel((pixel shr 8) and 0xff, 1)
                quantizeChannel(pixel and 0xff, 2)
            }
            input.rewind()
            if (scaled !== source) scaled.recycle()
            val preprocessMs = elapsedStaticMs(preprocessStart)

            output.rewind()
            val inferenceStart = System.nanoTime()
            interpreter.run(input, output)
            val inferenceMs = elapsedStaticMs(inferenceStart)

            output.rewind()
            val postprocessStart = System.nanoTime()
            val classIds = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
            var outputMinimum = Float.POSITIVE_INFINITY
            var outputMaximum = Float.NEGATIVE_INFINITY
            for (pixelIndex in classIds.indices) {
                var bestClass = 0
                var bestScore = Float.NEGATIVE_INFINITY
                repeat(CLASS_COUNT) { classIndex ->
                    val quantized = output.get().toInt()
                    val score = (quantized - outputZeroPoint) * outputScale
                    check(score.isFinite()) { "non-finite dequantized output" }
                    outputMinimum = minOf(outputMinimum, score)
                    outputMaximum = maxOf(outputMaximum, score)
                    if (score > bestScore) {
                        bestScore = score
                        bestClass = classIndex
                    }
                }
                classIds[pixelIndex] = bestClass
            }
            val postprocessMs = elapsedStaticMs(postprocessStart)

            val frameSize = FrameSize(source.width, source.height)
            val adapterStart = System.nanoTime()
            val detections = analyzer.analyze(
                DenseSemanticMask(INPUT_WIDTH, INPUT_HEIGHT, classIds),
                frameSize
            ).riskDetections
            val adapterMs = elapsedStaticMs(adapterStart)

            val decisionStart = System.nanoTime()
            kernel.processFrame(
                detections = detections,
                frameSize = frameSize,
                profile = AlertProfile.STANDARD,
                scenario = AssistScenario.GENERAL,
                metrics = DetectorMetrics(
                    totalMs = 0L,
                    preprocessMs = preprocessMs.roundToInt().toLong(),
                    inferenceMs = inferenceMs.roundToInt().toLong(),
                    postprocessMs = (postprocessMs + adapterMs).roundToInt().toLong(),
                    fps = 0f,
                    modelStatus = "QNN_HTP_QUANTIZED"
                ),
                feedbackGateway = gateway,
                nowMs = nowMs
            )
            val decisionMs = elapsedStaticMs(decisionStart)
            return PipelineSample(
                elapsedSinceStartMs = 0.0,
                preprocessMs = preprocessMs,
                inferenceMs = inferenceMs,
                postprocessMs = postprocessMs,
                adapterMs = adapterMs,
                decisionMs = decisionMs,
                totalMs = elapsedStaticMs(totalStart),
                uniqueClasses = classIds.distinct().sorted(),
                outputMinimum = outputMinimum,
                outputMaximum = outputMaximum,
                detectionCount = detections.size
            )
        }

        fun profilingResult(): ByteArray = delegate.profilingResult ?: ByteArray(0)

        override fun close() {
            interpreter.close()
            delegate.performanceRelease()
            delegate.close()
        }

        private fun quantizeChannel(channel: Int, channelIndex: Int) {
            val normalized = (channel / 255f - IMAGENET_MEAN[channelIndex]) /
                IMAGENET_STD[channelIndex]
            val quantized = (normalized / inputScale + inputZeroPoint)
                .roundToInt()
                .coerceIn(-128, 127)
            input.put(quantized.toByte())
        }

        private fun elapsedStaticMs(startNs: Long): Double =
            (System.nanoTime() - startNs) / 1_000_000.0
    }

    private object RisksegR0Taxonomy : TraversabilityTaxonomy {
        override fun traversabilityFor(classId: Int): TraversabilityClass = when (classId) {
            WALKABLE -> TraversabilityClass.SAFE_TO_WALK
            BLOCKING_OBSTACLE -> TraversabilityClass.OBSTACLE
            BOUNDARY_LEVEL_CHANGE, UNKNOWN_NONWALKABLE -> TraversabilityClass.NOT_SAFE_TO_WALK
            else -> error("RISKSEG-R0 class outside 0..3: $classId")
        }

        override fun isNavigationHazard(classId: Int): Boolean =
            classId == BLOCKING_OBSTACLE || classId == BOUNDARY_LEVEL_CHANGE

        override fun riskLabelFor(classId: Int): String? = when (classId) {
            BLOCKING_OBSTACLE -> "generic obstacle"
            BOUNDARY_LEVEL_CHANGE -> "curb"
            else -> null
        }

        override fun permitsBoundaryDetection(classId: Int): Boolean =
            classId == BOUNDARY_LEVEL_CHANGE

        override fun isBoundaryEvidence(classId: Int): Boolean =
            classId == BOUNDARY_LEVEL_CHANGE
    }

    private class PlannerGateway : FeedbackGateway {
        override fun resetSession() = Unit

        override fun notify(
            risk: com.linnan.blindassist.risk.RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
            val plan = FeedbackPlanner.planFor(risk, profile = profile, scenario = scenario)
            return if (plan == null) {
                FeedbackDecision(null, false, FeedbackReason.NO_FEEDBACK_RISK)
            } else {
                FeedbackDecision(plan, true, FeedbackReason.TRIGGERED)
            }
        }
    }

    private data class PipelineSample(
        val elapsedSinceStartMs: Double,
        val preprocessMs: Double,
        val inferenceMs: Double,
        val postprocessMs: Double,
        val adapterMs: Double,
        val decisionMs: Double,
        val totalMs: Double,
        val uniqueClasses: List<Int>,
        val outputMinimum: Float,
        val outputMaximum: Float,
        val detectionCount: Int
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("preprocess_ms", preprocessMs)
            .put("inference_ms", inferenceMs)
            .put("dequantize_argmax_ms", postprocessMs)
            .put("mask_adapter_ms", adapterMs)
            .put("decision_event_chain_ms", decisionMs)
            .put("total_ms", totalMs)
            .put("argmax_unique_classes", JSONArray(uniqueClasses))
            .put("output_float_minimum", outputMinimum.toDouble())
            .put("output_float_maximum", outputMaximum.toDouble())
            .put("finite", outputMinimum.isFinite() && outputMaximum.isFinite())
            .put("detection_count", detectionCount)
    }

    private data class TimingStats(
        val count: Int,
        val p50: Double,
        val p95: Double,
        val maximum: Double
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("count", count)
            .put("p50", p50)
            .put("p95", p95)
            .put("maximum", maximum)
    }

    private data class CacheFile(
        val relativePath: String,
        val sizeBytes: Long,
        val sha256: String
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("relative_path", relativePath)
            .put("size_bytes", sizeBytes)
            .put("sha256", sha256)
    }

    private companion object {
        const val MODEL_ASSET =
            "riskseg_pidnet/pidnet_s_512x288_4class_preflight_full_integer_quant.tflite"
        const val CANARY_ASSET = "riskseg_pidnet/train_rgb_non_eval.png"
        const val MODEL_TOKEN_PREFIX = "riskseg_pidnet_s_512x288_int8"
        const val CACHE_DIR_PREFIX = "riskseg-pidnet-qnn"
        const val SKEL_DIR_NAME = "riskseg-qnn-skel-v2-47"
        const val REPORT_FILE = "preflight.json"
        const val ARG_DURATION_MS = "risksegDurationMs"
        const val ARG_RUN_ROLE = "risksegRunRole"
        const val RUN_ROLE_TECHNICAL_PREFLIGHT = "TECHNICAL_PREFLIGHT"
        const val RUN_ROLE_TRAINED_FINAL = "TRAINED_FINAL"
        const val DEFAULT_SMOKE_DURATION_MS = 15_000L
        const val MIN_DURATION_MS = 1_000L
        const val MAX_DURATION_MS = 900_000L
        const val FORMAL_DURATION_MS = 600_000L
        const val INITIAL_WINDOW_MS = 120_000L
        const val FINAL_WINDOW_MS = 120_000L
        const val FRAME_STEP_MS = 100L
        const val WARMUP_RUNS = 20
        const val CPU_THREADS = 4
        const val INPUT_WIDTH = 512
        const val INPUT_HEIGHT = 288
        const val CLASS_COUNT = 4
        const val WALKABLE = 0
        const val BLOCKING_OBSTACLE = 1
        const val BOUNDARY_LEVEL_CHANGE = 2
        const val UNKNOWN_NONWALKABLE = 3
        val INPUT_SHAPE = intArrayOf(1, INPUT_HEIGHT, INPUT_WIDTH, 3)
        val OUTPUT_SHAPE = intArrayOf(1, INPUT_HEIGHT, INPUT_WIDTH, CLASS_COUNT)
        val CLASS_ORDER = listOf(
            "walkable",
            "blocking_obstacle",
            "boundary_level_change",
            "unknown_nonwalkable"
        )
        val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
