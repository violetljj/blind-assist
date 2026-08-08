package com.linnan.blindassist.benchmark

import android.graphics.Bitmap
import android.media.MediaExtractor
import android.media.MediaMetadataRetriever
import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackGateway
import com.linnan.blindassist.feedback.FeedbackPlanner
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.session.AssistDecisionKernel
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.system.measureNanoTime

/**
 * Device half of the USTRF U0 baseline adapter.
 *
 * A host bridge stages only the frozen request, sanitized inference manifest,
 * hash-bound fixed-arm artifact/config and source video in the target app's
 * private files directory. Video decode, shipped TFLite YOLO inference and the
 * shared production decision kernel all execute in this instrumentation process.
 */
@RunWith(AndroidJUnit4::class)
class UstrfU0BaselineAdapterDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun runBaselineYoloGeometryAdapter() {
        runAdapter(BASELINE_PROFILE)
    }

    @Test
    fun runDetectorBBoxExplicitRouteAdapter() {
        runAdapter(ROUTE_PROFILE)
    }

    private fun runAdapter(profile: AdapterProfile) {
        val required = arguments.getString(profile.requiredArgument)?.toBooleanStrictOrNull() ?: false
        val requestRelative = arguments.getString(ARG_REQUEST)
        if (requestRelative.isNullOrBlank()) {
            assumeTrue("USTRF U0 ${profile.armId} adapter request was not supplied", !required)
            return
        }
        check(required) { "USTRF U0 device adapter must be invoked with required=true" }
        val manifestRelative = requireArgument(ARG_INFERENCE_MANIFEST)
        val videoRelative = requireArgument(ARG_VIDEO)
        val ledgerRelative = requireArgument(ARG_LEDGER)
        val artifactRelative = requireArgument(ARG_ARTIFACT)
        val thresholdRelative = requireArgument(ARG_THRESHOLD_CONFIG)
        val outputRelative = requireArgument(ARG_OUTPUT)
        val routeRelative = if (profile.usesExplicitRoute) requireArgument(ARG_ROUTE) else null

        val requestFile = privateFile(requestRelative, "request")
        val manifestFile = privateFile(manifestRelative, "inference manifest")
        val videoFile = privateFile(videoRelative, "input video")
        val ledgerFile = privateFile(ledgerRelative, "capture frame ledger")
        val artifactFile = privateFile(artifactRelative, "fold artifact")
        val thresholdFile = privateFile(thresholdRelative, "threshold config")
        val routeFile = routeRelative?.let { privateFile(it, "explicit route input") }
        val outputFile = privateOutputFile(outputRelative)
        val request = JSONObject(requestFile.readText(Charsets.UTF_8))
        val manifest = JSONObject(manifestFile.readText(Charsets.UTF_8))
        val threshold = JSONObject(thresholdFile.readText(Charsets.UTF_8))
        validateInputs(
            profile, request, manifest, threshold, requestFile, manifestFile,
            videoFile, ledgerFile, artifactFile, routeFile
        )
        val routeEpisode = routeFile?.let {
            UstrfU0ExplicitRouteDetectionGate.parseEpisode(
                JSONObject(it.readText(Charsets.UTF_8)),
                request.getString("episode_id")
            )
        }
        val routeGateConfig = if (profile.usesExplicitRoute) {
            UstrfU0RouteGateConfig(
                minimumRouteConfidence = threshold.getDouble("minimum_route_confidence"),
                maximumRouteAgeMs = threshold.getLong("maximum_route_age_ms"),
                corridorHalfWidthFrameRatio = threshold.getDouble("corridor_half_width_frame_ratio"),
                obstacleFootprintHeightRatio = threshold.getDouble("obstacle_footprint_height_ratio")
            )
        } else null

        val detector = TfliteYoloDetector(testContext)
        check(detector.isReady) { "default detector failed to initialize: ${detector.statusMessage}" }
        val modelSha256 = sha256Asset(TfliteYoloDetector.MODEL_ASSET)
        val labelsSha256 = sha256Asset(TfliteYoloDetector.LABELS_ASSET)
        check(modelSha256 == threshold.getString("model_asset_sha256")) {
            "shipped model asset differs from baseline adapter config"
        }
        check(labelsSha256 == threshold.getString("labels_asset_sha256")) {
            "shipped labels asset differs from baseline adapter config"
        }
        val decisionKernel = AssistDecisionKernel()
        val feedbackGateway = PlannerFeedbackGateway()
        decisionKernel.startSession(0L)
        feedbackGateway.resetSession()
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(videoFile.absolutePath)
        val sourceSamples = videoSamples(videoFile)
        val maximumDecodePtsErrorUs = threshold.getLong("maximum_decode_pts_error_us")
        val outputFrames = JSONArray()
        val decodedFrames = JSONArray()
        val routeFrameReceipts = JSONArray()
        try {
            val requestFrames = request.getJSONArray("frames")
            for (index in 0 until requestFrames.length()) {
                val requested = requestFrames.getJSONObject(index)
                val videoPtsMs = requested.getLong("video_pts_ms")
                val requestedPtsUs = videoPtsMs * 1_000L
                val selectedSource = sourceSamples.minBy { abs(it.ptsUs - requestedPtsUs) }
                val selectedSourcePtsUs = selectedSource.ptsUs
                val ptsErrorUs = abs(selectedSourcePtsUs - requestedPtsUs)
                check(ptsErrorUs <= maximumDecodePtsErrorUs) {
                    "nearest encoded video sample for ${requested.getString("frame_id")} is ${ptsErrorUs}us away"
                }
                var bitmap: Bitmap? = null
                val decodeNanos = measureNanoTime {
                    bitmap = retriever.getFrameAtTime(
                        selectedSourcePtsUs,
                        MediaMetadataRetriever.OPTION_CLOSEST
                    )
                }
                val decoded = checkNotNull(bitmap) {
                    "cannot decode ${requested.getString("frame_id")} at ${videoPtsMs}ms"
                }
                try {
                    val detectorResult = detector.detect(decoded)
                    val gateResult = routeEpisode?.let { episode ->
                        UstrfU0ExplicitRouteDetectionGate.gate(
                            episode = episode,
                            frameTimestampMs = videoPtsMs,
                            detections = detectorResult.detections,
                            frameSize = detectorResult.frameSize,
                            config = checkNotNull(routeGateConfig)
                        )
                    }
                    val kernelDetections = gateResult?.retainedDetections ?: detectorResult.detections
                    val decision = decisionKernel.processFrame(
                        detections = kernelDetections,
                        frameSize = detectorResult.frameSize,
                        profile = AlertProfile.STANDARD,
                        scenario = AssistScenario.GENERAL,
                        metrics = detectorResult.metrics,
                        feedbackGateway = feedbackGateway,
                        nowMs = videoPtsMs
                    )
                    outputFrames.put(JSONObject(requested.toString()).put("decision", decisionJson(request, decision)))
                    if (routeEpisode != null && gateResult != null && routeGateConfig != null) {
                        routeFrameReceipts.put(
                            UstrfU0ExplicitRouteDetectionGate.receiptJson(
                                routeEpisode,
                                requested.getString("frame_id"),
                                videoPtsMs,
                                gateResult,
                                routeGateConfig
                            )
                        )
                    }
                    decodedFrames.put(
                        JSONObject()
                            .put("frame_id", requested.getString("frame_id"))
                            .put("video_pts_ms", videoPtsMs)
                            .put("requested_pts_us", requestedPtsUs)
                            .put("selected_source_pts_us", selectedSourcePtsUs)
                            .put("pts_error_us", ptsErrorUs)
                            .put("encoded_sample_sha256", selectedSource.encodedSampleSha256)
                            .put("decode_duration_ms", decodeNanos / 1_000_000.0)
                            .put("decoded_rgba8888_sha256", sha256CanonicalRgba8888(decoded))
                            .put("width", decoded.width)
                            .put("height", decoded.height)
                            .put("preprocess_ms", detectorResult.metrics.preprocessMs)
                            .put("inference_ms", detectorResult.metrics.inferenceMs)
                            .put("postprocess_ms", detectorResult.metrics.postprocessMs)
                            .put("total_detect_ms", detectorResult.metrics.totalMs)
                            .put("detection_count", detectorResult.detections.size)
                            .put("kernel_input_detection_count", kernelDetections.size)
                    )
                } finally {
                    decoded.recycle()
                }
            }
        } finally {
            retriever.release()
            detector.close()
        }

        val output = JSONObject(request.toString())
        output.remove("schema")
        output.remove("frames")
        output.put("schema", OUTPUT_SCHEMA)
        output.put("execution_completed", true)
        output.put("failure_count", 0)
        output.put("abstained", false)
        output.put("frames", outputFrames)
        output.put(
            "android_backend_receipt",
            backendReceipt(
                profile = profile,
                request = request,
                threshold = threshold,
                requestSha256 = sha256File(requestFile),
                modelSha256 = modelSha256,
                labelsSha256 = labelsSha256,
                decodedFrames = decodedFrames
            )
        )
        if (profile.usesExplicitRoute) {
            output.put(
                "route_conditioning_receipt",
                routeConditioningReceipt(
                    profile = profile,
                    request = request,
                    threshold = threshold,
                    routeFile = checkNotNull(routeFile),
                    routeEpisode = checkNotNull(routeEpisode),
                    frameReceipts = routeFrameReceipts
                )
            )
        }
        outputFile.parentFile?.mkdirs()
        outputFile.writeText(output.toString(2), Charsets.UTF_8)
        Log.i(TAG, "USTRF_U0_BASELINE_OUTPUT ${outputFile.absolutePath}")
    }

    private fun validateInputs(
        profile: AdapterProfile,
        request: JSONObject,
        manifest: JSONObject,
        threshold: JSONObject,
        requestFile: File,
        manifestFile: File,
        videoFile: File,
        ledgerFile: File,
        artifactFile: File,
        routeFile: File?
    ) {
        check(request.getString("schema") == REQUEST_SCHEMA)
        check(request.getString("arm_id") == profile.armId)
        check(request.getString("candidate_adapter_id") == profile.adapterId)
        check(request.getString("kernel_execution_backend_id") == BACKEND_ID)
        check(request.getString("shared_decision_kernel_contract_id") == AssistDecisionKernel.CONTRACT_ID)
        check(request.getString("decision_profile_id") == AlertProfile.STANDARD.name)
        check(request.getString("fit_policy") == "fixed_no_fit_v1")
        check(request.getString("event_identity_policy") == "kernel_native_optional_v1")
        check(request.getString("route_input_policy") == profile.routeInputPolicy)
        check(!request.getBoolean("synthetic_fixture"))
        check(!request.getBoolean("blind_accessed"))
        check(!request.getBoolean("future_inputs_used"))
        check(!request.getBoolean("production_model_replacement_authorized"))
        check(request.getJSONObject("decision_cadence").getInt("canonical_step_ms") == 500)
        check(request.getJSONArray("frames").length() > 0)

        check(manifest.getString("schema") == SANITIZED_MANIFEST_SCHEMA)
        check(manifest.getString("arm_id") == profile.armId)
        check(manifest.getString("episode_id") == request.getString("episode_id"))
        if (profile.usesExplicitRoute) {
            val expectedRoute = checkNotNull(routeFile)
            check(manifest.getString("adapter_route_input_sha256") == sha256File(expectedRoute))
            check(manifest.getString("adapter_route_source_episode_id") == request.getString("episode_id"))
            check(request.getString("adapter_route_input_sha256") == sha256File(expectedRoute))
            check(request.getString("adapter_route_source_episode_id") == request.getString("episode_id"))
            check(request.getString("truth_route_intent_sha256") == sha256File(expectedRoute))
        } else {
            check(routeFile == null)
            check(manifest.isNull("adapter_route_input_path"))
            check(manifest.isNull("adapter_route_input_sha256"))
            check(manifest.isNull("adapter_route_source_episode_id"))
            check(request.isNull("adapter_route_input_sha256"))
            check(request.isNull("adapter_route_source_episode_id"))
        }
        check(!manifest.getBoolean("review_fields_present"))
        check(!manifest.getBoolean("adjudication_fields_present"))
        check(!manifest.getBoolean("event_label_fields_present"))
        check(!manifest.getBoolean("blind_accessed"))
        check(!manifest.getBoolean("future_inputs_used"))
        check(sha256File(manifestFile) == request.getString("sanitized_inference_manifest_sha256"))
        check(sha256File(videoFile) == request.getString("input_video_sha256"))
        check(manifest.getString("input_video_sha256") == request.getString("input_video_sha256"))
        check(sha256File(ledgerFile) == request.getString("source_capture_frame_ledger_sha256"))
        check(manifest.getString("capture_frame_ledger_sha256") == request.getString("source_capture_frame_ledger_sha256"))
        val ledger = JSONObject(ledgerFile.readText(Charsets.UTF_8))
        check(ledger.getString("schema") == "blindassist_capture_frame_ledger_v1")
        check(ledger.getString("episode_id") == request.getString("episode_id"))
        val ledgerFrames = ledger.getJSONArray("frames")
        val requestFrames = request.getJSONArray("frames")
        check(ledgerFrames.length() == requestFrames.length())
        for (index in 0 until requestFrames.length()) {
            val ledgerFrame = ledgerFrames.getJSONObject(index)
            val requestFrame = requestFrames.getJSONObject(index)
            for (key in FRAME_BINDING_KEYS) check(ledgerFrame.get(key) == requestFrame.get(key))
        }
        check(sha256File(artifactFile) == request.getString("fold_artifact_sha256"))

        check(threshold.getString("schema") == profile.thresholdSchema)
        check(threshold.getString("arm_id") == profile.armId)
        check(threshold.getString("candidate_adapter_id") == profile.adapterId)
        check(threshold.getString("kernel_execution_backend_id") == BACKEND_ID)
        check(threshold.getString("model_asset_name") == TfliteYoloDetector.MODEL_ASSET)
        check(threshold.getString("labels_asset_name") == TfliteYoloDetector.LABELS_ASSET)
        check(threshold.getInt("input_size") == TfliteYoloDetector.INPUT_SIZE)
        check(threshold.getDouble("confidence_threshold").toFloat() == TfliteYoloDetector.CONFIDENCE_THRESHOLD)
        check(threshold.getDouble("iou_threshold").toFloat() == TfliteYoloDetector.IOU_THRESHOLD)
        check(threshold.getString("detector_runtime") == DETECTOR_RUNTIME)
        check(threshold.getString("frame_decode_policy") == FRAME_DECODE_POLICY)
        check(threshold.getString("decoded_payload_contract") == DECODED_PAYLOAD_CONTRACT)
        check(threshold.getLong("maximum_decode_pts_error_us") in 0L..20_000L)
        check(threshold.getString("instrumentation_test_class") == profile.testMethod)
        requireSha(threshold.getString("device_adapter_implementation_sha256"))
        requireSha(threshold.getString("host_adapter_implementation_sha256"))
        if (profile.usesExplicitRoute) {
            check(threshold.getString("route_gate_contract_id") == UstrfU0ExplicitRouteDetectionGate.GATE_CONTRACT_ID)
            check(threshold.getString("unknown_route_policy") == UstrfU0ExplicitRouteDetectionGate.UNKNOWN_ROUTE_POLICY)
            check(threshold.getDouble("minimum_route_confidence") in 0.0..1.0)
            check(threshold.getLong("maximum_route_age_ms") in 0L..1_000L)
            check(threshold.getDouble("corridor_half_width_frame_ratio") in 0.0..0.5)
            check(threshold.getDouble("obstacle_footprint_height_ratio") in 0.0..1.0)
            requireSha(threshold.getString("route_gate_implementation_sha256"))
        }
        check(sha256File(requestFile).length == 64)
    }

    private fun decisionJson(
        request: JSONObject,
        result: com.linnan.blindassist.session.AssistFrameResult
    ): JSONObject {
        val feedback = result.feedbackDecision
        val event = result.evaluation.riskEvent
        return JSONObject()
            .put("raw_risk_level", result.evaluation.rawRisk.level.name)
            .put("stable_risk_level", result.evaluation.stableRisk.level.name)
            .put("event_id", event.eventId ?: JSONObject.NULL)
            .put("event_state", event.state?.name ?: JSONObject.NULL)
            .put("candidate_adapter_id", request.getString("candidate_adapter_id"))
            .put(
                "feedback_receipt",
                JSONObject()
                    .put("outcome", feedbackOutcome(feedback))
                    .put("kernel_feedback_reason", feedback.reason.name)
                    .put("delivered", feedback.triggered)
                    .put("adapter_id", request.getString("feedback_adapter_id"))
                    .put("kernel_contract_id", AssistDecisionKernel.CONTRACT_ID)
            )
    }

    private fun feedbackOutcome(decision: FeedbackDecision): String = when (decision.reason) {
        FeedbackReason.TRIGGERED -> "TRIGGERED"
        FeedbackReason.EVENT_ALREADY_ALERTED -> "DUPLICATE_SUPPRESSED"
        FeedbackReason.FEEDBACK_UNAVAILABLE -> "UNAVAILABLE"
        else -> "NO_ALERT"
    }

    private fun backendReceipt(
        profile: AdapterProfile,
        request: JSONObject,
        threshold: JSONObject,
        requestSha256: String,
        modelSha256: String,
        labelsSha256: String,
        decodedFrames: JSONArray
    ): JSONObject {
        val packageInfo = targetContext.packageManager.getPackageInfo(targetContext.packageName, 0)
        return JSONObject()
            .put("schema", BACKEND_RECEIPT_SCHEMA)
            .put("backend_id", BACKEND_ID)
            .put("adapter_output_origin", "on_device_instrumentation_v1")
            .put("instrumentation_component", INSTRUMENTATION_COMPONENT)
            .put("instrumentation_test_class", profile.testMethod)
            .put("target_package", targetContext.packageName)
            .put("target_version_name", packageInfo.versionName ?: "unknown")
            .put("target_version_code", packageInfo.longVersionCode)
            .put("target_apk_sha256", sha256File(File(targetContext.applicationInfo.sourceDir)))
            .put("test_apk_sha256", sha256File(File(testContext.applicationInfo.sourceDir)))
            .put("device_model", Build.MODEL)
            .put("device_api_level", Build.VERSION.SDK_INT)
            .put("build_fingerprint_sha256", sha256Bytes(Build.FINGERPRINT.toByteArray(Charsets.UTF_8)))
            .put("model_asset_name", TfliteYoloDetector.MODEL_ASSET)
            .put("model_asset_sha256", modelSha256)
            .put("labels_asset_name", TfliteYoloDetector.LABELS_ASSET)
            .put("labels_asset_sha256", labelsSha256)
            .put("input_size", TfliteYoloDetector.INPUT_SIZE)
            .put("confidence_threshold", threshold.getDouble("confidence_threshold"))
            .put("iou_threshold", threshold.getDouble("iou_threshold"))
            .put("detector_runtime", DETECTOR_RUNTIME)
            .put("shared_decision_kernel_contract_id", AssistDecisionKernel.CONTRACT_ID)
            .put("device_adapter_implementation_sha256", threshold.getString("device_adapter_implementation_sha256"))
            .put("host_adapter_implementation_sha256", threshold.getString("host_adapter_implementation_sha256"))
            .put("request_sha256", requestSha256)
            .put("sanitized_inference_manifest_sha256", request.getString("sanitized_inference_manifest_sha256"))
            .put("input_video_sha256", request.getString("input_video_sha256"))
            .put("source_capture_frame_ledger_sha256", request.getString("source_capture_frame_ledger_sha256"))
            .put("frame_decode_policy", FRAME_DECODE_POLICY)
            .put("decoded_payload_contract", DECODED_PAYLOAD_CONTRACT)
            .put("maximum_decode_pts_error_us", threshold.getLong("maximum_decode_pts_error_us"))
            .put("requested_frame_count", request.getJSONArray("frames").length())
            .put("decoded_frame_count", decodedFrames.length())
            .put("decoded_frames", decodedFrames)
    }

    private fun routeConditioningReceipt(
        profile: AdapterProfile,
        request: JSONObject,
        threshold: JSONObject,
        routeFile: File,
        routeEpisode: UstrfU0ExplicitRouteEpisode,
        frameReceipts: JSONArray
    ): JSONObject = JSONObject()
        .put("schema", ROUTE_RECEIPT_SCHEMA)
        .put("arm_id", profile.armId)
        .put("candidate_adapter_id", profile.adapterId)
        .put("route_input_policy", profile.routeInputPolicy)
        .put("route_input_sha256", sha256File(routeFile))
        .put("route_episode_id", routeEpisode.episodeId)
        .put("route_parent_source_id", routeEpisode.parentSourceId)
        .put("route_provider_type", routeEpisode.providerType)
        .put("route_provider_id", routeEpisode.providerId)
        .put("projection_receipt_id", routeEpisode.projectionReceiptId)
        .put("route_gate_contract_id", UstrfU0ExplicitRouteDetectionGate.GATE_CONTRACT_ID)
        .put("route_gate_implementation_sha256", threshold.getString("route_gate_implementation_sha256"))
        .put("unknown_route_policy", UstrfU0ExplicitRouteDetectionGate.UNKNOWN_ROUTE_POLICY)
        .put("minimum_route_confidence", threshold.getDouble("minimum_route_confidence"))
        .put("maximum_route_age_ms", threshold.getLong("maximum_route_age_ms"))
        .put("corridor_half_width_frame_ratio", threshold.getDouble("corridor_half_width_frame_ratio"))
        .put("obstacle_footprint_height_ratio", threshold.getDouble("obstacle_footprint_height_ratio"))
        .put("route_sample_policy", request.getJSONObject("decision_cadence").getString("route_sample_policy"))
        .put("future_inputs_used", false)
        .put("risk_model_inferred_route", false)
        .put("frame_count", frameReceipts.length())
        .put("frames", frameReceipts)

    private fun privateFile(relative: String, label: String): File {
        val root = targetContext.filesDir.canonicalFile
        val file = File(root, relative).canonicalFile
        check(file.toPath().startsWith(root.toPath())) { "$label escapes target private files" }
        check(file.isFile) { "$label is missing: $file" }
        return file
    }

    private fun privateOutputFile(relative: String): File {
        val root = targetContext.filesDir.canonicalFile
        val file = File(root, relative).canonicalFile
        check(file.toPath().startsWith(root.toPath())) { "output escapes target private files" }
        return file
    }

    private fun requireArgument(name: String): String =
        arguments.getString(name)?.takeIf(String::isNotBlank) ?: error("missing instrumentation argument: $name")

    private fun sha256File(file: File): String = file.inputStream().use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        digest.digest().toHex()
    }

    private fun sha256Asset(assetName: String): String = testContext.assets.open(assetName).use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        digest.digest().toHex()
    }

    private fun sha256CanonicalRgba8888(bitmap: Bitmap): String {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val buffer = ByteBuffer.allocate(pixels.size * 4)
        pixels.forEach { argb ->
            buffer.put(((argb shr 16) and 0xff).toByte())
            buffer.put(((argb shr 8) and 0xff).toByte())
            buffer.put((argb and 0xff).toByte())
            buffer.put(((argb ushr 24) and 0xff).toByte())
        }
        return sha256Bytes(buffer.array())
    }

    private fun videoSamples(videoFile: File): List<VideoSample> {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(videoFile.absolutePath)
            val videoTrack = (0 until extractor.trackCount).firstOrNull { index ->
                extractor.getTrackFormat(index).getString(android.media.MediaFormat.KEY_MIME)?.startsWith("video/") == true
            } ?: error("input video has no video track")
            extractor.selectTrack(videoTrack)
            val values = mutableListOf<VideoSample>()
            while (extractor.sampleTime >= 0L) {
                val sampleSize = extractor.sampleSize
                check(sampleSize in 1..MAX_ENCODED_SAMPLE_BYTES.toLong()) {
                    "encoded video sample size is invalid: $sampleSize"
                }
                val buffer = ByteBuffer.allocate(sampleSize.toInt())
                val read = extractor.readSampleData(buffer, 0)
                check(read == sampleSize.toInt()) { "encoded video sample read is incomplete" }
                val bytes = ByteArray(read)
                buffer.rewind()
                buffer.get(bytes)
                values += VideoSample(extractor.sampleTime, sha256Bytes(bytes))
                if (!extractor.advance()) break
            }
            check(values.isNotEmpty()) { "input video has no encoded video samples" }
            return values.distinctBy { it.ptsUs }.sortedBy { it.ptsUs }
        } finally {
            extractor.release()
        }
    }

    private data class VideoSample(val ptsUs: Long, val encodedSampleSha256: String)

    private fun sha256Bytes(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).toHex()

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }

    private fun requireSha(value: String) {
        check(value.length == 64 && value.all { it in '0'..'9' || it in 'a'..'f' })
    }

    private class PlannerFeedbackGateway : FeedbackGateway {
        override fun resetSession() = Unit

        override fun notify(
            risk: RiskResult,
            profile: AlertProfile,
            scenario: AssistScenario
        ): FeedbackDecision {
            val plan = FeedbackPlanner.planFor(risk, profile = profile, scenario = scenario)
            return if (plan == null) {
                FeedbackDecision(null, triggered = false, reason = FeedbackReason.NO_FEEDBACK_RISK)
            } else {
                FeedbackDecision(plan, triggered = true, reason = FeedbackReason.TRIGGERED)
            }
        }
    }

    private data class AdapterProfile(
        val armId: String,
        val adapterId: String,
        val routeInputPolicy: String,
        val thresholdSchema: String,
        val testMethod: String,
        val requiredArgument: String,
        val usesExplicitRoute: Boolean
    )

    private companion object {
        const val BACKEND_ID = "android_kotlin_assist_decision_kernel_v1"
        const val REQUEST_SCHEMA = "blindassist_ustrf_sc_u0_candidate_adapter_request_v1"
        const val OUTPUT_SCHEMA = "blindassist_ustrf_sc_u0_candidate_adapter_output_v1"
        const val SANITIZED_MANIFEST_SCHEMA = "blindassist_ustrf_sc_u0_sanitized_inference_manifest_v1"
        const val BACKEND_RECEIPT_SCHEMA = "blindassist_ustrf_sc_u0_android_backend_receipt_v1"
        const val ROUTE_RECEIPT_SCHEMA = "blindassist_ustrf_sc_u0_route_conditioning_receipt_v1"
        const val FRAME_DECODE_POLICY = "android_media_metadata_retriever_closest_v1"
        const val DECODED_PAYLOAD_CONTRACT = "rgba8888_row_major_android_getpixels_v1"
        const val DETECTOR_RUNTIME = "tflite_cpu_4_threads_v1"
        const val BASELINE_TEST_METHOD =
            "com.linnan.blindassist.benchmark.UstrfU0BaselineAdapterDeviceTest#runBaselineYoloGeometryAdapter"
        const val ROUTE_TEST_METHOD =
            "com.linnan.blindassist.benchmark.UstrfU0BaselineAdapterDeviceTest#runDetectorBBoxExplicitRouteAdapter"
        const val INSTRUMENTATION_COMPONENT =
            "com.linnan.blindassist.benchmark/androidx.test.runner.AndroidJUnitRunner"
        const val ARG_BASELINE_REQUIRED = "ustrfU0BaselineRequired"
        const val ARG_ROUTE_REQUIRED = "ustrfU0BBoxRouteRequired"
        const val ARG_REQUEST = "ustrfU0Request"
        const val ARG_INFERENCE_MANIFEST = "ustrfU0InferenceManifest"
        const val ARG_VIDEO = "ustrfU0Video"
        const val ARG_LEDGER = "ustrfU0Ledger"
        const val ARG_ARTIFACT = "ustrfU0Artifact"
        const val ARG_THRESHOLD_CONFIG = "ustrfU0ThresholdConfig"
        const val ARG_ROUTE = "ustrfU0ExplicitRoute"
        const val ARG_OUTPUT = "ustrfU0Output"
        const val TAG = "UstrfU0BaselineAdapter"
        const val MAX_ENCODED_SAMPLE_BYTES = 32 * 1024 * 1024
        val FRAME_BINDING_KEYS = listOf(
            "frame_id", "frame_index", "capture_timestamp_ns", "video_pts_ms", "frame_payload_sha256"
        )
        val BASELINE_PROFILE = AdapterProfile(
            armId = "baseline_yolo_geometry",
            adapterId = "yolo_geometry_adapter_v1",
            routeInputPolicy = "no_route_input_v1",
            thresholdSchema = "blindassist_ustrf_sc_u0_android_baseline_adapter_config_v1",
            testMethod = BASELINE_TEST_METHOD,
            requiredArgument = ARG_BASELINE_REQUIRED,
            usesExplicitRoute = false
        )
        val ROUTE_PROFILE = AdapterProfile(
            armId = "detector_bbox_explicit_route",
            adapterId = "detector_bbox_explicit_route_adapter_v1",
            routeInputPolicy = "episode_explicit_causal_route_v1",
            thresholdSchema = "blindassist_ustrf_sc_u0_android_bbox_route_adapter_config_v1",
            testMethod = ROUTE_TEST_METHOD,
            requiredArgument = ARG_ROUTE_REQUIRED,
            usesExplicitRoute = true
        )
    }
}
