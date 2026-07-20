package com.linnan.blindassist.benchmark

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.feedback.FeedbackPlanner
import com.linnan.blindassist.feedback.FeedbackPlannerConfig
import com.linnan.blindassist.feedback.FeedbackReason
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.RiskAnalyzer
import com.linnan.blindassist.risk.RiskAnalyzerConfig
import com.linnan.blindassist.risk.RiskEventTracker
import com.linnan.blindassist.risk.RiskEventTrackerConfig
import com.linnan.blindassist.risk.TemporalRiskTracker
import com.linnan.blindassist.session.LowConfidenceSidePersonConfirmation
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * Runs the shipped default detector and production risk/feedback path on public
 * video evidence frames without embedding a label or asserting a safety metric.
 * The resulting JSON is consumed only by the separate silver-label comparator.
 */
@RunWith(AndroidJUnit4::class)
class PublicVideoEdgeInferenceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun inferPublicEvidenceFramesAndWriteEventReport() {
        val required = arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() ?: false
        val specAsset = "$ASSET_PREFIX/dataset_spec.json"
        if (!assetExists(specAsset)) {
            assumeTrue("public video inference assets are absent", !required)
            return
        }
        val spec = JSONObject(readAssetText(specAsset))
        assertEquals(SPEC_SCHEMA, spec.getString("schema"))
        assertFalse("public inference set must be inference-only", !spec.getBoolean("inference_only"))
        assertFalse("public inference set must not contain human truth", spec.getBoolean("human_event_truth_present"))
        assertFalse("public inference set must not authorize model replacement", spec.getBoolean("production_model_replacement_authorized"))
        val silverSha = spec.getString("silver_manifest_sha256")
        assertEquals(64, silverSha.length)

        val frames = readAssetText("$ASSET_PREFIX/manifest.jsonl")
            .lineSequence()
            .filter { it.isNotBlank() }
            .map { JSONObject(it) }
            .sortedWith(compareBy<JSONObject> { it.getInt("episode_order") }.thenBy { it.getInt("evidence_order") })
            .toList()
        assertFalse("public inference manifest has no frames", frames.isEmpty())
        val riskConfigName = arguments.getString(ARG_RISK_CONFIG) ?: RISK_CONFIG_CURRENT
        val feedbackPlannerConfig = feedbackPlannerConfigFor(riskConfigName)
        val detector = TfliteYoloDetector(testContext)
        assertNotNull("detector status is null", detector.statusMessage)
        check(detector.isReady) { "default detector failed to initialize: ${detector.statusMessage}" }
        val riskAnalyzer = RiskAnalyzer(riskAnalyzerConfigFor(riskConfigName))
        val temporalTracker = TemporalRiskTracker()
        val eventTracker = RiskEventTracker(eventTrackerConfigFor(riskConfigName))
        val lowConfidenceConfirmation = LowConfidenceSidePersonConfirmation()
        val rows = JSONArray()
        val episodes = linkedMapOf<String, MutableList<JSONObject>>()
        var activeEpisode: String? = null
        try {
            frames.forEach { frame ->
                val episodeId = frame.getString("episode_id")
                if (episodeId != activeEpisode) {
                    temporalTracker.reset()
                    eventTracker.reset()
                    lowConfidenceConfirmation.reset()
                    activeEpisode = episodeId
                }
                val assetPath = "$ASSET_PREFIX/${frame.getString("image_path")}"
                val bitmap = testContext.assets.open(assetPath).use { stream -> BitmapFactory.decodeStream(stream) }
                checkNotNull(bitmap) { "cannot decode public evidence frame: $assetPath" }
                val result = try {
                    val frameSize = FrameSize(bitmap.width, bitmap.height)
                    val detections = detector.detect(bitmap).detections
                    val stableRisk = temporalTracker.update(
                        riskAnalyzer.analyze(detections, frameSize),
                        frame.getInt("evidence_order").toLong() * FRAME_STEP_MS
                    )
                    val event = eventTracker.update(stableRisk, frame.getInt("evidence_order").toLong() * FRAME_STEP_MS)
                    val feedback = when {
                        event.suppressesFeedback -> FeedbackDecision(null, false, FeedbackReason.EVENT_ALREADY_ALERTED)
                        !lowConfidenceConfirmation.isConfirmed(stableRisk) -> FeedbackDecision(null, false, FeedbackReason.UNSTABLE_RISK)
                        FeedbackPlanner.planFor(
                            stableRisk,
                            scenario = AssistScenario.GENERAL,
                            config = feedbackPlannerConfig
                        ) == null ->
                            FeedbackDecision(null, false, FeedbackReason.NO_FEEDBACK_RISK)
                        else -> FeedbackDecision(
                            FeedbackPlanner.planFor(
                                stableRisk,
                                scenario = AssistScenario.GENERAL,
                                config = feedbackPlannerConfig
                            ),
                            true,
                            FeedbackReason.TRIGGERED
                        )
                    }
                    val completedEvent = eventTracker.recordFeedback(event, feedback)
                    JSONObject()
                        .put("episode_id", episodeId)
                        .put("evidence_order", frame.getInt("evidence_order"))
                        .put("timeline_frame_index", frame.opt("timeline_frame_index"))
                        .put("source_frame_index", frame.opt("source_frame_index"))
                        .put("image_sha256", frame.getString("image_sha256"))
                        .put("edge_should_alert", feedback.triggered)
                        .put("feedback_reason", feedback.reason.name)
                        .put("risk_level", stableRisk.level.name)
                        .put("risk_direction", stableRisk.direction.name)
                        .put("risk_proximity", stableRisk.proximity.name)
                        .put("risk_approach_trend", stableRisk.approachTrend.name)
                        .put("risk_score", stableRisk.riskScore.toDouble())
                        .put("risk_source", riskSourceJson(stableRisk))
                        .put("risk_event_id", completedEvent.eventId)
                        .put("detected_labels", JSONArray(detections.map { it.label }))
                } finally {
                    bitmap.recycle()
                }
                rows.put(result)
                episodes.getOrPut(episodeId) { mutableListOf() }.add(result)
            }
        } finally {
            detector.close()
        }
        val episodeResults = JSONArray()
        episodes.forEach { (episodeId, values) ->
            val alertRows = values.filter { it.getBoolean("edge_should_alert") }
            episodeResults.put(
                JSONObject()
                    .put("episode_id", episodeId)
                    .put("edge_should_alert", alertRows.isNotEmpty())
                    .put("edge_event_ids", JSONArray(alertRows.mapNotNull { it.optString("risk_event_id").takeIf(String::isNotBlank) }))
                    .put("frame_count", values.size)
            )
        }
        val output = JSONObject()
            .put("schema", EDGE_REPORT_SCHEMA)
            .put("created_at_utc", utcNow())
            .put("device_under_test", "instrumentation-connected-device")
            .put("model_asset", TfliteYoloDetector.MODEL_ASSET)
            .put("model_asset_sha256", sha256Asset(TfliteYoloDetector.MODEL_ASSET))
            .put("risk_config", riskConfigName)
            .put("replay_frame_step_ms", FRAME_STEP_MS)
            .put("silver_manifest_sha256", silverSha)
            .put("source_manifest_sha256", spec.getString("source_manifest_sha256"))
            .put("human_event_truth_present", false)
            .put("inference_only", true)
            .put("production_model_replacement_authorized", false)
            .put("episodes", episodeResults)
            .put("frames", rows)
            .put("important_limit", "This is a device inference report, not a safety accuracy evaluation. GPT/VLM silver labels are intentionally absent from its input.")
        val root = File(requireNotNull(targetContext.getExternalFilesDir(null)), "public-video-edge-inference")
        val outputDir = File(root, utcNow().replace(":", "").replace("-", ""))
        check(outputDir.mkdirs()) { "cannot create device inference output: $outputDir" }
        File(outputDir, "edge_events.json").writeText(output.toString(2), Charsets.UTF_8)
        Log.i(TAG, "PUBLIC_VIDEO_EDGE_REPORT ${File(outputDir, "edge_events.json").absolutePath}")
    }

    private fun assetExists(assetPath: String): Boolean = runCatching { testContext.assets.open(assetPath).close() }.isSuccess

    private fun readAssetText(assetPath: String): String =
        testContext.assets.open(assetPath).bufferedReader(Charsets.UTF_8).use { it.readText() }

    private fun sha256Asset(assetPath: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        testContext.assets.open(assetPath).use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun riskSourceJson(risk: com.linnan.blindassist.risk.RiskResult): Any {
        val detection = risk.sourceDetection ?: return JSONObject.NULL
        return JSONObject()
            .put("label", detection.label)
            .put("source", detection.source.name)
            .put("confidence", detection.confidence.toDouble())
            .put("area_ratio", detection.areaRatio.toDouble())
            .put("bounding_box", JSONObject()
                .put("left", detection.boundingBox.left.toDouble())
                .put("top", detection.boundingBox.top.toDouble())
                .put("right", detection.boundingBox.right.toDouble())
                .put("bottom", detection.boundingBox.bottom.toDouble())
            )
            .put("score_breakdown", JSONObject()
                .put("confidence", risk.scoreBreakdown.confidence.toDouble())
                .put("direction", risk.scoreBreakdown.directionWeight.toDouble())
                .put("proximity", risk.scoreBreakdown.proximityWeight.toDouble())
                .put("bottom", risk.scoreBreakdown.bottomPosition.toDouble())
                .put("area", risk.scoreBreakdown.area.toDouble())
                .put("center_lane", risk.scoreBreakdown.centerLane.toDouble())
                .put("total", risk.scoreBreakdown.total.toDouble())
            )
    }

    private fun utcNow(): String = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).apply {
        timeZone = TimeZone.getTimeZone("UTC")
    }.format(Date())

    private fun riskAnalyzerConfigFor(name: String): RiskAnalyzerConfig = when (name) {
        RISK_CONFIG_CURRENT -> RiskAnalyzerConfig.Default
        RISK_CONFIG_APPROACHING_CENTER_PERSON_CANDIDATE -> RiskAnalyzerConfig.Default
        RISK_CONFIG_CENTER_NEAR_STRICT -> RiskAnalyzerConfig.CenterNearStrict
        // Experimental only: increases side-object near thresholds while
        // leaving centre-path thresholds unchanged for causal comparison.
        RISK_CONFIG_SIDE_NEAR_STRICT -> RiskAnalyzerConfig.Default.copy(
            nearBottomRatio = 0.68f,
            nearAreaRatio = 0.16f
        )
        // Candidate taxonomy probe only; never selected by the app's default policy.
        RISK_CONFIG_ANIMAL_AWARE_CANDIDATE -> RiskAnalyzerConfig.Default.copy(
            includeHorseAsRiskTarget = true
        )
        else -> error("Unsupported publicVideoRiskConfig: $name")
    }

    private fun feedbackPlannerConfigFor(name: String): FeedbackPlannerConfig = when (name) {
        RISK_CONFIG_APPROACHING_CENTER_PERSON_CANDIDATE ->
            FeedbackPlannerConfig(enableApproachingCenterPersonMidAlert = true)
        else -> FeedbackPlannerConfig()
    }

    private fun eventTrackerConfigFor(name: String): RiskEventTrackerConfig = when (name) {
        RISK_CONFIG_APPROACHING_CENTER_PERSON_CANDIDATE ->
            RiskEventTrackerConfig(trackCenterObjectDetectorPerson = true)
        else -> RiskEventTrackerConfig()
    }

    private companion object {
        const val ASSET_PREFIX = "public_video_inference"
        const val SPEC_SCHEMA = "blindassist_public_video_edge_inference_set_v1"
        const val EDGE_REPORT_SCHEMA = "blindassist_public_video_edge_event_report_v1"
        const val ARG_REQUIRED = "publicVideoInferenceRequired"
        const val ARG_RISK_CONFIG = "publicVideoRiskConfig"
        const val RISK_CONFIG_CURRENT = "current"
        const val RISK_CONFIG_CENTER_NEAR_STRICT = "center_near_strict"
        const val RISK_CONFIG_SIDE_NEAR_STRICT = "side_near_strict"
        const val RISK_CONFIG_ANIMAL_AWARE_CANDIDATE = "animal_aware_candidate"
        const val RISK_CONFIG_APPROACHING_CENTER_PERSON_CANDIDATE = "approaching_center_person_candidate"
        const val FRAME_STEP_MS = 100L
        const val TAG = "PublicVideoEdgeInference"
    }
}
