package com.linnan.blindassist.benchmark

import android.media.MediaMetadataRetriever
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest
import kotlin.math.abs

/** R1.1 diagnostic replay: exact target frames, exact polygons, target-only event attribution. */
@RunWith(AndroidJUnit4::class)
class UstrfCrossCameraR11DeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun replayTargetLedger() {
        val inputRelative = arguments.getString(ARG_INPUT)
        if (inputRelative.isNullOrBlank()) {
            assumeTrue("R1.1 diagnostic input was not supplied", arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() != true)
            return
        }
        check(arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() == true) { "R1.1 replay requires required=true" }
        val input = JSONObject(privateFile(inputRelative, "input").readText(Charsets.UTF_8))
        check(input.getString("schema") == INPUT_SCHEMA)
        check(input.getString("diagnostic_set_role") == DIAGNOSTIC_ROLE)
        check(!input.getBoolean("training_authorized") && !input.getBoolean("production_model_replacement_authorized"))
        val ledgerFile = privateFile(input.getString("target_ledger_path"), "target ledger")
        val projectionFile = privateFile(input.getString("projection_receipt_path"), "projection receipt")
        check(sha256File(ledgerFile) == input.getString("target_ledger_sha256"))
        check(sha256File(projectionFile) == input.getString("projection_receipt_sha256"))
        val ledger = JSONObject(ledgerFile.readText(Charsets.UTF_8))
        val projection = JSONObject(projectionFile.readText(Charsets.UTF_8))
        check(ledger.getString("schema") == LEDGER_SCHEMA && projection.getString("schema") == PROJECTION_SCHEMA)
        check(projection.getString("target_ledger_sha256") == input.getString("target_ledger_sha256"))
        val ledgerEvents = ledger.getJSONArray("events").associateByString("event_id")
        val projectionEvents = projection.getJSONArray("events").associateByString("event_id")
        val videos = input.getJSONArray("sources").associateByString("event_id")

        val detector = TfliteYoloDetector(testContext)
        check(detector.isReady) { "default detector failed to initialize: ${detector.statusMessage}" }
        val sourceResults = JSONArray()
        try {
            ledgerEvents.forEach { (eventId, event) ->
                sourceResults.put(replayEvent(eventId, event, checkNotNull(projectionEvents[eventId]),
                    checkNotNull(videos[eventId]), detector))
            }
        } finally {
            detector.close()
        }
        val output = JSONObject()
            .put("schema", OUTPUT_SCHEMA)
            .put("diagnostic_set_role", DIAGNOSTIC_ROLE)
            .put("target_ledger_sha256", input.getString("target_ledger_sha256"))
            .put("projection_receipt_sha256", input.getString("projection_receipt_sha256"))
            .put("model_asset", TfliteYoloDetector.MODEL_ASSET)
            .put("model_asset_sha256", sha256Asset(TfliteYoloDetector.MODEL_ASSET))
            .put("labels_asset", TfliteYoloDetector.LABELS_ASSET)
            .put("labels_asset_sha256", sha256Asset(TfliteYoloDetector.LABELS_ASSET))
            .put("model_label_inventory", JSONArray(detector.labels))
            .put("target_match_contract_id", UstrfTargetInstanceMatcher.CONTRACT_ID)
            .put("uncertainty_frame_ratios", JSONArray(UNCERTAINTY_RATIOS))
            .put("threshold_fit", false).put("parameter_search", false).put("training_performed", false)
            .put("held_out_gate_authorized", false).put("sources", sourceResults)
        privateOutputFile(requireNotNull(arguments.getString(ARG_OUTPUT))).writeText(output.toString(2), Charsets.UTF_8)
        Log.i(TAG, "USTRF_CROSSCAM_R11_OUTPUT")
    }

    private fun replayEvent(
        eventId: String, event: JSONObject, projection: JSONObject, videoInput: JSONObject,
        detector: TfliteYoloDetector
    ): JSONObject {
        check(projection.getString("projection_mode") in setOf("per_frame", "stable_windows"))
        val target = event.getJSONObject("target_instance")
        val targetId = target.getString("target_instance_id")
        val allowlist = target.getJSONArray("detector_label_allowlist").toStringList()
        val eligibleLabels = detector.labels.filter { it in allowlist }
        val coverageStatus = if (eligibleLabels.isEmpty()) "unsupported_taxonomy" else "supported"
        val projectionFrames = projection.getJSONArray("frames").associateByString("frame_id")
        val video = privateFile(videoInput.getString("video_path"), "$eventId video")
        check(sha256File(video) == videoInput.getString("video_sha256"))
        val retriever = MediaMetadataRetriever().apply { setDataSource(video.absolutePath) }
        val frameRows = JSONArray()
        var visibleCount = 0
        var matchedCount = 0
        var zeroDetectionCount = 0
        var targetInsideCount = 0
        var cooccurrenceCount = 0
        var cooccurrenceInsideCount = 0
        var cooccurrenceInsideFrameCount = 0
        var geometryParity = true
        try {
            val frames = target.getJSONArray("frames")
            for (index in 0 until frames.length()) {
                val frozen = frames.getJSONObject(index)
                if (frozen.getString("visibility") != "visible") continue
                visibleCount++
                val frameId = frozen.getString("frame_id")
                val projected = checkNotNull(projectionFrames[frameId]) { "$eventId/$frameId missing exact projection" }
                check(projected.getLong("timestamp_ms") == frozen.getLong("timestamp_ms"))
                check(projected.getString("frame_sha256") == frozen.getString("frame_sha256"))
                check(projected.getString("status") == "admitted")
                val bitmap = checkNotNull(retriever.getFrameAtTime(frozen.getLong("timestamp_ms") * 1_000,
                    MediaMetadataRetriever.OPTION_CLOSEST)) { "$eventId/$frameId decode failed" }
                try {
                    val annotationWidth = frozen.getInt("frame_width")
                    val annotationHeight = frozen.getInt("frame_height")
                    val annotationAspect = annotationWidth.toDouble() / annotationHeight
                    val decodedAspect = bitmap.width.toDouble() / bitmap.height
                    check((annotationWidth > annotationHeight) == (bitmap.width > bitmap.height)) {
                        "$eventId/$frameId decode orientation drift"
                    }
                    check(abs(annotationAspect - decodedAspect) / annotationAspect <= MAX_ASPECT_RATIO_DRIFT) {
                        "$eventId/$frameId decode aspect drift: ${bitmap.width}x${bitmap.height} vs ${annotationWidth}x${annotationHeight}"
                    }
                    val detections = detector.detect(bitmap).detections
                    if (detections.isEmpty()) zeroDetectionCount++
                    val candidates = detections.mapIndexed { detectionIndex, detection ->
                        TargetMatchCandidate("d$detectionIndex", detection.label, detection.boundingBox)
                    }
                    val targetBox = frozen.getJSONArray("bbox_xyxy_norm").toPixelBox(bitmap.width, bitmap.height)
                    val match = UstrfTargetInstanceMatcher.match(targetBox, detector.labels, allowlist, candidates)
                    val polygon = projected.getJSONArray("route_polygon_xy_norm").toPolygon()
                    val detectionRows = JSONArray()
                    var matchedRelation: String? = null
                    var frameHasCooccurrenceInside = false
                    detections.forEachIndexed { detectionIndex, detection ->
                        val detectionId = "d$detectionIndex"
                        val profiles = UNCERTAINTY_RATIOS.map { ratio ->
                            UstrfCrossCameraProjectedCorridorGate.classify(polygon, detection.boundingBox,
                                FrameSize(bitmap.width, bitmap.height), ratio).relation.toOutputName()
                        }
                        val relation = if (profiles.distinct().size == 1) profiles.first() else "uncertain_boundary"
                        val role = if (detectionId == match.matchedDetectionId) "matched_target" else "cooccurrence"
                        if (role == "matched_target") matchedRelation = relation else {
                            cooccurrenceCount++
                            if (relation == "inside") {
                                cooccurrenceInsideCount++
                                frameHasCooccurrenceInside = true
                            }
                        }
                        detectionRows.put(JSONObject().put("detection_id", detectionId).put("label", detection.label)
                            .put("confidence", detection.confidence.toDouble()).put("bbox_xyxy_px", JSONArray(listOf(
                                detection.boundingBox.left, detection.boundingBox.top, detection.boundingBox.right, detection.boundingBox.bottom)))
                            .put("target_iou", UstrfTargetInstanceMatcher.iou(targetBox, detection.boundingBox))
                            .put("target_match_role", role).put("profile_relations", JSONArray(profiles))
                            .put("robust_relation", relation))
                    }
                    if (match.status == "matched") {
                        matchedCount++
                        if (matchedRelation == "inside") targetInsideCount++
                        if (matchedRelation != target.getString("expected_route_relation")) geometryParity = false
                    }
                    if (frameHasCooccurrenceInside) cooccurrenceInsideFrameCount++
                    frameRows.put(JSONObject().put("frame_id", frameId).put("timestamp_ms", frozen.getLong("timestamp_ms"))
                        .put("projection_frame_id", frameId).put("detection_count", detections.size)
                        .put("annotation_frame_width", annotationWidth).put("annotation_frame_height", annotationHeight)
                        .put("decoded_frame_width", bitmap.width).put("decoded_frame_height", bitmap.height)
                        .put("target_match_status", match.status).put("matched_detection_id", match.matchedDetectionId)
                        .put("matched_target_robust_relation", matchedRelation).put("detections", detectionRows))
                } finally { bitmap.recycle() }
            }
        } finally { retriever.release() }
        val expectedRelation = target.getString("expected_route_relation")
        val eventRecall: Any = if (coverageStatus == "supported" && expectedRelation == "inside") {
            if (targetInsideCount > 0) 1 else 0
        } else JSONObject.NULL
        val falseAlarm: Any = if (coverageStatus == "supported" && expectedRelation == "outside") {
            targetInsideCount > 0
        } else JSONObject.NULL
        return JSONObject().put("event_id", eventId).put("source_id", event.getString("source_id"))
            .put("target_instance_id", targetId).put("expected_route_relation", expectedRelation)
            .put("detector_coverage", JSONObject().put("status", coverageStatus).put("target_semantic_type", target.getString("semantic_type"))
                .put("eligible_labels", JSONArray(eligibleLabels)))
            .put("visible_target_frame_count", visibleCount).put("target_match_frame_count", matchedCount)
            .put("zero_detection_frame_count", zeroDetectionCount).put("target_robust_inside_frame_count", targetInsideCount)
            .put("cooccurrence_detection_count", cooccurrenceCount).put("cooccurrence_robust_inside_count", cooccurrenceInsideCount)
            .put("legacy_any_inside_alert_frame_count", cooccurrenceInsideFrameCount)
            .put("cooccurrence_runtime_alert_count", cooccurrenceInsideFrameCount)
            .put("android_oracle_geometry_parity", if (matchedCount > 0) geometryParity else JSONObject.NULL)
            .put("event_recall", eventRecall).put("false_alarm", falseAlarm).put("frames", frameRows)
    }

    private fun privateFile(relative: String, label: String): File {
        check(relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        return File(root, relative).canonicalFile.also { check(it.path.startsWith(root.path + File.separator) && it.isFile) { "$label unavailable" } }
    }
    private fun privateOutputFile(relative: String): File {
        check(relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        return File(root, relative).canonicalFile.also { check(it.path.startsWith(root.path + File.separator)); it.parentFile?.mkdirs() }
    }
    private fun JSONArray.associateByString(key: String) = (0 until length()).associate { getJSONObject(it).getString(key) to getJSONObject(it) }
    private fun JSONArray.toStringList() = (0 until length()).map { getString(it) }
    private fun JSONArray.toPolygon() = (0 until length()).map { index -> getJSONArray(index).let { CrossCameraCorridorPoint(it.getDouble(0), it.getDouble(1)) } }
    private fun JSONArray.toPixelBox(width: Int, height: Int) = BoundingBox(
        (getDouble(0) * width).toFloat(), (getDouble(1) * height).toFloat(),
        (getDouble(2) * width).toFloat(), (getDouble(3) * height).toFloat())
    private fun CrossCameraCorridorRelation.toOutputName() = when (this) {
        CrossCameraCorridorRelation.INSIDE -> "inside"
        CrossCameraCorridorRelation.OUTSIDE -> "outside"
        CrossCameraCorridorRelation.UNCERTAIN_BOUNDARY -> "uncertain_boundary"
    }
    private fun sha256File(file: File) = file.inputStream().use { digest(it) }
    private fun sha256Asset(path: String) = testContext.assets.open(path).use { digest(it) }
    private fun digest(input: java.io.InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256"); val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private companion object {
        const val INPUT_SCHEMA = "blindassist_ustrf_crosscam_target_aware_android_input_v2"
        const val OUTPUT_SCHEMA = "blindassist_ustrf_crosscam_target_aware_android_output_v2"
        const val LEDGER_SCHEMA = "blindassist_ustrf_crosscam_target_instance_ledger_v1"
        const val PROJECTION_SCHEMA = "blindassist_ustrf_crosscam_frame_projection_receipt_v2"
        const val DIAGNOSTIC_ROLE = "seen_diagnostic_not_held_out"
        const val ARG_REQUIRED = "ustrfCrosscamR11Required"
        const val ARG_INPUT = "ustrfCrosscamR11Input"
        const val ARG_OUTPUT = "ustrfCrosscamR11Output"
        const val TAG = "UstrfCrosscamR11"
        const val MAX_ASPECT_RATIO_DRIFT = 0.005
        val UNCERTAINTY_RATIOS = listOf(0.01, 0.02, 0.03)
    }
}
