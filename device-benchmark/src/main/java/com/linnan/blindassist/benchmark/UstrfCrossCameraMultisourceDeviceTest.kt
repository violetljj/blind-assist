package com.linnan.blindassist.benchmark

import android.media.MediaMetadataRetriever
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/** Benchmark-only replay of the preregistered six-source projected-corridor geometry gate. */
@RunWith(AndroidJUnit4::class)
class UstrfCrossCameraMultisourceDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun replayFrozenSources() {
        val required = arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() ?: false
        val inputRelative = arguments.getString(ARG_INPUT)
        if (inputRelative.isNullOrBlank()) {
            assumeTrue("cross-camera multisource input was not supplied", !required)
            return
        }
        check(required) { "cross-camera multisource replay requires required=true" }
        val inputFile = privateFile(inputRelative, "input")
        val outputFile = privateOutputFile(requireNotNull(arguments.getString(ARG_OUTPUT)))
        val input = JSONObject(inputFile.readText(Charsets.UTF_8))
        check(input.getString("schema") == INPUT_SCHEMA)
        check(input.getString("preregistration_sha256").matches(SHA_REGEX))
        check(input.getJSONArray("uncertainty_frame_ratios").toDoubleList() == UNCERTAINTY_RATIOS)
        check(!input.getBoolean("threshold_fit"))
        check(!input.getBoolean("training_authorized"))
        check(!input.getBoolean("production_model_replacement_authorized"))

        val detector = TfliteYoloDetector(testContext)
        check(detector.isReady) { "default detector failed to initialize: ${detector.statusMessage}" }
        val sourceResults = JSONArray()
        try {
            val sources = input.getJSONArray("sources")
            for (sourceIndex in 0 until sources.length()) {
                val source = sources.getJSONObject(sourceIndex)
                val result = runCatching { replaySource(source, detector) }.getOrElse { error ->
                    JSONObject()
                        .put("source_id", source.optString("source_id", "unknown-source-$sourceIndex"))
                        .put("event_id", source.optString("event_id", "unknown-event-$sourceIndex"))
                        .put("expected_class", source.optString("expected_class", "unknown"))
                        .put("window_ms", JSONArray(listOf(
                            source.optLong("window_start_ms", -1L), source.optLong("window_end_ms", -1L)
                        )))
                        .put("status", "unresolved")
                        .put("unresolved_reason", error.message ?: error.javaClass.simpleName)
                        .put("frame_count", 0)
                        .put("detection_relation_counts", JSONObject(mapOf("inside" to 0, "outside" to 0, "uncertain" to 0)))
                        .put("profile_summaries", JSONArray(UNCERTAINTY_RATIOS.map { ratio ->
                            JSONObject().put("uncertainty_frame_ratio", ratio)
                                .put("inside_count", 0).put("outside_count", 0).put("uncertain_count", 0)
                        }))
                        .put("robust_inside_frame_count", 0)
                        .put("robust_inside", false)
                        .put("event_recall", JSONObject.NULL)
                        .put("false_alarm", JSONObject.NULL)
                        .put("frames", JSONArray())
                }
                sourceResults.put(result)
            }
        } finally {
            detector.close()
        }
        val output = JSONObject()
            .put("schema", OUTPUT_SCHEMA)
            .put("created_at_utc", utcNow())
            .put("device", JSONObject()
                .put("manufacturer", android.os.Build.MANUFACTURER)
                .put("model", android.os.Build.MODEL)
                .put("sdk", android.os.Build.VERSION.SDK_INT))
            .put("model_asset", TfliteYoloDetector.MODEL_ASSET)
            .put("model_asset_sha256", sha256Asset(TfliteYoloDetector.MODEL_ASSET))
            .put("labels_asset", TfliteYoloDetector.LABELS_ASSET)
            .put("labels_asset_sha256", sha256Asset(TfliteYoloDetector.LABELS_ASSET))
            .put("gate_contract_id", UstrfCrossCameraProjectedCorridorGate.CONTRACT_ID)
            .put("preregistration_sha256", input.getString("preregistration_sha256"))
            .put("uncertainty_frame_ratios", JSONArray(UNCERTAINTY_RATIOS))
            .put("threshold_fit", false)
            .put("parameter_search", false)
            .put("thresholds_changed", false)
            .put("training_performed", false)
            .put("pexels_used_for_gate", false)
            .put("training_authorized", false)
            .put("production_model_replacement_authorized", false)
            .put("sources", sourceResults)
        outputFile.writeText(output.toString(2), Charsets.UTF_8)
        Log.i(TAG, "USTRF_CROSSCAM_MULTISOURCE_OUTPUT ${outputFile.absolutePath}")
    }

    private fun replaySource(source: JSONObject, detector: TfliteYoloDetector): JSONObject {
        val sourceId = source.getString("source_id")
        val expectedClass = source.getString("expected_class")
        check(expectedClass == "positive" || expectedClass == "negative")
        val eventId = source.getString("event_id")
        val startMs = source.getLong("window_start_ms")
        val endMs = source.getLong("window_end_ms")
        val projectionStatus = source.getString("projection_status")
        if (projectionStatus != ADMITTED_PROJECTION_STATUS) {
            return unresolvedProjectionSource(source, eventId, startMs, endMs, projectionStatus)
        }
        val video = privateFile(source.getString("video_path"), "$sourceId video")
        check(sha256File(video) == source.getString("video_sha256"))
        val projection = privateFile(source.getString("projection_receipt_path"), "$sourceId projection receipt")
        check(sha256File(projection) == source.getString("projection_receipt_sha256"))
        val projectionJson = JSONObject(projection.readText(Charsets.UTF_8))
        check(projectionJson.getString("schema") == PROJECTION_SCHEMA)
        check(!projectionJson.getBoolean("training_authorized"))
        check(!projectionJson.getBoolean("android_runtime_change_authorized"))
        val polygonJson = source.getJSONArray("route_polygon_xy_norm")
        check(projectionJson.getJSONArray("route_polygon_xy_norm").toString() == polygonJson.toString())
        val polygon = polygonJson.toPolygon()
        val stepMs = source.getLong("replay_step_ms")
        check(startMs >= 0 && endMs > startMs && stepMs == FROZEN_STEP_MS)

        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(video.absolutePath)
        val frames = JSONArray()
        val relationCounts = linkedMapOf("inside" to 0, "outside" to 0, "uncertain" to 0)
        val profileCounts = UNCERTAINTY_RATIOS.map {
            linkedMapOf("inside" to 0, "outside" to 0, "uncertain" to 0)
        }
        var robustInsideFrameCount = 0
        try {
            var timestampMs = startMs
            while (timestampMs < endMs) {
                val bitmap = checkNotNull(retriever.getFrameAtTime(timestampMs * 1_000, MediaMetadataRetriever.OPTION_CLOSEST)) {
                    "cannot decode $sourceId at ${timestampMs}ms"
                }
                try {
                    val detections = detector.detect(bitmap).detections
                    val detectionRows = JSONArray()
                    var frameHasRobustInside = false
                    detections.forEach { detection ->
                        val profileResults = UNCERTAINTY_RATIOS.map { ratio ->
                            UstrfCrossCameraProjectedCorridorGate.classify(
                                polygon, detection.boundingBox, FrameSize(bitmap.width, bitmap.height), ratio
                            )
                        }
                        profileResults.forEachIndexed { profileIndex, result ->
                            val profileRelation = result.relation.toOutputName()
                            profileCounts[profileIndex][profileRelation] =
                                checkNotNull(profileCounts[profileIndex][profileRelation]) + 1
                        }
                        val relation = if (profileResults.map { it.relation }.distinct().size == 1) {
                            profileResults.first().relation.toOutputName()
                        } else {
                            "uncertain"
                        }
                        relationCounts[relation] = checkNotNull(relationCounts[relation]) + 1
                        if (relation == "inside") frameHasRobustInside = true
                        detectionRows.put(JSONObject()
                            .put("label", detection.label)
                            .put("confidence", detection.confidence.toDouble())
                            .put("bbox", JSONArray(listOf(
                                detection.boundingBox.left.toDouble(), detection.boundingBox.top.toDouble(),
                                detection.boundingBox.right.toDouble(), detection.boundingBox.bottom.toDouble()
                            )))
                            .put("profile_relations", JSONArray(profileResults.map { it.relation.toOutputName() }))
                            .put("robust_relation", relation))
                    }
                    if (frameHasRobustInside) robustInsideFrameCount++
                    frames.put(JSONObject()
                        .put("timestamp_ms", timestampMs)
                        .put("frame_width", bitmap.width)
                        .put("frame_height", bitmap.height)
                        .put("detection_count", detections.size)
                        .put("robust_inside", frameHasRobustInside)
                        .put("detections", detectionRows))
                } finally {
                    bitmap.recycle()
                }
                timestampMs += stepMs
            }
        } finally {
            retriever.release()
        }
        val eventRecall = if (expectedClass == "positive") if (robustInsideFrameCount > 0) 1 else 0 else JSONObject.NULL
        val falseAlarm = if (expectedClass == "negative") robustInsideFrameCount > 0 else JSONObject.NULL
        val profileSummaries = JSONArray()
        UNCERTAINTY_RATIOS.forEachIndexed { index, ratio ->
            profileSummaries.put(JSONObject()
                .put("uncertainty_frame_ratio", ratio)
                .put("inside_count", profileCounts[index].getValue("inside"))
                .put("outside_count", profileCounts[index].getValue("outside"))
                .put("uncertain_count", profileCounts[index].getValue("uncertain")))
        }
        return JSONObject()
            .put("source_id", sourceId)
            .put("event_id", eventId)
            .put("expected_class", expectedClass)
            .put("window_ms", JSONArray(listOf(startMs, endMs)))
            .put("status", "resolved")
            .put("unresolved_reason", JSONObject.NULL)
            .put("video_sha256", source.getString("video_sha256"))
            .put("projection_receipt_sha256", source.getString("projection_receipt_sha256"))
            .put("route_polygon_xy_norm", polygonJson)
            .put("frame_count", frames.length())
            .put("detection_relation_counts", JSONObject(relationCounts as Map<*, *>))
            .put("profile_summaries", profileSummaries)
            .put("robust_inside_frame_count", robustInsideFrameCount)
            .put("robust_inside", robustInsideFrameCount > 0)
            .put("event_recall", eventRecall)
            .put("false_alarm", falseAlarm)
            .put("frames", frames)
    }

    private fun unresolvedProjectionSource(
        source: JSONObject,
        eventId: String,
        startMs: Long,
        endMs: Long,
        reason: String
    ): JSONObject {
        val emptyProfiles = JSONArray(UNCERTAINTY_RATIOS.map { ratio ->
            JSONObject()
                .put("uncertainty_frame_ratio", ratio)
                .put("inside_count", 0)
                .put("outside_count", 0)
                .put("uncertain_count", 0)
        })
        return JSONObject()
            .put("source_id", source.getString("source_id"))
            .put("event_id", eventId)
            .put("expected_class", source.getString("expected_class"))
            .put("window_ms", JSONArray(listOf(startMs, endMs)))
            .put("status", "unresolved")
            .put("unresolved_reason", reason)
            .put("video_sha256", source.opt("video_sha256"))
            .put("projection_receipt_sha256", source.opt("projection_receipt_sha256"))
            .put("frame_count", 0)
            .put("detection_relation_counts", JSONObject(mapOf("inside" to 0, "outside" to 0, "uncertain" to 0)))
            .put("profile_summaries", emptyProfiles)
            .put("robust_inside_frame_count", 0)
            .put("robust_inside", false)
            .put("event_recall", JSONObject.NULL)
            .put("false_alarm", JSONObject.NULL)
            .put("frames", JSONArray())
    }

    private fun privateFile(relative: String, label: String): File {
        check(relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        val file = File(root, relative).canonicalFile
        check(file.path.startsWith(root.path + File.separator) && file.isFile) { "$label is unavailable" }
        return file
    }

    private fun privateOutputFile(relative: String): File {
        check(relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        val file = File(root, relative).canonicalFile
        check(file.path.startsWith(root.path + File.separator))
        check(file.parentFile?.mkdirs() == true || file.parentFile?.isDirectory == true)
        return file
    }

    private fun JSONArray.toDoubleList() = (0 until length()).map { getDouble(it) }
    private fun JSONArray.toPolygon() = (0 until length()).map { index ->
        val point = getJSONArray(index)
        CrossCameraCorridorPoint(point.getDouble(0), point.getDouble(1))
    }
    private fun CrossCameraCorridorRelation.toOutputName() = when (this) {
        CrossCameraCorridorRelation.INSIDE -> "inside"
        CrossCameraCorridorRelation.OUTSIDE -> "outside"
        CrossCameraCorridorRelation.UNCERTAIN_BOUNDARY -> "uncertain"
    }
    private fun sha256File(file: File): String = file.inputStream().use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        digest.digest().joinToString("") { "%02x".format(it) }
    }
    private fun sha256Asset(path: String): String = testContext.assets.open(path).use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        digest.digest().joinToString("") { "%02x".format(it) }
    }
    private fun utcNow(): String = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).apply {
        timeZone = TimeZone.getTimeZone("UTC")
    }.format(Date())

    private companion object {
        const val INPUT_SCHEMA = "blindassist_ustrf_crosscam_multisource_android_input_v1"
        const val OUTPUT_SCHEMA = "blindassist_ustrf_crosscam_multisource_android_output_v1"
        const val PROJECTION_SCHEMA = "blindassist_ustrf_crosscam_route_projection_receipt_v1"
        const val ARG_REQUIRED = "ustrfCrosscamMultisourceRequired"
        const val ARG_INPUT = "ustrfCrosscamMultisourceInput"
        const val ARG_OUTPUT = "ustrfCrosscamMultisourceOutput"
        const val FROZEN_STEP_MS = 500L
        const val ADMITTED_PROJECTION_STATUS = "admitted_static_proxy_for_full_window"
        const val TAG = "UstrfCrosscamMultisource"
        val UNCERTAINTY_RATIOS = listOf(0.01, 0.02, 0.03)
        val SHA_REGEX = Regex("^[0-9a-f]{64}$")
    }
}
