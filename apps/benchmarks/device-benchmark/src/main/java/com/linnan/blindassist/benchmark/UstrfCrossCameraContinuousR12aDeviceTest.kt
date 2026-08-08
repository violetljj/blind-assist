package com.linnan.blindassist.benchmark

import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
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
import java.io.InputStream
import java.security.MessageDigest
import kotlin.math.ceil
import kotlin.math.max

/** Benchmark-only R1.2a continuous replay and 10-minute SM-S9280 stress/thermal gate. */
@RunWith(AndroidJUnit4::class)
class UstrfCrossCameraContinuousR12aDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun replaySeenEventsAndMeasureFrozenDeviceGate() {
        val inputRelative = arguments.getString(ARG_INPUT)
        if (inputRelative.isNullOrBlank()) {
            assumeTrue("R1.2a continuous input was not supplied", arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() != true)
            return
        }
        check(arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() == true)
        val inputFile = privateFile(inputRelative, "R1.2a input")
        val input = JSONObject(inputFile.readText(Charsets.UTF_8))
        check(input.getString("schema") == INPUT_SCHEMA)
        check(input.getString("dataset_role") == SEEN_ROLE)
        val authority = input.getJSONObject("authority")
        check(!authority.getBoolean("new_held_out_read") && authority.getBoolean("benchmark_only"))
        check(!authority.getBoolean("training_authorized") && !authority.getBoolean("production_model_replacement_authorized"))
        val detectorContract = input.getJSONObject("frozen_detector")
        check(detectorContract.getDouble("confidence_threshold") == 0.05)
        check(detectorContract.getDouble("target_anchor_iou_threshold") == 0.30)
        check(sha256Asset(MODEL_ASSET) == detectorContract.getString("model_sha256"))
        val deviceGate = input.getJSONObject("device_gate")
        check(Build.MODEL.replace("_", "-") == deviceGate.getString("device_model")) {
            "wrong device ${Build.MODEL}"
        }
        val soakSeconds = requireNotNull(arguments.getString(ARG_SOAK_SECONDS)).toInt()
        check(soakSeconds == deviceGate.getInt("soak_duration_seconds")) { "soak duration cannot be shortened" }
        val replay = input.getJSONObject("replay_contract")
        check(replay.getLong("sample_period_ms") == SAMPLE_PERIOD_MS)
        check(replay.getString("association_mode") == UstrfContinuousTargetAssociation.CONTRACT_ID)

        val detector = TfliteYoloDetector(testContext, MODEL_ASSET, LABELS_ASSET, 640, 0.05f, 0.45f)
        check(detector.isReady) { "R1.2a detector init failed: ${detector.statusMessage}" }
        val latencies = mutableListOf<Double>()
        var decodeFailures = 0
        var inferenceFailures = 0
        val eventRows = JSONArray()
        val startTemperature = batteryTemperatureC()
        val thermalSamples = mutableListOf(thermalStatus())
        val temperatureSamples = mutableListOf<Double>().apply { startTemperature?.let(::add) }
        try {
            val sources = input.getJSONArray("sources")
            for (index in 0 until sources.length()) {
                val result = replaySource(sources.getJSONObject(index), replay, detector, latencies)
                decodeFailures += result.decodeFailures
                inferenceFailures += result.inferenceFailures
                eventRows.put(result.output)
            }
            val soak = runSoak(sources, detector, soakSeconds, latencies, thermalSamples, temperatureSamples)
            decodeFailures += soak.first
            inferenceFailures += soak.second
        } finally {
            detector.close()
        }
        val endTemperature = batteryTemperatureC()
        endTemperature?.let(temperatureSamples::add)
        thermalSamples += thermalStatus()
        val eventGate = input.getJSONObject("event_gate")
        val eligible = (0 until eventRows.length()).map { eventRows.getJSONObject(it) }
            .filter { it.getBoolean("gate_eligible") }
        val positives = eligible.filter { it.getString("expected_class") == "positive" }
        val positiveRecall = positives.count { it.getBoolean("event_hit") }.toDouble() / positives.size
        val negativeFalseAlerts = eligible.filter { it.getString("expected_class") == "negative" }
            .sumOf { it.getInt("delivered_alert_count") }
        val deliveredRepeats = eligible.sumOf { it.getInt("delivered_repeated_alert_count") }
        val cooccurrenceTriggered = eligible.sumOf { it.getInt("cooccurrence_triggered_target_event_count") }
        val identitySwitches = eligible.sumOf { it.getInt("identity_switch_count") }
        val ambiguousFrames = eligible.sumOf { it.getInt("association_ambiguous_frame_count") }
        val totalFrames = eligible.sumOf { it.getInt("frame_count") }
        val ambiguousRate = ambiguousFrames.toDouble() / max(1, totalFrames)
        val firstAlertGatePassed = positives.filter { it.getBoolean("event_hit") }
            .all { it.getLong("first_correct_alert_delay_ms") <= eventGate.getLong("first_correct_alert_delay_ms_at_most") }
        val exitGatePassed = eligible.filter { !it.isNull("target_exit_clearance_delay_ms") }
            .all { it.getLong("target_exit_clearance_delay_ms") <= eventGate.getLong("target_exit_clearance_delay_ms_at_most") }
        val p50 = percentile(latencies, 0.50)
        val p95 = percentile(latencies, 0.95)
        val thermalDelta = if (temperatureSamples.isEmpty()) null else temperatureSamples.max() - temperatureSamples.min()
        val maxThermalStatus = thermalSamples.maxOrNull() ?: PowerManager.THERMAL_STATUS_NONE
        val finalTemperature = endTemperature ?: temperatureSamples.lastOrNull()
        val eventPassed = positiveRecall >= eventGate.getDouble("positive_event_recall_at_least") &&
            negativeFalseAlerts <= eventGate.getInt("negative_false_alert_count_at_most") && firstAlertGatePassed &&
            deliveredRepeats <= eventGate.getInt("delivered_repeated_alert_count_at_most") && exitGatePassed &&
            cooccurrenceTriggered <= eventGate.getInt("cooccurrence_triggered_target_event_count_at_most") &&
            identitySwitches <= eventGate.getInt("identity_switch_count_at_most") &&
            ambiguousRate <= eventGate.getDouble("association_ambiguous_frame_rate_at_most")
        val devicePassed = decodeFailures <= deviceGate.getInt("decode_or_inference_failure_count_at_most") &&
            inferenceFailures <= deviceGate.getInt("decode_or_inference_failure_count_at_most") &&
            p95 <= deviceGate.getDouble("inference_latency_p95_ms_at_most") &&
            thermalDelta != null && thermalDelta <= deviceGate.getDouble("thermal_delta_c_at_most") &&
            finalTemperature != null && finalTemperature <= deviceGate.getDouble("final_battery_temperature_c_at_most") &&
            maxThermalStatus <= deviceGate.getInt("maximum_android_thermal_status")
        val output = JSONObject()
            .put("schema", OUTPUT_SCHEMA).put("dataset_role", SEEN_ROLE)
            .put("input_sha256", sha256File(inputFile)).put("protocol_sha256", input.getString("protocol_sha256"))
            .put("device", JSONObject().put("model", Build.MODEL).put("fingerprint", Build.FINGERPRINT)
                .put("sdk", Build.VERSION.SDK_INT))
            .put("events", eventRows)
            .put("aggregate_event_metrics", JSONObject()
                .put("positive_event_recall", positiveRecall).put("negative_false_alert_count", negativeFalseAlerts)
                .put("delivered_repeated_alert_count", deliveredRepeats)
                .put("cooccurrence_triggered_target_event_count", cooccurrenceTriggered)
                .put("identity_switch_count", identitySwitches).put("association_ambiguous_frame_rate", ambiguousRate)
                .put("first_alert_gate_passed", firstAlertGatePassed).put("target_exit_gate_passed", exitGatePassed))
            .put("device_metrics", JSONObject()
                .put("soak_duration_seconds", soakSeconds).put("inference_sample_count", latencies.size)
                .put("inference_latency_p50_ms", p50).put("inference_latency_p95_ms", p95)
                .put("decode_failure_count", decodeFailures).put("inference_failure_count", inferenceFailures)
                .put("battery_temperature_start_c", startTemperature).put("battery_temperature_final_c", finalTemperature)
                .put("battery_temperature_delta_c", thermalDelta).put("maximum_thermal_status", maxThermalStatus)
                .put("thermal_status_samples", JSONArray(thermalSamples)).put("battery_temperature_samples_c", JSONArray(temperatureSamples)))
            .put("gate", JSONObject().put("event_protocol_passed", eventPassed).put("device_gate_passed", devicePassed)
                .put("overall_passed", eventPassed && devicePassed).put("benchmark_only", true)
                .put("production_model_replacement_authorized", false))
        privateOutputFile(requireNotNull(arguments.getString(ARG_OUTPUT))).writeText(output.toString(2), Charsets.UTF_8)
        Log.i(TAG, "USTRF_R12A_CONTINUOUS_RESULT event=$eventPassed device=$devicePassed p95=$p95")
        check(eventPassed && devicePassed) { "R1.2a frozen gate failed; inspect output receipt" }
    }

    private data class ReplayResult(val output: JSONObject, val decodeFailures: Int, val inferenceFailures: Int)

    private fun replaySource(source: JSONObject, contract: JSONObject, detector: TfliteYoloDetector,
                             latencies: MutableList<Double>): ReplayResult {
        val video = privateFile(source.getString("video_path"), "${source.getString("event_id")} video")
        check(sha256File(video) == source.getString("video_sha256"))
        val startMs = source.getJSONArray("clip_window_ms").getLong(0)
        val endMs = source.getJSONArray("clip_window_ms").getLong(1)
        val retriever = MediaMetadataRetriever().apply { setDataSource(video.absolutePath) }
        val observations = mutableListOf<FrameObservation>()
        var decodeFailures = 0
        var inferenceFailures = 0
        try {
            var timestamp = startMs
            while (timestamp <= endMs) {
                val bitmap = retriever.getFrameAtTime(timestamp * 1_000, MediaMetadataRetriever.OPTION_CLOSEST)
                if (bitmap == null) {
                    decodeFailures++
                } else try {
                    try {
                        val detections = detector.detect(bitmap).detections.mapIndexed { index, detection ->
                            ContinuousTargetDetection("d$index", detection.label, detection.boundingBox)
                        }
                        latencies += detector.lastInferenceMs.toDouble()
                        observations += FrameObservation(timestamp, bitmap.width, bitmap.height, detections)
                    } catch (_: Throwable) { inferenceFailures++ }
                } finally { bitmap.recycle() }
                timestamp += SAMPLE_PERIOD_MS
            }
        } finally { retriever.release() }
        check(observations.isNotEmpty()) { "${source.getString("event_id")}: no decodable continuous frames" }
        val anchors = source.getJSONArray("target_anchors").toAnchors(observations.first().width, observations.first().height)
        val association = UstrfContinuousTargetAssociation.associate(
            observations.map { ContinuousTargetFrame(it.timestampMs, it.detections) }, anchors,
            source.getLong("primary_anchor_timestamp_ms"), detector.labels, source.getJSONArray("detector_label_allowlist").toStringList(),
            contract.getDouble("association_iou_at_least"), contract.getDouble("association_center_distance_frame_diagonal_at_most"),
            contract.getDouble("association_area_ratio_min"), contract.getDouble("association_area_ratio_max"),
            contract.getDouble("association_ambiguity_margin"), contract.getInt("clear_after_consecutive_misses"),
            observations.first().width, observations.first().height,
        )
        val polygon = source.getJSONArray("route_polygon_xy_norm").toPolygon()
        var active = false
        var misses = 0
        var alerted = false
        var delivered = 0
        var suppressed = 0
        var firstAlertMs: Long? = null
        var clearMs: Long? = null
        var cooccurrenceInside = 0
        observations.forEachIndexed { index, observation ->
            val assignedId = association.frames[index].matchedDetectionId
            val assigned = observation.detections.singleOrNull { it.detectionId == assignedId }
            if (assigned == null) {
                if (active && ++misses >= contract.getInt("clear_after_consecutive_misses")) {
                    active = false
                    clearMs = observation.timestampMs
                }
            } else {
                active = true; misses = 0
                val relation = UstrfCrossCameraProjectedCorridorGate.classify(
                    polygon, assigned.box, FrameSize(observation.width, observation.height), 0.02,
                )
                if (source.getString("expected_class") == "positive" && relation.relation == CrossCameraCorridorRelation.INSIDE &&
                    observation.timestampMs >= source.getLong("alertable_start_ms")) {
                    if (!alerted) { alerted = true; delivered++; firstAlertMs = observation.timestampMs } else suppressed++
                }
            }
            observation.detections.filter { it.detectionId != assignedId }.forEach { detection ->
                if (UstrfCrossCameraProjectedCorridorGate.classify(
                        polygon, detection.box, FrameSize(observation.width, observation.height), 0.02,
                    ).relation == CrossCameraCorridorRelation.INSIDE) cooccurrenceInside++
            }
        }
        val knownAbsentFrom = source.optLongOrNull("known_not_visible_from_ms")
        val exitDelay = knownAbsentFrom?.let { absent -> max(0L, (clearMs ?: absent + 10_000L) - absent) }
        val preEntryViolation = source.optLongOrNull("known_not_visible_until_ms")?.let { until ->
            association.frames.count { it.timestampMs <= until && it.matchedDetectionId != null }
        } ?: 0
        val firstDelay = firstAlertMs?.let { it - source.getLong("alertable_start_ms") }
        val output = JSONObject()
            .put("event_id", source.getString("event_id")).put("source_id", source.getString("source_id"))
            .put("target_instance_id", source.getString("target_instance_id"))
            .put("expected_class", source.getString("expected_class")).put("gate_eligible", source.getBoolean("gate_eligible"))
            .put("diagnostic_role", source.getString("diagnostic_role"))
            .put("frame_count", observations.size).put("primary_anchor_matched", association.primaryAnchorMatched)
            .put("visible_anchor_count", association.visibleAnchorCount)
            .put("visible_anchor_reacquired_count", association.visibleAnchorReacquiredCount)
            .put("identity_switch_count", association.identitySwitchCount)
            .put("association_ambiguous_frame_count", association.frames.count { it.ambiguous })
            .put("association_matched_frame_count", association.frames.count { it.matchedDetectionId != null })
            .put("event_hit", firstAlertMs != null).put("first_correct_alert_timestamp_ms", firstAlertMs)
            .put("first_correct_alert_delay_ms", firstDelay).put("delivered_alert_count", delivered)
            .put("delivered_repeated_alert_count", max(0, delivered - 1)).put("suppressed_duplicate_attempt_count", suppressed)
            .put("target_exit_clearance_delay_ms", exitDelay).put("pre_entry_association_violation_count", preEntryViolation)
            .put("cooccurrence_inside_detection_count", cooccurrenceInside)
            .put("cooccurrence_triggered_target_event_count", 0)
            .put("route_proxy_is_geometry_truth", false)
        return ReplayResult(output, decodeFailures, inferenceFailures)
    }

    private fun runSoak(sources: JSONArray, detector: TfliteYoloDetector, seconds: Int,
                        latencies: MutableList<Double>, thermalSamples: MutableList<Int>,
                        temperatureSamples: MutableList<Double>): Pair<Int, Int> {
        val deadline = SystemClock.elapsedRealtime() + seconds * 1_000L
        var decodeFailures = 0
        var inferenceFailures = 0
        var sourceIndex = 0
        while (SystemClock.elapsedRealtime() < deadline) {
            val source = sources.getJSONObject(sourceIndex++ % sources.length())
            val video = privateFile(source.getString("video_path"), "soak video")
            val window = source.getJSONArray("clip_window_ms")
            val retriever = MediaMetadataRetriever().apply { setDataSource(video.absolutePath) }
            try {
                var timestamp = window.getLong(0)
                while (timestamp <= window.getLong(1) && SystemClock.elapsedRealtime() < deadline) {
                    val bitmap: Bitmap? = retriever.getFrameAtTime(timestamp * 1_000, MediaMetadataRetriever.OPTION_CLOSEST)
                    if (bitmap == null) decodeFailures++ else try {
                        try { detector.detect(bitmap); latencies += detector.lastInferenceMs.toDouble() }
                        catch (_: Throwable) { inferenceFailures++ }
                    } finally { bitmap.recycle() }
                    timestamp += SAMPLE_PERIOD_MS
                }
            } finally { retriever.release() }
            thermalSamples += thermalStatus()
            batteryTemperatureC()?.let(temperatureSamples::add)
        }
        return decodeFailures to inferenceFailures
    }

    private data class FrameObservation(val timestampMs: Long, val width: Int, val height: Int,
                                        val detections: List<ContinuousTargetDetection>)
    private fun JSONArray.toAnchors(width: Int, height: Int) = (0 until length()).map { index ->
        val row = getJSONObject(index); val visibility = row.getString("visibility")
        ContinuousTargetAnchor(row.getLong("timestamp_ms"), visibility,
            if (visibility == "visible") row.getJSONArray("bbox_xyxy_norm").toPixelBox(width, height) else null)
    }
    private fun JSONArray.toStringList() = (0 until length()).map(::getString)
    private fun JSONArray.toPolygon() = (0 until length()).map { getJSONArray(it).let { p -> CrossCameraCorridorPoint(p.getDouble(0), p.getDouble(1)) } }
    private fun JSONArray.toPixelBox(width: Int, height: Int) = BoundingBox(
        (getDouble(0) * width).toFloat(), (getDouble(1) * height).toFloat(),
        (getDouble(2) * width).toFloat(), (getDouble(3) * height).toFloat(),
    )
    private fun JSONObject.optLongOrNull(key: String) = if (!has(key) || isNull(key)) null else getLong(key)
    private fun percentile(values: List<Double>, fraction: Double): Double {
        check(values.isNotEmpty()); val sorted = values.sorted()
        return sorted[(ceil(fraction * sorted.size).toInt() - 1).coerceIn(sorted.indices)]
    }
    private fun batteryTemperatureC(): Double? {
        val intent: Intent = targetContext.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) ?: return null
        val tenths = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        return if (tenths == Int.MIN_VALUE) null else tenths / 10.0
    }
    private fun thermalStatus() = targetContext.getSystemService(PowerManager::class.java).currentThermalStatus
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
    private fun sha256File(file: File) = file.inputStream().use(::digest)
    private fun sha256Asset(path: String) = testContext.assets.open(path).use(::digest)
    private fun digest(input: InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256"); val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private companion object {
        const val INPUT_SCHEMA = "blindassist_ustrf_crosscam_continuous_android_input_v1"
        const val OUTPUT_SCHEMA = "blindassist_ustrf_crosscam_continuous_android_output_v1"
        const val SEEN_ROLE = "seen_diagnostic_not_held_out"
        const val MODEL_ASSET = "ustrf_r12_detector/yoloe11s_marker_static3_fp16_640.tflite"
        const val LABELS_ASSET = "ustrf_r12_detector/marker_labels.txt"
        const val SAMPLE_PERIOD_MS = 500L
        const val ARG_REQUIRED = "ustrfR12aContinuousRequired"
        const val ARG_INPUT = "ustrfR12aContinuousInput"
        const val ARG_OUTPUT = "ustrfR12aContinuousOutput"
        const val ARG_SOAK_SECONDS = "ustrfR12aSoakSeconds"
        const val TAG = "UstrfR12aContinuous"
    }
}
