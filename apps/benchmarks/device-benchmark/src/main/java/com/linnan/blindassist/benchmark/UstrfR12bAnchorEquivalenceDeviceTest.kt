package com.linnan.blindassist.benchmark

import android.graphics.BitmapFactory
import android.media.MediaMetadataRetriever
import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** P0: prove static canary anchors and Android video replay frames have equivalent target outcomes. */
@RunWith(AndroidJUnit4::class)
class UstrfR12bAnchorEquivalenceDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun auditStaticCanaryAgainstSameTimestampVideoDecode() {
        if (arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() != true) {
            assumeTrue("R1.2b P0 was not requested", false)
            return
        }
        check(arguments.getString(ARG_PROTOCOL_SHA) == PROTOCOL_SHA)
        val input = JSONObject(privateFile(requireNotNull(arguments.getString(ARG_R11_INPUT)), "R1.1 input")
            .readText(Charsets.UTF_8))
        check(input.getString("diagnostic_set_role") == "seen_diagnostic_not_held_out")
        val videos = input.getJSONArray("sources").associateByString("event_id")
        val manifest = JSONObject(testContext.assets.open(MANIFEST_ASSET).bufferedReader().use { it.readText() })
        check(!manifest.getJSONObject("authority").getBoolean("r12_sources_read"))
        val detector = TfliteYoloDetector(testContext, MODEL_ASSET, LABELS_ASSET, 640, 0.05f, 0.45f)
        check(detector.isReady)
        val rows = JSONArray()
        var parityCount = 0
        try {
            val frames = manifest.getJSONArray("frames")
            for (index in 0 until frames.length()) {
                val frame = frames.getJSONObject(index)
                val staticBitmap = testContext.assets.open(frame.getString("image_asset")).use { BitmapFactory.decodeStream(it) }
                checkNotNull(staticBitmap)
                val videoRow = checkNotNull(videos[frame.getString("event_id")])
                val video = privateFile(videoRow.getString("video_path"), "R1.1 source video")
                val retriever = MediaMetadataRetriever().apply { setDataSource(video.absolutePath) }
                val videoBitmap = try {
                    retriever.getFrameAtTime(frame.getLong("timestamp_ms") * 1_000, MediaMetadataRetriever.OPTION_CLOSEST)
                } finally { retriever.release() }
                checkNotNull(videoBitmap)
                try {
                    val staticResult = targetStatus(detector, staticBitmap, frame)
                    val videoResult = targetStatus(detector, videoBitmap, frame)
                    val dimensionParity = staticBitmap.width == videoBitmap.width && staticBitmap.height == videoBitmap.height
                    val orientationParity = (staticBitmap.width > staticBitmap.height) == (videoBitmap.width > videoBitmap.height)
                    val statusParity = staticResult.first == videoResult.first
                    val labelParity = staticResult.second == videoResult.second
                    val passed = dimensionParity && orientationParity && statusParity && labelParity
                    if (passed) parityCount++
                    rows.put(JSONObject().put("event_id", frame.getString("event_id"))
                        .put("frame_id", frame.getString("frame_id")).put("timestamp_ms", frame.getLong("timestamp_ms"))
                        .put("static_size", JSONArray(listOf(staticBitmap.width, staticBitmap.height)))
                        .put("video_size", JSONArray(listOf(videoBitmap.width, videoBitmap.height)))
                        .put("static_target_status", staticResult.first).put("video_target_status", videoResult.first)
                        .put("static_matched_label", staticResult.second).put("video_matched_label", videoResult.second)
                        .put("dimension_parity", dimensionParity).put("orientation_parity", orientationParity)
                        .put("target_status_parity", statusParity).put("matched_label_parity", labelParity)
                        .put("frame_equivalence_passed", passed))
                } finally { staticBitmap.recycle(); videoBitmap.recycle() }
            }
        } finally { detector.close() }
        val passed = parityCount == manifest.getJSONArray("frames").length()
        val output = JSONObject().put("schema", OUTPUT_SCHEMA).put("protocol_sha256", PROTOCOL_SHA)
            .put("dataset_role", "seen_r11_anchor_equivalence_not_r12_or_r13")
            .put("device", JSONObject().put("model", Build.MODEL).put("sdk", Build.VERSION.SDK_INT))
            .put("frame_count", manifest.getJSONArray("frames").length()).put("parity_frame_count", parityCount)
            .put("frames", rows).put("p0_anchor_equivalence_passed", passed)
            .put("failure_action", if (passed) JSONObject.NULL else "use_hash_bound_exact_timestamp_frames_or_fix_decoder_semantics")
            .put("authority", JSONObject().put("r12_sources_read", false).put("r13_sources_read", false)
                .put("detector_or_annotation_changed", false).put("production_authorized", false))
        privateOutputFile(requireNotNull(arguments.getString(ARG_OUTPUT))).writeText(output.toString(2), Charsets.UTF_8)
        Log.i(TAG, "R12B_P0_ANCHOR_EQUIVALENCE pass=$passed parity=$parityCount")
        check(passed) { "R1.2b P0 anchor equivalence failed; continuous detector attribution is blocked" }
    }

    private fun targetStatus(detector: TfliteYoloDetector, bitmap: android.graphics.Bitmap,
                             frame: JSONObject): Pair<String, String?> {
        val detections = detector.detect(bitmap).detections
        val candidates = detections.mapIndexed { index, detection ->
            TargetMatchCandidate("d$index", detection.label, detection.boundingBox)
        }
        val result = UstrfTargetInstanceMatcher.match(
            frame.getJSONArray("target_bbox_xyxy_norm").toPixelBox(bitmap.width, bitmap.height), detector.labels,
            frame.getJSONArray("detector_label_allowlist").toStringList(), candidates,
        )
        val label = result.matchedDetectionId?.let { id -> candidates.single { it.detectionId == id }.label }
        return result.status to label
    }
    private fun JSONArray.associateByString(key: String) = (0 until length()).associate {
        getJSONObject(it).getString(key) to getJSONObject(it)
    }
    private fun JSONArray.toStringList() = (0 until length()).map(::getString)
    private fun JSONArray.toPixelBox(width: Int, height: Int) = BoundingBox(
        (getDouble(0) * width).toFloat(), (getDouble(1) * height).toFloat(),
        (getDouble(2) * width).toFloat(), (getDouble(3) * height).toFloat(),
    )
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

    private companion object {
        const val PROTOCOL_SHA = "c56f49d19c8afd469e86c4ad970f5c994f827a260971907f746cd5c296b33757"
        const val MANIFEST_ASSET = "ustrf_r12_detector/canary_manifest.json"
        const val MODEL_ASSET = "ustrf_r12_detector/yoloe11s_marker_static3_fp16_640.tflite"
        const val LABELS_ASSET = "ustrf_r12_detector/marker_labels.txt"
        const val OUTPUT_SCHEMA = "blindassist_ustrf_crosscam_anchor_equivalence_output_v1"
        const val ARG_REQUIRED = "ustrfR12bAnchorEquivalenceRequired"
        const val ARG_PROTOCOL_SHA = "ustrfR12bProtocolSha"
        const val ARG_R11_INPUT = "ustrfR12bR11Input"
        const val ARG_OUTPUT = "ustrfR12bOutput"
        const val TAG = "UstrfR12bP0"
    }
}
