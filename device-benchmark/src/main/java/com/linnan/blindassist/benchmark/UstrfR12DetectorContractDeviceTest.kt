package com.linnan.blindassist.benchmark

import android.graphics.BitmapFactory
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
import java.io.InputStream
import java.security.MessageDigest

/** Benchmark-only parser canary for the frozen R1.2 three-class YOLOE export. */
@RunWith(AndroidJUnit4::class)
class UstrfR12DetectorContractDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun verifyStaticExportAndParserOnSeenR11Frames() {
        val required = arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() == true
        if (!required) {
            assumeTrue("R1.2 detector parser canary was not requested", false)
            return
        }
        val expectedModelSha = requireNotNull(arguments.getString(ARG_MODEL_SHA))
        val expectedLabelsSha = requireNotNull(arguments.getString(ARG_LABELS_SHA))
        check(sha256Asset(MODEL_ASSET) == expectedModelSha)
        check(sha256Asset(LABELS_ASSET) == expectedLabelsSha)
        val manifest = JSONObject(readAssetText(MANIFEST_ASSET))
        check(manifest.getString("schema") == MANIFEST_SCHEMA)
        check(manifest.getString("dataset_role") == "seen_r11_diagnostic_parser_canary_not_r12_held_out")
        check(!manifest.getJSONObject("authority").getBoolean("r12_sources_read"))

        val detector = TfliteYoloDetector(
            context = testContext,
            modelAssetName = MODEL_ASSET,
            labelsAssetName = LABELS_ASSET,
            inputSize = 640,
            confidenceThreshold = 0.05f,
            iouThreshold = 0.45f,
        )
        check(detector.isReady) { "R1.2 detector failed to initialize: ${detector.statusMessage}" }
        check(detector.labels == FROZEN_LABELS)
        var totalDetections = 0
        var matchedFrames = 0
        val matchedLabels = linkedSetOf<String>()
        val frameRows = JSONArray()
        try {
            val frames = manifest.getJSONArray("frames")
            for (index in 0 until frames.length()) {
                val frame = frames.getJSONObject(index)
                val asset = frame.getString("image_asset")
                check(sha256Asset(asset) == frame.getString("image_sha256"))
                val bitmap = testContext.assets.open(asset).use { BitmapFactory.decodeStream(it) }
                checkNotNull(bitmap) { "cannot decode $asset" }
                try {
                    val detections = detector.detect(bitmap).detections
                    totalDetections += detections.size
                    val targetBox = frame.getJSONArray("target_bbox_xyxy_norm")
                        .toPixelBox(bitmap.width, bitmap.height)
                    val allowlist = frame.getJSONArray("detector_label_allowlist").toStringList()
                    val candidates = detections.mapIndexed { detectionIndex, detection ->
                        TargetMatchCandidate("d$detectionIndex", detection.label, detection.boundingBox)
                    }
                    check(candidates.all { it.label in FROZEN_LABELS }) {
                        "parser emitted a label outside the frozen inventory: ${candidates.map { it.label }}"
                    }
                    val match = UstrfTargetInstanceMatcher.match(
                        targetBox, detector.labels, allowlist, candidates
                    )
                    if (match.status == "matched") {
                        matchedFrames++
                        val matched = candidates.single { it.detectionId == match.matchedDetectionId }
                        matchedLabels += matched.label
                    }
                    frameRows.put(
                        JSONObject()
                            .put("event_id", frame.getString("event_id"))
                            .put("frame_id", frame.getString("frame_id"))
                            .put("detection_count", detections.size)
                            .put("target_match_status", match.status)
                            .put("matched_detection_id", match.matchedDetectionId)
                            .put("detections", JSONArray().also { rows ->
                                detections.forEachIndexed { detectionIndex, detection ->
                                    rows.put(
                                        JSONObject()
                                            .put("detection_id", "d$detectionIndex")
                                            .put("label", detection.label)
                                            .put("confidence", detection.confidence.toDouble())
                                            .put("target_iou", UstrfTargetInstanceMatcher.iou(
                                                targetBox, detection.boundingBox
                                            ))
                                    )
                                }
                            })
                    )
                } finally {
                    bitmap.recycle()
                }
            }
        } finally {
            detector.close()
        }
        val output = JSONObject()
            .put("schema", OUTPUT_SCHEMA)
            .put("dataset_role", "seen_r11_diagnostic_parser_canary_not_r12_held_out")
            .put("model_asset_sha256", expectedModelSha)
            .put("labels_asset_sha256", expectedLabelsSha)
            .put("model_label_inventory", JSONArray(detector.labels))
            .put("frame_count", manifest.getJSONArray("frames").length())
            .put("total_detection_count", totalDetections)
            .put("target_match_frame_count", matchedFrames)
            .put("matched_labels", JSONArray(matchedLabels.toList()))
            .put("frames", frameRows)
            .put("r12_sources_read", false)
            .put("training_performed", false)
            .put("production_model_replacement_authorized", false)
        privateOutputFile(requireNotNull(arguments.getString(ARG_OUTPUT)))
            .writeText(output.toString(2), Charsets.UTF_8)
        check(totalDetections > 0) { "parser returned zero detections on all seen canary frames" }
        check(matchedFrames > 0) { "parser produced no target match on seen canary frames" }
        Log.i(TAG, "USTRF_R12_DETECTOR_CONTRACT_OK")
    }

    private fun JSONArray.toStringList() = (0 until length()).map { getString(it) }
    private fun JSONArray.toPixelBox(width: Int, height: Int) = BoundingBox(
        (getDouble(0) * width).toFloat(),
        (getDouble(1) * height).toFloat(),
        (getDouble(2) * width).toFloat(),
        (getDouble(3) * height).toFloat(),
    )
    private fun readAssetText(path: String) = testContext.assets.open(path)
        .bufferedReader().use { it.readText() }
    private fun sha256Asset(path: String) = testContext.assets.open(path).use { digest(it) }
    private fun digest(input: InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
    private fun privateOutputFile(relative: String): File {
        check(relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        return File(root, relative).canonicalFile.also {
            check(it.path.startsWith(root.path + File.separator))
            it.parentFile?.mkdirs()
        }
    }

    private companion object {
        const val MODEL_ASSET = "ustrf_r12_detector/yoloe11s_marker_static3_fp16_640.tflite"
        const val LABELS_ASSET = "ustrf_r12_detector/marker_labels.txt"
        const val MANIFEST_ASSET = "ustrf_r12_detector/canary_manifest.json"
        const val MANIFEST_SCHEMA = "blindassist_ustrf_r12_android_parser_canary_v1"
        const val OUTPUT_SCHEMA = "blindassist_ustrf_r12_android_parser_output_v1"
        const val ARG_REQUIRED = "ustrfR12DetectorContractRequired"
        const val ARG_MODEL_SHA = "ustrfR12ExpectedModelSha"
        const val ARG_LABELS_SHA = "ustrfR12ExpectedLabelsSha"
        const val ARG_OUTPUT = "ustrfR12DetectorContractOutput"
        const val TAG = "UstrfR12Detector"
        val FROZEN_LABELS = listOf("traffic cone", "delineator", "bollard")
    }
}
