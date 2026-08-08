package com.linnan.blindassist.benchmark

import android.graphics.BitmapFactory
import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.model.BoundingBox
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.InputStream
import java.security.MessageDigest

/** Admits hash-bound host frames only when all nine frozen canary anchors reproduce on device. */
@RunWith(AndroidJUnit4::class)
class UstrfR12bExactFrameAdmissionDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun proveExactFrameTargetStatusParity() {
        if (arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() != true) {
            assumeTrue("R1.2b exact-frame admission was not requested", false)
            return
        }
        val inputFile = privateFile(requireNotNull(arguments.getString(ARG_INPUT)), "R1.2b exact-frame input")
        val input = JSONObject(inputFile.readText(Charsets.UTF_8))
        check(input.getString("schema") == INPUT_SCHEMA)
        check(input.getString("transport_contract_sha256") == TRANSPORT_SHA)
        check(input.getString("dataset_role") == SEEN_ROLE)
        val authority = input.getJSONObject("authority")
        check(!authority.getBoolean("new_held_out_read") && !authority.getBoolean("r13_sources_read"))
        check(authority.getBoolean("benchmark_only") && !authority.getBoolean("training_authorized"))
        val candidate = input.getJSONObject("candidate")
        check(candidate.getString("candidate_id") == CANDIDATE_ID)
        check(candidate.getInt("input_size") == 640 && candidate.getString("execution_backend") == "gpu_delegate")
        check(candidate.getDouble("confidence_threshold") == 0.05)
        check(candidate.getDouble("target_anchor_iou_threshold") == 0.30)
        check(candidate.getDouble("nms_iou_threshold") == 0.45)
        check(sha256Asset(MODEL_ASSET) == candidate.getString("model_sha256"))
        val sourceByEvent = input.getJSONArray("sources").associateByString("event_id")
        val manifest = JSONObject(readAssetText(MANIFEST_ASSET))
        check(manifest.getJSONArray("frames").length() == REQUIRED_FRAME_COUNT)
        val detector = UstrfBenchmarkYoloDetector(
            testContext, MODEL_ASSET, LABELS_ASSET, 640, 0.05f, 0.45f,
            UstrfBenchmarkYoloDetector.Backend.GPU_DELEGATE,
        )
        val rows = JSONArray()
        var parityCount = 0
        var failures = 0
        try {
            check(detector.actualBackend == UstrfBenchmarkYoloDetector.Backend.GPU_DELEGATE)
            val frames = manifest.getJSONArray("frames")
            for (index in 0 until frames.length()) {
                val anchor = frames.getJSONObject(index)
                val eventId = anchor.getString("event_id")
                val timestampMs = anchor.getLong("timestamp_ms")
                val exactRow = findFrame(checkNotNull(sourceByEvent[eventId]).getJSONArray("frames"), timestampMs)
                val exactFile = privateFile(exactRow.getString("image_path"), "$eventId/$timestampMs exact frame")
                check(sha256File(exactFile) == exactRow.getString("image_sha256"))
                val staticBitmap = testContext.assets.open(anchor.getString("image_asset")).use(BitmapFactory::decodeStream)
                val exactBitmap = BitmapFactory.decodeFile(exactFile.absolutePath)
                if (staticBitmap == null || exactBitmap == null) {
                    failures++
                    continue
                }
                try {
                    val staticResult = targetStatus(detector, staticBitmap, anchor)
                    val exactResult = targetStatus(detector, exactBitmap, anchor)
                    val dimensionParity = staticBitmap.width == exactBitmap.width && staticBitmap.height == exactBitmap.height
                    val orientationParity = (staticBitmap.width > staticBitmap.height) == (exactBitmap.width > exactBitmap.height)
                    val statusParity = staticResult.first == exactResult.first
                    val labelParity = staticResult.second == exactResult.second
                    val passed = dimensionParity && orientationParity && statusParity && labelParity
                    if (passed) parityCount++
                    rows.put(JSONObject().put("event_id", eventId).put("frame_id", anchor.getString("frame_id"))
                        .put("timestamp_ms", timestampMs)
                        .put("static_size", JSONArray(listOf(staticBitmap.width, staticBitmap.height)))
                        .put("exact_size", JSONArray(listOf(exactBitmap.width, exactBitmap.height)))
                        .put("static_target_status", staticResult.first).put("exact_target_status", exactResult.first)
                        .put("static_matched_label", staticResult.second).put("exact_matched_label", exactResult.second)
                        .put("dimension_parity", dimensionParity).put("orientation_parity", orientationParity)
                        .put("target_status_parity", statusParity).put("matched_label_parity", labelParity)
                        .put("frame_equivalence_passed", passed))
                } catch (_: Throwable) {
                    failures++
                } finally {
                    staticBitmap.recycle(); exactBitmap.recycle()
                }
            }
        } finally {
            detector.close()
        }
        val passed = failures == 0 && parityCount == REQUIRED_FRAME_COUNT
        val output = JSONObject().put("schema", OUTPUT_SCHEMA).put("input_sha256", sha256File(inputFile))
            .put("transport_contract_sha256", TRANSPORT_SHA).put("candidate_id", CANDIDATE_ID)
            .put("dataset_role", "seen_r11_exact_frame_admission_not_r12_or_r13")
            .put("device", JSONObject().put("model", Build.MODEL).put("sdk", Build.VERSION.SDK_INT))
            .put("frame_count", REQUIRED_FRAME_COUNT).put("parity_frame_count", parityCount)
            .put("failure_count", failures).put("frames", rows).put("exact_frame_admission_passed", passed)
            .put("authority", JSONObject().put("r12_sources_read", false).put("r13_sources_read", false)
                .put("detector_or_annotation_changed", false).put("benchmark_only", true)
                .put("production_authorized", false))
        privateOutputFile(requireNotNull(arguments.getString(ARG_OUTPUT))).writeText(output.toString(2), Charsets.UTF_8)
        Log.i(TAG, "R12B_EXACT_FRAME_ADMISSION pass=$passed parity=$parityCount")
        check(passed) { "R1.2b exact-frame admission failed; continuous replay remains blocked" }
    }

    private fun targetStatus(detector: UstrfBenchmarkYoloDetector, bitmap: android.graphics.Bitmap,
                             frame: JSONObject): Pair<String, String?> {
        val candidates = detector.detect(bitmap).mapIndexed { index, detection ->
            TargetMatchCandidate("d$index", detection.label, detection.boundingBox)
        }
        val result = UstrfTargetInstanceMatcher.match(
            frame.getJSONArray("target_bbox_xyxy_norm").toPixelBox(bitmap.width, bitmap.height), detector.labels,
            frame.getJSONArray("detector_label_allowlist").toStringList(), candidates,
        )
        val label = result.matchedDetectionId?.let { id -> candidates.single { it.detectionId == id }.label }
        return result.status to label
    }
    private fun findFrame(frames: JSONArray, timestampMs: Long): JSONObject =
        (0 until frames.length()).map(frames::getJSONObject).single { it.getLong("timestamp_ms") == timestampMs }
    private fun JSONArray.associateByString(key: String) = (0 until length()).associate {
        getJSONObject(it).getString(key) to getJSONObject(it)
    }
    private fun JSONArray.toStringList() = (0 until length()).map(::getString)
    private fun JSONArray.toPixelBox(width: Int, height: Int) = BoundingBox(
        (getDouble(0) * width).toFloat(), (getDouble(1) * height).toFloat(),
        (getDouble(2) * width).toFloat(), (getDouble(3) * height).toFloat(),
    )
    private fun readAssetText(path: String) = testContext.assets.open(path).bufferedReader().use { it.readText() }
    private fun privateFile(relative: String, label: String): File {
        check(relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        return File(root, relative).canonicalFile.also {
            check(it.path.startsWith(root.path + File.separator) && it.isFile) { "$label unavailable" }
        }
    }
    private fun privateOutputFile(relative: String): File {
        check(relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        return File(root, relative).canonicalFile.also {
            check(it.path.startsWith(root.path + File.separator)); it.parentFile?.mkdirs()
        }
    }
    private fun sha256File(file: File) = file.inputStream().use(::digest)
    private fun sha256Asset(path: String) = testContext.assets.open(path).use(::digest)
    private fun digest(input: InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256"); val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private companion object {
        const val INPUT_SCHEMA = "blindassist_ustrf_crosscam_exact_frame_android_input_v1"
        const val OUTPUT_SCHEMA = "blindassist_ustrf_crosscam_exact_frame_admission_output_v1"
        const val TRANSPORT_SHA = "86945f31f98e576a042af090d58330b99270b7adc5f3ab3ce4632fa45aa03009"
        const val CANDIDATE_ID = "r12b_c1_same640_gpu_delegate"
        const val SEEN_ROLE = "seen_diagnostic_not_held_out"
        const val MANIFEST_ASSET = "ustrf_r12_detector/canary_manifest.json"
        const val MODEL_ASSET = "ustrf_r12_detector/yoloe11s_marker_static3_fp16_640.tflite"
        const val LABELS_ASSET = "ustrf_r12_detector/marker_labels.txt"
        const val REQUIRED_FRAME_COUNT = 9
        const val ARG_REQUIRED = "ustrfR12bExactFrameRequired"
        const val ARG_INPUT = "ustrfR12bExactFrameInput"
        const val ARG_OUTPUT = "ustrfR12bOutput"
        const val TAG = "UstrfR12bExactFrame"
    }
}
