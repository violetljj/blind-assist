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
import kotlin.math.ceil

/** R1.2b/R1.2c fixed nine-frame seen canary. It can select only a full continuous replay candidate. */
@RunWith(AndroidJUnit4::class)
class UstrfR12bMobileCanaryDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val targetContext = instrumentation.targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun profileFrozenCandidateWithoutR12OrR13Sources() {
        if (arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() != true) {
            assumeTrue("R1.2b canary was not requested", false)
            return
        }
        val protocolSha = arguments.getString(ARG_PROTOCOL_SHA)
        check(protocolSha == R12B_PROTOCOL_SHA || protocolSha == R12C_V2_PROTOCOL_SHA) {
            "R1.2b/R1.2c protocol hash mismatch"
        }
        val candidateId = requireNotNull(arguments.getString(ARG_CANDIDATE_ID))
        val modelAsset = requireNotNull(arguments.getString(ARG_MODEL_ASSET))
        check(modelAsset.startsWith("ustrf_r12_detector/") || modelAsset.startsWith("ustrf_r12b_detector/"))
        val expectedModelSha = requireNotNull(arguments.getString(ARG_MODEL_SHA))
        val inputSize = requireNotNull(arguments.getString(ARG_INPUT_SIZE)).toInt()
        val requestedBackend = when (requireNotNull(arguments.getString(ARG_BACKEND))) {
            "cpu" -> UstrfBenchmarkYoloDetector.Backend.CPU
            "gpu_delegate" -> UstrfBenchmarkYoloDetector.Backend.GPU_DELEGATE
            else -> error("unsupported backend")
        }
        check(sha256Asset(modelAsset) == expectedModelSha)
        val manifest = JSONObject(readAssetText(MANIFEST_ASSET))
        check(manifest.getString("schema") == MANIFEST_SCHEMA)
        check(manifest.getString("dataset_role") == "seen_r11_diagnostic_parser_canary_not_r12_held_out")
        check(!manifest.getJSONObject("authority").getBoolean("r12_sources_read"))
        check(manifest.getJSONArray("frames").length() == REQUIRED_FRAME_COUNT)

        val detector = UstrfBenchmarkYoloDetector(
            testContext, modelAsset, LABELS_ASSET, inputSize, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD,
            requestedBackend,
        )
        check(detector.isReady) { "candidate init failed: ${detector.statusMessage}" }
        val preprocess = mutableListOf<Double>()
        val inference = mutableListOf<Double>()
        val postprocess = mutableListOf<Double>()
        val total = mutableListOf<Double>()
        var failures = 0
        val emittedLabels = linkedSetOf<String>()
        val perFrameStatuses = Array(REQUIRED_FRAME_COUNT) { mutableListOf<String>() }
        try {
            val first = manifest.getJSONArray("frames").getJSONObject(0)
            repeat(WARMUP_ITERATIONS) {
                val bitmap = testContext.assets.open(first.getString("image_asset")).use { BitmapFactory.decodeStream(it) }
                checkNotNull(bitmap).useRecycled { detector.detect(it) }
            }
            repeat(MEASUREMENT_REPETITIONS) {
                val frames = manifest.getJSONArray("frames")
                for (index in 0 until frames.length()) {
                    val frame = frames.getJSONObject(index)
                    val bitmap = testContext.assets.open(frame.getString("image_asset")).use { BitmapFactory.decodeStream(it) }
                    if (bitmap == null) {
                        failures++
                        perFrameStatuses[index] += "decode_failure"
                        continue
                    }
                    bitmap.useRecycled { decoded ->
                        try {
                            val detections = detector.detect(decoded)
                            preprocess += detector.lastPreprocessMs.toDouble()
                            inference += detector.lastInferenceMs.toDouble()
                            postprocess += detector.lastPostprocessMs.toDouble()
                            total += detector.lastTotalDetectMs.toDouble()
                            emittedLabels += detections.map { it.label }
                            val candidates = detections.mapIndexed { detectionIndex, detection ->
                                TargetMatchCandidate("d$detectionIndex", detection.label, detection.boundingBox)
                            }
                            val result = UstrfTargetInstanceMatcher.match(
                                frame.getJSONArray("target_bbox_xyxy_norm").toPixelBox(decoded.width, decoded.height),
                                detector.labels, frame.getJSONArray("detector_label_allowlist").toStringList(), candidates,
                            )
                            perFrameStatuses[index] += result.status
                        } catch (_: Throwable) {
                            failures++
                            perFrameStatuses[index] += "inference_failure"
                        }
                    }
                }
            }
        } finally { detector.close() }
        val stableMatchedFrames = perFrameStatuses.count { statuses ->
            statuses.size == MEASUREMENT_REPETITIONS && statuses.all { it == "matched" }
        }
        val actualBackend = detector.actualBackend
        val backendPassed = actualBackend == requestedBackend
        val canaryPassed = failures == 0 && backendPassed && stableMatchedFrames >= REQUIRED_MATCHED_FRAMES &&
            emittedLabels.size == REQUIRED_LABEL_COUNT && percentile(inference, 0.95) <= INFERENCE_P95_LIMIT_MS &&
            percentile(total, 0.95) <= TOTAL_P95_LIMIT_MS
        val output = JSONObject()
            .put("schema", OUTPUT_SCHEMA).put("protocol_sha256", protocolSha).put("candidate_id", candidateId)
            .put("dataset_role", "seen_r11_diagnostic_parser_canary_not_r12_or_r13")
            .put("device", JSONObject().put("model", Build.MODEL).put("sdk", Build.VERSION.SDK_INT)
                .put("fingerprint", Build.FINGERPRINT))
            .put("model_asset", modelAsset).put("model_sha256", expectedModelSha).put("input_size", inputSize)
            .put("requested_backend", requestedBackend.name.lowercase())
            .put("actual_backend", actualBackend.name.lowercase()).put("backend_gate_passed", backendPassed)
            .put("warmup_iterations", WARMUP_ITERATIONS).put("measurement_repetitions", MEASUREMENT_REPETITIONS)
            .put("measurement_count", inference.size).put("failure_count", failures)
            .put("stable_target_match_frame_count", stableMatchedFrames)
            .put("emitted_labels", JSONArray(emittedLabels.toList().sorted()))
            .put("timing_ms", JSONObject()
                .put("preprocess_p50", percentile(preprocess, 0.50)).put("preprocess_p95", percentile(preprocess, 0.95))
                .put("inference_p50", percentile(inference, 0.50)).put("inference_p95", percentile(inference, 0.95))
                .put("postprocess_p50", percentile(postprocess, 0.50)).put("postprocess_p95", percentile(postprocess, 0.95))
                .put("total_p50", percentile(total, 0.50)).put("total_p95", percentile(total, 0.95)))
            .put("frame_statuses", JSONArray(perFrameStatuses.map { JSONArray(it) }))
            .put("gate", JSONObject().put("canary_passed", canaryPassed)
                .put("authorizes_full_seen_continuous_replay_only", canaryPassed)
                .put("production_model_replacement_authorized", false))
            .put("authority", JSONObject().put("r12_sources_read", false).put("r13_sources_read", false)
                .put("thresholds_changed", false).put("training_performed", false).put("benchmark_only", true))
        privateOutputFile(requireNotNull(arguments.getString(ARG_OUTPUT))).writeText(output.toString(2), Charsets.UTF_8)
        Log.i(TAG, "R12B_CANARY candidate=$candidateId backend=$actualBackend pass=$canaryPassed")
        check(canaryPassed) { "R1.2b candidate canary failed; inspect output receipt" }
    }

    private inline fun <T> android.graphics.Bitmap.useRecycled(block: (android.graphics.Bitmap) -> T): T = try { block(this) } finally { recycle() }
    private fun JSONArray.toStringList() = (0 until length()).map(::getString)
    private fun JSONArray.toPixelBox(width: Int, height: Int) = BoundingBox(
        (getDouble(0) * width).toFloat(), (getDouble(1) * height).toFloat(),
        (getDouble(2) * width).toFloat(), (getDouble(3) * height).toFloat(),
    )
    private fun percentile(values: List<Double>, fraction: Double): Double {
        check(values.isNotEmpty()); val sorted = values.sorted()
        return sorted[(ceil(fraction * sorted.size).toInt() - 1).coerceIn(sorted.indices)]
    }
    private fun readAssetText(path: String) = testContext.assets.open(path).bufferedReader().use { it.readText() }
    private fun sha256Asset(path: String) = testContext.assets.open(path).use(::digest)
    private fun digest(input: InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256"); val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
    private fun privateOutputFile(relative: String): File {
        check(relative.isNotBlank() && !relative.startsWith('/') && !relative.contains(".."))
        val root = requireNotNull(targetContext.getExternalFilesDir(null)).canonicalFile
        return File(root, relative).canonicalFile.also { check(it.path.startsWith(root.path + File.separator)); it.parentFile?.mkdirs() }
    }

    private companion object {
        const val R12B_PROTOCOL_SHA = "c56f49d19c8afd469e86c4ad970f5c994f827a260971907f746cd5c296b33757"
        const val R12C_V2_PROTOCOL_SHA = "10f245b316fa86a55b5bae5d275939d668f7d34ba12a51287ad38ce961ba5e0a"
        const val MANIFEST_ASSET = "ustrf_r12_detector/canary_manifest.json"
        const val MANIFEST_SCHEMA = "blindassist_ustrf_r12_android_parser_canary_v1"
        const val LABELS_ASSET = "ustrf_r12_detector/marker_labels.txt"
        const val OUTPUT_SCHEMA = "blindassist_ustrf_crosscam_mobile_canary_output_v1"
        const val WARMUP_ITERATIONS = 3
        const val MEASUREMENT_REPETITIONS = 3
        const val REQUIRED_FRAME_COUNT = 9
        const val REQUIRED_MATCHED_FRAMES = 6
        const val REQUIRED_LABEL_COUNT = 3
        const val CONFIDENCE_THRESHOLD = 0.05f
        const val NMS_IOU_THRESHOLD = 0.45f
        const val INFERENCE_P95_LIMIT_MS = 120.0
        const val TOTAL_P95_LIMIT_MS = 160.0
        const val ARG_REQUIRED = "ustrfR12bCanaryRequired"
        const val ARG_PROTOCOL_SHA = "ustrfR12bProtocolSha"
        const val ARG_CANDIDATE_ID = "ustrfR12bCandidateId"
        const val ARG_MODEL_ASSET = "ustrfR12bModelAsset"
        const val ARG_MODEL_SHA = "ustrfR12bModelSha"
        const val ARG_INPUT_SIZE = "ustrfR12bInputSize"
        const val ARG_BACKEND = "ustrfR12bBackend"
        const val ARG_OUTPUT = "ustrfR12bOutput"
        const val TAG = "UstrfR12bCanary"
    }
}
