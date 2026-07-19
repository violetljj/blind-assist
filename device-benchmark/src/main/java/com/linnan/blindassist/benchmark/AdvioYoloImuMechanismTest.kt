package com.linnan.blindassist.benchmark

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.vision.TfliteYoloDetector
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.security.MessageDigest

/**
 * Noncommercial, inference-only ADVIO mechanism probe.
 *
 * It proves only that a synchronized public RGB+gyro sequence can traverse the default detector
 * and the benchmark-only YOLO+IMU feature window. It has no event labels and must never be read
 * as an alert-quality, calibration, or production result.
 */
@RunWith(AndroidJUnit4::class)
class AdvioYoloImuMechanismTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val testContext = instrumentation.context
    private val arguments = InstrumentationRegistry.getArguments()

    @Test
    fun syncedPublicRgbAndGyroProduceFiniteCausalFeatureWindows() {
        val required = arguments.getString(ARG_REQUIRED)?.toBooleanStrictOrNull() ?: false
        val specPath = "$ASSET_PREFIX/dataset_spec.json"
        if (!assetExists(specPath)) {
            assumeTrue("ADVIO mechanism fixture is absent", !required)
            return
        }
        val spec = JSONObject(readAssetText(specPath))
        assertEquals(SPEC_SCHEMA, spec.getString("schema"))
        assertFalse("public source has no event truth", spec.getBoolean("human_event_truth_present"))
        assertFalse("fixture must not train", spec.getBoolean("training_authorized"))
        assertFalse("fixture must not calibrate", spec.getBoolean("calibration_authorized"))
        assertFalse("fixture must not replace production", spec.getBoolean("production_model_replacement_authorized"))

        val rows = readAssetText("$ASSET_PREFIX/manifest.jsonl")
            .lineSequence()
            .filter { it.isNotBlank() }
            .map(::JSONObject)
            .sortedBy { it.getInt("evidence_order") }
            .toList()
        assertEquals(spec.getInt("frame_count"), rows.size)
        val featureWindow = YoloImuCausalWindow()
        val detector = TfliteYoloDetector(testContext)
        check(detector.isReady) { "default detector failed to initialize: ${detector.statusMessage}" }
        var windows = 0
        var framesWithDetections = 0
        var framesWithSelected = 0
        var maximumSpatial = 0f
        var maximumMotion = 0f
        var maximumAbsoluteRotation = 0f
        try {
            rows.forEach { row ->
                val imagePath = "$ASSET_PREFIX/${row.getString("image_path")}"
                assertEquals(row.getString("image_sha256"), sha256Asset(imagePath))
                val bitmap = testContext.assets.open(imagePath).use(BitmapFactory::decodeStream)
                checkNotNull(bitmap) { "cannot decode $imagePath" }
                val features = try {
                    val detections = detector.detect(bitmap).detections
                    if (detections.isNotEmpty()) framesWithDetections += 1
                    val imu = row.getJSONObject("imu")
                    val rotation = imu.getDouble("rotation_radians").toFloat()
                    maximumAbsoluteRotation = maxOf(maximumAbsoluteRotation, kotlin.math.abs(rotation))
                    featureWindow.append(
                        YoloImuFeatureFrame(
                            timestampNanos = row.getLong("timestamp_nanos"),
                            detections = detections,
                            imu = YoloImuMotion(
                                translationX = imu.getDouble("translation_x").toFloat(),
                                translationY = imu.getDouble("translation_y").toFloat(),
                                logScale = imu.getDouble("log_scale").toFloat(),
                                rotationRadians = rotation,
                                observed = imu.getBoolean("observed")
                            )
                        )
                    )
                } finally {
                    bitmap.recycle()
                }
                features?.let { sequence ->
                    windows += 1
                    assertTrue("spatial feature is non-finite", sequence.spatialSequence.all { it.isFinite() })
                    assertTrue("motion feature is non-finite", sequence.motionSequence.all { it.isFinite() })
                    maximumSpatial = maxOf(maximumSpatial, sequence.spatialSequence.maxOrNull() ?: 0f)
                    maximumMotion = maxOf(maximumMotion, sequence.motionSequence.maxOf { kotlin.math.abs(it) })
                    if (sequence.motionSequence[(YoloImuCausalWindow.MOTION_VALUES_PER_FRAME * 7) + 7] > 0f) {
                        framesWithSelected += 1
                    }
                }
            }
        } finally {
            detector.close()
        }
        assertEquals(rows.size - 7, windows)
        assertTrue("feature transport must preserve nonzero gyro", maximumAbsoluteRotation > 0f)
        assertTrue("feature vectors must have bounded evidence", maximumSpatial in 0f..1f)

        val output = JSONObject()
            .put("schema", REPORT_SCHEMA)
            .put("source_id", spec.getString("source_id"))
            .put("source_license", spec.getString("source_license"))
            .put("allowed_use", spec.getString("allowed_use"))
            .put("frame_count", rows.size)
            .put("feature_windows", windows)
            .put("frames_with_yolo_detections", framesWithDetections)
            .put("windows_with_selected_detection", framesWithSelected)
            .put("maximum_abs_gyro_rotation_radians", maximumAbsoluteRotation.toDouble())
            .put("maximum_spatial_value", maximumSpatial.toDouble())
            .put("maximum_abs_motion_value", maximumMotion.toDouble())
            .put("human_event_truth_present", false)
            .put("training_authorized", false)
            .put("calibration_authorized", false)
            .put("production_model_replacement_authorized", false)
            .put("important_limit", "Feature-transport mechanism report only; no alert, safety, calibration, or model-quality conclusion.")
        // Instrumentation APKs are removed by Gradle after the run, so write only this compact,
        // non-image summary to logcat and capture it in the local experiment receipt.
        Log.i(TAG, "ADVIO_YOLO_IMU_SUMMARY ${output}")
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

    private companion object {
        const val ASSET_PREFIX = "public_video_inference"
        const val ARG_REQUIRED = "advioYoloImuRequired"
        const val SPEC_SCHEMA = "blindassist_advio_yolo_imu_mechanism_probe_v0"
        const val REPORT_SCHEMA = "blindassist_advio_yolo_imu_mechanism_report_v0"
        const val TAG = "AdvioYoloImuProbe"
    }
}
