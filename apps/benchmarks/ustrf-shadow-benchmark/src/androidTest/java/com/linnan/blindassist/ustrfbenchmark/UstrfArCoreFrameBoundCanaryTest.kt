package com.linnan.blindassist.ustrfbenchmark

import android.Manifest
import android.content.Intent
import android.media.Image
import android.os.Build
import android.os.Bundle
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.google.ar.core.Anchor
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.io.PrintWriter
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.TimeUnit

/**
 * Benchmark-only capture of evidence bound to one [Session.update] [Frame].
 *
 * This test owns the only ARCore Session in its process. It writes raw observations and does not
 * decide whether a geometry gate passed; the host validator recomputes every denominator and gate.
 * No BlindAssist App runtime code or navigation output is reachable from this benchmark package.
 */
@RunWith(AndroidJUnit4::class)
class UstrfArCoreFrameBoundCanaryTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun capturesSingleFrameBoundEvidenceWithoutUserActionOrRuntimeAuthority() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val frameAttempts = arguments.getString(ARG_FRAME_ATTEMPTS)
            ?.toIntOrNull()
            ?.coerceIn(1, MAX_FRAME_ATTEMPTS)
            ?: DEFAULT_FRAME_ATTEMPTS
        val runId = arguments.getString(ARG_RUN_ID)
            ?.takeIf { RUN_ID.matches(it) }
            ?: "autonomous-${System.currentTimeMillis()}"
        val outputRoot = File(instrumentation.targetContext.filesDir, "$OUTPUT_PARENT/$runId")
        check(outputRoot.mkdirs() || outputRoot.isDirectory) { "cannot create canary output directory" }
        val rawFile = File(outputRoot, RAW_FILE)
        val deviceReceiptFile = File(outputRoot, DEVICE_RECEIPT_FILE)
        val summaryFile = File(outputRoot, SUMMARY_FILE)
        val startedAtMs = System.currentTimeMillis()

        val activity = instrumentation.startActivitySync(
            Intent(instrumentation.targetContext, UstrfArCoreBenchmarkActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ) as UstrfArCoreBenchmarkActivity
        assertTrue("ARCore benchmark GL host did not initialize", activity.awaitGlReady())
        assertTrue("ARCore benchmark texture was not allocated", activity.cameraTextureName() != 0)

        var availability = ArCoreApk.Availability.UNKNOWN_CHECKING
        repeat(AVAILABILITY_ATTEMPTS) {
            availability = ArCoreApk.getInstance().checkAvailability(activity)
            if (!availability.isTransient) return@repeat
            Thread.sleep(AVAILABILITY_RETRY_MS)
        }

        val deviceReceipt = deviceReceipt(
            runId = runId,
            availability = availability,
            startedAtMs = startedAtMs,
            targetPackage = instrumentation.targetContext.packageName,
        )
        deviceReceiptFile.writeText(deviceReceipt.toString(2), Charsets.UTF_8)

        var captureError: String? = null
        var depthSupported = false
        var attemptedUpdates = 0
        var writtenRows = 0
        var session: Session? = null
        var anchor: Anchor? = null
        var anchorId: String? = null
        var anchorCreatedFrameIndex: Int? = null
        var anchorCreatedFrameTimestampNs: Long? = null

        try {
            PrintWriter(FileOutputStream(rawFile, false).bufferedWriter(Charsets.UTF_8)).use { rawWriter ->
                if (availability == ArCoreApk.Availability.SUPPORTED_INSTALLED) {
                    activity.runOnGlThreadAndWait(
                        timeoutSeconds = maxOf(30L, TimeUnit.MILLISECONDS.toSeconds(frameAttempts * 80L) + 30L)
                    ) {
                        session = Session(activity).also { created ->
                            val config = Config(created)
                            depthSupported = created.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
                            if (depthSupported) config.depthMode = Config.DepthMode.AUTOMATIC
                            config.updateMode = Config.UpdateMode.BLOCKING
                            created.configure(config)
                            created.setCameraTextureName(activity.cameraTextureName())
                            created.resume()
                        }
                        try {
                            repeat(frameAttempts) { frameIndex ->
                                attemptedUpdates++
                                val frame = requireNotNull(session).update()
                                if (anchor == null && frame.camera.trackingState == TrackingState.TRACKING) {
                                    anchor = requireNotNull(session).createAnchor(frame.camera.pose)
                                    anchorId = UUID.randomUUID().toString()
                                    anchorCreatedFrameIndex = frameIndex
                                    anchorCreatedFrameTimestampNs = frame.timestamp
                                }
                                rawWriter.println(
                                    observeFrame(
                                        frame = frame,
                                        frameIndex = frameIndex,
                                        runId = runId,
                                        depthSupported = depthSupported,
                                        anchor = anchor,
                                        anchorId = anchorId,
                                        anchorCreatedFrameIndex = anchorCreatedFrameIndex,
                                        anchorCreatedFrameTimestampNs = anchorCreatedFrameTimestampNs,
                                    ).toString()
                                )
                                rawWriter.flush()
                                writtenRows++
                            }
                        } finally {
                            anchor?.detach()
                            session?.pause()
                            session?.close()
                        }
                    }
                } else {
                    captureError = "ARCORE_${availability.name}"
                }
            }
        } catch (throwable: Throwable) {
            captureError = "${throwable.javaClass.name}:${throwable.message.orEmpty()}"
        } finally {
            activity.runOnUiThread { activity.finish() }
        }

        val summary = JSONObject()
            .put("schema", SUMMARY_SCHEMA)
            .put("run_id", runId)
            .put("capture_completed", captureError == null && writtenRows == frameAttempts)
            .put("capture_error", captureError ?: JSONObject.NULL)
            .put("frame_attempts_requested", frameAttempts)
            .put("session_update_attempt_count", attemptedUpdates)
            .put("raw_frame_row_count", writtenRows)
            .put("depth_mode_automatic_supported", depthSupported)
            .put("raw_frames_file", RAW_FILE)
            .put("raw_frames_sha256", sha256File(rawFile))
            .put("device_receipt_file", DEVICE_RECEIPT_FILE)
            .put("device_receipt_sha256", sha256File(deviceReceiptFile))
            .put("capture_started_at_epoch_ms", startedAtMs)
            .put("capture_completed_at_epoch_ms", System.currentTimeMillis())
            .put("evidence_boundary", evidenceBoundary())
        summaryFile.writeText(summary.toString(2), Charsets.UTF_8)

        instrumentation.sendStatus(2, Bundle().apply {
            putString(REPORT_KEY, summary.toString())
            putString("ustrf_arcore_canary_run_id", runId)
            putString("ustrf_arcore_canary_output_relative", "$OUTPUT_PARENT/$runId")
        })

        assertTrue("device receipt was not materialized", deviceReceiptFile.isFile)
        assertTrue("canary summary was not materialized", summaryFile.isFile)
    }

    private fun observeFrame(
        frame: Frame,
        frameIndex: Int,
        runId: String,
        depthSupported: Boolean,
        anchor: Anchor?,
        anchorId: String?,
        anchorCreatedFrameIndex: Int?,
        anchorCreatedFrameTimestampNs: Long?,
    ): JSONObject {
        val frameTimestampNs = frame.timestamp
        val cameraTimestampNs = frame.androidCameraTimestamp
        val camera = frame.camera
        val cameraPose = camera.pose
        val viewMatrix = FloatArray(16).also { camera.getViewMatrix(it, 0) }
        val projectionMatrix = FloatArray(16).also {
            camera.getProjectionMatrix(it, 0, PROJECTION_NEAR_M, PROJECTION_FAR_M)
        }
        val androidSensorPose = frame.androidSensorPose
        val imageIntrinsics = camera.imageIntrinsics
        val textureIntrinsics = camera.textureIntrinsics
        val cameraImage = acquireImage(frame::acquireCameraImage)
        val rawDepthImage = if (depthSupported) acquireImage(frame::acquireRawDepthImage16Bits) else {
            unavailableImage("DEPTH_MODE_AUTOMATIC_UNSUPPORTED")
        }
        val rawConfidenceImage = if (depthSupported) acquireImage(frame::acquireRawDepthConfidenceImage) else {
            unavailableImage("DEPTH_MODE_AUTOMATIC_UNSUPPORTED")
        }
        val anchorReceipt = if (anchor == null || anchorId == null) {
            JSONObject()
                .put("available", false)
                .put("reference_mode", "UNAVAILABLE")
                .put("error", "ANCHOR_NOT_CREATED_WITH_TRACKING_CAMERA")
        } else {
            val anchorPose = anchor.pose
            val anchorFromCamera = anchorPose.inverse().compose(cameraPose)
            JSONObject()
                .put("available", true)
                .put("anchor_id", anchorId)
                .put("created_frame_index", anchorCreatedFrameIndex)
                .put("created_frame_timestamp_ns", anchorCreatedFrameTimestampNs)
                .put("tracking_state", anchor.trackingState.name)
                .put("reference_mode", INTER_FRAME_REFERENCE_MODE)
                .put("world_from_anchor", poseJson(anchorPose))
                .put("anchor_from_camera", poseJson(anchorFromCamera))
        }

        return JSONObject()
            .put("schema", RAW_SCHEMA)
            .put("run_id", runId)
            .put("frame_index", frameIndex)
            .put("session_update_index", frameIndex)
            .put("frame_timestamp_ns", frameTimestampNs)
            .put("android_camera_timestamp_ns", cameraTimestampNs)
            .put("camera_texture_name", frame.cameraTextureName)
            .put("tracking_state", camera.trackingState.name)
            .put("tracking_failure_reason", camera.trackingFailureReason.name)
            .put("camera_image", cameraImage)
            .put("raw_depth_image", rawDepthImage)
            .put("raw_confidence_image", rawConfidenceImage)
            .put("source_alignment", JSONObject()
                .put("camera_image_matches_frame_timestamp", imageTimestamp(cameraImage) == frameTimestampNs)
                .put("camera_image_matches_android_camera_timestamp", imageTimestamp(cameraImage) == cameraTimestampNs)
                .put("raw_depth_matches_frame_timestamp", imageTimestamp(rawDepthImage) == frameTimestampNs)
                .put("raw_confidence_matches_frame_timestamp", imageTimestamp(rawConfidenceImage) == frameTimestampNs))
            .put("intrinsics", JSONObject()
                .put("image", intrinsicsJson(imageIntrinsics.imageDimensions, imageIntrinsics.focalLength, imageIntrinsics.principalPoint))
                .put("texture", intrinsicsJson(textureIntrinsics.imageDimensions, textureIntrinsics.focalLength, textureIntrinsics.principalPoint)))
            .put("transforms", JSONObject()
                .put("world_from_camera", poseJson(cameraPose))
                .put("camera_view_matrix", floatArrayJson(viewMatrix))
                .put("camera_projection_matrix", floatArrayJson(projectionMatrix))
                .put("world_from_android_sensor", poseJson(androidSensorPose)))
            .put("anchor", anchorReceipt)
            .put("evidence_boundary", evidenceBoundary())
    }

    private fun acquireImage(acquire: () -> Image): JSONObject {
        val image = try {
            acquire()
        } catch (_: NotYetAvailableException) {
            return unavailableImage("NOT_YET_AVAILABLE")
        } catch (error: RuntimeException) {
            return unavailableImage("${error.javaClass.name}:${error.message.orEmpty()}")
        }
        return try {
            val digest = MessageDigest.getInstance("SHA-256")
            val planes = JSONArray()
            image.planes.forEachIndexed { index, plane ->
                digest.update("plane:$index:${plane.rowStride}:${plane.pixelStride}:".toByteArray(Charsets.UTF_8))
                val buffer = plane.buffer.duplicate()
                val bytes = ByteArray(minOf(HASH_CHUNK_BYTES, maxOf(1, buffer.remaining())))
                while (buffer.hasRemaining()) {
                    val count = minOf(buffer.remaining(), bytes.size)
                    buffer.get(bytes, 0, count)
                    digest.update(bytes, 0, count)
                }
                planes.put(JSONObject()
                    .put("index", index)
                    .put("row_stride_bytes", plane.rowStride)
                    .put("pixel_stride_bytes", plane.pixelStride)
                    .put("buffer_remaining_bytes", plane.buffer.remaining()))
            }
            JSONObject()
                .put("available", true)
                .put("timestamp_ns", image.timestamp)
                .put("width_px", image.width)
                .put("height_px", image.height)
                .put("format", image.format)
                .put("plane_count", image.planes.size)
                .put("planes", planes)
                .put("content_sha256", digest.digest().joinToString("") { "%02x".format(it) })
                .put("error", JSONObject.NULL)
        } finally {
            image.close()
        }
    }

    private fun unavailableImage(error: String) = JSONObject()
        .put("available", false)
        .put("timestamp_ns", JSONObject.NULL)
        .put("content_sha256", JSONObject.NULL)
        .put("error", error)

    private fun imageTimestamp(receipt: JSONObject): Long? =
        if (receipt.optBoolean("available", false)) receipt.optLong("timestamp_ns") else null

    private fun intrinsicsJson(dimensions: IntArray, focal: FloatArray, principal: FloatArray) = JSONObject()
        .put("width_px", dimensions[0])
        .put("height_px", dimensions[1])
        .put("focal_x_px", focal[0].toDouble())
        .put("focal_y_px", focal[1].toDouble())
        .put("principal_x_px", principal[0].toDouble())
        .put("principal_y_px", principal[1].toDouble())

    private fun poseJson(pose: Pose): JSONObject {
        val matrix = FloatArray(16).also { pose.toMatrix(it, 0) }
        return JSONObject()
            .put("translation_m", JSONArray(listOf(pose.tx(), pose.ty(), pose.tz())))
            .put("rotation_quaternion_xyzw", JSONArray(listOf(pose.qx(), pose.qy(), pose.qz(), pose.qw())))
            .put("matrix_4x4_column_major", floatArrayJson(matrix))
    }

    private fun floatArrayJson(values: FloatArray) = JSONArray(values.map(Float::toDouble))

    private fun deviceReceipt(
        runId: String,
        availability: ArCoreApk.Availability,
        startedAtMs: Long,
        targetPackage: String,
    ) = JSONObject()
        .put("schema", DEVICE_SCHEMA)
        .put("run_id", runId)
        .put("capture_started_at_epoch_ms", startedAtMs)
        .put("device", JSONObject()
            .put("manufacturer", Build.MANUFACTURER)
            .put("brand", Build.BRAND)
            .put("model", Build.MODEL)
            .put("device", Build.DEVICE)
            .put("hardware", Build.HARDWARE)
            .put("build_fingerprint", Build.FINGERPRINT)
            .put("android_sdk_int", Build.VERSION.SDK_INT)
            .put("android_release", Build.VERSION.RELEASE))
        .put("arcore", JSONObject()
            .put("availability", availability.name)
            .put("sdk_dependency_version", ARCORE_SDK_VERSION))
        .put("capture_package", targetPackage)
        .put("capture_test", this::class.java.name)
        .put("session_ownership", "EXCLUSIVE_SINGLE_SESSION")
        .put("autonomous_capture", true)
        .put("user_motion_instruction", false)
        .put("evidence_boundary", evidenceBoundary())

    private fun evidenceBoundary() = JSONObject()
        .put("benchmark_only", true)
        .put("app_runtime_involved", false)
        .put("navigation_output_issued", false)
        .put("training_authority", false)
        .put("production_authorized", false)
        .put("human_truth", false)

    private fun sha256File(file: File): String {
        if (!file.isFile) return ""
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(HASH_CHUNK_BYTES)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private companion object {
        const val ARG_FRAME_ATTEMPTS = "ustrfArcoreFrameAttempts"
        const val ARG_RUN_ID = "ustrfArcoreRunId"
        const val DEFAULT_FRAME_ATTEMPTS = 150
        const val MAX_FRAME_ATTEMPTS = 1_800
        const val AVAILABILITY_ATTEMPTS = 10
        const val AVAILABILITY_RETRY_MS = 200L
        const val HASH_CHUNK_BYTES = 64 * 1024
        const val PROJECTION_NEAR_M = 0.1f
        const val PROJECTION_FAR_M = 100.0f
        const val OUTPUT_PARENT = "ustrf-arcore-frame-bound"
        const val RAW_FILE = "raw_frames.jsonl"
        const val DEVICE_RECEIPT_FILE = "device_receipt.json"
        const val SUMMARY_FILE = "summary.json"
        const val RAW_SCHEMA = "blindassist_ustrf_arcore_single_frame_observation_v1"
        const val SUMMARY_SCHEMA = "blindassist_ustrf_arcore_frame_bound_canary_summary_v1"
        const val DEVICE_SCHEMA = "blindassist_ustrf_arcore_frame_bound_device_receipt_v1"
        const val INTER_FRAME_REFERENCE_MODE = "INTER_FRAME_STABLE"
        const val ARCORE_SDK_VERSION = "1.33.0"
        const val REPORT_KEY = "ustrf_arcore_frame_bound_canary"
        val RUN_ID = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
    }
}
