package com.linnan.blindassist.ustrfbenchmark

import android.Manifest
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.util.Size
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.security.MessageDigest
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/** Physical P0/R2 capture entrypoint. It is isolated from the BlindAssist app and feedback path. */
@ExperimentalCamera2Interop
@RunWith(AndroidJUnit4::class)
class KnownHeightPhoneShadowCaptureTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun captureHashedRgbaFramesOrEmitPhysicalReceiptHold() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val args = InstrumentationRegistry.getArguments()
        val required = listOf("sessionId", "phase", "deviceSerial", "cameraHeightM", "cameraHeightUncertaintyM", "mountProfileId", "referenceManifest", "referenceManifestSha256")
        val missing = required.filter { args.getString(it).isNullOrBlank() }
        val sessionId = args.getString("sessionId") ?: "missing-physical-receipt"
        val root = requireNotNull(context.getExternalFilesDir("known-height-phone-shadow")).resolve(sessionId).apply { mkdirs() }
        if (missing.isNotEmpty()) {
            val hold = JSONObject()
                .put("schema", "blindassist_known_height_phone_capture_hold_v1")
                .put("status", "HOLD_MISSING_PHYSICAL_RECEIPTS")
                .put("missing_arguments", JSONArray(missing))
                .put("authorization", JSONObject().put("capture", false).put("app_runtime", false).put("production", false))
            root.resolve("capture_hold.json").writeText(hold.toString(2))
            instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, hold.toString()) })
            return
        }

        val height = args.getString("cameraHeightM")!!.toDoubleOrNull()
        val uncertainty = args.getString("cameraHeightUncertaintyM")!!.toDoubleOrNull()
        if (height == null || height !in 0.8..2.2 || uncertainty == null || uncertainty !in 0.0..0.02) {
            throw AssertionError("measured height must be 0.8..2.2 m and uncertainty <= 0.02 m")
        }
        val frameDirectory = root.resolve("images").apply { mkdirs() }
        val provider = ProcessCameraProvider.getInstance(context).get(10, TimeUnit.SECONDS)
        val owner = TestLifecycleOwner()
        val executor = Executors.newSingleThreadExecutor()
        val captured = CountDownLatch(REQUIRED_FRAMES)
        val nextFrame = AtomicInteger()
        val rows = arrayOfNulls<JSONObject>(REQUIRED_FRAMES)
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(executor) { image ->
            val index = nextFrame.getAndIncrement()
            try {
                if (index >= REQUIRED_FRAMES) return@setAnalyzer
                val output = frameDirectory.resolve("frame_%06d.png".format(index))
                saveRgbaPng(image.width, image.height, image.planes[0].rowStride, image.planes[0].pixelStride, image.planes[0].buffer, output)
                rows[index] = JSONObject()
                    .put("frame_id", index)
                    .put("capture_timestamp_ns", image.imageInfo.timestamp)
                    .put("rotation_degrees", image.imageInfo.rotationDegrees)
                    .put("width", image.width)
                    .put("height", image.height)
                    .put("rgb_file", "images/${output.name}")
                    .put("rgb_sha256", sha256(output))
                captured.countDown()
            } finally {
                image.close()
            }
        }

        var observedCameraId = ""
        try {
            instrumentation.runOnMainSync {
                provider.unbindAll()
                owner.resume()
                val camera = provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
                observedCameraId = Camera2CameraInfo.from(camera.cameraInfo).cameraId
            }
            assertTrue("insufficient frames for frozen P0 target", captured.await(60, TimeUnit.SECONDS))
        } finally {
            instrumentation.runOnMainSync { provider.unbindAll(); owner.destroy() }
            analysis.clearAnalyzer()
            executor.shutdown()
            executor.awaitTermination(5, TimeUnit.SECONDS)
        }
        val completeRows = rows.map { requireNotNull(it) }
        assertTrue("capture timestamps must be strictly monotonic", completeRows.zipWithNext().all { (left, right) -> right.getLong("capture_timestamp_ns") > left.getLong("capture_timestamp_ns") })
        root.resolve("frames.json").writeText(JSONArray(completeRows).toString(2))
        val intrinsicsReceipt = observedIntrinsicsReceipt(context.getSystemService(CameraManager::class.java), observedCameraId)
        root.resolve("intrinsics.json").writeText(intrinsicsReceipt.first)
        val receipt = JSONObject()
            .put("schema", "blindassist_known_height_phone_capture_receipt_v1")
            .put("protocol_id", PROTOCOL_ID)
            .put("model_id", MODEL_ID)
            .put("status", "CAPTURED_PENDING_HOST_PREFLIGHT")
            .put("session_id", sessionId)
            .put("phase", args.getString("phase"))
            .put("device_serial", args.getString("deviceSerial"))
            .put("camera_id", observedCameraId)
            .put("camera_height_m", height)
            .put("camera_height_uncertainty_m", uncertainty)
            .put("mount_profile_id", args.getString("mountProfileId"))
            .put("intrinsics_sha256", intrinsicsReceipt.second)
            .put("frame_manifest", "frames.json")
            .put("reference_manifest", args.getString("referenceManifest"))
            .put("reference_manifest_sha256", args.getString("referenceManifestSha256")!!.uppercase())
            .put("authorization", JSONObject().put("shadow_capture_only", true).put("app_runtime", false).put("production", false))
        root.resolve("receipt.json").writeText(receipt.toString(2))
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, receipt.toString()) })
    }

    private fun saveRgbaPng(width: Int, height: Int, rowStride: Int, pixelStride: Int, source: ByteBuffer, output: File) {
        require(pixelStride == 4) { "expected RGBA_8888 pixel stride" }
        val packed = ByteBuffer.allocate(width * height * 4)
        val row = ByteArray(width * 4)
        val input = source.duplicate()
        repeat(height) { y -> input.position(y * rowStride); input.get(row); packed.put(row) }
        packed.flip()
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(packed)
        FileOutputStream(output).use { check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)) }
        bitmap.recycle()
    }

    private fun observedIntrinsicsReceipt(manager: CameraManager, cameraId: String): Pair<String, String> {
        val characteristics = manager.getCameraCharacteristics(cameraId)
        val receipt = JSONObject()
            .put("camera_id", cameraId)
            .put("intrinsic_calibration", JSONArray(characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)?.toList() ?: emptyList<Float>()))
            .put("distortion", JSONArray(characteristics.get(CameraCharacteristics.LENS_DISTORTION)?.toList() ?: emptyList<Float>()))
            .put("sensor_orientation_degrees", characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION))
            .put("active_array", characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)?.flattenToString())
        val rendered = receipt.toString()
        return rendered to sha256(rendered.toByteArray())
    }

    private fun sha256(file: File): String = file.inputStream().use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1024 * 1024)
        while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
        digest.digest().joinToString("") { "%02X".format(it) }
    }
    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02X".format(it) }

    private class TestLifecycleOwner : LifecycleOwner {
        private val registry = LifecycleRegistry(this)
        override val lifecycle: Lifecycle get() = registry
        fun resume() { registry.currentState = Lifecycle.State.RESUMED }
        fun destroy() { registry.currentState = Lifecycle.State.DESTROYED }
    }

    private companion object {
        const val REQUIRED_FRAMES = 120
        const val PROTOCOL_ID = "KNOWN_HEIGHT_PHONE_SHADOW_P0_R2_20260804"
        const val MODEL_ID = "CAMERA_CONDITIONED_SCALE_STUDENT_R0_FINAL_5P"
        const val REPORT_KEY = "known_height_phone_shadow_capture_report"
    }
}
