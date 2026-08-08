package com.linnan.blindassist.benchmark

import android.Manifest
import android.app.Instrumentation
import android.content.Context
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.util.Size
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
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
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
class CameraXSensorTransformAuditTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun auditCameraXAuthoritativeSensorToBufferTransformAndMappedIntrinsics() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val provider = ProcessCameraProvider.getInstance(context).get(PROVIDER_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        val owner = TestLifecycleOwner()
        val executor = Executors.newSingleThreadExecutor()
        val latch = CountDownLatch(1)
        var captured: CapturedTransform? = null
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(productionResolutionSelector())
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(executor) { image ->
            try {
                if (captured == null) captured = CapturedTransform.from(image)
                latch.countDown()
            } finally {
                image.close()
            }
        }

        var cameraId = "unknown"
        try {
            instrumentation.runOnMainSync {
                provider.unbindAll()
                owner.resume()
                val camera = provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
                cameraId = Camera2CameraInfo.from(camera.cameraInfo).cameraId
            }
            assertTrue("CameraX did not produce a frame", latch.await(CAPTURE_TIMEOUT_SECONDS, TimeUnit.SECONDS))
        } finally {
            instrumentation.runOnMainSync {
                provider.unbindAll()
                owner.destroy()
            }
            analysis.clearAnalyzer()
            executor.shutdown()
            executor.awaitTermination(2, TimeUnit.SECONDS)
        }

        val frame = requireNotNull(captured)
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val characteristics = manager.getCameraCharacteristics(cameraId)
        val active = requireNotNull(characteristics[CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE])
        val intrinsics = requireNotNull(characteristics[CameraCharacteristics.LENS_INTRINSIC_CALIBRATION])
        val principalSensor = floatArrayOf(intrinsics[2], intrinsics[3])
        val principalBuffer = principalSensor.copyOf().also { frame.sensorToBuffer.mapPoints(it) }
        val activeCorners = floatArrayOf(
            active.left.toFloat(), active.top.toFloat(),
            active.right.toFloat(), active.top.toFloat(),
            active.right.toFloat(), active.bottom.toFloat(),
            active.left.toFloat(), active.bottom.toFloat()
        ).also { frame.sensorToBuffer.mapPoints(it) }
        val principalDisplay = mapBufferToDisplay(
            principalBuffer[0], principalBuffer[1], frame.width, frame.height, frame.rotationDegrees
        )
        val values = FloatArray(9).also { frame.sensorToBuffer.getValues(it) }
        val report = JSONObject()
            .put("schema", "blindassist_camerax_sensor_transform_intrinsic_audit_v1")
            .put("camera_id", cameraId)
            .put("active_array", JSONArray(listOf(active.left, active.top, active.right, active.bottom)))
            .put("sensor_intrinsics", JSONArray(intrinsics.toList()))
            .put("buffer_size", JSONArray(listOf(frame.width, frame.height)))
            .put("crop_rect", JSONArray(frame.cropRect.toList()))
            .put("rotation_degrees", frame.rotationDegrees)
            .put("sensor_to_buffer_matrix_row_major", JSONArray(values.toList()))
            .put("mapped_active_array_corners_buffer", JSONArray(activeCorners.toList()))
            .put("principal_sensor", JSONArray(principalSensor.toList()))
            .put("principal_buffer", JSONArray(principalBuffer.toList()))
            .put("display_size", JSONArray(if (frame.rotationDegrees % 180 == 0) listOf(frame.width, frame.height) else listOf(frame.height, frame.width)))
            .put("principal_display", JSONArray(principalDisplay.toList()))
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("transform_observed", true)
                .put("real_reprojection_validated", false)
                .put("app_runtime_authorized", false)
                .put("production_authorized", false))
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("sensor-to-buffer matrix is not affine", values[6] == 0f && values[7] == 0f && values[8] == 1f)
        assertTrue("mapped principal is outside buffer", principalBuffer[0] in 0f..frame.width.toFloat() && principalBuffer[1] in 0f..frame.height.toFloat())
        assertTrue("mapped principal is outside display", principalDisplay[0] in 0f..frame.displayWidth().toFloat() && principalDisplay[1] in 0f..frame.displayHeight().toFloat())
    }

    private data class CapturedTransform(
        val width: Int,
        val height: Int,
        val cropRect: IntArray,
        val rotationDegrees: Int,
        val sensorToBuffer: Matrix
    ) {
        fun displayWidth() = if (rotationDegrees % 180 == 0) width else height
        fun displayHeight() = if (rotationDegrees % 180 == 0) height else width

        companion object {
            fun from(image: ImageProxy): CapturedTransform {
                val crop = image.cropRect
                return CapturedTransform(
                    image.width,
                    image.height,
                    intArrayOf(crop.left, crop.top, crop.right, crop.bottom),
                    image.imageInfo.rotationDegrees,
                    Matrix(image.imageInfo.sensorToBufferTransformMatrix)
                )
            }
        }
    }

    private fun mapBufferToDisplay(x: Float, y: Float, width: Int, height: Int, rotation: Int): FloatArray = when ((rotation % 360 + 360) % 360) {
        90 -> floatArrayOf(height - 1f - y, x)
        180 -> floatArrayOf(width - 1f - x, height - 1f - y)
        270 -> floatArrayOf(y, width - 1f - x)
        else -> floatArrayOf(x, y)
    }

    private class TestLifecycleOwner : LifecycleOwner {
        private val registry = LifecycleRegistry(this)
        override val lifecycle: Lifecycle get() = registry
        fun resume() { registry.currentState = Lifecycle.State.RESUMED }
        fun destroy() { registry.currentState = Lifecycle.State.DESTROYED }
    }

    private fun productionResolutionSelector(): ResolutionSelector = ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
        .setResolutionStrategy(
            ResolutionStrategy(Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
        )
        .build()

    private companion object {
        const val ANALYSIS_WIDTH = 640
        const val ANALYSIS_HEIGHT = 480
        const val PROVIDER_TIMEOUT_SECONDS = 10L
        const val CAPTURE_TIMEOUT_SECONDS = 10L
        const val REPORT_KEY = "r833_report"
    }
}
