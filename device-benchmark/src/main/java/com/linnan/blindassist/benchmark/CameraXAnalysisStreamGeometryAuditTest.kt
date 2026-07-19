package com.linnan.blindassist.benchmark

import android.Manifest
import android.app.Instrumentation
import android.os.Bundle
import android.os.SystemClock
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
class CameraXAnalysisStreamGeometryAuditTest {
    @get:Rule
    val cameraPermission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun auditProductionMatchedAnalysisStreamGeometryAndClock() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val provider = ProcessCameraProvider.getInstance(instrumentation.targetContext)
            .get(PROVIDER_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        val owner = TestLifecycleOwner()
        val executor = Executors.newSingleThreadExecutor()
        val latch = CountDownLatch(REQUIRED_FRAMES)
        val rows = mutableListOf<FrameRow>()
        val lock = Any()
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(productionResolutionSelector())
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(executor) { image ->
            try {
                synchronized(lock) {
                    if (rows.size < REQUIRED_FRAMES) rows += FrameRow.from(image, SystemClock.elapsedRealtimeNanos())
                }
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
            assertTrue(
                "CameraX did not produce $REQUIRED_FRAMES frames within ${CAPTURE_TIMEOUT_SECONDS}s",
                latch.await(CAPTURE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            )
        } finally {
            instrumentation.runOnMainSync {
                provider.unbindAll()
                owner.destroy()
            }
            analysis.clearAnalyzer()
            executor.shutdown()
            executor.awaitTermination(2, TimeUnit.SECONDS)
        }

        val frozenRows = synchronized(lock) { rows.toList() }
        val timestamps = frozenRows.map { it.timestampNs }
        val intervals = timestamps.zipWithNext { a, b -> b - a }
        val report = JSONObject()
            .put("schema", "blindassist_camerax_analysis_stream_geometry_audit_v1")
            .put("camera_id", cameraId)
            .put("requested_resolution", JSONArray(listOf(ANALYSIS_WIDTH, ANALYSIS_HEIGHT)))
            .put("requested_aspect_ratio", "4:3")
            .put("output_format", "RGBA_8888")
            .put("backpressure", "KEEP_ONLY_LATEST")
            .put("frame_count", frozenRows.size)
            .put("unique_frame_geometries", JSONArray(frozenRows.distinctBy { it.geometryKey() }.map { it.toJson() }))
            .put("timestamp_monotonic", timestamps.zipWithNext().all { (a, b) -> b > a })
            .put("median_timestamp_interval_ms", intervals.medianNanos() / 1_000_000.0)
            .put("callback_minus_capture_timestamp_ms", JSONObject()
                .put("minimum", frozenRows.minOf { it.callbackElapsedRealtimeNs - it.timestampNs } / 1_000_000.0)
                .put("median", frozenRows.map { it.callbackElapsedRealtimeNs - it.timestampNs }.medianNanos() / 1_000_000.0)
                .put("maximum", frozenRows.maxOf { it.callbackElapsedRealtimeNs - it.timestampNs } / 1_000_000.0))
            .put("authorization", JSONObject()
                .put("benchmark_only", true)
                .put("intrinsic_stream_mapping_validated", false)
                .put("real_reprojection_validated", false)
                .put("app_runtime_authorized", false)
                .put("production_authorized", false))
        instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })

        assertTrue("no CameraX frames captured", frozenRows.size == REQUIRED_FRAMES)
        assertTrue("CameraX timestamps are not strictly monotonic", timestamps.zipWithNext().all { (a, b) -> b > a })
    }

    private data class FrameRow(
        val width: Int,
        val height: Int,
        val cropLeft: Int,
        val cropTop: Int,
        val cropRight: Int,
        val cropBottom: Int,
        val rotationDegrees: Int,
        val rowStride: Int,
        val pixelStride: Int,
        val timestampNs: Long,
        val callbackElapsedRealtimeNs: Long
    ) {
        fun geometryKey() = listOf(width, height, cropLeft, cropTop, cropRight, cropBottom, rotationDegrees, rowStride, pixelStride)

        fun toJson() = JSONObject()
            .put("buffer_size", JSONArray(listOf(width, height)))
            .put("crop_rect", JSONArray(listOf(cropLeft, cropTop, cropRight, cropBottom)))
            .put("crop_is_full_buffer", cropLeft == 0 && cropTop == 0 && cropRight == width && cropBottom == height)
            .put("rotation_degrees", rotationDegrees)
            .put("display_size_after_rotation", JSONArray(if (rotationDegrees % 180 == 0) listOf(width, height) else listOf(height, width)))
            .put("row_stride", rowStride)
            .put("pixel_stride", pixelStride)

        companion object {
            fun from(image: ImageProxy, callbackNs: Long): FrameRow {
                val crop = image.cropRect
                val plane = image.planes.first()
                return FrameRow(
                    image.width, image.height, crop.left, crop.top, crop.right, crop.bottom,
                    image.imageInfo.rotationDegrees, plane.rowStride, plane.pixelStride,
                    image.imageInfo.timestamp, callbackNs
                )
            }
        }
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
            ResolutionStrategy(
                Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
            )
        )
        .build()

    private fun List<Long>.medianNanos(): Double {
        if (isEmpty()) return Double.NaN
        val sorted = sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) (sorted[middle - 1] + sorted[middle]) / 2.0 else sorted[middle].toDouble()
    }

    private companion object {
        const val ANALYSIS_WIDTH = 640
        const val ANALYSIS_HEIGHT = 480
        const val REQUIRED_FRAMES = 30
        const val PROVIDER_TIMEOUT_SECONDS = 10L
        const val CAPTURE_TIMEOUT_SECONDS = 15L
        const val REPORT_KEY = "r832_report"
    }
}
