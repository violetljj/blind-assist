package com.linnan.blindassist.benchmark

import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot

@RunWith(AndroidJUnit4::class)
class GravityReprojectionAdaptiveRescreenTest {
    @Test
    fun rescreenFrozenR836LowLightFrameWithoutRecapture() {
        assertTrue("OpenCV local runtime failed to load", OpenCVLoader.initLocal())
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val inputDir = File(requireNotNull(instrumentation.targetContext.getExternalFilesDir(null)), "r836-gravity-reprojection")
        val imageFile = File(inputDir, "r836_display_frame.png")
        val metadataFile = File(inputDir, "r836_gravity_reprojection.json")
        assertTrue("r836 image missing", imageFile.isFile)
        assertTrue("r836 metadata missing", metadataFile.isFile)
        val metadata = JSONObject(metadataFile.readText(Charsets.UTF_8))
        val vpJson = metadata.getJSONArray("predicted_vertical_vanishing_point")
        val vp = doubleArrayOf(vpJson.getDouble(0), vpJson.getDouble(1))
        val bitmap = requireNotNull(BitmapFactory.decodeFile(imageFile.absolutePath))

        val rgba = Mat()
        val gray = Mat()
        val enhanced = Mat()
        val blurred = Mat()
        val edges = Mat()
        val lines = Mat()
        try {
            Utils.bitmapToMat(bitmap, rgba)
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.createCLAHE(3.0, Size(8.0, 8.0)).apply(gray, enhanced)
            Imgproc.GaussianBlur(enhanced, blurred, Size(3.0, 3.0), 0.8)
            Imgproc.Canny(blurred, edges, 15.0, 50.0)
            Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 15, 20.0, 20.0)
            val metrics = ArrayList<LineMetric>()
            for (row in 0 until lines.rows()) {
                val line = lines.get(row, 0) ?: continue
                if (line.size < 4) continue
                val dx = line[2] - line[0]
                val dy = line[3] - line[1]
                val length = hypot(dx, dy)
                if (length < 20.0) continue
                val midpointX = (line[0] + line[2]) / 2.0
                val midpointY = (line[1] + line[3]) / 2.0
                val expectedX = vp[0] - midpointX
                val expectedY = vp[1] - midpointY
                val denominator = length * hypot(expectedX, expectedY)
                if (denominator <= 1e-9) continue
                val cosine = abs((dx * expectedX + dy * expectedY) / denominator).coerceIn(0.0, 1.0)
                metrics += LineMetric(length, Math.toDegrees(acos(cosine)))
            }
            val totalLength = metrics.sumOf { it.lengthPx }
            val aligned = metrics.filter { it.angleErrorDegrees <= ALIGNMENT_THRESHOLD_DEGREES }
            val alignedLength = aligned.sumOf { it.lengthPx }
            val support = if (totalLength > 0.0) alignedLength / totalLength else 0.0
            val informative = metrics.size >= 5 && aligned.size >= 2 && alignedLength >= 80.0
            val pass = informative && support >= MINIMUM_ALIGNED_LENGTH_FRACTION
            val report = JSONObject()
                .put("schema", "blindassist_gravity_reprojection_adaptive_rescreen_v1")
                .put("source_frame", imageFile.absolutePath)
                .put("source_frame_timestamp_ns", metadata.getLong("frame_timestamp_ns"))
                .put("recapture_performed", false)
                .put("preprocessing", "CLAHE clip=3 tile=8x8; Gaussian 3x3 sigma=.8; Canny 15/50")
                .put("hough", "threshold=15 minLength=20 maxGap=20")
                .put("candidate_line_count", metrics.size)
                .put("aligned_line_count", aligned.size)
                .put("total_line_length_px", totalLength)
                .put("aligned_line_length_px", alignedLength)
                .put("aligned_line_length_fraction", support)
                .put("alignment_threshold_degrees", ALIGNMENT_THRESHOLD_DEGREES)
                .put("ten_smallest_angle_errors_degrees", JSONArray(metrics.map { it.angleErrorDegrees }.sorted().take(10)))
                .put("informative", informative)
                .put("pass", pass)
                .put("authorization", JSONObject()
                    .put("benchmark_only", true)
                    .put("gravity_axis_only", true)
                    .put("full_3d_reprojection_validated", false)
                    .put("app_runtime_authorized", false)
                    .put("production_authorized", false))
            val outputFile = File(inputDir, "r836a_adaptive_rescreen.json")
            outputFile.writeText(report.toString(2), Charsets.UTF_8)
            instrumentation.sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
            assertTrue("r836a report missing", outputFile.isFile && outputFile.length() > 0)
        } finally {
            rgba.release(); gray.release(); enhanced.release(); blurred.release(); edges.release(); lines.release(); bitmap.recycle()
        }
    }

    private data class LineMetric(val lengthPx: Double, val angleErrorDegrees: Double)

    private companion object {
        const val ALIGNMENT_THRESHOLD_DEGREES = 10.0
        const val MINIMUM_ALIGNED_LENGTH_FRACTION = 0.10
        const val REPORT_KEY = "r836a_report"
    }
}
