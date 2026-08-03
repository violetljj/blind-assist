package com.linnan.blindassist.hftf

import android.os.Bundle
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.hftf.metricdepth.KnownHeightScaleStudent
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class KnownHeightScaleStudentDeviceCanaryTest {
    @Test
    fun frozenHeadMatchesGoldenAndReportsHeadOnlyLatency() {
        val model = KnownHeightScaleStudent.frozen()
        val vectors = listOf(
            Golden(
                doubleArrayOf(0.054827538806422646, 0.403815962251523, 0.06463503918506175, -0.920194298115153, -0.2690751919611397, 0.013142168972656092, 0.239020763271592, 1.355777762706651, 2.2660235192064153, 2.0270027559348223),
                -0.5081528521151187,
                0.601605808013639,
            ),
            Golden(
                doubleArrayOf(0.6614339109960818, 0.5927967825548801, 0.17260446233201596, -0.8136351253722058, -0.03744822274849721, 0.016647115067781416, 0.750796989537317, 1.889579549554052, 2.8791293457271085, 2.540046878788808),
                -0.5351612167887914,
                0.5855748780331561,
            ),
            Golden(
                doubleArrayOf(-0.5517788333832365, 0.21483514194816586, -0.04333438396189246, -1.0267534708581, -0.5007021611737822, 0.009637222877530768, -0.27275546299413295, 0.8219759758592501, 1.652917692685722, 1.5139586330808363),
                -0.48114448744144617,
                0.6180756070878617,
            ),
        )
        vectors.forEach { golden ->
            val prediction = model.predict(golden.features)
            assertTrue(prediction is KnownHeightScaleStudent.Prediction.Valid)
            prediction as KnownHeightScaleStudent.Prediction.Valid
            assertEquals(golden.logScale, prediction.logScale, 1e-12)
            assertEquals(golden.scale, prediction.scale, 1e-12)
        }

        repeat(2_000) { model.predict(vectors[it % vectors.size].features) }
        val iterations = 100_000
        val startedNs = SystemClock.elapsedRealtimeNanos()
        repeat(iterations) { model.predict(vectors[it % vectors.size].features) }
        val elapsedNs = SystemClock.elapsedRealtimeNanos() - startedNs
        val report = JSONObject()
            .put("schema", "blindassist_known_height_scale_student_device_canary_v1")
            .put("model_id", KnownHeightScaleStudent.MODEL_ID)
            .put("golden_vectors_passed", vectors.size)
            .put("iterations", iterations)
            .put("head_only_mean_latency_us", elapsedNs.toDouble() / iterations / 1_000.0)
            .put("full_depth_pipeline_latency_status", "NOT_EVALUABLE_NO_EQUIVALENT_ANDROID_EXPORT")
            .put("authorization", JSONObject().put("shadow_only", true).put("app_runtime", false).put("production", false))
        InstrumentationRegistry.getInstrumentation().sendStatus(2, Bundle().apply { putString(REPORT_KEY, report.toString()) })
    }

    private data class Golden(val features: DoubleArray, val logScale: Double, val scale: Double)

    private companion object {
        const val REPORT_KEY = "known_height_scale_student_device_canary_report"
    }
}
