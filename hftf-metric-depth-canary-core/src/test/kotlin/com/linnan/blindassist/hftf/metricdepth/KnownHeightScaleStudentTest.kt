package com.linnan.blindassist.hftf.metricdepth

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class KnownHeightScaleStudentTest {
    private val student = KnownHeightScaleStudent.frozen()

    @Test
    fun frozenGoldenVectorsMatchFloat64Reference() {
        goldenVectors.forEach { golden ->
            val prediction = student.predict(golden.features)
            assertTrue(golden.id, prediction is KnownHeightScaleStudent.Prediction.Valid)
            val result = prediction as KnownHeightScaleStudent.Prediction.Valid
            assertEquals(golden.id, golden.logScale, result.logScale, 1e-12)
            assertEquals(golden.id, golden.scale, result.scale, 1e-12)
        }
    }

    @Test
    fun invalidFeaturesAndOutOfRangeScaleFailClosed() {
        val invalid = student.predict(DoubleArray(10) { Double.NaN })
        assertTrue(invalid is KnownHeightScaleStudent.Prediction.Unknown)
        assertEquals("INVALID_RUNTIME_FEATURES", (invalid as KnownHeightScaleStudent.Prediction.Unknown).reason)
        val extreme = goldenVectors.first().features.copyOf()
        extreme[0] += 100.0 * 0.6066063721896592
        val outOfRange = student.predict(extreme)
        assertTrue(outOfRange is KnownHeightScaleStudent.Prediction.Unknown)
        assertEquals(
            "STUDENT_SCALE_OUT_OF_RANGE",
            (outOfRange as KnownHeightScaleStudent.Prediction.Unknown).reason,
        )
    }

    @Test
    fun featureOrderAndIdentityAreFrozen() {
        assertEquals(10, KnownHeightScaleStudent.FEATURE_NAMES.size)
        assertEquals("log_r0_known_height_scale", KnownHeightScaleStudent.FEATURE_NAMES.first())
        assertEquals("log_da_depth_q90_over_q10", KnownHeightScaleStudent.FEATURE_NAMES.last())
        assertTrue(KnownHeightScaleStudent.EXTERNAL_CONFIRMATION_RESULT_SHA256.length == 64)
    }

    private data class Golden(
        val id: String,
        val features: DoubleArray,
        val logScale: Double,
        val scale: Double,
    )

    private val goldenVectors = listOf(
        Golden(
            "training_feature_mean",
            doubleArrayOf(0.054827538806422646, 0.403815962251523, 0.06463503918506175, -0.920194298115153, -0.2690751919611397, 0.013142168972656092, 0.239020763271592, 1.355777762706651, 2.2660235192064153, 2.0270027559348223),
            -0.5081528521151187,
            0.601605808013639,
        ),
        Golden(
            "training_feature_mean_plus_one_std",
            doubleArrayOf(0.6614339109960818, 0.5927967825548801, 0.17260446233201596, -0.8136351253722058, -0.03744822274849721, 0.016647115067781416, 0.750796989537317, 1.889579549554052, 2.8791293457271085, 2.540046878788808),
            -0.5351612167887914,
            0.5855748780331561,
        ),
        Golden(
            "training_feature_mean_minus_one_std",
            doubleArrayOf(-0.5517788333832365, 0.21483514194816586, -0.04333438396189246, -1.0267534708581, -0.5007021611737822, 0.009637222877530768, -0.27275546299413295, 0.8219759758592501, 1.652917692685722, 1.5139586330808363),
            -0.48114448744144617,
            0.6180756070878617,
        ),
    )
}
