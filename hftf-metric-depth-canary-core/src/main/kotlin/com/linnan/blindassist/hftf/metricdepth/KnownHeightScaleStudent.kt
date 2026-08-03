package com.linnan.blindassist.hftf.metricdepth

import kotlin.math.exp

/**
 * Immutable ridge head for the camera-conditioned scale student.
 *
 * This scorer accepts the ten already-extracted runtime features. It cannot fit or update weights,
 * access sensor depth/truth, mutate app decisions, or silently clamp an invalid scale.
 */
class KnownHeightScaleStudent private constructor(
    private val mean: DoubleArray,
    private val standardDeviation: DoubleArray,
    private val weights: DoubleArray,
    private val minimumScale: Double,
    private val maximumScale: Double,
) {
    fun predict(features: DoubleArray): Prediction {
        if (features.size != FEATURE_NAMES.size || features.any { !it.isFinite() }) {
            return Prediction.Unknown("INVALID_RUNTIME_FEATURES")
        }
        var logScale = weights[0]
        for (index in features.indices) {
            logScale += ((features[index] - mean[index]) / standardDeviation[index]) * weights[index + 1]
        }
        val scale = exp(logScale)
        if (!scale.isFinite() || scale !in minimumScale..maximumScale) {
            return Prediction.Unknown("STUDENT_SCALE_OUT_OF_RANGE", logScale, scale)
        }
        return Prediction.Valid(logScale, scale)
    }

    sealed interface Prediction {
        data class Valid(val logScale: Double, val scale: Double) : Prediction
        data class Unknown(
            val reason: String,
            val logScale: Double? = null,
            val scale: Double? = null,
        ) : Prediction
    }

    companion object {
        const val MODEL_ID = "CAMERA_CONDITIONED_SCALE_STUDENT_R0_FINAL_5P"
        const val EXTERNAL_CONFIRMATION_RESULT_SHA256 =
            "D2A8A1E091CB946078B1E8F6857749290088E247597A2BBC8F52E96D40BCCD43"

        val FEATURE_NAMES = listOf(
            "log_r0_known_height_scale",
            "log_known_camera_height_m",
            "r0_plane_normal_x",
            "r0_plane_normal_y",
            "r0_plane_normal_z",
            "r0_normalized_plane_residual",
            "log_da_depth_q10",
            "log_da_depth_q50",
            "log_da_depth_q90",
            "log_da_depth_q90_over_q10",
        )

        fun frozen(): KnownHeightScaleStudent = KnownHeightScaleStudent(
            mean = doubleArrayOf(
                0.054827538806422646,
                0.403815962251523,
                0.06463503918506175,
                -0.920194298115153,
                -0.2690751919611397,
                0.013142168972656092,
                0.239020763271592,
                1.355777762706651,
                2.2660235192064153,
                2.0270027559348223,
            ),
            standardDeviation = doubleArrayOf(
                0.6066063721896592,
                0.18898082030335714,
                0.10796942314695421,
                0.10655917274294714,
                0.2316269692126425,
                0.0035049460951253244,
                0.5117762262657249,
                0.5338017868474009,
                0.6131058265206931,
                0.513044122853986,
            ),
            weights = doubleArrayOf(
                -0.5081528521151187,
                0.30413064115576427,
                -0.025915112809860287,
                -0.04074743272902221,
                0.021670916889792353,
                -0.004125099976613373,
                -0.012474930743520215,
                -0.10277673078821535,
                -0.04824681366499122,
                -0.10070295481669889,
                -0.017820847190307586,
            ),
            minimumScale = 0.25,
            maximumScale = 4.0,
        )
    }
}
