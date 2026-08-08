package com.linnan.blindassist.hftf

import com.linnan.blindassist.hftf.metricdepth.KnownHeightGroundPipeline

/** JNI port of the frozen known-height geometry arm; canonical Kotlin remains the authority. */
object Dav2NativeGeometry {
    fun evaluate(
        depth: FloatArray,
        width: Int,
        height: Int,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        cameraHeightM: Double,
    ): Any {
        val result = nativeEvaluate(depth, width, height, fx, fy, cx, cy, cameraHeightM)
        val status = result[0].toInt()
        if (status != STATUS_VALID) return KnownHeightGroundPipeline.Unknown(REASONS.getValue(status))
        require(result.size == RESULT_SIZE) { "invalid native geometry result size=${result.size}" }
        return KnownHeightGroundPipeline.Geometry(
            relativeHeight = result[1],
            normalizedMedianResidual = result[2],
            inlierFraction = result[3],
            normal = result.copyOfRange(4, 7),
            features = result.copyOfRange(7, RESULT_SIZE),
        )
    }

    fun evaluateDirect(
        depth: java.nio.ByteBuffer,
        width: Int,
        height: Int,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        cameraHeightM: Double,
    ): Any {
        require(depth.isDirect) { "depth must be a direct buffer" }
        require(depth.limit() >= width * height * 4) { "direct depth buffer is too small" }
        return decodeResult(nativeEvaluateDirect(depth, width, height, fx, fy, cx, cy, cameraHeightM))
    }

    private fun decodeResult(result: DoubleArray): Any {
        val status = result[0].toInt()
        if (status != STATUS_VALID) return KnownHeightGroundPipeline.Unknown(REASONS.getValue(status))
        require(result.size == RESULT_SIZE) { "invalid native geometry result size=${result.size}" }
        return KnownHeightGroundPipeline.Geometry(
            relativeHeight = result[1],
            normalizedMedianResidual = result[2],
            inlierFraction = result[3],
            normal = result.copyOfRange(4, 7),
            features = result.copyOfRange(7, RESULT_SIZE),
        )
    }

    private external fun nativeEvaluate(
        depth: FloatArray,
        width: Int,
        height: Int,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        cameraHeightM: Double,
    ): DoubleArray

    private external fun nativeEvaluateDirect(
        depth: java.nio.ByteBuffer,
        width: Int,
        height: Int,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        cameraHeightM: Double,
    ): DoubleArray

    private const val STATUS_VALID = 0
    private const val RESULT_SIZE = 17
    private val REASONS = mapOf(
        1 to "INVALID_INPUT",
        2 to "INSUFFICIENT_GROUND_CANDIDATES",
        3 to "DEGENERATE_RELATIVE_DEPTH",
        4 to "NO_GROUND_CONSENSUS",
        5 to "DEGENERATE_RELATIVE_HEIGHT",
        6 to "GROUND_ORIENTATION_REJECTED",
        7 to "GROUND_SUPPORT_REJECTED",
        8 to "GROUND_RESIDUAL_REJECTED",
        9 to "SCALE_OUT_OF_RANGE",
        10 to "INSUFFICIENT_VALID_DEPTH",
    )

    init {
        System.loadLibrary("dav2_preprocess_native")
    }
}
