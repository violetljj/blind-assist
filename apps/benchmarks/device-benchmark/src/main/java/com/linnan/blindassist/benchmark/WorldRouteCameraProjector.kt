package com.linnan.blindassist.benchmark

import kotlin.math.abs
import kotlin.math.sqrt

internal data class WorldRoutePoint(
    val horizonMs: Long,
    val eastM: Double,
    val northM: Double,
    val upM: Double
)

internal data class CameraProjectionReceipt(
    val receiptId: String,
    val timestampMs: Long,
    val validUntilTimestampMs: Long,
    val confidence: Double,
    val cameraOriginEnuM: DoubleArray,
    /** Row-major rotation mapping a world ENU delta to camera x-right, y-down, z-forward. */
    val worldEnuToCamera: DoubleArray,
    val fxPx: Double,
    val fyPx: Double,
    val cxPx: Double,
    val cyPx: Double,
    val frameWidthPx: Int,
    val frameHeightPx: Int
)

internal data class ProjectedRoute(
    val routeValid: Boolean,
    val waypoints: List<ExplicitRouteWaypoint>,
    val failureReason: String? = null
)

/** Benchmark-only pinhole projection. It consumes an external pose receipt; it never infers pose. */
internal object WorldRouteCameraProjector {
    private val requiredHorizons = listOf(1_000L, 2_000L, 3_000L)
    private const val MINIMUM_DEPTH_M = 0.05
    private const val MAXIMUM_RECEIPT_AGE_MS = 1_000L
    private const val MINIMUM_CONFIDENCE = 0.5
    private const val ROTATION_TOLERANCE = 1e-3

    fun project(
        points: List<WorldRoutePoint>,
        receipt: CameraProjectionReceipt,
        frameTimestampMs: Long
    ): ProjectedRoute {
        val invalid = { reason: String -> ProjectedRoute(false, emptyList(), reason) }
        if (receipt.receiptId.isBlank()) return invalid("missing_receipt")
        if (!receipt.confidence.isFinite() || receipt.confidence < MINIMUM_CONFIDENCE) {
            return invalid("low_confidence")
        }
        if (receipt.timestampMs > frameTimestampMs || frameTimestampMs > receipt.validUntilTimestampMs ||
            frameTimestampMs - receipt.timestampMs > MAXIMUM_RECEIPT_AGE_MS ||
            receipt.validUntilTimestampMs - receipt.timestampMs > MAXIMUM_RECEIPT_AGE_MS
        ) return invalid("stale_or_future_receipt")
        if (!validIntrinsics(receipt)) return invalid("invalid_intrinsics")
        if (!validRotation(receipt.worldEnuToCamera) ||
            receipt.cameraOriginEnuM.size != 3 || receipt.cameraOriginEnuM.any { !it.isFinite() }
        ) return invalid("invalid_pose")

        val byHorizon = points.groupBy { it.horizonMs }
        if (points.size != requiredHorizons.size || requiredHorizons.any { byHorizon[it]?.size != 1 }) {
            return invalid("invalid_horizons")
        }
        val rotation = receipt.worldEnuToCamera
        val waypoints = requiredHorizons.map { horizon ->
            val point = byHorizon.getValue(horizon).single()
            val world = doubleArrayOf(point.eastM, point.northM, point.upM)
            if (world.any { !it.isFinite() }) return invalid("non_finite_point")
            val delta = DoubleArray(3) { world[it] - receipt.cameraOriginEnuM[it] }
            val camera = DoubleArray(3) { row ->
                rotation[row * 3] * delta[0] + rotation[row * 3 + 1] * delta[1] +
                    rotation[row * 3 + 2] * delta[2]
            }
            if (camera[2] <= MINIMUM_DEPTH_M) return invalid("point_behind_or_too_close")
            val xNorm = (receipt.fxPx * camera[0] / camera[2] + receipt.cxPx) / receipt.frameWidthPx
            val yNorm = (receipt.fyPx * camera[1] / camera[2] + receipt.cyPx) / receipt.frameHeightPx
            if (!xNorm.isFinite() || !yNorm.isFinite() || xNorm !in 0.0..1.0 || yNorm !in 0.0..1.0) {
                return invalid("projection_out_of_frame")
            }
            ExplicitRouteWaypoint(horizon, xNorm, yNorm)
        }
        return ProjectedRoute(true, waypoints)
    }

    private fun validIntrinsics(receipt: CameraProjectionReceipt): Boolean =
        receipt.frameWidthPx > 0 && receipt.frameHeightPx > 0 &&
            receipt.fxPx.isFinite() && receipt.fyPx.isFinite() &&
            receipt.cxPx.isFinite() && receipt.cyPx.isFinite() &&
            receipt.fxPx > 0.0 && receipt.fyPx > 0.0 &&
            receipt.cxPx in 0.0..receipt.frameWidthPx.toDouble() &&
            receipt.cyPx in 0.0..receipt.frameHeightPx.toDouble()

    private fun validRotation(matrix: DoubleArray): Boolean {
        if (matrix.size != 9 || matrix.any { !it.isFinite() }) return false
        fun dot(a: Int, b: Int) = (0..2).sumOf { matrix[a * 3 + it] * matrix[b * 3 + it] }
        for (row in 0..2) {
            if (abs(sqrt(dot(row, row)) - 1.0) > ROTATION_TOLERANCE) return false
        }
        if (abs(dot(0, 1)) > ROTATION_TOLERANCE || abs(dot(0, 2)) > ROTATION_TOLERANCE ||
            abs(dot(1, 2)) > ROTATION_TOLERANCE
        ) return false
        val determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
            matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
            matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
        return abs(determinant - 1.0) <= ROTATION_TOLERANCE
    }
}
