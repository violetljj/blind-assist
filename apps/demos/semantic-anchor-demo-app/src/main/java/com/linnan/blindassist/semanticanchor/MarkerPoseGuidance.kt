package com.linnan.blindassist.semanticanchor

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sqrt

internal data class PixelPoint(val x: Double, val y: Double)

internal data class CameraIntrinsics(
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
) {
    init {
        require(fx > 0.0 && fy > 0.0)
    }
}

internal data class MarkerPoseEstimate(
    val payload: String,
    val centerXNormalized: Double,
    val centerBearingDegrees: Double,
    val rangeMeters: Double,
    val lateralMeters: Double,
    val markerYawDegrees: Double,
    val waypointLateralMeters: Double,
    val waypointForwardMeters: Double,
    val reprojectionErrorPixels: Double,
)

internal enum class GuidanceArm { CENTER_BASELINE, PNP_POSE }

internal enum class GuidancePhase { SEARCH, ALIGN, ADVANCE, LOST, ARRIVE }

internal data class MarkerGuidance(
    val phase: GuidancePhase,
    val command: String,
    val detail: String,
    val estimate: MarkerPoseEstimate? = null,
)

/**
 * Square-planar pose from four clockwise image corners beginning at top-left.
 * The implementation is a calibrated planar PnP specialization: solve the homography in
 * normalized camera coordinates, then recover the rotation columns and translation.
 */
internal object SquareMarkerPoseSolver {
    fun solve(
        payload: String,
        corners: List<PixelPoint>,
        intrinsics: CameraIntrinsics,
        markerSizeMeters: Double,
        standoffMeters: Double,
    ): MarkerPoseEstimate? {
        if (corners.size != 4 || markerSizeMeters <= 0.0 || standoffMeters < 0.0) return null
        val half = markerSizeMeters / 2.0
        val plane = listOf(
            PixelPoint(-half, -half),
            PixelPoint(half, -half),
            PixelPoint(half, half),
            PixelPoint(-half, half),
        )
        val normalized = corners.map {
            PixelPoint((it.x - intrinsics.cx) / intrinsics.fx, (it.y - intrinsics.cy) / intrinsics.fy)
        }
        val a = Array(8) { DoubleArray(8) }
        val b = DoubleArray(8)
        plane.indices.forEach { index ->
            val world = plane[index]
            val image = normalized[index]
            val row = index * 2
            a[row][0] = world.x
            a[row][1] = world.y
            a[row][2] = 1.0
            a[row][6] = -image.x * world.x
            a[row][7] = -image.x * world.y
            b[row] = image.x
            a[row + 1][3] = world.x
            a[row + 1][4] = world.y
            a[row + 1][5] = 1.0
            a[row + 1][6] = -image.y * world.x
            a[row + 1][7] = -image.y * world.y
            b[row + 1] = image.y
        }
        val h = solveLinearSystem(a, b) ?: return null
        val c1 = Vec3(h[0], h[3], h[6])
        val c2 = Vec3(h[1], h[4], h[7])
        val c3 = Vec3(h[2], h[5], 1.0)
        val scale = 2.0 / (c1.norm() + c2.norm())
        var r1 = (c1 * scale).normalized() ?: return null
        val c2Scaled = c2 * scale
        var r2 = (c2Scaled - r1 * r1.dot(c2Scaled)).normalized() ?: return null
        var normal = r1.cross(r2).normalized() ?: return null
        var translation = c3 * scale
        if (translation.z < 0.0) {
            translation = translation * -1.0
            r1 = r1 * -1.0
            r2 = r2 * -1.0
            normal = r1.cross(r2).normalized() ?: return null
        }
        val outwardNormal = if (normal.dot(translation) > 0.0) normal * -1.0 else normal
        val waypoint = translation + outwardNormal * standoffMeters
        val reprojection = plane.indices.map { index ->
            val world = plane[index]
            val camera = translation + r1 * world.x + r2 * world.y
            if (camera.z <= 1e-6) return null
            val u = intrinsics.fx * camera.x / camera.z + intrinsics.cx
            val v = intrinsics.fy * camera.y / camera.z + intrinsics.cy
            hypot(u - corners[index].x, v - corners[index].y)
        }.average()
        if (!reprojection.isFinite() || reprojection > 18.0) return null
        val centerX = corners.map(PixelPoint::x).average()
        val imageWidth = intrinsics.cx * 2.0
        return MarkerPoseEstimate(
            payload = payload,
            centerXNormalized = ((centerX - intrinsics.cx) / imageWidth).coerceIn(-0.5, 0.5),
            centerBearingDegrees = Math.toDegrees(atan2(translation.x, translation.z)),
            rangeMeters = sqrt(translation.x * translation.x + translation.y * translation.y + translation.z * translation.z),
            lateralMeters = translation.x,
            markerYawDegrees = Math.toDegrees(atan2(outwardNormal.x, -outwardNormal.z)),
            waypointLateralMeters = waypoint.x,
            waypointForwardMeters = waypoint.z,
            reprojectionErrorPixels = reprojection,
        )
    }

    private fun solveLinearSystem(input: Array<DoubleArray>, rhs: DoubleArray): DoubleArray? {
        val n = rhs.size
        val a = Array(n) { row -> input[row].copyOf() }
        val b = rhs.copyOf()
        for (column in 0 until n) {
            var pivot = column
            for (row in column + 1 until n) if (abs(a[row][column]) > abs(a[pivot][column])) pivot = row
            if (abs(a[pivot][column]) < 1e-10) return null
            a[column] = a[pivot].also { a[pivot] = a[column] }
            val swap = b[column]
            b[column] = b[pivot]
            b[pivot] = swap
            val divisor = a[column][column]
            for (j in column until n) a[column][j] /= divisor
            b[column] /= divisor
            for (row in 0 until n) {
                if (row == column) continue
                val factor = a[row][column]
                for (j in column until n) a[row][j] -= factor * a[column][j]
                b[row] -= factor * b[column]
            }
        }
        return b
    }

    private data class Vec3(val x: Double, val y: Double, val z: Double) {
        operator fun plus(other: Vec3) = Vec3(x + other.x, y + other.y, z + other.z)
        operator fun minus(other: Vec3) = Vec3(x - other.x, y - other.y, z - other.z)
        operator fun times(scale: Double) = Vec3(x * scale, y * scale, z * scale)
        fun dot(other: Vec3) = x * other.x + y * other.y + z * other.z
        fun cross(other: Vec3) = Vec3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x)
        fun norm() = sqrt(dot(this))
        fun normalized(): Vec3? = norm().takeIf { it > 1e-10 }?.let { this * (1.0 / it) }
    }
}

internal class MarkerPoseController(
    var arm: GuidanceArm = GuidanceArm.PNP_POSE,
    private val lateralToleranceMeters: Double = 0.12,
    private val yawToleranceDegrees: Double = 12.0,
    private val arrivalToleranceMeters: Double = 0.22,
    private val arrivalFramesRequired: Int = 2,
) {
    private var arrivalStreak = 0

    init {
        require(arrivalFramesRequired > 0)
    }

    fun reset() {
        arrivalStreak = 0
    }

    fun update(identityPhase: AnchorPhase, estimate: MarkerPoseEstimate?): MarkerGuidance {
        if (identityPhase == AnchorPhase.SEARCH) {
            reset()
            return MarkerGuidance(GuidancePhase.SEARCH, "SEARCH", "等待 exact QR ID")
        }
        if (identityPhase == AnchorPhase.LOST || estimate == null) {
            reset()
            return MarkerGuidance(GuidancePhase.LOST, "STOP", "目标丢失；等待同 ID fresh reacquire")
        }
        return when (arm) {
            GuidanceArm.CENTER_BASELINE -> centerBaseline(estimate)
            GuidanceArm.PNP_POSE -> pnpGuidance(estimate)
        }
    }

    private fun centerBaseline(estimate: MarkerPoseEstimate): MarkerGuidance {
        val candidate = when {
            estimate.centerXNormalized < -0.055 -> "LEFT"
            estimate.centerXNormalized > 0.055 -> "RIGHT"
            estimate.rangeMeters <= 0.72 -> "ARRIVE"
            else -> "FORWARD"
        }
        val command = stableArrival(candidate)
        return MarkerGuidance(
            phase = if (command == "ARRIVE") GuidancePhase.ARRIVE else if (command == "FORWARD") GuidancePhase.ADVANCE else GuidancePhase.ALIGN,
            command = command,
            detail = "中心基线 · x=${format(estimate.centerXNormalized)} · scale-range=${format(estimate.rangeMeters)}m",
            estimate = estimate,
        )
    }

    private fun pnpGuidance(estimate: MarkerPoseEstimate): MarkerGuidance {
        val waypointRange = hypot(estimate.waypointLateralMeters, estimate.waypointForwardMeters)
        val candidate = when {
            waypointRange <= arrivalToleranceMeters && abs(estimate.markerYawDegrees) <= yawToleranceDegrees -> "ARRIVE"
            abs(estimate.markerYawDegrees) > yawToleranceDegrees -> if (estimate.markerYawDegrees > 0.0) "TURN RIGHT" else "TURN LEFT"
            abs(estimate.waypointLateralMeters) > lateralToleranceMeters -> if (estimate.waypointLateralMeters > 0.0) "MOVE RIGHT" else "MOVE LEFT"
            else -> "FORWARD"
        }
        val command = stableArrival(candidate)
        val phase = when (command) {
            "ARRIVE" -> GuidancePhase.ARRIVE
            "FORWARD" -> GuidancePhase.ADVANCE
            else -> GuidancePhase.ALIGN
        }
        return MarkerGuidance(
            phase = phase,
            command = command,
            detail = "PnP · range=${format(estimate.rangeMeters)}m · lateral=${format(estimate.lateralMeters)}m · yaw=${format(estimate.markerYawDegrees)}deg · reproj=${format(estimate.reprojectionErrorPixels)}px",
            estimate = estimate,
        )
    }

    private fun stableArrival(candidate: String): String {
        if (candidate != "ARRIVE") {
            arrivalStreak = 0
            return candidate
        }
        arrivalStreak += 1
        return if (arrivalStreak >= arrivalFramesRequired) "ARRIVE" else "HOLD"
    }

    private fun format(value: Double) = "%.2f".format(java.util.Locale.US, value)
}
