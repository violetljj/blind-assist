package com.linnan.blindassist.benchmark

import kotlin.math.abs
import kotlin.math.sqrt

internal data class CameraPoseComposition(
    val valid: Boolean,
    val worldEnuToCameraSensor: DoubleArray = doubleArrayOf(),
    val failureReason: String? = null
)

/**
 * Benchmark-only composition of Android rotation-vector and Camera2 lens-pose metadata.
 * deviceToWorldEnu is the row-major matrix returned by SensorManager for ENU world axes.
 * lensPoseRotationXyzw maps Android device-sensor axes to camera-aligned sensor axes.
 * Output is not yet rotated/cropped/scaled into a CameraX analysis buffer.
 */
internal object AndroidCameraPoseComposer {
    private const val ROTATION_TOLERANCE = 1e-3
    private const val QUATERNION_TOLERANCE = 1e-3

    fun compose(
        deviceToWorldEnu: DoubleArray,
        lensPoseRotationXyzw: DoubleArray
    ): CameraPoseComposition {
        val invalid = { reason: String -> CameraPoseComposition(false, failureReason = reason) }
        if (!validRotation(deviceToWorldEnu)) return invalid("invalid_device_to_world_rotation")
        if (lensPoseRotationXyzw.size != 4 || lensPoseRotationXyzw.any { !it.isFinite() }) {
            return invalid("invalid_lens_pose_quaternion")
        }
        val norm = sqrt(lensPoseRotationXyzw.sumOf { it * it })
        if (abs(norm - 1.0) > QUATERNION_TOLERANCE) return invalid("invalid_lens_pose_quaternion")

        val x = lensPoseRotationXyzw[0] / norm
        val y = lensPoseRotationXyzw[1] / norm
        val z = lensPoseRotationXyzw[2] / norm
        val w = lensPoseRotationXyzw[3] / norm
        val cameraFromDevice = doubleArrayOf(
            1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
            2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w,
            2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y
        )
        val deviceFromWorld = transpose(deviceToWorldEnu)
        val cameraFromWorld = multiply(cameraFromDevice, deviceFromWorld)
        if (!validRotation(cameraFromWorld)) return invalid("invalid_composed_rotation")
        return CameraPoseComposition(true, cameraFromWorld)
    }

    private fun transpose(matrix: DoubleArray) = DoubleArray(9) { index ->
        val row = index / 3
        val column = index % 3
        matrix[column * 3 + row]
    }

    private fun multiply(left: DoubleArray, right: DoubleArray) = DoubleArray(9) { index ->
        val row = index / 3
        val column = index % 3
        (0..2).sumOf { k -> left[row * 3 + k] * right[k * 3 + column] }
    }

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
