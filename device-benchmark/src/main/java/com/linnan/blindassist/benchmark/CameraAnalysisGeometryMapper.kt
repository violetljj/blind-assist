package com.linnan.blindassist.benchmark

import kotlin.math.abs

internal data class AnalysisProjectionGeometry(
    val valid: Boolean,
    val worldEnuToDisplayCamera: DoubleArray = doubleArrayOf(),
    val fxPx: Double = Double.NaN,
    val fyPx: Double = Double.NaN,
    val cxPx: Double = Double.NaN,
    val cyPx: Double = Double.NaN,
    val displayWidthPx: Int = 0,
    val displayHeightPx: Int = 0,
    val failureReason: String? = null
)

/** Maps exact Camera2 sensor intrinsics and raw camera axes into the rotated CameraX analysis display. */
internal object CameraAnalysisGeometryMapper {
    private const val EPSILON = 1e-6

    fun map(
        sensorToBuffer: DoubleArray,
        sensorIntrinsicsFxFyCxCySkew: DoubleArray,
        bufferWidthPx: Int,
        bufferHeightPx: Int,
        rotationDegrees: Int,
        worldEnuToRawCamera: DoubleArray
    ): AnalysisProjectionGeometry {
        val invalid = { reason: String -> AnalysisProjectionGeometry(false, failureReason = reason) }
        if (sensorToBuffer.size != 9 || sensorToBuffer.any { !it.isFinite() } ||
            abs(sensorToBuffer[1]) > EPSILON || abs(sensorToBuffer[3]) > EPSILON ||
            abs(sensorToBuffer[6]) > EPSILON || abs(sensorToBuffer[7]) > EPSILON ||
            abs(sensorToBuffer[8] - 1.0) > EPSILON || sensorToBuffer[0] <= 0.0 || sensorToBuffer[4] <= 0.0
        ) return invalid("unsupported_sensor_to_buffer_transform")
        if (sensorIntrinsicsFxFyCxCySkew.size != 5 || sensorIntrinsicsFxFyCxCySkew.any { !it.isFinite() } ||
            sensorIntrinsicsFxFyCxCySkew[0] <= 0.0 || sensorIntrinsicsFxFyCxCySkew[1] <= 0.0 ||
            abs(sensorIntrinsicsFxFyCxCySkew[4]) > EPSILON
        ) return invalid("unsupported_sensor_intrinsics")
        if (bufferWidthPx <= 0 || bufferHeightPx <= 0 || !validRotation(worldEnuToRawCamera)) {
            return invalid("invalid_buffer_or_pose")
        }

        val fxBuffer = sensorToBuffer[0] * sensorIntrinsicsFxFyCxCySkew[0]
        val fyBuffer = sensorToBuffer[4] * sensorIntrinsicsFxFyCxCySkew[1]
        val cxBuffer = sensorToBuffer[0] * sensorIntrinsicsFxFyCxCySkew[2] + sensorToBuffer[2]
        val cyBuffer = sensorToBuffer[4] * sensorIntrinsicsFxFyCxCySkew[3] + sensorToBuffer[5]
        val rotation = ((rotationDegrees % 360) + 360) % 360
        val displayFromRaw: DoubleArray
        val fxDisplay: Double
        val fyDisplay: Double
        val cxDisplay: Double
        val cyDisplay: Double
        val displayWidth: Int
        val displayHeight: Int
        when (rotation) {
            0 -> {
                displayFromRaw = identity()
                fxDisplay = fxBuffer; fyDisplay = fyBuffer; cxDisplay = cxBuffer; cyDisplay = cyBuffer
                displayWidth = bufferWidthPx; displayHeight = bufferHeightPx
            }
            90 -> {
                displayFromRaw = doubleArrayOf(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
                fxDisplay = fyBuffer; fyDisplay = fxBuffer
                cxDisplay = bufferHeightPx - 1.0 - cyBuffer; cyDisplay = cxBuffer
                displayWidth = bufferHeightPx; displayHeight = bufferWidthPx
            }
            180 -> {
                displayFromRaw = doubleArrayOf(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
                fxDisplay = fxBuffer; fyDisplay = fyBuffer
                cxDisplay = bufferWidthPx - 1.0 - cxBuffer; cyDisplay = bufferHeightPx - 1.0 - cyBuffer
                displayWidth = bufferWidthPx; displayHeight = bufferHeightPx
            }
            270 -> {
                displayFromRaw = doubleArrayOf(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
                fxDisplay = fyBuffer; fyDisplay = fxBuffer
                cxDisplay = cyBuffer; cyDisplay = bufferWidthPx - 1.0 - cxBuffer
                displayWidth = bufferHeightPx; displayHeight = bufferWidthPx
            }
            else -> return invalid("unsupported_rotation")
        }
        val worldToDisplay = multiply(displayFromRaw, worldEnuToRawCamera)
        if (!validRotation(worldToDisplay) || cxDisplay !in 0.0..displayWidth.toDouble() || cyDisplay !in 0.0..displayHeight.toDouble()) {
            return invalid("invalid_mapped_geometry")
        }
        return AnalysisProjectionGeometry(
            true, worldToDisplay, fxDisplay, fyDisplay, cxDisplay, cyDisplay, displayWidth, displayHeight
        )
    }

    private fun multiply(left: DoubleArray, right: DoubleArray) = DoubleArray(9) { index ->
        val row = index / 3
        val column = index % 3
        (0..2).sumOf { k -> left[row * 3 + k] * right[k * 3 + column] }
    }

    private fun identity() = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    private fun validRotation(matrix: DoubleArray): Boolean {
        if (matrix.size != 9 || matrix.any { !it.isFinite() }) return false
        fun dot(a: Int, b: Int) = (0..2).sumOf { matrix[a * 3 + it] * matrix[b * 3 + it] }
        if ((0..2).any { abs(dot(it, it) - 1.0) > 1e-3 }) return false
        if (abs(dot(0, 1)) > 1e-3 || abs(dot(0, 2)) > 1e-3 || abs(dot(1, 2)) > 1e-3) return false
        val determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
            matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
            matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
        return abs(determinant - 1.0) <= 1e-3
    }
}
