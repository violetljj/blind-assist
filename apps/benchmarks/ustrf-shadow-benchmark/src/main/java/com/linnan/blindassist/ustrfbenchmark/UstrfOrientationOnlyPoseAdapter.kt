package com.linnan.blindassist.ustrfbenchmark

import android.hardware.SensorManager
import com.linnan.blindassist.ustrf.UstrfPoseSample
import com.linnan.blindassist.ustrf.UstrfPoseState
import com.linnan.blindassist.ustrf.UstrfVector3
import kotlin.math.atan2

/**
 * Benchmark-only rotation-vector adapter. It intentionally emits [UstrfPoseState.DEGRADED]:
 * rotation-vector has neither camera-to-body extrinsics nor translational VIO, so it must not
 * authorize a USTRF risk-field warp or a directional safety action.
 */
class UstrfOrientationOnlyPoseAdapter(
    private val worldFrame: String = "android-rotation-world-v1",
    private val confidence: Float = .5f
) {
    init { require(confidence in 0f..1f) }

    fun receipt(
        timestampNs: Long,
        rotationVector: FloatArray,
        cameraFrame: String,
        validUntilNs: Long
    ): UstrfPoseSample {
        require(timestampNs >= 0L && validUntilNs >= timestampNs)
        require(rotationVector.size >= 3)
        require(cameraFrame.isNotBlank())
        val matrix = FloatArray(9)
        SensorManager.getRotationMatrixFromVector(matrix, rotationVector)
        val yawRad = atan2(matrix[3], matrix[0])
        return UstrfPoseSample(
            timestampNs = timestampNs,
            worldFrame = worldFrame,
            cameraFrame = cameraFrame,
            worldCameraTranslationM = UstrfVector3(0f, 0f, 0f),
            yawRad = yawRad,
            gravityWorld = UstrfVector3(0f, -9.80665f, 0f),
            tracking = UstrfPoseState.DEGRADED,
            confidence = confidence,
            validUntilNs = validUntilNs
        )
    }
}
