package com.linnan.blindassist.ustrfbenchmark

import com.google.ar.core.Frame
import com.google.ar.core.TrackingState
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import com.linnan.blindassist.ustrf.UstrfVector3

/**
 * Benchmark-only raw ARCore observation. It deliberately preserves the frame-local coordinate
 * caveat instead of manufacturing an inter-frame-stable USTRF pose receipt.
 */
data class UstrfArCoreRawPoseReceipt(
    val sourceFrame: UstrfFrameStamp,
    val trackingState: String,
    val worldCameraTranslationM: UstrfVector3,
    /** ARCore quaternion in x, y, z, w order. */
    val rotationQuaternionXyzw: FloatArray,
    val worldFrameStability: String = "EPHEMERAL_PER_FRAME"
) {
    init {
        require(rotationQuaternionXyzw.size == 4)
        require(worldFrameStability == "EPHEMERAL_PER_FRAME")
    }
}

class UstrfArCoreRawPoseReceiptAdapter {
    fun observe(frame: Frame, sourceFrame: UstrfFrameStamp): UstrfArCoreRawPoseReceipt {
        require(frame.timestamp == sourceFrame.capturedAtNs) { "ARCore frame timestamp must bind to the USTRF source frame" }
        val pose = frame.camera.pose
        return UstrfArCoreRawPoseReceipt(
            sourceFrame = sourceFrame,
            trackingState = frame.camera.trackingState.name,
            worldCameraTranslationM = UstrfVector3(pose.tx(), pose.ty(), pose.tz()),
            rotationQuaternionXyzw = floatArrayOf(pose.qx(), pose.qy(), pose.qz(), pose.qw())
        )
    }

    fun isTracking(receipt: UstrfArCoreRawPoseReceipt): Boolean = receipt.trackingState == TrackingState.TRACKING.name
}
