package com.linnan.blindassist.ustrfbenchmark

import com.google.ar.core.Frame
import com.linnan.blindassist.ustrf.UstrfFrameStamp

/**
 * Raw ARCore image-intrinsics observation for audit only. It deliberately has no calibration
 * confidence or independent-verification claim, so it cannot satisfy USTRF metric-geometry gates.
 */
data class UstrfArCoreRawCameraIntrinsicsObservation(
    val sourceFrame: UstrfFrameStamp,
    val imageWidthPx: Int,
    val imageHeightPx: Int,
    val focalXpx: Float,
    val focalYpx: Float,
    val principalXpx: Float,
    val principalYpx: Float
) {
    fun signature(): String = "$imageWidthPx:$imageHeightPx:$focalXpx:$focalYpx:$principalXpx:$principalYpx"
}

class UstrfArCoreRawCameraIntrinsicsAdapter {
    fun observe(frame: Frame, sourceFrame: UstrfFrameStamp): UstrfArCoreRawCameraIntrinsicsObservation {
        require(frame.timestamp == sourceFrame.capturedAtNs) {
            "ARCore frame timestamp must bind to the USTRF source frame"
        }
        val intrinsics = frame.camera.imageIntrinsics
        val dimensions = intrinsics.imageDimensions
        val focal = intrinsics.focalLength
        val principal = intrinsics.principalPoint
        return UstrfArCoreRawCameraIntrinsicsObservation(
            sourceFrame = sourceFrame,
            imageWidthPx = dimensions[0],
            imageHeightPx = dimensions[1],
            focalXpx = focal[0],
            focalYpx = focal[1],
            principalXpx = principal[0],
            principalYpx = principal[1]
        )
    }
}
