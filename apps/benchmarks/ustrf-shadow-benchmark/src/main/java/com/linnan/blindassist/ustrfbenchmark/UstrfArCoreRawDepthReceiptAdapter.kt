package com.linnan.blindassist.ustrfbenchmark

import com.google.ar.core.Frame
import com.google.ar.core.exceptions.NotYetAvailableException
import com.linnan.blindassist.ustrf.UstrfFrameStamp

/**
 * Metadata-only raw-depth observation for the isolated benchmark.
 *
 * ARCore can return reprojected depth from an older frame. This adapter deliberately makes that
 * distinction explicit and never manufactures a [com.linnan.blindassist.ustrf.UstrfGeometryPacket].
 * Metric projection additionally needs a separately validated intrinsics, coordinate-transform,
 * and camera-to-body-extrinsics chain.
 */
enum class UstrfRawDepthFreshness {
    FRESH_SOURCE_ALIGNED,
    REPROJECTED_OR_STALE
}

data class UstrfArCoreRawDepthReceipt(
    val sourceFrame: UstrfFrameStamp,
    val depthTimestampNs: Long,
    val confidenceTimestampNs: Long,
    val depthWidth: Int,
    val depthHeight: Int,
    val depthRowStrideBytes: Int,
    val depthPixelStrideBytes: Int,
    val confidenceWidth: Int,
    val confidenceHeight: Int,
    val confidenceRowStrideBytes: Int,
    val confidencePixelStrideBytes: Int,
    val freshness: UstrfRawDepthFreshness
) {
    val isFreshForSourceFrame: Boolean
        get() = freshness == UstrfRawDepthFreshness.FRESH_SOURCE_ALIGNED
}

class UstrfArCoreRawDepthReceiptAdapter {
    /** Returns null only when ARCore has no raw depth/confidence image available for this update. */
    fun observeOrNull(frame: Frame, sourceFrame: UstrfFrameStamp): UstrfArCoreRawDepthReceipt? {
        require(frame.timestamp == sourceFrame.capturedAtNs) {
            "ARCore frame timestamp must bind to the USTRF source frame"
        }
        val depth = try {
            frame.acquireRawDepthImage16Bits()
        } catch (_: NotYetAvailableException) {
            return null
        }
        try {
            val confidence = try {
                frame.acquireRawDepthConfidenceImage()
            } catch (_: NotYetAvailableException) {
                return null
            }
            try {
                val freshness = if (
                    depth.timestamp == sourceFrame.capturedAtNs &&
                    confidence.timestamp == sourceFrame.capturedAtNs
                ) {
                    UstrfRawDepthFreshness.FRESH_SOURCE_ALIGNED
                } else {
                    UstrfRawDepthFreshness.REPROJECTED_OR_STALE
                }
                return UstrfArCoreRawDepthReceipt(
                    sourceFrame = sourceFrame,
                    depthTimestampNs = depth.timestamp,
                    confidenceTimestampNs = confidence.timestamp,
                    depthWidth = depth.width,
                    depthHeight = depth.height,
                    depthRowStrideBytes = depth.planes.first().rowStride,
                    depthPixelStrideBytes = depth.planes.first().pixelStride,
                    confidenceWidth = confidence.width,
                    confidenceHeight = confidence.height,
                    confidenceRowStrideBytes = confidence.planes.first().rowStride,
                    confidencePixelStrideBytes = confidence.planes.first().pixelStride,
                    freshness = freshness
                )
            } finally {
                confidence.close()
            }
        } finally {
            depth.close()
        }
    }
}
