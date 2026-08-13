package com.linnan.blindassist.ustrfbenchmark

import com.google.ar.core.Anchor
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import com.linnan.blindassist.ustrf.UstrfPoseSample
import com.linnan.blindassist.ustrf.UstrfPoseState
import com.linnan.blindassist.ustrf.UstrfVector3
import kotlin.math.atan2
import kotlin.math.sqrt

private const val TARO_ARCORE_ANCHOR_FRAME_PREFIX = "arcore-local-anchor-v1"

enum class TaroArCoreAnchorPoseAdmissionFailure {
    SOURCE_FRAME_MISMATCH,
    FRAME_TIMESTAMP_NOT_ADVANCING,
    CAMERA_NOT_TRACKING,
    ANCHOR_NOT_TRACKING,
    TRACKING_WARMUP_INCOMPLETE,
    RELATIVE_POSE_INVALID
}

sealed interface TaroArCoreAnchorPoseAdmission {
    data class Available(
        val cameraPose: UstrfPoseSample,
        val sessionToken: String,
        val continuousTrackingFrames: Int
    ) : TaroArCoreAnchorPoseAdmission {
        init {
            require(sessionToken.isNotBlank())
            require(continuousTrackingFrames > 0)
            require(cameraPose.worldFrame == "$TARO_ARCORE_ANCHOR_FRAME_PREFIX:$sessionToken")
        }
    }

    data class Unavailable(
        val failure: TaroArCoreAnchorPoseAdmissionFailure,
        val continuousTrackingFrames: Int
    ) : TaroArCoreAnchorPoseAdmission {
        init {
            require(continuousTrackingFrames >= 0)
        }
    }
}

/**
 * Benchmark-only bridge from an ARCore session to an anchor-relative camera pose.
 *
 * ARCore world-pose numbers are frame-local. This adapter creates one local anchor and expresses
 * each camera pose relative to the anchor's pose from the same Session.update() frame. The output
 * is admissible only to the TARO camera-history selector: it does not contain or authorize a
 * camera-to-body transform, risk-field warp, payload decode, guidance, or production behavior.
 *
 * All calls, including [close], must remain on the same ARCore/GL thread as [Session.update].
 */
class TaroArCoreAnchorPoseAdmissionAdapter(
    private val session: Session,
    private val sessionToken: String,
    private val minimumContinuousTrackingFrames: Int = 15,
    private val validForNs: Long = 1_000_000_000L
) : AutoCloseable {
    private var anchor: Anchor? = null
    private var lastObservedTimestampNs = -1L
    private var continuousTrackingFrames = 0

    init {
        require(sessionToken.isNotBlank())
        require(minimumContinuousTrackingFrames > 0)
        require(validForNs > 0L)
    }

    fun observe(frame: Frame, sourceFrame: UstrfFrameStamp): TaroArCoreAnchorPoseAdmission {
        if (frame.timestamp != sourceFrame.capturedAtNs) {
            return unavailable(TaroArCoreAnchorPoseAdmissionFailure.SOURCE_FRAME_MISMATCH)
        }
        if (frame.timestamp <= lastObservedTimestampNs) {
            return unavailable(TaroArCoreAnchorPoseAdmissionFailure.FRAME_TIMESTAMP_NOT_ADVANCING)
        }
        lastObservedTimestampNs = frame.timestamp
        if (frame.camera.trackingState != TrackingState.TRACKING) {
            resetAnchor()
            return unavailable(TaroArCoreAnchorPoseAdmissionFailure.CAMERA_NOT_TRACKING)
        }

        continuousTrackingFrames++
        val currentAnchor = anchor ?: session.createAnchor(frame.camera.pose).also { anchor = it }
        if (currentAnchor.trackingState != TrackingState.TRACKING) {
            resetAnchor()
            return unavailable(TaroArCoreAnchorPoseAdmissionFailure.ANCHOR_NOT_TRACKING)
        }
        if (continuousTrackingFrames < minimumContinuousTrackingFrames) {
            return unavailable(TaroArCoreAnchorPoseAdmissionFailure.TRACKING_WARMUP_INCOMPLETE)
        }

        val anchorRelativeCameraPose = currentAnchor.pose.inverse().compose(frame.camera.pose)
        val translation = anchorRelativeCameraPose.translation
        val forwardAxis = anchorRelativeCameraPose.zAxis
        if (!translation.all { it.isFinite() } || !forwardAxis.all { it.isFinite() }) {
            return unavailable(TaroArCoreAnchorPoseAdmissionFailure.RELATIVE_POSE_INVALID)
        }
        val horizontalForwardNorm = sqrt(
            (forwardAxis[0] * forwardAxis[0] + forwardAxis[2] * forwardAxis[2]).toDouble()
        ).toFloat()
        if (!horizontalForwardNorm.isFinite() || horizontalForwardNorm < MINIMUM_HORIZONTAL_FORWARD_NORM) {
            return unavailable(TaroArCoreAnchorPoseAdmissionFailure.RELATIVE_POSE_INVALID)
        }

        return TaroArCoreAnchorPoseAdmission.Available(
            cameraPose = UstrfPoseSample(
                timestampNs = sourceFrame.capturedAtNs,
                worldFrame = "$TARO_ARCORE_ANCHOR_FRAME_PREFIX:$sessionToken",
                cameraFrame = sourceFrame.coordinateFrame,
                worldCameraTranslationM = UstrfVector3(translation[0], translation[1], translation[2]),
                yawRad = atan2(forwardAxis[0], forwardAxis[2]),
                gravityWorld = UstrfVector3(0f, -9.80665f, 0f),
                tracking = UstrfPoseState.TRACKING,
                // This is binary contract admission, not calibrated sensor confidence.
                confidence = 1f,
                validUntilNs = safeAdd(sourceFrame.capturedAtNs, validForNs)
            ),
            sessionToken = sessionToken,
            continuousTrackingFrames = continuousTrackingFrames
        )
    }

    override fun close() {
        resetAnchor()
    }

    private fun resetAnchor() {
        anchor?.detach()
        anchor = null
        continuousTrackingFrames = 0
    }

    private fun unavailable(failure: TaroArCoreAnchorPoseAdmissionFailure) =
        TaroArCoreAnchorPoseAdmission.Unavailable(failure, continuousTrackingFrames)

    private fun safeAdd(first: Long, second: Long): Long =
        if (first > Long.MAX_VALUE - second) Long.MAX_VALUE else first + second

    private companion object {
        const val MINIMUM_HORIZONTAL_FORWARD_NORM = 1e-3f
    }
}
