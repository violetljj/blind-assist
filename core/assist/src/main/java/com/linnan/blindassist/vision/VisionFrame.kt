package com.linnan.blindassist.vision

import java.nio.ByteBuffer

enum class FrameClockDomain {
    /** Comparable with Android elapsed-realtime timestamps and sensor events on calibrated devices. */
    ANDROID_ELAPSED_REALTIME,
    /** Monotonic for this camera stream, but not proven comparable with another subsystem. */
    CAMERA_HARDWARE_UNMAPPED,
    /** Monotonic on an external device; not comparable with Android until explicitly synchronized. */
    EXTERNAL_DEVICE_MONOTONIC_UNMAPPED,
    /** A deterministic offline timeline; never claim that it is a live camera clock. */
    REPLAY_TIMELINE
}

data class RangingSample(
    val sampledAtNs: Long,
    val valid: Boolean,
    val rangeMm: Int?,
    val ageAtFrameReadyNs: Long?,
    val clockDomain: FrameClockDomain
) {
    init {
        require(sampledAtNs >= 0L)
        require(rangeMm == null || rangeMm >= 0)
        require(ageAtFrameReadyNs == null || ageAtFrameReadyNs >= 0L)
    }
}

/** Immutable capture identity. Decision/effect time is deliberately carried separately. */
data class FrameStamp(
    val frameId: Long,
    val capturedAtNs: Long,
    val receivedAtNs: Long,
    val sourceId: String,
    val coordinateFrame: String,
    val clockDomain: FrameClockDomain
) {
    init {
        require(frameId >= 0L)
        require(capturedAtNs >= 0L && receivedAtNs >= 0L)
        require(sourceId.isNotBlank() && coordinateFrame.isNotBlank())
        if (clockDomain == FrameClockDomain.ANDROID_ELAPSED_REALTIME) {
            require(receivedAtNs >= capturedAtNs) {
                "elapsed-realtime capture cannot arrive before it was captured"
            }
        }
    }
}

interface VisionFrame : AutoCloseable {
    val width: Int
    val height: Int
    val rotationDegrees: Int
    /** Live CameraX frames carry this; bitmap/replay/test frames may remain unstamped. */
    val frameStamp: FrameStamp? get() = null
    /** Optional ranging sample explicitly paired by the producing frame source. */
    val rangingSample: RangingSample? get() = null

    override fun close()
}

interface RgbaVisionFrame : VisionFrame {
    val buffer: ByteBuffer
    val rowStride: Int
    val pixelStride: Int
}
