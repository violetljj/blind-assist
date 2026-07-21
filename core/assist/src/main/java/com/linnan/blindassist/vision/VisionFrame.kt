package com.linnan.blindassist.vision

import java.nio.ByteBuffer

enum class FrameClockDomain {
    /** Comparable with Android elapsed-realtime timestamps and sensor events on calibrated devices. */
    ANDROID_ELAPSED_REALTIME,
    /** Monotonic for this camera stream, but not proven comparable with another subsystem. */
    CAMERA_HARDWARE_UNMAPPED,
    /** A deterministic offline timeline; never claim that it is a live camera clock. */
    REPLAY_TIMELINE
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

    override fun close()
}

interface RgbaVisionFrame : VisionFrame {
    val buffer: ByteBuffer
    val rowStride: Int
    val pixelStride: Int
}
