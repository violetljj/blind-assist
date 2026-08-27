package com.linnan.blindassist.vision

import java.nio.ByteBuffer

/** Pinhole calibration in the pixel coordinates of the frame that carries it. */
data class CameraIntrinsics(
    val focalLengthXPx: Float,
    val focalLengthYPx: Float,
    val principalPointXPx: Float,
    val principalPointYPx: Float,
    val coordinateWidthPx: Int,
    val coordinateHeightPx: Int
) {
    init {
        require(focalLengthXPx.isFinite() && focalLengthXPx > 0f)
        require(focalLengthYPx.isFinite() && focalLengthYPx > 0f)
        require(principalPointXPx.isFinite() && principalPointXPx in 0f..coordinateWidthPx.toFloat())
        require(principalPointYPx.isFinite() && principalPointYPx in 0f..coordinateHeightPx.toFloat())
        require(coordinateWidthPx > 0 && coordinateHeightPx > 0)
    }

    /** Match the clockwise display rotation used by the realtime detector. */
    fun rotatedForDisplay(rotationDegrees: Int): CameraIntrinsics {
        val rotation = ((rotationDegrees % 360) + 360) % 360
        require(rotation % 90 == 0) { "camera rotation must be a multiple of 90 degrees" }
        return when (rotation) {
            90 -> CameraIntrinsics(
                focalLengthXPx = focalLengthYPx,
                focalLengthYPx = focalLengthXPx,
                principalPointXPx = coordinateHeightPx - principalPointYPx,
                principalPointYPx = principalPointXPx,
                coordinateWidthPx = coordinateHeightPx,
                coordinateHeightPx = coordinateWidthPx
            )
            180 -> copy(
                principalPointXPx = coordinateWidthPx - principalPointXPx,
                principalPointYPx = coordinateHeightPx - principalPointYPx
            )
            270 -> CameraIntrinsics(
                focalLengthXPx = focalLengthYPx,
                focalLengthYPx = focalLengthXPx,
                principalPointXPx = principalPointYPx,
                principalPointYPx = coordinateWidthPx - principalPointXPx,
                coordinateWidthPx = coordinateHeightPx,
                coordinateHeightPx = coordinateWidthPx
            )
            else -> this
        }
    }
}

enum class FrameClockDomain {
    /** Comparable with Android elapsed-realtime timestamps and sensor events on calibrated devices. */
    ANDROID_ELAPSED_REALTIME,
    /** Monotonic for this camera stream, but not proven comparable with another subsystem. */
    CAMERA_HARDWARE_UNMAPPED,
    /** Monotonic on an external device; not comparable with Android until explicitly synchronized. */
    EXTERNAL_DEVICE_MONOTONIC_UNMAPPED,
    /** External monotonic timestamps mapped to Android elapsed realtime by bounded UDP midpoint sync. */
    EXTERNAL_DEVICE_MONOTONIC_MAPPED_TO_ANDROID,
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

/** Frame-bound acquisition timing. All Android fields use elapsed-realtime nanoseconds. */
data class ExternalFrameTiming(
    val deviceCaptureNs: Long,
    val deviceJpegReadyNs: Long,
    val deviceSendStartNs: Long?,
    val androidReadStartNs: Long,
    val androidFirstByteNs: Long,
    val androidJpegCompleteNs: Long,
    val androidDecodeStartNs: Long,
    val androidDecodeCompleteNs: Long,
    val androidRgbaCompleteNs: Long,
    val deviceMinusAndroidNs: Long?,
    val clockSyncRttNs: Long?,
    val clockSyncErrorBoundNs: Long?
) {
    init {
        require(deviceCaptureNs >= 0L && deviceJpegReadyNs >= deviceCaptureNs)
        require(deviceSendStartNs == null || deviceSendStartNs >= deviceJpegReadyNs)
        require(androidReadStartNs >= 0L)
        require(androidFirstByteNs >= androidReadStartNs)
        require(androidJpegCompleteNs >= androidFirstByteNs)
        require(androidDecodeStartNs >= androidJpegCompleteNs)
        require(androidDecodeCompleteNs >= androidDecodeStartNs)
        require(androidRgbaCompleteNs >= androidDecodeCompleteNs)
        require(clockSyncRttNs == null || clockSyncRttNs >= 0L)
        require(clockSyncErrorBoundNs == null || clockSyncErrorBoundNs >= 0L)
    }
}

data class ExternalFrameTransportDiagnostics(
    val jpegSizeBytes: Int,
    val wifiRssiDbm: Int?,
    val previousFrameSequence: Long?,
    val previousResponseWriteDurationNs: Long?,
    val androidBodyReadCalls: Int,
    val androidMaxBodyReadGapNs: Long
) {
    init {
        require(jpegSizeBytes > 0)
        require(previousFrameSequence == null || previousFrameSequence >= 0L)
        require(previousResponseWriteDurationNs == null || previousResponseWriteDurationNs >= 0L)
        require(androidBodyReadCalls > 0)
        require(androidMaxBodyReadGapNs >= 0L)
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
        if (clockDomain == FrameClockDomain.ANDROID_ELAPSED_REALTIME ||
            clockDomain == FrameClockDomain.EXTERNAL_DEVICE_MONOTONIC_MAPPED_TO_ANDROID
        ) {
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
    /** Optional calibration in the unrotated frame-buffer coordinate system. */
    val cameraIntrinsics: CameraIntrinsics? get() = null
    /** Live CameraX frames carry this; bitmap/replay/test frames may remain unstamped. */
    val frameStamp: FrameStamp? get() = null
    /** Optional ranging sample explicitly paired by the producing frame source. */
    val rangingSample: RangingSample? get() = null
    /** Optional external acquisition/decode timing carried with the exact frame. */
    val externalTiming: ExternalFrameTiming? get() = null
    /** Optional transport diagnostics carried with the exact external frame. */
    val externalTransportDiagnostics: ExternalFrameTransportDiagnostics? get() = null

    override fun close()
}

interface RgbaVisionFrame : VisionFrame {
    val buffer: ByteBuffer
    val rowStride: Int
    val pixelStride: Int
}

/** A frame whose decoded bitmap remains owned by the frame until [close]. */
interface NativeImageVisionFrame : VisionFrame {
    val nativeImage: Any
}
