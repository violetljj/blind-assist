package com.linnan.blindassist.ustrfbenchmark

import android.graphics.ImageFormat
import android.os.SystemClock
import com.google.ar.core.Frame
import com.google.ar.core.exceptions.NotYetAvailableException
import com.linnan.blindassist.hftf.metricdepth.D45Yuv420Image
import com.linnan.blindassist.hftf.metricdepth.D45Yuv420ToRgbaDecoder
import com.linnan.blindassist.hftf.metricdepth.D45YuvBytePlane
import com.linnan.blindassist.hftf.metricdepth.D45UnregisteredRawMetricDepthFrame
import com.linnan.blindassist.vision.DetectorFrameResult
import com.linnan.blindassist.vision.RgbaVisionFrame
import com.linnan.blindassist.vision.TfliteYoloDetector
import java.nio.ByteBuffer
import java.nio.ByteOrder

enum class D45ArCorePersonDetectorFailure {
    CAMERA_IMAGE_NOT_YET_AVAILABLE,
    CAMERA_IMAGE_ACQUISITION_FAILED,
    CAMERA_TIMESTAMP_MISMATCH,
    CAMERA_DIMENSION_MISMATCH,
    CAMERA_FORMAT_UNSUPPORTED,
    CAMERA_PLANE_COUNT_MISMATCH,
    YUV_DECODE_FAILED,
    DETECTOR_NOT_READY,
    DETECTOR_SOURCE_FRAME_MISMATCH
}

sealed interface D45ArCorePersonDetectorResult {
    data class Available(
        val detectorFrame: DetectorFrameResult,
        val cameraImageTimestampNs: Long,
        val yuvToRgbaLatencyNs: Long,
        val detectorLatencyNs: Long,
        val producedAtNs: Long
    ) : D45ArCorePersonDetectorResult

    data class Unavailable(
        val failure: D45ArCorePersonDetectorFailure,
        val detail: String? = null
    ) : D45ArCorePersonDetectorResult
}

/**
 * Runs the existing CPU YOLO detector on ARCore's exact current CPU camera image.
 *
 * This remains benchmark-only. It does not call the BlindAssist decision/risk/feedback runtime.
 */
class D45ArCorePersonDetectorAdapter(
    private val detector: TfliteYoloDetector,
    private val elapsedRealtimeNs: () -> Long = SystemClock::elapsedRealtimeNanos
) {
    fun observe(
        frame: Frame,
        rawObservation: D45UnregisteredRawMetricDepthFrame,
        detectorRotationDegrees: Int
    ): D45ArCorePersonDetectorResult {
        if (!detector.isReady) {
            return unavailable(
                D45ArCorePersonDetectorFailure.DETECTOR_NOT_READY,
                detector.statusMessage
            )
        }
        val image = try {
            frame.acquireCameraImage()
        } catch (_: NotYetAvailableException) {
            return unavailable(D45ArCorePersonDetectorFailure.CAMERA_IMAGE_NOT_YET_AVAILABLE)
        } catch (error: RuntimeException) {
            return unavailable(
                D45ArCorePersonDetectorFailure.CAMERA_IMAGE_ACQUISITION_FAILED,
                "${error.javaClass.name}:${error.message.orEmpty()}"
            )
        }
        try {
            if (image.timestamp != frame.androidCameraTimestamp) {
                return unavailable(
                    D45ArCorePersonDetectorFailure.CAMERA_TIMESTAMP_MISMATCH,
                    "frameAndroid=${frame.androidCameraTimestamp},image=${image.timestamp}"
                )
            }
            val intrinsics = rawObservation.sourceImageIntrinsics
            if (
                image.width != intrinsics.imageWidthPx ||
                image.height != intrinsics.imageHeightPx
            ) {
                return unavailable(
                    D45ArCorePersonDetectorFailure.CAMERA_DIMENSION_MISMATCH,
                    "image=${image.width}x${image.height}," +
                        "intrinsics=${intrinsics.imageWidthPx}x${intrinsics.imageHeightPx}"
                )
            }
            if (image.format != ImageFormat.YUV_420_888) {
                return unavailable(
                    D45ArCorePersonDetectorFailure.CAMERA_FORMAT_UNSUPPORTED,
                    "format=${image.format}"
                )
            }
            if (image.planes.size != EXPECTED_YUV_PLANE_COUNT) {
                return unavailable(
                    D45ArCorePersonDetectorFailure.CAMERA_PLANE_COUNT_MISMATCH,
                    "planes=${image.planes.size}"
                )
            }
            val decodeStartedAtNs = elapsedRealtimeNs()
            val rgba = try {
                D45Yuv420ToRgbaDecoder.decode(
                    D45Yuv420Image(
                        widthPx = image.width,
                        heightPx = image.height,
                        y = image.planes[0].asCorePlane(),
                        u = image.planes[1].asCorePlane(),
                        v = image.planes[2].asCorePlane()
                    )
                )
            } catch (error: IllegalArgumentException) {
                return unavailable(
                    D45ArCorePersonDetectorFailure.YUV_DECODE_FAILED,
                    error.message
                )
            }
            val decodedAtNs = elapsedRealtimeNs()
            val buffer = ByteBuffer
                .allocateDirect(rgba.bytes.size)
                .order(ByteOrder.nativeOrder())
                .apply {
                    put(rgba.bytes)
                    rewind()
                }
            val detectorStartedAtNs = elapsedRealtimeNs()
            val detectorFrame = detector.detect(
                D45RgbaVisionFrame(
                    width = rgba.widthPx,
                    height = rgba.heightPx,
                    rotationDegrees = detectorRotationDegrees,
                    frameStamp = rawObservation.sourceFrame,
                    buffer = buffer
                )
            )
            val producedAtNs = elapsedRealtimeNs()
            if (detectorFrame.sourceFrame != rawObservation.sourceFrame) {
                return unavailable(
                    D45ArCorePersonDetectorFailure.DETECTOR_SOURCE_FRAME_MISMATCH
                )
            }
            return D45ArCorePersonDetectorResult.Available(
                detectorFrame = detectorFrame,
                cameraImageTimestampNs = image.timestamp,
                yuvToRgbaLatencyNs = decodedAtNs - decodeStartedAtNs,
                detectorLatencyNs = producedAtNs - detectorStartedAtNs,
                producedAtNs = producedAtNs
            )
        } finally {
            image.close()
        }
    }

    private fun android.media.Image.Plane.asCorePlane() = D45YuvBytePlane(
        rowStrideBytes = rowStride,
        pixelStrideBytes = pixelStride,
        buffer = buffer
    )

    private fun unavailable(
        failure: D45ArCorePersonDetectorFailure,
        detail: String? = null
    ) = D45ArCorePersonDetectorResult.Unavailable(failure, detail)

    private data class D45RgbaVisionFrame(
        override val width: Int,
        override val height: Int,
        override val rotationDegrees: Int,
        override val frameStamp: com.linnan.blindassist.vision.FrameStamp,
        override val buffer: ByteBuffer,
        override val rowStride: Int = width * RGBA_CHANNELS,
        override val pixelStride: Int = RGBA_CHANNELS
    ) : RgbaVisionFrame {
        override fun close() = Unit
    }

    private companion object {
        const val EXPECTED_YUV_PLANE_COUNT = 3
        const val RGBA_CHANNELS = 4
    }
}
