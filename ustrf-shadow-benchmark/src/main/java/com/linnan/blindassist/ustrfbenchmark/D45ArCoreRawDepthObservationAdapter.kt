package com.linnan.blindassist.ustrfbenchmark

import android.os.SystemClock
import com.google.ar.core.Frame
import com.google.ar.core.exceptions.NotYetAvailableException
import com.linnan.blindassist.hftf.metricdepth.D45DecodedRawDepthRaster
import com.linnan.blindassist.hftf.metricdepth.D45RawDepthPlaneDecoder
import com.linnan.blindassist.hftf.metricdepth.D45StridedBytePlane
import com.linnan.blindassist.hftf.metricdepth.D45UnregisteredRawMetricDepthFrame
import com.linnan.blindassist.hftf.metricdepth.MetricDepthCameraIntrinsics
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import java.nio.ByteOrder

enum class D45ArCoreRawDepthFailure {
    CLOCK_DOMAIN_MISMATCH,
    DEPTH_NOT_YET_AVAILABLE,
    CONFIDENCE_NOT_YET_AVAILABLE,
    DEPTH_ACQUISITION_FAILED,
    CONFIDENCE_ACQUISITION_FAILED,
    DEPTH_TIMESTAMP_MISMATCH,
    CONFIDENCE_TIMESTAMP_MISMATCH,
    DIMENSION_MISMATCH,
    PLANE_COUNT_MISMATCH,
    DECODE_FAILED
}

sealed interface D45ArCoreRawDepthObservationResult {
    data class Available(
        val observation: D45UnregisteredRawMetricDepthFrame
    ) : D45ArCoreRawDepthObservationResult

    data class Unavailable(
        val failure: D45ArCoreRawDepthFailure,
        val detail: String? = null
    ) : D45ArCoreRawDepthObservationResult
}

/**
 * D45-only source adapter for the already isolated ARCore GL benchmark.
 *
 * It decodes pixels only when depth and confidence are bound to the exact current ARCore frame.
 * The result remains explicitly unregistered and therefore cannot reach the person sampler.
 */
class D45ArCoreRawDepthObservationAdapter(
    private val elapsedRealtimeNs: () -> Long = SystemClock::elapsedRealtimeNanos
) {
    fun observe(
        frame: Frame,
        frameId: Long
    ): D45ArCoreRawDepthObservationResult {
        val receivedAtNs = elapsedRealtimeNs()
        if (receivedAtNs < frame.timestamp) {
            return unavailable(D45ArCoreRawDepthFailure.CLOCK_DOMAIN_MISMATCH)
        }
        val depth = try {
            frame.acquireRawDepthImage16Bits()
        } catch (_: NotYetAvailableException) {
            return unavailable(D45ArCoreRawDepthFailure.DEPTH_NOT_YET_AVAILABLE)
        } catch (error: RuntimeException) {
            return unavailable(
                D45ArCoreRawDepthFailure.DEPTH_ACQUISITION_FAILED,
                "${error.javaClass.name}:${error.message.orEmpty()}"
            )
        }
        try {
            val confidence = try {
                frame.acquireRawDepthConfidenceImage()
            } catch (_: NotYetAvailableException) {
                return unavailable(D45ArCoreRawDepthFailure.CONFIDENCE_NOT_YET_AVAILABLE)
            } catch (error: RuntimeException) {
                return unavailable(
                    D45ArCoreRawDepthFailure.CONFIDENCE_ACQUISITION_FAILED,
                    "${error.javaClass.name}:${error.message.orEmpty()}"
                )
            }
            try {
                if (depth.timestamp != frame.timestamp) {
                    return unavailable(
                        D45ArCoreRawDepthFailure.DEPTH_TIMESTAMP_MISMATCH,
                        "frame=${frame.timestamp},depth=${depth.timestamp}"
                    )
                }
                if (confidence.timestamp != frame.timestamp) {
                    return unavailable(
                        D45ArCoreRawDepthFailure.CONFIDENCE_TIMESTAMP_MISMATCH,
                        "frame=${frame.timestamp},confidence=${confidence.timestamp}"
                    )
                }
                if (depth.width != confidence.width || depth.height != confidence.height) {
                    return unavailable(D45ArCoreRawDepthFailure.DIMENSION_MISMATCH)
                }
                if (depth.planes.size != 1 || confidence.planes.size != 1) {
                    return unavailable(
                        D45ArCoreRawDepthFailure.PLANE_COUNT_MISMATCH,
                        "depth=${depth.planes.size},confidence=${confidence.planes.size}"
                    )
                }
                val decoded = try {
                    decode(depth, confidence)
                } catch (error: IllegalArgumentException) {
                    return unavailable(
                        D45ArCoreRawDepthFailure.DECODE_FAILED,
                        error.message
                    )
                }
                val intrinsics = frame.camera.imageIntrinsics
                val dimensions = intrinsics.imageDimensions
                val focal = intrinsics.focalLength
                val principal = intrinsics.principalPoint
                val producedAtNs = elapsedRealtimeNs()
                val sourceFrame = FrameStamp(
                    frameId = frameId,
                    capturedAtNs = frame.timestamp,
                    receivedAtNs = receivedAtNs,
                    sourceId = SOURCE_ID,
                    coordinateFrame = SOURCE_CAMERA_FRAME,
                    clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
                )
                return D45ArCoreRawDepthObservationResult.Available(
                    D45UnregisteredRawMetricDepthFrame(
                        sourceFrame = sourceFrame,
                        sourceImageIntrinsics = MetricDepthCameraIntrinsics(
                            imageWidthPx = dimensions[0],
                            imageHeightPx = dimensions[1],
                            focalXpx = focal[0],
                            focalYpx = focal[1],
                            principalXpx = principal[0],
                            principalYpx = principal[1]
                        ),
                        raster = decoded,
                        depthTimestampNs = depth.timestamp,
                        confidenceTimestampNs = confidence.timestamp,
                        producedAtNs = producedAtNs
                    )
                )
            } finally {
                confidence.close()
            }
        } finally {
            depth.close()
        }
    }

    private fun decode(
        depth: android.media.Image,
        confidence: android.media.Image
    ): D45DecodedRawDepthRaster {
        val depthPlane = depth.planes.single()
        val confidencePlane = confidence.planes.single()
        return D45RawDepthPlaneDecoder.decode(
            depth = D45StridedBytePlane(
                widthPx = depth.width,
                heightPx = depth.height,
                rowStrideBytes = depthPlane.rowStride,
                pixelStrideBytes = depthPlane.pixelStride,
                buffer = depthPlane.buffer
            ),
            confidence = D45StridedBytePlane(
                widthPx = confidence.width,
                heightPx = confidence.height,
                rowStrideBytes = confidencePlane.rowStride,
                pixelStrideBytes = confidencePlane.pixelStride,
                buffer = confidencePlane.buffer
            ),
            depthByteOrder = ByteOrder.nativeOrder()
        )
    }

    private fun unavailable(
        failure: D45ArCoreRawDepthFailure,
        detail: String? = null
    ) = D45ArCoreRawDepthObservationResult.Unavailable(failure, detail)

    private companion object {
        const val SOURCE_ID = "arcore:raw-depth"
        const val SOURCE_CAMERA_FRAME = "arcore:camera-image"
    }
}
