package com.linnan.blindassist.ustrfbenchmark

import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.linnan.blindassist.hftf.metricdepth.D45CoordinateCorrespondence
import com.linnan.blindassist.hftf.metricdepth.D45FrameBoundMetricDepthRegistration
import com.linnan.blindassist.hftf.metricdepth.D45MetricDepthRegistrationFactory
import com.linnan.blindassist.hftf.metricdepth.D45MetricDepthRegistrationFailure
import com.linnan.blindassist.hftf.metricdepth.D45MetricDepthRegistrationResult
import com.linnan.blindassist.hftf.metricdepth.D45UnregisteredRawMetricDepthFrame
import com.linnan.blindassist.hftf.metricdepth.MetricImagePoint
import com.linnan.blindassist.model.FrameSize

enum class D45ArCoreRegistrationFailure {
    FRAME_TIMESTAMP_MISMATCH,
    CAMERA_DIMENSION_MISMATCH,
    COORDINATE_TRANSFORM_FAILED,
    REGISTRATION_REJECTED
}

sealed interface D45ArCoreRegistrationObservationResult {
    data class Available(
        val observation: D45FrameBoundMetricDepthRegistration
    ) : D45ArCoreRegistrationObservationResult

    data class Unavailable(
        val failure: D45ArCoreRegistrationFailure,
        val registrationFailure: D45MetricDepthRegistrationFailure? = null,
        val detail: String? = null
    ) : D45ArCoreRegistrationObservationResult
}

/**
 * Converts ARCore's current-frame CPU-image -> depth coordinate mapping into a checked affine.
 *
 * Detector orientation is explicit because BlindAssist boxes use the full rotated camera image,
 * while ARCore raw depth uses TEXTURE_NORMALIZED coordinates. No person samples are read here.
 */
class D45ArCoreMetricDepthRegistrationAdapter {
    fun observe(
        frame: Frame,
        rawObservation: D45UnregisteredRawMetricDepthFrame,
        detectorRotationDegrees: Int
    ): D45ArCoreRegistrationObservationResult {
        if (frame.timestamp != rawObservation.sourceFrame.capturedAtNs) {
            return unavailable(D45ArCoreRegistrationFailure.FRAME_TIMESTAMP_MISMATCH)
        }
        val intrinsics = frame.camera.imageIntrinsics
        val dimensions = intrinsics.imageDimensions
        if (
            dimensions[0] != rawObservation.sourceImageIntrinsics.imageWidthPx ||
            dimensions[1] != rawObservation.sourceImageIntrinsics.imageHeightPx
        ) {
            return unavailable(D45ArCoreRegistrationFailure.CAMERA_DIMENSION_MISMATCH)
        }
        val cameraSize = FrameSize(dimensions[0], dimensions[1])
        val depthSize = FrameSize(
            rawObservation.raster.widthPx,
            rawObservation.raster.heightPx
        )
        val cameraPoints = probePoints(cameraSize)
        val inputCoordinates = FloatArray(cameraPoints.size * COORDINATE_COMPONENTS)
        cameraPoints.forEachIndexed { index, point ->
            inputCoordinates[index * COORDINATE_COMPONENTS] = point.x
            inputCoordinates[index * COORDINATE_COMPONENTS + 1] = point.y
        }
        val textureCoordinates = FloatArray(inputCoordinates.size)
        try {
            frame.transformCoordinates2d(
                Coordinates2d.IMAGE_PIXELS,
                inputCoordinates,
                Coordinates2d.TEXTURE_NORMALIZED,
                textureCoordinates
            )
        } catch (error: RuntimeException) {
            return unavailable(
                D45ArCoreRegistrationFailure.COORDINATE_TRANSFORM_FAILED,
                detail = "${error.javaClass.name}:${error.message.orEmpty()}"
            )
        }
        if (textureCoordinates.any { !it.isFinite() }) {
            return unavailable(
                D45ArCoreRegistrationFailure.COORDINATE_TRANSFORM_FAILED,
                detail = "ARCore returned non-finite texture coordinates"
            )
        }
        val correspondences = cameraPoints.indices.map { index ->
            D45CoordinateCorrespondence(
                cameraImagePoint = cameraPoints[index],
                rawDepthPoint = MetricImagePoint(
                    x = textureCoordinates[index * COORDINATE_COMPONENTS] * depthSize.width,
                    y = textureCoordinates[index * COORDINATE_COMPONENTS + 1] * depthSize.height
                )
            )
        }
        return when (
            val result = D45MetricDepthRegistrationFactory.create(
                cameraImageSize = cameraSize,
                rawDepthSize = depthSize,
                detectorRotationDegrees = detectorRotationDegrees,
                correspondences = correspondences
            )
        ) {
            is D45MetricDepthRegistrationResult.Available ->
                D45ArCoreRegistrationObservationResult.Available(
                    D45FrameBoundMetricDepthRegistration(
                        sourceFrameId = rawObservation.sourceFrame.frameId,
                        sourceCapturedAtNs = rawObservation.sourceFrame.capturedAtNs,
                        transform = result.registration
                    )
                )

            is D45MetricDepthRegistrationResult.Unavailable ->
                unavailable(
                    failure = D45ArCoreRegistrationFailure.REGISTRATION_REJECTED,
                    registrationFailure = result.failure,
                    detail = result.detail
                )
        }
    }

    private fun probePoints(size: FrameSize) = listOf(
        MetricImagePoint(0f, 0f),
        MetricImagePoint(size.width.toFloat(), 0f),
        MetricImagePoint(0f, size.height.toFloat()),
        MetricImagePoint(size.width.toFloat(), size.height.toFloat()),
        MetricImagePoint(size.width / 2f, size.height / 2f),
        MetricImagePoint(size.width / 2f, 0f),
        MetricImagePoint(size.width / 2f, size.height.toFloat()),
        MetricImagePoint(0f, size.height / 2f),
        MetricImagePoint(size.width.toFloat(), size.height / 2f)
    )

    private fun unavailable(
        failure: D45ArCoreRegistrationFailure,
        registrationFailure: D45MetricDepthRegistrationFailure? = null,
        detail: String? = null
    ) = D45ArCoreRegistrationObservationResult.Unavailable(
        failure = failure,
        registrationFailure = registrationFailure,
        detail = detail
    )

    private companion object {
        const val COORDINATE_COMPONENTS = 2
    }
}
