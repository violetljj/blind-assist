package com.linnan.blindassist.hftf.metricdepth

import com.linnan.blindassist.model.FrameSize
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.round

data class MetricImagePoint(
    val x: Float,
    val y: Float
) {
    init {
        require(x.isFinite() && y.isFinite())
    }
}

/**
 * Continuous 2D affine transform. Pixel boxes use edge coordinates; raster samples use centres.
 */
data class MetricAffineTransform2D(
    val m00: Float,
    val m01: Float,
    val m02: Float,
    val m10: Float,
    val m11: Float,
    val m12: Float
) {
    init {
        require(listOf(m00, m01, m02, m10, m11, m12).all(Float::isFinite))
        require(abs(determinant()) > MIN_ABSOLUTE_DETERMINANT) {
            "affine transform must be invertible"
        }
    }

    fun map(point: MetricImagePoint) = MetricImagePoint(
        x = m00 * point.x + m01 * point.y + m02,
        y = m10 * point.x + m11 * point.y + m12
    )

    /** Returns a transform that applies this transform and then [next]. */
    fun then(next: MetricAffineTransform2D) = MetricAffineTransform2D(
        m00 = next.m00 * m00 + next.m01 * m10,
        m01 = next.m00 * m01 + next.m01 * m11,
        m02 = next.m00 * m02 + next.m01 * m12 + next.m02,
        m10 = next.m10 * m00 + next.m11 * m10,
        m11 = next.m10 * m01 + next.m11 * m11,
        m12 = next.m10 * m02 + next.m11 * m12 + next.m12
    )

    fun inverse(): MetricAffineTransform2D {
        val determinant = determinant()
        val inverse00 = m11 / determinant
        val inverse01 = -m01 / determinant
        val inverse10 = -m10 / determinant
        val inverse11 = m00 / determinant
        return MetricAffineTransform2D(
            m00 = inverse00,
            m01 = inverse01,
            m02 = -(inverse00 * m02 + inverse01 * m12),
            m10 = inverse10,
            m11 = inverse11,
            m12 = -(inverse10 * m02 + inverse11 * m12)
        )
    }

    private fun determinant(): Float = m00 * m11 - m01 * m10

    companion object {
        val IDENTITY = MetricAffineTransform2D(
            m00 = 1f,
            m01 = 0f,
            m02 = 0f,
            m10 = 0f,
            m11 = 1f,
            m12 = 0f
        )

        private const val MIN_ABSOLUTE_DETERMINANT = 1e-9f
    }
}

data class D45CoordinateCorrespondence(
    val cameraImagePoint: MetricImagePoint,
    val rawDepthPoint: MetricImagePoint
)

data class MetricDepthRegistrationTransform(
    val detectorDisplaySize: FrameSize,
    val cameraImageSize: FrameSize,
    val rawDepthSize: FrameSize,
    val detectorRotationDegrees: Int,
    val detectorToCameraImage: MetricAffineTransform2D,
    val cameraImageToRawDepth: MetricAffineTransform2D,
    val maximumFitResidualPx: Float,
    val transformId: String
) {
    init {
        require(detectorDisplaySize.width > 0 && detectorDisplaySize.height > 0)
        require(cameraImageSize.width > 0 && cameraImageSize.height > 0)
        require(rawDepthSize.width > 0 && rawDepthSize.height > 0)
        require(detectorRotationDegrees in VALID_ROTATIONS)
        require(maximumFitResidualPx.isFinite() && maximumFitResidualPx >= 0f)
        require(transformId.isNotBlank())
    }

    val detectorToRawDepth: MetricAffineTransform2D =
        detectorToCameraImage.then(cameraImageToRawDepth)

    val rawDepthToDetector: MetricAffineTransform2D = detectorToRawDepth.inverse()

    companion object {
        val VALID_ROTATIONS = setOf(0, 90, 180, 270)
    }
}

data class D45FrameBoundMetricDepthRegistration(
    val sourceFrameId: Long,
    val sourceCapturedAtNs: Long,
    val transform: MetricDepthRegistrationTransform
) {
    init {
        require(sourceFrameId >= 0L && sourceCapturedAtNs >= 0L)
    }
}

enum class D45MetricDepthRegistrationFailure {
    INVALID_DIMENSIONS,
    INVALID_DETECTOR_ROTATION,
    INSUFFICIENT_CORRESPONDENCES,
    DEGENERATE_BASIS,
    NONFINITE_COORDINATES,
    AFFINE_RESIDUAL_ABOVE_TOLERANCE
}

sealed interface D45MetricDepthRegistrationResult {
    data class Available(
        val registration: MetricDepthRegistrationTransform
    ) : D45MetricDepthRegistrationResult

    data class Unavailable(
        val failure: D45MetricDepthRegistrationFailure,
        val detail: String? = null
    ) : D45MetricDepthRegistrationResult
}

/**
 * Builds the exact detector-display -> CPU-camera -> raw-depth transform.
 *
 * ARCore supplies CPU-image -> depth correspondences. Detector rotation is kept separate because
 * BlindAssist detections live in the full, display-oriented camera image, not in an AR viewport.
 */
object D45MetricDepthRegistrationFactory {
    fun create(
        cameraImageSize: FrameSize,
        rawDepthSize: FrameSize,
        detectorRotationDegrees: Int,
        correspondences: List<D45CoordinateCorrespondence>,
        maximumAllowedResidualPx: Float = DEFAULT_MAXIMUM_RESIDUAL_PX
    ): D45MetricDepthRegistrationResult {
        if (!cameraImageSize.isPositive() || !rawDepthSize.isPositive()) {
            return unavailable(D45MetricDepthRegistrationFailure.INVALID_DIMENSIONS)
        }
        if (detectorRotationDegrees !in MetricDepthRegistrationTransform.VALID_ROTATIONS) {
            return unavailable(D45MetricDepthRegistrationFailure.INVALID_DETECTOR_ROTATION)
        }
        if (correspondences.size < MINIMUM_CORRESPONDENCES) {
            return unavailable(D45MetricDepthRegistrationFailure.INSUFFICIENT_CORRESPONDENCES)
        }
        if (
            correspondences.any {
                !it.cameraImagePoint.x.isFinite() ||
                    !it.cameraImagePoint.y.isFinite() ||
                    !it.rawDepthPoint.x.isFinite() ||
                    !it.rawDepthPoint.y.isFinite()
            }
        ) {
            return unavailable(D45MetricDepthRegistrationFailure.NONFINITE_COORDINATES)
        }

        val basis = selectNondegenerateBasis(correspondences)
            ?: return unavailable(D45MetricDepthRegistrationFailure.DEGENERATE_BASIS)
        val cameraToDepth = try {
            affineFromThreeCorrespondences(basis[0], basis[1], basis[2])
        } catch (error: IllegalArgumentException) {
            return unavailable(
                D45MetricDepthRegistrationFailure.DEGENERATE_BASIS,
                error.message
            )
        }
        val maximumResidual = correspondences.maxOf { correspondence ->
            val mapped = cameraToDepth.map(correspondence.cameraImagePoint)
            hypot(
                (mapped.x - correspondence.rawDepthPoint.x).toDouble(),
                (mapped.y - correspondence.rawDepthPoint.y).toDouble()
            ).toFloat()
        }
        if (
            !maximumAllowedResidualPx.isFinite() ||
            maximumAllowedResidualPx < 0f ||
            maximumResidual > maximumAllowedResidualPx
        ) {
            return unavailable(
                D45MetricDepthRegistrationFailure.AFFINE_RESIDUAL_ABOVE_TOLERANCE,
                "maximumResidualPx=$maximumResidual,allowedPx=$maximumAllowedResidualPx"
            )
        }

        val detectorToCamera = detectorToCameraTransform(
            cameraImageSize = cameraImageSize,
            rotationDegrees = detectorRotationDegrees
        )
        val detectorSize = if (detectorRotationDegrees % 180 == 0) {
            cameraImageSize
        } else {
            FrameSize(cameraImageSize.height, cameraImageSize.width)
        }
        val transformId = transformId(
            detectorDisplaySize = detectorSize,
            cameraImageSize = cameraImageSize,
            rawDepthSize = rawDepthSize,
            detectorRotationDegrees = detectorRotationDegrees,
            detectorToCamera = detectorToCamera,
            cameraToDepth = cameraToDepth
        )
        return D45MetricDepthRegistrationResult.Available(
            MetricDepthRegistrationTransform(
                detectorDisplaySize = detectorSize,
                cameraImageSize = cameraImageSize,
                rawDepthSize = rawDepthSize,
                detectorRotationDegrees = detectorRotationDegrees,
                detectorToCameraImage = detectorToCamera,
                cameraImageToRawDepth = cameraToDepth,
                maximumFitResidualPx = maximumResidual,
                transformId = transformId
            )
        )
    }

    private fun selectNondegenerateBasis(
        correspondences: List<D45CoordinateCorrespondence>
    ): List<D45CoordinateCorrespondence>? {
        for (first in 0 until correspondences.size - 2) {
            for (second in first + 1 until correspondences.size - 1) {
                for (third in second + 1 until correspondences.size) {
                    val candidate = listOf(
                        correspondences[first],
                        correspondences[second],
                        correspondences[third]
                    )
                    if (abs(sourceTriangleTwiceArea(candidate)) > MINIMUM_BASIS_AREA_PX2) {
                        return candidate
                    }
                }
            }
        }
        return null
    }

    private fun sourceTriangleTwiceArea(points: List<D45CoordinateCorrespondence>): Float {
        val first = points[0].cameraImagePoint
        val second = points[1].cameraImagePoint
        val third = points[2].cameraImagePoint
        return (second.x - first.x) * (third.y - first.y) -
            (third.x - first.x) * (second.y - first.y)
    }

    private fun affineFromThreeCorrespondences(
        first: D45CoordinateCorrespondence,
        second: D45CoordinateCorrespondence,
        third: D45CoordinateCorrespondence
    ): MetricAffineTransform2D {
        val x0 = first.cameraImagePoint.x
        val y0 = first.cameraImagePoint.y
        val x1 = second.cameraImagePoint.x
        val y1 = second.cameraImagePoint.y
        val x2 = third.cameraImagePoint.x
        val y2 = third.cameraImagePoint.y
        val determinant =
            x0 * (y1 - y2) +
                x1 * (y2 - y0) +
                x2 * (y0 - y1)
        require(abs(determinant) > MINIMUM_BASIS_AREA_PX2)

        fun coefficients(v0: Float, v1: Float, v2: Float): FloatArray {
            val a = (
                v0 * (y1 - y2) +
                    v1 * (y2 - y0) +
                    v2 * (y0 - y1)
                ) / determinant
            val b = (
                v0 * (x2 - x1) +
                    v1 * (x0 - x2) +
                    v2 * (x1 - x0)
                ) / determinant
            val c = (
                v0 * (x1 * y2 - x2 * y1) +
                    v1 * (x2 * y0 - x0 * y2) +
                    v2 * (x0 * y1 - x1 * y0)
                ) / determinant
            return floatArrayOf(a, b, c)
        }

        val x = coefficients(
            first.rawDepthPoint.x,
            second.rawDepthPoint.x,
            third.rawDepthPoint.x
        )
        val y = coefficients(
            first.rawDepthPoint.y,
            second.rawDepthPoint.y,
            third.rawDepthPoint.y
        )
        return MetricAffineTransform2D(
            m00 = x[0],
            m01 = x[1],
            m02 = x[2],
            m10 = y[0],
            m11 = y[1],
            m12 = y[2]
        )
    }

    private fun detectorToCameraTransform(
        cameraImageSize: FrameSize,
        rotationDegrees: Int
    ): MetricAffineTransform2D = when (rotationDegrees) {
        0 -> MetricAffineTransform2D.IDENTITY
        90 -> MetricAffineTransform2D(
            m00 = 0f,
            m01 = 1f,
            m02 = 0f,
            m10 = -1f,
            m11 = 0f,
            m12 = cameraImageSize.height.toFloat()
        )
        180 -> MetricAffineTransform2D(
            m00 = -1f,
            m01 = 0f,
            m02 = cameraImageSize.width.toFloat(),
            m10 = 0f,
            m11 = -1f,
            m12 = cameraImageSize.height.toFloat()
        )
        270 -> MetricAffineTransform2D(
            m00 = 0f,
            m01 = -1f,
            m02 = cameraImageSize.width.toFloat(),
            m10 = 1f,
            m11 = 0f,
            m12 = 0f
        )
        else -> error("rotation was validated before transform construction")
    }

    private fun transformId(
        detectorDisplaySize: FrameSize,
        cameraImageSize: FrameSize,
        rawDepthSize: FrameSize,
        detectorRotationDegrees: Int,
        detectorToCamera: MetricAffineTransform2D,
        cameraToDepth: MetricAffineTransform2D
    ): String {
        val values = listOf(
            detectorDisplaySize.width,
            detectorDisplaySize.height,
            cameraImageSize.width,
            cameraImageSize.height,
            rawDepthSize.width,
            rawDepthSize.height,
            detectorRotationDegrees
        )
        val coefficients = listOf(
            detectorToCamera.m00,
            detectorToCamera.m01,
            detectorToCamera.m02,
            detectorToCamera.m10,
            detectorToCamera.m11,
            detectorToCamera.m12,
            cameraToDepth.m00,
            cameraToDepth.m01,
            cameraToDepth.m02,
            cameraToDepth.m10,
            cameraToDepth.m11,
            cameraToDepth.m12
        )
        val bytes = ByteBuffer
            .allocate((values.size + coefficients.size) * Int.SIZE_BYTES)
            .order(ByteOrder.BIG_ENDIAN)
        values.forEach(bytes::putInt)
        coefficients.forEach { coefficient ->
            val quantized =
                round(coefficient / TRANSFORM_ID_COEFFICIENT_QUANTUM) *
                    TRANSFORM_ID_COEFFICIENT_QUANTUM
            bytes.putInt(quantized.toRawBits())
        }
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes.array())
        return "d45-arcore-registration-v1:" +
            digest.joinToString("") { "%02x".format(it) }
    }

    private fun FrameSize.isPositive(): Boolean = width > 0 && height > 0

    private fun unavailable(
        failure: D45MetricDepthRegistrationFailure,
        detail: String? = null
    ) = D45MetricDepthRegistrationResult.Unavailable(failure, detail)

    private const val MINIMUM_CORRESPONDENCES = 4
    private const val MINIMUM_BASIS_AREA_PX2 = 1e-3f
    private const val DEFAULT_MAXIMUM_RESIDUAL_PX = 0.25f
    private const val TRANSFORM_ID_COEFFICIENT_QUANTUM = 1e-6f
}

object D45MetricDepthFrameRegistrar {
    fun register(
        frame: D45UnregisteredRawMetricDepthFrame,
        registrationObservation: D45FrameBoundMetricDepthRegistration,
        validUntilNs: Long
    ): RegisteredMetricDepthFrame {
        require(
            registrationObservation.sourceFrameId == frame.sourceFrame.frameId &&
                registrationObservation.sourceCapturedAtNs == frame.sourceFrame.capturedAtNs
        ) {
            "registration observation is not bound to the raw-depth source frame"
        }
        val registration = registrationObservation.transform
        require(
            registration.cameraImageSize.width == frame.sourceImageIntrinsics.imageWidthPx &&
                registration.cameraImageSize.height == frame.sourceImageIntrinsics.imageHeightPx
        ) {
            "registration camera dimensions do not match ARCore image intrinsics"
        }
        require(
            registration.rawDepthSize.width == frame.raster.widthPx &&
                registration.rawDepthSize.height == frame.raster.heightPx
        ) {
            "registration raw-depth dimensions do not match decoded raster"
        }
        return RegisteredMetricDepthFrame(
            sourceFrame = frame.sourceFrame,
            detectorDisplaySize = registration.detectorDisplaySize,
            intrinsics = frame.sourceImageIntrinsics,
            depthMillimeters = frame.raster.depthMillimeters,
            confidence = frame.raster.confidence,
            source = MetricDepthSource.ARCORE_RAW_REGISTERED,
            registration = registration,
            producedAtNs = frame.producedAtNs,
            validUntilNs = validUntilNs
        )
    }
}
