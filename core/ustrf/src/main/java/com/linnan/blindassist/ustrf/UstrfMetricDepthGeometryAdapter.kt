package com.linnan.blindassist.ustrf

import kotlin.math.abs

/**
 * A metric depth raster after the source-specific registration Adapter has placed it in the
 * camera-intrinsics pixel grid.  This contract intentionally does not implement registration:
 * callers must bind the supplied raster to the independently verified transform receipt.
 */
data class UstrfRegisteredMetricDepthImage(
    val sourceFrame: UstrfFrameStamp,
    val widthPx: Int,
    val heightPx: Int,
    val depthCoordinateFrame: String,
    val registrationTransformId: String,
    /** One non-negative millimetre value for each registered pixel; zero is an invalid sample. */
    val depthMillimeters: IntArray,
    /** One [0, 1] measurement confidence for each registered pixel. */
    val confidence: FloatArray,
    val validUntilNs: Long
) {
    init {
        require(widthPx > 0 && heightPx > 0)
        require(depthCoordinateFrame.isNotBlank() && registrationTransformId.isNotBlank())
        require(depthMillimeters.size == widthPx * heightPx)
        require(confidence.size == widthPx * heightPx && confidence.all { it in 0f..1f })
        require(depthMillimeters.all { it >= 0 })
        require(validUntilNs >= sourceFrame.capturedAtNs)
    }
}

/** A body-local plane equation `normal dot point + offset = 0`, with y conventionally up. */
data class UstrfVerifiedGroundPlaneReceipt(
    val sourceFrame: UstrfFrameStamp,
    val bodyFrame: String,
    val normal: UstrfVector3,
    val offsetMeters: Float,
    val confidence: Float,
    val independentlyVerified: Boolean,
    val validUntilNs: Long
) {
    init {
        require(bodyFrame.isNotBlank())
        require(normal.x.isFinite() && normal.y.isFinite() && normal.z.isFinite())
        require(offsetMeters.isFinite() && confidence in 0f..1f)
        require(validUntilNs >= sourceFrame.capturedAtNs)
    }
}

data class UstrfMetricDepthGeometryAdapterConfig(
    val sampleStridePx: Int = 4,
    val minimumDepthMeters: Float = .20f,
    val maximumDepthMeters: Float = 5f,
    val minimumSampleConfidence: Float = .70f,
    val groundToleranceMeters: Float = .12f,
    val lowerBodyMaximumMeters: Float = 1.35f,
    val headMinimumMeters: Float = 1.35f
) {
    init {
        require(sampleStridePx >= 1)
        require(minimumDepthMeters > 0f && maximumDepthMeters >= minimumDepthMeters)
        require(minimumSampleConfidence in 0f..1f)
        require(groundToleranceMeters > 0f && lowerBodyMaximumMeters > groundToleranceMeters)
        require(headMinimumMeters >= lowerBodyMaximumMeters)
    }
}

enum class UstrfMetricDepthGeometryAdapterFailure {
    ADMISSION_FRAME_MISMATCH,
    DEPTH_FRAME_MISMATCH,
    DEPTH_DIMENSION_MISMATCH,
    DEPTH_REGISTRATION_MISMATCH,
    DEPTH_COORDINATE_FRAME_MISMATCH,
    DEPTH_STALE,
    GROUND_FRAME_MISMATCH,
    GROUND_BODY_FRAME_MISMATCH,
    GROUND_NOT_INDEPENDENTLY_VERIFIED,
    GROUND_STALE,
    GROUND_DEGENERATE
}

sealed interface UstrfMetricDepthGeometryAdapterResult {
    data class Available(
        val packet: UstrfGeometryPacket,
        val sampledPixelCount: Int,
        val admittedEvidenceCount: Int
    ) : UstrfMetricDepthGeometryAdapterResult

    data class Unavailable(val failure: UstrfMetricDepthGeometryAdapterFailure) : UstrfMetricDepthGeometryAdapterResult
}

/**
 * Offline/theory-only calibrated-depth front end for the USTRF geometry projector.
 *
 * It projects registered metric samples through pinned intrinsics and a full camera-to-body
 * transform, then uses an independently verified body-local ground plane to form traversable,
 * lower-body and head-obstacle evidence.  A missing or invalid depth sample is deliberately
 * ignored: absence of geometry is not a drop observation.  Detecting a drop requires a separate
 * negative-observation/visibility model and is therefore outside this adapter.
 */
class UstrfMetricDepthGeometryAdapter(
    private val config: UstrfMetricDepthGeometryAdapterConfig = UstrfMetricDepthGeometryAdapterConfig()
) {
    fun project(
        admission: UstrfMetricGeometryProjectionAdmission.Available,
        depth: UstrfRegisteredMetricDepthImage,
        intrinsics: UstrfCameraIntrinsicsReceipt,
        extrinsics: UstrfCameraBodyFullExtrinsicsReceipt,
        ground: UstrfVerifiedGroundPlaneReceipt,
        producedAtNs: Long
    ): UstrfMetricDepthGeometryAdapterResult {
        if (admission.sourceFrame != depth.sourceFrame) return unavailable(UstrfMetricDepthGeometryAdapterFailure.ADMISSION_FRAME_MISMATCH)
        if (depth.sourceFrame != ground.sourceFrame) return unavailable(UstrfMetricDepthGeometryAdapterFailure.GROUND_FRAME_MISMATCH)
        if (depth.sourceFrame.coordinateFrame != intrinsics.cameraFrame || depth.sourceFrame.coordinateFrame != extrinsics.cameraFrame) {
            return unavailable(UstrfMetricDepthGeometryAdapterFailure.DEPTH_FRAME_MISMATCH)
        }
        if (depth.widthPx != intrinsics.imageWidthPx || depth.heightPx != intrinsics.imageHeightPx) {
            return unavailable(UstrfMetricDepthGeometryAdapterFailure.DEPTH_DIMENSION_MISMATCH)
        }
        if (depth.registrationTransformId != admission.registrationTransformId) {
            return unavailable(UstrfMetricDepthGeometryAdapterFailure.DEPTH_REGISTRATION_MISMATCH)
        }
        if (depth.depthCoordinateFrame != admission.depthCoordinateFrame) {
            return unavailable(UstrfMetricDepthGeometryAdapterFailure.DEPTH_COORDINATE_FRAME_MISMATCH)
        }
        if (producedAtNs > depth.validUntilNs || producedAtNs > admission.validUntilNs) {
            return unavailable(UstrfMetricDepthGeometryAdapterFailure.DEPTH_STALE)
        }
        if (ground.bodyFrame != admission.bodyFrame) return unavailable(UstrfMetricDepthGeometryAdapterFailure.GROUND_BODY_FRAME_MISMATCH)
        if (!ground.independentlyVerified) return unavailable(UstrfMetricDepthGeometryAdapterFailure.GROUND_NOT_INDEPENDENTLY_VERIFIED)
        if (producedAtNs > ground.validUntilNs) return unavailable(UstrfMetricDepthGeometryAdapterFailure.GROUND_STALE)
        val normalLengthSquared = dot(ground.normal, ground.normal)
        if (normalLengthSquared < .99f || normalLengthSquared > 1.01f) return unavailable(UstrfMetricDepthGeometryAdapterFailure.GROUND_DEGENERATE)

        var sampled = 0
        val evidence = mutableListOf<UstrfMetricGeometryEvidence>()
        for (vertical in 0 until depth.heightPx step config.sampleStridePx) {
            for (horizontal in 0 until depth.widthPx step config.sampleStridePx) {
                sampled += 1
                val index = vertical * depth.widthPx + horizontal
                val depthMeters = depth.depthMillimeters[index] / 1_000f
                val sampleConfidence = depth.confidence[index]
                if (depthMeters !in config.minimumDepthMeters..config.maximumDepthMeters || sampleConfidence < config.minimumSampleConfidence) continue
                // Camera ray uses x=right, y=up, z=forward; image v increases downward.
                val cameraPoint = UstrfVector3(
                    (horizontal - intrinsics.principalXpx) * depthMeters / intrinsics.focalXpx,
                    -(vertical - intrinsics.principalYpx) * depthMeters / intrinsics.focalYpx,
                    depthMeters
                )
                val bodyPoint = rotate(cameraPoint, extrinsics.cameraToBodyQuaternionXyzw) + extrinsics.cameraToBodyTranslationM
                if (bodyPoint.z !in 0f..config.maximumDepthMeters) continue
                val planeDistance = dot(ground.normal, bodyPoint) + ground.offsetMeters
                val kindAndBand = when {
                    abs(planeDistance) <= config.groundToleranceMeters -> UstrfGeometryKind.TRAVERSABLE to UstrfHeightBand.GROUND
                    planeDistance > config.groundToleranceMeters && planeDistance < config.lowerBodyMaximumMeters ->
                        UstrfGeometryKind.OCCUPIED to UstrfHeightBand.LOWER_BODY
                    planeDistance >= config.headMinimumMeters -> UstrfGeometryKind.HEAD_OBSTACLE to UstrfHeightBand.HEAD
                    else -> null
                } ?: continue
                evidence += UstrfMetricGeometryEvidence(
                    forwardMeters = bodyPoint.z,
                    lateralMeters = bodyPoint.x,
                    heightBand = kindAndBand.second,
                    kind = kindAndBand.first,
                    confidence = minOf(sampleConfidence, ground.confidence),
                    source = "metric-depth-ground-plane",
                    validUntilNs = minOf(depth.validUntilNs, ground.validUntilNs, admission.validUntilNs)
                )
            }
        }
        val validUntilNs = minOf(depth.validUntilNs, ground.validUntilNs, admission.validUntilNs)
        return UstrfMetricDepthGeometryAdapterResult.Available(
            packet = UstrfGeometryPacket(depth.sourceFrame, producedAtNs, validUntilNs, UstrfDepthScale.METRIC, evidence),
            sampledPixelCount = sampled,
            admittedEvidenceCount = evidence.size
        )
    }

    private fun unavailable(failure: UstrfMetricDepthGeometryAdapterFailure) = UstrfMetricDepthGeometryAdapterResult.Unavailable(failure)

    private fun dot(left: UstrfVector3, right: UstrfVector3): Float =
        left.x * right.x + left.y * right.y + left.z * right.z

    private fun rotate(point: UstrfVector3, quaternion: FloatArray): UstrfVector3 {
        val qx = quaternion[0]
        val qy = quaternion[1]
        val qz = quaternion[2]
        val qw = quaternion[3]
        // q * point * conjugate(q), expanded to avoid an allocation-heavy quaternion type.
        val tx = 2f * (qy * point.z - qz * point.y)
        val ty = 2f * (qz * point.x - qx * point.z)
        val tz = 2f * (qx * point.y - qy * point.x)
        return UstrfVector3(
            point.x + qw * tx + (qy * tz - qz * ty),
            point.y + qw * ty + (qz * tx - qx * tz),
            point.z + qw * tz + (qx * ty - qy * tx)
        )
    }
}
