package com.linnan.blindassist.ustrf

import kotlin.math.abs
import kotlin.math.sqrt

/** The unit must be explicit before raw depth is allowed anywhere near metric projection. */
enum class UstrfRawDepthUnit { MILLIMETERS, UNKNOWN }

/**
 * Metadata-only receipt for a raw-depth/confidence pair. Pixel sampling and scene interpretation
 * remain Adapter responsibilities; this receipt establishes only whether those pixels could be
 * considered current-frame metric candidates.
 */
data class UstrfRawDepthCandidateReceipt(
    val sourceFrame: UstrfFrameStamp,
    val depthTimestampNs: Long,
    val confidenceTimestampNs: Long,
    val depthCoordinateFrame: String,
    val unit: UstrfRawDepthUnit,
    val validUntilNs: Long
) {
    init {
        require(depthTimestampNs >= 0L && confidenceTimestampNs >= 0L)
        require(depthCoordinateFrame.isNotBlank())
        require(validUntilNs >= sourceFrame.capturedAtNs)
    }
}

/** A pinned pinhole-camera calibration receipt; numbers alone are not evidence of verification. */
data class UstrfCameraIntrinsicsReceipt(
    val cameraFrame: String,
    val calibrationVersion: String,
    val imageWidthPx: Int,
    val imageHeightPx: Int,
    val focalXpx: Float,
    val focalYpx: Float,
    val principalXpx: Float,
    val principalYpx: Float,
    val verifiedAtNs: Long,
    val validUntilNs: Long,
    val confidence: Float,
    val independentlyVerified: Boolean
) {
    init {
        require(cameraFrame.isNotBlank() && calibrationVersion.isNotBlank())
        require(imageWidthPx > 0 && imageHeightPx > 0)
        require(focalXpx.isFinite() && focalXpx > 0f && focalYpx.isFinite() && focalYpx > 0f)
        require(principalXpx.isFinite() && principalYpx.isFinite())
        require(verifiedAtNs >= 0L && validUntilNs >= verifiedAtNs)
        require(confidence in 0f..1f)
    }
}

/**
 * Registration is intentionally separate from intrinsics: a raw depth image can use another
 * resolution/orientation. A matching calibration version is insufficient without a verified map.
 */
data class UstrfDepthCameraRegistrationReceipt(
    val depthCoordinateFrame: String,
    val cameraFrame: String,
    val calibrationVersion: String,
    val transformId: String,
    val verifiedAtNs: Long,
    val validUntilNs: Long,
    val confidence: Float,
    val independentlyVerified: Boolean
) {
    init {
        require(depthCoordinateFrame.isNotBlank() && cameraFrame.isNotBlank())
        require(calibrationVersion.isNotBlank() && transformId.isNotBlank())
        require(verifiedAtNs >= 0L && validUntilNs >= verifiedAtNs)
        require(confidence in 0f..1f)
    }
}

/**
 * Full SE(3) mount calibration required for height/ground geometry. This is deliberately not the
 * yaw-only [UstrfCameraBodyExtrinsicsReceipt] used by the planar pose seam.
 */
data class UstrfCameraBodyFullExtrinsicsReceipt(
    val cameraFrame: String,
    val bodyFrame: String,
    val cameraToBodyTranslationM: UstrfVector3,
    /** Unit quaternion in x, y, z, w order. */
    val cameraToBodyQuaternionXyzw: FloatArray,
    val calibrationId: String,
    val verifiedAtNs: Long,
    val validUntilNs: Long,
    val confidence: Float,
    val independentlyVerified: Boolean
) {
    init {
        require(cameraFrame.isNotBlank() && bodyFrame.isNotBlank() && calibrationId.isNotBlank())
        require(cameraToBodyQuaternionXyzw.size == 4 && cameraToBodyQuaternionXyzw.all { it.isFinite() })
        val norm = sqrt(cameraToBodyQuaternionXyzw.sumOf { (it * it).toDouble() }).toFloat()
        require(abs(norm - 1f) <= .01f) { "camera-to-body quaternion must be normalized" }
        require(verifiedAtNs >= 0L && validUntilNs >= verifiedAtNs)
        require(confidence in 0f..1f)
    }
}

enum class UstrfMetricGeometryAdmissionFailure {
    SOURCE_FRAME_MISMATCH,
    RAW_DEPTH_TIMESTAMP_MISMATCH,
    CONFIDENCE_TIMESTAMP_MISMATCH,
    RAW_DEPTH_UNIT_NOT_METRIC,
    RAW_DEPTH_STALE,
    INTRINSICS_CAMERA_FRAME_MISMATCH,
    INTRINSICS_CALIBRATION_VERSION_MISMATCH,
    INTRINSICS_NOT_INDEPENDENTLY_VERIFIED,
    INTRINSICS_STALE,
    INTRINSICS_LOW_CONFIDENCE,
    REGISTRATION_CAMERA_FRAME_MISMATCH,
    REGISTRATION_DEPTH_FRAME_MISMATCH,
    REGISTRATION_CALIBRATION_VERSION_MISMATCH,
    REGISTRATION_NOT_INDEPENDENTLY_VERIFIED,
    REGISTRATION_STALE,
    REGISTRATION_LOW_CONFIDENCE,
    EXTRINSICS_CAMERA_FRAME_MISMATCH,
    EXTRINSICS_NOT_INDEPENDENTLY_VERIFIED,
    EXTRINSICS_STALE,
    EXTRINSICS_LOW_CONFIDENCE
}

sealed interface UstrfMetricGeometryProjectionAdmission {
    /**
     * The inputs are eligible for a future pixel-to-body projection Adapter only. No scene labels,
     * ground plane, obstacle, drop, traversability, or [UstrfGeometryPacket] is created here.
     */
    data class Available(
        val sourceFrame: UstrfFrameStamp,
        val depthCoordinateFrame: String,
        val cameraFrame: String,
        val bodyFrame: String,
        val registrationTransformId: String,
        val validUntilNs: Long
    ) : UstrfMetricGeometryProjectionAdmission

    data class Unavailable(val failure: UstrfMetricGeometryAdmissionFailure) : UstrfMetricGeometryProjectionAdmission
}

/**
 * Fail-closed admission for a future metric geometry Adapter. It keeps current-frame raw depth,
 * calibration receipts, and a complete mount transform atomic, while deliberately leaving scene
 * interpretation and safety geometry to a separately tested Adapter.
 */
class UstrfMetricGeometryReceiptPromoter(
    private val minimumIntrinsicsConfidence: Float = .95f,
    private val minimumRegistrationConfidence: Float = .95f,
    private val minimumExtrinsicsConfidence: Float = .95f
) {
    init {
        require(minimumIntrinsicsConfidence in 0f..1f)
        require(minimumRegistrationConfidence in 0f..1f)
        require(minimumExtrinsicsConfidence in 0f..1f)
    }

    fun admit(
        rawDepth: UstrfRawDepthCandidateReceipt,
        capture: UstrfCaptureReceipt,
        intrinsics: UstrfCameraIntrinsicsReceipt,
        registration: UstrfDepthCameraRegistrationReceipt,
        extrinsics: UstrfCameraBodyFullExtrinsicsReceipt,
        decisionAtNs: Long
    ): UstrfMetricGeometryProjectionAdmission {
        val frame = capture.frame
        if (rawDepth.sourceFrame != frame) return unavailable(UstrfMetricGeometryAdmissionFailure.SOURCE_FRAME_MISMATCH)
        if (rawDepth.depthTimestampNs != frame.capturedAtNs) return unavailable(UstrfMetricGeometryAdmissionFailure.RAW_DEPTH_TIMESTAMP_MISMATCH)
        if (rawDepth.confidenceTimestampNs != frame.capturedAtNs) return unavailable(UstrfMetricGeometryAdmissionFailure.CONFIDENCE_TIMESTAMP_MISMATCH)
        if (rawDepth.unit != UstrfRawDepthUnit.MILLIMETERS) return unavailable(UstrfMetricGeometryAdmissionFailure.RAW_DEPTH_UNIT_NOT_METRIC)
        if (decisionAtNs > rawDepth.validUntilNs) return unavailable(UstrfMetricGeometryAdmissionFailure.RAW_DEPTH_STALE)

        if (intrinsics.cameraFrame != frame.coordinateFrame) return unavailable(UstrfMetricGeometryAdmissionFailure.INTRINSICS_CAMERA_FRAME_MISMATCH)
        if (intrinsics.calibrationVersion != capture.calibrationVersion) return unavailable(UstrfMetricGeometryAdmissionFailure.INTRINSICS_CALIBRATION_VERSION_MISMATCH)
        if (!intrinsics.independentlyVerified) return unavailable(UstrfMetricGeometryAdmissionFailure.INTRINSICS_NOT_INDEPENDENTLY_VERIFIED)
        if (decisionAtNs > intrinsics.validUntilNs) return unavailable(UstrfMetricGeometryAdmissionFailure.INTRINSICS_STALE)
        if (intrinsics.confidence < minimumIntrinsicsConfidence) return unavailable(UstrfMetricGeometryAdmissionFailure.INTRINSICS_LOW_CONFIDENCE)

        if (registration.cameraFrame != frame.coordinateFrame) return unavailable(UstrfMetricGeometryAdmissionFailure.REGISTRATION_CAMERA_FRAME_MISMATCH)
        if (registration.depthCoordinateFrame != rawDepth.depthCoordinateFrame) return unavailable(UstrfMetricGeometryAdmissionFailure.REGISTRATION_DEPTH_FRAME_MISMATCH)
        if (registration.calibrationVersion != capture.calibrationVersion) return unavailable(UstrfMetricGeometryAdmissionFailure.REGISTRATION_CALIBRATION_VERSION_MISMATCH)
        if (!registration.independentlyVerified) return unavailable(UstrfMetricGeometryAdmissionFailure.REGISTRATION_NOT_INDEPENDENTLY_VERIFIED)
        if (decisionAtNs > registration.validUntilNs) return unavailable(UstrfMetricGeometryAdmissionFailure.REGISTRATION_STALE)
        if (registration.confidence < minimumRegistrationConfidence) return unavailable(UstrfMetricGeometryAdmissionFailure.REGISTRATION_LOW_CONFIDENCE)

        if (extrinsics.cameraFrame != frame.coordinateFrame) return unavailable(UstrfMetricGeometryAdmissionFailure.EXTRINSICS_CAMERA_FRAME_MISMATCH)
        if (!extrinsics.independentlyVerified) return unavailable(UstrfMetricGeometryAdmissionFailure.EXTRINSICS_NOT_INDEPENDENTLY_VERIFIED)
        if (decisionAtNs > extrinsics.validUntilNs) return unavailable(UstrfMetricGeometryAdmissionFailure.EXTRINSICS_STALE)
        if (extrinsics.confidence < minimumExtrinsicsConfidence) return unavailable(UstrfMetricGeometryAdmissionFailure.EXTRINSICS_LOW_CONFIDENCE)

        return UstrfMetricGeometryProjectionAdmission.Available(
            sourceFrame = frame,
            depthCoordinateFrame = rawDepth.depthCoordinateFrame,
            cameraFrame = intrinsics.cameraFrame,
            bodyFrame = extrinsics.bodyFrame,
            registrationTransformId = registration.transformId,
            validUntilNs = minOf(rawDepth.validUntilNs, intrinsics.validUntilNs, registration.validUntilNs, extrinsics.validUntilNs)
        )
    }

    private fun unavailable(failure: UstrfMetricGeometryAdmissionFailure) = UstrfMetricGeometryProjectionAdmission.Unavailable(failure)
}
