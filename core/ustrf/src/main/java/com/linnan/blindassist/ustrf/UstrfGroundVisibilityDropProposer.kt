package com.linnan.blindassist.ustrf

/** Conservative, metric ground-visibility discontinuity proposer for offline geometry replay. */
data class UstrfGroundVisibilityDropConfig(
    val sampleStridePx: Int = 4,
    val minimumConfidence: Float = .80f,
    val minimumExpectedForwardMeters: Float = .50f,
    val maximumExpectedForwardMeters: Float = 5f,
    val minimumDepthExcessMeters: Float = .35f
) {
    init {
        require(sampleStridePx >= 1 && minimumConfidence in 0f..1f)
        require(minimumExpectedForwardMeters > 0f && maximumExpectedForwardMeters >= minimumExpectedForwardMeters)
        require(minimumDepthExcessMeters > 0f)
    }
}

enum class UstrfGroundVisibilityDropFailure { FRAME_MISMATCH, REGISTRATION_MISMATCH, DIMENSION_MISMATCH, STALE, GROUND_UNAVAILABLE }

sealed interface UstrfGroundVisibilityDropProposal {
    data class Available(val evidence: List<UstrfMetricGeometryEvidence>) : UstrfGroundVisibilityDropProposal
    data class Unavailable(val failure: UstrfGroundVisibilityDropFailure) : UstrfGroundVisibilityDropProposal
}

/**
 * Finds observed ground-ray depth that is materially farther than the local verified ground plane.
 * A zero/invalid depth sample is never a candidate: missing returns uncertainty upstream, not DROP.
 */
class UstrfGroundVisibilityDropProposer(private val config: UstrfGroundVisibilityDropConfig = UstrfGroundVisibilityDropConfig()) {
    fun propose(
        admission: UstrfMetricGeometryProjectionAdmission.Available,
        depth: UstrfRegisteredMetricDepthImage,
        intrinsics: UstrfCameraIntrinsicsReceipt,
        extrinsics: UstrfCameraBodyFullExtrinsicsReceipt,
        ground: UstrfVerifiedGroundPlaneReceipt,
        nowNs: Long
    ): UstrfGroundVisibilityDropProposal {
        if (depth.sourceFrame != admission.sourceFrame || ground.sourceFrame != admission.sourceFrame) return unavailable(UstrfGroundVisibilityDropFailure.FRAME_MISMATCH)
        if (depth.registrationTransformId != admission.registrationTransformId || depth.depthCoordinateFrame != admission.depthCoordinateFrame) return unavailable(UstrfGroundVisibilityDropFailure.REGISTRATION_MISMATCH)
        if (depth.widthPx != intrinsics.imageWidthPx || depth.heightPx != intrinsics.imageHeightPx) return unavailable(UstrfGroundVisibilityDropFailure.DIMENSION_MISMATCH)
        if (nowNs > depth.validUntilNs || nowNs > admission.validUntilNs || nowNs > ground.validUntilNs) return unavailable(UstrfGroundVisibilityDropFailure.STALE)
        if (!ground.independentlyVerified || ground.bodyFrame != admission.bodyFrame) return unavailable(UstrfGroundVisibilityDropFailure.GROUND_UNAVAILABLE)
        val candidates = mutableListOf<UstrfMetricGeometryEvidence>()
        for (v in 0 until depth.heightPx step config.sampleStridePx) for (u in 0 until depth.widthPx step config.sampleStridePx) {
            val index = v * depth.widthPx + u
            val observed = depth.depthMillimeters[index] / 1_000f
            if (observed <= 0f || depth.confidence[index] < config.minimumConfidence) continue
            val ray = UstrfVector3((u - intrinsics.principalXpx) / intrinsics.focalXpx, -(v - intrinsics.principalYpx) / intrinsics.focalYpx, 1f)
            val bodyRay = rotate(ray, extrinsics.cameraToBodyQuaternionXyzw)
            val numerator = dot(ground.normal, extrinsics.cameraToBodyTranslationM) + ground.offsetMeters
            val denominator = dot(ground.normal, bodyRay)
            if (denominator >= -.0001f) continue
            val expected = -numerator / denominator
            if (expected <= 0f || observed - expected < config.minimumDepthExcessMeters) continue
            val point = bodyRay * expected + extrinsics.cameraToBodyTranslationM
            if (point.z !in config.minimumExpectedForwardMeters..config.maximumExpectedForwardMeters) continue
            candidates += UstrfMetricGeometryEvidence(point.z, point.x, UstrfHeightBand.GROUND, UstrfGeometryKind.DROP,
                minOf(depth.confidence[index], ground.confidence), "ground-visibility-drop", minOf(depth.validUntilNs, ground.validUntilNs, admission.validUntilNs))
        }
        return UstrfGroundVisibilityDropProposal.Available(candidates)
    }

    private fun unavailable(failure: UstrfGroundVisibilityDropFailure) = UstrfGroundVisibilityDropProposal.Unavailable(failure)
    private fun dot(a: UstrfVector3, b: UstrfVector3) = a.x * b.x + a.y * b.y + a.z * b.z
    private fun rotate(p: UstrfVector3, q: FloatArray): UstrfVector3 {
        val tx = 2f * (q[1] * p.z - q[2] * p.y); val ty = 2f * (q[2] * p.x - q[0] * p.z); val tz = 2f * (q[0] * p.y - q[1] * p.x)
        return UstrfVector3(p.x + q[3] * tx + q[1] * tz - q[2] * ty, p.y + q[3] * ty + q[2] * tx - q[0] * tz, p.z + q[3] * tz + q[0] * ty - q[1] * tx)
    }
}
