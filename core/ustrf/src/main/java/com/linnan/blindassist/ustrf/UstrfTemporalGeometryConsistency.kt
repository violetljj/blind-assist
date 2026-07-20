package com.linnan.blindassist.ustrf

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

data class UstrfTemporalGeometryConsistencyConfig(
    val maximumPlanarResidualMeters: Float = .35f,
    val maximumHeightResidualMeters: Float = .30f
) {
    init {
        require(maximumPlanarResidualMeters > 0f && maximumHeightResidualMeters > 0f)
    }
}

data class UstrfTemporalGeometryMatch(
    val prior: UstrfMetricGeometryEvidence,
    val current: UstrfMetricGeometryEvidence,
    val planarResidualMeters: Float
)

enum class UstrfTemporalGeometryConsistencyFailure {
    PRIOR_NOT_METRIC,
    CURRENT_NOT_METRIC,
    PRIOR_STALE,
    CURRENT_STALE,
    POSE_NOT_VERIFIED,
    POSE_FRAME_MISMATCH
}

sealed interface UstrfTemporalGeometryConsistencyResult {
    data class Available(
        val matches: List<UstrfTemporalGeometryMatch>,
        val unmatchedPriorCount: Int,
        val unmatchedCurrentCount: Int
    ) : UstrfTemporalGeometryConsistencyResult

    data class Unavailable(val failure: UstrfTemporalGeometryConsistencyFailure) : UstrfTemporalGeometryConsistencyResult
}

/**
 * Offline evidence-audit seam for adjacent metric-geometry frames.
 *
 * Static geometry from the prior body frame is moved into the current body frame using the exact
 * verified pose delta, then only like-for-like geometry is matched within metric residual gates.
 * It intentionally emits matches rather than new risk evidence: a temporal match is supporting
 * evidence for later model evaluation, not a licence to replace a current-frame observation or to
 * emit a `DROP` from a visibility gap.
 */
class UstrfTemporalGeometryConsistency(
    private val config: UstrfTemporalGeometryConsistencyConfig = UstrfTemporalGeometryConsistencyConfig()
) {
    fun compare(
        prior: UstrfGeometryPacket,
        current: UstrfGeometryPacket,
        poseDelta: UstrfVerifiedPoseDelta,
        nowNs: Long
    ): UstrfTemporalGeometryConsistencyResult {
        if (prior.scale != UstrfDepthScale.METRIC) return unavailable(UstrfTemporalGeometryConsistencyFailure.PRIOR_NOT_METRIC)
        if (current.scale != UstrfDepthScale.METRIC) return unavailable(UstrfTemporalGeometryConsistencyFailure.CURRENT_NOT_METRIC)
        if (prior.validUntilNs < nowNs) return unavailable(UstrfTemporalGeometryConsistencyFailure.PRIOR_STALE)
        if (current.validUntilNs < nowNs) return unavailable(UstrfTemporalGeometryConsistencyFailure.CURRENT_STALE)
        if (!poseDelta.verifiedForOfflineReplay) return unavailable(UstrfTemporalGeometryConsistencyFailure.POSE_NOT_VERIFIED)
        if (poseDelta.fromFrame != prior.sourceFrame || poseDelta.toFrame != current.sourceFrame) {
            return unavailable(UstrfTemporalGeometryConsistencyFailure.POSE_FRAME_MISMATCH)
        }

        val usablePrior = prior.evidence.filter { it.validUntilNs >= nowNs && it.kind != UstrfGeometryKind.DROP }
        val usableCurrent = current.evidence.filter { it.validUntilNs >= nowNs && it.kind != UstrfGeometryKind.DROP }
        val claimedCurrent = mutableSetOf<Int>()
        val matches = mutableListOf<UstrfTemporalGeometryMatch>()
        usablePrior.forEach { evidence ->
            val warped = warp(evidence, poseDelta)
            val candidate = usableCurrent.indices
                .filter { it !in claimedCurrent }
                .filter { usableCurrent[it].kind == evidence.kind && usableCurrent[it].heightBand == evidence.heightBand }
                .map { index -> index to planarResidual(warped, usableCurrent[index]) }
                .filter { (_, residual) -> residual <= config.maximumPlanarResidualMeters }
                .minByOrNull { (_, residual) -> residual }
            if (candidate != null) {
                val currentIndex = candidate.first
                val currentEvidence = usableCurrent[currentIndex]
                if (heightResidual(warped, currentEvidence) <= config.maximumHeightResidualMeters) {
                    claimedCurrent += currentIndex
                    matches += UstrfTemporalGeometryMatch(evidence, currentEvidence, candidate.second)
                }
            }
        }
        return UstrfTemporalGeometryConsistencyResult.Available(
            matches = matches,
            unmatchedPriorCount = usablePrior.size - matches.size,
            unmatchedCurrentCount = usableCurrent.size - matches.size
        )
    }

    private fun unavailable(failure: UstrfTemporalGeometryConsistencyFailure) = UstrfTemporalGeometryConsistencyResult.Unavailable(failure)

    private fun warp(evidence: UstrfMetricGeometryEvidence, delta: UstrfVerifiedPoseDelta): UstrfMetricGeometryEvidence {
        val translatedForward = evidence.forwardMeters - delta.forwardMeters
        val translatedLateral = evidence.lateralMeters - delta.lateralMeters
        val cosine = cos(delta.yawRadians)
        val sine = sin(delta.yawRadians)
        return evidence.copy(
            forwardMeters = cosine * translatedForward + sine * translatedLateral,
            lateralMeters = -sine * translatedForward + cosine * translatedLateral
        )
    }

    private fun planarResidual(first: UstrfMetricGeometryEvidence, second: UstrfMetricGeometryEvidence): Float =
        sqrt((first.forwardMeters - second.forwardMeters) * (first.forwardMeters - second.forwardMeters) +
            (first.lateralMeters - second.lateralMeters) * (first.lateralMeters - second.lateralMeters))

    private fun heightResidual(first: UstrfMetricGeometryEvidence, second: UstrfMetricGeometryEvidence): Float =
        // Height bands are discrete in this contract. A match must keep the same band; metric
        // height will be added only with a future ground-plane residual receipt.
        if (first.heightBand == second.heightBand) 0f else Float.POSITIVE_INFINITY
}
