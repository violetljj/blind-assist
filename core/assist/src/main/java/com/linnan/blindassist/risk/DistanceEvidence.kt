package com.linnan.blindassist.risk

/**
 * Relative distance evidence used by experimental risk fusion.
 *
 * The value intentionally represents a proximity band rather than meters. A
 * monocular depth model cannot provide reliable metric distance without device
 * calibration, so BlindAssist keeps the user-facing contract as relative risk.
 */
data class DistanceEvidence(
    val band: ProximityBand,
    val confidence: Float,
    val source: DistanceEvidenceSource,
    val relativeDepthScore: Float
) {
    init {
        require(confidence in 0f..1f) { "confidence must be in [0, 1]" }
        require(relativeDepthScore in 0f..1f) { "relativeDepthScore must be in [0, 1]" }
    }
}

typealias DepthBandEvidence = DistanceEvidence

enum class DistanceEvidenceSource {
    GEOMETRY,
    MONOCULAR_DEPTH,
    ARCORE_DEPTH,
    OFFLINE_LABEL
}
