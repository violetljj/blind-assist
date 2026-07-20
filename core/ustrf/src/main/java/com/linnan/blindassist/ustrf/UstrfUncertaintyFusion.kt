package com.linnan.blindassist.ustrf

data class UstrfUncertaintyEvidence(
    val model: Float,
    val geometry: Float,
    val age: Float,
    val outOfDistribution: Float
) {
    init {
        listOf(model, geometry, age, outOfDistribution).forEach { require(it in 0f..1f) }
    }
}

/** Explicit uncertainty composition; calibration and threshold selection remain dataset-gated work. */
object UstrfUncertaintyFusion {
    fun fuse(evidence: UstrfUncertaintyEvidence): Float = 1f -
        (1f - evidence.model) *
        (1f - evidence.geometry) *
        (1f - evidence.age) *
        (1f - evidence.outOfDistribution)
}
