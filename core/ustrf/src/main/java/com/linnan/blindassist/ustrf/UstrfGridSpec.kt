package com.linnan.blindassist.ustrf

import kotlin.math.abs

/**
 * The single metric/raster contract shared by every USTRF grid consumer.
 *
 * [bodyWidthMeters] is a user/body-profile input used for structured shadow output. The
 * rasterised capsule may be wider because cells are discrete; both remain explicit here.
 */
data class UstrfGridSpec(
    val cellMeters: Float,
    val halfWidthCells: Int,
    val horizonCells: Int,
    val bodyCapsuleHalfWidthCells: Int,
    val candidateOffsets: List<Int>,
    val bodyWidthMeters: Float = .60f
) {
    init {
        require(cellMeters.isFinite() && cellMeters > 0f) { "cellMeters must be positive and finite" }
        require(halfWidthCells >= 1 && horizonCells >= 1) { "grid bounds must be positive" }
        require(bodyCapsuleHalfWidthCells >= 0) { "body capsule half width must be non-negative" }
        require(candidateOffsets.isNotEmpty()) { "at least one corridor candidate is required" }
        require(candidateOffsets.distinct().size == candidateOffsets.size) { "corridor candidates must be unique" }
        require(
            halfWidthCells >= candidateOffsets.maxOf { abs(it) } + bodyCapsuleHalfWidthCells
        ) { "candidate body envelope exceeds the risk-field lateral bounds" }
        require(bodyWidthMeters.isFinite() && bodyWidthMeters > 0f) { "body width must be positive and finite" }
        require(bodyWidthMeters <= rasterEnvelopeWidthMeters) {
            "body width exceeds its rasterised capsule"
        }
    }

    val lookaheadMeters: Float get() = horizonCells * cellMeters
    val rasterEnvelopeWidthMeters: Float get() = (bodyCapsuleHalfWidthCells * 2 + 1) * cellMeters

    fun contains(coordinate: UstrfGridCoordinate): Boolean =
        coordinate.lateral in -halfWidthCells..halfWidthCells && coordinate.forward in 0..horizonCells

    companion object {
        /** Compatibility profile for the original four-metre pure-Kotlin kernel. */
        val LEGACY_KERNEL = UstrfGridSpec(
            cellMeters = 1f,
            halfWidthCells = 2,
            horizonCells = 4,
            bodyCapsuleHalfWidthCells = 0,
            candidateOffsets = listOf(-1, 0, 1),
            bodyWidthMeters = .60f
        )

        /** Document target: 0-5 m at 0.5 m, with five body-envelope candidates. */
        val DOCUMENT_FIVE_METER = UstrfGridSpec(
            cellMeters = .5f,
            halfWidthCells = 3,
            horizonCells = 10,
            bodyCapsuleHalfWidthCells = 1,
            candidateOffsets = listOf(-2, -1, 0, 1, 2),
            bodyWidthMeters = .60f
        )

        val SYNTHETIC_CORRIDOR = DOCUMENT_FIVE_METER.copy(horizonCells = 4)
    }
}
