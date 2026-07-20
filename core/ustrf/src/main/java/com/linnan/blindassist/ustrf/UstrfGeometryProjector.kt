package com.linnan.blindassist.ustrf

import kotlin.math.floor
import kotlin.math.roundToInt

enum class UstrfDepthScale { METRIC, RELATIVE, UNKNOWN }
enum class UstrfHeightBand { GROUND, LOWER_BODY, TORSO, HEAD }
enum class UstrfGeometryKind { TRAVERSABLE, OCCUPIED, DROP, HEAD_OBSTACLE, UNKNOWN }

/** A pre-projected point/evidence receipt; image/depth Adapters must own their own calibration. */
data class UstrfMetricGeometryEvidence(
    val forwardMeters: Float,
    val lateralMeters: Float,
    val heightBand: UstrfHeightBand,
    val kind: UstrfGeometryKind,
    val confidence: Float,
    val source: String,
    val validUntilNs: Long
) {
    init {
        require(forwardMeters >= 0f)
        require(confidence in 0f..1f)
        require(source.isNotBlank())
        require(validUntilNs >= 0L)
        if (kind == UstrfGeometryKind.HEAD_OBSTACLE) require(heightBand == UstrfHeightBand.HEAD)
        if (kind == UstrfGeometryKind.DROP) require(heightBand == UstrfHeightBand.GROUND)
    }
}

data class UstrfGeometryPacket(
    val sourceFrame: UstrfFrameStamp,
    val producedAtNs: Long,
    val validUntilNs: Long,
    val scale: UstrfDepthScale,
    val evidence: List<UstrfMetricGeometryEvidence>
) {
    init {
        require(producedAtNs >= sourceFrame.capturedAtNs)
        require(validUntilNs >= producedAtNs)
    }
}

enum class UstrfGeometryProjectionFailure { SCALE_NOT_METRIC, STALE }

sealed interface UstrfGeometryProjection {
    data class Available(val observations: List<UstrfRiskObservation>) : UstrfGeometryProjection
    data class Unavailable(val failure: UstrfGeometryProjectionFailure) : UstrfGeometryProjection
}

class UstrfGeometryProjector(
    private val cellSizeMeters: Float = .5f,
    private val halfWidthCells: Int = 2,
    private val horizonCells: Int = 4
) {
    init {
        require(cellSizeMeters > 0f)
        require(halfWidthCells >= 1 && horizonCells >= 1)
    }

    fun project(packet: UstrfGeometryPacket, nowNs: Long): UstrfGeometryProjection {
        if (packet.scale != UstrfDepthScale.METRIC) {
            return UstrfGeometryProjection.Unavailable(UstrfGeometryProjectionFailure.SCALE_NOT_METRIC)
        }
        if (packet.validUntilNs < nowNs) {
            return UstrfGeometryProjection.Unavailable(UstrfGeometryProjectionFailure.STALE)
        }
        return UstrfGeometryProjection.Available(
            packet.evidence.filter { it.validUntilNs >= nowNs }.mapNotNull { evidence ->
                val coordinate = UstrfGridCoordinate(
                    lateral = (evidence.lateralMeters / cellSizeMeters).roundToInt(),
                    forward = floor(evidence.forwardMeters / cellSizeMeters).toInt()
                )
                if (coordinate.lateral !in -halfWidthCells..halfWidthCells || coordinate.forward !in 0..horizonCells) {
                    null
                } else {
                    observationFor(coordinate, evidence)
                }
            }
        )
    }

    private fun observationFor(
        coordinate: UstrfGridCoordinate,
        evidence: UstrfMetricGeometryEvidence
    ): UstrfRiskObservation {
        val confidence = evidence.confidence
        return UstrfRiskObservation(
            coordinate = coordinate,
            occupancy = if (evidence.kind == UstrfGeometryKind.OCCUPIED) confidence else 0f,
            traversability = if (evidence.kind == UstrfGeometryKind.TRAVERSABLE) confidence else 0f,
            dropRisk = if (evidence.kind == UstrfGeometryKind.DROP) confidence else 0f,
            headRisk = if (evidence.kind == UstrfGeometryKind.HEAD_OBSTACLE) confidence else 0f,
            dynamicTtcMs = null,
            uncertainty = if (evidence.kind == UstrfGeometryKind.UNKNOWN) confidence else 1f - confidence,
            source = evidence.source,
            validUntilNs = evidence.validUntilNs
        )
    }
}
