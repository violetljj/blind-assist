package com.linnan.blindassist.ustrf

import kotlin.math.cos
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sin

data class UstrfRiskFieldConfig(
    val halfWidthCells: Int = 2,
    val horizonCells: Int = 4,
    val cellMeters: Float = 1f,
    val staticLifetimeNs: Long = 2_000_000_000L,
    val dynamicLifetimeNs: Long = 700_000_000L
) {
    init {
        require(halfWidthCells >= 1 && horizonCells >= 1)
        require(cellMeters > 0f && cellMeters.isFinite())
        require(staticLifetimeNs > 0L && dynamicLifetimeNs > 0L)
    }
}

/**
 * A source-frame-bound local pose delta for offline replay only.
 *
 * ``forwardMeters`` and ``lateralMeters`` describe the current body origin in
 * the previous body frame. Positive yaw rotates the current body frame to the
 * left relative to that previous frame. The field builder uses the inverse
 * transform to carry static evidence into the current user-local grid.
 *
 * A device adapter cannot manufacture this receipt: it must bind exactly to the
 * two adjacent frame stamps and be explicitly verified by an independent
 * offline/geometry gate before a caller opts into warp.
 */
data class UstrfVerifiedPoseDelta(
    val fromFrame: UstrfFrameStamp,
    val toFrame: UstrfFrameStamp,
    val forwardMeters: Float,
    val lateralMeters: Float,
    val yawRadians: Float,
    val verifiedForOfflineReplay: Boolean
) {
    init {
        require(forwardMeters.isFinite() && lateralMeters.isFinite() && yawRadians.isFinite())
        require(fromFrame.coordinateFrame == toFrame.coordinateFrame)
        require(toFrame.frameId > fromFrame.frameId)
        require(toFrame.capturedAtNs > fromFrame.capturedAtNs)
    }
}

data class UstrfRiskCell(
    val occupancy: Float = 0f,
    val traversability: Float = 0f,
    val dropRisk: Float = 0f,
    val headRisk: Float = 0f,
    val dynamicTtcMs: Long? = null,
    val uncertainty: Float = 1f,
    val ageNs: Long = Long.MAX_VALUE,
    val sources: Set<String> = emptySet()
) {
    val isUnknown: Boolean get() = uncertainty >= UNKNOWN_THRESHOLD
    companion object { const val UNKNOWN_THRESHOLD = 0.70f }
}

data class UstrfRiskField(
    val frame: UstrfFrameStamp,
    val cells: Map<UstrfGridCoordinate, UstrfRiskCell>
) {
    fun cellAt(coordinate: UstrfGridCoordinate): UstrfRiskCell = cells[coordinate] ?: UstrfRiskCell()
}

/**
 * A deterministic user-local raster.
 *
 * Callers without a [UstrfVerifiedPoseDelta] preserve the previous local-grid
 * behavior. A warp is intentionally opt-in and rejects unverified or
 * frame-mismatched pose receipts rather than silently reusing stale geometry.
 */
class UstrfRiskFieldBuilder(private val config: UstrfRiskFieldConfig = UstrfRiskFieldConfig()) {
    private var previous: UstrfRiskField? = null

    fun update(packet: UstrfPerceptionPacket, poseDelta: UstrfVerifiedPoseDelta? = null): UstrfRiskField {
        val prior = previous
        if (prior != null) {
            require(packet.sourceFrame.capturedAtNs > prior.frame.capturedAtNs) { "non-monotonic capture time" }
            require(packet.sourceFrame.frameId > prior.frame.frameId) { "non-monotonic frame ID" }
            require(packet.sourceFrame.coordinateFrame == prior.frame.coordinateFrame) { "coordinate frame changed without warp Adapter" }
            poseDelta?.let { validatePoseDelta(it, prior.frame, packet.sourceFrame) }
        } else {
            require(poseDelta == null) { "first risk-field frame cannot carry a pose delta" }
        }
        val now = packet.sourceFrame.capturedAtNs
        val cells = prior?.let { priorField ->
            retainedPriorCells(priorField, now, poseDelta).toMutableMap()
        } ?: mutableMapOf()
        packet.observations.filter { it.validUntilNs >= now && inBounds(it.coordinate) }.forEach { observation ->
            val old = cells[observation.coordinate] ?: UstrfRiskCell()
            cells[observation.coordinate] = UstrfRiskCell(
                occupancy = max(old.occupancy, observation.occupancy),
                traversability = max(old.traversability, observation.traversability),
                dropRisk = max(old.dropRisk, observation.dropRisk),
                headRisk = max(old.headRisk, observation.headRisk),
                dynamicTtcMs = listOfNotNull(old.dynamicTtcMs, observation.dynamicTtcMs).minOrNull(),
                uncertainty = max(if (old.ageNs == Long.MAX_VALUE) 0f else old.uncertainty, observation.uncertainty),
                ageNs = 0L,
                sources = old.sources + observation.source
            )
        }
        return UstrfRiskField(packet.sourceFrame, cells).also { previous = it }
    }

    fun reset() { previous = null }

    private fun validatePoseDelta(delta: UstrfVerifiedPoseDelta, prior: UstrfFrameStamp, current: UstrfFrameStamp) {
        require(delta.verifiedForOfflineReplay) { "risk-field warp requires independently verified pose evidence" }
        require(delta.fromFrame == prior) { "pose delta must bind to the previous risk-field frame" }
        require(delta.toFrame == current) { "pose delta must bind to the current perception frame" }
    }

    private fun retainedPriorCells(
        prior: UstrfRiskField,
        now: Long,
        poseDelta: UstrfVerifiedPoseDelta?
    ): Map<UstrfGridCoordinate, UstrfRiskCell> {
        val elapsed = now - prior.frame.capturedAtNs
        val retained = mutableMapOf<UstrfGridCoordinate, UstrfRiskCell>()
        prior.cells.forEach { (coordinate, cell) ->
            val destination = poseDelta?.let { warp(coordinate, it) } ?: coordinate
            if (!inBounds(destination)) return@forEach
            val decayed = decay(cell, elapsed)
            retained[destination] = retained[destination]?.let { old -> merge(old, decayed) } ?: decayed
        }
        return retained
    }

    /** Converts prior-body grid coordinates into the current body grid. */
    private fun warp(coordinate: UstrfGridCoordinate, delta: UstrfVerifiedPoseDelta): UstrfGridCoordinate {
        val forward = coordinate.forward.toFloat() - delta.forwardMeters / config.cellMeters
        val lateral = coordinate.lateral.toFloat() - delta.lateralMeters / config.cellMeters
        val cosYaw = cos(delta.yawRadians)
        val sinYaw = sin(delta.yawRadians)
        return UstrfGridCoordinate(
            lateral = (-sinYaw * forward + cosYaw * lateral).roundToInt(),
            forward = (cosYaw * forward + sinYaw * lateral).roundToInt()
        )
    }

    private fun inBounds(point: UstrfGridCoordinate): Boolean =
        point.lateral in -config.halfWidthCells..config.halfWidthCells && point.forward in 0..config.horizonCells

    private fun decay(cell: UstrfRiskCell, deltaNs: Long): UstrfRiskCell {
        val lifetime = if (cell.dynamicTtcMs == null) config.staticLifetimeNs else config.dynamicLifetimeNs
        val retained = (1f - deltaNs.toFloat() / lifetime.toFloat()).coerceIn(0f, 1f)
        return cell.copy(
            occupancy = cell.occupancy * retained,
            traversability = cell.traversability * retained,
            dropRisk = cell.dropRisk * retained,
            headRisk = cell.headRisk * retained,
            dynamicTtcMs = cell.dynamicTtcMs?.takeIf { retained > 0f },
            uncertainty = 1f - (1f - cell.uncertainty) * retained,
            ageNs = if (cell.ageNs == Long.MAX_VALUE) Long.MAX_VALUE else cell.ageNs + deltaNs
        )
    }

    private fun merge(first: UstrfRiskCell, second: UstrfRiskCell): UstrfRiskCell = UstrfRiskCell(
        occupancy = max(first.occupancy, second.occupancy),
        traversability = max(first.traversability, second.traversability),
        dropRisk = max(first.dropRisk, second.dropRisk),
        headRisk = max(first.headRisk, second.headRisk),
        dynamicTtcMs = listOfNotNull(first.dynamicTtcMs, second.dynamicTtcMs).minOrNull(),
        uncertainty = max(first.uncertainty, second.uncertainty),
        ageNs = minOf(first.ageNs, second.ageNs),
        sources = first.sources + second.sources
    )
}
