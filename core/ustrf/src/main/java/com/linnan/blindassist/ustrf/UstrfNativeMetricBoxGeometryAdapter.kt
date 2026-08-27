package com.linnan.blindassist.ustrf

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.math.sqrt

/** Privileged, body-local 3-D box used by public-data geometry ceilings. */
data class UstrfNativeMetricBox(
    val sourceFrame: UstrfFrameStamp,
    val centerForwardMeters: Float,
    val centerLateralMeters: Float,
    val lengthMeters: Float,
    val widthMeters: Float,
    val yawRadians: Float,
    val bottomHeightAboveGroundMeters: Float,
    val topHeightAboveGroundMeters: Float,
    val confidence: Float,
    val source: String,
    val validUntilNs: Long
) {
    init {
        require(centerForwardMeters.isFinite() && centerLateralMeters.isFinite())
        require(lengthMeters.isFinite() && lengthMeters > 0f)
        require(widthMeters.isFinite() && widthMeters > 0f)
        require(yawRadians.isFinite())
        require(bottomHeightAboveGroundMeters.isFinite())
        require(topHeightAboveGroundMeters.isFinite() && topHeightAboveGroundMeters > bottomHeightAboveGroundMeters)
        require(confidence in 0f..1f)
        require(source.isNotBlank())
        require(validUntilNs >= sourceFrame.capturedAtNs)
    }
}

data class UstrfNativeMetricBoxGeometryConfig(
    val groundToleranceMeters: Float = .12f,
    val lowerBodyMaximumMeters: Float = 1.35f,
    val headMinimumMeters: Float = 1.35f,
    val headMaximumMeters: Float = 2.10f
) {
    init {
        require(groundToleranceMeters > 0f)
        require(lowerBodyMaximumMeters > groundToleranceMeters)
        require(headMinimumMeters >= lowerBodyMaximumMeters)
        require(headMaximumMeters > headMinimumMeters)
    }
}

enum class UstrfNativeMetricBoxGeometryFailure { SOURCE_FRAME_MISMATCH, STALE }

sealed interface UstrfNativeMetricBoxGeometryResult {
    data class Available(
        val packet: UstrfGeometryPacket,
        val rasterizedCellCount: Int
    ) : UstrfNativeMetricBoxGeometryResult

    data class Unavailable(
        val failure: UstrfNativeMetricBoxGeometryFailure
    ) : UstrfNativeMetricBoxGeometryResult
}

/**
 * Offline adapter from source-native 3-D boxes to the existing USTRF metric grid.
 *
 * A box intersecting the lower-body and head bands may emit both risks. Geometry
 * wholly above [UstrfNativeMetricBoxGeometryConfig.headMaximumMeters] is ignored,
 * so a visible ceiling or tree crown is not automatically a head warning.
 */
class UstrfNativeMetricBoxGeometryAdapter(
    private val gridSpec: UstrfGridSpec = UstrfGridSpec.DOCUMENT_FIVE_METER,
    private val config: UstrfNativeMetricBoxGeometryConfig = UstrfNativeMetricBoxGeometryConfig()
) {
    fun project(
        sourceFrame: UstrfFrameStamp,
        boxes: List<UstrfNativeMetricBox>,
        producedAtNs: Long
    ): UstrfNativeMetricBoxGeometryResult {
        if (boxes.any { it.sourceFrame != sourceFrame }) {
            return UstrfNativeMetricBoxGeometryResult.Unavailable(
                UstrfNativeMetricBoxGeometryFailure.SOURCE_FRAME_MISMATCH
            )
        }
        if (boxes.any { producedAtNs > it.validUntilNs }) {
            return UstrfNativeMetricBoxGeometryResult.Unavailable(
                UstrfNativeMetricBoxGeometryFailure.STALE
            )
        }
        val validUntilNs = boxes.minOfOrNull { it.validUntilNs } ?: producedAtNs
        val evidence = mutableListOf<UstrfMetricGeometryEvidence>()
        var rasterizedCells = 0
        boxes.forEach { box ->
            val lowerBody =
                box.topHeightAboveGroundMeters > config.groundToleranceMeters &&
                    box.bottomHeightAboveGroundMeters < config.lowerBodyMaximumMeters
            val head =
                box.topHeightAboveGroundMeters >= config.headMinimumMeters &&
                    box.bottomHeightAboveGroundMeters <= config.headMaximumMeters
            if (!lowerBody && !head) return@forEach
            cellsFor(box).forEach { coordinate ->
                rasterizedCells += 1
                val forwardMeters = coordinate.forward * gridSpec.cellMeters
                val lateralMeters = coordinate.lateral * gridSpec.cellMeters
                if (lowerBody) {
                    evidence += UstrfMetricGeometryEvidence(
                        forwardMeters = forwardMeters,
                        lateralMeters = lateralMeters,
                        heightBand = UstrfHeightBand.LOWER_BODY,
                        kind = UstrfGeometryKind.OCCUPIED,
                        confidence = box.confidence,
                        source = box.source,
                        validUntilNs = box.validUntilNs
                    )
                }
                if (head) {
                    evidence += UstrfMetricGeometryEvidence(
                        forwardMeters = forwardMeters,
                        lateralMeters = lateralMeters,
                        heightBand = UstrfHeightBand.HEAD,
                        kind = UstrfGeometryKind.HEAD_OBSTACLE,
                        confidence = box.confidence,
                        source = box.source,
                        validUntilNs = box.validUntilNs
                    )
                }
            }
        }
        return UstrfNativeMetricBoxGeometryResult.Available(
            UstrfGeometryPacket(
                sourceFrame = sourceFrame,
                producedAtNs = producedAtNs,
                validUntilNs = validUntilNs,
                scale = UstrfDepthScale.METRIC,
                evidence = evidence
            ),
            rasterizedCellCount = rasterizedCells
        )
    }

    private fun cellsFor(box: UstrfNativeMetricBox): List<UstrfGridCoordinate> {
        val cosine = cos(box.yawRadians)
        val sine = sin(box.yawRadians)
        val cellRadius = gridSpec.cellMeters * sqrt(2f) / 2f
        return buildList {
            for (forward in 0..gridSpec.horizonCells) {
                for (lateral in -gridSpec.halfWidthCells..gridSpec.halfWidthCells) {
                    val deltaForward = forward * gridSpec.cellMeters - box.centerForwardMeters
                    val deltaLateral = lateral * gridSpec.cellMeters - box.centerLateralMeters
                    val boxForward = deltaForward * cosine + deltaLateral * sine
                    val boxLateral = -deltaForward * sine + deltaLateral * cosine
                    val outsideForward = maxOf(0f, abs(boxForward) - box.lengthMeters / 2f)
                    val outsideLateral = maxOf(0f, abs(boxLateral) - box.widthMeters / 2f)
                    if (hypot(outsideForward, outsideLateral) <= cellRadius) {
                        add(UstrfGridCoordinate(lateral, forward))
                    }
                }
            }
        }
    }
}
