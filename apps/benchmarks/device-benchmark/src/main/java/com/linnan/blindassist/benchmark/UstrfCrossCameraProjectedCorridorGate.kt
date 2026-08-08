package com.linnan.blindassist.benchmark

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.FrameSize
import kotlin.math.abs
import kotlin.math.hypot

internal data class CrossCameraCorridorPoint(val xNorm: Double, val yNorm: Double)

internal enum class CrossCameraCorridorRelation {
    INSIDE,
    OUTSIDE,
    UNCERTAIN_BOUNDARY
}

internal data class CrossCameraCorridorClassification(
    val relation: CrossCameraCorridorRelation,
    val footpointXPx: Double,
    val footpointYPx: Double,
    val boundaryDistancePx: Double,
    val uncertaintyPx: Double,
    val nominalInside: Boolean
)

/**
 * Research-only projected route polygon gate.
 *
 * The bbox bottom centre is the object-agnostic ground-contact proxy. A detection is only called
 * INSIDE/OUTSIDE when its footpoint is farther than the registered projection uncertainty from
 * every route boundary. Boundary cases remain explicit uncertainty and are never upgraded to a
 * certain route-inside claim. This class is test-APK-only and does not change the U0 v1 gate.
 */
internal object UstrfCrossCameraProjectedCorridorGate {
    const val CONTRACT_ID = "bbox_bottom_center_projected_polygon_uncertainty_r1"

    fun classify(
        polygon: List<CrossCameraCorridorPoint>,
        detectionBox: BoundingBox,
        frameSize: FrameSize,
        projectionUncertaintyFrameRatio: Double
    ): CrossCameraCorridorClassification {
        require(frameSize.width > 0 && frameSize.height > 0)
        require(projectionUncertaintyFrameRatio.isFinite() && projectionUncertaintyFrameRatio in 0.0..0.25)
        require(detectionBox.left >= 0f && detectionBox.top >= 0f &&
            detectionBox.right <= frameSize.width && detectionBox.bottom <= frameSize.height &&
            detectionBox.left < detectionBox.right && detectionBox.top < detectionBox.bottom
        ) { "detection box is invalid" }
        validatePolygon(polygon)
        val polygonPx = polygon.map { Point(it.xNorm * frameSize.width, it.yNorm * frameSize.height) }
        val footpoint = Point(
            (detectionBox.left + detectionBox.right) / 2.0,
            detectionBox.bottom.toDouble()
        )
        val nominalInside = pointInPolygon(footpoint, polygonPx)
        val distance = polygonPx.indices.minOf { index ->
            pointSegmentDistance(footpoint, polygonPx[index], polygonPx[(index + 1) % polygonPx.size])
        }
        val uncertainty = projectionUncertaintyFrameRatio * frameSize.width
        val relation = if (distance <= uncertainty) {
            CrossCameraCorridorRelation.UNCERTAIN_BOUNDARY
        } else if (nominalInside) {
            CrossCameraCorridorRelation.INSIDE
        } else {
            CrossCameraCorridorRelation.OUTSIDE
        }
        return CrossCameraCorridorClassification(
            relation, footpoint.x, footpoint.y, distance, uncertainty, nominalInside
        )
    }

    private fun validatePolygon(polygon: List<CrossCameraCorridorPoint>) {
        require(polygon.size >= 3) { "route corridor polygon needs at least three points" }
        require(polygon.all { it.xNorm.isFinite() && it.yNorm.isFinite() && it.xNorm in 0.0..1.0 && it.yNorm in 0.0..1.0 }) {
            "route corridor polygon is outside normalized image space"
        }
        val twiceArea = polygon.indices.sumOf { index ->
            val current = polygon[index]
            val next = polygon[(index + 1) % polygon.size]
            current.xNorm * next.yNorm - next.xNorm * current.yNorm
        }
        require(abs(twiceArea) >= 1e-6) { "route corridor polygon has zero area" }
        val signs = mutableSetOf<Int>()
        for (index in polygon.indices) {
            val a = polygon[(index - 1 + polygon.size) % polygon.size]
            val b = polygon[index]
            val c = polygon[(index + 1) % polygon.size]
            val cross = (b.xNorm - a.xNorm) * (c.yNorm - b.yNorm) -
                (b.yNorm - a.yNorm) * (c.xNorm - b.xNorm)
            if (abs(cross) > 1e-9) signs += if (cross > 0.0) 1 else -1
        }
        require(signs.size <= 1) { "route corridor polygon must be convex and consistently ordered" }
    }

    private fun pointInPolygon(point: Point, polygon: List<Point>): Boolean {
        var inside = false
        for (index in polygon.indices) {
            val start = polygon[index]
            val end = polygon[(index + 1) % polygon.size]
            if ((start.y > point.y) != (end.y > point.y)) {
                val crossingX = (end.x - start.x) * (point.y - start.y) / (end.y - start.y) + start.x
                if (point.x < crossingX) inside = !inside
            }
        }
        return inside
    }

    private fun pointSegmentDistance(point: Point, start: Point, end: Point): Double {
        val dx = end.x - start.x
        val dy = end.y - start.y
        val lengthSquared = dx * dx + dy * dy
        if (lengthSquared == 0.0) return hypot(point.x - start.x, point.y - start.y)
        val ratio = (((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared).coerceIn(0.0, 1.0)
        return hypot(point.x - (start.x + ratio * dx), point.y - (start.y + ratio * dy))
    }

    private data class Point(val x: Double, val y: Double)
}
