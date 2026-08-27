package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfNativeMetricBoxGeometryAdapterTest {
    @Test
    fun nativeBoxesSeparateLowerBodyHeadAndNonActionableHighGeometry() {
        val frame = UstrfFrameStamp(9L, 1_000L, "body")
        val boxes = listOf(
            box(frame, 1f, 0f, 0f, 1.2f, "bollard"),
            box(frame, 2f, .5f, 1.5f, 1.9f, "low-branch"),
            box(frame, 3f, 0f, 2.3f, 3.0f, "tree-crown")
        )

        val available = UstrfNativeMetricBoxGeometryAdapter().project(frame, boxes, 1_100L)
            as UstrfNativeMetricBoxGeometryResult.Available
        val projected = UstrfGeometryProjector(UstrfGridSpec.DOCUMENT_FIVE_METER)
            .project(available.packet, 1_100L) as UstrfGeometryProjection.Available

        assertTrue(projected.observations.any { it.occupancy > 0f && it.source == "bollard" })
        assertTrue(projected.observations.any { it.headRisk > 0f && it.source == "low-branch" })
        assertEquals(0, projected.observations.count { it.source == "tree-crown" })
    }

    private fun box(
        frame: UstrfFrameStamp,
        forward: Float,
        lateral: Float,
        bottom: Float,
        top: Float,
        source: String
    ) = UstrfNativeMetricBox(
        sourceFrame = frame,
        centerForwardMeters = forward,
        centerLateralMeters = lateral,
        lengthMeters = .4f,
        widthMeters = .4f,
        yawRadians = 0f,
        bottomHeightAboveGroundMeters = bottom,
        topHeightAboveGroundMeters = top,
        confidence = 1f,
        source = source,
        validUntilNs = 2_000L
    )
}
