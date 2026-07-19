package com.linnan.blindassist.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class WorldRouteCameraProjectorTest {
    @Test
    fun identityPoseProjectsStraightGroundRoute() {
        val result = WorldRouteCameraProjector.project(straightGroundRoute(), receipt(), 1_000)

        assertTrue(result.failureReason, result.routeValid)
        result.waypoints.forEach {
            assertEquals(0.5, it.xNorm, 1e-12)
            assertEquals(0.75, it.yNorm, 1e-12)
        }
    }

    @Test
    fun signedCameraXProducesLeftAndRightWaypoints() {
        val left = WorldRouteCameraProjector.project(lateralRoute(-0.5), receipt(), 1_000)
        val right = WorldRouteCameraProjector.project(lateralRoute(0.5), receipt(), 1_000)

        assertTrue(left.waypoints.all { it.xNorm < 0.5 })
        assertTrue(right.waypoints.all { it.xNorm > 0.5 })
    }

    @Test
    fun translationReceiptUsesCameraRelativeWorldDelta() {
        val translated = receipt(origin = doubleArrayOf(10.0, 20.0, 0.0))
        val points = straightGroundRoute().map { it.copy(eastM = it.eastM + 10.0, northM = it.northM + 20.0) }
        val result = WorldRouteCameraProjector.project(points, translated, 1_000)

        assertTrue(result.routeValid)
        assertTrue(result.waypoints.all { it.xNorm == 0.5 && it.yNorm == 0.75 })
    }

    @Test
    fun staleFutureOrLowConfidenceReceiptFailsClosed() {
        val stale = receipt(timestampMs = 0, validUntilMs = 500)
        val future = receipt(timestampMs = 1_001, validUntilMs = 1_500)
        val low = receipt(confidence = 0.49)

        listOf(stale, future, low).forEach {
            assertFalse(WorldRouteCameraProjector.project(straightGroundRoute(), it, 1_000).routeValid)
        }
    }

    @Test
    fun invalidRotationBehindCameraOrOutOfFrameFailsClosed() {
        val scaledRotation = receipt(rotation = doubleArrayOf(2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        val behind = straightGroundRoute().map { it.copy(upM = -it.upM) }
        val outside = lateralRoute(10.0)

        assertFalse(WorldRouteCameraProjector.project(straightGroundRoute(), scaledRotation, 1_000).routeValid)
        assertFalse(WorldRouteCameraProjector.project(behind, receipt(), 1_000).routeValid)
        assertFalse(WorldRouteCameraProjector.project(outside, receipt(), 1_000).routeValid)
    }

    @Test
    fun projectedRouteFeedsAspectAwareGeometryAndLifecycle() {
        val projected = WorldRouteCameraProjector.project(straightGroundRoute(), receipt(), 1_000)
        val obstacle = listOf(ExplicitRouteObstacleBox(0.49, 0.74, 0.51, 0.76))
        val samples = listOf(1_000L, 2_000L).map { timestamp ->
            ExplicitRouteGeometryFusion.score(
                ExplicitRouteFrameInput(timestamp, timestamp + 500, 1.0, projected.routeValid,
                    1_000, 1_000, projected.waypoints, obstacle),
                obstacleExpansionObjectHeights = 0.0
            )
        }

        assertEquals(
            listOf(ExplicitRouteRiskTransition(ExplicitRouteRiskState.INTERVENTION_NEEDED, 2_000)),
            ExplicitRouteIntentFusion.decode(samples)
        )
    }

    private fun straightGroundRoute() = listOf(
        WorldRoutePoint(1_000, 0.0, 0.5, 2.0),
        WorldRoutePoint(2_000, 0.0, 1.0, 4.0),
        WorldRoutePoint(3_000, 0.0, 1.5, 6.0)
    )

    private fun lateralRoute(x: Double) = listOf(
        WorldRoutePoint(1_000, x, 0.5, 2.0),
        WorldRoutePoint(2_000, x, 1.0, 4.0),
        WorldRoutePoint(3_000, x, 1.5, 6.0)
    )

    private fun receipt(
        timestampMs: Long = 500,
        validUntilMs: Long = 1_500,
        confidence: Double = 1.0,
        origin: DoubleArray = doubleArrayOf(0.0, 0.0, 0.0),
        rotation: DoubleArray = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    ) = CameraProjectionReceipt(
        "pose_receipt_test", timestampMs, validUntilMs, confidence, origin, rotation,
        fxPx = 1_000.0, fyPx = 1_000.0, cxPx = 500.0, cyPx = 500.0,
        frameWidthPx = 1_000, frameHeightPx = 1_000
    )
}
