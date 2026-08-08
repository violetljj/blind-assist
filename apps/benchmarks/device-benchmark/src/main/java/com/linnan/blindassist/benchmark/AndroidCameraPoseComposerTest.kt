package com.linnan.blindassist.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AndroidCameraPoseComposerTest {
    @Test
    fun identityMetadataProducesIdentityWorldToCamera() {
        val result = AndroidCameraPoseComposer.compose(identity(), doubleArrayOf(0.0, 0.0, 0.0, 1.0))

        assertTrue(result.failureReason, result.valid)
        assertArrayEquals(identity(), result.worldEnuToCameraSensor, 1e-12)
    }

    @Test
    fun deviceToWorldIsInvertedBeforeLensPoseIsApplied() {
        val deviceToWorld = doubleArrayOf(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        val result = AndroidCameraPoseComposer.compose(deviceToWorld, doubleArrayOf(0.0, 0.0, 0.0, 1.0))

        assertTrue(result.valid)
        assertArrayEquals(
            doubleArrayOf(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            result.worldEnuToCameraSensor,
            1e-12
        )
    }

    @Test
    fun r830PrimaryCameraQuaternionMatchesOfficialXyzwFormula() {
        val q = doubleArrayOf(0.70710677, -0.70710677, 0.0, 0.0)
        val result = AndroidCameraPoseComposer.compose(identity(), q)

        assertTrue(result.valid)
        assertArrayEquals(
            doubleArrayOf(0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
            result.worldEnuToCameraSensor,
            1e-6
        )
    }

    @Test
    fun composedRotationCanFeedExistingWorldProjector() {
        val composed = AndroidCameraPoseComposer.compose(identity(), doubleArrayOf(0.0, 0.0, 0.0, 1.0))
        val receipt = CameraProjectionReceipt(
            receiptId = "r831_test",
            timestampMs = 500,
            validUntilTimestampMs = 1_500,
            confidence = 1.0,
            cameraOriginEnuM = doubleArrayOf(0.0, 0.0, 0.0),
            worldEnuToCamera = composed.worldEnuToCameraSensor,
            fxPx = 1_000.0,
            fyPx = 1_000.0,
            cxPx = 500.0,
            cyPx = 500.0,
            frameWidthPx = 1_000,
            frameHeightPx = 1_000
        )
        val points = listOf(1_000L, 2_000L, 3_000L).map { WorldRoutePoint(it, 0.0, 0.0, 4.0) }
        val projected = WorldRouteCameraProjector.project(points, receipt, 1_000)

        assertTrue(projected.failureReason, projected.routeValid)
        projected.waypoints.forEach {
            assertEquals(0.5, it.xNorm, 1e-12)
            assertEquals(0.5, it.yNorm, 1e-12)
        }
    }

    @Test
    fun nonRotationOrNonUnitQuaternionFailsClosed() {
        val scaled = doubleArrayOf(2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        assertFalse(AndroidCameraPoseComposer.compose(scaled, doubleArrayOf(0.0, 0.0, 0.0, 1.0)).valid)
        assertFalse(AndroidCameraPoseComposer.compose(identity(), doubleArrayOf(0.0, 0.0, 0.0, 2.0)).valid)
        assertFalse(AndroidCameraPoseComposer.compose(identity(), doubleArrayOf(Double.NaN, 0.0, 0.0, 1.0)).valid)
    }

    private fun identity() = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
}
