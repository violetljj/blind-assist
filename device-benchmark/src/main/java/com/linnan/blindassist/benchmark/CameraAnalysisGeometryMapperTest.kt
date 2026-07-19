package com.linnan.blindassist.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CameraAnalysisGeometryMapperTest {
    @Test
    fun r833TransformMapsExactIntrinsicsIntoRotatedAnalysisDisplay() {
        val geometry = r833Geometry(identity())

        assertTrue(geometry.failureReason, geometry.valid)
        assertEquals(480, geometry.displayWidthPx)
        assertEquals(640, geometry.displayHeightPx)
        assertEquals(434.69432, geometry.fxPx, 1e-4)
        assertEquals(433.90063, geometry.fyPx, 1e-4)
        assertEquals(238.98842, geometry.cxPx, 1e-4)
        assertEquals(320.20874, geometry.cyPx, 1e-4)
    }

    @Test
    fun ninetyDegreeRotationAlsoRotatesCameraAxes() {
        val geometry = r833Geometry(identity())

        assertEquals(0.0, geometry.worldEnuToDisplayCamera[0], 1e-12)
        assertEquals(-1.0, geometry.worldEnuToDisplayCamera[1], 1e-12)
        assertEquals(1.0, geometry.worldEnuToDisplayCamera[3], 1e-12)
        assertEquals(1.0, geometry.worldEnuToDisplayCamera[8], 1e-12)
    }

    @Test
    fun mappedGeometryFeedsWorldProjectorAtMappedPrincipalPoint() {
        val rawPose = doubleArrayOf(0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        val geometry = r833Geometry(rawPose)
        val receipt = CameraProjectionReceipt(
            "r834_test", 500, 1_500, 1.0, doubleArrayOf(0.0, 0.0, 0.0),
            geometry.worldEnuToDisplayCamera, geometry.fxPx, geometry.fyPx,
            geometry.cxPx, geometry.cyPx, geometry.displayWidthPx, geometry.displayHeightPx
        )
        val points = listOf(1_000L, 2_000L, 3_000L).map { WorldRoutePoint(it, 0.0, 0.0, -5.0) }
        val projected = WorldRouteCameraProjector.project(points, receipt, 1_000)

        assertTrue(projected.failureReason, projected.routeValid)
        projected.waypoints.forEach {
            assertEquals(geometry.cxPx / geometry.displayWidthPx, it.xNorm, 1e-12)
            assertEquals(geometry.cyPx / geometry.displayHeightPx, it.yNorm, 1e-12)
        }
    }

    @Test
    fun shearSkewUnsupportedRotationOrInvalidPoseFailClosed() {
        val shear = sensorToBuffer().also { it[1] = 0.01 }
        val skew = intrinsics().also { it[4] = 1.0 }
        val invalidPose = identity().also { it[0] = 2.0 }

        assertFalse(CameraAnalysisGeometryMapper.map(shear, intrinsics(), 640, 480, 90, identity()).valid)
        assertFalse(CameraAnalysisGeometryMapper.map(sensorToBuffer(), skew, 640, 480, 90, identity()).valid)
        assertFalse(CameraAnalysisGeometryMapper.map(sensorToBuffer(), intrinsics(), 640, 480, 45, identity()).valid)
        assertFalse(CameraAnalysisGeometryMapper.map(sensorToBuffer(), intrinsics(), 640, 480, 90, invalidPose).valid)
    }

    private fun r833Geometry(pose: DoubleArray) = CameraAnalysisGeometryMapper.map(
        sensorToBuffer(), intrinsics(), 640, 480, 90, pose
    )

    private fun sensorToBuffer() = doubleArrayOf(
        0.15686275, 0.0, 0.0,
        0.0, 0.15686275, 0.0,
        0.0, 0.0, 1.0
    )

    private fun intrinsics() = doubleArrayOf(2766.1165, 2771.1763, 2041.3307, 1530.0737, 0.0)

    private fun identity() = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
}
