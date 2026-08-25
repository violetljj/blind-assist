package com.linnan.blindassist.semanticanchor

import com.google.common.truth.Truth.assertThat
import org.junit.Test
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class MarkerPoseGuidanceTest {
    private val intrinsics = CameraIntrinsics(fx = 800.0, fy = 800.0, cx = 640.0, cy = 360.0)

    @Test
    fun planarPnpRecoversFrontalMetricPoseAndWaypoint() {
        val estimate = requireNotNull(
            SquareMarkerPoseSolver.solve(
                payload = "BLINDASSIST:ANCHOR:17",
                corners = projectedSquare(centerX = 0.12, centerZ = 1.65),
                intrinsics = intrinsics,
                markerSizeMeters = 0.16,
                standoffMeters = 0.65,
            ),
        )

        assertThat(estimate.rangeMeters).isWithin(0.01).of(1.654)
        assertThat(estimate.lateralMeters).isWithin(0.01).of(0.12)
        assertThat(estimate.markerYawDegrees).isWithin(0.1).of(0.0)
        assertThat(estimate.waypointForwardMeters).isWithin(0.01).of(1.0)
        assertThat(estimate.reprojectionErrorPixels).isWithin(0.001).of(0.0)
    }

    @Test
    fun pnpControllerStopsOnLostAndArrivesOnlyAtAlignedStandoff() {
        val controller = MarkerPoseController(arm = GuidanceArm.PNP_POSE)
        val far = requireNotNull(
            SquareMarkerPoseSolver.solve(
                "A17", projectedSquare(0.0, 1.65), intrinsics, 0.16, 0.65,
            ),
        )
        val near = requireNotNull(
            SquareMarkerPoseSolver.solve(
                "A17", projectedSquare(0.0, 0.78), intrinsics, 0.16, 0.65,
            ),
        )

        assertThat(controller.update(AnchorPhase.LOST, far).command).isEqualTo("STOP")
        assertThat(controller.update(AnchorPhase.LOCKED, far).command).isEqualTo("FORWARD")
        assertThat(controller.update(AnchorPhase.REACQUIRED, near).command).isEqualTo("HOLD")
        assertThat(controller.update(AnchorPhase.REACQUIRED, near).command).isEqualTo("ARRIVE")
    }

    @Test
    fun planarPnpRecoversObliqueMarkerOrientation() {
        val estimate = requireNotNull(
            SquareMarkerPoseSolver.solve(
                "A17", projectedSquare(centerX = -0.08, centerZ = 1.4, yawDegrees = 24.0),
                intrinsics, 0.16, 0.65,
            ),
        )

        assertThat(abs(estimate.markerYawDegrees)).isWithin(0.2).of(24.0)
        assertThat(estimate.reprojectionErrorPixels).isWithin(0.001).of(0.0)
    }

    @Test
    fun centerBaselineIgnoresPlaneWaypointAndTurnsFromImageCenter() {
        val controller = MarkerPoseController(arm = GuidanceArm.CENTER_BASELINE)
        val right = requireNotNull(
            SquareMarkerPoseSolver.solve(
                "A17", projectedSquare(0.35, 1.65), intrinsics, 0.16, 0.65,
            ),
        )

        assertThat(controller.update(AnchorPhase.LOCKED, right).command).isEqualTo("RIGHT")
    }

    private fun projectedSquare(centerX: Double, centerZ: Double, yawDegrees: Double = 0.0): List<PixelPoint> {
        val half = 0.08
        val yaw = Math.toRadians(yawDegrees)
        return listOf(-half to -half, half to -half, half to half, -half to half).map { (x, y) ->
            val cameraX = centerX + cos(yaw) * x
            val cameraZ = centerZ - sin(yaw) * x
            PixelPoint(
                intrinsics.fx * cameraX / cameraZ + intrinsics.cx,
                intrinsics.fy * y / cameraZ + intrinsics.cy,
            )
        }
    }
}
