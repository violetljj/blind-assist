package com.linnan.blindassist.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.FrameSize
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class UstrfCrossCameraProjectedCorridorGateTest {
    private val frame = FrameSize(640, 360)
    private val route = listOf(
        CrossCameraCorridorPoint(0.56, 0.98),
        CrossCameraCorridorPoint(0.92, 0.98),
        CrossCameraCorridorPoint(0.66, 0.44),
        CrossCameraCorridorPoint(0.55, 0.44)
    )

    @Test
    fun pexelsRoadCarIsOutsideAtNarrowUncertaintyAndUncertainAtWiderProfiles() {
        val car = BoundingBox(325.7878f, 203.0786f, 367.3694f, 243.1192f)
        assertEquals(
            CrossCameraCorridorRelation.OUTSIDE,
            UstrfCrossCameraProjectedCorridorGate.classify(route, car, frame, 0.01).relation
        )
        assertEquals(
            CrossCameraCorridorRelation.UNCERTAIN_BOUNDARY,
            UstrfCrossCameraProjectedCorridorGate.classify(route, car, frame, 0.02).relation
        )
        assertEquals(
            CrossCameraCorridorRelation.UNCERTAIN_BOUNDARY,
            UstrfCrossCameraProjectedCorridorGate.classify(route, car, frame, 0.03).relation
        )
    }

    @Test
    fun centralGroundContactRemainsInsideAndFarRoadContactRemainsOutside() {
        val central = BoundingBox(420f, 220f, 450f, 260f)
        val farRoad = BoundingBox(120f, 200f, 180f, 250f)
        assertEquals(
            CrossCameraCorridorRelation.INSIDE,
            UstrfCrossCameraProjectedCorridorGate.classify(route, central, frame, 0.03).relation
        )
        assertEquals(
            CrossCameraCorridorRelation.OUTSIDE,
            UstrfCrossCameraProjectedCorridorGate.classify(route, farRoad, frame, 0.03).relation
        )
    }

    @Test
    fun exactBoundaryIsUncertainAndMalformedPolygonFailsClosed() {
        // Bottom center is exactly the first projected-corridor vertex: (0.56, 0.98).
        val boundary = BoundingBox(348.4f, 300f, 368.4f, 352.8f)
        val boundaryResult = UstrfCrossCameraProjectedCorridorGate.classify(route, boundary, frame, 0.03)
        assertEquals(CrossCameraCorridorRelation.UNCERTAIN_BOUNDARY, boundaryResult.relation)
        assertThrows(IllegalArgumentException::class.java) {
            UstrfCrossCameraProjectedCorridorGate.classify(
                listOf(
                    CrossCameraCorridorPoint(0.5, 0.5),
                    CrossCameraCorridorPoint(0.5, 0.5),
                    CrossCameraCorridorPoint(0.5, 0.5)
                ),
                boundary,
                frame,
                0.02
            )
        }
    }
}
