package com.linnan.blindassist.goal

import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CausalActionGeometryTest {
    private val points = listOf(
        ActionVector3(-0.04, -0.02, 0.00),
        ActionVector3(0.04, -0.02, 0.01),
        ActionVector3(-0.03, 0.03, 0.02),
        ActionVector3(0.03, 0.04, -0.01),
        ActionVector3(0.00, -0.04, 0.04),
        ActionVector3(0.01, 0.01, -0.04)
    )

    @Test
    fun pairedTranslationLocksDirection() {
        val shift = ActionVector3(0.02, -0.01, 0.0)
        val estimate = CausalActionGeometryEstimator().estimate(
            points.map { ActionPointPair(it, it + shift) }
        )

        assertEquals(ActionGeometryBeliefState.LOCKED, estimate.state)
        assertEquals(ActionMotionType.TRANSLATION, estimate.motionType)
        assertTrue(requireNotNull(estimate.axis).x > 0.89)
        assertTrue(requireNotNull(estimate.rmsResidualMeters) < 1e-6)
    }

    @Test
    fun pairedRotationLocksAxisAndPivotLine() {
        val angle = 5.0 * PI / 180.0
        val pivot = ActionVector3(0.30, 0.10, 0.0)
        fun rotate(point: ActionVector3): ActionVector3 {
            val local = point - pivot
            return ActionVector3(
                pivot.x + cos(angle) * local.x - sin(angle) * local.y,
                pivot.y + sin(angle) * local.x + cos(angle) * local.y,
                point.z
            )
        }
        val estimate = CausalActionGeometryEstimator().estimate(
            points.map { ActionPointPair(it, rotate(it)) }
        )

        assertEquals(ActionGeometryBeliefState.LOCKED, estimate.state)
        assertEquals(ActionMotionType.ROTATION, estimate.motionType)
        assertTrue(requireNotNull(estimate.axis).z > 0.99)
        val recoveredPivot = requireNotNull(estimate.pivotLinePointMeters)
        assertEquals(pivot.x, recoveredPivot.x, 1e-4)
        assertEquals(pivot.y, recoveredPivot.y, 1e-4)
    }

    @Test
    fun staleCausalEvidenceRemainsUnknown() {
        val previous = frame(1L, 10L)
        val current = frame(2L, 20L)
        val evidence = CausalActionGeometryEvidence(
            sourceContractId = CausalActionGeometryAdmitter.CONTRACT_ID,
            sourceId = CausalActionGeometryAdmitter.PAIRED_RGBD_SOURCE_ID,
            goalId = "g",
            sessionId = "s",
            parentBindingId = "binding",
            previousFrame = previous,
            currentFrame = current,
            availableAtNs = 22L,
            validUntilNs = 25L,
            availabilityClockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME,
            pairs = points.map { ActionPointPair(it, it + ActionVector3(0.02, 0.0, 0.0)) }
        )

        val observation = CausalActionGeometryAdmitter.pairedRgbdSource().evaluate(
            evidence = evidence,
            goalId = "g",
            sessionId = "s",
            parentBindingId = "binding",
            currentFrame = current,
            decisionAtNs = 26L,
            decisionClockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
        )

        assertEquals(CausalActionGeometryDisposition.EVIDENCE_STALE, observation.disposition)
        assertEquals(ActionGeometryBeliefState.UNKNOWN, observation.state)
    }

    private fun frame(id: Long, capturedAtNs: Long) = FrameStamp(
        frameId = id,
        capturedAtNs = capturedAtNs,
        receivedAtNs = capturedAtNs + 1L,
        sourceId = "camera",
        coordinateFrame = "parent",
        clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
    )
}
