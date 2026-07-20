package com.linnan.blindassist.ustrf

import kotlin.math.PI
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfRiskFieldWarpTest {
    @Test
    fun verifiedForwardMotionMovesStaticEvidenceTowardTheUserBeforeFusion() {
        val builder = UstrfRiskFieldBuilder(UstrfRiskFieldConfig(staticLifetimeNs = 10_000L, dynamicLifetimeNs = 1_000L))
        val first = frame(1L, 100L)
        builder.update(packet(first, UstrfGridCoordinate(0, 3), .8f, "depth"))
        val second = frame(2L, 200L)

        val field = builder.update(
            UstrfPerceptionPacket(second, 200L, 500L, emptyList()),
            UstrfVerifiedPoseDelta(first, second, forwardMeters = 1f, lateralMeters = 0f, yawRadians = 0f, verifiedForOfflineReplay = true)
        )

        assertEquals(.792f, field.cellAt(UstrfGridCoordinate(0, 2)).occupancy, .0001f)
        assertEquals(100L, field.cellAt(UstrfGridCoordinate(0, 2)).ageNs)
        assertEquals(0f, field.cellAt(UstrfGridCoordinate(0, 3)).occupancy, .0001f)
    }

    @Test
    fun verifiedLeftTurnRotatesPriorEvidenceIntoTheCurrentBodyFrame() {
        val builder = UstrfRiskFieldBuilder()
        val first = frame(1L, 100L)
        builder.update(packet(first, UstrfGridCoordinate(0, 2), .7f, "depth"))
        val second = frame(2L, 200L)

        val field = builder.update(
            UstrfPerceptionPacket(second, 200L, 500L, emptyList()),
            UstrfVerifiedPoseDelta(first, second, 0f, 0f, (PI / 2.0).toFloat(), verifiedForOfflineReplay = true)
        )

        assertEquals(.7f, field.cellAt(UstrfGridCoordinate(-2, 0)).occupancy, .0001f)
    }

    @Test
    fun poseWarpFailsClosedForUnverifiedOrFrameMismatchedReceipts() {
        val builder = UstrfRiskFieldBuilder()
        val first = frame(1L, 100L)
        builder.update(packet(first, UstrfGridCoordinate(0, 1), .7f, "depth"))
        val second = frame(2L, 200L)
        val unverified = UstrfVerifiedPoseDelta(first, second, 0f, 0f, 0f, verifiedForOfflineReplay = false)
        try {
            builder.update(UstrfPerceptionPacket(second, 200L, 500L, emptyList()), unverified)
            throw AssertionError("expected unverified pose receipt to fail")
        } catch (error: IllegalArgumentException) {
            assertTrue(error.message!!.contains("verified pose"))
        }

        val third = frame(3L, 300L)
        val mismatched = UstrfVerifiedPoseDelta(first, third, 0f, 0f, 0f, verifiedForOfflineReplay = true)
        try {
            builder.update(UstrfPerceptionPacket(second, 200L, 500L, emptyList()), mismatched)
            throw AssertionError("expected frame-mismatched pose receipt to fail")
        } catch (error: IllegalArgumentException) {
            assertTrue(error.message!!.contains("current perception frame"))
        }
    }

    private fun frame(id: Long, atNs: Long) = UstrfFrameStamp(id, atNs, "offline-synthetic-v1")

    private fun packet(frame: UstrfFrameStamp, coordinate: UstrfGridCoordinate, occupancy: Float, source: String) =
        UstrfPerceptionPacket(frame, frame.capturedAtNs, frame.capturedAtNs + 1_000L, listOf(
            UstrfRiskObservation(coordinate, occupancy, 0f, 0f, 0f, null, 0f, source, frame.capturedAtNs + 1_000L)
        ))
}
