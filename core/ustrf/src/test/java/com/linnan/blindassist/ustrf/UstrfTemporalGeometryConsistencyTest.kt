package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfTemporalGeometryConsistencyTest {
    @Test
    fun staticObstacleMatchesAfterVerifiedForwardEgoMotion() {
        val first = frame(1L, 1_000L)
        val second = frame(2L, 2_000L)
        val result = UstrfTemporalGeometryConsistency().compare(
            packet(first, geometry(2f, 0f)),
            packet(second, geometry(1.5f, 0f)),
            UstrfVerifiedPoseDelta(first, second, .5f, 0f, 0f, true),
            2_000L
        )

        assertTrue(result is UstrfTemporalGeometryConsistencyResult.Available)
        val available = result as UstrfTemporalGeometryConsistencyResult.Available
        assertEquals(1, available.matches.size)
        assertEquals(0f, available.matches.single().planarResidualMeters, .0001f)
        assertEquals(0, available.unmatchedPriorCount)
        assertEquals(0, available.unmatchedCurrentCount)
    }

    @Test
    fun unverifiedOrWrongFramePoseCannotCreateTemporalMatch() {
        val first = frame(1L, 1_000L)
        val second = frame(2L, 2_000L)
        assertEquals(
            UstrfTemporalGeometryConsistencyResult.Unavailable(UstrfTemporalGeometryConsistencyFailure.POSE_NOT_VERIFIED),
            UstrfTemporalGeometryConsistency().compare(
                packet(first, geometry(2f, 0f)), packet(second, geometry(1.5f, 0f)),
                UstrfVerifiedPoseDelta(first, second, .5f, 0f, 0f, false), 2_000L
            )
        )
        assertEquals(
            UstrfTemporalGeometryConsistencyResult.Unavailable(UstrfTemporalGeometryConsistencyFailure.POSE_FRAME_MISMATCH),
            UstrfTemporalGeometryConsistency().compare(
                packet(first, geometry(2f, 0f)), packet(second, geometry(1.5f, 0f)),
                UstrfVerifiedPoseDelta(frame(0L, 500L), second, .5f, 0f, 0f, true), 2_000L
            )
        )
    }

    @Test
    fun differentGeometryKindsAndDropsAreNotUsedAsTemporalConfirmation() {
        val first = frame(1L, 1_000L)
        val second = frame(2L, 2_000L)
        val result = UstrfTemporalGeometryConsistency().compare(
            packet(first, geometry(2f, 0f, UstrfGeometryKind.DROP)),
            packet(second, geometry(1.5f, 0f, UstrfGeometryKind.DROP)),
            UstrfVerifiedPoseDelta(first, second, .5f, 0f, 0f, true),
            2_000L
        ) as UstrfTemporalGeometryConsistencyResult.Available

        assertEquals(0, result.matches.size)
        assertEquals(0, result.unmatchedPriorCount)
        assertEquals(0, result.unmatchedCurrentCount)
    }

    private fun frame(id: Long, timestampNs: Long) = UstrfFrameStamp(id, timestampNs, "offline-camera")

    private fun packet(frame: UstrfFrameStamp, evidence: UstrfMetricGeometryEvidence) = UstrfGeometryPacket(
        frame, frame.capturedAtNs, frame.capturedAtNs + 2_000L, UstrfDepthScale.METRIC, listOf(evidence)
    )

    private fun geometry(
        forward: Float,
        lateral: Float,
        kind: UstrfGeometryKind = UstrfGeometryKind.OCCUPIED
    ) = UstrfMetricGeometryEvidence(
        forwardMeters = forward,
        lateralMeters = lateral,
        heightBand = if (kind == UstrfGeometryKind.DROP) UstrfHeightBand.GROUND else UstrfHeightBand.LOWER_BODY,
        kind = kind,
        confidence = .9f,
        source = "fixture",
        validUntilNs = 4_000L
    )
}
