package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UstrfEgoCompensatedMotionTest {
    private val promoter = UstrfEgoCompensatedMotionPromoter()
    private val first = UstrfFrameStamp(1L, 1_000_000_000L, "offline-body")
    private val second = UstrfFrameStamp(2L, 2_000_000_000L, "offline-body")

    @Test
    fun subtractsUserForwardMotionBeforeEstimatingTargetRelativeVelocityAndTtc() {
        val result = promote(
            previous = UstrfVector2(3f, 0f),
            current = UstrfVector2(1f, 0f),
            delta = delta(forward = 1f)
        ) as UstrfEgoCompensatedMotionResolution.Available

        assertEquals(2f, result.egoCompensatedPreviousPositionMeters.forward)
        assertEquals(-1f, result.evidence.motion.relativeVelocityMetersPerSecond.forward)
        assertEquals(UstrfGridCoordinate(0, 1), result.evidence.coordinate)
        assertEquals(1_000L, UstrfTtcEstimator().estimate(result.evidence.motion, second.capturedAtNs)?.timeToClosestApproachMs)
    }

    @Test
    fun unverifiedOrWronglyBoundPoseCannotReachTtcEstimator() {
        val unverified = promote(UstrfVector2(3f, 0f), UstrfVector2(1f, 0f), delta(forward = 1f, verified = false))
        val wrongBinding = promoter.promote(
            pair(UstrfVector2(3f, 0f), UstrfVector2(1f, 0f)),
            UstrfVerifiedPoseDelta(UstrfFrameStamp(0L, 0L, "offline-body"), second, 1f, 0f, 0f, true),
            second.capturedAtNs
        )

        assertEquals(UstrfEgoCompensatedMotionFailure.POSE_DELTA_UNVERIFIED, (unverified as UstrfEgoCompensatedMotionResolution.Unavailable).failure)
        assertEquals(UstrfEgoCompensatedMotionFailure.POSE_DELTA_BINDING_MISMATCH, (wrongBinding as UstrfEgoCompensatedMotionResolution.Unavailable).failure)
    }

    @Test
    fun targetOutsideBodyGridIsRejectedInsteadOfCreatingAFalseTtcRisk() {
        val result = promote(UstrfVector2(3f, 0f), UstrfVector2(6f, 0f), delta())
        assertTrue(result is UstrfEgoCompensatedMotionResolution.Unavailable)
        assertEquals(UstrfEgoCompensatedMotionFailure.TARGET_OUTSIDE_LOCAL_GRID, (result as UstrfEgoCompensatedMotionResolution.Unavailable).failure)
    }

    private fun promote(previous: UstrfVector2, current: UstrfVector2, delta: UstrfVerifiedPoseDelta) =
        promoter.promote(pair(previous, current), delta, second.capturedAtNs)

    private fun pair(previous: UstrfVector2, current: UstrfVector2) = UstrfDynamicTrackPair(
        "target-1", first, second, previous, current, 1f, "offline-fixture", 3_000_000_000L
    )

    private fun delta(forward: Float = 0f, verified: Boolean = true) =
        UstrfVerifiedPoseDelta(first, second, forward, 0f, 0f, verified)
}
