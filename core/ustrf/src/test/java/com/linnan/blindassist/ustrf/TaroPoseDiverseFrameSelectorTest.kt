package com.linnan.blindassist.ustrf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TaroPoseDiverseFrameSelectorTest {
    @Test
    fun defaultIsFailClosedAndCannotSelect() {
        val reference = frame(10L, 1_000_000_000L)
        val result = TaroPoseDiverseFrameSelector().select(reference, pose(reference), emptyList())

        assertEquals(
            TaroPoseDiverseSelection.Unavailable(TaroPoseDiverseSelectionFailure.DISABLED),
            result
        )
    }

    @Test
    fun enabledSelectorMaximizesTranslationThenYaw() {
        val reference = frame(10L, 1_000_000_000L)
        val smaller = frame(7L, 500_000_000L)
        val sameTranslationLowerYaw = frame(8L, 600_000_000L)
        val selected = frame(9L, 700_000_000L)
        val result = TaroPoseDiverseFrameSelector(enabled = true).select(
            reference,
            pose(reference, x = 0f),
            listOf(
                TaroBufferedPoseFrame(smaller, pose(smaller, x = .4f, yaw = 1f)),
                TaroBufferedPoseFrame(sameTranslationLowerYaw, pose(sameTranslationLowerYaw, x = .6f, yaw = .2f)),
                TaroBufferedPoseFrame(selected, pose(selected, x = .6f, yaw = .4f))
            )
        ) as TaroPoseDiverseSelection.Available

        assertEquals(selected, result.selectedFrame)
        assertEquals(.6f, result.translationM, .0001f)
        assertEquals(.4f, result.yawDeltaRad, .0001f)
        assertEquals(TaroPoseDiverseFrameSelector.POLICY_ID, result.policyId)
    }

    @Test
    fun selectorRejectsFutureStaleLowConfidenceAndFrameMismatchedCandidates() {
        val reference = frame(10L, 2_000_000_000L)
        val tooOld = frame(1L, 0L)
        val tooRecent = frame(9L, 1_900_000_001L)
        val lowConfidence = frame(8L, 1_700_000_000L)
        val mismatch = UstrfFrameStamp(7L, 1_600_000_000L, "other-camera")
        val result = TaroPoseDiverseFrameSelector(enabled = true).select(
            reference,
            pose(reference),
            listOf(
                TaroBufferedPoseFrame(tooOld, pose(tooOld, x = 4f)),
                TaroBufferedPoseFrame(tooRecent, pose(tooRecent, x = 3f)),
                TaroBufferedPoseFrame(lowConfidence, pose(lowConfidence, x = 2f, confidence = .5f)),
                TaroBufferedPoseFrame(mismatch, pose(mismatch, x = 1f))
            )
        )

        assertEquals(
            TaroPoseDiverseSelection.Unavailable(TaroPoseDiverseSelectionFailure.NO_ELIGIBLE_BUFFERED_FRAME),
            result
        )
    }

    @Test
    fun invalidReferencePoseFailsClosed() {
        val reference = frame(10L, 1_000_000_000L)
        val result = TaroPoseDiverseFrameSelector(enabled = true).select(
            reference,
            pose(reference, tracking = UstrfPoseState.LOST),
            emptyList()
        )

        assertEquals(
            TaroPoseDiverseSelection.Unavailable(TaroPoseDiverseSelectionFailure.REFERENCE_POSE_INVALID),
            result
        )
    }

    @Test
    fun tieBreakPrefersSmallerGapThenStableFrameId() {
        val reference = frame(10L, 1_000_000_000L)
        val older = frame(7L, 500_000_000L)
        val newerLowId = frame(8L, 700_000_000L)
        val newerHighId = frame(9L, 700_000_000L)
        val result = TaroPoseDiverseFrameSelector(enabled = true).select(
            reference,
            pose(reference),
            listOf(older, newerLowId, newerHighId).map { candidate ->
                TaroBufferedPoseFrame(candidate, pose(candidate, x = .5f, yaw = .2f))
            }
        ) as TaroPoseDiverseSelection.Available

        assertEquals(newerHighId, result.selectedFrame)
        assertTrue(result.gapNs < reference.capturedAtNs - older.capturedAtNs)
    }

    private fun frame(id: Long, timestampNs: Long) = UstrfFrameStamp(id, timestampNs, "camera-v1")

    private fun pose(
        frame: UstrfFrameStamp,
        x: Float = 0f,
        yaw: Float = 0f,
        confidence: Float = .9f,
        tracking: UstrfPoseState = UstrfPoseState.TRACKING
    ) = UstrfPoseSample(
        timestampNs = frame.capturedAtNs,
        worldFrame = "world-v1",
        cameraFrame = frame.coordinateFrame,
        worldCameraTranslationM = UstrfVector3(x, 0f, 0f),
        yawRad = yaw,
        gravityWorld = UstrfVector3(0f, -9.80665f, 0f),
        tracking = tracking,
        confidence = confidence,
        validUntilNs = frame.capturedAtNs + 1_000_000_000L
    )
}
