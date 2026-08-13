package com.linnan.blindassist.ustrfbenchmark

import com.linnan.blindassist.ustrf.TaroPoseDiverseSelection
import com.linnan.blindassist.ustrf.TaroPoseDiverseSelectionFailure
import com.linnan.blindassist.ustrf.UstrfFrameStamp
import com.linnan.blindassist.ustrf.UstrfPoseSample
import com.linnan.blindassist.ustrf.UstrfPoseState
import com.linnan.blindassist.ustrf.UstrfVector3
import com.linnan.blindassist.ustrf.UstrfVioPoseAdmission
import com.linnan.blindassist.ustrf.UstrfVioPoseAdmissionFailure
import org.junit.Assert.assertEquals
import org.junit.Test

class TaroPoseDiverseCanaryHarnessTest {
    @Test
    fun unavailableVioAdmissionNeverEntersBuffer() {
        val harness = TaroPoseDiverseCanaryHarness()
        val rejectedFrame = frame(1L, 100_000_000L)
        val rejected = harness.observe(
            rejectedFrame,
            UstrfVioPoseAdmission.Unavailable(UstrfVioPoseAdmissionFailure.WORLD_FRAME_NOT_INTERFRAME_STABLE)
        ) as TaroPoseDiverseCanaryStep.AdmissionRejected
        val firstAvailable = frame(2L, 400_000_000L)
        val evaluated = harness.observe(firstAvailable, available(firstAvailable, x = .1f)) as TaroPoseDiverseCanaryStep.Evaluated

        assertEquals(0, rejected.bufferedFrameCount)
        assertEquals(
            TaroPoseDiverseSelection.Unavailable(TaroPoseDiverseSelectionFailure.NO_ELIGIBLE_BUFFERED_FRAME),
            evaluated.selection
        )
        assertEquals(1, evaluated.bufferedFrameCountAfterAppend)
    }

    @Test
    fun admittedPoseReceiptsDriveSelectorWithoutPixelPayload() {
        val harness = TaroPoseDiverseCanaryHarness()
        val first = frame(1L, 100_000_000L)
        val second = frame(2L, 400_000_000L)
        harness.observe(first, available(first, x = 0f))
        val step = harness.observe(second, available(second, x = .5f)) as TaroPoseDiverseCanaryStep.Evaluated
        val selected = step.selection as TaroPoseDiverseSelection.Available

        assertEquals(first, selected.selectedFrame)
        assertEquals(.5f, selected.translationM, .0001f)
        assertEquals(300_000_000L, selected.gapNs)
        assertEquals(2, step.bufferedFrameCountAfterAppend)
    }

    @Test
    fun expiredHistoryIsRemovedBeforeSelection() {
        val harness = TaroPoseDiverseCanaryHarness(maximumRetainedAgeNs = 1_000_000_000L)
        val old = frame(1L, 100_000_000L)
        val current = frame(2L, 1_200_000_001L)
        harness.observe(old, available(old, x = 0f))
        val step = harness.observe(current, available(current, x = 1f)) as TaroPoseDiverseCanaryStep.Evaluated

        assertEquals(
            TaroPoseDiverseSelection.Unavailable(TaroPoseDiverseSelectionFailure.NO_ELIGIBLE_BUFFERED_FRAME),
            step.selection
        )
        assertEquals(1, step.bufferedFrameCountAfterAppend)
    }

    private fun frame(id: Long, timestampNs: Long) = UstrfFrameStamp(id, timestampNs, "arcore-camera-v1")

    private fun available(frame: UstrfFrameStamp, x: Float) = UstrfVioPoseAdmission.Available(
        cameraPose = UstrfPoseSample(
            timestampNs = frame.capturedAtNs,
            worldFrame = "arcore-session-anchor-v1",
            cameraFrame = frame.coordinateFrame,
            worldCameraTranslationM = UstrfVector3(x, 0f, 0f),
            yawRad = 0f,
            gravityWorld = UstrfVector3(0f, -9.80665f, 0f),
            tracking = UstrfPoseState.TRACKING,
            confidence = .95f,
            validUntilNs = frame.capturedAtNs + 1_000_000_000L
        ),
        verifiedBodyFrame = "phone-body-v1"
    )
}
