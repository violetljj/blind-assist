package com.linnan.blindassist.session

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DualLoopJrdbReplayAdapterTest {
    @Test
    fun exactReplayAnnotationInputProducesSourceBoundShadowEvidence() {
        val proposal = DualLoopJrdbReplayAdapter.adapt(input())

        assertTrue(proposal is DualLoopJrdbReplayProposal.Available)
        val evidence = (proposal as DualLoopJrdbReplayProposal.Available).evidence
        assertEquals(DualLoopJrdbReplayAdapter.SOURCE_ID, evidence.sourceId)
        assertEquals(DetectionSource.OBJECT_DETECTOR, evidence.targetSource)
        assertEquals(
            DualLoopTargetProvenance.REPLAY_ANNOTATION,
            evidence.targetProvenance
        )
        assertEquals(FrameClockDomain.REPLAY_TIMELINE, evidence.availabilityClockDomain)
    }

    @Test
    fun liveOrDetectorProvenanceFailsClosed() {
        val live = DualLoopJrdbReplayAdapter.adapt(
            input().copy(
                currentFrame = stamp(2L, 2_000L, FrameClockDomain.ANDROID_ELAPSED_REALTIME)
            )
        )
        val detector = DualLoopJrdbReplayAdapter.adapt(
            input().copy(
                target = target().copy(source = DetectionSource.SEGMENTATION)
            )
        )

        assertEquals(
            "REPLAY_TIMELINE_REQUIRED",
            (live as DualLoopJrdbReplayProposal.Abstained).reason
        )
        assertEquals(
            "OBJECT_DETECTOR_BEHAVIOR_REQUIRED",
            (detector as DualLoopJrdbReplayProposal.Abstained).reason
        )
    }

    private fun input() = JrdbAnnotationConditionedLidarReplayInput(
        previousFrame = stamp(1L, 1_000L),
        currentFrame = stamp(2L, 2_000L),
        availableAtNs = 2_100L,
        validUntilNs = 102_100L,
        trackEpoch = "sequence:pedestrian:1",
        target = target(),
        signedApproachRatePerS = 0.4f,
        quality = 1f
    )

    private fun target() = Detection(
        classId = 0,
        label = "person",
        confidence = 1f,
        boundingBox = BoundingBox(10f, 20f, 30f, 40f),
        frameSize = FrameSize(3760, 480),
        source = DetectionSource.OBJECT_DETECTOR
    )

    private fun stamp(
        frameId: Long,
        capturedAtNs: Long,
        clockDomain: FrameClockDomain = FrameClockDomain.REPLAY_TIMELINE
    ) = FrameStamp(
        frameId = frameId,
        capturedAtNs = capturedAtNs,
        receivedAtNs = capturedAtNs,
        sourceId = "jrdb-sequence",
        coordinateFrame = DualLoopJrdbReplayAdapter.COORDINATE_FRAME,
        clockDomain = clockDomain
    )
}
