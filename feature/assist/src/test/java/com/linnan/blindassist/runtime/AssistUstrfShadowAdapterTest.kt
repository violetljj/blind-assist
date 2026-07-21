package com.linnan.blindassist.runtime

import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.session.DetectorMetrics
import com.linnan.blindassist.ustrf.UstrfSafetyAction
import com.linnan.blindassist.ustrf.UstrfSafetyReason
import com.linnan.blindassist.vision.DetectorFrameResult
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AssistUstrfShadowAdapterTest {
    @Test
    fun liveStampedFrameIsRecordedFailClosedWithoutBecomingFeedbackAuthority() {
        val events = mutableListOf<AssistUstrfShadowEvent>()
        val adapter = AssistUstrfShadowAdapter(sink = AssistUstrfShadowSink(events::add))

        adapter.observe(frame(FrameClockDomain.ANDROID_ELAPSED_REALTIME), decisionAtNs = 1_200L)

        val event = events.single() as AssistUstrfShadowEvent.Recorded
        assertEquals(UstrfSafetyAction.STOP_AND_REASSESS, event.record.decision.action)
        assertTrue(UstrfSafetyReason.GEOMETRY_UNAVAILABLE in event.record.decision.reasons)
        assertTrue(event.record.structuredOutput.shadowOnly)
        assertEquals(null, event.record.field)
    }

    @Test
    fun unmappedCameraClockAbstainsInsteadOfComparingDifferentTimeDomains() {
        val events = mutableListOf<AssistUstrfShadowEvent>()
        val adapter = AssistUstrfShadowAdapter(sink = AssistUstrfShadowSink(events::add))

        adapter.observe(frame(FrameClockDomain.CAMERA_HARDWARE_UNMAPPED), decisionAtNs = 1_200L)

        val event = events.single() as AssistUstrfShadowEvent.Abstained
        assertEquals("camera_clock_domain_unmapped", event.reason)
    }

    private fun frame(clockDomain: FrameClockDomain) = DetectorFrameResult(
        detections = emptyList(),
        frameSize = FrameSize(640, 480),
        metrics = DetectorMetrics(1L, 0L, 1L, 0L, 30f, "ready"),
        sourceFrame = FrameStamp(
            frameId = 7L,
            capturedAtNs = 1_000L,
            receivedAtNs = 1_100L,
            sourceId = "camera2:0",
            coordinateFrame = "camera2:0:analysis-buffer",
            clockDomain = clockDomain
        )
    )
}
