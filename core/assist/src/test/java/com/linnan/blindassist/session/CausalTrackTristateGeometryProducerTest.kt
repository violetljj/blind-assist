package com.linnan.blindassist.session

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.exp

class CausalTrackTristateGeometryProducerTest {
    private val frameSize = FrameSize(1000, 1000)

    @Test
    fun sevenUnanimouslyGrowingFramesConfirmApproach() {
        val result = runSeries { timeS -> 100f * exp(0.30f * timeS) }

        assertEquals(DualLoopCorrectionDecision.CONFIRM_APPROACH, result.correctionDecision)
        assertTrue(result.signedApproachRatePerS!! >= 0.2f)
        assertNull(result.sourceAbstentionReason)
    }

    @Test
    fun sevenUnanimouslyShrinkingFramesContradictApproach() {
        val result = runSeries { timeS -> 140f * exp(-0.30f * timeS) }

        assertEquals(DualLoopCorrectionDecision.CONTRADICT_APPROACH, result.correctionDecision)
        assertTrue(result.signedApproachRatePerS!! <= -0.2f)
        assertNull(result.sourceAbstentionReason)
    }

    @Test
    fun mixedDirectionAbstainsEvenWhenOverallSlopeIsPositive() {
        val heights = listOf(100f, 103f, 106f, 105f, 112f, 116f, 120f)
        val producer = CausalTrackTristateGeometryProducer()
        var result: DualLoopGeometryEvidence? = null
        heights.forEachIndexed { index, height ->
            result = producer.produce(stamp(index), detection(height), decisionAtNs(index))
        }

        assertEquals(DualLoopCorrectionDecision.ABSTAIN, result?.correctionDecision)
        assertEquals("TREND_NOT_SELECTIVE", result?.sourceAbstentionReason)
    }

    @Test
    fun targetSwitchAndLargeGapStartNewEpochAndHistory() {
        val producer = CausalTrackTristateGeometryProducer()
        val first = producer.produce(stamp(0), detection(100f), decisionAtNs(0))!!
        val switched = producer.produce(
            stamp(1),
            detection(104f).copy(classId = 1, label = "bicycle"),
            decisionAtNs(1)
        )!!
        val afterGap = producer.produce(
            stamp(10, capturedAtNs = 2_000_000_000L),
            detection(108f).copy(classId = 1, label = "bicycle"),
            2_010_000_000L
        )!!

        assertNotEquals(first.trackEpoch, switched.trackEpoch)
        assertNotEquals(switched.trackEpoch, afterGap.trackEpoch)
        assertEquals("INSUFFICIENT_CONTIGUOUS_HISTORY", switched.sourceAbstentionReason)
        assertEquals("INSUFFICIENT_CONTIGUOUS_HISTORY", afterGap.sourceAbstentionReason)
    }

    private fun runSeries(heightAt: (Float) -> Float): DualLoopGeometryEvidence {
        val producer = CausalTrackTristateGeometryProducer()
        var result: DualLoopGeometryEvidence? = null
        repeat(7) { index ->
            val timeS = index * 0.1f
            result = producer.produce(
                stamp(index),
                detection(heightAt(timeS)),
                decisionAtNs(index)
            )
        }
        return requireNotNull(result)
    }

    private fun detection(height: Float) = Detection(
        classId = 0,
        label = "person",
        confidence = 0.95f,
        boundingBox = BoundingBox(400f, 100f, 600f, 100f + height),
        frameSize = frameSize
    )

    private fun stamp(
        index: Int,
        capturedAtNs: Long = 1_000_000_000L + index * 100_000_000L
    ) = FrameStamp(
        frameId = index.toLong(),
        capturedAtNs = capturedAtNs,
        receivedAtNs = capturedAtNs + 1_000_000L,
        sourceId = "camera2:0",
        coordinateFrame = "camera2:0:analysis-buffer",
        clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
    )

    private fun decisionAtNs(index: Int) =
        1_010_000_000L + index * 100_000_000L
}
