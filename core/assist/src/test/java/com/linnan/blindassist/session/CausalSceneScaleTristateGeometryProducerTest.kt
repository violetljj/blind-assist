package com.linnan.blindassist.session

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CausalSceneScaleTristateGeometryProducerTest {
    private val frameSize = FrameSize(1000, 1000)

    @Test
    fun twoUniquelyMatchedShrinkingTargetsContradict() {
        val producer = CausalSceneScaleTristateGeometryProducer()
        producer.produce(
            stamp(0),
            listOf(detection(0, 250f, 350f), detection(1, 200f, 700f)),
            selectedTarget = detection(0, 250f, 350f),
            decisionAtNs = 1_010_000_000L
        )
        val selected = detection(0, 225f, 350f)
        val result = producer.produce(
            stamp(1),
            listOf(selected, detection(1, 180f, 700f)),
            selectedTarget = selected,
            decisionAtNs = 1_110_000_000L
        )!!

        assertEquals(DualLoopCorrectionDecision.CONTRADICT_APPROACH, result.correctionDecision)
        assertTrue(result.signedApproachRatePerS!! < -0.05f)
        assertTrue(result.quality!! >= 0.5f)
        assertNull(result.sourceAbstentionReason)
    }

    @Test
    fun oneMatchOrCollectiveGrowthAbstains() {
        val producer = CausalSceneScaleTristateGeometryProducer()
        producer.produce(
            stamp(0),
            listOf(detection(0, 200f, 350f)),
            selectedTarget = detection(0, 200f, 350f),
            decisionAtNs = 1_010_000_000L
        )
        val selected = detection(0, 220f, 350f)
        val oneMatch = producer.produce(
            stamp(1),
            listOf(selected),
            selectedTarget = selected,
            decisionAtNs = 1_110_000_000L
        )!!
        assertEquals(DualLoopCorrectionDecision.ABSTAIN, oneMatch.correctionDecision)
        assertEquals("INSUFFICIENT_SCENE_MATCHES", oneMatch.sourceAbstentionReason)

        producer.reset()
        val first = listOf(detection(0, 200f, 350f), detection(1, 160f, 700f))
        producer.produce(stamp(0), first, first[0], 1_010_000_000L)
        val second = listOf(detection(0, 220f, 350f), detection(1, 180f, 700f))
        val growth = producer.produce(stamp(1), second, second[0], 1_110_000_000L)!!
        assertEquals(DualLoopCorrectionDecision.ABSTAIN, growth.correctionDecision)
        assertEquals("SCENE_NOT_COLLECTIVELY_RECEDING", growth.sourceAbstentionReason)
    }

    @Test
    fun rejectedPairDoesNotConsumeTrackNeededByLaterUniqueMatch() {
        val producer = CausalSceneScaleTristateGeometryProducer()
        val previous = listOf(
            boxDetection(300f, 500f, 400f),
            boxDetection(500f, 700f, 400f)
        )
        producer.produce(stamp(0), previous, previous[0], 1_010_000_000L)
        val current = listOf(
            boxDetection(400f, 600f, 350f),
            boxDetection(610f, 810f, 350f)
        )
        val result = producer.produce(stamp(1), current, current[0], 1_110_000_000L)!!

        assertEquals(DualLoopCorrectionDecision.CONTRADICT_APPROACH, result.correctionDecision)
        assertTrue(result.signedApproachRatePerS!! < -0.05f)
    }

    @Test
    fun bidirectionalSourceConfirmsCollectiveGrowthAndKeepsDeadband() {
        val producer = CausalSceneScaleTristateGeometryProducer.bidirectional()
        val first = listOf(detection(0, 200f, 350f), detection(1, 160f, 700f))
        producer.produce(stamp(0), first, first[0], 1_010_000_000L)
        val growing = listOf(detection(0, 220f, 350f), detection(1, 180f, 700f))
        val confirm = producer.produce(
            stamp(1),
            growing,
            growing[0],
            1_110_000_000L
        )!!

        assertEquals(
            CausalSceneScaleTristateGeometryProducer.BIDIRECTIONAL_SOURCE_ID,
            confirm.sourceId
        )
        assertEquals(
            DualLoopCorrectionDecision.CONFIRM_APPROACH,
            confirm.correctionDecision
        )
        assertTrue(confirm.signedApproachRatePerS!! > 0.05f)
        assertNull(confirm.sourceAbstentionReason)

        producer.reset()
        val deadbandFirst = listOf(
            detection(0, 200f, 350f),
            detection(1, 160f, 700f)
        )
        producer.produce(
            stamp(0),
            deadbandFirst,
            deadbandFirst[0],
            1_010_000_000L
        )
        val deadbandSecond = listOf(
            detection(0, 200.5f, 350f),
            detection(1, 160.4f, 700f)
        )
        val abstain = producer.produce(
            stamp(1),
            deadbandSecond,
            deadbandSecond[0],
            1_110_000_000L
        )!!
        assertEquals(
            DualLoopCorrectionDecision.ABSTAIN,
            abstain.correctionDecision
        )
        assertEquals(
            "SCENE_RATE_IN_DEADBAND",
            abstain.sourceAbstentionReason
        )
    }

    private fun detection(classId: Int, height: Float, centerX: Float) = Detection(
        classId = classId,
        label = if (classId == 0) "person" else "car",
        confidence = 0.95f,
        boundingBox = BoundingBox(centerX - 80f, 100f, centerX + 80f, 100f + height),
        frameSize = frameSize
    )

    private fun boxDetection(left: Float, right: Float, height: Float) = Detection(
        classId = 0,
        label = "person",
        confidence = 0.95f,
        boundingBox = BoundingBox(left, 100f, right, 100f + height),
        frameSize = frameSize
    )

    private fun stamp(index: Int) = FrameStamp(
        frameId = index.toLong(),
        capturedAtNs = 1_000_000_000L + index * 100_000_000L,
        receivedAtNs = 1_001_000_000L + index * 100_000_000L,
        sourceId = "camera2:0",
        coordinateFrame = "camera2:0:analysis-buffer",
        clockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
    )
}
