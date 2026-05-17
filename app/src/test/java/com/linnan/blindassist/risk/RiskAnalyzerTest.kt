package com.linnan.blindassist.risk

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RiskAnalyzerTest {
    private val analyzer = RiskAnalyzer()
    private val frame = FrameSize(1000, 1000)

    @Test
    fun centerCriticalPersonIsHighRisk() {
        val result = analyzer.analyze(
            listOf(detection("person", BoundingBox(390f, 140f, 610f, 780f))),
            frame
        )

        assertEquals(RiskLevel.HIGH, result.level)
        assertEquals(RiskDirection.CENTER, result.direction)
        assertEquals(ProximityBand.CRITICAL, result.proximity)
        assertEquals("前方迫近 有人", result.message)
        assertTrue(result.urgencyScore > 0f)
    }

    @Test
    fun leftNearCarIsMediumRisk() {
        val result = analyzer.analyze(
            listOf(detection("car", BoundingBox(20f, 330f, 260f, 720f))),
            frame
        )

        assertEquals(RiskLevel.MEDIUM, result.level)
        assertEquals(RiskDirection.LEFT, result.direction)
        assertEquals(ProximityBand.NEAR, result.proximity)
        assertEquals("左前方近处 有车辆", result.message)
    }

    @Test
    fun centerNearVehicleIsHighRisk() {
        val result = analyzer.analyze(
            listOf(detection("bus", BoundingBox(410f, 330f, 590f, 650f))),
            frame
        )

        assertEquals(RiskLevel.HIGH, result.level)
        assertEquals(RiskDirection.CENTER, result.direction)
        assertEquals(ProximityBand.NEAR, result.proximity)
    }

    @Test
    fun midDistanceTargetIsLowRisk() {
        val result = analyzer.analyze(
            listOf(detection("chair", BoundingBox(450f, 240f, 560f, 500f))),
            frame
        )

        assertEquals(RiskLevel.LOW, result.level)
        assertEquals(RiskDirection.CENTER, result.direction)
        assertEquals(ProximityBand.MID, result.proximity)
        assertEquals("前方中距 有障碍", result.message)
    }

    @Test
    fun farTargetDoesNotTriggerFeedbackLevelRisk() {
        val result = analyzer.analyze(
            listOf(detection("person", BoundingBox(450f, 120f, 520f, 280f))),
            frame
        )

        assertEquals(RiskLevel.NONE, result.level)
        assertEquals(ProximityBand.FAR, result.proximity)
        assertEquals("未发现风险", result.message)
    }

    @Test
    fun highestUrgencyTargetWinsOverHigherConfidenceFarTarget() {
        val farHighConfidence = detection(
            label = "person",
            box = BoundingBox(450f, 120f, 520f, 280f),
            confidence = 0.99f
        )
        val nearLowerConfidence = detection(
            label = "car",
            box = BoundingBox(20f, 330f, 260f, 720f),
            confidence = 0.72f
        )

        val result = analyzer.analyze(listOf(farHighConfidence, nearLowerConfidence), frame)

        assertEquals("car", result.sourceDetection?.label)
        assertEquals(ProximityBand.NEAR, result.proximity)
        assertEquals(RiskLevel.MEDIUM, result.level)
    }

    @Test
    fun lowConfidenceIsIgnored() {
        val result = analyzer.analyze(
            listOf(detection("person", BoundingBox(420f, 300f, 580f, 760f), confidence = 0.2f)),
            frame
        )

        assertEquals(RiskLevel.NONE, result.level)
        assertEquals(ProximityBand.FAR, result.proximity)
    }

    @Test
    fun unrelatedClassIsIgnored() {
        val result = analyzer.analyze(
            listOf(detection("banana", BoundingBox(420f, 300f, 580f, 760f))),
            frame
        )

        assertEquals(RiskLevel.NONE, result.level)
    }

    private fun detection(
        label: String,
        box: BoundingBox,
        confidence: Float = 0.9f
    ): Detection {
        return Detection(
            classId = 0,
            label = label,
            confidence = confidence,
            boundingBox = box,
            frameSize = frame
        )
    }
}
