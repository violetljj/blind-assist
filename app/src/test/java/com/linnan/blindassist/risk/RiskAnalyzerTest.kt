package com.linnan.blindassist.risk

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import org.junit.Assert.assertEquals
import org.junit.Test

class RiskAnalyzerTest {
    private val analyzer = RiskAnalyzer()
    private val frame = FrameSize(1000, 1000)

    @Test
    fun centerNearPersonIsHighRisk() {
        val result = analyzer.analyze(
            listOf(detection("person", BoundingBox(420f, 300f, 580f, 760f))),
            frame
        )

        assertEquals(RiskLevel.HIGH, result.level)
        assertEquals(RiskDirection.CENTER, result.direction)
        assertEquals("前方有人", result.message)
    }

    @Test
    fun leftNearCarIsMediumRisk() {
        val result = analyzer.analyze(
            listOf(detection("car", BoundingBox(20f, 330f, 260f, 720f))),
            frame
        )

        assertEquals(RiskLevel.MEDIUM, result.level)
        assertEquals(RiskDirection.LEFT, result.direction)
        assertEquals("左前方有车辆", result.message)
    }

    @Test
    fun lowConfidenceIsIgnored() {
        val result = analyzer.analyze(
            listOf(
                Detection(
                    classId = 0,
                    label = "person",
                    confidence = 0.2f,
                    boundingBox = BoundingBox(420f, 300f, 580f, 760f),
                    frameSize = frame
                )
            ),
            frame
        )

        assertEquals(RiskLevel.NONE, result.level)
    }

    @Test
    fun unrelatedClassIsIgnored() {
        val result = analyzer.analyze(
            listOf(detection("banana", BoundingBox(420f, 300f, 580f, 760f))),
            frame
        )

        assertEquals(RiskLevel.NONE, result.level)
    }

    private fun detection(label: String, box: BoundingBox): Detection {
        return Detection(
            classId = 0,
            label = label,
            confidence = 0.9f,
            boundingBox = box,
            frameSize = frame
        )
    }
}
