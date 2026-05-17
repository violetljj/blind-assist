package com.linnan.blindassist.risk

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import org.junit.Assert.assertEquals
import org.junit.Test

class RiskStabilizerTest {
    private val frame = FrameSize(1000, 1000)

    @Test
    fun highRiskConfirmsImmediately() {
        val stabilizer = RiskStabilizer()

        val stable = stabilizer.update(risk(RiskLevel.HIGH, RiskDirection.CENTER, "前方有人"), nowMs = 100L)

        assertEquals(RiskLevel.HIGH, stable.level)
        assertEquals(RiskDirection.CENTER, stable.direction)
        assertEquals("前方有人", stable.message)
    }

    @Test
    fun mediumRiskRequiresTwoMatchingFrames() {
        val stabilizer = RiskStabilizer()
        val medium = risk(RiskLevel.MEDIUM, RiskDirection.LEFT, "左前方有车辆")

        val first = stabilizer.update(medium, nowMs = 100L)
        val second = stabilizer.update(medium, nowMs = 130L)

        assertEquals(RiskLevel.NONE, first.level)
        assertEquals(RiskLevel.MEDIUM, second.level)
        assertEquals(RiskDirection.LEFT, second.direction)
    }

    @Test
    fun singleMediumFrameDoesNotTriggerAfterDisappearing() {
        val stabilizer = RiskStabilizer()

        stabilizer.update(risk(RiskLevel.MEDIUM, RiskDirection.RIGHT, "右前方有车辆"), nowMs = 100L)
        val stable = stabilizer.update(noRisk(), nowMs = 130L)

        assertEquals(RiskLevel.NONE, stable.level)
        assertEquals(RiskDirection.NONE, stable.direction)
    }

    @Test
    fun directionChangeResetsPendingMediumRisk() {
        val stabilizer = RiskStabilizer()

        stabilizer.update(risk(RiskLevel.MEDIUM, RiskDirection.LEFT, "左前方有车辆"), nowMs = 100L)
        val changed = stabilizer.update(risk(RiskLevel.MEDIUM, RiskDirection.RIGHT, "右前方有车辆"), nowMs = 130L)

        assertEquals(RiskLevel.NONE, changed.level)
        assertEquals(RiskDirection.NONE, changed.direction)
    }

    @Test
    fun shortNoRiskGapKeepsLastConfirmedAlertThenClears() {
        val stabilizer = RiskStabilizer()
        val medium = risk(RiskLevel.MEDIUM, RiskDirection.LEFT, "左前方有车辆")

        stabilizer.update(medium, nowMs = 100L)
        stabilizer.update(medium, nowMs = 130L)
        val held = stabilizer.update(noRisk(), nowMs = 700L)
        val cleared = stabilizer.update(noRisk(), nowMs = 731L)

        assertEquals(RiskLevel.MEDIUM, held.level)
        assertEquals(RiskDirection.LEFT, held.direction)
        assertEquals(RiskLevel.NONE, cleared.level)
        assertEquals(RiskDirection.NONE, cleared.direction)
    }

    private fun noRisk(): RiskResult {
        return RiskResult(RiskLevel.NONE, RiskDirection.NONE, "未发现风险")
    }

    private fun risk(level: RiskLevel, direction: RiskDirection, message: String): RiskResult {
        return RiskResult(
            level = level,
            direction = direction,
            message = message,
            sourceDetection = Detection(
                classId = 0,
                label = "person",
                confidence = 0.9f,
                boundingBox = BoundingBox(420f, 300f, 580f, 760f),
                frameSize = frame
            )
        )
    }
}
