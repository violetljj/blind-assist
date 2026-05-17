package com.linnan.blindassist.risk

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import org.junit.Assert.assertEquals
import org.junit.Test

class RiskStabilizerTest {
    private val frame = FrameSize(1000, 1000)

    @Test
    fun criticalRiskConfirmsImmediately() {
        val stabilizer = RiskStabilizer()

        val stable = stabilizer.update(
            risk(RiskLevel.HIGH, RiskDirection.CENTER, ProximityBand.CRITICAL, "前方迫近 有人"),
            nowMs = 100L
        )

        assertEquals(RiskLevel.HIGH, stable.level)
        assertEquals(RiskDirection.CENTER, stable.direction)
        assertEquals(ProximityBand.CRITICAL, stable.proximity)
        assertEquals("前方迫近 有人", stable.message)
    }

    @Test
    fun nearMediumRiskRequiresTwoMatchingFrames() {
        val stabilizer = RiskStabilizer()
        val medium = risk(RiskLevel.MEDIUM, RiskDirection.LEFT, ProximityBand.NEAR, "左前方近处 有车辆")

        val first = stabilizer.update(medium, nowMs = 100L)
        val second = stabilizer.update(medium, nowMs = 130L)

        assertEquals(RiskLevel.NONE, first.level)
        assertEquals(RiskLevel.MEDIUM, second.level)
        assertEquals(RiskDirection.LEFT, second.direction)
        assertEquals(ProximityBand.NEAR, second.proximity)
    }

    @Test
    fun quietProfileRequiresThreeMediumFrames() {
        val stabilizer = RiskStabilizer()
        val medium = risk(RiskLevel.MEDIUM, RiskDirection.LEFT, ProximityBand.NEAR, "左前方近处 有车辆")

        val first = stabilizer.update(medium, profile = AlertProfile.QUIET, nowMs = 100L)
        val second = stabilizer.update(medium, profile = AlertProfile.QUIET, nowMs = 130L)
        val third = stabilizer.update(medium, profile = AlertProfile.QUIET, nowMs = 160L)

        assertEquals(RiskLevel.NONE, first.level)
        assertEquals(RiskLevel.NONE, second.level)
        assertEquals(RiskLevel.MEDIUM, third.level)
    }

    @Test
    fun sensitiveProfileConfirmsMediumOnFirstFrame() {
        val stabilizer = RiskStabilizer()
        val medium = risk(RiskLevel.MEDIUM, RiskDirection.LEFT, ProximityBand.NEAR, "左前方近处 有车辆")

        val stable = stabilizer.update(medium, profile = AlertProfile.SENSITIVE, nowMs = 100L)

        assertEquals(RiskLevel.MEDIUM, stable.level)
        assertEquals(RiskDirection.LEFT, stable.direction)
    }

    @Test
    fun singleNearMediumFrameDoesNotTriggerAfterDisappearing() {
        val stabilizer = RiskStabilizer()

        stabilizer.update(
            risk(RiskLevel.MEDIUM, RiskDirection.RIGHT, ProximityBand.NEAR, "右前方近处 有车辆"),
            nowMs = 100L
        )
        val stable = stabilizer.update(noRisk(), nowMs = 130L)

        assertEquals(RiskLevel.NONE, stable.level)
        assertEquals(RiskDirection.NONE, stable.direction)
    }

    @Test
    fun directionChangeResetsPendingMediumRisk() {
        val stabilizer = RiskStabilizer()

        stabilizer.update(
            risk(RiskLevel.MEDIUM, RiskDirection.LEFT, ProximityBand.NEAR, "左前方近处 有车辆"),
            nowMs = 100L
        )
        val changed = stabilizer.update(
            risk(RiskLevel.MEDIUM, RiskDirection.RIGHT, ProximityBand.NEAR, "右前方近处 有车辆"),
            nowMs = 130L
        )

        assertEquals(RiskLevel.NONE, changed.level)
        assertEquals(RiskDirection.NONE, changed.direction)
    }

    @Test
    fun proximityDowngradeResetsPendingAlertButKeepsLowRiskForUi() {
        val stabilizer = RiskStabilizer()

        stabilizer.update(
            risk(RiskLevel.MEDIUM, RiskDirection.LEFT, ProximityBand.NEAR, "左前方近处 有车辆"),
            nowMs = 100L
        )
        val downgraded = stabilizer.update(
            risk(RiskLevel.LOW, RiskDirection.LEFT, ProximityBand.MID, "左前方中距 有车辆"),
            nowMs = 130L
        )

        assertEquals(RiskLevel.LOW, downgraded.level)
        assertEquals(RiskDirection.LEFT, downgraded.direction)
        assertEquals(ProximityBand.MID, downgraded.proximity)
    }

    @Test
    fun proximityUpgradeCanConfirmMediumRiskSooner() {
        val stabilizer = RiskStabilizer()

        stabilizer.update(
            risk(RiskLevel.MEDIUM, RiskDirection.LEFT, ProximityBand.MID, "左前方中距 有车辆"),
            nowMs = 100L
        )
        val upgraded = stabilizer.update(
            risk(RiskLevel.MEDIUM, RiskDirection.LEFT, ProximityBand.NEAR, "左前方近处 有车辆"),
            nowMs = 130L
        )

        assertEquals(RiskLevel.MEDIUM, upgraded.level)
        assertEquals(ProximityBand.NEAR, upgraded.proximity)
    }

    @Test
    fun shortNoRiskGapKeepsLastConfirmedAlertThenClears() {
        val stabilizer = RiskStabilizer()
        val medium = risk(RiskLevel.MEDIUM, RiskDirection.LEFT, ProximityBand.NEAR, "左前方近处 有车辆")

        stabilizer.update(medium, nowMs = 100L)
        stabilizer.update(medium, nowMs = 130L)
        val held = stabilizer.update(noRisk(), nowMs = 700L)
        val cleared = stabilizer.update(noRisk(), nowMs = 731L)

        assertEquals(RiskLevel.MEDIUM, held.level)
        assertEquals(RiskDirection.LEFT, held.direction)
        assertEquals(ProximityBand.NEAR, held.proximity)
        assertEquals(RiskLevel.NONE, cleared.level)
        assertEquals(RiskDirection.NONE, cleared.direction)
    }

    @Test
    fun quietProfileClearsHeldAlertSoonerThanStandard() {
        val stabilizer = RiskStabilizer()
        val high = risk(RiskLevel.HIGH, RiskDirection.CENTER, ProximityBand.CRITICAL, "前方迫近 有人")

        stabilizer.update(high, profile = AlertProfile.QUIET, nowMs = 100L)
        val cleared = stabilizer.update(noRisk(), profile = AlertProfile.QUIET, nowMs = 551L)

        assertEquals(RiskLevel.NONE, cleared.level)
    }

    @Test
    fun sensitiveProfileHoldsAlertLongerThanStandard() {
        val stabilizer = RiskStabilizer()
        val high = risk(RiskLevel.HIGH, RiskDirection.CENTER, ProximityBand.CRITICAL, "前方迫近 有人")

        stabilizer.update(high, profile = AlertProfile.SENSITIVE, nowMs = 100L)
        val held = stabilizer.update(noRisk(), profile = AlertProfile.SENSITIVE, nowMs = 850L)

        assertEquals(RiskLevel.HIGH, held.level)
    }

    private fun noRisk(): RiskResult {
        return RiskResult(RiskLevel.NONE, RiskDirection.NONE, "未发现风险")
    }

    private fun risk(
        level: RiskLevel,
        direction: RiskDirection,
        proximity: ProximityBand,
        message: String
    ): RiskResult {
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
            ),
            proximity = proximity,
            urgencyScore = proximity.ordinal.toFloat()
        )
    }
}
