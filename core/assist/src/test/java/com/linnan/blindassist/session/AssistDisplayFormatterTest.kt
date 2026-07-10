package com.linnan.blindassist.session

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskEvidenceState
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AssistDisplayFormatterTest {
    private val frame = FrameSize(1000, 1000)

    @Test
    fun targetLineUsesCurrentFrameTargetAndPrimaryTarget() {
        val risk = risk(
            level = RiskLevel.MEDIUM,
            direction = RiskDirection.RIGHT,
            proximity = ProximityBand.NEAR,
            label = "person"
        )

        val line = AssistDisplayFormatter.targetLine(risk, risk, detectionCount = 2)

        assertEquals("当前目标 2 个 · 主要目标 人", line)
    }

    @Test
    fun targetLineExplainsHeldRiskWhenCurrentFrameHasNoDetections() {
        val raw = RiskResult(RiskLevel.NONE, RiskDirection.NONE, "持续检测中")
        val stable = risk(
            level = RiskLevel.MEDIUM,
            direction = RiskDirection.RIGHT,
            proximity = ProximityBand.NEAR,
            label = "person"
        )

        val line = AssistDisplayFormatter.targetLine(raw, stable, detectionCount = 0)
        val careLine = AssistDisplayFormatter.careTargetLine(raw, stable, detectionCount = 0)

        assertEquals("提醒保持：上一帧 人 · 当前帧 0 个", line)
        assertEquals("提醒短暂保持，当前帧未重新锁定目标", careLine)
    }

    @Test
    fun targetLineAvoidsPrimaryTargetWhenNoStableRiskSource() {
        val raw = RiskResult(RiskLevel.NONE, RiskDirection.NONE, "持续检测中")

        val line = AssistDisplayFormatter.targetLine(raw, raw, detectionCount = 0)

        assertEquals("当前未检测到达到提醒条件的支持目标，请继续确认周围环境。", line)
    }

    @Test
    fun targetLineReportsDetectionsWithoutAlertSource() {
        val raw = RiskResult(RiskLevel.NONE, RiskDirection.NONE, "持续检测中")

        val line = AssistDisplayFormatter.targetLine(raw, raw, detectionCount = 3)

        assertEquals("当前目标 3 个 · 暂无需要提醒的主要目标", line)
    }

    @Test
    fun noneDetailDistinguishesSupportedAndUnsupportedEvidenceInBothLanguages() {
        val unsupported = RiskResult(RiskLevel.NONE, RiskDirection.NONE, "monitoring")
        val supported = unsupported.copy(evidenceState = RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE)

        assertEquals(
            "当前未检测到达到提醒条件的支持目标，请继续确认周围环境。",
            AssistDisplayFormatter.detailFor(unsupported)
        )
        assertEquals(
            "检测到模型支持的目标，当前未达到提醒条件。",
            AssistDisplayFormatter.detailFor(supported)
        )
        assertEquals(
            "No supported target currently meets the reminder threshold. Keep checking your surroundings.",
            AssistDisplayFormatter.detailFor(unsupported, com.linnan.blindassist.localization.AppLanguage.EN)
        )
        assertEquals(
            "A supported target is detected, but it does not currently meet the reminder threshold.",
            AssistDisplayFormatter.detailFor(supported, com.linnan.blindassist.localization.AppLanguage.EN)
        )
    }

    @Test
    fun detailTextUsesActionOrientedGuidance() {
        val high = risk(
            level = RiskLevel.HIGH,
            direction = RiskDirection.CENTER,
            proximity = ProximityBand.NEAR,
            label = "bus"
        )
        val medium = risk(
            level = RiskLevel.MEDIUM,
            direction = RiskDirection.LEFT,
            proximity = ProximityBand.NEAR,
            label = "chair"
        )

        assertTrue(AssistDisplayFormatter.detailFor(high).contains("请减速"))
        assertTrue(AssistDisplayFormatter.detailFor(medium).contains("谨慎通过"))
    }

    private fun risk(
        level: RiskLevel,
        direction: RiskDirection,
        proximity: ProximityBand,
        label: String
    ): RiskResult {
        return RiskResult(
            level = level,
            direction = direction,
            message = "测试风险",
            sourceDetection = Detection(
                classId = 0,
                label = label,
                confidence = 0.9f,
                boundingBox = BoundingBox(390f, 300f, 610f, 760f),
                frameSize = frame
            ),
            proximity = proximity,
            urgencyScore = 6.2f
        )
    }
}
