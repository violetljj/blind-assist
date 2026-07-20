package com.linnan.blindassist.feedback

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.ApproachTrend
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskFusionReason
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class FeedbackPlannerTest {
    private val frame = FrameSize(1000, 1000)

    @Test
    fun stableSegmentationMediumMidProducesAlert() {
        assertNotNull(FeedbackPlanner.planFor(segmentationRisk(RiskFusionReason.STABILITY_PROMOTED.name)))
    }

    @Test
    fun approachingSegmentationMediumMidProducesAlert() {
        assertNotNull(FeedbackPlanner.planFor(segmentationRisk(RiskFusionReason.MOTION_PROMOTED.name)))
    }

    @Test
    fun singleFrameSegmentationMediumMidDoesNotProduceAlert() {
        assertNull(FeedbackPlanner.planFor(segmentationRisk("GEOMETRY_ONLY")))
    }

    @Test
    fun approachingCenterPersonMidIsOptInAndDefaultRemainsSilent() {
        val risk = segmentationRisk("GEOMETRY_ONLY").copy(
            sourceDetection = Detection(
                classId = 0,
                label = "person",
                confidence = 0.8f,
                boundingBox = BoundingBox(400f, 300f, 600f, 700f),
                frameSize = frame,
                source = DetectionSource.OBJECT_DETECTOR
            ),
            approachTrend = ApproachTrend.APPROACHING
        )
        assertNull(FeedbackPlanner.planFor(risk))
        assertNotNull(
            FeedbackPlanner.planFor(
                risk,
                config = FeedbackPlannerConfig(enableApproachingCenterPersonMidAlert = true)
            )
        )
    }

    private fun segmentationRisk(summary: String): RiskResult = RiskResult(
        level = RiskLevel.MEDIUM,
        direction = com.linnan.blindassist.risk.RiskDirection.CENTER,
        message = "test",
        sourceDetection = Detection(
            classId = 10_015,
            label = "stairs",
            confidence = 1f,
            boundingBox = BoundingBox(400f, 400f, 600f, 520f),
            frameSize = frame,
            source = DetectionSource.SEGMENTATION
        ),
        proximity = ProximityBand.MID,
        scoreBreakdown = com.linnan.blindassist.risk.RiskScoreBreakdown(fusionSummary = summary)
    )
}
