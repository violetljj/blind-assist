package com.linnan.blindassist.feedback

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
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
