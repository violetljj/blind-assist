package com.linnan.blindassist.risk

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TemporalRiskTrackerTest {
    private val frame = FrameSize(1000, 1000)
    private val analyzer = RiskAnalyzer()

    @Test
    fun approachingCenterFarTargetPromotesToLowWithTrendScore() {
        val tracker = TemporalRiskTracker()

        tracker.update(raw("person", BoundingBox(450f, 120f, 520f, 280f)), nowMs = 100L)
        tracker.update(raw("person", BoundingBox(445f, 130f, 525f, 310f)), nowMs = 200L)
        val result = tracker.update(raw("person", BoundingBox(440f, 140f, 530f, 340f)), nowMs = 300L)

        assertEquals(ApproachTrend.APPROACHING, result.approachTrend)
        assertEquals(RiskLevel.LOW, result.level)
        assertEquals(ProximityBand.FAR, result.proximity)
        assertTrue(result.riskScore > result.scoreBreakdown.total - 0.001f)
        assertTrue(result.scoreBreakdown.approachTrend > 0f)
        assertEquals(RiskFusionReason.MOTION_PROMOTED.name, result.scoreBreakdown.fusionSummary)
    }

    @Test
    fun approachingCenterMidTargetPromotesToMedium() {
        val tracker = TemporalRiskTracker()

        tracker.update(raw("chair", BoundingBox(450f, 220f, 560f, 460f)), nowMs = 100L)
        tracker.update(raw("chair", BoundingBox(445f, 230f, 565f, 480f)), nowMs = 200L)
        val result = tracker.update(raw("chair", BoundingBox(440f, 240f, 570f, 520f)), nowMs = 300L)

        assertEquals(ApproachTrend.APPROACHING, result.approachTrend)
        assertEquals(RiskLevel.MEDIUM, result.level)
        assertEquals(ProximityBand.MID, result.proximity)
    }

    @Test
    fun approachingSideTargetDoesNotPromoteToHigh() {
        val tracker = TemporalRiskTracker()

        tracker.update(raw("car", BoundingBox(20f, 300f, 250f, 650f)), nowMs = 100L)
        tracker.update(raw("car", BoundingBox(20f, 315f, 260f, 690f)), nowMs = 200L)
        val result = tracker.update(raw("car", BoundingBox(20f, 330f, 270f, 730f)), nowMs = 300L)

        assertEquals(ApproachTrend.APPROACHING, result.approachTrend)
        assertEquals(RiskDirection.LEFT, result.direction)
        assertEquals(RiskLevel.MEDIUM, result.level)
    }

    @Test
    fun recedingTargetDoesNotBoostRisk() {
        val tracker = TemporalRiskTracker()

        tracker.update(raw("person", BoundingBox(440f, 140f, 530f, 340f)), nowMs = 100L)
        tracker.update(raw("person", BoundingBox(445f, 130f, 525f, 310f)), nowMs = 200L)
        val result = tracker.update(raw("person", BoundingBox(450f, 120f, 520f, 280f)), nowMs = 300L)

        assertEquals(ApproachTrend.RECEDING, result.approachTrend)
        assertEquals(RiskLevel.NONE, result.level)
        assertEquals(0f, result.scoreBreakdown.approachTrend, 0.001f)
        assertEquals(RiskFusionReason.GEOMETRY_ONLY.name, result.scoreBreakdown.fusionSummary)
    }

    @Test
    fun staleFrameGapResetsApproachTrack() {
        val tracker = TemporalRiskTracker()

        tracker.update(raw("person", BoundingBox(450f, 120f, 520f, 280f)), nowMs = 100L)
        tracker.update(raw("person", BoundingBox(445f, 130f, 525f, 310f)), nowMs = 200L)
        val result = tracker.update(raw("person", BoundingBox(440f, 140f, 530f, 340f)), nowMs = 1200L)

        assertEquals(ApproachTrend.UNKNOWN, result.approachTrend)
        assertEquals(RiskLevel.NONE, result.level)
    }

    @Test
    fun depthEvidenceCanMarkApproaching() {
        val tracker = TemporalRiskTracker()

        tracker.update(
            rawWithDepth(ProximityBand.FAR, 0.60f, BoundingBox(450f, 120f, 520f, 280f)),
            nowMs = 100L
        )
        tracker.update(
            rawWithDepth(ProximityBand.MID, 0.64f, BoundingBox(450f, 120f, 520f, 280f)),
            nowMs = 200L
        )
        val result = tracker.update(
            rawWithDepth(ProximityBand.NEAR, 0.76f, BoundingBox(450f, 120f, 520f, 280f)),
            nowMs = 300L
        )

        assertEquals(ApproachTrend.APPROACHING, result.approachTrend)
        assertTrue(result.scoreBreakdown.approachTrend > 0f)
        assertEquals(
            "${RiskFusionReason.DEPTH_REJECTED_LARGE_PROMOTION.name}+${RiskFusionReason.MOTION_PROMOTED.name}",
            result.scoreBreakdown.fusionSummary
        )
    }

    @Test
    fun resetClearsApproachTrack() {
        val tracker = TemporalRiskTracker()

        tracker.update(raw("person", BoundingBox(450f, 120f, 520f, 280f)), nowMs = 100L)
        tracker.update(raw("person", BoundingBox(445f, 130f, 525f, 310f)), nowMs = 200L)
        tracker.reset()
        val result = tracker.update(raw("person", BoundingBox(440f, 140f, 530f, 340f)), nowMs = 300L)

        assertEquals(ApproachTrend.UNKNOWN, result.approachTrend)
        assertEquals(RiskLevel.NONE, result.level)
    }

    private fun raw(label: String, box: BoundingBox): RiskResult {
        return analyzer.analyze(listOf(detection(label, box)), frame)
    }

    private fun rawWithDepth(
        depthBand: ProximityBand,
        depthScore: Float,
        box: BoundingBox
    ): RiskResult {
        return analyzer.analyze(
            listOf(
                detection(
                    label = "person",
                    box = box,
                    distanceEvidence = DistanceEvidence(
                        band = depthBand,
                        confidence = 0.8f,
                        source = DistanceEvidenceSource.MONOCULAR_DEPTH,
                        relativeDepthScore = depthScore
                    )
                )
            ),
            frame
        )
    }

    private fun detection(
        label: String,
        box: BoundingBox,
        distanceEvidence: DistanceEvidence? = null
    ): Detection {
        return Detection(
            classId = 0,
            label = label,
            confidence = 0.9f,
            boundingBox = box,
            frameSize = frame,
            distanceEvidence = distanceEvidence
        )
    }
}
