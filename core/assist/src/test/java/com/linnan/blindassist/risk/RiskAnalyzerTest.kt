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
        assertEquals("前方很近，放慢", result.message)
        assertTrue(result.urgencyScore > 0f)
        assertEquals(result.urgencyScore, result.riskScore, 0.001f)
        assertEquals(result.riskScore, result.scoreBreakdown.total, 0.001f)
        assertTrue(result.scoreBreakdown.bottomPosition > 0f)
        assertTrue(result.scoreBreakdown.area > 0f)
        assertTrue(result.scoreBreakdown.centerLane > 0f)
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
        assertEquals("左前方近处，注意避让", result.message)
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
        assertEquals("前方近处，减速", result.message)
    }

    @Test
    fun centerBottomBoundaryTargetIsNearHighRisk() {
        val result = analyzer.analyze(
            listOf(detection("person", BoundingBox(400f, 120f, 600f, 580f))),
            frame
        )

        assertEquals(RiskLevel.HIGH, result.level)
        assertEquals(RiskDirection.CENTER, result.direction)
        assertEquals(ProximityBand.NEAR, result.proximity)
        assertEquals("前方近处，减速", result.message)
    }

    @Test
    fun centerAreaBoundaryTargetIsNearHighRisk() {
        val result = analyzer.analyze(
            listOf(detection("person", BoundingBox(362.5f, 150f, 637.5f, 550f))),
            frame
        )

        assertEquals(RiskLevel.HIGH, result.level)
        assertEquals(RiskDirection.CENTER, result.direction)
        assertEquals(ProximityBand.NEAR, result.proximity)
        assertEquals("前方近处，减速", result.message)
    }

    @Test
    fun sideTargetsKeepOriginalNearThresholdsAtCenterBoundarySize() {
        val left = analyzer.analyze(
            listOf(detection("person", BoundingBox(100f, 150f, 400f, 550f))),
            frame
        )
        val right = analyzer.analyze(
            listOf(detection("person", BoundingBox(600f, 150f, 900f, 550f))),
            frame
        )

        assertEquals(RiskLevel.LOW, left.level)
        assertEquals(RiskDirection.LEFT, left.direction)
        assertEquals(ProximityBand.MID, left.proximity)
        assertEquals(RiskLevel.LOW, right.level)
        assertEquals(RiskDirection.RIGHT, right.direction)
        assertEquals(ProximityBand.MID, right.proximity)
    }

    @Test
    fun rightNearTargetUsesGuidanceMessage() {
        val result = analyzer.analyze(
            listOf(detection("chair", BoundingBox(740f, 330f, 980f, 720f))),
            frame
        )

        assertEquals(RiskLevel.MEDIUM, result.level)
        assertEquals(RiskDirection.RIGHT, result.direction)
        assertEquals(ProximityBand.NEAR, result.proximity)
        assertEquals("右前方近处，注意避让", result.message)
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
    fun directionBoundariesClassifyLeftCenterAndRight() {
        val left = analyzer.analyze(listOf(detection("chair", BoundingBox(40f, 330f, 260f, 720f))), frame)
        val center = analyzer.analyze(listOf(detection("chair", BoundingBox(420f, 330f, 580f, 650f))), frame)
        val right = analyzer.analyze(listOf(detection("chair", BoundingBox(740f, 330f, 960f, 720f))), frame)

        assertEquals(RiskDirection.LEFT, left.direction)
        assertEquals(RiskDirection.CENTER, center.direction)
        assertEquals(RiskDirection.RIGHT, right.direction)
    }

    @Test
    fun farTargetDoesNotTriggerFeedbackLevelRisk() {
        val result = analyzer.analyze(
            listOf(detection("person", BoundingBox(450f, 120f, 520f, 280f))),
            frame
        )

        assertEquals(RiskLevel.NONE, result.level)
        assertEquals(ProximityBand.FAR, result.proximity)
        assertEquals(RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE, result.evidenceState)
        assertEquals("检测到模型支持的目标，当前未达到提醒条件。", result.message)
    }

    @Test
    fun monocularDepthEvidenceConservativelyPromotesCenteredMidGeometryOneStep() {
        val evidence = DistanceEvidence(
            band = ProximityBand.NEAR,
            confidence = 0.82f,
            source = DistanceEvidenceSource.MONOCULAR_DEPTH,
            relativeDepthScore = 0.7f
        )
        val result = analyzer.analyze(
            listOf(
                detection(
                    label = "person",
                    box = BoundingBox(450f, 240f, 560f, 500f),
                    distanceEvidence = evidence
                )
            ),
            frame
        )

        assertEquals(RiskLevel.HIGH, result.level)
        assertEquals(RiskDirection.CENTER, result.direction)
        assertEquals(ProximityBand.NEAR, result.proximity)
        assertEquals(evidence, result.distanceEvidence)
        assertEquals(RiskFusionReason.DEPTH_PROMOTED.name, result.scoreBreakdown.fusionSummary)
    }

    @Test
    fun lowConfidenceDepthEvidenceFallsBackToGeometry() {
        val result = analyzer.analyze(
            listOf(
                detection(
                    label = "person",
                    box = BoundingBox(450f, 120f, 520f, 280f),
                    distanceEvidence = DistanceEvidence(
                        band = ProximityBand.CRITICAL,
                        confidence = 0.3f,
                        source = DistanceEvidenceSource.MONOCULAR_DEPTH,
                        relativeDepthScore = 0.95f
                    )
                )
            ),
            frame
        )

        assertEquals(RiskLevel.NONE, result.level)
        assertEquals(ProximityBand.FAR, result.proximity)
        assertEquals(RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE, result.evidenceState)
        assertEquals(null, result.distanceEvidence)
        assertEquals(
            RiskFusionReason.DEPTH_REJECTED_LOW_CONFIDENCE.name,
            result.scoreBreakdown.fusionSummary
        )
    }

    @Test
    fun depthEvidenceThatLooksFartherDoesNotDowngradeGeometryRisk() {
        val evidence = DistanceEvidence(
            band = ProximityBand.FAR,
            confidence = 0.9f,
            source = DistanceEvidenceSource.MONOCULAR_DEPTH,
            relativeDepthScore = 0.1f
        )

        val result = analyzer.analyze(
            listOf(
                detection(
                    label = "person",
                    box = BoundingBox(400f, 120f, 600f, 580f),
                    distanceEvidence = evidence
                )
            ),
            frame
        )

        assertEquals(ProximityBand.NEAR, result.proximity)
        assertEquals(RiskLevel.HIGH, result.level)
        assertEquals(evidence, result.distanceEvidence)
        assertEquals(
            RiskFusionReason.DEPTH_REJECTED_NOT_CLOSER.name,
            result.scoreBreakdown.fusionSummary
        )
    }

    @Test
    fun sideDepthEvidenceCanOnlyPromoteToMediumRisk() {
        val evidence = DistanceEvidence(
            band = ProximityBand.NEAR,
            confidence = 0.9f,
            source = DistanceEvidenceSource.MONOCULAR_DEPTH,
            relativeDepthScore = 0.8f
        )

        val result = analyzer.analyze(
            listOf(
                detection(
                    label = "person",
                    box = BoundingBox(100f, 150f, 400f, 550f),
                    distanceEvidence = evidence
                )
            ),
            frame
        )

        assertEquals(RiskDirection.LEFT, result.direction)
        assertEquals(ProximityBand.NEAR, result.proximity)
        assertEquals(RiskLevel.MEDIUM, result.level)
        assertEquals(RiskFusionReason.DEPTH_PROMOTED.name, result.scoreBreakdown.fusionSummary)
    }

    @Test
    fun configurableDepthFusionCanCapFarToCriticalPromotion() {
        val conservativeAnalyzer = RiskAnalyzer(
            RiskAnalyzerConfig.Default.copy(
                distanceEvidenceMaxPromotionSteps = 1,
                rejectLargeDistanceEvidencePromotion = false
            )
        )
        val evidence = DistanceEvidence(
            band = ProximityBand.CRITICAL,
            confidence = 0.9f,
            source = DistanceEvidenceSource.MONOCULAR_DEPTH,
            relativeDepthScore = 0.95f
        )

        val result = conservativeAnalyzer.analyze(
            listOf(
                detection(
                    label = "person",
                    box = BoundingBox(450f, 120f, 520f, 280f),
                    distanceEvidence = evidence
                )
            ),
            frame
        )

        assertEquals(ProximityBand.MID, result.proximity)
        assertEquals(RiskLevel.LOW, result.level)
        assertEquals(evidence, result.distanceEvidence)
    }

    @Test
    fun defaultConservativeDepthFusionRejectsLargeConflicts() {
        val result = analyzer.analyze(
            listOf(
                detection(
                    label = "person",
                    box = BoundingBox(450f, 120f, 520f, 280f),
                    distanceEvidence = DistanceEvidence(
                        band = ProximityBand.CRITICAL,
                        confidence = 0.9f,
                        source = DistanceEvidenceSource.MONOCULAR_DEPTH,
                        relativeDepthScore = 0.95f
                    )
                )
            ),
            frame
        )

        assertEquals(RiskLevel.NONE, result.level)
        assertEquals(ProximityBand.FAR, result.proximity)
        assertEquals(RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE, result.evidenceState)
        assertEquals(
            RiskFusionReason.DEPTH_REJECTED_LARGE_PROMOTION.name,
            result.scoreBreakdown.fusionSummary
        )
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
        assertEquals(RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE, result.evidenceState)
    }

    @Test
    fun unrelatedClassIsIgnored() {
        val result = analyzer.analyze(
            listOf(detection("banana", BoundingBox(420f, 300f, 580f, 760f))),
            frame
        )

        assertEquals(RiskLevel.NONE, result.level)
        assertEquals(RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE, result.evidenceState)
    }

    @Test
    fun emptyFrameHasNoSupportedTargetEvidence() {
        val result = analyzer.analyze(emptyList(), frame)

        assertEquals(RiskLevel.NONE, result.level)
        assertEquals(RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE, result.evidenceState)
        assertEquals("当前未检测到达到提醒条件的支持目标，请继续确认周围环境。", result.message)
    }

    private fun detection(
        label: String,
        box: BoundingBox,
        confidence: Float = 0.9f,
        distanceEvidence: DistanceEvidence? = null
    ): Detection {
        return Detection(
            classId = 0,
            label = label,
            confidence = confidence,
            boundingBox = box,
            frameSize = frame,
            distanceEvidence = distanceEvidence
        )
    }
}
