package com.linnan.blindassist.risk

import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TraversabilitySegmentationAnalyzerTest {
    @Test
    fun keepsCurbAsBoundaryEvidenceAndExtractsPole() {
        val width = 20
        val height = 20
        val ids = IntArray(width * height) { 3 }
        fill(ids, width, 8, 12, 12, 14, 2)
        fill(ids, width, 10, 14, 12, 20, 24)

        val result = TraversabilitySegmentationAnalyzer(
            TraversabilityAnalyzerConfig(minimumRegionPixels = 4, minimumRegionAreaRatio = 0f)
        ).analyze(DenseSemanticMask(width, height, ids), FrameSize(200, 200))

        assertEquals(setOf("pole"), result.riskDetections.map { it.label }.toSet())
        assertTrue(result.riskDetections.all { it.source == DetectionSource.SEGMENTATION })
        assertTrue(result.safeCoverage > 0.5f)
        assertTrue(result.obstacleCoverage > 0f)
    }

    @Test
    fun reusesWorkingBuffersForSameMaskSize() {
        val ids = IntArray(20 * 20) { 3 }
        val analyzer = TraversabilitySegmentationAnalyzer()
        analyzer.analyze(DenseSemanticMask(20, 20, ids), FrameSize(200, 200))
        val allocations = analyzer.allocations
        analyzer.analyze(DenseSemanticMask(20, 20, ids), FrameSize(200, 200))
        assertEquals(allocations, analyzer.allocations)
    }

    @Test
    fun ignoresHazardOutsideProjectedCorridor() {
        val width = 20
        val height = 20
        val ids = IntArray(width * height) { 3 }
        fill(ids, width, 0, 12, 1, 20, 20)

        val result = TraversabilitySegmentationAnalyzer(
            TraversabilityAnalyzerConfig(minimumRegionPixels = 2, minimumRegionAreaRatio = 0f)
        ).analyze(DenseSemanticMask(width, height, ids), FrameSize(200, 200))

        assertTrue(result.riskDetections.isEmpty())
    }

    @Test
    fun keepsStairsAsBlindAssistHazard() {
        assertEquals(TraversabilityClass.OBSTACLE, BlindAssistSanpoTaxonomy.traversabilityFor(15))
        assertEquals("stairs", BlindAssistSanpoTaxonomy.riskLabelFor(15))
    }

    @Test
    fun keepsEdgeAlignedLongGenericObstacleForDiagnosticsButDisablesTemporalPromotion() {
        val width = 30
        val height = 30
        val ids = IntArray(width * height) { 3 }
        fill(ids, width, 0, 10, 30, 14, 20)

        val result = TraversabilitySegmentationAnalyzer(
            TraversabilityAnalyzerConfig(minimumRegionPixels = 2, minimumRegionAreaRatio = 0f)
        ).analyze(DenseSemanticMask(width, height, ids), FrameSize(300, 300))

        assertEquals(listOf("generic obstacle"), result.riskDetections.map { it.label })
        assertTrue(!result.riskDetections.single().temporalPromotionEligible)
    }

    @Test
    fun keepsCompactCenterGenericObstacle() {
        val width = 30
        val height = 30
        val ids = IntArray(width * height) { 3 }
        fill(ids, width, 12, 20, 18, 29, 20)

        val result = TraversabilitySegmentationAnalyzer(
            TraversabilityAnalyzerConfig(minimumRegionPixels = 2, minimumRegionAreaRatio = 0f)
        ).analyze(DenseSemanticMask(width, height, ids), FrameSize(300, 300))

        assertEquals(listOf("generic obstacle"), result.riskDetections.map { it.label })
        assertTrue(result.riskDetections.single().temporalPromotionEligible)
    }

    @Test
    fun learnedBoundaryNeedsCorridorIntrusionWhileUnknownRemainsDiagnosticOnly() {
        val width = 30
        val height = 30
        val ids = IntArray(width * height) { BlindAssistLearnedTraversabilityTaxonomy.WALKABLE }
        // Long right-edge boundary is parallel to the path and must not produce evidence.
        fill(ids, width, 24, 10, 30, 30, BlindAssistLearnedTraversabilityTaxonomy.BOUNDARY_STEP_CURB)
        // Unknown non-walkable changes coverage but never emits a detection by itself.
        fill(ids, width, 10, 20, 14, 28, BlindAssistLearnedTraversabilityTaxonomy.UNKNOWN_NONWALKABLE)

        val result = TraversabilitySegmentationAnalyzer(
            TraversabilityAnalyzerConfig(minimumRegionPixels = 2, minimumRegionAreaRatio = 0f),
            BlindAssistLearnedTraversabilityTaxonomy
        ).analyze(DenseSemanticMask(width, height, ids), FrameSize(300, 300))

        assertTrue(result.riskDetections.isEmpty())
        assertTrue(result.notSafeCoverage > 0f)
    }

    @Test
    fun learnedCentralObstaclePreservesSegmentationSource() {
        val width = 30
        val height = 30
        val ids = IntArray(width * height) { BlindAssistLearnedTraversabilityTaxonomy.WALKABLE }
        fill(ids, width, 12, 20, 18, 29, BlindAssistLearnedTraversabilityTaxonomy.OBSTACLE)

        val result = TraversabilitySegmentationAnalyzer(
            TraversabilityAnalyzerConfig(minimumRegionPixels = 2, minimumRegionAreaRatio = 0f),
            BlindAssistLearnedTraversabilityTaxonomy
        ).analyze(DenseSemanticMask(width, height, ids), FrameSize(300, 300))

        assertEquals(listOf("segmentation obstacle"), result.riskDetections.map { it.label })
        assertEquals(DetectionSource.SEGMENTATION, result.riskDetections.single().source)
    }

    private fun fill(
        target: IntArray,
        width: Int,
        left: Int,
        top: Int,
        right: Int,
        bottom: Int,
        value: Int
    ) {
        for (y in top until bottom) {
            for (x in left until right) target[y * width + x] = value
        }
    }
}
