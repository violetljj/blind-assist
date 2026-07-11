package com.linnan.blindassist.risk

import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TraversabilitySegmentationAnalyzerTest {
    @Test
    fun extractsCurbAndPoleInsideWalkingCorridor() {
        val width = 20
        val height = 20
        val ids = IntArray(width * height) { 3 }
        fill(ids, width, 8, 12, 12, 14, 2)
        fill(ids, width, 10, 14, 12, 20, 24)

        val result = TraversabilitySegmentationAnalyzer(
            TraversabilityAnalyzerConfig(minimumRegionPixels = 4, minimumRegionAreaRatio = 0f)
        ).analyze(DenseSemanticMask(width, height, ids), FrameSize(200, 200))

        assertEquals(setOf("curb", "pole"), result.riskDetections.map { it.label }.toSet())
        assertTrue(result.riskDetections.all { it.source == DetectionSource.SEGMENTATION })
        assertTrue(result.safeCoverage > 0.5f)
        assertTrue(result.obstacleCoverage > 0f)
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
