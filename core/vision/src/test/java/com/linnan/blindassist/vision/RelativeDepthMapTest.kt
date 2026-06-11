package com.linnan.blindassist.vision

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.ProximityBand
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class RelativeDepthMapTest {
    private val frame = FrameSize(100, 100)
    private val fullBox = BoundingBox(0f, 0f, 100f, 100f)

    @Test
    fun sampleEvidenceMapsHigherClosenessToNearerBand() {
        val map = RelativeDepthMap(
            width = 4,
            height = 4,
            closeness = FloatArray(16) { index -> if (index >= 8) 0.9f else 0.1f }
        )

        val evidence = map.sampleEvidence(fullBox, frame)

        assertEquals(ProximityBand.CRITICAL, evidence?.band)
        assertEquals(0.9f, evidence?.relativeDepthScore ?: 0f, 0.0001f)
    }

    @Test
    fun percentileCanUseLowerDepthSampleAfterPolarityIsInverted() {
        val map = RelativeDepthMap(
            width = 4,
            height = 4,
            closeness = floatArrayOf(
                0.1f, 0.1f, 0.1f, 0.1f,
                0.1f, 0.1f, 0.1f, 0.1f,
                0.2f, 0.2f, 0.8f, 0.8f,
                0.2f, 0.2f, 0.8f, 0.8f
            )
        )

        val evidence = map.sampleEvidence(
            fullBox,
            frame,
            DepthEvidenceSamplingConfig(samplePercentile = 0.25f)
        )

        assertEquals(ProximityBand.FAR, evidence?.band)
        assertEquals(0.2f, evidence?.relativeDepthScore ?: 0f, 0.0001f)
    }

    @Test
    fun innerCropExcludesNoisyBoxEdges() {
        val map = RelativeDepthMap(
            width = 6,
            height = 6,
            closeness = FloatArray(36) { index ->
                val x = index % 6
                val y = index / 6
                if (x == 0 || x == 5 || y == 0 || y == 5) 0.95f else 0.2f
            }
        )

        val evidence = map.sampleEvidence(
            fullBox,
            frame,
            DepthEvidenceSamplingConfig(innerCropRatio = 0.5f, lowerHalfOnly = false)
        )

        assertEquals(ProximityBand.FAR, evidence?.band)
        assertEquals(0.2f, evidence?.relativeDepthScore ?: 0f, 0.0001f)
    }

    @Test
    fun sampleEvidenceRejectsWeakLocalSignal() {
        val flatMap = RelativeDepthMap(width = 4, height = 4, closeness = FloatArray(16) { 0.6f })

        val evidence = flatMap.sampleEvidence(
            fullBox,
            frame,
            DepthEvidenceSamplingConfig(minLocalRange = 0.1f)
        )

        assertNull(evidence)
    }

    @Test
    fun sampleEvidenceRejectsWhenSampleCountIsTooSmall() {
        val map = RelativeDepthMap(width = 2, height = 2, closeness = FloatArray(4) { 0.9f })

        val evidence = map.sampleEvidence(
            fullBox,
            frame,
            DepthEvidenceSamplingConfig(minSamples = 5)
        )

        assertNull(evidence)
    }

    @Test
    fun sampleEvidenceRejectsWhenConfidenceThresholdIsTooHigh() {
        val map = RelativeDepthMap(
            width = 4,
            height = 4,
            closeness = FloatArray(16) { index -> if (index % 2 == 0) 0.1f else 0.9f }
        )

        val evidence = map.sampleEvidence(
            fullBox,
            frame,
            DepthEvidenceSamplingConfig(minConfidence = 0.9f)
        )

        assertNull(evidence)
    }
}
