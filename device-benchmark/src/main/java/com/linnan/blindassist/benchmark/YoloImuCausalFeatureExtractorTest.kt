package com.linnan.blindassist.benchmark

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class YoloImuCausalFeatureExtractorTest {
    @Test
    fun emptyFrameProducesClearFeatureEvidenceWithoutAnAlert() {
        val result = YoloImuCausalFeatureExtractor().extract(YoloImuFeatureFrame(1L, emptyList()))

        assertFalse(result.hasSelectedDetection)
        assertEquals(128, result.spatialGridNhwc.size)
        assertEquals(20, result.motion.size)
        assertTrue(result.spatialGridNhwc.all { it == 0f })
        assertTrue(result.motion.all { it == 0f })
    }

    @Test
    fun centeredNearFieldBoxHasMoreCorridorEvidenceThanLateralBox() {
        val centered = YoloImuCausalFeatureExtractor().extract(
            YoloImuFeatureFrame(1L, listOf(detection(left = 400f, top = 650f, right = 600f, bottom = 950f)))
        )
        val lateral = YoloImuCausalFeatureExtractor().extract(
            YoloImuFeatureFrame(1L, listOf(detection(left = 0f, top = 650f, right = 160f, bottom = 950f)))
        )

        assertTrue(centered.motion[8] > lateral.motion[8])
        assertEquals(0.06f, centered.motion[7], 1e-6f)
        assertEquals(1f, centered.motion[8], 1e-6f)
        assertEquals(0.054f, centered.motion[16], 1e-6f)
        assertTrue(centered.spatialGridNhwc.filterIndexed { index, _ -> index % 8 == 2 }.sum() >
            lateral.spatialGridNhwc.filterIndexed { index, _ -> index % 8 == 2 }.sum())
    }

    @Test
    fun motionIsCarriedThroughAndTemporalDeltasAreCausal() {
        val extractor = YoloImuCausalFeatureExtractor()
        val first = extractor.extract(
            YoloImuFeatureFrame(
                timestampNanos = 10L,
                detections = listOf(detection(left = 400f, top = 600f, right = 520f, bottom = 800f)),
                imu = YoloImuMotion(0.1f, -0.2f, 0.3f, 0.4f, observed = true)
            )
        )
        val second = extractor.extract(
            YoloImuFeatureFrame(20L, listOf(detection(left = 430f, top = 580f, right = 570f, bottom = 820f)))
        )

        assertEquals(0.1f, first.motion[0], 0f)
        assertEquals(-0.2f, first.motion[1], 0f)
        assertEquals(1f, first.motion[4], 0f)
        assertEquals(0f, first.motion[10], 0f)
        assertTrue(second.motion[10] > 0f)
        assertTrue(second.motion[11] > 0f)
    }

    @Test
    fun invalidImuAndNewObjectFailClosedForTemporalSignals() {
        val extractor = YoloImuCausalFeatureExtractor()
        extractor.extract(YoloImuFeatureFrame(10L, listOf(detection(classId = 1))))
        val result = extractor.extract(
            YoloImuFeatureFrame(
                timestampNanos = 20L,
                detections = listOf(detection(classId = 2)),
                imu = YoloImuMotion(Float.NaN, 1f, 1f, 1f, observed = true)
            )
        )

        assertEquals(0f, result.motion[4], 0f)
        assertEquals(0f, result.motion[10], 0f)
        assertEquals(0f, result.motion[11], 0f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun nonMonotonicTimestampsAreRejected() {
        val extractor = YoloImuCausalFeatureExtractor()
        extractor.extract(YoloImuFeatureFrame(10L, emptyList()))
        extractor.extract(YoloImuFeatureFrame(10L, emptyList()))
    }

    @Test
    fun eightFramePackerUsesChronologicalCausalWindowAndRetainsNoBoxes() {
        val packer = YoloImuCausalWindow()
        repeat(7) { index ->
            assertEquals(null, packer.append(YoloImuFeatureFrame((index + 1).toLong(), listOf(detection()))))
        }
        val result = packer.append(YoloImuFeatureFrame(8L, listOf(detection())))!!

        assertEquals(8 * 4 * 4 * 8, result.spatialSequence.size)
        assertEquals(8 * 20, result.motionSequence.size)
        assertEquals(0.1f, result.motionSequence[5], 0f)
        assertEquals(0.1f, result.motionSequence[20 + 5], 0f)
        assertTrue(result.motionSequence.all { it.isFinite() })
    }

    @Test
    fun deviceMicrobenchmarkLogsPureFeatureExtractionCost() {
        val detections = listOf(
            detection(left = 390f, top = 580f, right = 570f, bottom = 900f),
            detection(classId = 2, left = 90f, top = 520f, right = 250f, bottom = 760f),
            detection(classId = 3, left = 700f, top = 440f, right = 820f, bottom = 680f),
            detection(classId = 4, left = 470f, top = 230f, right = 560f, bottom = 410f)
        )
        val extractor = YoloImuCausalFeatureExtractor()
        repeat(300) { index -> extractor.extract(YoloImuFeatureFrame((index + 1).toLong(), detections)) }

        val samplesNanos = LongArray(2_000) { index ->
            val start = System.nanoTime()
            extractor.extract(YoloImuFeatureFrame((index + 301).toLong(), detections))
            System.nanoTime() - start
        }.sorted()
        val p50Micros = samplesNanos[samplesNanos.size / 2] / 1_000.0
        val p95Micros = samplesNanos[(samplesNanos.size * 95 / 100).coerceAtMost(samplesNanos.lastIndex)] / 1_000.0
        Log.i("YoloImuFeatureBench", "samples=2000 p50_us=%.3f p95_us=%.3f".format(java.util.Locale.US, p50Micros, p95Micros))

        // A pure feature probe must remain far below the 1.953 ms end-to-end headroom.
        assertTrue("p95_us=$p95Micros", p95Micros < 1_000.0)
    }

    private fun detection(
        classId: Int = 1,
        left: Float = 400f,
        top: Float = 600f,
        right: Float = 600f,
        bottom: Float = 900f
    ) = Detection(
        classId = classId,
        label = "test",
        confidence = 0.9f,
        boundingBox = BoundingBox(left, top, right, bottom),
        frameSize = FrameSize(1_000, 1_000)
    )
}
