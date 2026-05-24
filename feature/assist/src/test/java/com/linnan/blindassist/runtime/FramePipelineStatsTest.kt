package com.linnan.blindassist.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class FramePipelineStatsTest {
    @Test
    fun tracksDropRateAndInferencePercentiles() {
        val stats = FramePipelineStats(maxInferenceSamples = 10)
        repeat(5) { stats.onReceived() }

        stats.onProcessed(10L)
        stats.onProcessed(20L)
        stats.onProcessed(30L)
        stats.onDroppedBusy()
        val snapshot = stats.onDroppedDetectorUnavailable()

        assertEquals(5L, snapshot.received)
        assertEquals(3L, snapshot.processed)
        assertEquals(1L, snapshot.droppedBusy)
        assertEquals(1L, snapshot.droppedDetectorUnavailable)
        assertEquals(0.4f, snapshot.droppedFrameRate, 0.001f)
        assertEquals(20L, snapshot.inferenceP50Ms)
        assertEquals(30L, snapshot.inferenceP95Ms)
    }

    @Test
    fun resetClearsCountersAndSamples() {
        val stats = FramePipelineStats()
        stats.onReceived()
        stats.onProcessed(42L)
        stats.reset()

        val snapshot = stats.snapshot()

        assertEquals(0L, snapshot.received)
        assertEquals(0L, snapshot.processed)
        assertEquals(0f, snapshot.droppedFrameRate, 0.001f)
        assertEquals(0L, snapshot.inferenceP50Ms)
        assertEquals(0L, snapshot.inferenceP95Ms)
    }
}
