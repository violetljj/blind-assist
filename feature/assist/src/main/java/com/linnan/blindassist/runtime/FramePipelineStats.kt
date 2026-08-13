package com.linnan.blindassist.runtime

import kotlin.math.ceil

internal class FramePipelineStats(
    private val maxInferenceSamples: Int = DEFAULT_MAX_INFERENCE_SAMPLES
) {
    private val inferenceSamples = ArrayDeque<Long>()
    private var received: Long = 0L
    private var processed: Long = 0L
    private var droppedBusy: Long = 0L
    private var droppedInactive: Long = 0L
    private var droppedDetectorUnavailable: Long = 0L

    @Synchronized
    fun onReceived() {
        received += 1
    }

    @Synchronized
    fun onDroppedBusy(): FramePipelineStatsSnapshot {
        droppedBusy += 1
        return snapshotLocked()
    }

    @Synchronized
    fun onDroppedInactive(): FramePipelineStatsSnapshot {
        droppedInactive += 1
        return snapshotLocked()
    }

    @Synchronized
    fun onDroppedDetectorUnavailable(): FramePipelineStatsSnapshot {
        droppedDetectorUnavailable += 1
        return snapshotLocked()
    }

    @Synchronized
    fun onProcessed(inferenceMs: Long): FramePipelineStatsSnapshot {
        processed += 1
        inferenceSamples += inferenceMs
        while (inferenceSamples.size > maxInferenceSamples) {
            inferenceSamples.removeFirst()
        }
        return snapshotLocked()
    }

    @Synchronized
    fun snapshot(): FramePipelineStatsSnapshot = snapshotLocked()

    @Synchronized
    fun reset() {
        inferenceSamples.clear()
        received = 0L
        processed = 0L
        droppedBusy = 0L
        droppedInactive = 0L
        droppedDetectorUnavailable = 0L
    }

    private fun snapshotLocked(): FramePipelineStatsSnapshot {
        val dropped = droppedBusy + droppedInactive + droppedDetectorUnavailable
        val total = processed + dropped
        val sortedInferenceSamples = inferenceSamples.sorted()
        return FramePipelineStatsSnapshot(
            received = received,
            processed = processed,
            droppedBusy = droppedBusy,
            droppedInactive = droppedInactive,
            droppedDetectorUnavailable = droppedDetectorUnavailable,
            droppedFrameRate = if (total == 0L) 0f else dropped.toFloat() / total.toFloat(),
            inferenceP50Ms = percentile(sortedInferenceSamples, 0.50f),
            inferenceP95Ms = percentile(sortedInferenceSamples, 0.95f)
        )
    }

    private fun percentile(sortedSamples: List<Long>, percentile: Float): Long {
        if (sortedSamples.isEmpty()) return 0L
        val index = ceil((sortedSamples.size - 1) * percentile).toInt().coerceIn(0, sortedSamples.lastIndex)
        return sortedSamples[index]
    }

    companion object {
        private const val DEFAULT_MAX_INFERENCE_SAMPLES = 120
    }
}

internal data class FramePipelineStatsSnapshot(
    val received: Long,
    val processed: Long,
    val droppedBusy: Long,
    val droppedInactive: Long,
    val droppedDetectorUnavailable: Long,
    val droppedFrameRate: Float,
    val inferenceP50Ms: Long,
    val inferenceP95Ms: Long
)
