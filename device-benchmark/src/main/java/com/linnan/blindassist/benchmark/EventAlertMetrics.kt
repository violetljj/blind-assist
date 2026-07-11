package com.linnan.blindassist.benchmark

/**
 * Event-level alert accounting for continuous evaluation data.
 *
 * A frame contributes only when its annotation expects an alert.  One accepted alert anywhere
 * in that window satisfies the event; frame-level recall remains a separate diagnostic metric.
 */
internal data class EventAlertSample(
    val riskEventId: String?,
    val sequenceId: String?,
    val fallbackFrameId: String,
    val expectedShouldAlert: Boolean,
    val expectedCritical: Boolean,
    val actualAlert: Boolean
)

internal data class EventAlertSummary(
    val eventCount: Int,
    val hitCount: Int,
    val criticalEventMissCount: Int
) {
    val recall: Double
        get() = if (eventCount == 0) 0.0 else hitCount.toDouble() / eventCount.toDouble()
}

internal object EventAlertMetrics {
    fun summarize(samples: List<EventAlertSample>): EventAlertSummary {
        val windows = samples
            .filter { it.expectedShouldAlert }
            .groupBy { sample ->
                sample.riskEventId?.takeIf { it.isNotBlank() }
                    ?: sample.sequenceId?.takeIf { it.isNotBlank() }
                    ?: "frame:${sample.fallbackFrameId}"
            }
        val hitCount = windows.values.count { window -> window.any { it.actualAlert } }
        val criticalEventMissCount = windows.values.count { window ->
            window.any { it.expectedCritical } && window.none { it.actualAlert }
        }
        return EventAlertSummary(
            eventCount = windows.size,
            hitCount = hitCount,
            criticalEventMissCount = criticalEventMissCount
        )
    }
}
