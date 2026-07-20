package com.linnan.blindassist.benchmark

/**
 * Separates user-delivered repeats from attempts that the event gate correctly suppressed.
 *
 * Samples are grouped by the reviewed expected event identity. This deliberately exposes a
 * runtime event-ID split as regeneration instead of letting it hide a second delivered alert.
 */
internal data class EventLifecycleSample(
    val riskEventId: String?,
    val sequenceId: String?,
    val fallbackFrameId: String,
    val frameIndex: Int?,
    val expectedEventPhase: String?,
    val expectedShouldAlert: Boolean,
    val actualAlert: Boolean,
    val suppressedDuplicateAttempt: Boolean,
    val runtimeEventId: String?
)

internal data class EventLifecycleSummary(
    val observedEventCount: Int,
    val deliveredAlertCount: Int,
    val deliveredRepeatedAlertCount: Int,
    val suppressedDuplicateAttemptCount: Int,
    val passedEventCount: Int,
    val clearedPassedEventCount: Int,
    val eventRegenerationCount: Int,
    val meanPostEventAlertLatencyFrames: Double,
    val falseAlertCount: Int,
    val sequenceDurationMs: Long
) {
    val deliveredRepeatedAlertRate: Double
        get() = if (deliveredAlertCount == 0) 0.0 else {
            deliveredRepeatedAlertCount.toDouble() / deliveredAlertCount.toDouble()
        }

    val postEventClearanceRate: Double
        get() = if (passedEventCount == 0) 0.0 else {
            clearedPassedEventCount.toDouble() / passedEventCount.toDouble()
        }

    val falseAlertsPerMinute: Double?
        get() = if (sequenceDurationMs <= 0L) null else {
            falseAlertCount * 60_000.0 / sequenceDurationMs.toDouble()
        }
}

internal object EventLifecycleMetrics {
    fun summarize(samples: List<EventLifecycleSample>): EventLifecycleSummary {
        val events = samples.groupBy(::eventKey)
        val deliveredAlertCount = samples.count { it.actualAlert }
        val deliveredRepeatedAlertCount = events.values.sumOf { event ->
            (event.count { it.actualAlert } - 1).coerceAtLeast(0)
        }
        val passedEvents = events.values.filter { event ->
            event.any { it.expectedEventPhase == PASSED_EVENT_PHASE }
        }
        val clearedPassedEventCount = passedEvents.count { event ->
            event.none { it.expectedEventPhase == PASSED_EVENT_PHASE && it.actualAlert }
        }
        val postEventLatencies = passedEvents.map { event ->
            val passed = event.filter { it.expectedEventPhase == PASSED_EVENT_PHASE }
                .sortedBy { it.frameIndex ?: Int.MAX_VALUE }
            val firstPassedFrame = passed.firstOrNull()?.frameIndex
            val lastPostEventAlertFrame = passed.lastOrNull { it.actualAlert }?.frameIndex
            if (firstPassedFrame == null || lastPostEventAlertFrame == null) 0.0 else {
                (lastPostEventAlertFrame - firstPassedFrame + 1).coerceAtLeast(0).toDouble()
            }
        }
        val sequenceDurationMs = samples
            .filter { it.sequenceId?.isNotBlank() == true && it.frameIndex != null }
            .groupBy { requireNotNull(it.sequenceId) }
            .values
            .sumOf { sequence ->
                val indices = sequence.mapNotNull { it.frameIndex }
                if (indices.isEmpty()) 0L else {
                    (indices.max().toLong() - indices.min().toLong() + 1L) * FRAME_STEP_MS
                }
            }
        return EventLifecycleSummary(
            observedEventCount = events.size,
            deliveredAlertCount = deliveredAlertCount,
            deliveredRepeatedAlertCount = deliveredRepeatedAlertCount,
            suppressedDuplicateAttemptCount = samples.count { it.suppressedDuplicateAttempt },
            passedEventCount = passedEvents.size,
            clearedPassedEventCount = clearedPassedEventCount,
            eventRegenerationCount = events.values.sumOf { event ->
                (event.mapNotNull { it.runtimeEventId?.takeIf(String::isNotBlank) }.toSet().size - 1)
                    .coerceAtLeast(0)
            },
            meanPostEventAlertLatencyFrames = if (postEventLatencies.isEmpty()) 0.0 else {
                postEventLatencies.average()
            },
            falseAlertCount = samples.count { !it.expectedShouldAlert && it.actualAlert },
            sequenceDurationMs = sequenceDurationMs
        )
    }

    private fun eventKey(sample: EventLifecycleSample): String =
        sample.riskEventId?.takeIf { it.isNotBlank() }
            ?: sample.sequenceId?.takeIf { it.isNotBlank() }
            ?: "frame:${sample.fallbackFrameId}"

    private const val PASSED_EVENT_PHASE = "PASSED"
    private const val FRAME_STEP_MS = 100L
}
