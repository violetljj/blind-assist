package com.linnan.blindassist.ustrf

/**
 * A reference-free trigger candidate. It intentionally carries no image payload and no safety
 * action. HIGH_ENTROPY must provide the fast-loop uncertainty that caused the request.
 */
data class UstrfSlowLoopTriggerRequest(
    val trigger: UstrfSlowLoopTrigger,
    val sourceFrame: UstrfFrameStamp,
    val observedAtNs: Long,
    val uncertainty: Float? = null
) {
    init {
        require(observedAtNs >= sourceFrame.capturedAtNs)
        require(uncertainty == null || uncertainty in 0f..1f)
        require(trigger != UstrfSlowLoopTrigger.HIGH_ENTROPY || uncertainty != null)
    }
}

enum class UstrfSlowLoopEventSuppression {
    SOURCE_FRAME_ROLLBACK,
    COORDINATE_FRAME_CHANGED,
    HIGH_ENTROPY_BELOW_THRESHOLD,
    MINIMUM_INTER_EVENT_INTERVAL,
    PERIODIC_KEYFRAME_INTERVAL_NOT_REACHED
}

sealed interface UstrfSlowLoopEventGateResult {
    data class Accepted(val event: UstrfSlowLoopEvent) : UstrfSlowLoopEventGateResult
    data class Suppressed(val reason: UstrfSlowLoopEventSuppression) : UstrfSlowLoopEventGateResult
}

/**
 * Stateful, deterministic slow-loop scheduler for the phone's reference-free shadow route.
 * It has no model implementation and cannot create a corridor, route, speed or safety decision.
 */
class UstrfSlowLoopEventGate(
    private val minimumInterEventNs: Long = 500_000_000L,
    private val periodicKeyframeIntervalNs: Long = 2_000_000_000L,
    private val eventTtlNs: Long = 2_000_000_000L,
    private val highEntropyThreshold: Float = .60f
) {
    private var priorFrame: UstrfFrameStamp? = null
    private var lastAcceptedAtNs: Long? = null

    init {
        require(minimumInterEventNs > 0L && periodicKeyframeIntervalNs >= minimumInterEventNs && eventTtlNs > 0L)
        require(highEntropyThreshold in 0f..1f)
    }

    fun request(input: UstrfSlowLoopTriggerRequest): UstrfSlowLoopEventGateResult {
        priorFrame?.let { prior ->
            if (input.sourceFrame.frameId <= prior.frameId || input.sourceFrame.capturedAtNs <= prior.capturedAtNs) {
                return UstrfSlowLoopEventGateResult.Suppressed(UstrfSlowLoopEventSuppression.SOURCE_FRAME_ROLLBACK)
            }
            if (input.sourceFrame.coordinateFrame != prior.coordinateFrame) {
                return UstrfSlowLoopEventGateResult.Suppressed(UstrfSlowLoopEventSuppression.COORDINATE_FRAME_CHANGED)
            }
        }
        priorFrame = input.sourceFrame
        if (input.trigger == UstrfSlowLoopTrigger.HIGH_ENTROPY && input.uncertainty!! < highEntropyThreshold) {
            return UstrfSlowLoopEventGateResult.Suppressed(UstrfSlowLoopEventSuppression.HIGH_ENTROPY_BELOW_THRESHOLD)
        }
        val previousAcceptedAt = lastAcceptedAtNs
        if (previousAcceptedAt != null) {
            val elapsed = input.observedAtNs - previousAcceptedAt
            if (input.trigger == UstrfSlowLoopTrigger.PERIODIC_KEYFRAME && elapsed < periodicKeyframeIntervalNs) {
                return UstrfSlowLoopEventGateResult.Suppressed(UstrfSlowLoopEventSuppression.PERIODIC_KEYFRAME_INTERVAL_NOT_REACHED)
            }
            if (elapsed < minimumInterEventNs) {
                return UstrfSlowLoopEventGateResult.Suppressed(UstrfSlowLoopEventSuppression.MINIMUM_INTER_EVENT_INTERVAL)
            }
        }
        val event = UstrfSlowLoopEvent(
            eventId = "slow-${input.sourceFrame.coordinateFrame}-${input.sourceFrame.frameId}",
            trigger = input.trigger,
            queryFrame = input.sourceFrame,
            issuedAtNs = input.observedAtNs,
            validUntilNs = input.observedAtNs + eventTtlNs
        )
        lastAcceptedAtNs = input.observedAtNs
        return UstrfSlowLoopEventGateResult.Accepted(event)
    }

    fun reset() {
        priorFrame = null
        lastAcceptedAtNs = null
    }
}
