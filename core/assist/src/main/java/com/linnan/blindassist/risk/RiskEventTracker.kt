package com.linnan.blindassist.risk

import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import kotlin.math.abs

/** Tracks one centre-path segmentation hazard from first approach until it is passed. */
enum class RiskEventState {
    APPROACHING,
    ALERTED,
    PASSED_OR_RECEDING,
    CLEARED
}

enum class RiskEventClearReason {
    LEFT_CENTER_CORRIDOR,
    THREE_RECEDING_OR_MISSING_FRAMES,
    REPLACED_BY_NEW_TARGET,
    SESSION_RESET
}

data class RiskEventSnapshot(
    val eventId: String? = null,
    val state: RiskEventState? = null,
    val suppressesFeedback: Boolean = false,
    val clearReason: RiskEventClearReason? = null
) {
    companion object {
        fun none() = RiskEventSnapshot()
    }
}

data class RiskEventTrackerConfig(
    val centerCorridorHalfWidthRatio: Float = 0.42f,
    val maxCenterDeltaForSameEvent: Float = 0.25f,
    val clearAfterRecedingOrMissingFrames: Int = 3,
    val postPassReappearanceHoldMs: Long = 1_000L,
    val postPassMaxCenterDeltaForSameEvent: Float = 0.5f,
    val trackCenterObjectDetectorPerson: Boolean = false
) {
    init {
        require(centerCorridorHalfWidthRatio in 0f..0.5f)
        require(maxCenterDeltaForSameEvent in 0f..1f)
        require(clearAfterRecedingOrMissingFrames >= 2)
        require(postPassReappearanceHoldMs >= 0L)
        require(postPassMaxCenterDeltaForSameEvent in 0f..1f)
    }
}

/**
 * Event-level feedback gate for segmentation candidates. Object-detector persons remain
 * untouched unless an explicitly configured experiment opts them in.
 */
class RiskEventTracker(
    private val config: RiskEventTrackerConfig = RiskEventTrackerConfig()
) {
    private var active: ActiveEvent? = null
    private var recentlyPassed: PassedEvent? = null
    private var nextId = 1

    fun update(risk: RiskResult, nowMs: Long = System.currentTimeMillis()): RiskEventSnapshot {
        expirePassedEvent(nowMs)
        val detection = risk.sourceDetection
        if (!isTrackedCandidate(detection)) {
            return onMissingOrNonCentralCandidate(nowMs)
        }
        val candidate = requireNotNull(detection)
        if (!isInCenterCorridor(candidate)) {
            clear(RiskEventClearReason.LEFT_CENTER_CORRIDOR, nowMs)
            return RiskEventSnapshot.none()
        }

        recentlyPassed?.takeIf { it.matches(candidate, config) }?.let { passed ->
            return passed.snapshot()
        }

        val current = active
        if (current == null || !current.matches(candidate, config)) {
            if (current != null) clear(RiskEventClearReason.REPLACED_BY_NEW_TARGET, nowMs)
            active = ActiveEvent(id = "seg-${nextId++}", label = candidate.label, centerXRatio = centerRatio(candidate))
        }
        val event = requireNotNull(active)
        event.centerXRatio = centerRatio(candidate)
        if (risk.approachTrend == ApproachTrend.RECEDING) {
            event.state = RiskEventState.PASSED_OR_RECEDING
            event.recedingOrMissingFrames += 1
            if (event.recedingOrMissingFrames >= config.clearAfterRecedingOrMissingFrames) {
                clear(RiskEventClearReason.THREE_RECEDING_OR_MISSING_FRAMES, nowMs)
                return RiskEventSnapshot.none()
            }
        } else {
            event.recedingOrMissingFrames = 0
        }
        return snapshot(event)
    }

    /** Tracks one frame-bound, object-agnostic route-risk identity without manufacturing a box. */
    fun updateExternalEvidence(
        risk: RiskResult,
        eventKey: String,
        nowMs: Long = System.currentTimeMillis()
    ): RiskEventSnapshot {
        require(risk.sourceDetection == null) { "external risk evidence must not contain a detection" }
        require(eventKey.isNotBlank()) { "external risk event key must be non-blank" }
        expirePassedEvent(nowMs)
        if (risk.level == RiskLevel.NONE || risk.evidenceState == RiskEvidenceState.NO_SUPPORTED_TARGET_EVIDENCE) {
            return onMissingOrNonCentralCandidate(nowMs)
        }
        recentlyPassed?.takeIf { it.matchesExternal(eventKey) }?.let { return it.snapshot() }
        val current = active
        if (current == null || !current.matchesExternal(eventKey)) {
            if (current != null) clear(RiskEventClearReason.REPLACED_BY_NEW_TARGET, nowMs)
            active = ActiveEvent(
                id = "risk-${nextId++}",
                label = "external-risk-evidence",
                centerXRatio = 0.5f,
                externalEventKey = eventKey
            )
        }
        val event = requireNotNull(active)
        if (risk.approachTrend == ApproachTrend.RECEDING) {
            event.state = RiskEventState.PASSED_OR_RECEDING
            event.recedingOrMissingFrames += 1
            if (event.recedingOrMissingFrames >= config.clearAfterRecedingOrMissingFrames) {
                clear(RiskEventClearReason.THREE_RECEDING_OR_MISSING_FRAMES, nowMs)
                return RiskEventSnapshot.none()
            }
        } else {
            event.recedingOrMissingFrames = 0
        }
        return snapshot(event)
    }

    /** Mark the active event only after speech or vibration was actually accepted. */
    fun recordFeedback(snapshot: RiskEventSnapshot, decision: FeedbackDecision): RiskEventSnapshot {
        val event = active ?: return RiskEventSnapshot.none()
        if (snapshot.eventId == event.id && decision.triggered) {
            event.state = RiskEventState.ALERTED
            event.wasAlerted = true
            event.recedingOrMissingFrames = 0
        }
        return snapshot(event)
    }

    fun reset() {
        clear(RiskEventClearReason.SESSION_RESET, System.currentTimeMillis())
        active = null
        recentlyPassed = null
    }

    private fun onMissingOrNonCentralCandidate(nowMs: Long): RiskEventSnapshot {
        val event = active ?: return RiskEventSnapshot.none()
        event.state = RiskEventState.PASSED_OR_RECEDING
        event.recedingOrMissingFrames += 1
        if (event.recedingOrMissingFrames >= config.clearAfterRecedingOrMissingFrames) {
            clear(RiskEventClearReason.THREE_RECEDING_OR_MISSING_FRAMES, nowMs)
            return RiskEventSnapshot.none()
        }
        return snapshot(event)
    }

    private fun snapshot(event: ActiveEvent): RiskEventSnapshot = RiskEventSnapshot(
        eventId = event.id,
        state = event.state,
        suppressesFeedback = event.wasAlerted,
        clearReason = event.clearReason
    )

    private fun expirePassedEvent(nowMs: Long) {
        if (recentlyPassed?.expiresAtMs?.let { nowMs >= it } == true) {
            recentlyPassed = null
        }
    }

    private fun clear(reason: RiskEventClearReason, nowMs: Long) {
        active?.let {
            if (reason == RiskEventClearReason.THREE_RECEDING_OR_MISSING_FRAMES &&
                it.wasAlerted && config.postPassReappearanceHoldMs > 0L
            ) {
                recentlyPassed = PassedEvent(
                    id = it.id,
                    label = it.label,
                    centerXRatio = it.centerXRatio,
                    expiresAtMs = nowMs + config.postPassReappearanceHoldMs,
                    externalEventKey = it.externalEventKey
                )
            }
            it.state = RiskEventState.CLEARED
            it.clearReason = reason
        }
        active = null
    }

    private fun isTrackedCandidate(detection: Detection?): Boolean =
        detection?.source == DetectionSource.SEGMENTATION ||
            (config.trackCenterObjectDetectorPerson &&
                detection?.source == DetectionSource.OBJECT_DETECTOR &&
                detection.label == "person")

    private fun isInCenterCorridor(detection: Detection): Boolean =
        abs(centerRatio(detection) - 0.5f) <= config.centerCorridorHalfWidthRatio

    private fun centerRatio(detection: Detection): Float =
        detection.boundingBox.centerX / detection.frameSize.width.toFloat()

    private data class ActiveEvent(
        val id: String,
        val label: String,
        var centerXRatio: Float,
        var state: RiskEventState = RiskEventState.APPROACHING,
        var wasAlerted: Boolean = false,
        var recedingOrMissingFrames: Int = 0,
        var clearReason: RiskEventClearReason? = null,
        val externalEventKey: String? = null
    ) {
        fun matches(detection: Detection, config: RiskEventTrackerConfig): Boolean =
            externalEventKey == null && label == detection.label &&
                abs(centerXRatio - detection.boundingBox.centerX / detection.frameSize.width) <=
                config.maxCenterDeltaForSameEvent

        fun matchesExternal(eventKey: String): Boolean = externalEventKey == eventKey
    }

    private data class PassedEvent(
        val id: String,
        val label: String,
        val centerXRatio: Float,
        val expiresAtMs: Long,
        val externalEventKey: String? = null
    ) {
        fun matches(detection: Detection, config: RiskEventTrackerConfig): Boolean =
            externalEventKey == null && label == detection.label &&
                abs(centerXRatio - detection.boundingBox.centerX / detection.frameSize.width) <=
                config.postPassMaxCenterDeltaForSameEvent

        fun matchesExternal(eventKey: String): Boolean = externalEventKey == eventKey

        fun snapshot(): RiskEventSnapshot = RiskEventSnapshot(
            eventId = id,
            state = RiskEventState.PASSED_OR_RECEDING,
            suppressesFeedback = true,
            clearReason = RiskEventClearReason.THREE_RECEDING_OR_MISSING_FRAMES
        )
    }
}
