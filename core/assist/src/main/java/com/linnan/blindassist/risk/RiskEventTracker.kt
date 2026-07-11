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
    val clearAfterRecedingOrMissingFrames: Int = 3
) {
    init {
        require(centerCorridorHalfWidthRatio in 0f..0.5f)
        require(maxCenterDeltaForSameEvent in 0f..1f)
        require(clearAfterRecedingOrMissingFrames >= 2)
    }
}

/**
 * Event-level feedback gate for segmentation candidates. YOLO detections deliberately pass
 * through untouched so existing detector reminder behaviour is preserved.
 */
class RiskEventTracker(
    private val config: RiskEventTrackerConfig = RiskEventTrackerConfig()
) {
    private var active: ActiveEvent? = null
    private var nextId = 1

    fun update(risk: RiskResult): RiskEventSnapshot {
        val detection = risk.sourceDetection
        if (!isTrackedCandidate(detection)) {
            return onMissingOrNonCentralCandidate()
        }
        val candidate = requireNotNull(detection)
        if (!isInCenterCorridor(candidate)) {
            clear(RiskEventClearReason.LEFT_CENTER_CORRIDOR)
            return RiskEventSnapshot.none()
        }

        val current = active
        if (current == null || !current.matches(candidate, config)) {
            if (current != null) clear(RiskEventClearReason.REPLACED_BY_NEW_TARGET)
            active = ActiveEvent(id = "seg-${nextId++}", label = candidate.label, centerXRatio = centerRatio(candidate))
        }
        val event = requireNotNull(active)
        event.centerXRatio = centerRatio(candidate)
        if (risk.approachTrend == ApproachTrend.RECEDING) {
            event.state = RiskEventState.PASSED_OR_RECEDING
            event.recedingOrMissingFrames += 1
            if (event.recedingOrMissingFrames >= config.clearAfterRecedingOrMissingFrames) {
                clear(RiskEventClearReason.THREE_RECEDING_OR_MISSING_FRAMES)
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
        clear(RiskEventClearReason.SESSION_RESET)
        active = null
    }

    private fun onMissingOrNonCentralCandidate(): RiskEventSnapshot {
        val event = active ?: return RiskEventSnapshot.none()
        event.state = RiskEventState.PASSED_OR_RECEDING
        event.recedingOrMissingFrames += 1
        if (event.recedingOrMissingFrames >= config.clearAfterRecedingOrMissingFrames) {
            clear(RiskEventClearReason.THREE_RECEDING_OR_MISSING_FRAMES)
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

    private fun clear(reason: RiskEventClearReason) {
        active?.let {
            it.state = RiskEventState.CLEARED
            it.clearReason = reason
        }
        active = null
    }

    private fun isTrackedCandidate(detection: Detection?): Boolean =
        detection?.source == DetectionSource.SEGMENTATION

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
        var clearReason: RiskEventClearReason? = null
    ) {
        fun matches(detection: Detection, config: RiskEventTrackerConfig): Boolean =
            label == detection.label && abs(centerXRatio - detection.boundingBox.centerX / detection.frameSize.width) <= config.maxCenterDeltaForSameEvent
    }
}
