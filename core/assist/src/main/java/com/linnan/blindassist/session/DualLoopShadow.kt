package com.linnan.blindassist.session

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.DetectionSource
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp

/**
 * The only dual-loop runtime modes currently implemented.
 *
 * There is intentionally no active/actuating mode. The second loop may be
 * observed in an isolated build, but it cannot alter risk, event, or feedback.
 */
enum class DualLoopRuntimeMode {
    OFF,
    SHADOW_ABSTAIN_ONLY
}

enum class DualLoopShadowDisposition {
    OFF,
    EVIDENCE_ABSENT,
    SOURCE_NOT_ADMITTED,
    SOURCE_ID_INVALID,
    SOURCE_ABSTAINED,
    SOURCE_FRAME_MISSING,
    CURRENT_FRAME_MISMATCH,
    PREVIOUS_FRAME_INVALID,
    CLOCK_DOMAIN_MISMATCH,
    EVIDENCE_NOT_AVAILABLE,
    EVIDENCE_STALE,
    TARGET_NOT_SELECTED,
    TARGET_AMBIGUOUS,
    NONFINITE_RATE,
    LOW_QUALITY,
    ADMITTED_SHADOW
}

/**
 * Frame- and target-bound geometry evidence for the non-actuating second loop.
 *
 * It cannot carry a risk level, event identity, message, or feedback decision.
 * A producer that has no usable estimate must set [sourceAbstentionReason].
 */
data class DualLoopGeometryEvidence(
    val sourceContractId: String,
    val sourceId: String,
    val previousFrame: FrameStamp,
    val currentFrame: FrameStamp,
    val availableAtNs: Long,
    val validUntilNs: Long,
    val availabilityClockDomain: FrameClockDomain,
    val trackEpoch: String,
    val targetClassId: Int,
    val targetLabel: String,
    val targetBoundingBox: BoundingBox,
    val targetFrameSize: FrameSize,
    val targetSource: DetectionSource,
    val signedApproachRatePerS: Float?,
    val quality: Float?,
    val sourceAbstentionReason: String? = null
)

data class DualLoopSourceIdentity(
    val sourceContractId: String,
    val sourceId: String
)

data class DualLoopShadowObservation(
    val mode: DualLoopRuntimeMode,
    val disposition: DualLoopShadowDisposition,
    val sourceId: String? = null,
    val signedApproachRatePerS: Float? = null,
    val quality: Float? = null
) {
    val productionRiskUnchanged: Boolean
        get() = true
    val eventMutationAllowed: Boolean
        get() = false
    val feedbackMutationAllowed: Boolean
        get() = false
    val admitted: Boolean
        get() = disposition == DualLoopShadowDisposition.ADMITTED_SHADOW

    companion object {
        fun off(): DualLoopShadowObservation = DualLoopShadowObservation(
            mode = DualLoopRuntimeMode.OFF,
            disposition = DualLoopShadowDisposition.OFF
        )
    }
}

/**
 * Fail-closed admission for shadow-only geometry evidence.
 *
 * The production allowlist is empty. Tests or a separately isolated caller may
 * inject an explicit allowlist to verify the contract, but even admitted
 * evidence remains observational and cannot reach the event/feedback seam.
 */
class DualLoopShadowAdmitter(
    admittedSourceIdentities: Set<DualLoopSourceIdentity> = emptySet(),
    private val minimumQuality: Float = DEFAULT_MINIMUM_QUALITY
) {
    private val admittedSourceIdentities = admittedSourceIdentities.toSet()

    init {
        require(minimumQuality.isFinite() && minimumQuality in 0f..1f)
    }

    fun evaluate(
        mode: DualLoopRuntimeMode,
        evidence: DualLoopGeometryEvidence?,
        sourceFrame: FrameStamp?,
        detections: List<Detection>,
        baselineRisk: RiskResult,
        decisionAtNs: Long,
        decisionClockDomain: FrameClockDomain?
    ): DualLoopShadowObservation {
        if (mode == DualLoopRuntimeMode.OFF) return DualLoopShadowObservation.off()
        if (evidence == null) return abstain(DualLoopShadowDisposition.EVIDENCE_ABSENT)
        if (evidence.sourceContractId.isBlank() || evidence.sourceId.isBlank()) {
            return abstain(DualLoopShadowDisposition.SOURCE_ID_INVALID, evidence)
        }
        if (evidence.identity() !in admittedSourceIdentities) {
            return abstain(DualLoopShadowDisposition.SOURCE_NOT_ADMITTED, evidence)
        }
        if (!evidence.sourceAbstentionReason.isNullOrBlank()) {
            return abstain(DualLoopShadowDisposition.SOURCE_ABSTAINED, evidence)
        }
        if (sourceFrame == null) {
            return abstain(DualLoopShadowDisposition.SOURCE_FRAME_MISSING, evidence)
        }
        if (evidence.currentFrame != sourceFrame) {
            return abstain(DualLoopShadowDisposition.CURRENT_FRAME_MISMATCH, evidence)
        }
        if (!validFramePair(evidence.previousFrame, evidence.currentFrame, evidence.trackEpoch)) {
            return abstain(DualLoopShadowDisposition.PREVIOUS_FRAME_INVALID, evidence)
        }
        if (decisionClockDomain == null ||
            evidence.availabilityClockDomain != decisionClockDomain ||
            evidence.currentFrame.clockDomain != decisionClockDomain ||
            decisionClockDomain == FrameClockDomain.CAMERA_HARDWARE_UNMAPPED
        ) {
            return abstain(DualLoopShadowDisposition.CLOCK_DOMAIN_MISMATCH, evidence)
        }
        if (evidence.availableAtNs < evidence.currentFrame.capturedAtNs ||
            decisionAtNs < evidence.availableAtNs
        ) {
            return abstain(DualLoopShadowDisposition.EVIDENCE_NOT_AVAILABLE, evidence)
        }
        if (evidence.validUntilNs < evidence.availableAtNs || decisionAtNs > evidence.validUntilNs) {
            return abstain(DualLoopShadowDisposition.EVIDENCE_STALE, evidence)
        }

        val matchingTargets = detections.filter { detection -> evidence.matches(detection) }
        if (matchingTargets.size > 1) {
            return abstain(DualLoopShadowDisposition.TARGET_AMBIGUOUS, evidence)
        }
        val selected = baselineRisk.sourceDetection
        if (matchingTargets.singleOrNull() == null || selected == null || !evidence.matches(selected)) {
            return abstain(DualLoopShadowDisposition.TARGET_NOT_SELECTED, evidence)
        }
        val rate = evidence.signedApproachRatePerS
        if (rate == null || !rate.isFinite()) {
            return abstain(DualLoopShadowDisposition.NONFINITE_RATE, evidence)
        }
        val quality = evidence.quality
        if (quality == null || !quality.isFinite() || quality < minimumQuality || quality > 1f) {
            return abstain(DualLoopShadowDisposition.LOW_QUALITY, evidence)
        }
        return DualLoopShadowObservation(
            mode = mode,
            disposition = DualLoopShadowDisposition.ADMITTED_SHADOW,
            sourceId = evidence.sourceId,
            signedApproachRatePerS = rate,
            quality = quality
        )
    }

    private fun abstain(
        disposition: DualLoopShadowDisposition,
        evidence: DualLoopGeometryEvidence? = null
    ): DualLoopShadowObservation = DualLoopShadowObservation(
        mode = DualLoopRuntimeMode.SHADOW_ABSTAIN_ONLY,
        disposition = disposition,
        sourceId = evidence?.sourceId
    )

    private fun validFramePair(
        previous: FrameStamp,
        current: FrameStamp,
        trackEpoch: String
    ): Boolean {
        return trackEpoch.isNotBlank() &&
            previous.sourceId == current.sourceId &&
            previous.coordinateFrame == current.coordinateFrame &&
            previous.clockDomain == current.clockDomain &&
            previous.frameId < current.frameId &&
            previous.capturedAtNs < current.capturedAtNs
    }

    private fun DualLoopGeometryEvidence.matches(detection: Detection): Boolean {
        return targetClassId == detection.classId &&
            targetLabel == detection.label &&
            targetBoundingBox == detection.boundingBox &&
            targetFrameSize == detection.frameSize &&
            targetSource == detection.source
    }

    private fun DualLoopGeometryEvidence.identity(): DualLoopSourceIdentity =
        DualLoopSourceIdentity(sourceContractId, sourceId)

    companion object {
        const val CONTRACT_ID = "blindassist_dual_loop_geometry_shadow_input_v1"
        const val DEFAULT_MINIMUM_QUALITY = 0.50f
    }
}
