package com.linnan.blindassist.ustrf

import kotlin.math.atan2

/**
 * Vocabulary expected by the USTRF-SC document's downstream feedback contract.
 *
 * This is a shadow-only report. It is derived from, and cannot override, the fast-loop
 * [UstrfSafetyDecision]. A device adapter must never treat it as a motor or user-direction
 * authority until independent device evidence and GPT/Codex safety-review gates are satisfied.
 */
enum class UstrfStructuredAction {
    CONTINUE,
    ADJUST_LEFT,
    ADJUST_RIGHT,
    SLOW_DOWN,
    STOP,
    SCAN
}

data class UstrfStructuredSafetyOutput(
    val action: UstrfStructuredAction,
    /** Positive is the configured leftward heading convention. */
    val headingDeltaRadians: Float,
    val speedScale: Float,
    val risk: Float,
    val confidence: Float,
    val corridorWidthMeters: Float?,
    val validUntilNs: Long,
    val reasons: Set<UstrfSafetyReason>,
    val shadowOnly: Boolean = true
) {
    init {
        require(headingDeltaRadians.isFinite())
        require(speedScale in 0f..1f)
        require(risk in 0f..1f)
        require(confidence in 0f..1f)
        require(corridorWidthMeters == null || (corridorWidthMeters.isFinite() && corridorWidthMeters > 0f))
        require(validUntilNs >= 0L)
        require(reasons.isNotEmpty())
        require(shadowOnly) { "structured USTRF output must remain shadow-only" }
    }
}

/** Maps a signed lateral cell coordinate to the target hardware's left/right convention. */
enum class UstrfLateralConvention { LEFT_POSITIVE, RIGHT_POSITIVE }

class UstrfStructuredSafetyOutputMapper(
    val gridSpec: UstrfGridSpec = UstrfGridSpec.LEGACY_KERNEL,
    private val lateralConvention: UstrfLateralConvention = UstrfLateralConvention.LEFT_POSITIVE
) {
    fun map(
        decision: UstrfSafetyDecision,
        selected: UstrfCorridorCandidate?
    ): UstrfStructuredSafetyOutput {
        require(UstrfSafetyReason.SHADOW_ONLY in decision.reasons) {
            "structured output requires a shadow-only supervisor decision"
        }
        val selectedMatchesDecision = selected != null &&
            selected.offsetCells == decision.experimentalCorridorOffsetCells && selected.hardSafe
        val nominalOnly = decision.reasons == setOf(UstrfSafetyReason.SHADOW_ONLY) && selectedMatchesDecision
        val action = when (decision.action) {
            UstrfSafetyAction.STOP_AND_REASSESS -> UstrfStructuredAction.STOP
            UstrfSafetyAction.SCAN -> UstrfStructuredAction.SCAN
            UstrfSafetyAction.SLOW_DOWN -> when {
                !nominalOnly -> UstrfStructuredAction.SLOW_DOWN
                selected!!.offsetCells == 0 -> UstrfStructuredAction.CONTINUE
                isLeftward(selected.offsetCells) -> UstrfStructuredAction.ADJUST_LEFT
                else -> UstrfStructuredAction.ADJUST_RIGHT
            }
        }
        val heading = if (action == UstrfStructuredAction.ADJUST_LEFT || action == UstrfStructuredAction.ADJUST_RIGHT) {
            val signedOffsetMeters = selected!!.offsetCells * gridSpec.cellMeters
            atan2(signedOffsetMeters, gridSpec.lookaheadMeters).toFloat() * if (lateralConvention == UstrfLateralConvention.LEFT_POSITIVE) 1f else -1f
        } else 0f
        val speed = when (action) {
            UstrfStructuredAction.CONTINUE -> 1f
            UstrfStructuredAction.ADJUST_LEFT, UstrfStructuredAction.ADJUST_RIGHT -> .70f
            UstrfStructuredAction.SLOW_DOWN -> .50f
            UstrfStructuredAction.STOP, UstrfStructuredAction.SCAN -> 0f
        }
        return UstrfStructuredSafetyOutput(
            action = action,
            headingDeltaRadians = heading,
            speedScale = speed,
            risk = decision.risk,
            confidence = decision.confidence,
            corridorWidthMeters = selected?.let { gridSpec.bodyWidthMeters },
            validUntilNs = decision.validUntilNs,
            reasons = decision.reasons
        )
    }

    private fun isLeftward(offsetCells: Int): Boolean = when (lateralConvention) {
        UstrfLateralConvention.LEFT_POSITIVE -> offsetCells > 0
        UstrfLateralConvention.RIGHT_POSITIVE -> offsetCells < 0
    }
}
