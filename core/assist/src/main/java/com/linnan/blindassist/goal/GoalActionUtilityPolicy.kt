package com.linnan.blindassist.goal

import kotlin.math.ln
import kotlin.math.sqrt

data class GoalActionUtilityContext(
    val deficit: GoalObservationDeficit,
    val bearing: CameraRelativeBearing
)

data class GoalActionUtilityStats(
    val context: GoalActionUtilityContext,
    val action: GoalCopilotAction,
    val improved: Long,
    val noGain: Long,
    val contradicted: Long
) {
    val evaluated: Long get() = improved + noGain + contradicted
    val empiricalUtility: Double get() = if (evaluated == 0L) 0.0 else improved.toDouble() / evaluated
}

data class GoalActionUtilitySelection(
    val context: GoalActionUtilityContext,
    val failedAction: GoalCopilotAction,
    val action: GoalCopilotAction,
    val posteriorMean: Double,
    val upperConfidenceUtility: Double,
    val contextEvaluations: Long
)

/**
 * Bounded online utility learner for observation-seeking actions.
 *
 * Only execution-confirmed, comparable outcomes may update it. Candidate sets
 * encode product-safe exploration boundaries; the learner ranks actions inside
 * that boundary and can be shared across goal-controller sessions.
 */
class GoalActionUtilityPolicy(
    private val explorationStrength: Double = DEFAULT_EXPLORATION_STRENGTH,
    private val maxRememberedReceipts: Int = DEFAULT_MAX_REMEMBERED_RECEIPTS
) {
    private data class MutableStats(
        var improved: Long = 0,
        var noGain: Long = 0,
        var contradicted: Long = 0
    ) {
        val evaluated: Long get() = improved + noGain + contradicted
    }

    private val stats = mutableMapOf<Pair<GoalActionUtilityContext, GoalCopilotAction>, MutableStats>()
    private val observedReceipts = linkedSetOf<String>()

    init {
        require(explorationStrength >= 0.0 && explorationStrength.isFinite())
        require(maxRememberedReceipts > 0)
    }

    /** Returns true only when a new authoritative outcome changed the policy. */
    fun observe(outcome: GoalObservationActionOutcome): Boolean {
        if (outcome.state == GoalObservationActionOutcomeState.UNKNOWN) return false
        if (!observedReceipts.add(outcome.receipt.receiptId)) return false
        while (observedReceipts.size > maxRememberedReceipts) {
            observedReceipts.remove(observedReceipts.first())
        }
        val context = GoalActionUtilityContext(
            deficit = outcome.receipt.priorDeficit,
            bearing = outcome.receipt.priorBearing
        )
        val counts = stats.getOrPut(context to outcome.receipt.action) { MutableStats() }
        when (outcome.state) {
            GoalObservationActionOutcomeState.IMPROVED -> counts.improved += 1
            GoalObservationActionOutcomeState.NO_GAIN -> counts.noGain += 1
            GoalObservationActionOutcomeState.CONTRADICTED -> counts.contradicted += 1
            GoalObservationActionOutcomeState.UNKNOWN -> Unit
        }
        return true
    }

    fun selectRepair(outcome: GoalObservationActionOutcome): GoalActionUtilitySelection? {
        if (outcome.state !in setOf(
                GoalObservationActionOutcomeState.NO_GAIN,
                GoalObservationActionOutcomeState.CONTRADICTED
            )
        ) return null
        val context = GoalActionUtilityContext(
            deficit = outcome.receipt.priorDeficit,
            bearing = outcome.receipt.priorBearing
        )
        val candidates = allowedActions(context).filter { it != outcome.receipt.action }
        if (candidates.isEmpty()) return null
        val contextEvaluations = stats
            .filterKeys { it.first == context }
            .values
            .sumOf(MutableStats::evaluated)
        val ranked = candidates.mapIndexed { index, action ->
            val counts = stats[context to action] ?: MutableStats()
            val posteriorMean = (counts.improved + 1.0) / (counts.evaluated + 2.0)
            val bonus = explorationStrength * sqrt(
                ln(contextEvaluations + 2.0) / (counts.evaluated + 1.0)
            )
            CandidateUtility(index, action, posteriorMean, posteriorMean + bonus)
        }.maxWithOrNull(
            compareBy<CandidateUtility> { it.upperConfidenceUtility }
                .thenBy { -it.priorityIndex }
        ) ?: return null
        return GoalActionUtilitySelection(
            context = context,
            failedAction = outcome.receipt.action,
            action = ranked.action,
            posteriorMean = ranked.posteriorMean,
            upperConfidenceUtility = ranked.upperConfidenceUtility,
            contextEvaluations = contextEvaluations
        )
    }

    fun snapshot(): List<GoalActionUtilityStats> = stats.map { (key, counts) ->
        GoalActionUtilityStats(
            context = key.first,
            action = key.second,
            improved = counts.improved,
            noGain = counts.noGain,
            contradicted = counts.contradicted
        )
    }.sortedWith(
        compareBy<GoalActionUtilityStats>(
            { it.context.deficit.name },
            { it.context.bearing.name },
            { it.action.name }
        )
    )

    private fun allowedActions(context: GoalActionUtilityContext): List<GoalCopilotAction> {
        val directedPan = when (context.bearing) {
            CameraRelativeBearing.LEFT -> GoalCopilotAction.PAN_LEFT_TO_IDENTITY
            CameraRelativeBearing.RIGHT -> GoalCopilotAction.PAN_RIGHT_TO_IDENTITY
            CameraRelativeBearing.FORWARD,
            CameraRelativeBearing.UNKNOWN -> GoalCopilotAction.HOLD_STEADY_LOCALIZE
        }
        val directedScan = when (context.bearing) {
            CameraRelativeBearing.LEFT -> GoalCopilotAction.SCAN_LAST_LEFT
            CameraRelativeBearing.RIGHT -> GoalCopilotAction.SCAN_LAST_RIGHT
            CameraRelativeBearing.FORWARD -> GoalCopilotAction.SCAN_LAST_FORWARD
            CameraRelativeBearing.UNKNOWN -> GoalCopilotAction.SWEEP_SEARCH
        }
        val actions = when (context.deficit) {
            GoalObservationDeficit.NO_LOCALIZABLE_EVIDENCE,
            GoalObservationDeficit.TARGET_NOT_PROPOSED -> listOf(
                GoalCopilotAction.SWEEP_SEARCH,
                directedScan,
                GoalCopilotAction.HOLD_STEADY_LOCALIZE
            )
            GoalObservationDeficit.DECISIVE_IDENTITY_UNREADABLE -> buildList {
                add(GoalCopilotAction.SIDESTEP_FOR_DISAMBIGUATION)
                add(GoalCopilotAction.HOLD_STEADY_CONFIRM)
                if (context.bearing == CameraRelativeBearing.FORWARD) {
                    add(GoalCopilotAction.APPROACH_FOR_IDENTITY)
                }
                add(directedPan)
            }
            GoalObservationDeficit.ASSOCIATION_AMBIGUOUS -> listOf(
                GoalCopilotAction.HOLD_STEADY_CONFIRM,
                GoalCopilotAction.SIDESTEP_FOR_DISAMBIGUATION,
                directedPan,
                GoalCopilotAction.SWEEP_SEARCH
            )
            GoalObservationDeficit.REACQUIRE_CONFIRMATION_PENDING -> listOf(
                GoalCopilotAction.HOLD_STEADY_CONFIRM,
                directedScan,
                GoalCopilotAction.SWEEP_SEARCH,
                GoalCopilotAction.SIDESTEP_FOR_DISAMBIGUATION
            )
            GoalObservationDeficit.METRIC_POSITION_UNKNOWN,
            GoalObservationDeficit.ENDPOINT_NOT_READY -> listOf(
                GoalCopilotAction.HOLD_STEADY_LOCALIZE,
                GoalCopilotAction.SIDESTEP_FOR_DISAMBIGUATION,
                GoalCopilotAction.SWEEP_SEARCH
            )
            GoalObservationDeficit.FUNCTIONAL_TARGET_UNKNOWN -> listOf(
                GoalCopilotAction.SIDESTEP_FOR_DISAMBIGUATION,
                GoalCopilotAction.HOLD_STEADY_CONFIRM,
                directedPan,
                GoalCopilotAction.SWEEP_SEARCH
            )
            GoalObservationDeficit.NONE -> listOf(
                GoalCopilotAction.HOLD_STEADY_LOCALIZE,
                directedPan,
                GoalCopilotAction.SIDESTEP_FOR_DISAMBIGUATION
            )
        }
        return actions.distinct()
    }

    private data class CandidateUtility(
        val priorityIndex: Int,
        val action: GoalCopilotAction,
        val posteriorMean: Double,
        val upperConfidenceUtility: Double
    )

    companion object {
        const val DEFAULT_EXPLORATION_STRENGTH = 0.35
        const val DEFAULT_MAX_REMEMBERED_RECEIPTS = 2_048
    }
}
