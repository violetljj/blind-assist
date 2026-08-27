package com.linnan.blindassist.semanticanchor

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln

internal data class OcrGoalLockUpdate(
    val phase: AnchorPhase,
    val selected: AnchorCandidate?,
    val hitStreak: Int,
    val missStreak: Int,
    val beliefScore: Double,
    val completionEvidenceFrames: Int,
    val lockAcquired: Boolean,
    val reacquired: Boolean,
    val evidence: String,
    val guidance: MarkerGuidance,
)

/**
 * Candidate-bound L10-R0 belief for natural-text goals.
 *
 * Semantic text admits a proposal. Short motion memory keeps the live binding stable, while a
 * slower spatial prototype supports explicit reacquisition. A live lock never jumps to another
 * same-text proposal: a rejected association must first pass through LOST and two fresh hits.
 */
internal class OcrGoalLockController(
    private val hitsToAcquire: Int = 2,
    private val missesToLose: Int = 5,
    private val completionFramesRequired: Int = 3,
    private val nearScaleThreshold: Double = 0.13,
) {
    private var phase = AnchorPhase.SEARCH
    private var hitStreak = 0
    private var missStreak = 0
    private var lostFrames = 0
    private var nearHits = 0
    private var belief = INITIAL_BELIEF
    private var shortBox: NormalizedBox? = null
    private var longBox: NormalizedBox? = null
    private var pendingBox: NormalizedBox? = null
    private var velocityX = 0.0
    private var bindingId = 0
    private var lastCandidate: AnchorCandidate? = null

    init {
        require(hitsToAcquire > 0)
        require(missesToLose > 0)
        require(completionFramesRequired > 0)
        require(nearScaleThreshold > 0.0)
    }

    fun reset() {
        phase = AnchorPhase.SEARCH
        hitStreak = 0
        missStreak = 0
        lostFrames = 0
        nearHits = 0
        belief = INITIAL_BELIEF
        shortBox = null
        longBox = null
        pendingBox = null
        velocityX = 0.0
        bindingId = 0
        lastCandidate = null
    }

    fun initialGuidance() = MarkerGuidance(
        phase = GuidancePhase.SEARCH,
        command = "SEARCH",
        detail = "扫描文字目标 · candidate-bound belief",
    )

    fun update(target: AnchorTarget, candidates: List<AnchorCandidate>): OcrGoalLockUpdate {
        if (phase == AnchorPhase.TASK_COMPLETE) {
            return result(
                selected = lastCandidate,
                evidence = "TASK COMPLETE · reset to start a new goal",
                guidance = MarkerGuidance(
                    GuidancePhase.ARRIVE,
                    "TASK COMPLETE",
                    "三帧居中近距视觉证据已闭环 · 非米制距离",
                ),
            )
        }

        val matches = candidates.filter { target.matches(it.value) }
        val selection = select(matches)
        val selected = selection.candidate
        if (selected == null) return miss(matches, selection.reason)
        return hit(selected, selection.score)
    }

    private fun select(matches: List<AnchorCandidate>): Selection {
        if (matches.isEmpty()) return Selection(reason = "NO TARGET TEXT")
        return when (phase) {
            AnchorPhase.SEARCH,
            AnchorPhase.TARGET_FOUND,
            -> {
                if (pendingBox == null) {
                    matches.singleOrNull()?.let { Selection(it, 1.0, "UNIQUE SEMANTIC") }
                        ?: Selection(reason = "AMBIGUOUS · repeated target text")
                } else {
                    rank(matches, pendingBox, threshold = 0.38, margin = 0.10)
                }
            }
            AnchorPhase.LOCKED,
            AnchorPhase.REACQUIRED,
            AnchorPhase.NEAR,
            -> rankAgainstDualMemory(matches)
            AnchorPhase.LOST -> {
                when {
                    pendingBox != null -> rank(matches, pendingBox, threshold = 0.34, margin = 0.10)
                    longBox != null && lostFrames <= MEMORY_SEARCH_FRAMES ->
                        rank(matches, longBox, threshold = 0.26, margin = 0.10)
                    else -> matches.singleOrNull()?.let { Selection(it, 0.72, "EXPLICIT REBIND") }
                        ?: Selection(reason = "AMBIGUOUS · reacquire candidates")
                }
            }
            AnchorPhase.TASK_COMPLETE -> Selection(reason = "COMPLETE")
        }
    }

    private fun rankAgainstDualMemory(matches: List<AnchorCandidate>): Selection {
        val short = shortBox ?: return matches.singleOrNull()?.let { Selection(it, 1.0, "TEXT TRACK") }
            ?: Selection(reason = "AMBIGUOUS · repeated target text")
        val predicted = short.shiftX(velocityX)
        val scored = matches.mapNotNull { candidate ->
            val box = candidate.bounds ?: return@mapNotNull null
            val shortScore = spatialScore(predicted, box)
            val longScore = longBox?.let { spatialScore(it, box) } ?: shortScore
            Scored(candidate, 0.76 * shortScore + 0.24 * longScore)
        }.sortedByDescending(Scored::score)
        return chooseRanked(scored, threshold = 0.40, margin = 0.10)
    }

    private fun rank(
        matches: List<AnchorCandidate>,
        reference: NormalizedBox?,
        threshold: Double,
        margin: Double,
    ): Selection {
        if (reference == null) {
            return matches.singleOrNull()?.let { Selection(it, 1.0, "SEMANTIC") }
                ?: Selection(reason = "AMBIGUOUS · repeated target text")
        }
        val scored = matches.mapNotNull { candidate ->
            candidate.bounds?.let { Scored(candidate, spatialScore(reference, it)) }
        }.sortedByDescending(Scored::score)
        return chooseRanked(scored, threshold, margin)
    }

    private fun chooseRanked(scored: List<Scored>, threshold: Double, margin: Double): Selection {
        val best = scored.firstOrNull() ?: return Selection(reason = "SWITCH REJECTED · no geometry")
        if (best.score < threshold) return Selection(reason = "SWITCH REJECTED · spatial contradiction")
        val runnerUp = scored.getOrNull(1)
        if (runnerUp != null && best.score - runnerUp.score < margin) {
            return Selection(reason = "AMBIGUOUS · association margin")
        }
        return Selection(best.candidate, best.score, "DUAL MEMORY")
    }

    private fun spatialScore(reference: NormalizedBox, candidate: NormalizedBox): Double {
        val centerDistance = hypot(
            candidate.centerX - reference.centerX,
            candidate.centerY - reference.centerY,
        )
        val motion = exp(-centerDistance / 0.18)
        val areaRatio = (candidate.area / reference.area).coerceIn(0.05, 20.0)
        val scale = exp(-abs(ln(areaRatio)) / 1.10)
        return 0.54 * motion + 0.28 * scale + 0.18 * reference.iou(candidate)
    }

    private fun hit(candidate: AnchorCandidate, associationScore: Double): OcrGoalLockUpdate {
        lastCandidate = candidate
        val previousPhase = phase
        val wasSearching = previousPhase == AnchorPhase.SEARCH || previousPhase == AnchorPhase.TARGET_FOUND
        val wasLost = previousPhase == AnchorPhase.LOST
        missStreak = 0
        lostFrames = 0
        hitStreak += 1
        val evidenceStrength = 2.0 + 1.3 * associationScore.coerceIn(0.0, 1.0)
        belief = clamp(0.72 * belief + evidenceStrength, -6.0, 8.0)

        var lockAcquired = false
        var reacquired = false
        if (wasSearching || wasLost) {
            pendingBox = candidate.bounds
            if (hitStreak >= hitsToAcquire && belief >= ACQUIRE_BELIEF) {
                phase = if (wasLost) AnchorPhase.REACQUIRED else AnchorPhase.LOCKED
                lockAcquired = !wasLost
                reacquired = wasLost
                if (lockAcquired) bindingId += 1
                bind(candidate, initialize = true)
                pendingBox = null
            } else {
                phase = if (wasLost) AnchorPhase.LOST else AnchorPhase.TARGET_FOUND
            }
        } else {
            bind(candidate, initialize = false)
            if (previousPhase == AnchorPhase.REACQUIRED) phase = AnchorPhase.LOCKED
        }

        if (phase == AnchorPhase.LOCKED || phase == AnchorPhase.REACQUIRED || phase == AnchorPhase.NEAR) {
            val near = candidate.bounds?.let(::isNearEvidence) == true
            nearHits = if (near) nearHits + 1 else 0
            phase = when {
                nearHits >= completionFramesRequired && belief >= COMPLETE_BELIEF -> AnchorPhase.TASK_COMPLETE
                nearHits > 0 && phase != AnchorPhase.REACQUIRED -> AnchorPhase.NEAR
                phase == AnchorPhase.REACQUIRED -> AnchorPhase.REACQUIRED
                else -> AnchorPhase.LOCKED
            }
        } else {
            nearHits = 0
        }

        val guidance = guidanceFor(candidate)
        val geometry = candidate.bounds?.let {
            "x=${format(it.centerX - 0.5)} · scale=${format(it.visualScale)}"
        } ?: "geometry unavailable"
        return result(
            selected = candidate,
            lockAcquired = lockAcquired,
            reacquired = reacquired,
            evidence = when (phase) {
                AnchorPhase.TARGET_FOUND -> "TARGET FOUND · ${candidate.value}"
                AnchorPhase.REACQUIRED -> "REACQUIRED · ${candidate.value} · $geometry"
                AnchorPhase.TASK_COMPLETE -> "TASK COMPLETE · ${candidate.value} · $geometry"
                else -> "TRACK $bindingId · ${candidate.value} · $geometry"
            },
            guidance = guidance,
        )
    }

    private fun bind(candidate: AnchorCandidate, initialize: Boolean) {
        val box = candidate.bounds ?: return
        if (initialize || shortBox == null) {
            shortBox = box
            longBox = box
            velocityX = 0.0
            return
        }
        val previous = shortBox ?: box
        val instantaneousVelocity = box.centerX - previous.centerX
        velocityX = 0.65 * velocityX + 0.35 * instantaneousVelocity
        shortBox = previous.blend(box, 0.70)
        longBox = (longBox ?: box).blend(box, 0.18)
    }

    private fun isNearEvidence(box: NormalizedBox): Boolean =
        abs(box.centerX - 0.5) <= 0.10 &&
            box.centerY in 0.18..0.82 &&
            box.visualScale >= nearScaleThreshold

    private fun guidanceFor(candidate: AnchorCandidate): MarkerGuidance {
        if (phase == AnchorPhase.TARGET_FOUND) {
            return MarkerGuidance(
                GuidancePhase.SEARCH,
                "TARGET FOUND",
                "第 $hitStreak/$hitsToAcquire 帧 · belief=${format(belief)}",
            )
        }
        if (phase == AnchorPhase.TASK_COMPLETE) {
            return MarkerGuidance(
                GuidancePhase.ARRIVE,
                "TASK COMPLETE",
                "$nearHits/$completionFramesRequired 帧居中近距视觉证据 · 非米制距离",
            )
        }
        val box = candidate.bounds
        if (box == null) {
            return MarkerGuidance(
                GuidancePhase.ADVANCE,
                if (phase == AnchorPhase.REACQUIRED) "REACQUIRED" else "LOCKED",
                "文字匹配已锁定；当前帧没有候选几何",
            )
        }
        if (phase == AnchorPhase.NEAR) {
            return MarkerGuidance(
                GuidancePhase.ALIGN,
                "NEAR",
                "$nearHits/$completionFramesRequired 帧视觉近距证据 · 保持居中",
            )
        }
        val command = when {
            box.centerX < 0.40 -> "LEFT"
            box.centerX > 0.60 -> "RIGHT"
            else -> "FORWARD"
        }
        return MarkerGuidance(
            phase = if (command == "FORWARD") GuidancePhase.ADVANCE else GuidancePhase.ALIGN,
            command = command,
            detail = "track=$bindingId · belief=${format(belief)} · x=${format(box.centerX - 0.5)} · scale=${format(box.visualScale)}",
        )
    }

    private fun miss(matches: List<AnchorCandidate>, reason: String): OcrGoalLockUpdate {
        hitStreak = 0
        nearHits = 0
        missStreak += 1
        belief = 0.90 * belief - if (missStreak <= 3) 0.55 else 0.85

        when (phase) {
            AnchorPhase.SEARCH,
            AnchorPhase.TARGET_FOUND,
            -> {
                phase = AnchorPhase.SEARCH
                pendingBox = null
            }
            AnchorPhase.LOCKED,
            AnchorPhase.REACQUIRED,
            AnchorPhase.NEAR,
            -> if (missStreak >= missesToLose) {
                phase = AnchorPhase.LOST
                pendingBox = null
                lostFrames = 1
            }
            AnchorPhase.LOST -> lostFrames += 1
            AnchorPhase.TASK_COMPLETE -> Unit
        }

        val guidance = when (phase) {
            AnchorPhase.LOST -> MarkerGuidance(
                GuidancePhase.LOST,
                scanCommand(),
                "主动重捕获 · lost=$lostFrames · belief=${format(belief)}",
            )
            AnchorPhase.SEARCH -> initialGuidance()
            else -> MarkerGuidance(
                GuidancePhase.ALIGN,
                "HOLD",
                "短时缺证据 $missStreak/$missesToLose · 不切换候选",
            )
        }
        val evidence = when {
            reason.startsWith("AMBIGUOUS") -> reason
            reason.startsWith("SWITCH REJECTED") -> "$reason · matches=${matches.size}"
            matches.isEmpty() -> "NO TARGET TEXT"
            else -> reason
        }
        return result(selected = null, evidence = evidence, guidance = guidance)
    }

    private fun scanCommand(): String {
        if (lostFrames <= 5) {
            longBox?.let { return if (it.centerX < 0.5) "SCAN LEFT" else "SCAN RIGHT" }
        }
        val sweep = (lostFrames - 6).coerceAtLeast(0) % 12
        return if (sweep < 4) "SCAN RIGHT" else "SCAN LEFT"
    }

    private fun result(
        selected: AnchorCandidate?,
        evidence: String,
        guidance: MarkerGuidance,
        lockAcquired: Boolean = false,
        reacquired: Boolean = false,
    ) = OcrGoalLockUpdate(
        phase = phase,
        selected = selected,
        hitStreak = hitStreak,
        missStreak = missStreak,
        beliefScore = belief,
        completionEvidenceFrames = nearHits,
        lockAcquired = lockAcquired,
        reacquired = reacquired,
        evidence = evidence,
        guidance = guidance,
    )

    private fun format(value: Double) = "%.2f".format(java.util.Locale.US, value)

    private data class Scored(val candidate: AnchorCandidate, val score: Double)

    private data class Selection(
        val candidate: AnchorCandidate? = null,
        val score: Double = 0.0,
        val reason: String = "",
    )

    private companion object {
        const val INITIAL_BELIEF = -2.2
        const val ACQUIRE_BELIEF = 3.2
        const val COMPLETE_BELIEF = 4.4
        const val MEMORY_SEARCH_FRAMES = 8

        fun clamp(value: Double, low: Double, high: Double) = maxOf(low, minOf(high, value))
    }
}
